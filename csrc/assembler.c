#include "lana/assembler.h"
#include "lana/vm.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LANA_ASSEMBLER_MAX_FIXUPS 4096u
#define LANA_ASSEMBLER_INDEX_CAPACITY (LANA_ASSEMBLER_MAX_FIXUPS * 2u)

typedef struct {
    char name[64];
    uint32_t offset;
} Label;

typedef struct {
    char name[64];
    uint32_t instruction;
} Fixup;

typedef struct {
    char name[128];
    uint32_t instruction;
} FunctionFixup;

static size_t name_hash(const char *name) {
    size_t hash = (size_t)2166136261u;
    while (*name != '\0') {
        hash ^= (unsigned char)*name++;
        hash *= (size_t)16777619u;
    }
    return hash;
}

static void index_clear(size_t *slots) {
    size_t index;
    for (index = 0u; index < LANA_ASSEMBLER_INDEX_CAPACITY; ++index)
        slots[index] = SIZE_MAX;
}

static bool label_index_add(size_t *slots, const Label *labels,
                            size_t label_index) {
    size_t slot = name_hash(labels[label_index].name) &
                  (LANA_ASSEMBLER_INDEX_CAPACITY - 1u);
    size_t probe;
    for (probe = 0u; probe < LANA_ASSEMBLER_INDEX_CAPACITY; ++probe) {
        if (slots[slot] == SIZE_MAX) {
            slots[slot] = label_index;
            return true;
        }
        if (strcmp(labels[slots[slot]].name, labels[label_index].name) == 0)
            return true;
        slot = (slot + 1u) & (LANA_ASSEMBLER_INDEX_CAPACITY - 1u);
    }
    return false;
}

static size_t label_index_find(const size_t *slots, const Label *labels,
                               const char *name) {
    size_t slot = name_hash(name) & (LANA_ASSEMBLER_INDEX_CAPACITY - 1u);
    size_t probe;
    for (probe = 0u; probe < LANA_ASSEMBLER_INDEX_CAPACITY; ++probe) {
        size_t index = slots[slot];
        if (index == SIZE_MAX) return SIZE_MAX;
        if (strcmp(labels[index].name, name) == 0) return index;
        slot = (slot + 1u) & (LANA_ASSEMBLER_INDEX_CAPACITY - 1u);
    }
    return SIZE_MAX;
}

static bool function_index_add(size_t *slots, const LanaChunk *chunk,
                               size_t function_index) {
    size_t slot = name_hash(chunk->functions[function_index].name) &
                  (LANA_ASSEMBLER_INDEX_CAPACITY - 1u);
    size_t probe;
    for (probe = 0u; probe < LANA_ASSEMBLER_INDEX_CAPACITY; ++probe) {
        if (slots[slot] == SIZE_MAX) {
            slots[slot] = function_index;
            return true;
        }
        if (strcmp(chunk->functions[slots[slot]].name,
                   chunk->functions[function_index].name) == 0)
            return true;
        slot = (slot + 1u) & (LANA_ASSEMBLER_INDEX_CAPACITY - 1u);
    }
    return false;
}

static size_t function_index_find(const size_t *slots, const LanaChunk *chunk,
                                  const char *name) {
    size_t slot = name_hash(name) & (LANA_ASSEMBLER_INDEX_CAPACITY - 1u);
    size_t probe;
    for (probe = 0u; probe < LANA_ASSEMBLER_INDEX_CAPACITY; ++probe) {
        size_t index = slots[slot];
        if (index == SIZE_MAX) return SIZE_MAX;
        if (strcmp(chunk->functions[index].name, name) == 0) return index;
        slot = (slot + 1u) & (LANA_ASSEMBLER_INDEX_CAPACITY - 1u);
    }
    return SIZE_MAX;
}

static char *trim(char *text) {
    char *end;
    while (isspace((unsigned char)*text)) ++text;
    if (*text == '\0') return text;
    end = text + strlen(text) - 1;
    while (end > text && isspace((unsigned char)*end)) *end-- = '\0';
    return text;
}

static bool parse_register(const char *text, uint32_t *out) {
    char *end;
    unsigned long value;
    if (text == NULL || (text[0] != 'R' && text[0] != 'r')) return false;
    errno = 0; value = strtoul(text + 1, &end, 10);
    if (errno != 0 || *end != '\0' || value >= LANA_MAX_REGISTERS) return false;
    *out = (uint32_t)value; return true;
}

static bool parse_number(const char *text, double *out) {
    char *end;
    errno = 0; *out = strtod(text, &end);
    return errno == 0 && end != text && *end == '\0';
}

static LanaError add_number(LanaChunk *chunk, double number, uint32_t *index) {
    return lana_chunk_add_constant(chunk, lana_value_number(number), index);
}

static LanaError number_constant(LanaChunk *chunk, const char *text, uint32_t *index) {
    char *end;
    unsigned long parsed;
    double number;
    if ((text[0] == 'K' || text[0] == 'k') && text[1] != '\0') {
        errno = 0;
        parsed = strtoul(text + 1, &end, 10);
        if (errno != 0 || *end != '\0' || parsed >= chunk->constant_count ||
            chunk->constants[parsed].type != VAL_NUMBER) return LANA_ERR_CONSTANT;
        *index = (uint32_t)parsed;
        return LANA_OK;
    }
    if (!parse_number(text, &number)) return LANA_ERR_FORMAT;
    return add_number(chunk, number, index);
}

static LanaError string_constant(LanaChunk *chunk, const char *text, uint32_t *index) {
    char *end;
    unsigned long parsed;
    if (text == NULL || *text == '\0') return LANA_ERR_FORMAT;
    if ((text[0] != 'S' && text[0] != 's') ||
        !isdigit((unsigned char)text[1]))
        return lana_chunk_add_constant(chunk, lana_value_string(text), index);
    errno = 0;
    parsed = strtoul(text + 1, &end, 10);
    if (errno != 0 || *end != '\0' || parsed >= chunk->constant_count ||
        chunk->constants[parsed].type != VAL_STRING) return LANA_ERR_CONSTANT;
    *index = (uint32_t)parsed;
    return LANA_OK;
}

static int transform_id(const char *name) {
    if (strcmp(name, "invert") == 0) return LANA_TRANSFORM_INVERT;
    if (strcmp(name, "neutralize") == 0) return LANA_TRANSFORM_NEUTRALIZE;
    return -1;
}

static int measure_id_current(const char *name) {
    if (strcmp(name, "probability") == 0) return LANA_MEASURE_PROBABILITY;
    if (strcmp(name, "distribution") == 0) return LANA_MEASURE_DISTRIBUTION;
    if (strcmp(name, "sample") == 0) return LANA_MEASURE_SAMPLE;
    return -1;
}

static int basis_id(const char *name) {
    if (strcmp(name, "computational") == 0) return LANA_MEASURE_BASIS_COMPUTATIONAL;
    if (strcmp(name, "x") == 0) return LANA_MEASURE_BASIS_X;
    if (strcmp(name, "y") == 0) return LANA_MEASURE_BASIS_Y;
    return -1;
}

static bool sample_count(const char *text, uint32_t *out) {
    char *end;
    unsigned long long value;
    if (text == NULL || *text == '\0' || text[0] == '-') return false;
    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno != 0 || *end != '\0' || value == 0u || value > UINT32_MAX)
        return false;
    *out = (uint32_t)value;
    return true;
}

static int field_id(const char *name) {
    if (strcmp(name, "p") == 0) return 0;
    if (strcmp(name, "d") == 0 || strcmp(name, "d_re") == 0) return 1;
    if (strcmp(name, "d_im") == 0) return 2;
    return -1;
}

static int index_id(const char *name) {
    if (strcmp(name, "timestamp") == 0) return 0;
    if (strcmp(name, "source") == 0) return 1;
    if (strcmp(name, "weight") == 0) return 2;
    if (strcmp(name, "confidence") == 0) return 3;
    return -1;
}

static int binary_id(const char *name) {
    if (strcmp(name, "+") == 0 || strcmp(name, "add") == 0) return LANA_BINARY_ADD;
    if (strcmp(name, "-") == 0 || strcmp(name, "sub") == 0) return LANA_BINARY_SUBTRACT;
    if (strcmp(name, "*") == 0 || strcmp(name, "mul") == 0) return LANA_BINARY_MULTIPLY;
    if (strcmp(name, "/") == 0 || strcmp(name, "div") == 0) return LANA_BINARY_DIVIDE;
    return -1;
}

static int compare_id(const char *name) {
    if (strcmp(name, "==") == 0) return LANA_COMPARE_EQUAL;
    if (strcmp(name, "!=") == 0) return LANA_COMPARE_NOT_EQUAL;
    if (strcmp(name, "<") == 0) return LANA_COMPARE_LESS;
    if (strcmp(name, "<=") == 0) return LANA_COMPARE_LESS_EQUAL;
    if (strcmp(name, ">") == 0) return LANA_COMPARE_GREATER;
    if (strcmp(name, ">=") == 0) return LANA_COMPARE_GREATER_EQUAL;
    return -1;
}

static int host_call_id(const char *name) {
    static const char *names[] = {
        "args", "read_text", "write_text", "now", "random", "assert",
        "map_new", "map_has", "map_get", "map_set", "map_keys", "index_get", "index_set", "json_parse", "json_stringify",
        "csv_read", "csv_write", "string_length", "string_byte_at",
        "string_slice", "string_concat", "number_to_string", "array_new",
        "array_push", "string_hex", "string_join", "array_length",
        "string_unescape", "path_resolve", "sample_record",
        "information_new", "claim_new", "claim_value", "claim_proposition",
        "claim_status", "planned_effect_new", "planned_effect_execute",
        "planned_effect_status", "shared_information", "shared_grant",
        "shared_revoke", "shared_snapshot", "shared_at", "shared_observe",
        "shared_revision", "shared_identity", "shared_wait",
        "information_inspect", "directory_list", "directory_create",
        "path_exists", "write_text_atomic", "hash_update"
    };
    size_t index;
    for (index = 0; index < sizeof(names) / sizeof(names[0]); ++index)
        if (strcmp(name, names[index]) == 0) return (int)index;
    return -1;
}

static int hex_digit(char character) {
    if (character >= '0' && character <= '9') return character - '0';
    if (character >= 'a' && character <= 'f') return character - 'a' + 10;
    if (character >= 'A' && character <= 'F') return character - 'A' + 10;
    return -1;
}

static LanaError add_hex_string(LanaChunk *chunk, const char *encoded, uint32_t *index) {
    size_t encoded_length = strcmp(encoded, "-") == 0 ? 0u : strlen(encoded);
    size_t output_index;
    char *decoded;
    LanaError result;
    if (encoded_length % 2u != 0u) return LANA_ERR_FORMAT;
    decoded = malloc(encoded_length / 2u + 1u);
    if (decoded == NULL) return LANA_ERR_OOM;
    for (output_index = 0; output_index < encoded_length / 2u; ++output_index) {
        int high = hex_digit(encoded[output_index * 2u]);
        int low = hex_digit(encoded[output_index * 2u + 1u]);
        if (high < 0 || low < 0) { free(decoded); return LANA_ERR_FORMAT; }
        decoded[output_index] = (char)((high << 4) | low);
        if (decoded[output_index] == '\0') { free(decoded); return LANA_ERR_FORMAT; }
    }
    decoded[encoded_length / 2u] = '\0';
    result = lana_chunk_add_constant(chunk, lana_value_string(decoded), index);
    free(decoded);
    return result;
}

static bool copy_name(char destination[64], const char *source) {
    size_t length = strlen(source);
    if (length == 0 || length >= 64u) return false;
    memcpy(destination, source, length + 1u); return true;
}

static LanaError emit_line(LanaChunk *chunk, char **tokens, size_t count, uint32_t line,
                         Fixup *fixups, size_t *fixup_count,
                         FunctionFixup *function_fixups,
                         size_t *function_fixup_count, LanaErrorInfo *error) {
    LanaInstruction ins = {.a = LANA_NO_OPERAND, .b = LANA_NO_OPERAND, .c = LANA_NO_OPERAND,
                         .imm = LANA_NO_OPERAND, .line = line};
    uint32_t a, b, c;
    int id;
    double number;
    LanaError result;
#define EXPECT(n) do { if (count != (n)) { lana_error_set(error, LANA_ERR_FORMAT, chunk->code_count, OP_NOP, line, "%s expects %u operands", tokens[0], (unsigned)((n)-1u)); return LANA_ERR_FORMAT; } } while (0)
#define REG(token, destination) do { if (!parse_register((token), &(destination))) { lana_error_set(error, LANA_ERR_REGISTER, chunk->code_count, OP_NOP, line, "invalid register %s", (token)); return LANA_ERR_REGISTER; } } while (0)
    if (strcmp(tokens[0], "NOP") == 0) { EXPECT(1); ins.opcode = OP_NOP; }
    else if (strcmp(tokens[0], "HALT") == 0) { EXPECT(1); ins.opcode = OP_HALT; }
    else if (strcmp(tokens[0], "LOAD_CONST") == 0) {
        EXPECT(3); REG(tokens[1], a); ins.opcode = OP_LOAD_CONST; ins.a = a;
        if (strcmp(tokens[2], "true") == 0) result = lana_chunk_add_constant(chunk, lana_value_bool(true), &ins.imm);
        else if (strcmp(tokens[2], "false") == 0) result = lana_chunk_add_constant(chunk, lana_value_bool(false), &ins.imm);
        else if (strcmp(tokens[2], "null") == 0) result = lana_chunk_add_constant(chunk, lana_value_null(), &ins.imm);
        else if (parse_number(tokens[2], &number)) result = add_number(chunk, number, &ins.imm);
        else result = lana_chunk_add_constant(chunk, lana_value_string(tokens[2]), &ins.imm);
        if (result != LANA_OK) return result;
    } else if (strcmp(tokens[0], "LOAD_STRING") == 0) {
        EXPECT(3); REG(tokens[1], a); ins.opcode = OP_LOAD_CONST; ins.a = a;
        result = add_hex_string(chunk, tokens[2], &ins.imm);
        if (result != LANA_OK) return result;
    } else if (strcmp(tokens[0], "STATE_NEW") == 0) {
        EXPECT(5);
        REG(tokens[1], a);
        result = number_constant(chunk, tokens[2], &b);
        if (result != LANA_OK) return result;
        result = number_constant(chunk, tokens[3], &c);
        if (result != LANA_OK) return result;
        result = number_constant(chunk, tokens[4], &ins.imm);
        if (result != LANA_OK) return result;
        ins.opcode = OP_STATE_NEW;
        ins.a = a;
        ins.b = b;
        ins.c = c;
    } else if (strcmp(tokens[0], "STATE_BUILD") == 0) {
        EXPECT(5);
        REG(tokens[1], a);
        REG(tokens[2], b);
        REG(tokens[3], c);
        REG(tokens[4], ins.imm);
        ins.opcode = OP_STATE_BUILD;
        ins.a = a;
        ins.b = b;
        ins.c = c;
    } else if (strcmp(tokens[0], "MOVE") == 0) {
        EXPECT(3); REG(tokens[1], a); REG(tokens[2], b);
        ins.opcode = OP_MOVE;
        ins.a = a; ins.b = b;
    } else if (strcmp(tokens[0], "TRANSFORM") == 0) {
        EXPECT(4);
        REG(tokens[1], a);
        REG(tokens[2], b);
        id = transform_id(tokens[3]);
        if (id < 0) return LANA_ERR_TRANSFORM;
        ins.opcode = OP_TRANSFORM;
        ins.a = a;
        ins.b = b;
        ins.c = (uint32_t)id;
        ins.imm = 0u;
    } else if (strcmp(tokens[0], "MEASURE") == 0) {
        EXPECT(4);
        REG(tokens[1], a);
        id = measure_id_current(tokens[2]);
        REG(tokens[3], b);
        if (id < 0) return LANA_ERR_MEASURE;
        ins.opcode = OP_MEASURE;
        ins.a = a;
        ins.b = b;
        ins.c = (uint32_t)id;
        ins.imm = 0u;
    } else if (strcmp(tokens[0], "MEASURE_BASIS") == 0) {
        EXPECT(5);
        REG(tokens[1], a);
        REG(tokens[2], b);
        id = basis_id(tokens[3]);
        if (id < 0) return LANA_ERR_MEASURE;
        result = measure_id_current(tokens[4]);
        if (result < 0) return LANA_ERR_MEASURE;
        ins.opcode = OP_MEASURE_BASIS;
        ins.a = a;
        ins.b = b;
        ins.c = (uint32_t)id;
        ins.imm = (uint32_t)result;
    } else if (strcmp(tokens[0], "ESTIMATE_MEASURE_PROBABILITY") == 0 ||
               strcmp(tokens[0], "ESTIMATE_MEASURE_DISTRIBUTION") == 0) {
        EXPECT(5);
        REG(tokens[1], a);
        REG(tokens[2], b);
        id = basis_id(tokens[3]);
        if (id < 0) return LANA_ERR_MEASURE;
        if (!sample_count(tokens[4], &ins.imm)) return LANA_ERR_FORMAT;
        ins.opcode = strcmp(tokens[0], "ESTIMATE_MEASURE_PROBABILITY") == 0
                         ? OP_ESTIMATE_MEASURE_PROBABILITY
                         : OP_ESTIMATE_MEASURE_DISTRIBUTION;
        ins.a = a;
        ins.b = b;
        ins.c = (uint32_t)id;
    } else if (strcmp(tokens[0], "APPEND") == 0) {
        EXPECT(4);
        REG(tokens[1], a);
        REG(tokens[2], b);
        REG(tokens[3], c);
        ins.opcode = OP_APPEND;
        ins.a = a;
        ins.b = b;
        ins.c = c;
        ins.imm = 0u;
    } else if (strcmp(tokens[0], "JOINT_BUILD") == 0) {
        uint32_t joint_count;
        EXPECT(5); REG(tokens[1], a); REG(tokens[2], b);
        if (!sample_count(tokens[3], &joint_count) || b + joint_count > LANA_MAX_REGISTERS)
            return LANA_ERR_REGISTER;
        result = string_constant(chunk, tokens[4], &ins.imm);
        if (result != LANA_OK) return result;
        ins.opcode = OP_JOINT_BUILD; ins.a = a; ins.b = b; ins.c = joint_count;
    } else if (strcmp(tokens[0], "JOINT_PROJECT") == 0) {
        EXPECT(4); REG(tokens[1], a); REG(tokens[2], b);
        result = string_constant(chunk, tokens[3], &ins.c);
        if (result != LANA_OK) return result;
        ins.opcode = OP_JOINT_PROJECT; ins.a = a; ins.b = b;
    } else if (strcmp(tokens[0], "JOINT_CONDITION") == 0) {
        EXPECT(5); REG(tokens[1], a); REG(tokens[2], b);
        result = string_constant(chunk, tokens[3], &ins.c);
        if (result != LANA_OK) return result;
        REG(tokens[4], ins.imm);
        ins.opcode = OP_JOINT_CONDITION; ins.a = a; ins.b = b;
    } else if (strcmp(tokens[0], "JOINT_BUILD_FINITE") == 0) {
        EXPECT(4); REG(tokens[1], a); REG(tokens[2], b);
        result = string_constant(chunk, tokens[3], &ins.c);
        if (result != LANA_OK) return result;
        ins.opcode = OP_JOINT_BUILD_FINITE; ins.a = a; ins.b = b; ins.imm = 0u;
    } else if (strcmp(tokens[0], "JOINT_RENAME") == 0) {
        EXPECT(5); REG(tokens[1], a); REG(tokens[2], b);
        result = string_constant(chunk, tokens[3], &ins.c);
        if (result != LANA_OK) return result;
        result = string_constant(chunk, tokens[4], &ins.imm);
        if (result != LANA_OK) return result;
        ins.opcode = OP_JOINT_RENAME; ins.a = a; ins.b = b;
    } else if (strcmp(tokens[0], "POSSIBILITY_BUILD") == 0 ||
               strcmp(tokens[0], "INFO_SAMPLE") == 0) {
        EXPECT(3); REG(tokens[1], a); REG(tokens[2], b);
        ins.opcode = strcmp(tokens[0], "POSSIBILITY_BUILD") == 0
            ? OP_POSSIBILITY_BUILD : OP_INFO_SAMPLE;
        ins.a = a; ins.b = b; ins.c = 0u; ins.imm = 0u;
    } else if (strcmp(tokens[0], "PATH_SPLIT") == 0) {
        EXPECT(3); REG(tokens[1], a);
        ins.opcode = OP_PATH_SPLIT; ins.a = a; ins.b = 0u; ins.c = 0u;
        if (*fixup_count >= LANA_ASSEMBLER_MAX_FIXUPS ||
            !copy_name(fixups[*fixup_count].name, tokens[2])) return LANA_ERR_LIMIT;
        fixups[*fixup_count].instruction = (uint32_t)chunk->code_count;
        ++*fixup_count;
    } else if (strcmp(tokens[0], "PATH_JOIN") == 0) {
        EXPECT(1); ins.opcode = OP_PATH_JOIN;
        ins.a = 0u; ins.b = 0u; ins.c = 0u; ins.imm = 0u;
    } else if (strcmp(tokens[0], "OBSERVE") == 0) {
        EXPECT(5); REG(tokens[1], a); REG(tokens[2], b);
        result = string_constant(chunk, tokens[3], &ins.c);
        if (result != LANA_OK) return result;
        REG(tokens[4], ins.imm);
        ins.opcode = OP_OBSERVE; ins.a = a; ins.b = b;
    } else if (strcmp(tokens[0], "EVIDENCE") == 0 ||
               strcmp(tokens[0], "ASSUME") == 0) {
        EXPECT(4); REG(tokens[1], a); REG(tokens[2], b);
        result = string_constant(chunk, tokens[3], &ins.c);
        if (result != LANA_OK) return result;
        ins.opcode = strcmp(tokens[0], "EVIDENCE") == 0
                         ? OP_EVIDENCE : OP_ASSUME;
        ins.a = a; ins.b = b; ins.imm = 0u;
    } else if (strcmp(tokens[0], "DERIVATION") == 0 ||
               strcmp(tokens[0], "EXPLAIN") == 0) {
        EXPECT(3); REG(tokens[1], a); REG(tokens[2], b);
        ins.opcode = strcmp(tokens[0], "DERIVATION") == 0
                         ? OP_DERIVATION : OP_EXPLAIN;
        ins.a = a; ins.b = b; ins.c = 0u; ins.imm = 0u;
    } else if (strcmp(tokens[0], "JOINT_SAMPLE") == 0 ||
               strcmp(tokens[0], "RESOLVE") == 0) {
        EXPECT(3); REG(tokens[1], a); REG(tokens[2], b);
        ins.opcode = strcmp(tokens[0], "JOINT_SAMPLE") == 0
                         ? OP_JOINT_SAMPLE : OP_RESOLVE;
        ins.a = a; ins.b = b; ins.c = 0u; ins.imm = 0u;
    } else if (strcmp(tokens[0], "SAMPLE_STATE_DIST") == 0) {
        EXPECT(3);
        REG(tokens[1], a);
        REG(tokens[2], b);
        ins.opcode = OP_SAMPLE_STATE_DIST;
        ins.a = a;
        ins.b = b;
        ins.c = 0u;
        ins.imm = 0u;
    } else if (strcmp(tokens[0], "GET_FIELD") == 0 || strcmp(tokens[0], "GET_INDEX") == 0) {
        EXPECT(4); REG(tokens[1], a); id = strcmp(tokens[0], "GET_FIELD") == 0 ? field_id(tokens[2]) : index_id(tokens[2]); REG(tokens[3], b); if (id < 0) return LANA_ERR_FORMAT;
        ins.opcode = strcmp(tokens[0], "GET_FIELD") == 0 ? OP_GET_FIELD : OP_GET_INDEX; ins.a = a; ins.b = b; ins.c = (uint32_t)id;
    } else if (strcmp(tokens[0], "SET_INDEX") == 0) {
        EXPECT(4); REG(tokens[1], a); id = index_id(tokens[2]); REG(tokens[3], c); if (id < 0) return LANA_ERR_FORMAT;
        ins.opcode = OP_SET_INDEX; ins.a = a; ins.b = (uint32_t)id; ins.c = c;
    } else if (strcmp(tokens[0], "HISTORY") == 0) {
        EXPECT(4); REG(tokens[1], a); REG(tokens[3], b);
        id = strcmp(tokens[2], "latest") == 0 ? LANA_HISTORY_LATEST : strcmp(tokens[2], "duration") == 0 ? LANA_HISTORY_DURATION : -1;
        if (id < 0) return LANA_ERR_HISTORY;
        ins.opcode = OP_HISTORY_CONFIG;
        ins.a = a;
        ins.b = b;
        ins.c = (uint32_t)id;
    } else if (strcmp(tokens[0], "PREVIOUS") == 0 || strcmp(tokens[0], "CHANGE") == 0 || strcmp(tokens[0], "VELOCITY") == 0) {
        EXPECT(3); REG(tokens[1], a); REG(tokens[2], b);
        ins.opcode = strcmp(tokens[0], "PREVIOUS") == 0 ? OP_PREVIOUS : strcmp(tokens[0], "CHANGE") == 0 ? OP_CHANGE : OP_VELOCITY; ins.a = a; ins.b = b;
    } else if (strcmp(tokens[0], "BINARY") == 0 || strcmp(tokens[0], "COMPARE") == 0) {
        EXPECT(5); REG(tokens[1], a); id = strcmp(tokens[0], "BINARY") == 0 ? binary_id(tokens[2]) : compare_id(tokens[2]); REG(tokens[3], b); REG(tokens[4], c); if (id < 0) return LANA_ERR_FORMAT;
        ins.opcode = strcmp(tokens[0], "BINARY") == 0 ? OP_BINARY : OP_COMPARE; ins.a = a; ins.b = b; ins.c = c; ins.imm = (uint32_t)id;
    } else if (strcmp(tokens[0], "UNARY") == 0) {
        EXPECT(4); REG(tokens[2], a); REG(tokens[3], b); id = strcmp(tokens[1], "-") == 0 ? 0 : strcmp(tokens[1], "!") == 0 ? 1 : -1; if (id < 0) return LANA_ERR_FORMAT;
        ins.opcode = OP_UNARY; ins.a = a; ins.b = b; ins.imm = (uint32_t)id;
    } else if (strcmp(tokens[0], "JUMP") == 0 || strcmp(tokens[0], "JUMP_IF_TRUE") == 0 || strcmp(tokens[0], "JUMP_IF_FALSE") == 0) {
        size_t label_token = 1;
        if (strcmp(tokens[0], "JUMP") == 0) { EXPECT(2); ins.opcode = OP_JUMP; }
        else { EXPECT(3); REG(tokens[1], a); ins.a = a; label_token = 2; ins.opcode = strcmp(tokens[0], "JUMP_IF_TRUE") == 0 ? OP_JUMP_IF_TRUE : OP_JUMP_IF_FALSE; }
        if (*fixup_count >= LANA_ASSEMBLER_MAX_FIXUPS || !copy_name(fixups[*fixup_count].name, tokens[label_token])) return LANA_ERR_LIMIT;
        fixups[*fixup_count].instruction = (uint32_t)chunk->code_count; ++*fixup_count;
    } else if (strcmp(tokens[0], "ARRAY_NEW") == 0) {
        EXPECT(4); REG(tokens[1], a); REG(tokens[2], b); number = strtod(tokens[3], NULL);
        if (number < 0 || number > LANA_MAX_REGISTERS) return LANA_ERR_LIMIT;
        ins.opcode = OP_ARRAY_NEW; ins.a = a; ins.b = b; ins.c = (uint32_t)number;
    } else if (strcmp(tokens[0], "ARRAY_GET") == 0 || strcmp(tokens[0], "ARRAY_SET") == 0) {
        EXPECT(4); REG(tokens[1], a); REG(tokens[2], b); REG(tokens[3], c);
        ins.opcode = strcmp(tokens[0], "ARRAY_GET") == 0 ? OP_ARRAY_GET : OP_ARRAY_SET; ins.a = a; ins.b = b; ins.c = c;
    } else if (strcmp(tokens[0], "CALL") == 0) {
        size_t function_index;
        EXPECT(5); REG(tokens[2], a); number = strtod(tokens[3], NULL); REG(tokens[4], b);
        for (function_index = 0; function_index < chunk->function_count; ++function_index)
            if (strcmp(tokens[1], chunk->functions[function_index].name) == 0) break;
        if (number < 0.0 || number > LANA_MAX_REGISTERS) return LANA_ERR_FORMAT;
        if (function_index == chunk->function_count) {
            if (*function_fixup_count >= LANA_ASSEMBLER_MAX_FIXUPS ||
                strlen(tokens[1]) >= sizeof(function_fixups[0].name)) return LANA_ERR_LIMIT;
            strcpy(function_fixups[*function_fixup_count].name, tokens[1]);
            function_fixups[*function_fixup_count].instruction = (uint32_t)chunk->code_count;
            ++*function_fixup_count;
            function_index = 0u;
        }
        ins.opcode = OP_CALL; ins.a = b; ins.b = (uint32_t)function_index; ins.c = a; ins.imm = (uint32_t)number;
    } else if (strcmp(tokens[0], "FORK") == 0) {
        size_t function_index;
        EXPECT(5); REG(tokens[2], a); number = strtod(tokens[3], NULL); REG(tokens[4], b);
        for (function_index = 0; function_index < chunk->function_count; ++function_index)
            if (strcmp(tokens[1], chunk->functions[function_index].name) == 0) break;
        if (number < 0.0 || number > LANA_MAX_REGISTERS || floor(number) != number) return LANA_ERR_FORMAT;
        if (function_index == chunk->function_count) {
            if (*function_fixup_count >= LANA_ASSEMBLER_MAX_FIXUPS ||
                strlen(tokens[1]) >= sizeof(function_fixups[0].name)) return LANA_ERR_LIMIT;
            strcpy(function_fixups[*function_fixup_count].name, tokens[1]);
            function_fixups[*function_fixup_count].instruction = (uint32_t)chunk->code_count;
            ++*function_fixup_count;
            function_index = 0u;
        }
        ins.opcode = OP_FORK; ins.a = b; ins.b = (uint32_t)function_index; ins.c = a; ins.imm = (uint32_t)number;
    } else if (strcmp(tokens[0], "JOIN") == 0 || strcmp(tokens[0], "JOIN_ALL") == 0) {
        EXPECT(3); REG(tokens[1], a); REG(tokens[2], b);
        ins.opcode = strcmp(tokens[0], "JOIN") == 0 ? OP_JOIN : OP_JOIN_ALL; ins.a = a; ins.b = b;
    } else if (strcmp(tokens[0], "JOIN_TIMEOUT") == 0) {
        EXPECT(4); REG(tokens[1], a); REG(tokens[2], b); REG(tokens[3], c);
        ins.opcode = OP_JOIN_TIMEOUT; ins.a = a; ins.b = b; ins.c = c;
    } else if (strcmp(tokens[0], "CANCEL") == 0) {
        EXPECT(2); REG(tokens[1], a); ins.opcode = OP_CANCEL; ins.a = a;
    } else if (strcmp(tokens[0], "TASKGROUP_ENTER") == 0 || strcmp(tokens[0], "TASKGROUP_EXIT") == 0) {
        EXPECT(1); ins.opcode = strcmp(tokens[0], "TASKGROUP_ENTER") == 0 ? OP_TASKGROUP_ENTER : OP_TASKGROUP_EXIT;
    } else if (strcmp(tokens[0], "HOST_CALL") == 0) {
        EXPECT(5); id = host_call_id(tokens[1]); REG(tokens[2], a); number = strtod(tokens[3], NULL); REG(tokens[4], b);
        if (id < 0 || number < 0.0 || number > LANA_MAX_REGISTERS || floor(number) != number) return LANA_ERR_FORMAT;
        ins.opcode = OP_HOST_CALL; ins.a = b; ins.b = (uint32_t)id; ins.c = a; ins.imm = (uint32_t)number;
    } else if (strcmp(tokens[0], "RETURN") == 0 || strcmp(tokens[0], "PRINT") == 0) {
        EXPECT(2); REG(tokens[1], a); ins.opcode = strcmp(tokens[0], "RETURN") == 0 ? OP_RETURN : OP_PRINT; ins.a = a;
    } else {
        lana_error_set(error, LANA_ERR_OPCODE, chunk->code_count, OP_NOP, line, "unknown instruction %s", tokens[0]); return LANA_ERR_OPCODE;
    }
    result = lana_chunk_emit(chunk, ins);
    return result;
#undef EXPECT
#undef REG
}

LanaError lana_assemble_file(const char *path, LanaChunk *chunk, LanaErrorInfo *error) {
    FILE *file = fopen(path, "r");
    char buffer[4096];
    uint32_t line = 0;
    uint32_t source_line = 0;
    Label labels[LANA_ASSEMBLER_MAX_FIXUPS]; size_t label_count = 0;
    Fixup fixups[LANA_ASSEMBLER_MAX_FIXUPS]; size_t fixup_count = 0;
    FunctionFixup function_fixups[LANA_ASSEMBLER_MAX_FIXUPS]; size_t function_fixup_count = 0;
    size_t label_index[LANA_ASSEMBLER_INDEX_CAPACITY];
    size_t function_index[LANA_ASSEMBLER_INDEX_CAPACITY];
    LanaError result = LANA_OK;
    if (file == NULL) { lana_error_set(error, LANA_ERR_IO, 0, OP_NOP, 0, "cannot open %s", path); return LANA_ERR_IO; }
    lana_chunk_init(chunk);
    while (fgets(buffer, sizeof(buffer), file) != NULL) {
        char *tokens[16], *cursor, *comment, *token; size_t count = 0, length;
        ++line; comment = strchr(buffer, '#'); if (comment != NULL) *comment = '\0'; cursor = trim(buffer);
        if (*cursor == '\0') continue;
        length = strlen(cursor);
        if (cursor[length - 1u] == ':') {
            cursor[length - 1u] = '\0';
            if (label_count >= LANA_ASSEMBLER_MAX_FIXUPS || !copy_name(labels[label_count].name, trim(cursor))) { result = LANA_ERR_LIMIT; break; }
            labels[label_count++].offset = (uint32_t)chunk->code_count; continue;
        }
        token = strtok(cursor, " \t\r\n,");
        while (token != NULL && count < 16u) { tokens[count++] = token; token = strtok(NULL, " \t\r\n,"); }
        if (token != NULL || count == 0) { result = LANA_ERR_FORMAT; break; }
        if (strcmp(tokens[0], ".function") == 0) {
            char *end; unsigned long arity, registers;
            if (count != 4u) { result = LANA_ERR_FORMAT; break; }
            arity = strtoul(tokens[2], &end, 10); if (*end != '\0') { result = LANA_ERR_FORMAT; break; }
            registers = strtoul(tokens[3], &end, 10); if (*end != '\0') { result = LANA_ERR_FORMAT; break; }
            result = lana_chunk_add_function(chunk, tokens[1], (uint32_t)chunk->code_count,
                                           (uint32_t)registers, (uint32_t)arity, NULL);
            if (result != LANA_OK) break;
            continue;
        }
        if (strcmp(tokens[0], ".version") == 0) {
            char *end;
            unsigned long version;
            if (count != 2u) { result = LANA_ERR_FORMAT; break; }
            errno = 0;
            version = strtoul(tokens[1], &end, 10);
            if (errno != 0 || *end != '\0' || version > UINT32_MAX) {
                result = LANA_ERR_FORMAT;
                break;
            }
            if ((uint32_t)version != LABC_VERSION) { result = LANA_ERR_INCOMPATIBLE_FORMAT; break; }
            chunk->version = LABC_VERSION;
            continue;
        }
        if (strcmp(tokens[0], ".line") == 0) {
            char *end; unsigned long parsed;
            if (count != 2u) { result = LANA_ERR_FORMAT; break; }
            parsed = strtoul(tokens[1], &end, 10);
            if (*end != '\0' || parsed > UINT32_MAX) { result = LANA_ERR_FORMAT; break; }
            source_line = (uint32_t)parsed;
            continue;
        }
        result = emit_line(chunk, tokens, count, source_line == 0u ? line : source_line,
                           fixups, &fixup_count, function_fixups,
                           &function_fixup_count, error);
        if (result != LANA_OK) break;
    }
    (void)fclose(file);
    if (result == LANA_OK) {
        size_t fixup_index, index;
        index_clear(label_index);
        for (index = 0u; index < label_count; ++index) {
            if (!label_index_add(label_index, labels, index)) {
                result = LANA_ERR_LIMIT;
                break;
            }
        }
        for (fixup_index = 0; result == LANA_OK &&
                              fixup_index < fixup_count; ++fixup_index) {
            index = label_index_find(label_index, labels,
                                     fixups[fixup_index].name);
            if (index == SIZE_MAX) { lana_error_set(error, LANA_ERR_JUMP, fixups[fixup_index].instruction, OP_JUMP, 0, "unknown label %s", fixups[fixup_index].name); result = LANA_ERR_JUMP; break; }
            chunk->code[fixups[fixup_index].instruction].imm =
                labels[index].offset;
        }
    }
    if (result == LANA_OK) {
        size_t fixup_index, index;
        index_clear(function_index);
        for (index = 0u; index < chunk->function_count; ++index) {
            if (!function_index_add(function_index, chunk, index)) {
                result = LANA_ERR_LIMIT;
                break;
            }
        }
        for (fixup_index = 0; result == LANA_OK &&
                              fixup_index < function_fixup_count; ++fixup_index) {
            index = function_index_find(function_index, chunk,
                                        function_fixups[fixup_index].name);
            if (index == SIZE_MAX) {
                lana_error_set(error, LANA_ERR_FORMAT, function_fixups[fixup_index].instruction,
                             OP_CALL, 0, "unknown function %s",
                             function_fixups[fixup_index].name);
                result = LANA_ERR_FORMAT; break;
            }
            chunk->code[function_fixups[fixup_index].instruction].b =
                (uint32_t)index;
        }
    }
    if (result == LANA_OK) result = lana_chunk_verify(chunk, error);
    if (result != LANA_OK) {
        if (error != NULL && error->code == LANA_OK)
            lana_error_set(error, result, chunk->code_count, OP_NOP, line,
                         "assembly failed near source line %u", line);
        lana_chunk_free(chunk);
    }
    return result;
}
