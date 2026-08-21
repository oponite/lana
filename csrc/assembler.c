#include "ss/assembler.h"
#include "ss/vm.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    if (errno != 0 || *end != '\0' || value >= SS_MAX_REGISTERS) return false;
    *out = (uint32_t)value; return true;
}

static bool parse_number(const char *text, double *out) {
    char *end;
    errno = 0; *out = strtod(text, &end);
    return errno == 0 && end != text && *end == '\0';
}

static SSError add_number(SSChunk *chunk, double number, uint32_t *index) {
    return ss_chunk_add_constant(chunk, ss_value_number(number), index);
}

static SSError number_constant(SSChunk *chunk, const char *text, uint32_t *index) {
    char *end;
    unsigned long parsed;
    double number;
    if ((text[0] == 'K' || text[0] == 'k') && text[1] != '\0') {
        errno = 0;
        parsed = strtoul(text + 1, &end, 10);
        if (errno != 0 || *end != '\0' || parsed >= chunk->constant_count ||
            chunk->constants[parsed].type != VAL_NUMBER) return SS_ERR_CONSTANT;
        *index = (uint32_t)parsed;
        return SS_OK;
    }
    if (!parse_number(text, &number)) return SS_ERR_FORMAT;
    return add_number(chunk, number, index);
}

static SSError string_constant(SSChunk *chunk, const char *text, uint32_t *index) {
    char *end;
    unsigned long parsed;
    if (text == NULL || *text == '\0') return SS_ERR_FORMAT;
    if ((text[0] != 'S' && text[0] != 's') ||
        !isdigit((unsigned char)text[1]))
        return ss_chunk_add_constant(chunk, ss_value_string(text), index);
    errno = 0;
    parsed = strtoul(text + 1, &end, 10);
    if (errno != 0 || *end != '\0' || parsed >= chunk->constant_count ||
        chunk->constants[parsed].type != VAL_STRING) return SS_ERR_CONSTANT;
    *index = (uint32_t)parsed;
    return SS_OK;
}

static int measure_id(const char *name) {
    if (strcmp(name, "distribution") == 0) return SS_MEASURE_DISTRIBUTION;
    if (strcmp(name, "probability") == 0) return SS_MEASURE_PROBABILITY;
    if (strcmp(name, "sample") == 0) return SS_MEASURE_SAMPLE;
    if (strcmp(name, "collapse") == 0) return SS_MEASURE_COLLAPSE;
    return -1;
}

static int transform_id(const char *name) {
    static const char *names[] = {"decay", "reinforce", "invert", "neutralize", "shift", "clamp", "reset_d"};
    size_t index;
    for (index = 0; index < sizeof(names) / sizeof(names[0]); ++index)
        if (strcmp(name, names[index]) == 0) return (int)index;
    return -1;
}

static int transform_id_v3(const char *name) {
    if (strcmp(name, "invert") == 0) return SS_TRANSFORM_V3_INVERT;
    if (strcmp(name, "neutralize") == 0) return SS_TRANSFORM_V3_NEUTRALIZE;
    return -1;
}

static int measure_id_v3(const char *name) {
    if (strcmp(name, "probability") == 0) return SS_MEASURE_V3_PROBABILITY;
    if (strcmp(name, "distribution") == 0) return SS_MEASURE_V3_DISTRIBUTION;
    if (strcmp(name, "sample") == 0) return SS_MEASURE_V3_SAMPLE;
    return -1;
}

static int basis_id(const char *name) {
    if (strcmp(name, "computational") == 0) return SS_MEASURE_BASIS_COMPUTATIONAL;
    if (strcmp(name, "x") == 0) return SS_MEASURE_BASIS_X;
    if (strcmp(name, "y") == 0) return SS_MEASURE_BASIS_Y;
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
    if (strcmp(name, "+") == 0 || strcmp(name, "add") == 0) return SS_BINARY_ADD;
    if (strcmp(name, "-") == 0 || strcmp(name, "sub") == 0) return SS_BINARY_SUBTRACT;
    if (strcmp(name, "*") == 0 || strcmp(name, "mul") == 0) return SS_BINARY_MULTIPLY;
    if (strcmp(name, "/") == 0 || strcmp(name, "div") == 0) return SS_BINARY_DIVIDE;
    return -1;
}

static int compare_id(const char *name) {
    if (strcmp(name, "==") == 0) return SS_COMPARE_EQUAL;
    if (strcmp(name, "!=") == 0) return SS_COMPARE_NOT_EQUAL;
    if (strcmp(name, "<") == 0) return SS_COMPARE_LESS;
    if (strcmp(name, "<=") == 0) return SS_COMPARE_LESS_EQUAL;
    if (strcmp(name, ">") == 0) return SS_COMPARE_GREATER;
    if (strcmp(name, ">=") == 0) return SS_COMPARE_GREATER_EQUAL;
    return -1;
}

static int aggregation_id(const char *name) {
    if (strcmp(name, "weighted") == 0) return SS_AGG_WEIGHTED;
    if (strcmp(name, "mean") == 0) return SS_AGG_MEAN;
    if (strcmp(name, "sequential") == 0) return SS_AGG_SEQUENTIAL;
    if (strcmp(name, "strongest") == 0) return SS_AGG_STRONGEST;
    return -1;
}

static int host_call_id(const char *name) {
    static const char *names[] = {
        "args", "read_text", "write_text", "now", "random", "assert",
        "ml_fit_ridge", "ml_fit_logistic", "ml_predict", "ml_standardize",
        "ml_polynomial_features", "ml_rbf_features", "ml_regression_metrics",
        "ml_classification_metrics", "ml_model_write", "ml_model_read",
        "map_new", "index_get", "index_set", "json_parse", "json_stringify",
        "csv_read", "csv_write", "string_length", "string_byte_at",
        "string_slice", "string_concat", "number_to_string", "array_new",
        "array_push", "string_hex", "string_join", "array_length",
        "string_unescape", "path_resolve"
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

static SSError add_hex_string(SSChunk *chunk, const char *encoded, uint32_t *index) {
    size_t encoded_length = strcmp(encoded, "-") == 0 ? 0u : strlen(encoded);
    size_t output_index;
    char *decoded;
    SSError result;
    if (encoded_length % 2u != 0u) return SS_ERR_FORMAT;
    decoded = malloc(encoded_length / 2u + 1u);
    if (decoded == NULL) return SS_ERR_OOM;
    for (output_index = 0; output_index < encoded_length / 2u; ++output_index) {
        int high = hex_digit(encoded[output_index * 2u]);
        int low = hex_digit(encoded[output_index * 2u + 1u]);
        if (high < 0 || low < 0) { free(decoded); return SS_ERR_FORMAT; }
        decoded[output_index] = (char)((high << 4) | low);
        if (decoded[output_index] == '\0') { free(decoded); return SS_ERR_FORMAT; }
    }
    decoded[encoded_length / 2u] = '\0';
    result = ss_chunk_add_constant(chunk, ss_value_string(decoded), index);
    free(decoded);
    return result;
}

static bool copy_name(char destination[64], const char *source) {
    size_t length = strlen(source);
    if (length == 0 || length >= 64u) return false;
    memcpy(destination, source, length + 1u); return true;
}

static SSError emit_line(SSChunk *chunk, char **tokens, size_t count, uint32_t line,
                         Fixup *fixups, size_t *fixup_count,
                         FunctionFixup *function_fixups,
                         size_t *function_fixup_count, SSErrorInfo *error) {
    SSInstruction ins = {.a = SS_NO_OPERAND, .b = SS_NO_OPERAND, .c = SS_NO_OPERAND,
                         .imm = SS_NO_OPERAND, .line = line};
    uint32_t a, b, c;
    int id;
    double number;
    SSError result;
#define EXPECT(n) do { if (count != (n)) { ss_error_set(error, SS_ERR_FORMAT, chunk->code_count, OP_NOP, line, "%s expects %u operands", tokens[0], (unsigned)((n)-1u)); return SS_ERR_FORMAT; } } while (0)
#define REG(token, destination) do { if (!parse_register((token), &(destination))) { ss_error_set(error, SS_ERR_REGISTER, chunk->code_count, OP_NOP, line, "invalid register %s", (token)); return SS_ERR_REGISTER; } } while (0)
    if (strcmp(tokens[0], "NOP") == 0) { EXPECT(1); ins.opcode = OP_NOP; }
    else if (strcmp(tokens[0], "HALT") == 0) { EXPECT(1); ins.opcode = OP_HALT; }
    else if (strcmp(tokens[0], "LOAD_CONST") == 0) {
        EXPECT(3); REG(tokens[1], a); ins.opcode = OP_LOAD_CONST; ins.a = a;
        if (strcmp(tokens[2], "true") == 0) result = ss_chunk_add_constant(chunk, ss_value_bool(true), &ins.imm);
        else if (strcmp(tokens[2], "false") == 0) result = ss_chunk_add_constant(chunk, ss_value_bool(false), &ins.imm);
        else if (strcmp(tokens[2], "null") == 0) result = ss_chunk_add_constant(chunk, ss_value_null(), &ins.imm);
        else if (parse_number(tokens[2], &number)) result = add_number(chunk, number, &ins.imm);
        else result = ss_chunk_add_constant(chunk, ss_value_string(tokens[2]), &ins.imm);
        if (result != SS_OK) return result;
    } else if (strcmp(tokens[0], "LOAD_STRING") == 0) {
        EXPECT(3); REG(tokens[1], a); ins.opcode = OP_LOAD_CONST; ins.a = a;
        result = add_hex_string(chunk, tokens[2], &ins.imm);
        if (result != SS_OK) return result;
    } else if (strcmp(tokens[0], "STATE_NEW") == 0) {
        EXPECT(4); REG(tokens[1], a);
        if (!parse_number(tokens[2], &number)) return SS_ERR_FORMAT;
        result = add_number(chunk, number, &b); if (result != SS_OK) return result;
        if (!parse_number(tokens[3], &number)) return SS_ERR_FORMAT;
        result = add_number(chunk, number, &c); if (result != SS_OK) return result;
        ins.opcode = OP_STATE_NEW; ins.a = a; ins.b = b; ins.c = c;
    } else if (strcmp(tokens[0], "STATE_BUILD") == 0) {
        EXPECT(4); REG(tokens[1], a); REG(tokens[2], b); REG(tokens[3], c);
        ins.opcode = OP_STATE_BUILD; ins.a = a; ins.b = b; ins.c = c;
    } else if (strcmp(tokens[0], "STATE_NEW_V3") == 0) {
        EXPECT(5);
        REG(tokens[1], a);
        result = number_constant(chunk, tokens[2], &b);
        if (result != SS_OK) return result;
        result = number_constant(chunk, tokens[3], &c);
        if (result != SS_OK) return result;
        result = number_constant(chunk, tokens[4], &ins.imm);
        if (result != SS_OK) return result;
        ins.opcode = OP_STATE_NEW_V3;
        ins.a = a;
        ins.b = b;
        ins.c = c;
    } else if (strcmp(tokens[0], "STATE_BUILD_V3") == 0) {
        EXPECT(5);
        REG(tokens[1], a);
        REG(tokens[2], b);
        REG(tokens[3], c);
        REG(tokens[4], ins.imm);
        ins.opcode = OP_STATE_BUILD_V3;
        ins.a = a;
        ins.b = b;
        ins.c = c;
    } else if (strcmp(tokens[0], "MOVE") == 0 || strcmp(tokens[0], "STATE_SET") == 0 ||
               strcmp(tokens[0], "STATE_COPY") == 0 || strcmp(tokens[0], "APPLY") == 0) {
        EXPECT(3); REG(tokens[1], a); REG(tokens[2], b);
        ins.opcode = strcmp(tokens[0], "MOVE") == 0 ? OP_MOVE : strcmp(tokens[0], "STATE_SET") == 0 ? OP_STATE_SET : strcmp(tokens[0], "STATE_COPY") == 0 ? OP_STATE_COPY : OP_APPLY;
        ins.a = a; ins.b = b;
    } else if (strcmp(tokens[0], "TRANSFORM") == 0) {
        if (count < 3 || count > 5) return SS_ERR_FORMAT;
        REG(tokens[1], a); id = transform_id(tokens[2]); if (id < 0) return SS_ERR_TRANSFORM;
        ins.opcode = OP_TRANSFORM; ins.a = a; ins.b = (uint32_t)id; ins.c = 0; ins.imm = (uint32_t)(count - 3u);
        if (count > 3) { REG(tokens[3], ins.c); if (count > 4) { REG(tokens[4], b); if (b != ins.c + 1u) return SS_ERR_REGISTER; } }
    } else if (strcmp(tokens[0], "TRANSFORM_V3") == 0) {
        EXPECT(4);
        REG(tokens[1], a);
        REG(tokens[2], b);
        id = transform_id_v3(tokens[3]);
        if (id < 0) return SS_ERR_TRANSFORM;
        ins.opcode = OP_TRANSFORM_V3;
        ins.a = a;
        ins.b = b;
        ins.c = (uint32_t)id;
        ins.imm = 0u;
    } else if (strcmp(tokens[0], "APPLY_MANY") == 0) {
        EXPECT(5); REG(tokens[1], a); number = strtod(tokens[2], NULL); REG(tokens[3], c); id = aggregation_id(tokens[4]);
        if (number < 1.0 || number > SS_MAX_REGISTERS || floor(number) != number || id < 0) return SS_ERR_FORMAT;
        ins.opcode = OP_APPLY_MANY; ins.a = a; ins.b = (uint32_t)number; ins.c = c; ins.imm = (uint32_t)id;
    } else if (strcmp(tokens[0], "COMPOSE_MERGE") == 0 || strcmp(tokens[0], "COMPOSE_JOINT") == 0) {
        EXPECT(4); REG(tokens[1], a); REG(tokens[2], b); REG(tokens[3], c);
        ins.opcode = strcmp(tokens[0], "COMPOSE_MERGE") == 0 ? OP_COMPOSE_MERGE : OP_COMPOSE_JOINT; ins.a = a; ins.b = b; ins.c = c;
    } else if (strcmp(tokens[0], "COMPOSE_UPDATE") == 0) {
        EXPECT(3); REG(tokens[1], a); REG(tokens[2], b); ins.opcode = OP_COMPOSE_UPDATE; ins.a = a; ins.b = b;
    } else if (strcmp(tokens[0], "MEASURE") == 0) {
        EXPECT(4); REG(tokens[1], a); id = measure_id(tokens[2]); REG(tokens[3], b); if (id < 0) return SS_ERR_MEASURE;
        ins.opcode = OP_MEASURE; ins.a = a; ins.b = b; ins.c = (uint32_t)id;
    } else if (strcmp(tokens[0], "MEASURE_V3") == 0) {
        EXPECT(4);
        REG(tokens[1], a);
        id = measure_id_v3(tokens[2]);
        REG(tokens[3], b);
        if (id < 0) return SS_ERR_MEASURE;
        ins.opcode = OP_MEASURE_V3;
        ins.a = a;
        ins.b = b;
        ins.c = (uint32_t)id;
        ins.imm = 0u;
    } else if (strcmp(tokens[0], "MEASURE_BASIS_V4") == 0) {
        EXPECT(5);
        REG(tokens[1], a);
        REG(tokens[2], b);
        id = basis_id(tokens[3]);
        if (id < 0) return SS_ERR_MEASURE;
        result = measure_id_v3(tokens[4]);
        if (result < 0) return SS_ERR_MEASURE;
        ins.opcode = OP_MEASURE_BASIS_V4;
        ins.a = a;
        ins.b = b;
        ins.c = (uint32_t)id;
        ins.imm = (uint32_t)result;
    } else if (strcmp(tokens[0], "ESTIMATE_MEASURE_PROBABILITY_V4") == 0 ||
               strcmp(tokens[0], "ESTIMATE_MEASURE_DISTRIBUTION_V4") == 0) {
        EXPECT(5);
        REG(tokens[1], a);
        REG(tokens[2], b);
        id = basis_id(tokens[3]);
        if (id < 0) return SS_ERR_MEASURE;
        if (!sample_count(tokens[4], &ins.imm)) return SS_ERR_FORMAT;
        ins.opcode = strcmp(tokens[0], "ESTIMATE_MEASURE_PROBABILITY_V4") == 0
                         ? OP_ESTIMATE_MEASURE_PROBABILITY_V4
                         : OP_ESTIMATE_MEASURE_DISTRIBUTION_V4;
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
    } else if (strcmp(tokens[0], "JOINT_BUILD_V5") == 0) {
        uint32_t joint_count;
        EXPECT(5); REG(tokens[1], a); REG(tokens[2], b);
        if (!sample_count(tokens[3], &joint_count) || b + joint_count > SS_MAX_REGISTERS)
            return SS_ERR_REGISTER;
        result = string_constant(chunk, tokens[4], &ins.imm);
        if (result != SS_OK) return result;
        ins.opcode = OP_JOINT_BUILD_V5; ins.a = a; ins.b = b; ins.c = joint_count;
    } else if (strcmp(tokens[0], "JOINT_PROJECT_V5") == 0) {
        EXPECT(4); REG(tokens[1], a); REG(tokens[2], b);
        result = string_constant(chunk, tokens[3], &ins.c);
        if (result != SS_OK) return result;
        ins.opcode = OP_JOINT_PROJECT_V5; ins.a = a; ins.b = b;
    } else if (strcmp(tokens[0], "JOINT_CONDITION_V5") == 0) {
        EXPECT(5); REG(tokens[1], a); REG(tokens[2], b);
        result = string_constant(chunk, tokens[3], &ins.c);
        if (result != SS_OK) return result;
        REG(tokens[4], ins.imm);
        ins.opcode = OP_JOINT_CONDITION_V5; ins.a = a; ins.b = b;
    } else if (strcmp(tokens[0], "JOINT_BUILD_FINITE_V5") == 0) {
        EXPECT(4); REG(tokens[1], a); REG(tokens[2], b);
        result = string_constant(chunk, tokens[3], &ins.c);
        if (result != SS_OK) return result;
        ins.opcode = OP_JOINT_BUILD_FINITE_V5; ins.a = a; ins.b = b; ins.imm = 0u;
    } else if (strcmp(tokens[0], "JOINT_RENAME_V5") == 0) {
        EXPECT(5); REG(tokens[1], a); REG(tokens[2], b);
        result = string_constant(chunk, tokens[3], &ins.c);
        if (result != SS_OK) return result;
        result = string_constant(chunk, tokens[4], &ins.imm);
        if (result != SS_OK) return result;
        ins.opcode = OP_JOINT_RENAME_V5; ins.a = a; ins.b = b;
    } else if (strcmp(tokens[0], "POSSIBILITY_BUILD_V5") == 0 ||
               strcmp(tokens[0], "INFO_SAMPLE_V5") == 0) {
        EXPECT(3); REG(tokens[1], a); REG(tokens[2], b);
        ins.opcode = strcmp(tokens[0], "POSSIBILITY_BUILD_V5") == 0
            ? OP_POSSIBILITY_BUILD_V5 : OP_INFO_SAMPLE_V5;
        ins.a = a; ins.b = b; ins.c = 0u; ins.imm = 0u;
    } else if (strcmp(tokens[0], "PATH_SPLIT_V5") == 0) {
        EXPECT(3); REG(tokens[1], a);
        ins.opcode = OP_PATH_SPLIT_V5; ins.a = a; ins.b = 0u; ins.c = 0u;
        if (*fixup_count >= 1024u ||
            !copy_name(fixups[*fixup_count].name, tokens[2])) return SS_ERR_LIMIT;
        fixups[*fixup_count].instruction = (uint32_t)chunk->code_count;
        ++*fixup_count;
    } else if (strcmp(tokens[0], "PATH_JOIN_V5") == 0) {
        EXPECT(1); ins.opcode = OP_PATH_JOIN_V5;
        ins.a = 0u; ins.b = 0u; ins.c = 0u; ins.imm = 0u;
    } else if (strcmp(tokens[0], "OBSERVE_V5") == 0) {
        EXPECT(5); REG(tokens[1], a); REG(tokens[2], b);
        result = string_constant(chunk, tokens[3], &ins.c);
        if (result != SS_OK) return result;
        REG(tokens[4], ins.imm);
        ins.opcode = OP_OBSERVE_V5; ins.a = a; ins.b = b;
    } else if (strcmp(tokens[0], "JOINT_SAMPLE_V5") == 0 ||
               strcmp(tokens[0], "RESOLVE_V5") == 0) {
        EXPECT(3); REG(tokens[1], a); REG(tokens[2], b);
        ins.opcode = strcmp(tokens[0], "JOINT_SAMPLE_V5") == 0
                         ? OP_JOINT_SAMPLE_V5 : OP_RESOLVE_V5;
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
        EXPECT(4); REG(tokens[1], a); id = strcmp(tokens[0], "GET_FIELD") == 0 ? field_id(tokens[2]) : index_id(tokens[2]); REG(tokens[3], b); if (id < 0) return SS_ERR_FORMAT;
        ins.opcode = strcmp(tokens[0], "GET_FIELD") == 0 ? OP_GET_FIELD : OP_GET_INDEX; ins.a = a; ins.b = b; ins.c = (uint32_t)id;
    } else if (strcmp(tokens[0], "SET_INDEX") == 0) {
        EXPECT(4); REG(tokens[1], a); id = index_id(tokens[2]); REG(tokens[3], c); if (id < 0) return SS_ERR_FORMAT;
        ins.opcode = OP_SET_INDEX; ins.a = a; ins.b = (uint32_t)id; ins.c = c;
    } else if (strcmp(tokens[0], "HISTORY") == 0) {
        EXPECT(4); REG(tokens[1], a); REG(tokens[3], b);
        id = strcmp(tokens[2], "latest") == 0 ? SS_HISTORY_LATEST : strcmp(tokens[2], "duration") == 0 ? SS_HISTORY_DURATION : -1;
        if (id < 0) return SS_ERR_HISTORY;
        ins.opcode = OP_HISTORY_CONFIG;
        ins.a = a;
        ins.b = b;
        ins.c = (uint32_t)id;
    } else if (strcmp(tokens[0], "PREVIOUS") == 0 || strcmp(tokens[0], "CHANGE") == 0 || strcmp(tokens[0], "VELOCITY") == 0) {
        EXPECT(3); REG(tokens[1], a); REG(tokens[2], b);
        ins.opcode = strcmp(tokens[0], "PREVIOUS") == 0 ? OP_PREVIOUS : strcmp(tokens[0], "CHANGE") == 0 ? OP_CHANGE : OP_VELOCITY; ins.a = a; ins.b = b;
    } else if (strcmp(tokens[0], "BINARY") == 0 || strcmp(tokens[0], "COMPARE") == 0) {
        EXPECT(5); REG(tokens[1], a); id = strcmp(tokens[0], "BINARY") == 0 ? binary_id(tokens[2]) : compare_id(tokens[2]); REG(tokens[3], b); REG(tokens[4], c); if (id < 0) return SS_ERR_FORMAT;
        ins.opcode = strcmp(tokens[0], "BINARY") == 0 ? OP_BINARY : OP_COMPARE; ins.a = a; ins.b = b; ins.c = c; ins.imm = (uint32_t)id;
    } else if (strcmp(tokens[0], "UNARY") == 0) {
        EXPECT(4); REG(tokens[2], a); REG(tokens[3], b); id = strcmp(tokens[1], "-") == 0 ? 0 : strcmp(tokens[1], "!") == 0 ? 1 : -1; if (id < 0) return SS_ERR_FORMAT;
        ins.opcode = OP_UNARY; ins.a = a; ins.b = b; ins.imm = (uint32_t)id;
    } else if (strcmp(tokens[0], "JUMP") == 0 || strcmp(tokens[0], "JUMP_IF_TRUE") == 0 || strcmp(tokens[0], "JUMP_IF_FALSE") == 0) {
        size_t label_token = 1;
        if (strcmp(tokens[0], "JUMP") == 0) { EXPECT(2); ins.opcode = OP_JUMP; }
        else { EXPECT(3); REG(tokens[1], a); ins.a = a; label_token = 2; ins.opcode = strcmp(tokens[0], "JUMP_IF_TRUE") == 0 ? OP_JUMP_IF_TRUE : OP_JUMP_IF_FALSE; }
        if (*fixup_count >= 1024u || !copy_name(fixups[*fixup_count].name, tokens[label_token])) return SS_ERR_LIMIT;
        fixups[*fixup_count].instruction = (uint32_t)chunk->code_count; ++*fixup_count;
    } else if (strcmp(tokens[0], "ARRAY_NEW") == 0) {
        EXPECT(4); REG(tokens[1], a); REG(tokens[2], b); number = strtod(tokens[3], NULL);
        if (number < 0 || number > SS_MAX_REGISTERS) return SS_ERR_LIMIT;
        ins.opcode = OP_ARRAY_NEW; ins.a = a; ins.b = b; ins.c = (uint32_t)number;
    } else if (strcmp(tokens[0], "ARRAY_GET") == 0 || strcmp(tokens[0], "ARRAY_SET") == 0) {
        EXPECT(4); REG(tokens[1], a); REG(tokens[2], b); REG(tokens[3], c);
        ins.opcode = strcmp(tokens[0], "ARRAY_GET") == 0 ? OP_ARRAY_GET : OP_ARRAY_SET; ins.a = a; ins.b = b; ins.c = c;
    } else if (strcmp(tokens[0], "CALL") == 0) {
        size_t function_index;
        EXPECT(5); REG(tokens[2], a); number = strtod(tokens[3], NULL); REG(tokens[4], b);
        for (function_index = 0; function_index < chunk->function_count; ++function_index)
            if (strcmp(tokens[1], chunk->functions[function_index].name) == 0) break;
        if (number < 0.0 || number > SS_MAX_REGISTERS) return SS_ERR_FORMAT;
        if (function_index == chunk->function_count) {
            if (*function_fixup_count >= 1024u ||
                strlen(tokens[1]) >= sizeof(function_fixups[0].name)) return SS_ERR_LIMIT;
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
        if (number < 0.0 || number > SS_MAX_REGISTERS || floor(number) != number) return SS_ERR_FORMAT;
        if (function_index == chunk->function_count) {
            if (*function_fixup_count >= 1024u ||
                strlen(tokens[1]) >= sizeof(function_fixups[0].name)) return SS_ERR_LIMIT;
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
        if (id < 0 || number < 0.0 || number > SS_MAX_REGISTERS || floor(number) != number) return SS_ERR_FORMAT;
        ins.opcode = OP_HOST_CALL; ins.a = b; ins.b = (uint32_t)id; ins.c = a; ins.imm = (uint32_t)number;
    } else if (strcmp(tokens[0], "RETURN") == 0 || strcmp(tokens[0], "PRINT") == 0) {
        EXPECT(2); REG(tokens[1], a); ins.opcode = strcmp(tokens[0], "RETURN") == 0 ? OP_RETURN : OP_PRINT; ins.a = a;
    } else {
        ss_error_set(error, SS_ERR_OPCODE, chunk->code_count, OP_NOP, line, "unknown instruction %s", tokens[0]); return SS_ERR_OPCODE;
    }
    result = ss_chunk_emit(chunk, ins);
    return result;
#undef EXPECT
#undef REG
}

SSError ss_assemble_file(const char *path, SSChunk *chunk, SSErrorInfo *error) {
    FILE *file = fopen(path, "r");
    char buffer[4096];
    uint32_t line = 0;
    uint32_t source_line = 0;
    Label labels[1024]; size_t label_count = 0;
    Fixup fixups[1024]; size_t fixup_count = 0;
    FunctionFixup function_fixups[1024]; size_t function_fixup_count = 0;
    SSError result = SS_OK;
    if (file == NULL) { ss_error_set(error, SS_ERR_IO, 0, OP_NOP, 0, "cannot open %s", path); return SS_ERR_IO; }
    ss_chunk_init(chunk);
    while (fgets(buffer, sizeof(buffer), file) != NULL) {
        char *tokens[16], *cursor, *comment, *token; size_t count = 0, length;
        ++line; comment = strchr(buffer, '#'); if (comment != NULL) *comment = '\0'; cursor = trim(buffer);
        if (*cursor == '\0') continue;
        length = strlen(cursor);
        if (cursor[length - 1u] == ':') {
            cursor[length - 1u] = '\0';
            if (label_count >= 1024u || !copy_name(labels[label_count].name, trim(cursor))) { result = SS_ERR_LIMIT; break; }
            labels[label_count++].offset = (uint32_t)chunk->code_count; continue;
        }
        token = strtok(cursor, " \t\r\n,");
        while (token != NULL && count < 16u) { tokens[count++] = token; token = strtok(NULL, " \t\r\n,"); }
        if (token != NULL || count == 0) { result = SS_ERR_FORMAT; break; }
        if (strcmp(tokens[0], ".function") == 0) {
            char *end; unsigned long arity, registers;
            if (count != 4u) { result = SS_ERR_FORMAT; break; }
            arity = strtoul(tokens[2], &end, 10); if (*end != '\0') { result = SS_ERR_FORMAT; break; }
            registers = strtoul(tokens[3], &end, 10); if (*end != '\0') { result = SS_ERR_FORMAT; break; }
            result = ss_chunk_add_function(chunk, tokens[1], (uint32_t)chunk->code_count,
                                           (uint32_t)registers, (uint32_t)arity, NULL);
            if (result != SS_OK) break;
            continue;
        }
        if (strcmp(tokens[0], ".version") == 0) {
            char *end;
            unsigned long version;
            if (count != 2u) { result = SS_ERR_FORMAT; break; }
            errno = 0;
            version = strtoul(tokens[1], &end, 10);
            if (errno != 0 || *end != '\0' || version > UINT32_MAX) {
                result = SS_ERR_FORMAT;
                break;
            }
            chunk->version = (uint32_t)version;
            continue;
        }
        if (strcmp(tokens[0], ".line") == 0) {
            char *end; unsigned long parsed;
            if (count != 2u) { result = SS_ERR_FORMAT; break; }
            parsed = strtoul(tokens[1], &end, 10);
            if (*end != '\0' || parsed > UINT32_MAX) { result = SS_ERR_FORMAT; break; }
            source_line = (uint32_t)parsed;
            continue;
        }
        result = emit_line(chunk, tokens, count, source_line == 0u ? line : source_line,
                           fixups, &fixup_count, function_fixups,
                           &function_fixup_count, error);
        if (result != SS_OK) break;
    }
    (void)fclose(file);
    if (result == SS_OK) {
        size_t fixup_index, label_index;
        for (fixup_index = 0; fixup_index < fixup_count; ++fixup_index) {
            bool found = false;
            for (label_index = 0; label_index < label_count; ++label_index)
                if (strcmp(fixups[fixup_index].name, labels[label_index].name) == 0) {
                    chunk->code[fixups[fixup_index].instruction].imm = labels[label_index].offset; found = true; break;
                }
            if (!found) { ss_error_set(error, SS_ERR_JUMP, fixups[fixup_index].instruction, OP_JUMP, 0, "unknown label %s", fixups[fixup_index].name); result = SS_ERR_JUMP; break; }
        }
    }
    if (result == SS_OK) {
        size_t fixup_index, function_index;
        for (fixup_index = 0; fixup_index < function_fixup_count; ++fixup_index) {
            for (function_index = 0; function_index < chunk->function_count; ++function_index)
                if (strcmp(function_fixups[fixup_index].name,
                           chunk->functions[function_index].name) == 0) break;
            if (function_index == chunk->function_count) {
                ss_error_set(error, SS_ERR_FORMAT, function_fixups[fixup_index].instruction,
                             OP_CALL, 0, "unknown function %s",
                             function_fixups[fixup_index].name);
                result = SS_ERR_FORMAT; break;
            }
            chunk->code[function_fixups[fixup_index].instruction].b =
                (uint32_t)function_index;
        }
    }
    if (result == SS_OK) result = ss_chunk_verify(chunk, error);
    if (result != SS_OK) {
        if (error != NULL && error->code == SS_OK)
            ss_error_set(error, result, chunk->code_count, OP_NOP, line,
                         "assembly failed near source line %u", line);
        ss_chunk_free(chunk);
    }
    return result;
}
