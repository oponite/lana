#include "lana/bytecode.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

static char *copy_string(const char *source) {
    size_t length = strlen(source);
    char *copy = malloc(length + 1u);
    if (copy != NULL) {
        memcpy(copy, source, length + 1u);
    }
    return copy;
}

void lana_chunk_init(LanaChunk *chunk) {
    memset(chunk, 0, sizeof(*chunk));
    chunk->version = LABC_VERSION;
}

void lana_chunk_free(LanaChunk *chunk) {
    size_t index;
    if (chunk == NULL) return;
    for (index = 0; index < chunk->constant_count; ++index) {
        if (chunk->constants[index].type == VAL_STRING) {
            free((void *)chunk->constants[index].as.string);
        }
    }
    for (index = 0; index < chunk->function_count; ++index) {
        free(chunk->functions[index].name);
    }
    free(chunk->constants);
    free(chunk->functions);
    free(chunk->code);
    lana_chunk_init(chunk);
}

LanaError lana_chunk_add_constant(LanaChunk *chunk, Value value, uint32_t *index) {
    Value *constants;
    if (chunk->constant_count == UINT32_MAX) return LANA_ERR_LIMIT;
    if (chunk->constant_count == chunk->constant_capacity) {
        size_t capacity = chunk->constant_capacity == 0 ? 16u : chunk->constant_capacity * 2u;
        constants = realloc(chunk->constants, capacity * sizeof(*constants));
        if (constants == NULL) return LANA_ERR_OOM;
        chunk->constants = constants;
        chunk->constant_capacity = capacity;
    }
    if (value.type == VAL_STRING) {
        char *string = copy_string(value.as.string);
        if (string == NULL) return LANA_ERR_OOM;
        value.as.string = string;
    }
    chunk->constants[chunk->constant_count] = value;
    if (index != NULL) *index = (uint32_t)chunk->constant_count;
    ++chunk->constant_count;
    return LANA_OK;
}

LanaError lana_chunk_add_function(LanaChunk *chunk, const char *name, uint32_t entry,
                              uint32_t register_count, uint32_t arity,
                              uint32_t *index) {
    LanaFunction *functions;
    char *owned_name;
    if (name == NULL || register_count > LANA_MAX_REGISTERS || arity > register_count ||
        chunk->function_count == UINT32_MAX) return LANA_ERR_FORMAT;
    if (chunk->function_count == chunk->function_capacity) {
        size_t capacity = chunk->function_capacity == 0 ? 8u : chunk->function_capacity * 2u;
        functions = realloc(chunk->functions, capacity * sizeof(*functions));
        if (functions == NULL) return LANA_ERR_OOM;
        chunk->functions = functions; chunk->function_capacity = capacity;
    }
    owned_name = copy_string(name);
    if (owned_name == NULL) return LANA_ERR_OOM;
    chunk->functions[chunk->function_count] = (LanaFunction){owned_name, entry, register_count, arity};
    if (index != NULL) *index = (uint32_t)chunk->function_count;
    ++chunk->function_count;
    return LANA_OK;
}

LanaError lana_chunk_emit(LanaChunk *chunk, LanaInstruction instruction) {
    LanaInstruction *code;
    if (chunk->code_count == chunk->code_capacity) {
        size_t capacity = chunk->code_capacity == 0 ? 32u : chunk->code_capacity * 2u;
        code = realloc(chunk->code, capacity * sizeof(*code));
        if (code == NULL) return LANA_ERR_OOM;
        chunk->code = code;
        chunk->code_capacity = capacity;
    }
    chunk->code[chunk->code_count++] = instruction;
    return LANA_OK;
}

const char *lana_opcode_name(uint8_t opcode) {
    static const char *names[] = {
        "NOP", "LOAD_CONST", "MOVE", "STATE_NEW", "STATE_BUILD",
        "TRANSFORM", "MEASURE", "APPEND", "SAMPLE_STATE_DIST",
        "MEASURE_BASIS", "ESTIMATE_MEASURE_PROBABILITY",
        "ESTIMATE_MEASURE_DISTRIBUTION", "GET_FIELD", "GET_INDEX", "SET_INDEX",
        "HISTORY_CONFIG", "PREVIOUS", "CHANGE", "VELOCITY", "BINARY", "UNARY",
        "COMPARE", "JUMP", "JUMP_IF_TRUE", "JUMP_IF_FALSE", "ARRAY_NEW",
        "ARRAY_GET", "ARRAY_SET", "CALL", "RETURN", "PRINT", "HALT",
        "FORK", "JOIN", "JOIN_TIMEOUT", "JOIN_ALL", "CANCEL",
        "TASKGROUP_ENTER", "TASKGROUP_EXIT", "HOST_CALL",
        "JOINT_BUILD", "JOINT_PROJECT", "JOINT_CONDITION",
        "JOINT_SAMPLE", "RESOLVE", "JOINT_BUILD_FINITE",
        "JOINT_RENAME", "POSSIBILITY_BUILD", "PATH_SPLIT",
        "PATH_JOIN", "OBSERVE", "INFO_SAMPLE", "EVIDENCE",
        "ASSUME", "DERIVATION", "EXPLAIN"
    };
    return opcode < OP_COUNT ? names[opcode] : "UNKNOWN";
}

static bool register_valid(uint32_t value) { return value < LANA_MAX_REGISTERS; }
static bool constant_valid(const LanaChunk *chunk, uint32_t value) {
    return value < chunk->constant_count;
}

static LanaError verify_register(LanaErrorInfo *error, size_t ip,
                               const LanaInstruction *instruction, uint32_t reg) {
    if (!register_valid(reg)) {
        lana_error_set(error, LANA_ERR_REGISTER, ip, instruction->opcode,
                     instruction->line, "register R%u is out of range", reg);
        return LANA_ERR_REGISTER;
    }
    return LANA_OK;
}

LanaError lana_chunk_verify(const LanaChunk *chunk, LanaErrorInfo *error) {
    size_t ip, function_index;
    if (error != NULL) memset(error, 0, sizeof(*error));
    if (chunk == NULL || chunk->code_count == 0 || chunk->entry >= chunk->code_count) {
        lana_error_set(error, LANA_ERR_FORMAT, 0, OP_NOP, 0, "chunk has no valid entry point");
        return LANA_ERR_FORMAT;
    }
    if (chunk->version != LABC_VERSION) {
        lana_error_set(error, LANA_ERR_FORMAT, 0, OP_NOP, 0,
                     "unsupported LABC version %u", chunk->version);
        return LANA_ERR_FORMAT;
    }
    for (function_index = 0; function_index < chunk->function_count; ++function_index) {
        const LanaFunction *function = &chunk->functions[function_index];
        if (function->entry >= chunk->code_count || function->register_count > LANA_MAX_REGISTERS ||
            function->arity > function->register_count) {
            lana_error_set(error, LANA_ERR_FORMAT, function->entry, OP_CALL, 0,
                         "function %zu has invalid metadata", function_index);
            return LANA_ERR_FORMAT;
        }
    }
    for (ip = 0; ip < chunk->code_count; ++ip) {
        const LanaInstruction *ins = &chunk->code[ip];
        LanaError result = LANA_OK;
        if (ins->opcode >= OP_COUNT) {
            lana_error_set(error, LANA_ERR_OPCODE, ip, ins->opcode, ins->line,
                         "unknown opcode %u", ins->opcode);
            return LANA_ERR_OPCODE;
        }
        switch ((OpCode)ins->opcode) {
            case OP_NOP: case OP_HALT: case OP_TASKGROUP_ENTER: case OP_TASKGROUP_EXIT: break;
            case OP_LOAD_CONST:
                result = verify_register(error, ip, ins, ins->a);
                if (result == LANA_OK && !constant_valid(chunk, ins->imm)) {
                    lana_error_set(error, LANA_ERR_CONSTANT, ip, ins->opcode, ins->line,
                                 "constant %u is out of range", ins->imm);
                    result = LANA_ERR_CONSTANT;
                }
                break;
            case OP_STATE_NEW: {
                LanaState state;
                result = verify_register(error, ip, ins, ins->a);
                if (result == LANA_OK && (!constant_valid(chunk, ins->b) ||
                    !constant_valid(chunk, ins->c) || !constant_valid(chunk, ins->imm)))
                    result = LANA_ERR_CONSTANT;
                if (result == LANA_OK && (chunk->constants[ins->b].type != VAL_NUMBER ||
                    chunk->constants[ins->c].type != VAL_NUMBER ||
                    chunk->constants[ins->imm].type != VAL_NUMBER)) result = LANA_ERR_TYPE;
                if (result == LANA_OK)
                    result = lana_state_make_complex(chunk->constants[ins->b].as.number,
                                                   chunk->constants[ins->c].as.number,
                                                   chunk->constants[ins->imm].as.number,
                                                   &state);
                break;
            }
            case OP_STATE_BUILD:
                result = verify_register(error, ip, ins, ins->a);
                if (result == LANA_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == LANA_OK) result = verify_register(error, ip, ins, ins->c);
                if (result == LANA_OK) result = verify_register(error, ip, ins, ins->imm);
                break;
            case OP_TRANSFORM:
                result = verify_register(error, ip, ins, ins->a);
                if (result == LANA_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == LANA_OK && ins->c > LANA_TRANSFORM_NEUTRALIZE)
                    result = LANA_ERR_TRANSFORM;
                if (result == LANA_OK && ins->imm != 0u) result = LANA_ERR_FORMAT;
                break;
            case OP_MEASURE:
                result = verify_register(error, ip, ins, ins->a);
                if (result == LANA_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == LANA_OK && ins->c > LANA_MEASURE_SAMPLE)
                    result = LANA_ERR_MEASURE;
                if (result == LANA_OK && ins->imm != 0u) result = LANA_ERR_FORMAT;
                break;
            case OP_APPEND:
                result = verify_register(error, ip, ins, ins->a);
                if (result == LANA_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == LANA_OK) result = verify_register(error, ip, ins, ins->c);
                if (result == LANA_OK && ins->imm != 0u) result = LANA_ERR_FORMAT;
                break;
            case OP_SAMPLE_STATE_DIST:
                result = verify_register(error, ip, ins, ins->a);
                if (result == LANA_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == LANA_OK && (ins->c != 0u || ins->imm != 0u))
                    result = LANA_ERR_FORMAT;
                break;
            case OP_MEASURE_BASIS:
                result = verify_register(error, ip, ins, ins->a);
                if (result == LANA_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == LANA_OK && ins->c > LANA_MEASURE_BASIS_Y)
                    result = LANA_ERR_MEASURE;
                if (result == LANA_OK && ins->imm > LANA_MEASURE_SAMPLE)
                    result = LANA_ERR_MEASURE;
                break;
            case OP_ESTIMATE_MEASURE_PROBABILITY:
            case OP_ESTIMATE_MEASURE_DISTRIBUTION:
                result = verify_register(error, ip, ins, ins->a);
                if (result == LANA_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == LANA_OK && ins->c > LANA_MEASURE_BASIS_Y)
                    result = LANA_ERR_MEASURE;
                if (result == LANA_OK && ins->imm == 0u)
                    result = LANA_ERR_FORMAT;
                break;
            case OP_JOINT_BUILD:
                result = verify_register(error, ip, ins, ins->a);
                if (result == LANA_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == LANA_OK && (ins->c == 0u || ins->b + ins->c > LANA_MAX_REGISTERS))
                    result = LANA_ERR_REGISTER;
                if (result == LANA_OK && !constant_valid(chunk, ins->imm)) result = LANA_ERR_CONSTANT;
                if (result == LANA_OK && chunk->constants[ins->imm].type != VAL_STRING)
                    result = LANA_ERR_TYPE;
                break;
            case OP_JOINT_PROJECT:
                result = verify_register(error, ip, ins, ins->a);
                if (result == LANA_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == LANA_OK && !constant_valid(chunk, ins->c)) result = LANA_ERR_CONSTANT;
                if (result == LANA_OK && chunk->constants[ins->c].type != VAL_STRING)
                    result = LANA_ERR_TYPE;
                break;
            case OP_JOINT_CONDITION:
                result = verify_register(error, ip, ins, ins->a);
                if (result == LANA_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == LANA_OK) result = verify_register(error, ip, ins, ins->imm);
                if (result == LANA_OK && !constant_valid(chunk, ins->c)) result = LANA_ERR_CONSTANT;
                if (result == LANA_OK && chunk->constants[ins->c].type != VAL_STRING)
                    result = LANA_ERR_TYPE;
                break;
            case OP_JOINT_SAMPLE:
            case OP_RESOLVE:
                result = verify_register(error, ip, ins, ins->a);
                if (result == LANA_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == LANA_OK && (ins->c != 0u || ins->imm != 0u))
                    result = LANA_ERR_FORMAT;
                break;
            case OP_JOINT_BUILD_FINITE:
                result = verify_register(error, ip, ins, ins->a);
                if (result == LANA_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == LANA_OK && !constant_valid(chunk, ins->c)) result = LANA_ERR_CONSTANT;
                if (result == LANA_OK && chunk->constants[ins->c].type != VAL_STRING)
                    result = LANA_ERR_TYPE;
                if (result == LANA_OK && ins->imm != 0u) result = LANA_ERR_FORMAT;
                break;
            case OP_JOINT_RENAME:
                result = verify_register(error, ip, ins, ins->a);
                if (result == LANA_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == LANA_OK && (!constant_valid(chunk, ins->c) ||
                                        !constant_valid(chunk, ins->imm)))
                    result = LANA_ERR_CONSTANT;
                if (result == LANA_OK && (chunk->constants[ins->c].type != VAL_STRING ||
                                        chunk->constants[ins->imm].type != VAL_STRING))
                    result = LANA_ERR_TYPE;
                break;
            case OP_POSSIBILITY_BUILD:
            case OP_INFO_SAMPLE:
                result = verify_register(error, ip, ins, ins->a);
                if (result == LANA_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == LANA_OK && (ins->c != 0u || ins->imm != 0u))
                    result = LANA_ERR_FORMAT;
                break;
            case OP_PATH_SPLIT:
                result = verify_register(error, ip, ins, ins->a);
                if (result == LANA_OK && ins->imm >= chunk->code_count) result = LANA_ERR_JUMP;
                if (result == LANA_OK && (ins->b != 0u || ins->c != 0u))
                    result = LANA_ERR_FORMAT;
                break;
            case OP_PATH_JOIN:
                if (ins->a != 0u || ins->b != 0u || ins->c != 0u || ins->imm != 0u)
                    result = LANA_ERR_FORMAT;
                break;
            case OP_OBSERVE:
                result = verify_register(error, ip, ins, ins->a);
                if (result == LANA_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == LANA_OK) result = verify_register(error, ip, ins, ins->imm);
                if (result == LANA_OK && !constant_valid(chunk, ins->c)) result = LANA_ERR_CONSTANT;
                if (result == LANA_OK && chunk->constants[ins->c].type != VAL_STRING)
                    result = LANA_ERR_TYPE;
                break;
            case OP_EVIDENCE:
            case OP_ASSUME:
                result = verify_register(error, ip, ins, ins->a);
                if (result == LANA_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == LANA_OK && !constant_valid(chunk, ins->c)) result = LANA_ERR_CONSTANT;
                if (result == LANA_OK && chunk->constants[ins->c].type != VAL_STRING)
                    result = LANA_ERR_TYPE;
                if (result == LANA_OK && ins->imm != 0u) result = LANA_ERR_FORMAT;
                break;
            case OP_DERIVATION:
            case OP_EXPLAIN:
                result = verify_register(error, ip, ins, ins->a);
                if (result == LANA_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == LANA_OK && (ins->c != 0u || ins->imm != 0u))
                    result = LANA_ERR_FORMAT;
                break;
            case OP_JUMP:
                if (ins->imm >= chunk->code_count) result = LANA_ERR_JUMP;
                if (result != LANA_OK) lana_error_set(error, result, ip, ins->opcode, ins->line, "jump target %u is invalid", ins->imm);
                break;
            case OP_JUMP_IF_TRUE: case OP_JUMP_IF_FALSE:
                result = verify_register(error, ip, ins, ins->a);
                if (result == LANA_OK && ins->imm >= chunk->code_count) {
                    lana_error_set(error, LANA_ERR_JUMP, ip, ins->opcode, ins->line,
                                 "jump target %u is invalid", ins->imm);
                    result = LANA_ERR_JUMP;
                }
                break;
            case OP_MOVE:
            case OP_PREVIOUS: case OP_CHANGE: case OP_VELOCITY: case OP_RETURN:
                result = verify_register(error, ip, ins, ins->a);
                if (result == LANA_OK && ins->opcode != OP_RETURN)
                    result = verify_register(error, ip, ins, ins->b);
                break;
            case OP_ARRAY_GET: case OP_ARRAY_SET:
                result = verify_register(error, ip, ins, ins->a);
                if (result == LANA_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == LANA_OK) result = verify_register(error, ip, ins, ins->c);
                break;
            case OP_GET_FIELD: case OP_GET_INDEX: case OP_SET_INDEX:
            case OP_HISTORY_CONFIG: case OP_BINARY: case OP_COMPARE:
                result = verify_register(error, ip, ins, ins->a);
                if (result == LANA_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == LANA_OK && (ins->opcode == OP_BINARY || ins->opcode == OP_COMPARE || ins->opcode == OP_SET_INDEX))
                    result = verify_register(error, ip, ins, ins->c);
                break;
            case OP_UNARY: case OP_PRINT:
                result = verify_register(error, ip, ins, ins->a);
                if (result == LANA_OK && ins->opcode == OP_UNARY) result = verify_register(error, ip, ins, ins->b);
                break;
            case OP_ARRAY_NEW:
                result = verify_register(error, ip, ins, ins->a);
                if (result == LANA_OK && (ins->b >= LANA_MAX_REGISTERS || ins->b + ins->c > LANA_MAX_REGISTERS)) result = LANA_ERR_REGISTER;
                break;
            case OP_CALL:
                result = verify_register(error, ip, ins, ins->a);
                if (result == LANA_OK && ins->b >= chunk->function_count) result = LANA_ERR_FORMAT;
                if (result == LANA_OK && (ins->c >= LANA_MAX_REGISTERS || ins->c + ins->imm > LANA_MAX_REGISTERS)) result = LANA_ERR_REGISTER;
                break;
            case OP_FORK:
                result = verify_register(error, ip, ins, ins->a);
                if (result == LANA_OK && ins->b >= chunk->function_count) result = LANA_ERR_FORMAT;
                if (result == LANA_OK && (ins->c >= LANA_MAX_REGISTERS || ins->c + ins->imm > LANA_MAX_REGISTERS)) result = LANA_ERR_REGISTER;
                if (result == LANA_OK && ins->imm != chunk->functions[ins->b].arity) result = LANA_ERR_TYPE;
                break;
            case OP_JOIN: case OP_JOIN_ALL:
                result = verify_register(error, ip, ins, ins->a);
                if (result == LANA_OK) result = verify_register(error, ip, ins, ins->b);
                break;
            case OP_JOIN_TIMEOUT:
                result = verify_register(error, ip, ins, ins->a);
                if (result == LANA_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == LANA_OK) result = verify_register(error, ip, ins, ins->c);
                break;
            case OP_CANCEL:
                result = verify_register(error, ip, ins, ins->a);
                break;
            case OP_HOST_CALL:
                result = verify_register(error, ip, ins, ins->a);
                if (result == LANA_OK && ins->b >= LANA_HOST_COUNT)
                    result = LANA_ERR_FORMAT;
                if (result == LANA_OK && (ins->c >= LANA_MAX_REGISTERS || ins->c + ins->imm > LANA_MAX_REGISTERS)) result = LANA_ERR_REGISTER;
                break;
            case OP_COUNT: result = LANA_ERR_OPCODE; break;
        }
        if (result != LANA_OK) {
            if (error != NULL && error->code == LANA_OK)
                lana_error_set(error, result, ip, ins->opcode, ins->line, "invalid operands for %s", lana_opcode_name(ins->opcode));
            return result;
        }
    }
    return LANA_OK;
}

static bool write_u8(FILE *file, uint8_t value) { return fwrite(&value, 1, 1, file) == 1; }
static bool write_u32(FILE *file, uint32_t value) {
    uint8_t bytes[4] = {(uint8_t)value, (uint8_t)(value >> 8u), (uint8_t)(value >> 16u), (uint8_t)(value >> 24u)};
    return fwrite(bytes, 1, sizeof(bytes), file) == sizeof(bytes);
}
static bool write_u64(FILE *file, uint64_t value) {
    uint8_t bytes[8];
    size_t index;
    for (index = 0; index < 8; ++index) bytes[index] = (uint8_t)(value >> (index * 8u));
    return fwrite(bytes, 1, sizeof(bytes), file) == sizeof(bytes);
}
static bool read_u8(FILE *file, uint8_t *value) { return fread(value, 1, 1, file) == 1; }
static bool read_u32(FILE *file, uint32_t *value) {
    uint8_t bytes[4];
    if (fread(bytes, 1, sizeof(bytes), file) != sizeof(bytes)) return false;
    *value = (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8u) |
             ((uint32_t)bytes[2] << 16u) | ((uint32_t)bytes[3] << 24u);
    return true;
}
static bool read_u64(FILE *file, uint64_t *value) {
    uint8_t bytes[8];
    size_t index;
    if (fread(bytes, 1, sizeof(bytes), file) != sizeof(bytes)) return false;
    *value = 0;
    for (index = 0; index < 8; ++index) *value |= (uint64_t)bytes[index] << (index * 8u);
    return true;
}

LanaError lana_chunk_write_file(const LanaChunk *chunk, const char *path, LanaErrorInfo *error) {
    FILE *file;
    size_t index;
    LanaError result = lana_chunk_verify(chunk, error);
    if (result != LANA_OK) return result;
    file = fopen(path, "wb");
    if (file == NULL) {
        lana_error_set(error, LANA_ERR_IO, 0, OP_NOP, 0, "cannot open %s: %s", path, strerror(errno));
        return LANA_ERR_IO;
    }
    if (fwrite("LABC", 1, 4, file) != 4 || !write_u32(file, chunk->version) ||
        !write_u32(file, (uint32_t)chunk->constant_count) ||
        !write_u32(file, (uint32_t)chunk->function_count) ||
        !write_u32(file, (uint32_t)chunk->code_count) || !write_u32(file, chunk->entry)) goto io_error;
    for (index = 0; index < chunk->constant_count; ++index) {
        const Value *value = &chunk->constants[index];
        if (!write_u8(file, (uint8_t)value->type)) goto io_error;
        if (value->type == VAL_NUMBER) {
            uint64_t bits;
            memcpy(&bits, &value->as.number, sizeof(bits));
            if (!write_u64(file, bits)) goto io_error;
        } else if (value->type == VAL_BOOL) {
            if (!write_u8(file, value->as.boolean ? 1u : 0u)) goto io_error;
        } else if (value->type == VAL_STRING) {
            uint32_t length = (uint32_t)strlen(value->as.string);
            if (!write_u32(file, length) || fwrite(value->as.string, 1, length, file) != length) goto io_error;
        } else if (value->type != VAL_NULL) {
            lana_error_set(error, LANA_ERR_FORMAT, 0, OP_NOP, 0, "constant type %s is not serializable", lana_value_type_name(value->type));
            (void)fclose(file);
            return LANA_ERR_FORMAT;
        }
    }
    for (index = 0; index < chunk->function_count; ++index) {
        const LanaFunction *function = &chunk->functions[index];
        uint32_t length = (uint32_t)strlen(function->name);
        if (!write_u32(file, length) || fwrite(function->name, 1, length, file) != length ||
            !write_u32(file, function->entry) || !write_u32(file, function->register_count) ||
            !write_u32(file, function->arity)) goto io_error;
    }
    for (index = 0; index < chunk->code_count; ++index) {
        const LanaInstruction *ins = &chunk->code[index];
        if (!write_u8(file, ins->opcode) || !write_u32(file, ins->a) || !write_u32(file, ins->b) ||
            !write_u32(file, ins->c) || !write_u32(file, ins->imm) || !write_u32(file, ins->line)) goto io_error;
    }
    if (fclose(file) != 0) goto close_error;
    return LANA_OK;
io_error:
    lana_error_set(error, LANA_ERR_IO, 0, OP_NOP, 0, "failed writing %s", path);
    (void)fclose(file);
    return LANA_ERR_IO;
close_error:
    lana_error_set(error, LANA_ERR_IO, 0, OP_NOP, 0, "failed closing %s", path);
    return LANA_ERR_IO;
}

LanaError lana_chunk_read_file(LanaChunk *chunk, const char *path, LanaErrorInfo *error) {
    FILE *file = fopen(path, "rb");
    char magic[4];
    long file_size;
    uint32_t constants, functions, instructions, index;
    LanaError result = LANA_OK;
    if (file == NULL) {
        lana_error_set(error, LANA_ERR_IO, 0, OP_NOP, 0, "cannot open %s: %s", path, strerror(errno));
        return LANA_ERR_IO;
    }
    lana_chunk_init(chunk);
    if (fseek(file, 0, SEEK_END) != 0 || (file_size = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0) {
        result = LANA_ERR_IO; goto failure;
    }
    if (file_size > 64L * 1024L * 1024L) {
        result = LANA_ERR_LIMIT; goto failure;
    }
    if (fread(magic, 1, 4, file) != 4) {
        result = LANA_ERR_FORMAT; goto failure;
    }
    if (memcmp(magic, "LABC", 4) != 0) {
        result = LANA_ERR_INCOMPATIBLE_FORMAT; goto failure;
    }
    if (!read_u32(file, &chunk->version) || !read_u32(file, &constants) ||
        !read_u32(file, &functions) || !read_u32(file, &instructions) || !read_u32(file, &chunk->entry)) {
        result = LANA_ERR_FORMAT; goto failure;
    }
    if (chunk->version != LABC_VERSION) { result = LANA_ERR_INCOMPATIBLE_FORMAT; goto failure; }
    if (constants > 100000u || functions > 10000u || instructions > 1000000u) {
        result = LANA_ERR_LIMIT; goto failure;
    }
    if ((uint64_t)constants > (uint64_t)file_size ||
        (uint64_t)functions > (uint64_t)file_size / 16u ||
        (uint64_t)instructions > (uint64_t)file_size / 21u) {
        result = LANA_ERR_FORMAT; goto failure;
    }
    for (index = 0; index < constants; ++index) {
        uint8_t type;
        Value value = lana_value_null();
        if (!read_u8(file, &type) || type > VAL_STRING) { result = LANA_ERR_FORMAT; goto failure; }
        if (type == VAL_NUMBER) {
            uint64_t bits;
            if (!read_u64(file, &bits)) { result = LANA_ERR_FORMAT; goto failure; }
            memcpy(&value.as.number, &bits, sizeof(bits)); value.type = VAL_NUMBER;
        } else if (type == VAL_BOOL) {
            uint8_t boolean;
            if (!read_u8(file, &boolean) || boolean > 1u) { result = LANA_ERR_FORMAT; goto failure; }
            value = lana_value_bool(boolean != 0);
        } else if (type == VAL_STRING) {
            uint32_t length;
            char *string;
            if (!read_u32(file, &length) || length > 10000000u) { result = LANA_ERR_FORMAT; goto failure; }
            string = malloc((size_t)length + 1u);
            if (string == NULL) { result = LANA_ERR_OOM; goto failure; }
            if (fread(string, 1, length, file) != length) { free(string); result = LANA_ERR_FORMAT; goto failure; }
            string[length] = '\0'; value = lana_value_string(string);
            result = lana_chunk_add_constant(chunk, value, NULL); free(string);
            if (result != LANA_OK) goto failure;
            continue;
        }
        result = lana_chunk_add_constant(chunk, value, NULL);
        if (result != LANA_OK) goto failure;
    }
    if (functions > 0) {
        chunk->functions = calloc(functions, sizeof(*chunk->functions));
        if (chunk->functions == NULL) { result = LANA_ERR_OOM; goto failure; }
        chunk->function_capacity = functions;
    }
    for (index = 0; index < functions; ++index) {
        uint32_t length;
        LanaFunction *function = &chunk->functions[index];
        if (!read_u32(file, &length) || length > 1000000u) { result = LANA_ERR_FORMAT; goto failure; }
        function->name = malloc((size_t)length + 1u);
        if (function->name == NULL) { result = LANA_ERR_OOM; goto failure; }
        ++chunk->function_count;
        if (fread(function->name, 1, length, file) != length || !read_u32(file, &function->entry) ||
            !read_u32(file, &function->register_count) || !read_u32(file, &function->arity)) {
            result = LANA_ERR_FORMAT; goto failure;
        }
        function->name[length] = '\0';
    }
    for (index = 0; index < instructions; ++index) {
        LanaInstruction ins = {0};
        if (!read_u8(file, &ins.opcode) || !read_u32(file, &ins.a) || !read_u32(file, &ins.b) ||
            !read_u32(file, &ins.c) || !read_u32(file, &ins.imm) || !read_u32(file, &ins.line)) {
            result = LANA_ERR_FORMAT; goto failure;
        }
        result = lana_chunk_emit(chunk, ins);
        if (result != LANA_OK) goto failure;
    }
    if (fgetc(file) != EOF || ferror(file)) {
        result = LANA_ERR_FORMAT; goto failure;
    }
    (void)fclose(file);
    result = lana_chunk_verify(chunk, error);
    if (result != LANA_OK) lana_chunk_free(chunk);
    return result;
failure:
    lana_error_set(error, result, 0, OP_NOP, 0, "invalid LABC file %s", path);
    (void)fclose(file);
    lana_chunk_free(chunk);
    return result;
}

void lana_disassemble_instruction(const LanaChunk *chunk, size_t offset, FILE *out) {
    const LanaInstruction *ins = &chunk->code[offset];
    static const char *measure_names_current[] = {"probability", "distribution", "sample"};
    (void)fprintf(out, "%04zu %-20s ", offset, lana_opcode_name(ins->opcode));
    switch ((OpCode)ins->opcode) {
        case OP_STATE_NEW:
            (void)fprintf(out,
                          "R%u p=K%u(%.12g) d_re=K%u(%.12g) d_im=K%u(%.12g)",
                          ins->a, ins->b, chunk->constants[ins->b].as.number,
                          ins->c, chunk->constants[ins->c].as.number,
                          ins->imm, chunk->constants[ins->imm].as.number);
            break;
        case OP_STATE_BUILD:
            (void)fprintf(out, "R%u p=R%u d_re=R%u d_im=R%u", ins->imm,
                          ins->a, ins->b, ins->c);
            break;
        case OP_TRANSFORM:
            (void)fprintf(out, "R%u <- %s(R%u)", ins->a,
                          ins->c == LANA_TRANSFORM_INVERT ? "invert" : "neutralize",
                          ins->b);
            break;
        case OP_MEASURE:
            (void)fprintf(out, "R%u %s -> R%u", ins->a,
                          ins->c <= LANA_MEASURE_SAMPLE ? measure_names_current[ins->c] : "unknown",
                          ins->b);
            break;
        case OP_APPEND:
            (void)fprintf(out, "R%u R%u -> R%u", ins->a, ins->b, ins->c);
            break;
        case OP_SAMPLE_STATE_DIST:
            (void)fprintf(out, "R%u -> R%u", ins->a, ins->b);
            break;
        case OP_MEASURE_BASIS:
            (void)fprintf(out, "R%u basis=%s mode=%s -> R%u", ins->a,
                          ins->c == LANA_MEASURE_BASIS_COMPUTATIONAL ? "computational" :
                          ins->c == LANA_MEASURE_BASIS_X ? "x" : "y",
                          ins->imm == LANA_MEASURE_PROBABILITY ? "probability" :
                          ins->imm == LANA_MEASURE_DISTRIBUTION ? "distribution" : "sample",
                          ins->b);
            break;
        case OP_ESTIMATE_MEASURE_PROBABILITY:
        case OP_ESTIMATE_MEASURE_DISTRIBUTION:
            (void)fprintf(out, "R%u basis=%s samples=%u -> R%u", ins->a,
                          ins->c == LANA_MEASURE_BASIS_COMPUTATIONAL ? "computational" :
                          ins->c == LANA_MEASURE_BASIS_X ? "x" : "y",
                          ins->imm, ins->b);
            break;
        case OP_JOINT_BUILD:
            (void)fprintf(out, "R%u..R%u descriptor[%u] -> R%u", ins->b,
                          ins->b + ins->c - 1u, ins->imm, ins->a);
            break;
        case OP_JOINT_PROJECT:
            (void)fprintf(out, "R%u names[%u] -> R%u", ins->a, ins->c, ins->b);
            break;
        case OP_JOINT_CONDITION:
            (void)fprintf(out, "R%u %s[%u]=R%u -> R%u", ins->a,
                          "condition", ins->c, ins->imm, ins->b);
            break;
        case OP_JOINT_SAMPLE:
            (void)fprintf(out, "R%u -> R%u", ins->a, ins->b);
            break;
        case OP_RESOLVE:
            (void)fprintf(out, "R%u -> R%u", ins->a, ins->b);
            break;
        case OP_JOINT_BUILD_FINITE:
            (void)fprintf(out, "rows=R%u names[%u] -> R%u", ins->a, ins->c, ins->b);
            break;
        case OP_JOINT_RENAME:
            (void)fprintf(out, "R%u name[%u]->name[%u] -> R%u",
                          ins->a, ins->c, ins->imm, ins->b);
            break;
        case OP_POSSIBILITY_BUILD:
            (void)fprintf(out, "R%u -> R%u", ins->a, ins->b);
            break;
        case OP_PATH_SPLIT:
            (void)fprintf(out, "R%u false->%u", ins->a, ins->imm);
            break;
        case OP_PATH_JOIN:
            (void)fprintf(out, "join");
            break;
        case OP_OBSERVE:
            (void)fprintf(out, "R%u name[%u]=R%u -> R%u", ins->a, ins->c,
                          ins->imm, ins->b);
            break;
        case OP_INFO_SAMPLE:
            (void)fprintf(out, "R%u -> R%u", ins->a, ins->b);
            break;
        case OP_EVIDENCE:
        case OP_ASSUME:
            (void)fprintf(out, "R%u label[%u] -> R%u", ins->a, ins->c, ins->b);
            break;
        case OP_DERIVATION:
        case OP_EXPLAIN:
            (void)fprintf(out, "R%u -> R%u", ins->a, ins->b);
            break;
        case OP_FORK:
            (void)fprintf(out, "function[%u] R%u argc=%u -> R%u", ins->b, ins->c, ins->imm, ins->a);
            break;
        case OP_JOIN: case OP_JOIN_ALL:
            (void)fprintf(out, "R%u -> R%u", ins->a, ins->b);
            break;
        case OP_JOIN_TIMEOUT:
            (void)fprintf(out, "R%u timeout=R%u -> R%u", ins->a, ins->b, ins->c);
            break;
        case OP_CANCEL: (void)fprintf(out, "R%u", ins->a); break;
        case OP_LOAD_CONST:
            (void)fprintf(out, "R%u constant[%u]", ins->a, ins->imm); break;
        case OP_MOVE:
            (void)fprintf(out, "R%u <- R%u", ins->a, ins->b); break;
        case OP_JUMP: (void)fprintf(out, "-> %u", ins->imm); break;
        case OP_JUMP_IF_TRUE: case OP_JUMP_IF_FALSE:
            (void)fprintf(out, "R%u -> %u", ins->a, ins->imm); break;
        case OP_PRINT: case OP_RETURN: (void)fprintf(out, "R%u", ins->a); break;
        case OP_HALT: case OP_NOP: break;
        default:
            (void)fprintf(out, "a=%u b=%u c=%u imm=%u", ins->a, ins->b, ins->c, ins->imm);
            break;
    }
    (void)fprintf(out, "  ; line %u\n", ins->line);
}

void lana_disassemble(const LanaChunk *chunk, FILE *out) {
    size_t offset;
    (void)fprintf(out, "LABC v%u entry=%u constants=%zu functions=%zu instructions=%zu\n",
                  chunk->version, chunk->entry, chunk->constant_count,
                  chunk->function_count, chunk->code_count);
    for (offset = 0; offset < chunk->code_count; ++offset)
        lana_disassemble_instruction(chunk, offset, out);
}
