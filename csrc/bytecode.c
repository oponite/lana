#include "ss/bytecode.h"

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

void ss_chunk_init(SSChunk *chunk) {
    memset(chunk, 0, sizeof(*chunk));
    chunk->version = SSBC_VERSION;
}

void ss_chunk_free(SSChunk *chunk) {
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
    ss_chunk_init(chunk);
}

SSError ss_chunk_add_constant(SSChunk *chunk, Value value, uint32_t *index) {
    Value *constants;
    if (chunk->constant_count == UINT32_MAX) return SS_ERR_LIMIT;
    if (chunk->constant_count == chunk->constant_capacity) {
        size_t capacity = chunk->constant_capacity == 0 ? 16u : chunk->constant_capacity * 2u;
        constants = realloc(chunk->constants, capacity * sizeof(*constants));
        if (constants == NULL) return SS_ERR_OOM;
        chunk->constants = constants;
        chunk->constant_capacity = capacity;
    }
    if (value.type == VAL_STRING) {
        char *string = copy_string(value.as.string);
        if (string == NULL) return SS_ERR_OOM;
        value.as.string = string;
    }
    chunk->constants[chunk->constant_count] = value;
    if (index != NULL) *index = (uint32_t)chunk->constant_count;
    ++chunk->constant_count;
    return SS_OK;
}

SSError ss_chunk_add_function(SSChunk *chunk, const char *name, uint32_t entry,
                              uint32_t register_count, uint32_t arity,
                              uint32_t *index) {
    SSFunction *functions;
    char *owned_name;
    if (name == NULL || register_count > SS_MAX_REGISTERS || arity > register_count ||
        chunk->function_count == UINT32_MAX) return SS_ERR_FORMAT;
    if (chunk->function_count == chunk->function_capacity) {
        size_t capacity = chunk->function_capacity == 0 ? 8u : chunk->function_capacity * 2u;
        functions = realloc(chunk->functions, capacity * sizeof(*functions));
        if (functions == NULL) return SS_ERR_OOM;
        chunk->functions = functions; chunk->function_capacity = capacity;
    }
    owned_name = copy_string(name);
    if (owned_name == NULL) return SS_ERR_OOM;
    chunk->functions[chunk->function_count] = (SSFunction){owned_name, entry, register_count, arity};
    if (index != NULL) *index = (uint32_t)chunk->function_count;
    ++chunk->function_count;
    return SS_OK;
}

SSError ss_chunk_emit(SSChunk *chunk, SSInstruction instruction) {
    SSInstruction *code;
    if (chunk->code_count == chunk->code_capacity) {
        size_t capacity = chunk->code_capacity == 0 ? 32u : chunk->code_capacity * 2u;
        code = realloc(chunk->code, capacity * sizeof(*code));
        if (code == NULL) return SS_ERR_OOM;
        chunk->code = code;
        chunk->code_capacity = capacity;
    }
    chunk->code[chunk->code_count++] = instruction;
    return SS_OK;
}

const char *ss_opcode_name(uint8_t opcode) {
    static const char *names[] = {
        "NOP", "LOAD_CONST", "MOVE", "STATE_NEW", "STATE_SET", "STATE_COPY",
        "APPLY", "APPLY_MANY", "TRANSFORM", "COMPOSE_MERGE", "COMPOSE_UPDATE",
        "COMPOSE_JOINT", "MEASURE", "GET_FIELD", "GET_INDEX", "SET_INDEX",
        "HISTORY_CONFIG", "PREVIOUS", "CHANGE", "VELOCITY", "BINARY", "UNARY",
        "COMPARE", "JUMP", "JUMP_IF_TRUE", "JUMP_IF_FALSE", "ARRAY_NEW",
        "ARRAY_GET", "ARRAY_SET", "CALL", "RETURN", "PRINT", "HALT",
        "STATE_BUILD", "FORK", "JOIN", "JOIN_TIMEOUT", "JOIN_ALL", "CANCEL",
        "TASKGROUP_ENTER", "TASKGROUP_EXIT", "HOST_CALL", "STATE_NEW_V3",
        "STATE_BUILD_V3", "TRANSFORM_V3", "MEASURE_V3", "APPEND",
        "SAMPLE_STATE_DIST", "MEASURE_BASIS_V4",
        "ESTIMATE_MEASURE_PROBABILITY_V4", "ESTIMATE_MEASURE_DISTRIBUTION_V4",
        "JOINT_BUILD_V5", "JOINT_PROJECT_V5", "JOINT_CONDITION_V5",
        "JOINT_SAMPLE_V5", "RESOLVE_V5", "JOINT_BUILD_FINITE_V5",
        "JOINT_RENAME_V5", "POSSIBILITY_BUILD_V5", "PATH_SPLIT_V5",
        "PATH_JOIN_V5", "OBSERVE_V5", "INFO_SAMPLE_V5"
    };
    return opcode < OP_COUNT ? names[opcode] : "UNKNOWN";
}

static bool register_valid(uint32_t value) { return value < SS_MAX_REGISTERS; }
static bool constant_valid(const SSChunk *chunk, uint32_t value) {
    return value < chunk->constant_count;
}

static SSError verify_register(SSErrorInfo *error, size_t ip,
                               const SSInstruction *instruction, uint32_t reg) {
    if (!register_valid(reg)) {
        ss_error_set(error, SS_ERR_REGISTER, ip, instruction->opcode,
                     instruction->line, "register R%u is out of range", reg);
        return SS_ERR_REGISTER;
    }
    return SS_OK;
}

SSError ss_chunk_verify(const SSChunk *chunk, SSErrorInfo *error) {
    size_t ip, function_index;
    if (error != NULL) memset(error, 0, sizeof(*error));
    if (chunk == NULL || chunk->code_count == 0 || chunk->entry >= chunk->code_count) {
        ss_error_set(error, SS_ERR_FORMAT, 0, OP_NOP, 0, "chunk has no valid entry point");
        return SS_ERR_FORMAT;
    }
    if (chunk->version < SSBC_MIN_VERSION || chunk->version > SSBC_VERSION) {
        ss_error_set(error, SS_ERR_FORMAT, 0, OP_NOP, 0,
                     "unsupported SSBC version %u", chunk->version);
        return SS_ERR_FORMAT;
    }
    for (function_index = 0; function_index < chunk->function_count; ++function_index) {
        const SSFunction *function = &chunk->functions[function_index];
        if (function->entry >= chunk->code_count || function->register_count > SS_MAX_REGISTERS ||
            function->arity > function->register_count) {
            ss_error_set(error, SS_ERR_FORMAT, function->entry, OP_CALL, 0,
                         "function %zu has invalid metadata", function_index);
            return SS_ERR_FORMAT;
        }
    }
    for (ip = 0; ip < chunk->code_count; ++ip) {
        const SSInstruction *ins = &chunk->code[ip];
        SSError result = SS_OK;
        if (ins->opcode >= OP_COUNT) {
            ss_error_set(error, SS_ERR_OPCODE, ip, ins->opcode, ins->line,
                         "unknown opcode %u", ins->opcode);
            return SS_ERR_OPCODE;
        }
        if (chunk->version == 1u && ins->opcode > OP_HALT) {
            ss_error_set(error, SS_ERR_OPCODE, ip, ins->opcode, ins->line,
                         "Bytecode v1 cannot contain %s", ss_opcode_name(ins->opcode));
            return SS_ERR_OPCODE;
        }
        if (chunk->version == 2u && ins->opcode > OP_HOST_CALL) {
            ss_error_set(error, SS_ERR_OPCODE, ip, ins->opcode, ins->line,
                         "Bytecode v2 cannot contain %s", ss_opcode_name(ins->opcode));
            return SS_ERR_OPCODE;
        }
        if (chunk->version < 4u && ins->opcode > OP_SAMPLE_STATE_DIST) {
            ss_error_set(error, SS_ERR_OPCODE, ip, ins->opcode, ins->line,
                         "Bytecode v%u cannot contain %s", chunk->version,
                         ss_opcode_name(ins->opcode));
            return SS_ERR_OPCODE;
        }
        if (chunk->version < 5u && ins->opcode > OP_ESTIMATE_MEASURE_DISTRIBUTION_V4) {
            ss_error_set(error, SS_ERR_OPCODE, ip, ins->opcode, ins->line,
                         "Bytecode v%u cannot contain %s", chunk->version,
                         ss_opcode_name(ins->opcode));
            return SS_ERR_OPCODE;
        }
        switch ((OpCode)ins->opcode) {
            case OP_NOP: case OP_HALT: case OP_TASKGROUP_ENTER: case OP_TASKGROUP_EXIT: break;
            case OP_LOAD_CONST:
                result = verify_register(error, ip, ins, ins->a);
                if (result == SS_OK && !constant_valid(chunk, ins->imm)) {
                    ss_error_set(error, SS_ERR_CONSTANT, ip, ins->opcode, ins->line,
                                 "constant %u is out of range", ins->imm);
                    result = SS_ERR_CONSTANT;
                }
                break;
            case OP_STATE_NEW:
                result = verify_register(error, ip, ins, ins->a);
                if (result == SS_OK && (!constant_valid(chunk, ins->b) ||
                    !constant_valid(chunk, ins->c))) {
                    ss_error_set(error, SS_ERR_CONSTANT, ip, ins->opcode, ins->line,
                                 "STATE_NEW uses an invalid constant");
                    result = SS_ERR_CONSTANT;
                }
                if (result == SS_OK && (chunk->constants[ins->b].type != VAL_NUMBER ||
                    chunk->constants[ins->c].type != VAL_NUMBER)) {
                    ss_error_set(error, SS_ERR_TYPE, ip, ins->opcode, ins->line,
                                 "STATE_NEW requires numeric p and d constants");
                    result = SS_ERR_TYPE;
                }
                break;
            case OP_STATE_BUILD:
                result = verify_register(error, ip, ins, ins->a);
                if (result == SS_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == SS_OK) result = verify_register(error, ip, ins, ins->c);
                break;
            case OP_STATE_NEW_V3: {
                SSState state;
                result = verify_register(error, ip, ins, ins->a);
                if (result == SS_OK && (!constant_valid(chunk, ins->b) ||
                    !constant_valid(chunk, ins->c) || !constant_valid(chunk, ins->imm)))
                    result = SS_ERR_CONSTANT;
                if (result == SS_OK && (chunk->constants[ins->b].type != VAL_NUMBER ||
                    chunk->constants[ins->c].type != VAL_NUMBER ||
                    chunk->constants[ins->imm].type != VAL_NUMBER)) result = SS_ERR_TYPE;
                if (result == SS_OK)
                    result = ss_state_make_complex(chunk->constants[ins->b].as.number,
                                                   chunk->constants[ins->c].as.number,
                                                   chunk->constants[ins->imm].as.number,
                                                   &state);
                break;
            }
            case OP_STATE_BUILD_V3:
                result = verify_register(error, ip, ins, ins->a);
                if (result == SS_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == SS_OK) result = verify_register(error, ip, ins, ins->c);
                if (result == SS_OK) result = verify_register(error, ip, ins, ins->imm);
                break;
            case OP_TRANSFORM_V3:
                result = verify_register(error, ip, ins, ins->a);
                if (result == SS_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == SS_OK && ins->c > SS_TRANSFORM_V3_NEUTRALIZE)
                    result = SS_ERR_TRANSFORM;
                if (result == SS_OK && ins->imm != 0u) result = SS_ERR_FORMAT;
                break;
            case OP_MEASURE_V3:
                result = verify_register(error, ip, ins, ins->a);
                if (result == SS_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == SS_OK && ins->c > SS_MEASURE_V3_SAMPLE)
                    result = SS_ERR_MEASURE;
                if (result == SS_OK && ins->imm != 0u) result = SS_ERR_FORMAT;
                break;
            case OP_APPEND:
                result = verify_register(error, ip, ins, ins->a);
                if (result == SS_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == SS_OK) result = verify_register(error, ip, ins, ins->c);
                if (result == SS_OK && ins->imm != 0u) result = SS_ERR_FORMAT;
                break;
            case OP_SAMPLE_STATE_DIST:
                result = verify_register(error, ip, ins, ins->a);
                if (result == SS_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == SS_OK && (ins->c != 0u || ins->imm != 0u))
                    result = SS_ERR_FORMAT;
                break;
            case OP_MEASURE_BASIS_V4:
                result = verify_register(error, ip, ins, ins->a);
                if (result == SS_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == SS_OK && ins->c > SS_MEASURE_BASIS_Y)
                    result = SS_ERR_MEASURE;
                if (result == SS_OK && ins->imm > SS_MEASURE_V3_SAMPLE)
                    result = SS_ERR_MEASURE;
                break;
            case OP_ESTIMATE_MEASURE_PROBABILITY_V4:
            case OP_ESTIMATE_MEASURE_DISTRIBUTION_V4:
                result = verify_register(error, ip, ins, ins->a);
                if (result == SS_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == SS_OK && ins->c > SS_MEASURE_BASIS_Y)
                    result = SS_ERR_MEASURE;
                if (result == SS_OK && ins->imm == 0u)
                    result = SS_ERR_FORMAT;
                break;
            case OP_JOINT_BUILD_V5:
                result = verify_register(error, ip, ins, ins->a);
                if (result == SS_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == SS_OK && (ins->c == 0u || ins->b + ins->c > SS_MAX_REGISTERS))
                    result = SS_ERR_REGISTER;
                if (result == SS_OK && !constant_valid(chunk, ins->imm)) result = SS_ERR_CONSTANT;
                if (result == SS_OK && chunk->constants[ins->imm].type != VAL_STRING)
                    result = SS_ERR_TYPE;
                break;
            case OP_JOINT_PROJECT_V5:
                result = verify_register(error, ip, ins, ins->a);
                if (result == SS_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == SS_OK && !constant_valid(chunk, ins->c)) result = SS_ERR_CONSTANT;
                if (result == SS_OK && chunk->constants[ins->c].type != VAL_STRING)
                    result = SS_ERR_TYPE;
                break;
            case OP_JOINT_CONDITION_V5:
                result = verify_register(error, ip, ins, ins->a);
                if (result == SS_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == SS_OK) result = verify_register(error, ip, ins, ins->imm);
                if (result == SS_OK && !constant_valid(chunk, ins->c)) result = SS_ERR_CONSTANT;
                if (result == SS_OK && chunk->constants[ins->c].type != VAL_STRING)
                    result = SS_ERR_TYPE;
                break;
            case OP_JOINT_SAMPLE_V5:
            case OP_RESOLVE_V5:
                result = verify_register(error, ip, ins, ins->a);
                if (result == SS_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == SS_OK && (ins->c != 0u || ins->imm != 0u))
                    result = SS_ERR_FORMAT;
                break;
            case OP_JOINT_BUILD_FINITE_V5:
                result = verify_register(error, ip, ins, ins->a);
                if (result == SS_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == SS_OK && !constant_valid(chunk, ins->c)) result = SS_ERR_CONSTANT;
                if (result == SS_OK && chunk->constants[ins->c].type != VAL_STRING)
                    result = SS_ERR_TYPE;
                if (result == SS_OK && ins->imm != 0u) result = SS_ERR_FORMAT;
                break;
            case OP_JOINT_RENAME_V5:
                result = verify_register(error, ip, ins, ins->a);
                if (result == SS_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == SS_OK && (!constant_valid(chunk, ins->c) ||
                                        !constant_valid(chunk, ins->imm)))
                    result = SS_ERR_CONSTANT;
                if (result == SS_OK && (chunk->constants[ins->c].type != VAL_STRING ||
                                        chunk->constants[ins->imm].type != VAL_STRING))
                    result = SS_ERR_TYPE;
                break;
            case OP_POSSIBILITY_BUILD_V5:
            case OP_INFO_SAMPLE_V5:
                result = verify_register(error, ip, ins, ins->a);
                if (result == SS_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == SS_OK && (ins->c != 0u || ins->imm != 0u))
                    result = SS_ERR_FORMAT;
                break;
            case OP_PATH_SPLIT_V5:
                result = verify_register(error, ip, ins, ins->a);
                if (result == SS_OK && ins->imm >= chunk->code_count) result = SS_ERR_JUMP;
                if (result == SS_OK && (ins->b != 0u || ins->c != 0u))
                    result = SS_ERR_FORMAT;
                break;
            case OP_PATH_JOIN_V5:
                if (ins->a != 0u || ins->b != 0u || ins->c != 0u || ins->imm != 0u)
                    result = SS_ERR_FORMAT;
                break;
            case OP_OBSERVE_V5:
                result = verify_register(error, ip, ins, ins->a);
                if (result == SS_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == SS_OK) result = verify_register(error, ip, ins, ins->imm);
                if (result == SS_OK && !constant_valid(chunk, ins->c)) result = SS_ERR_CONSTANT;
                if (result == SS_OK && chunk->constants[ins->c].type != VAL_STRING)
                    result = SS_ERR_TYPE;
                break;
            case OP_JUMP:
                if (ins->imm >= chunk->code_count) result = SS_ERR_JUMP;
                if (result != SS_OK) ss_error_set(error, result, ip, ins->opcode, ins->line, "jump target %u is invalid", ins->imm);
                break;
            case OP_JUMP_IF_TRUE: case OP_JUMP_IF_FALSE:
                result = verify_register(error, ip, ins, ins->a);
                if (result == SS_OK && ins->imm >= chunk->code_count) {
                    ss_error_set(error, SS_ERR_JUMP, ip, ins->opcode, ins->line,
                                 "jump target %u is invalid", ins->imm);
                    result = SS_ERR_JUMP;
                }
                break;
            case OP_MOVE: case OP_STATE_SET: case OP_STATE_COPY: case OP_APPLY:
            case OP_PREVIOUS: case OP_CHANGE: case OP_VELOCITY: case OP_RETURN:
                result = verify_register(error, ip, ins, ins->a);
                if (result == SS_OK && ins->opcode != OP_RETURN)
                    result = verify_register(error, ip, ins, ins->b);
                break;
            case OP_TRANSFORM:
                result = verify_register(error, ip, ins, ins->a);
                if (result == SS_OK && ins->b >= SS_TRANSFORM_RESET_D + 1u) result = SS_ERR_TRANSFORM;
                if (result == SS_OK && ins->imm > 0) result = verify_register(error, ip, ins, ins->c);
                break;
            case OP_APPLY_MANY:
                result = verify_register(error, ip, ins, ins->a);
                if (result == SS_OK && (ins->b == 0u || ins->a + ins->b > SS_MAX_REGISTERS)) result = SS_ERR_REGISTER;
                if (result == SS_OK) result = verify_register(error, ip, ins, ins->c);
                if (result == SS_OK && ins->imm > SS_AGG_STRONGEST) result = SS_ERR_COMPOSE;
                break;
            case OP_COMPOSE_MERGE: case OP_COMPOSE_JOINT:
            case OP_ARRAY_GET: case OP_ARRAY_SET:
                result = verify_register(error, ip, ins, ins->a);
                if (result == SS_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == SS_OK) result = verify_register(error, ip, ins, ins->c);
                break;
            case OP_COMPOSE_UPDATE:
                result = verify_register(error, ip, ins, ins->a);
                if (result == SS_OK) result = verify_register(error, ip, ins, ins->b);
                break;
            case OP_MEASURE:
                result = verify_register(error, ip, ins, ins->a);
                if (result == SS_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == SS_OK && ins->c > SS_MEASURE_COLLAPSE) result = SS_ERR_MEASURE;
                break;
            case OP_GET_FIELD: case OP_GET_INDEX: case OP_SET_INDEX:
            case OP_HISTORY_CONFIG: case OP_BINARY: case OP_COMPARE:
                result = verify_register(error, ip, ins, ins->a);
                if (result == SS_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == SS_OK && (ins->opcode == OP_BINARY || ins->opcode == OP_COMPARE || ins->opcode == OP_SET_INDEX))
                    result = verify_register(error, ip, ins, ins->c);
                break;
            case OP_UNARY: case OP_PRINT:
                result = verify_register(error, ip, ins, ins->a);
                if (result == SS_OK && ins->opcode == OP_UNARY) result = verify_register(error, ip, ins, ins->b);
                break;
            case OP_ARRAY_NEW:
                result = verify_register(error, ip, ins, ins->a);
                if (result == SS_OK && (ins->b >= SS_MAX_REGISTERS || ins->b + ins->c > SS_MAX_REGISTERS)) result = SS_ERR_REGISTER;
                break;
            case OP_CALL:
                result = verify_register(error, ip, ins, ins->a);
                if (result == SS_OK && ins->b >= chunk->function_count) result = SS_ERR_FORMAT;
                if (result == SS_OK && (ins->c >= SS_MAX_REGISTERS || ins->c + ins->imm > SS_MAX_REGISTERS)) result = SS_ERR_REGISTER;
                break;
            case OP_FORK:
                result = verify_register(error, ip, ins, ins->a);
                if (result == SS_OK && ins->b >= chunk->function_count) result = SS_ERR_FORMAT;
                if (result == SS_OK && (ins->c >= SS_MAX_REGISTERS || ins->c + ins->imm > SS_MAX_REGISTERS)) result = SS_ERR_REGISTER;
                if (result == SS_OK && ins->imm != chunk->functions[ins->b].arity) result = SS_ERR_TYPE;
                break;
            case OP_JOIN: case OP_JOIN_ALL:
                result = verify_register(error, ip, ins, ins->a);
                if (result == SS_OK) result = verify_register(error, ip, ins, ins->b);
                break;
            case OP_JOIN_TIMEOUT:
                result = verify_register(error, ip, ins, ins->a);
                if (result == SS_OK) result = verify_register(error, ip, ins, ins->b);
                if (result == SS_OK) result = verify_register(error, ip, ins, ins->c);
                break;
            case OP_CANCEL:
                result = verify_register(error, ip, ins, ins->a);
                break;
            case OP_HOST_CALL:
                result = verify_register(error, ip, ins, ins->a);
                if (result == SS_OK &&
                    ((chunk->version < 3u && ins->b > SS_HOST_ML_MODEL_READ) ||
                     ins->b > SS_HOST_PATH_RESOLVE)) result = SS_ERR_FORMAT;
                if (result == SS_OK && (ins->c >= SS_MAX_REGISTERS || ins->c + ins->imm > SS_MAX_REGISTERS)) result = SS_ERR_REGISTER;
                break;
            case OP_COUNT: result = SS_ERR_OPCODE; break;
        }
        if (result != SS_OK) {
            if (error != NULL && error->code == SS_OK)
                ss_error_set(error, result, ip, ins->opcode, ins->line, "invalid operands for %s", ss_opcode_name(ins->opcode));
            return result;
        }
    }
    return SS_OK;
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

SSError ss_chunk_write_file(const SSChunk *chunk, const char *path, SSErrorInfo *error) {
    FILE *file;
    size_t index;
    SSError result = ss_chunk_verify(chunk, error);
    if (result != SS_OK) return result;
    file = fopen(path, "wb");
    if (file == NULL) {
        ss_error_set(error, SS_ERR_IO, 0, OP_NOP, 0, "cannot open %s: %s", path, strerror(errno));
        return SS_ERR_IO;
    }
    if (fwrite("SSBC", 1, 4, file) != 4 || !write_u32(file, chunk->version) ||
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
            ss_error_set(error, SS_ERR_FORMAT, 0, OP_NOP, 0, "constant type %s is not serializable", ss_value_type_name(value->type));
            (void)fclose(file);
            return SS_ERR_FORMAT;
        }
    }
    for (index = 0; index < chunk->function_count; ++index) {
        const SSFunction *function = &chunk->functions[index];
        uint32_t length = (uint32_t)strlen(function->name);
        if (!write_u32(file, length) || fwrite(function->name, 1, length, file) != length ||
            !write_u32(file, function->entry) || !write_u32(file, function->register_count) ||
            !write_u32(file, function->arity)) goto io_error;
    }
    for (index = 0; index < chunk->code_count; ++index) {
        const SSInstruction *ins = &chunk->code[index];
        if (!write_u8(file, ins->opcode) || !write_u32(file, ins->a) || !write_u32(file, ins->b) ||
            !write_u32(file, ins->c) || !write_u32(file, ins->imm) || !write_u32(file, ins->line)) goto io_error;
    }
    if (fclose(file) != 0) goto close_error;
    return SS_OK;
io_error:
    ss_error_set(error, SS_ERR_IO, 0, OP_NOP, 0, "failed writing %s", path);
    (void)fclose(file);
    return SS_ERR_IO;
close_error:
    ss_error_set(error, SS_ERR_IO, 0, OP_NOP, 0, "failed closing %s", path);
    return SS_ERR_IO;
}

SSError ss_chunk_read_file(SSChunk *chunk, const char *path, SSErrorInfo *error) {
    FILE *file = fopen(path, "rb");
    char magic[4];
    uint32_t constants, functions, instructions, index;
    SSError result = SS_OK;
    if (file == NULL) {
        ss_error_set(error, SS_ERR_IO, 0, OP_NOP, 0, "cannot open %s: %s", path, strerror(errno));
        return SS_ERR_IO;
    }
    ss_chunk_init(chunk);
    if (fread(magic, 1, 4, file) != 4 || memcmp(magic, "SSBC", 4) != 0 ||
        !read_u32(file, &chunk->version) || !read_u32(file, &constants) ||
        !read_u32(file, &functions) || !read_u32(file, &instructions) || !read_u32(file, &chunk->entry)) {
        result = SS_ERR_FORMAT; goto failure;
    }
    if (chunk->version < SSBC_MIN_VERSION || chunk->version > SSBC_VERSION) { result = SS_ERR_FORMAT; goto failure; }
    if (constants > 100000u || functions > 10000u || instructions > 1000000u) {
        result = SS_ERR_LIMIT; goto failure;
    }
    for (index = 0; index < constants; ++index) {
        uint8_t type;
        Value value = ss_value_null();
        if (!read_u8(file, &type) || type > VAL_STRING) { result = SS_ERR_FORMAT; goto failure; }
        if (type == VAL_NUMBER) {
            uint64_t bits;
            if (!read_u64(file, &bits)) { result = SS_ERR_FORMAT; goto failure; }
            memcpy(&value.as.number, &bits, sizeof(bits)); value.type = VAL_NUMBER;
        } else if (type == VAL_BOOL) {
            uint8_t boolean;
            if (!read_u8(file, &boolean) || boolean > 1u) { result = SS_ERR_FORMAT; goto failure; }
            value = ss_value_bool(boolean != 0);
        } else if (type == VAL_STRING) {
            uint32_t length;
            char *string;
            if (!read_u32(file, &length) || length > 10000000u) { result = SS_ERR_FORMAT; goto failure; }
            string = malloc((size_t)length + 1u);
            if (string == NULL) { result = SS_ERR_OOM; goto failure; }
            if (fread(string, 1, length, file) != length) { free(string); result = SS_ERR_FORMAT; goto failure; }
            string[length] = '\0'; value = ss_value_string(string);
            result = ss_chunk_add_constant(chunk, value, NULL); free(string);
            if (result != SS_OK) goto failure;
            continue;
        }
        result = ss_chunk_add_constant(chunk, value, NULL);
        if (result != SS_OK) goto failure;
    }
    if (functions > 0) {
        chunk->functions = calloc(functions, sizeof(*chunk->functions));
        if (chunk->functions == NULL) { result = SS_ERR_OOM; goto failure; }
        chunk->function_capacity = functions;
    }
    for (index = 0; index < functions; ++index) {
        uint32_t length;
        SSFunction *function = &chunk->functions[index];
        if (!read_u32(file, &length) || length > 1000000u) { result = SS_ERR_FORMAT; goto failure; }
        function->name = malloc((size_t)length + 1u);
        if (function->name == NULL) { result = SS_ERR_OOM; goto failure; }
        if (fread(function->name, 1, length, file) != length || !read_u32(file, &function->entry) ||
            !read_u32(file, &function->register_count) || !read_u32(file, &function->arity)) {
            result = SS_ERR_FORMAT; goto failure;
        }
        function->name[length] = '\0'; ++chunk->function_count;
    }
    for (index = 0; index < instructions; ++index) {
        SSInstruction ins = {0};
        if (!read_u8(file, &ins.opcode) || !read_u32(file, &ins.a) || !read_u32(file, &ins.b) ||
            !read_u32(file, &ins.c) || !read_u32(file, &ins.imm) || !read_u32(file, &ins.line)) {
            result = SS_ERR_FORMAT; goto failure;
        }
        result = ss_chunk_emit(chunk, ins);
        if (result != SS_OK) goto failure;
    }
    (void)fclose(file);
    return ss_chunk_verify(chunk, error);
failure:
    ss_error_set(error, result, 0, OP_NOP, 0, "invalid SSBC file %s", path);
    (void)fclose(file);
    ss_chunk_free(chunk);
    return result;
}

void ss_disassemble_instruction(const SSChunk *chunk, size_t offset, FILE *out) {
    const SSInstruction *ins = &chunk->code[offset];
    static const char *measure_names[] = {"distribution", "probability", "sample", "collapse"};
    static const char *measure_names_v3[] = {"probability", "distribution", "sample"};
    (void)fprintf(out, "%04zu %-20s ", offset, ss_opcode_name(ins->opcode));
    switch ((OpCode)ins->opcode) {
        case OP_STATE_NEW:
            (void)fprintf(out, "R%u p=%.12g d=%.12g", ins->a,
                          chunk->constants[ins->b].as.number,
                          chunk->constants[ins->c].as.number);
            break;
        case OP_STATE_BUILD:
            (void)fprintf(out, "R%u p=R%u d=R%u", ins->c, ins->a, ins->b);
            break;
        case OP_STATE_NEW_V3:
            (void)fprintf(out,
                          "R%u p=K%u(%.12g) d_re=K%u(%.12g) d_im=K%u(%.12g)",
                          ins->a, ins->b, chunk->constants[ins->b].as.number,
                          ins->c, chunk->constants[ins->c].as.number,
                          ins->imm, chunk->constants[ins->imm].as.number);
            break;
        case OP_STATE_BUILD_V3:
            (void)fprintf(out, "R%u p=R%u d_re=R%u d_im=R%u", ins->imm,
                          ins->a, ins->b, ins->c);
            break;
        case OP_TRANSFORM_V3:
            (void)fprintf(out, "R%u <- %s(R%u)", ins->a,
                          ins->c == SS_TRANSFORM_V3_INVERT ? "invert" : "neutralize",
                          ins->b);
            break;
        case OP_MEASURE_V3:
            (void)fprintf(out, "R%u %s -> R%u", ins->a,
                          ins->c <= SS_MEASURE_V3_SAMPLE ? measure_names_v3[ins->c] : "unknown",
                          ins->b);
            break;
        case OP_APPEND:
            (void)fprintf(out, "R%u R%u -> R%u", ins->a, ins->b, ins->c);
            break;
        case OP_SAMPLE_STATE_DIST:
            (void)fprintf(out, "R%u -> R%u", ins->a, ins->b);
            break;
        case OP_MEASURE_BASIS_V4:
            (void)fprintf(out, "R%u basis=%s mode=%s -> R%u", ins->a,
                          ins->c == SS_MEASURE_BASIS_COMPUTATIONAL ? "computational" :
                          ins->c == SS_MEASURE_BASIS_X ? "x" : "y",
                          ins->imm == SS_MEASURE_V3_PROBABILITY ? "probability" :
                          ins->imm == SS_MEASURE_V3_DISTRIBUTION ? "distribution" : "sample",
                          ins->b);
            break;
        case OP_ESTIMATE_MEASURE_PROBABILITY_V4:
        case OP_ESTIMATE_MEASURE_DISTRIBUTION_V4:
            (void)fprintf(out, "R%u basis=%s samples=%u -> R%u", ins->a,
                          ins->c == SS_MEASURE_BASIS_COMPUTATIONAL ? "computational" :
                          ins->c == SS_MEASURE_BASIS_X ? "x" : "y",
                          ins->imm, ins->b);
            break;
        case OP_JOINT_BUILD_V5:
            (void)fprintf(out, "R%u..R%u descriptor[%u] -> R%u", ins->b,
                          ins->b + ins->c - 1u, ins->imm, ins->a);
            break;
        case OP_JOINT_PROJECT_V5:
            (void)fprintf(out, "R%u names[%u] -> R%u", ins->a, ins->c, ins->b);
            break;
        case OP_JOINT_CONDITION_V5:
            (void)fprintf(out, "R%u %s[%u]=R%u -> R%u", ins->a,
                          "condition", ins->c, ins->imm, ins->b);
            break;
        case OP_JOINT_SAMPLE_V5:
            (void)fprintf(out, "R%u -> R%u", ins->a, ins->b);
            break;
        case OP_RESOLVE_V5:
            (void)fprintf(out, "R%u -> R%u", ins->a, ins->b);
            break;
        case OP_JOINT_BUILD_FINITE_V5:
            (void)fprintf(out, "rows=R%u names[%u] -> R%u", ins->a, ins->c, ins->b);
            break;
        case OP_JOINT_RENAME_V5:
            (void)fprintf(out, "R%u name[%u]->name[%u] -> R%u",
                          ins->a, ins->c, ins->imm, ins->b);
            break;
        case OP_POSSIBILITY_BUILD_V5:
            (void)fprintf(out, "R%u -> R%u", ins->a, ins->b);
            break;
        case OP_PATH_SPLIT_V5:
            (void)fprintf(out, "R%u false->%u", ins->a, ins->imm);
            break;
        case OP_PATH_JOIN_V5:
            (void)fprintf(out, "join");
            break;
        case OP_OBSERVE_V5:
            (void)fprintf(out, "R%u name[%u]=R%u -> R%u", ins->a, ins->c,
                          ins->imm, ins->b);
            break;
        case OP_INFO_SAMPLE_V5:
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
        case OP_APPLY: (void)fprintf(out, "R%u -> R%u", ins->a, ins->b); break;
        case OP_MEASURE:
            (void)fprintf(out, "R%u %s -> R%u", ins->a,
                          ins->c <= SS_MEASURE_COLLAPSE ? measure_names[ins->c] : "unknown",
                          ins->b);
            break;
        case OP_LOAD_CONST:
            (void)fprintf(out, "R%u constant[%u]", ins->a, ins->imm); break;
        case OP_MOVE: case OP_STATE_SET: case OP_STATE_COPY:
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

void ss_disassemble(const SSChunk *chunk, FILE *out) {
    size_t offset;
    (void)fprintf(out, "SSBC v%u entry=%u constants=%zu functions=%zu instructions=%zu\n",
                  chunk->version, chunk->entry, chunk->constant_count,
                  chunk->function_count, chunk->code_count);
    for (offset = 0; offset < chunk->code_count; ++offset)
        ss_disassemble_instruction(chunk, offset, out);
}
