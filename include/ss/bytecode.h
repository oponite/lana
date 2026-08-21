#ifndef SS_BYTECODE_H
#define SS_BYTECODE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "ss/error.h"
#include "ss/value.h"

#define SSBC_VERSION 5u
#define SSBC_MIN_VERSION 1u
#define SS_NO_OPERAND UINT32_MAX
#define SS_MAX_REGISTERS 256u
#define SS_MAX_CALL_FRAMES 64u

typedef enum {
    OP_NOP = 0,
    OP_LOAD_CONST,
    OP_MOVE,
    OP_STATE_NEW,
    OP_STATE_SET,
    OP_STATE_COPY,
    OP_APPLY,
    OP_APPLY_MANY,
    OP_TRANSFORM,
    OP_COMPOSE_MERGE,
    OP_COMPOSE_UPDATE,
    OP_COMPOSE_JOINT,
    OP_MEASURE,
    OP_GET_FIELD,
    OP_GET_INDEX,
    OP_SET_INDEX,
    OP_HISTORY_CONFIG,
    OP_PREVIOUS,
    OP_CHANGE,
    OP_VELOCITY,
    OP_BINARY,
    OP_UNARY,
    OP_COMPARE,
    OP_JUMP,
    OP_JUMP_IF_TRUE,
    OP_JUMP_IF_FALSE,
    OP_ARRAY_NEW,
    OP_ARRAY_GET,
    OP_ARRAY_SET,
    OP_CALL,
    OP_RETURN,
    OP_PRINT,
    OP_HALT,

    /* Bytecode v2 additions are appended to preserve every v1 opcode value. */
    OP_STATE_BUILD,
    OP_FORK,
    OP_JOIN,
    OP_JOIN_TIMEOUT,
    OP_JOIN_ALL,
    OP_CANCEL,
    OP_TASKGROUP_ENTER,
    OP_TASKGROUP_EXIT,
    OP_HOST_CALL,

    /* Bytecode v3 / Lana 1.0 additions. */
    OP_STATE_NEW_V3,
    OP_STATE_BUILD_V3,
    OP_TRANSFORM_V3,
    OP_MEASURE_V3,
    OP_APPEND,
    OP_SAMPLE_STATE_DIST,
    /* Bytecode v4 / basis-aware measurement additions. */
    OP_MEASURE_BASIS_V4,
    OP_ESTIMATE_MEASURE_PROBABILITY_V4,
    OP_ESTIMATE_MEASURE_DISTRIBUTION_V4,
    /* Bytecode v5 / Information layer. */
    OP_JOINT_BUILD_V5,
    OP_JOINT_PROJECT_V5,
    OP_JOINT_CONDITION_V5,
    OP_JOINT_SAMPLE_V5,
    OP_RESOLVE_V5,
    OP_JOINT_BUILD_FINITE_V5,
    OP_JOINT_RENAME_V5,
    OP_POSSIBILITY_BUILD_V5,
    OP_PATH_SPLIT_V5,
    OP_PATH_JOIN_V5,
    OP_OBSERVE_V5,
    OP_INFO_SAMPLE_V5,
    OP_COUNT
} OpCode;

typedef enum {
    SS_HOST_ARGS = 0,
    SS_HOST_READ_TEXT,
    SS_HOST_WRITE_TEXT,
    SS_HOST_NOW,
    SS_HOST_RANDOM,
    SS_HOST_ASSERT
    ,SS_HOST_ML_FIT_RIDGE
    ,SS_HOST_ML_FIT_LOGISTIC
    ,SS_HOST_ML_PREDICT
    ,SS_HOST_ML_STANDARDIZE
    ,SS_HOST_ML_POLYNOMIAL
    ,SS_HOST_ML_RBF
    ,SS_HOST_ML_REGRESSION_METRICS
    ,SS_HOST_ML_CLASSIFICATION_METRICS
    ,SS_HOST_ML_MODEL_WRITE
    ,SS_HOST_ML_MODEL_READ
    ,SS_HOST_MAP_NEW
    ,SS_HOST_INDEX_GET
    ,SS_HOST_INDEX_SET
    ,SS_HOST_JSON_PARSE
    ,SS_HOST_JSON_STRINGIFY
    ,SS_HOST_CSV_READ
    ,SS_HOST_CSV_WRITE
    ,SS_HOST_STRING_LENGTH
    ,SS_HOST_STRING_BYTE_AT
    ,SS_HOST_STRING_SLICE
    ,SS_HOST_STRING_CONCAT
    ,SS_HOST_NUMBER_TO_STRING
    ,SS_HOST_ARRAY_NEW
    ,SS_HOST_ARRAY_PUSH
    ,SS_HOST_STRING_HEX
    ,SS_HOST_STRING_JOIN
    ,SS_HOST_ARRAY_LENGTH
    ,SS_HOST_STRING_UNESCAPE
    ,SS_HOST_PATH_RESOLVE
} SSHostCallId;

typedef enum {
    SS_MEASURE_DISTRIBUTION = 0,
    SS_MEASURE_PROBABILITY,
    SS_MEASURE_SAMPLE,
    SS_MEASURE_COLLAPSE
} SSMeasureId;

typedef enum {
    SS_TRANSFORM_DECAY = 0,
    SS_TRANSFORM_REINFORCE,
    SS_TRANSFORM_INVERT,
    SS_TRANSFORM_NEUTRALIZE,
    SS_TRANSFORM_SHIFT,
    SS_TRANSFORM_CLAMP,
    SS_TRANSFORM_RESET_D
} SSTransformId;

typedef enum {
    SS_TRANSFORM_V3_INVERT = 0,
    SS_TRANSFORM_V3_NEUTRALIZE = 1
} SSTransformIdV3;

typedef enum {
    SS_MEASURE_V3_PROBABILITY = 0,
    SS_MEASURE_V3_DISTRIBUTION = 1,
    SS_MEASURE_V3_SAMPLE = 2
} SSMeasureIdV3;

typedef enum {
    SS_MEASURE_BASIS_COMPUTATIONAL = 0,
    SS_MEASURE_BASIS_X = 1,
    SS_MEASURE_BASIS_Y = 2
} SSMeasureBasisId;

typedef enum {
    SS_BINARY_ADD = 0,
    SS_BINARY_SUBTRACT,
    SS_BINARY_MULTIPLY,
    SS_BINARY_DIVIDE
} SSBinaryId;

typedef enum {
    SS_COMPARE_EQUAL = 0,
    SS_COMPARE_NOT_EQUAL,
    SS_COMPARE_LESS,
    SS_COMPARE_LESS_EQUAL,
    SS_COMPARE_GREATER,
    SS_COMPARE_GREATER_EQUAL
} SSCompareId;

typedef enum {
    SS_AGG_WEIGHTED = 0,
    SS_AGG_MEAN,
    SS_AGG_SEQUENTIAL,
    SS_AGG_STRONGEST
} SSAggregationId;

typedef struct {
    uint8_t opcode;
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint32_t imm;
    uint32_t line;
} SSInstruction;

typedef struct {
    char *name;
    uint32_t entry;
    uint32_t register_count;
    uint32_t arity;
} SSFunction;

typedef struct {
    uint32_t version;
    uint32_t entry;
    SSInstruction *code;
    size_t code_count;
    size_t code_capacity;
    Value *constants;
    size_t constant_count;
    size_t constant_capacity;
    SSFunction *functions;
    size_t function_count;
    size_t function_capacity;
} SSChunk;

void ss_chunk_init(SSChunk *chunk);
void ss_chunk_free(SSChunk *chunk);
SSError ss_chunk_add_constant(SSChunk *chunk, Value value, uint32_t *index);
SSError ss_chunk_add_function(SSChunk *chunk, const char *name, uint32_t entry,
                              uint32_t register_count, uint32_t arity,
                              uint32_t *index);
SSError ss_chunk_emit(SSChunk *chunk, SSInstruction instruction);
SSError ss_chunk_verify(const SSChunk *chunk, SSErrorInfo *error);
SSError ss_chunk_write_file(const SSChunk *chunk, const char *path, SSErrorInfo *error);
SSError ss_chunk_read_file(SSChunk *chunk, const char *path, SSErrorInfo *error);
const char *ss_opcode_name(uint8_t opcode);
void ss_disassemble_instruction(const SSChunk *chunk, size_t offset, FILE *out);
void ss_disassemble(const SSChunk *chunk, FILE *out);

#endif
