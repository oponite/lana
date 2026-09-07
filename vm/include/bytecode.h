#ifndef LANA_BYTECODE_H
#define LANA_BYTECODE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "error.h"
#include "value.h"

#define LABC_VERSION 2u
#define LABC_VERSION_1 1u
#define LABC_VERSION_3 3u
#define LABC_VERSION_4 4u
#define LANA_NO_OPERAND UINT32_MAX
#define LANA_MAX_REGISTERS 256u
#define LANA_MAX_CALL_FRAMES 64u

typedef enum {
    OP_NOP = 0,
    OP_LOAD_CONST,
    OP_MOVE,
    OP_STATE_NEW,
    OP_STATE_BUILD,
    OP_TRANSFORM,
    OP_MEASURE,
    OP_APPEND,
    OP_SAMPLE_STATE_DIST,
    OP_MEASURE_BASIS,
    OP_ESTIMATE_MEASURE_PROBABILITY,
    OP_ESTIMATE_MEASURE_DISTRIBUTION,
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

    OP_FORK,
    OP_JOIN,
    OP_JOIN_TIMEOUT,
    OP_JOIN_ALL,
    OP_CANCEL,
    OP_TASKGROUP_ENTER,
    OP_TASKGROUP_EXIT,
    OP_HOST_CALL,

    OP_JOINT_BUILD,
    OP_JOINT_PROJECT,
    OP_JOINT_CONDITION,
    OP_JOINT_SAMPLE,
    OP_RESOLVE,
    OP_JOINT_BUILD_FINITE,
    OP_JOINT_RENAME,
    OP_POSSIBILITY_BUILD,
    OP_PATH_SPLIT,
    OP_PATH_JOIN,
    OP_OBSERVE,
    OP_INFO_SAMPLE,
    OP_EVIDENCE,
    OP_ASSUME,
    OP_DERIVATION,
    OP_EXPLAIN,
    /* Lana 2.0 ISA operations. */
    OP_MIX,
    OP_MAP,
    OP_SUPPORT,
    OP_EXPECT,
    OP_VALIDATE,
    OP_REVISION,
    OP_ATTENUATE,
    OP_TRACE_DISTANCE,
    OP_APPEND_REDUNDANT,
    OP_APPEND_FULL_REDUNDANCY,
    OP_APPEND_COMPLEMENTARY,
    /* Lana 2.0 ADTs and pattern matching. */
    OP_ADT_BUILD,
    OP_ADT_CASE,
    OP_ADT_GET,
    /* Lana 2.0 lazy bounded datasets. */
    OP_LAZY,
    OP_FORCE,
    /* Lana 2.0 deterministic resampling. */
    OP_BOOTSTRAP,
    /* Lana 2.1 reverse-mode autodiff (LIP-011): load a function value so it
     * can be passed to the `grad`/`vjp` host calls. */
    OP_LOAD_FUNCTION,
    /* Lana 2.1 generator suspension (LIP-022 §2). */
    OP_GENERATOR,
    OP_YIELD,
    OP_NEXT,
    /* Lana 2.2 async/await (LIP-024). Operand semantics:
     *   OP_ASYNC <target> <arg0> <argcount> <dest> — create a cold future by
     *     calling async function <target> with <argcount> arguments starting at
     *     register <arg0>, storing the future in <dest>. The body is NOT
     *     executed (mirrors OP_GENERATOR).
     *   OP_AWAIT <future> <dest> — suspend the current async frame until the
     *     future in <future> completes, then store its result in <dest> and
     *     yield control to the event loop.
     *   OP_RUN_ASYNC <future> <dest> — run the event loop to completion on the
     *     future in <future>, then store its result in <dest>. */
    OP_ASYNC,
    OP_AWAIT,
    OP_RUN_ASYNC,
    OP_COUNT
} OpCode;

typedef enum {
    LANA_HOST_ARGS = 0,
    LANA_HOST_READ_TEXT,
    LANA_HOST_WRITE_TEXT,
    LANA_HOST_NOW,
    LANA_HOST_RANDOM,
    LANA_HOST_ASSERT,
    LANA_HOST_MAP_NEW,
    LANA_HOST_MAP_HAS,
    LANA_HOST_MAP_GET,
    LANA_HOST_MAP_SET,
    LANA_HOST_MAP_KEYS,
    LANA_HOST_INDEX_GET,
    LANA_HOST_INDEX_SET,
    LANA_HOST_JSON_PARSE,
    LANA_HOST_JSON_STRINGIFY,
    LANA_HOST_CSV_READ,
    LANA_HOST_CSV_WRITE,
    LANA_HOST_STRING_LENGTH,
    LANA_HOST_STRING_BYTE_AT,
    LANA_HOST_STRING_SLICE,
    LANA_HOST_STRING_CONCAT,
    LANA_HOST_NUMBER_TO_STRING,
    LANA_HOST_ARRAY_NEW,
    LANA_HOST_ARRAY_PUSH,
    LANA_HOST_STRING_HEX,
    LANA_HOST_STRING_JOIN,
    LANA_HOST_ARRAY_LENGTH,
    LANA_HOST_STRING_UNESCAPE,
    LANA_HOST_PATH_RESOLVE,
    LANA_HOST_SAMPLE_RECORD,
    /* Lana 1.0 reactive Information and definite-effect metadata. */
    LANA_HOST_INFORMATION_NEW,
    LANA_HOST_CLAIM_NEW,
    LANA_HOST_CLAIM_VALUE,
    LANA_HOST_CLAIM_PROPOSITION,
    LANA_HOST_CLAIM_STATUS,
    LANA_HOST_PLANNED_EFFECT_NEW,
    LANA_HOST_PLANNED_EFFECT_EXECUTE,
    LANA_HOST_PLANNED_EFFECT_STATUS,
    /* Lana 1.0 process-local shared Information capabilities. */
    LANA_HOST_SHARED_INFORMATION,
    LANA_HOST_SHARED_GRANT,
    LANA_HOST_SHARED_REVOKE,
    LANA_HOST_SHARED_SNAPSHOT,
    LANA_HOST_SHARED_AT,
    LANA_HOST_SHARED_OBSERVE,
    LANA_HOST_SHARED_REVISION,
    LANA_HOST_SHARED_IDENTITY,
    LANA_HOST_SHARED_WAIT,
    LANA_HOST_INFORMATION_INSPECT,
    /* Contract-neutral filesystem facts for Lana-owned project tooling. */
    LANA_HOST_DIRECTORY_LIST,
    LANA_HOST_DIRECTORY_CREATE,
    LANA_HOST_PATH_EXISTS,
    LANA_HOST_WRITE_TEXT_ATOMIC,
    LANA_HOST_HASH_UPDATE,
    /* Lana 2.0 lazy bounded datasets. */
    LANA_HOST_LAZY_BOUND,
    /* Lana 2.0 declared correlation (bivariate Bernoulli joint law). */
    LANA_HOST_CORRELATED,
    /* Lana 2.0 surprisal (natural-log information content in nats). */
    LANA_HOST_SURPRISAL,
    LANA_HOST_TENSOR_ALLOC,
    LANA_HOST_TENSOR_ZEROS,
    LANA_HOST_TENSOR_ONES,
    LANA_HOST_TENSOR_EYE,
    LANA_HOST_TENSOR_DTYPE,
    LANA_HOST_TENSOR_SHAPE,
    LANA_HOST_TENSOR_NDIM,
    LANA_HOST_TENSOR_ADD,
    LANA_HOST_TENSOR_SUB,
    LANA_HOST_TENSOR_MUL,
    LANA_HOST_TENSOR_DIV,
    LANA_HOST_TENSOR_MATMUL,
    LANA_HOST_TENSOR_SUM,
    LANA_HOST_TENSOR_MEAN,
    LANA_HOST_TENSOR_MAX,
    LANA_HOST_TENSOR_MIN,
    LANA_HOST_TENSOR,
    LANA_HOST_TENSOR_COMPLEX,
    LANA_HOST_GRANT,
    LANA_HOST_REVOKE,
    /* Lana 2.1 immutable sets (LIP-022 §1). */
    LANA_HOST_SET_NEW,
    LANA_HOST_SET_ADD,
    LANA_HOST_SET_CONTAINS,
    LANA_HOST_SET_UNION,
    LANA_HOST_SET_INTERSECT,
    LANA_HOST_SET_DIFFERENCE,
    /* Lana 2.1 installed stdlib (LIP-016): read an environment variable. */
    LANA_HOST_GETENV,
    /* Lana 2.1 stdlib (LIP-016): reseed the RNG and floor a number. */
    LANA_HOST_RANDOM_SEED,
    LANA_HOST_FLOOR,
    /* Lana 2.1 data interchange (LIP-023): parse a number from text. */
    LANA_HOST_STRING_TO_NUMBER,
    /* Lana 2.1 data interchange (LIP-023): runtime type name of a value. */
    LANA_HOST_TYPE_OF,
    /* Lana 2.1 text processing (LIP-021 §3): explicit formatting. */
    LANA_HOST_FORMAT,
    LANA_HOST_FORMAT_NUMBER,
    /* Lana 2.1 text processing (LIP-021 §1/§4): Unicode code points and
     * simple case mapping. */
    LANA_HOST_CHAR_LENGTH,
    LANA_HOST_STRING_CODEPOINT_SLICE,
    LANA_HOST_TO_UPPER,
    LANA_HOST_TO_LOWER,
    /* Lana 2.1 text processing (LIP-021 §2): linear-time regular expressions. */
    LANA_HOST_REGEX_COMPILE,
    LANA_HOST_REGEX_MATCH,
    LANA_HOST_REGEX_SEARCH,
    LANA_HOST_REGEX_REPLACE,
    LANA_HOST_GPU_MATMUL,
    /* Lana 2.1 linear algebra on STATEs (LIP-005): N-qubit density operators,
     * POVMs, channels, and observables as first-class values over the tensor
     * substrate. */
    LANA_HOST_DENSITY_OPERATOR,
    LANA_HOST_POVM,
    LANA_HOST_CHANNEL,
    LANA_HOST_OBSERVABLE,
    LANA_HOST_TENSOR_PRODUCT,
    LANA_HOST_PARTIAL_TRACE,
    LANA_HOST_MEASURE_WITH,
    LANA_HOST_APPLY_TO,
    LANA_HOST_EXPECT,
    LANA_HOST_MIX,
    LANA_HOST_TRACE_DISTANCE,
    LANA_HOST_IS_SEPARABLE,
    LANA_HOST_TO_STATE,
    /* Lana 2.1 reverse-mode automatic differentiation (LIP-011). */
    LANA_HOST_GRAD,
    LANA_HOST_VJP,
    /* Lana 2.1 auditable, replayable training primitive (LIP-006). */
    LANA_HOST_SGD,
    LANA_HOST_ADAM,
    LANA_HOST_TRAIN,
    /* Lana 2.1 Bayesian inference as a first-class training mode (LIP-009). */
    LANA_HOST_MCMC,
    LANA_HOST_VI,
    LANA_HOST_SMC,
    LANA_HOST_INFER,
    /* Lana 2.1 incremental / online learning (LIP-010). */
    LANA_HOST_UPDATE,
    /* Lana 2.1 whole-run reproducibility and resumability (LIP-014). */
    LANA_HOST_RESUME,
    /* Lana 2.1 differentiable STATE tensors (LIP-007): a tensor whose elements
     * are N-qubit density matrices, with differentiable APPEND, MEASURE, and
     * TRANSFORM applied element-wise. */
    LANA_HOST_STATE_TENSOR,
    LANA_HOST_APPEND,
    LANA_HOST_MEASURE,
    LANA_HOST_TRANSFORM,
    /* Lana 2.2 async/await (LIP-024 §4). */
    LANA_HOST_RUN_ASYNC,
    LANA_HOST_FUTURE_ALL,
    LANA_HOST_FUTURE_RACE,
    LANA_HOST_SLEEP,
    /* Lana 2.1 relational algebra over lazy datasets (LIP-015). */
    LANA_HOST_DATASET,
    LANA_HOST_DATASET_FILTER,
    LANA_HOST_DATASET_MAP,
    LANA_HOST_DATASET_SELECT,
    LANA_HOST_DATASET_LIMIT,
    LANA_HOST_DATASET_SORT,
    LANA_HOST_DATASET_GROUP_BY,
    LANA_HOST_DATASET_AGGREGATE,
    LANA_HOST_DATASET_JOIN,
    LANA_HOST_DATASET_MATERIALIZE,
    LANA_HOST_DATASET_EXPLAIN,
    /* Lana 2.1 durable pipeline: store, policy, ledger (LIP-015 §3, LIP-012). */
    LANA_HOST_STORE_OPEN,
    LANA_HOST_STORE_PUT,
    LANA_HOST_STORE_GET,
    LANA_HOST_STORE_DELETE,
    LANA_HOST_STORE_COMMIT,
    LANA_HOST_STORE_SCAN,
    LANA_HOST_STORE_CURRENT_REVISION,
    LANA_HOST_POLICY_EVALUATE,
    LANA_HOST_POLICY_STORE_DECISION,
    LANA_HOST_LEDGER_APPEND,
    LANA_HOST_LEDGER_QUERY,
    /* Lana 2.1 data layer: MVCC, optimistic commit, adapters (LIP-015 §3, §5). */
    LANA_HOST_STORE_GET_AT,
    LANA_HOST_STORE_SNAPSHOT,
    LANA_HOST_STORE_COMMIT_IF,
    LANA_HOST_ADAPTER_LOAD,
    LANA_HOST_ADAPTER_FETCH,
    /* Lana 2.1 two-way FFI (LIP-018). */
    LANA_HOST_FFI_DECLARE,
    LANA_HOST_FFI_LOAD,
    LANA_HOST_FFI_CALL,
    /* Lana 2.1 networking and HTTP (LIP-019). */
    LANA_HOST_HTTP_GET,
    LANA_HOST_HTTP_POST,
    LANA_HOST_SOCKET_CONNECT,
    LANA_HOST_SOCKET_SEND,
    LANA_HOST_SOCKET_RECV,
    LANA_HOST_SOCKET_CLOSE,
    LANA_HOST_COUNT
} LanaHostCallId;

typedef enum {
    LANA_MEASURE_PROBABILITY = 0,
    LANA_MEASURE_DISTRIBUTION,
    LANA_MEASURE_SAMPLE
} LanaMeasureId;

typedef enum {
    LANA_TRANSFORM_INVERT = 0,
    LANA_TRANSFORM_NEUTRALIZE
} LanaTransformId;

typedef enum {
    LANA_MEASURE_BASIS_COMPUTATIONAL = 0,
    LANA_MEASURE_BASIS_X = 1,
    LANA_MEASURE_BASIS_Y = 2
} LanaMeasureBasisId;

typedef enum {
    LANA_OBSERVABLE_PROBABILITY = 0
} LanaObservableId;

typedef enum {
    LANA_BINARY_ADD = 0,
    LANA_BINARY_SUBTRACT,
    LANA_BINARY_MULTIPLY,
    LANA_BINARY_DIVIDE
} LanaBinaryId;

typedef enum {
    LANA_COMPARE_EQUAL = 0,
    LANA_COMPARE_NOT_EQUAL,
    LANA_COMPARE_LESS,
    LANA_COMPARE_LESS_EQUAL,
    LANA_COMPARE_GREATER,
    LANA_COMPARE_GREATER_EQUAL
} LanaCompareId;

typedef struct {
    uint8_t opcode;
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint32_t imm;
    uint32_t line;
} LanaInstruction;

typedef struct {
    char *name;
    uint32_t entry;
    uint32_t register_count;
    uint32_t arity;
} LanaFunction;

typedef struct {
    uint32_t version;
    uint32_t entry;
    LanaInstruction *code;
    size_t code_count;
    size_t code_capacity;
    Value *constants;
    size_t constant_count;
    size_t constant_capacity;
    LanaFunction *functions;
    size_t function_count;
    size_t function_capacity;
} LanaChunk;

void lana_chunk_init(LanaChunk *chunk);
void lana_chunk_free(LanaChunk *chunk);
LanaError lana_chunk_add_constant(LanaChunk *chunk, Value value, uint32_t *index);
LanaError lana_chunk_add_function(LanaChunk *chunk, const char *name, uint32_t entry,
                              uint32_t register_count, uint32_t arity,
                              uint32_t *index);
LanaError lana_chunk_emit(LanaChunk *chunk, LanaInstruction instruction);
LanaError lana_chunk_verify(const LanaChunk *chunk, LanaErrorInfo *error);
LanaError lana_chunk_write_file(const LanaChunk *chunk, const char *path, LanaErrorInfo *error);
LanaError lana_chunk_read_file(LanaChunk *chunk, const char *path, LanaErrorInfo *error);
const char *lana_opcode_name(uint8_t opcode);
void lana_disassemble_instruction(const LanaChunk *chunk, size_t offset, FILE *out);
void lana_disassemble(const LanaChunk *chunk, FILE *out);

#endif
