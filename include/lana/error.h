#ifndef LANA_ERROR_H
#define LANA_ERROR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define LANA_ERROR_PATH_CAPACITY 256u
#define LANA_ERROR_OPERATION_CAPACITY 64u
#define LANA_ERROR_MESSAGE_CAPACITY 256u
#define LANA_ERROR_DETAIL_CAPACITY 128u
#define LANA_ERROR_MAX_CAUSES 8u

typedef enum {
    LANA_OK = 0,
    LANA_ERR_INVALID_STATE,
    LANA_ERR_INVALID_PROBABILITY,
    LANA_ERR_INVALID_DEPENDENCY,
    LANA_ERR_TYPE,
    LANA_ERR_REGISTER,
    LANA_ERR_OPCODE,
    LANA_ERR_CONSTANT,
    LANA_ERR_JUMP,
    LANA_ERR_TRANSFORM,
    LANA_ERR_COMPOSE,
    LANA_ERR_MEASURE,
    LANA_ERR_HISTORY,
    LANA_ERR_FORMAT,
    LANA_ERR_INCOMPATIBLE_FORMAT,
    LANA_ERR_IO,
    LANA_ERR_OOM,
    LANA_ERR_LIMIT,
    LANA_ERR_TASK,
    LANA_ERR_CANCELLED,
    LANA_ERR_TIMEOUT,
    LANA_ERR_INVALID_TRANSFORM_RESULT,
    LANA_ERR_UNSUPPORTED_OPERATION,
    LANA_ERR_UNSUPPORTED_EXACT_MEASUREMENT,
    LANA_ERR_INVALID_DISTRIBUTION,
    LANA_ERR_BUDGET_EXHAUSTED,
    LANA_ERR_KEY,
    LANA_ERR_PARSE,
    LANA_ERR_ASSERTION,
    LANA_ERR_INVALID_CONDITIONING,
    LANA_ERR_UNRESOLVED_VALUE,
    LANA_ERR_PATH_LIMIT,
    LANA_ERR_CAPABILITY,
    LANA_ERR_CONFLICT
} LanaError;

typedef enum {
    LANA_ERROR_KIND_NONE = 0,
    LANA_ERROR_KIND_VALIDATION,
    LANA_ERROR_KIND_TYPE,
    LANA_ERROR_KIND_PARSE,
    LANA_ERROR_KIND_BYTECODE,
    LANA_ERROR_KIND_IO,
    LANA_ERROR_KIND_RESOURCE_LIMIT,
    LANA_ERROR_KIND_TASK,
    LANA_ERROR_KIND_CANCELLATION,
    LANA_ERROR_KIND_TIMEOUT,
    LANA_ERROR_KIND_UNSUPPORTED,
    LANA_ERROR_KIND_RESOLUTION,
    LANA_ERROR_KIND_ASSERTION
} LanaErrorKind;

typedef enum {
    LANA_RESOLUTION_REASON_NONE = 0,
    LANA_RESOLUTION_REASON_NO_ALTERNATIVES,
    LANA_RESOLUTION_REASON_MULTIPLE_ALTERNATIVES,
    LANA_RESOLUTION_REASON_CONTRADICTION,
    LANA_RESOLUTION_REASON_INVALID_CONDITIONING,
    LANA_RESOLUTION_REASON_UNSUPPORTED_EXACT,
    LANA_RESOLUTION_REASON_CANCELLED,
    LANA_RESOLUTION_REASON_RESOURCE_LIMIT
} LanaResolutionReason;

typedef enum {
    LANA_EXACT_SUPPORT_UNKNOWN = 0,
    LANA_EXACT_SUPPORT_AVAILABLE,
    LANA_EXACT_SUPPORT_UNAVAILABLE
} LanaExactSupport;

typedef enum {
    LANA_RESOURCE_NONE = 0,
    LANA_RESOURCE_MEMORY,
    LANA_RESOURCE_INSTRUCTIONS,
    LANA_RESOURCE_TASKS,
    LANA_RESOURCE_PATHS,
    LANA_RESOURCE_SAMPLES,
    LANA_RESOURCE_TIME
} LanaResourceKind;

typedef struct {
    char path[LANA_ERROR_PATH_CAPACITY];
    uint32_t start_line;
    uint32_t start_column;
    uint32_t end_line;
    uint32_t end_column;
} LanaSourceSpan;

typedef struct {
    LanaError code;
    LanaErrorKind kind;
    char operation[LANA_ERROR_OPERATION_CAPACITY];
    char message[LANA_ERROR_DETAIL_CAPACITY];
} LanaErrorCause;

typedef struct {
    bool present;
    uint64_t task_lineage;
    char reason[LANA_ERROR_DETAIL_CAPACITY];
} LanaCancellationContext;

typedef struct {
    bool present;
    LanaResourceKind resource;
    uint64_t limit;
    uint64_t observed;
    char unit[32];
} LanaResourceLimitContext;

typedef struct {
    LanaError code;
    LanaErrorKind kind;
    size_t ip;
    uint8_t opcode;
    uint32_t line;
    char function[128];
    char message[LANA_ERROR_MESSAGE_CAPACITY];
    bool has_derivation;
    uint64_t derivation_task_lineage;
    uint64_t derivation_local_sequence;

    LanaSourceSpan source;
    char operation[LANA_ERROR_OPERATION_CAPACITY];
    LanaErrorCause causes[LANA_ERROR_MAX_CAUSES];
    size_t cause_count;
    bool cause_chain_truncated;
    LanaResolutionReason resolution_reason;
    bool has_remaining_alternatives;
    size_t remaining_alternatives;
    LanaExactSupport exact_support;
    char exact_support_detail[LANA_ERROR_DETAIL_CAPACITY];
    LanaCancellationContext cancellation;
    LanaResourceLimitContext resource_limit;
} LanaErrorInfo;

const char *lana_error_name(LanaError error);
LanaErrorKind lana_error_kind_from_code(LanaError error);
const char *lana_error_kind_name(LanaErrorKind kind);
const char *lana_resolution_reason_name(LanaResolutionReason reason);
const char *lana_exact_support_name(LanaExactSupport support);
const char *lana_resource_kind_name(LanaResourceKind resource);

void lana_error_set(LanaErrorInfo *info, LanaError code, size_t ip, uint8_t opcode,
                    uint32_t line, const char *format, ...);
void lana_error_set_source_span(LanaErrorInfo *info, const char *path,
                                uint32_t start_line, uint32_t start_column,
                                uint32_t end_line, uint32_t end_column);
void lana_error_set_operation(LanaErrorInfo *info, const char *operation);
bool lana_error_add_cause(LanaErrorInfo *info, const LanaErrorInfo *cause);
void lana_error_set_resolution(LanaErrorInfo *info, LanaResolutionReason reason,
                               size_t remaining_alternatives);
void lana_error_set_exact_support(LanaErrorInfo *info, LanaExactSupport support,
                                  const char *detail);
void lana_error_set_cancellation(LanaErrorInfo *info, uint64_t task_lineage,
                                 const char *reason);
void lana_error_set_resource_limit(LanaErrorInfo *info, LanaResourceKind resource,
                                   uint64_t limit, uint64_t observed,
                                   const char *unit);

#endif
