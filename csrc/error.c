#include "lana/error.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static void copy_text(char *destination, size_t capacity, const char *source) {
    if (capacity == 0u) return;
    if (source == NULL) {
        destination[0] = '\0';
        return;
    }
    (void)snprintf(destination, capacity, "%s", source);
}

const char *lana_error_name(LanaError error) {
    static const char *names[] = {
        "LANA_OK", "LANA_ERR_INVALID_STATE", "LANA_ERR_INVALID_PROBABILITY",
        "LANA_ERR_INVALID_DEPENDENCY", "LANA_ERR_TYPE", "LANA_ERR_REGISTER",
        "LANA_ERR_OPCODE", "LANA_ERR_CONSTANT", "LANA_ERR_JUMP",
        "LANA_ERR_TRANSFORM", "LANA_ERR_COMPOSE", "LANA_ERR_MEASURE",
        "LANA_ERR_HISTORY", "LANA_ERR_FORMAT", "LANA_ERR_INCOMPATIBLE_FORMAT",
        "LANA_ERR_IO", "LANA_ERR_OOM",
        "LANA_ERR_LIMIT", "LANA_ERR_TASK", "LANA_ERR_CANCELLED", "LANA_ERR_TIMEOUT",
        "LANA_ERR_INVALID_TRANSFORM_RESULT", "LANA_ERR_UNSUPPORTED_OPERATION",
        "LANA_ERR_UNSUPPORTED_EXACT_MEASUREMENT", "LANA_ERR_INVALID_DISTRIBUTION",
        "LANA_ERR_BUDGET_EXHAUSTED", "LANA_ERR_KEY", "LANA_ERR_PARSE",
        "LANA_ERR_ASSERTION", "LANA_ERR_INVALID_CONDITIONING",
        "LANA_ERR_UNRESOLVED_VALUE", "LANA_ERR_PATH_LIMIT",
        "LANA_ERR_CAPABILITY", "LANA_ERR_CONFLICT"
    };
    if ((size_t)error >= sizeof(names) / sizeof(names[0])) return "LANA_ERR_UNKNOWN";
    return names[error];
}

LanaErrorKind lana_error_kind_from_code(LanaError error) {
    switch (error) {
        case LANA_OK: return LANA_ERROR_KIND_NONE;
        case LANA_ERR_TYPE: return LANA_ERROR_KIND_TYPE;
        case LANA_ERR_PARSE: return LANA_ERROR_KIND_PARSE;
        case LANA_ERR_REGISTER:
        case LANA_ERR_OPCODE:
        case LANA_ERR_CONSTANT:
        case LANA_ERR_JUMP:
        case LANA_ERR_FORMAT:
        case LANA_ERR_INCOMPATIBLE_FORMAT:
            return LANA_ERROR_KIND_BYTECODE;
        case LANA_ERR_IO: return LANA_ERROR_KIND_IO;
        case LANA_ERR_OOM:
        case LANA_ERR_LIMIT:
        case LANA_ERR_BUDGET_EXHAUSTED:
        case LANA_ERR_PATH_LIMIT:
            return LANA_ERROR_KIND_RESOURCE_LIMIT;
        case LANA_ERR_TASK: return LANA_ERROR_KIND_TASK;
        case LANA_ERR_CANCELLED: return LANA_ERROR_KIND_CANCELLATION;
        case LANA_ERR_TIMEOUT: return LANA_ERROR_KIND_TIMEOUT;
        case LANA_ERR_UNSUPPORTED_OPERATION:
        case LANA_ERR_UNSUPPORTED_EXACT_MEASUREMENT:
            return LANA_ERROR_KIND_UNSUPPORTED;
        case LANA_ERR_INVALID_CONDITIONING:
        case LANA_ERR_UNRESOLVED_VALUE:
            return LANA_ERROR_KIND_RESOLUTION;
        case LANA_ERR_ASSERTION:
            return LANA_ERROR_KIND_ASSERTION;
        case LANA_ERR_INVALID_STATE:
        case LANA_ERR_INVALID_PROBABILITY:
        case LANA_ERR_INVALID_DEPENDENCY:
        case LANA_ERR_TRANSFORM:
        case LANA_ERR_COMPOSE:
        case LANA_ERR_MEASURE:
        case LANA_ERR_HISTORY:
        case LANA_ERR_INVALID_TRANSFORM_RESULT:
        case LANA_ERR_INVALID_DISTRIBUTION:
        case LANA_ERR_KEY:
        case LANA_ERR_CAPABILITY:
        case LANA_ERR_CONFLICT:
            return LANA_ERROR_KIND_VALIDATION;
    }
    return LANA_ERROR_KIND_ASSERTION;
}

const char *lana_error_kind_name(LanaErrorKind kind) {
    static const char *names[] = {
        "none", "validation", "type", "parse", "bytecode", "io", "resource-limit",
        "task", "cancellation", "timeout", "unsupported", "resolution", "assertion"
    };
    if ((size_t)kind >= sizeof(names) / sizeof(names[0])) return "unknown";
    return names[kind];
}

const char *lana_resolution_reason_name(LanaResolutionReason reason) {
    static const char *names[] = {
        "none", "no-alternatives", "multiple-alternatives", "contradiction",
        "invalid-conditioning", "unsupported-exact", "cancelled", "resource-limit"
    };
    if ((size_t)reason >= sizeof(names) / sizeof(names[0])) return "unknown";
    return names[reason];
}

const char *lana_exact_support_name(LanaExactSupport support) {
    static const char *names[] = {"unknown", "available", "unavailable"};
    if ((size_t)support >= sizeof(names) / sizeof(names[0])) return "unknown";
    return names[support];
}

const char *lana_resource_kind_name(LanaResourceKind resource) {
    static const char *names[] = {
        "none", "memory", "instructions", "tasks", "paths", "samples", "time"
    };
    if ((size_t)resource >= sizeof(names) / sizeof(names[0])) return "unknown";
    return names[resource];
}

void lana_error_set(LanaErrorInfo *info, LanaError code, size_t ip, uint8_t opcode,
                    uint32_t line, const char *format, ...) {
    va_list arguments;
    if (info == NULL) return;

    memset(info, 0, sizeof(*info));
    info->code = code;
    info->kind = lana_error_kind_from_code(code);
    info->ip = ip;
    info->opcode = opcode;
    info->line = line;
    info->source.start_line = line;
    info->source.end_line = line;
    if (format == NULL) return;
    va_start(arguments, format);
    (void)vsnprintf(info->message, sizeof(info->message), format, arguments);
    va_end(arguments);
}

void lana_error_set_source_span(LanaErrorInfo *info, const char *path,
                                uint32_t start_line, uint32_t start_column,
                                uint32_t end_line, uint32_t end_column) {
    if (info == NULL) return;
    copy_text(info->source.path, sizeof(info->source.path), path);
    info->source.start_line = start_line;
    info->source.start_column = start_column;
    info->source.end_line = end_line;
    info->source.end_column = end_column;
    info->line = start_line;
}

void lana_error_set_operation(LanaErrorInfo *info, const char *operation) {
    if (info == NULL) return;
    copy_text(info->operation, sizeof(info->operation), operation);
}

bool lana_error_add_cause(LanaErrorInfo *info, const LanaErrorInfo *cause) {
    LanaErrorCause *destination;
    if (info == NULL || cause == NULL) return false;
    if (info->cause_count >= LANA_ERROR_MAX_CAUSES) {
        info->cause_chain_truncated = true;
        return false;
    }
    destination = &info->causes[info->cause_count++];
    destination->code = cause->code;
    destination->kind = cause->kind;
    copy_text(destination->operation, sizeof(destination->operation), cause->operation);
    copy_text(destination->message, sizeof(destination->message), cause->message);
    return true;
}

void lana_error_set_resolution(LanaErrorInfo *info, LanaResolutionReason reason,
                               size_t remaining_alternatives) {
    if (info == NULL) return;
    info->resolution_reason = reason;
    info->has_remaining_alternatives = true;
    info->remaining_alternatives = remaining_alternatives;
}

void lana_error_set_exact_support(LanaErrorInfo *info, LanaExactSupport support,
                                  const char *detail) {
    if (info == NULL) return;
    info->exact_support = support;
    copy_text(info->exact_support_detail, sizeof(info->exact_support_detail), detail);
}

void lana_error_set_cancellation(LanaErrorInfo *info, uint64_t task_lineage,
                                 const char *reason) {
    if (info == NULL) return;
    info->cancellation.present = true;
    info->cancellation.task_lineage = task_lineage;
    copy_text(info->cancellation.reason, sizeof(info->cancellation.reason), reason);
}

void lana_error_set_resource_limit(LanaErrorInfo *info, LanaResourceKind resource,
                                   uint64_t limit, uint64_t observed,
                                   const char *unit) {
    if (info == NULL) return;
    info->resource_limit.present = true;
    info->resource_limit.resource = resource;
    info->resource_limit.limit = limit;
    info->resource_limit.observed = observed;
    copy_text(info->resource_limit.unit, sizeof(info->resource_limit.unit), unit);
}
