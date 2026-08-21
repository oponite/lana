#include "ss/error.h"

#include <stdarg.h>
#include <stdio.h>

const char *ss_error_name(SSError error) {
    static const char *names[] = {
        "SS_OK", "SS_ERR_INVALID_STATE", "SS_ERR_INVALID_PROBABILITY",
        "SS_ERR_INVALID_DEPENDENCY", "SS_ERR_TYPE", "SS_ERR_REGISTER",
        "SS_ERR_OPCODE", "SS_ERR_CONSTANT", "SS_ERR_JUMP",
        "SS_ERR_TRANSFORM", "SS_ERR_COMPOSE", "SS_ERR_MEASURE",
        "SS_ERR_HISTORY", "SS_ERR_FORMAT", "SS_ERR_IO", "SS_ERR_OOM",
        "SS_ERR_LIMIT", "SS_ERR_TASK", "SS_ERR_CANCELLED", "SS_ERR_TIMEOUT",
        "SS_ERR_INVALID_TRANSFORM_RESULT", "SS_ERR_UNSUPPORTED_OPERATION",
        "SS_ERR_UNSUPPORTED_EXACT_MEASUREMENT", "SS_ERR_INVALID_DISTRIBUTION",
        "SS_ERR_BUDGET_EXHAUSTED", "SS_ERR_KEY", "SS_ERR_PARSE",
        "SS_ERR_ASSERTION", "SS_ERR_INVALID_CONDITIONING",
        "SS_ERR_UNRESOLVED_VALUE", "SS_ERR_PATH_LIMIT"
    };
    if ((size_t)error >= sizeof(names) / sizeof(names[0])) {
        return "SS_ERR_UNKNOWN";
    }
    return names[error];
}

void ss_error_set(SSErrorInfo *info, SSError code, size_t ip, uint8_t opcode,
                  uint32_t line, const char *format, ...) {
    va_list arguments;
    if (info == NULL) {
        return;
    }
    info->code = code;
    info->ip = ip;
    info->opcode = opcode;
    info->line = line;
    va_start(arguments, format);
    (void)vsnprintf(info->message, sizeof(info->message), format, arguments);
    va_end(arguments);
}
