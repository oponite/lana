#ifndef SS_ERROR_H
#define SS_ERROR_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    SS_OK = 0,
    SS_ERR_INVALID_STATE,
    SS_ERR_INVALID_PROBABILITY,
    SS_ERR_INVALID_DEPENDENCY,
    SS_ERR_TYPE,
    SS_ERR_REGISTER,
    SS_ERR_OPCODE,
    SS_ERR_CONSTANT,
    SS_ERR_JUMP,
    SS_ERR_TRANSFORM,
    SS_ERR_COMPOSE,
    SS_ERR_MEASURE,
    SS_ERR_HISTORY,
    SS_ERR_FORMAT,
    SS_ERR_IO,
    SS_ERR_OOM,
    SS_ERR_LIMIT,
    SS_ERR_TASK,
    SS_ERR_CANCELLED,
    SS_ERR_TIMEOUT
} SSError;

typedef struct {
    SSError code;
    size_t ip;
    uint8_t opcode;
    uint32_t line;
    char message[256];
} SSErrorInfo;

const char *ss_error_name(SSError error);
void ss_error_set(SSErrorInfo *info, SSError code, size_t ip, uint8_t opcode,
                  uint32_t line, const char *format, ...);

#endif
