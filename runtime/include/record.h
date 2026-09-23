#ifndef LANA_RECORD_H
#define LANA_RECORD_H

#include <stddef.h>
#include <stdint.h>

#define LANA_RECORD_ABI_VERSION 1u

typedef struct LanaRecord LanaRecord;
typedef struct LanaRecordBuffer {
    size_t struct_size;
    uint8_t *data;
    size_t length;
} LanaRecordBuffer;

uint32_t lana_record_abi_version(void);
int lana_record_parse(const uint8_t *data, size_t length, LanaRecord **out);
int lana_record_json(const LanaRecord *record, LanaRecordBuffer *out);
void lana_record_free(LanaRecord *record);
void lana_record_buffer_free(LanaRecordBuffer *buffer);

/* This header describes the shipped Rust record subset. Decision, inspection,
 * and Execution record APIs remain unimplemented; the native C bridge is a
 * separate ABI. Free output buffers before reusing them. Input pointers must
 * address at least length bytes; returned buffers belong to lana-ffi. */

#endif
