#ifndef LANA_CODEC_H
#define LANA_CODEC_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "value.h"
#include "error.h"

typedef struct {
    unsigned char *data;
    size_t length;
    size_t capacity;
} LanaBuffer;

LanaError lana_codec_encode_value(LanaBuffer *buf, Value value);
LanaError lana_codec_decode_value(LanaBuffer *buf, size_t *offset, Value *out_value);
LanaError lana_codec_decode_document(LanaBuffer *buf, Value *out_value);

#endif
