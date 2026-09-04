#ifndef LANA_SHA256_H
#define LANA_SHA256_H

#include <stddef.h>
#include <stdint.h>

#define LANA_SHA256_DIGEST_SIZE 32u

typedef struct {
    uint32_t state[8];
    uint64_t length;
    unsigned char block[64];
    size_t block_length;
} LanaSha256;

void lana_sha256_init(LanaSha256 *context);
void lana_sha256_update(LanaSha256 *context, const void *data, size_t length);
void lana_sha256_final(LanaSha256 *context,
                       unsigned char out[LANA_SHA256_DIGEST_SIZE]);
void lana_sha256(const void *data, size_t length,
                 unsigned char out[LANA_SHA256_DIGEST_SIZE]);

#endif
