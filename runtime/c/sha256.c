#include "sha256.h"

#include <string.h>

static const uint32_t round_constants[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
    0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
    0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
    0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
    0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
    0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
    0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
    0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

static uint32_t rotate_right(uint32_t value, unsigned count) {
    return (value >> count) | (value << (32u - count));
}

static void transform(LanaSha256 *context, const unsigned char block[64]) {
    uint32_t words[64], a, b, c, d, e, f, g, h;
    size_t index;
    for (index = 0u; index < 16u; ++index) {
        size_t offset = index * 4u;
        words[index] = ((uint32_t)block[offset] << 24u) |
                       ((uint32_t)block[offset + 1u] << 16u) |
                       ((uint32_t)block[offset + 2u] << 8u) |
                       (uint32_t)block[offset + 3u];
    }
    for (; index < 64u; ++index) {
        uint32_t s0 = rotate_right(words[index - 15u], 7u) ^ rotate_right(words[index - 15u], 18u) ^ (words[index - 15u] >> 3u);
        uint32_t s1 = rotate_right(words[index - 2u], 17u) ^ rotate_right(words[index - 2u], 19u) ^ (words[index - 2u] >> 10u);
        words[index] = words[index - 16u] + s0 + words[index - 7u] + s1;
    }
    a = context->state[0]; b = context->state[1]; c = context->state[2]; d = context->state[3];
    e = context->state[4]; f = context->state[5]; g = context->state[6]; h = context->state[7];
    for (index = 0u; index < 64u; ++index) {
        uint32_t s1 = rotate_right(e, 6u) ^ rotate_right(e, 11u) ^ rotate_right(e, 25u);
        uint32_t choose = (e & f) ^ (~e & g);
        uint32_t temporary1 = h + s1 + choose + round_constants[index] + words[index];
        uint32_t s0 = rotate_right(a, 2u) ^ rotate_right(a, 13u) ^ rotate_right(a, 22u);
        uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temporary2 = s0 + majority;
        h = g; g = f; f = e; e = d + temporary1;
        d = c; c = b; b = a; a = temporary1 + temporary2;
    }
    context->state[0] += a; context->state[1] += b; context->state[2] += c;
    context->state[3] += d; context->state[4] += e; context->state[5] += f;
    context->state[6] += g; context->state[7] += h;
}

void lana_sha256_init(LanaSha256 *context) {
    if (context == NULL) return;
    context->state[0] = 0x6a09e667u; context->state[1] = 0xbb67ae85u;
    context->state[2] = 0x3c6ef372u; context->state[3] = 0xa54ff53au;
    context->state[4] = 0x510e527fu; context->state[5] = 0x9b05688cu;
    context->state[6] = 0x1f83d9abu; context->state[7] = 0x5be0cd19u;
    context->length = 0u; context->block_length = 0u;
}

void lana_sha256_update(LanaSha256 *context, const void *data, size_t length) {
    const unsigned char *bytes = data;
    if (context == NULL || (data == NULL && length != 0u)) return;
    while (length > 0u) {
        size_t available = 64u - context->block_length;
        size_t copied = length < available ? length : available;
        memcpy(context->block + context->block_length, bytes, copied);
        context->block_length += copied; context->length += copied; bytes += copied; length -= copied;
        if (context->block_length == 64u) { transform(context, context->block); context->block_length = 0u; }
    }
}

void lana_sha256_final(LanaSha256 *context, unsigned char out[LANA_SHA256_DIGEST_SIZE]) {
    uint64_t bits;
    size_t index;
    if (context == NULL || out == NULL) return;
    bits = context->length * 8u;
    context->block[context->block_length++] = 0x80u;
    if (context->block_length > 56u) {
        memset(context->block + context->block_length, 0, 64u - context->block_length);
        transform(context, context->block); context->block_length = 0u;
    }
    memset(context->block + context->block_length, 0, 56u - context->block_length);
    for (index = 0u; index < 8u; ++index) context->block[63u - index] = (unsigned char)(bits >> (index * 8u));
    transform(context, context->block);
    for (index = 0u; index < 8u; ++index) {
        out[index * 4u] = (unsigned char)(context->state[index] >> 24u);
        out[index * 4u + 1u] = (unsigned char)(context->state[index] >> 16u);
        out[index * 4u + 2u] = (unsigned char)(context->state[index] >> 8u);
        out[index * 4u + 3u] = (unsigned char)context->state[index];
    }
}

void lana_sha256(const void *data, size_t length, unsigned char out[LANA_SHA256_DIGEST_SIZE]) {
    LanaSha256 context;
    lana_sha256_init(&context); lana_sha256_update(&context, data, length); lana_sha256_final(&context, out);
}
