#include "metal.h"

bool lana_metal_sgemm(size_t m, size_t k, size_t n,
                      const float *a, const float *b, float *c) {
    (void)m;
    (void)k;
    (void)n;
    (void)a;
    (void)b;
    (void)c;
    return false;
}

bool lana_metal_available(void) {
    return false;
}

void *lana_metal_buffer_create(const void *bytes, size_t length) {
    (void)bytes;
    (void)length;
    return NULL;
}

void lana_metal_buffer_release(void *buffer) {
    (void)buffer;
}

void *lana_metal_buffer_contents(void *buffer) {
    (void)buffer;
    return NULL;
}

bool lana_metal_buffer_copy(void *buffer, void *bytes, size_t length) {
    (void)buffer;
    (void)bytes;
    (void)length;
    return false;
}

bool lana_metal_buffer_sgemm(size_t m, size_t k, size_t n,
                             void *a, size_t a_offset,
                             void *b, size_t b_offset,
                             void *c, size_t c_offset) {
    (void)m;
    (void)k;
    (void)n;
    (void)a;
    (void)a_offset;
    (void)b;
    (void)b_offset;
    (void)c;
    (void)c_offset;
    return false;
}
