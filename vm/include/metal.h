#ifndef LANA_METAL_H
#define LANA_METAL_H

#include <stdbool.h>
#include <stddef.h>

/* Explicit GPU matmul (LIP-004 section 5). A single float32 general matrix
 * multiply on the system Metal device: c = a . b where a is m x k, b is k x n,
 * and c is m x n, all row-major float32. The caller downcasts binary64 inputs
 * to float32 before the call and upcasts the result after — never silently.
 *
 * Returns true on success. Returns false when no Metal device is available
 * (non-Apple build, or MTLCreateSystemDefaultDevice returns NULL) or when a
 * Metal call fails; the caller maps that to LANA_ERR_UNSUPPORTED_OPERATION. */
bool lana_metal_sgemm(size_t m, size_t k, size_t n,
                      const float *a, const float *b, float *c);

#endif
