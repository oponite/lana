#ifndef LANA_BACKEND_H
#define LANA_BACKEND_H

#include <stdbool.h>
#include <stddef.h>

/* The single native backend dispatch point for dense tensor math (LIP-004
 * section 5). Both VMs route `matmul` through this interface so C11 and Rust
 * issue identical backend calls per platform and stay byte-identical. */

typedef struct {
    size_t m, k, n;   /* C = A(m x k) . B(k x n), row-major throughout */
    bool is_complex;  /* components interleaved: [re, im] per element */
    bool fp32;        /* LIP-027: accumulate in binary32 (f32/f16/bf16 inputs) */
    const double *a;  /* m * k elements (complex: 2 * m * k doubles) */
    size_t lda;       /* row stride of a in elements, >= k, or 1 when k == 0 */
    const double *b;  /* k * n elements */
    size_t ldb;       /* row stride of b in elements, >= n, or 1 when n == 0 */
    double *c;        /* m * n result elements, written, never read */
} LanaGemmCall;

/* One row-major general matrix multiply: c = a . b. A zero outer dimension
 * writes nothing; a zero inner dimension writes zeros (the sum over an empty
 * contraction is zero). Operands must already be row-major with the given
 * leading dimensions; callers pack anything that cannot be expressed that
 * way into contiguous scratch first. */
void lana_backend_gemm(const LanaGemmCall *call);

#endif
