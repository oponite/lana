#include "backend.h"

#include <stdlib.h>

/* Native matmul dispatch (LIP-004 section 5). The Accelerate build defines
 * LANA_BLAS_ACCELERATE and links the framework; every other build runs the
 * same naive loop the previous matmul used, so both VMs keep issuing the
 * same sums in the same order on a given platform. */

#ifdef LANA_BLAS_ACCELERATE
/* Accelerate's CBLAS follows the original ATLAS argument order: the size
 * parameters are (M, N, K), not netlib's (M, K, N). The entry points are
 * declared directly so the pedantic build never includes vendor headers. */
enum { LANA_CBLAS_ROW_MAJOR = 101, LANA_CBLAS_NO_TRANS = 111 };

void cblas_sgemm(int order, int transa, int transb, int m, int n, int k,
                 float alpha, const float *a, int lda, const float *b,
                 int ldb, float beta, float *c, int ldc);
void cblas_dgemm(int order, int transa, int transb, int m, int n, int k,
                 double alpha, const double *a, int lda, const double *b,
                 int ldb, double beta, double *c, int ldc);
void cblas_zgemm(int order, int transa, int transb, int m, int n, int k,
                 const void *alpha, const void *a, int lda, const void *b,
                 int ldb, const void *beta, void *c, int ldc);

/* LIP-027 item 9: route the fp32-accumulation matmul (f32/f16/bf16 inputs)
 * through cblas_sgemm. The caller stages every element as a double (half /
 * bf16 / f32 storage is already widened to double), so pack each operand's
 * logical elements into a float scratch, run the fp32 BLAS product, and store
 * the float result upcast to a double. Returns nonzero when the call was
 * handled; returns zero (so the caller falls back to the naive loop) when a
 * dimension exceeds the BLAS int range, a scratch size would overflow, or the
 * scratch cannot be allocated. */
static int gemm_sgemm(const LanaGemmCall *call) {
    size_t lda = call->lda != 0u ? call->lda : 1u;
    size_t ldb = call->ldb != 0u ? call->ldb : 1u;
    size_t m = call->m, k = call->k, n = call->n;
    if (m > 2147483647u || k > 2147483647u || n > 2147483647u ||
        lda > 2147483647u || ldb > 2147483647u)
        return 0;
    /* A packed product is a[k][n] and fa[m][k]: sizes cannot grow past the
     * caller's own packed double buffers, which live under the VM memory
     * limit, but guard the multiply anyway so a huge m*n*k cannot overflow. */
    size_t a_cnt = m * k, b_cnt = k * n, c_cnt = m * n;
    if ((k != 0u && a_cnt / k != m) || (n != 0u && b_cnt / n != k) ||
        (n != 0u && c_cnt / n != m))
        return 0;
    float *fa = malloc((a_cnt != 0u ? a_cnt : 1u) * sizeof(*fa));
    float *fb = malloc((b_cnt != 0u ? b_cnt : 1u) * sizeof(*fb));
    float *fc = malloc((c_cnt != 0u ? c_cnt : 1u) * sizeof(*fc));
    if (fa == NULL || fb == NULL || fc == NULL) {
        free(fa);
        free(fb);
        free(fc);
        return 0;
    }
    for (size_t i = 0; i < m; ++i)
        for (size_t p = 0; p < k; ++p)
            fa[i * k + p] = (float)call->a[i * lda + p];
    for (size_t p = 0; p < k; ++p)
        for (size_t j = 0; j < n; ++j)
            fb[p * n + j] = (float)call->b[p * ldb + j];
    /* Row-major, no transpose. lda' >= max(1,k), ldb' >= max(1,n), ldc = n. A
     * zero inner dimension leaves the accumulator at zero (beta = 0). */
    cblas_sgemm(LANA_CBLAS_ROW_MAJOR, LANA_CBLAS_NO_TRANS, LANA_CBLAS_NO_TRANS,
                (int)m, (int)n, (int)k, 1.0f, fa, (int)(k != 0u ? k : 1u), fb,
                (int)(n != 0u ? n : 1u), 0.0f, fc, (int)n);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            call->c[i * n + j] = (double)fc[i * n + j];
    free(fa);
    free(fb);
    free(fc);
    return 1;
}
#endif

void lana_backend_gemm(const LanaGemmCall *call) {
    if (call->m == 0u || call->n == 0u) {
        return;
    }
    /* LIP-027: binary32 accumulation for f32/f16/bf16 inputs. On macOS the
     * Accelerate cblas_sgemm handles it (item 9); elsewhere a naive loop
     * accumulates in float, operation-for-operation the same as the Rust
     * backend's fp32 fallback so both VMs stay byte-identical. Complex never
     * takes this path (it stays binary64). */
    if (call->fp32) {
#ifdef LANA_BLAS_ACCELERATE
        if (gemm_sgemm(call)) return;
#endif
        size_t lda = call->lda != 0u ? call->lda : 1u;
        size_t ldb = call->ldb != 0u ? call->ldb : 1u;
        for (size_t i = 0; i < call->m; ++i) {
            for (size_t j = 0; j < call->n; ++j) {
                float acc = 0.0f;
                for (size_t p = 0; p < call->k; ++p) {
                    float av = (float)call->a[i * lda + p];
                    float bv = (float)call->b[p * ldb + j];
                    acc += av * bv;
                }
                call->c[i * call->n + j] = (double)acc;
            }
        }
        return;
    }
    /* Leading dimensions are at least 1 so a zero inner dimension cannot
     * trip BLAS parameter validation (nothing is read when k is 0). */
    size_t lda = call->lda != 0u ? call->lda : 1u;
    size_t ldb = call->ldb != 0u ? call->ldb : 1u;
#ifdef LANA_BLAS_ACCELERATE
    /* CBLAS sizes are int. Any tensor that fits the memory limit is far
     * below INT_MAX, but fall back rather than truncate if that ever fails. */
    if (call->m <= 2147483647u && call->k <= 2147483647u &&
        call->n <= 2147483647u && lda <= 2147483647u && ldb <= 2147483647u) {
        int m = (int)call->m;
        int k = (int)call->k;
        int n = (int)call->n;
        if (!call->is_complex) {
            cblas_dgemm(LANA_CBLAS_ROW_MAJOR, LANA_CBLAS_NO_TRANS,
                        LANA_CBLAS_NO_TRANS, m, n, k, 1.0, call->a, (int)lda,
                        call->b, (int)ldb, 0.0, call->c, n);
        } else {
            static const double one[2] = {1.0, 0.0};
            static const double zero[2] = {0.0, 0.0};
            cblas_zgemm(LANA_CBLAS_ROW_MAJOR, LANA_CBLAS_NO_TRANS,
                        LANA_CBLAS_NO_TRANS, m, n, k, one, call->a, (int)lda,
                        call->b, (int)ldb, zero, call->c, n);
        }
        return;
    }
#endif
    /* Portable fallback: beta = 0 means c is write-only, so no clearing pass
     * is needed; a zero inner dimension leaves every accumulator at zero. */
    size_t mult = call->is_complex ? 2u : 1u;
    for (size_t i = 0; i < call->m; ++i) {
        for (size_t j = 0; j < call->n; ++j) {
            double acc_re = 0.0;
            double acc_im = 0.0;
            for (size_t p = 0; p < call->k; ++p) {
                const double *av = &call->a[(i * lda + p) * mult];
                const double *bv = &call->b[(p * ldb + j) * mult];
                if (call->is_complex) {
                    acc_re += av[0] * bv[0] - av[1] * bv[1];
                    acc_im += av[0] * bv[1] + av[1] * bv[0];
                } else {
                    acc_re += av[0] * bv[0];
                }
            }
            double *cv = &call->c[(i * call->n + j) * mult];
            cv[0] = acc_re;
            if (call->is_complex) {
                cv[1] = acc_im;
            }
        }
    }
}
