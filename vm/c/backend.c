#include "backend.h"

/* Native matmul dispatch (LIP-004 section 5). The Accelerate build defines
 * LANA_BLAS_ACCELERATE and links the framework; every other build runs the
 * same naive loop the previous matmul used, so both VMs keep issuing the
 * same sums in the same order on a given platform. */

#ifdef LANA_BLAS_ACCELERATE
/* Accelerate's CBLAS follows the original ATLAS argument order: the size
 * parameters are (M, N, K), not netlib's (M, K, N). The entry points are
 * declared directly so the pedantic build never includes vendor headers. */
enum { LANA_CBLAS_ROW_MAJOR = 101, LANA_CBLAS_NO_TRANS = 111 };

void cblas_dgemm(int order, int transa, int transb, int m, int n, int k,
                 double alpha, const double *a, int lda, const double *b,
                 int ldb, double beta, double *c, int ldc);
void cblas_zgemm(int order, int transa, int transb, int m, int n, int k,
                 const void *alpha, const void *a, int lda, const void *b,
                 int ldb, const void *beta, void *c, int ldc);
#endif

void lana_backend_gemm(const LanaGemmCall *call) {
    if (call->m == 0u || call->n == 0u) {
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