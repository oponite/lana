//! Native matmul backend dispatch (LIP-004 section 5), mirroring
//! `vm/c/backend.c`. Both VMs must select the same backend on a given
//! platform so C/Rust differential byte identity holds: macOS links the
//! Accelerate framework, everything else runs the portable loop through the
//! same dispatch point.

#[cfg(target_os = "macos")]
mod accelerate {
    // Accelerate's CBLAS follows the original ATLAS argument order: the size
    // parameters are (M, N, K), not netlib's (M, K, N). Declaring the two
    // entry points directly keeps the build free of vendored bindings.
    const ROW_MAJOR: i32 = 101;
    const NO_TRANS: i32 = 111;

    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        fn cblas_dgemm(
            order: i32, transa: i32, transb: i32, m: i32, n: i32, k: i32,
            alpha: f64, a: *const f64, lda: i32, b: *const f64, ldb: i32,
            beta: f64, c: *mut f64, ldc: i32,
        );
        fn cblas_zgemm(
            order: i32, transa: i32, transb: i32, m: i32, n: i32, k: i32,
            alpha: *const f64, a: *const f64, lda: i32, b: *const f64,
            ldb: i32, beta: *const f64, c: *mut f64, ldc: i32,
        );
    }

    /// Returns false when a dimension or leading dimension exceeds the BLAS
    /// `int` range; any tensor under the VM memory limit is far below it,
    /// but the caller falls back to the portable loop rather than truncate.
    pub fn gemm(
        m: usize, k: usize, n: usize, complex: bool,
        a: *const f64, lda: usize, b: *const f64, ldb: usize, c: *mut f64,
    ) -> bool {
        let lda = if lda == 0 { 1 } else { lda };
        let ldb = if ldb == 0 { 1 } else { ldb };
        if m > i32::MAX as usize || k > i32::MAX as usize || n > i32::MAX as usize
            || lda > i32::MAX as usize || ldb > i32::MAX as usize
        {
            return false;
        }
        unsafe {
            if complex {
                let one = [1.0f64, 0.0];
                let zero = [0.0f64, 0.0];
                cblas_zgemm(
                    ROW_MAJOR, NO_TRANS, NO_TRANS,
                    m as i32, n as i32, k as i32,
                    one.as_ptr(), a, lda as i32, b, ldb as i32,
                    zero.as_ptr(), c, n as i32,
                );
            } else {
                cblas_dgemm(
                    ROW_MAJOR, NO_TRANS, NO_TRANS,
                    m as i32, n as i32, k as i32,
                    1.0, a, lda as i32, b, ldb as i32, 0.0, c, n as i32,
                );
            }
        }
        true
    }
}

#[cfg(target_os = "macos")]
use accelerate::gemm as blas_gemm;

#[cfg(not(target_os = "macos"))]
fn blas_gemm(
    _: usize, _: usize, _: usize, _: bool,
    _: *const f64, _: usize, _: *const f64, _: usize, _: *mut f64,
) -> bool {
    false
}

/// One row-major general matrix multiply: `c = a . b` where `a` is `m x k`,
/// `b` is `k x n`, and complex components are interleaved. `c` is written,
/// never read, so a zero inner dimension writes zeros. Operands must already
/// be row-major with the given leading dimensions; callers pack anything that
/// cannot be expressed that way into contiguous scratch first.
///
/// LIP-027: when `fp32` is set, each input element is downcast to float and
/// the dot product accumulates in binary32 (for f32/f16/bf16 inputs); the
/// result is upcast to double. Complex never takes this path.
pub fn backend_gemm(
    m: usize, k: usize, n: usize, complex: bool, fp32: bool,
    a: *const f64, lda: usize, b: *const f64, ldb: usize, c: *mut f64,
) {
    if m == 0 || n == 0 {
        return;
    }
    if fp32 {
        // Operation-for-operation the same loop as the C backend's fp32 path so
        // both VMs stay byte-identical.
        let lda = if lda == 0 { 1 } else { lda };
        let ldb = if ldb == 0 { 1 } else { ldb };
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    let av = unsafe { *a.add(i * lda + p) } as f32;
                    let bv = unsafe { *b.add(p * ldb + j) } as f32;
                    acc += av * bv;
                }
                unsafe { *c.add(i * n + j) = acc as f64 };
            }
        }
        return;
    }
    if blas_gemm(m, k, n, complex, a, lda, b, ldb, c) {
        return;
    }
    // Portable fallback, operation-for-operation the same loop as the C
    // backend's fallback so both VMs stay byte-identical off Apple platforms.
    let mult = if complex { 2 } else { 1 };
    let lda = if lda == 0 { 1 } else { lda };
    let ldb = if ldb == 0 { 1 } else { ldb };
    for i in 0..m {
        for j in 0..n {
            let mut acc_re = 0.0f64;
            let mut acc_im = 0.0f64;
            for p in 0..k {
                let av = unsafe { a.add((i * lda + p) * mult) };
                let bv = unsafe { b.add((p * ldb + j) * mult) };
                unsafe {
                    if complex {
                        let (ar, aim) = (*av, *av.add(1));
                        let (br, bim) = (*bv, *bv.add(1));
                        acc_re += ar * br - aim * bim;
                        acc_im += ar * bim + aim * br;
                    } else {
                        acc_re += *av * *bv;
                    }
                }
            }
            let cv = unsafe { c.add((i * n + j) * mult) };
            unsafe {
                *cv = acc_re;
                if complex {
                    *cv.add(1) = acc_im;
                }
            }
        }
    }
}