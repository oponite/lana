//! Differential fuzz targets for the Lana Rust runtime boundary.
//!
//! This crate exposes the Rust `lana-bytecode` loader + verifier as:
//!
//! * a libFuzzer-compatible entry point (`LLVMFuzzerTestOneInput`), mirroring
//!   the C11 reference target `tests/unit/fuzz_bytecode.c`, and
//! * a plain `check` / `outcome` API reused by the `fuzz-driver` (single-input
//!   driver) and `fuzz-diff` (differential driver) binaries.
//!
//! The differential driver shells out to the C11 `lanavm verify` subcommand and
//! compares accept/reject plus the stable `LANA_ERR_*` error code on identical
//! inputs. See `tests/differential/run_fuzz_diff.sh` for the end-to-end wiring.

use lana_bytecode::{loader, verifier, LanaError, LanaErrorInfo};

/// Run the Rust loader and, on success, the verifier over `data`.
///
/// Never panics: every failure path is a `Result`. Returns the first error the
/// C11 loader/verifier would produce for the same bytes, or `Ok(())`.
pub fn check(data: &[u8]) -> Result<(), LanaErrorInfo> {
    let chunk = loader::load(data)?;
    verifier::verify(&chunk)
}

/// The accept/reject outcome of [`check`], reduced to the stable error code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Ok,
    Err(LanaError),
}

impl Outcome {
    /// The stable C11 name for this outcome, e.g. `LANA_OK` or `LANA_ERR_FORMAT`.
    pub fn name(self) -> &'static str {
        match self {
            Outcome::Ok => "LANA_OK",
            Outcome::Err(code) => code.name(),
        }
    }
}

/// Reduce [`check`] to an [`Outcome`] for differential comparison.
pub fn outcome(data: &[u8]) -> Outcome {
    match check(data) {
        Ok(()) => Outcome::Ok,
        Err(info) => Outcome::Err(info.code),
    }
}

/// libFuzzer entry point. Mirrors `tests/unit/fuzz_bytecode.c`: run the loader and,
/// if it loads, the verifier. Never aborts and never panics on arbitrary input.
///
/// # Safety
/// `data` must point to `size` readable bytes, or be null when `size == 0`.
#[no_mangle]
pub extern "C" fn LLVMFuzzerTestOneInput(data: *const u8, size: usize) -> i32 {
    let bytes = if data.is_null() {
        &[][..]
    } else {
        // SAFETY: libFuzzer guarantees `data` is valid for `size` bytes.
        unsafe { std::slice::from_raw_parts(data, size) }
    };
    let _ = check(bytes);
    0
}
