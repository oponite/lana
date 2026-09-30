//! Fuzz targets for the Lana Rust runtime boundary.
//!
//! This crate exposes the Rust `lana-bytecode` loader + verifier through a
//! plain `check` / `outcome` API reused by cargo-fuzz and `fuzz-driver`.

use lana_bytecode::{loader, verifier, LanaError, LanaErrorInfo};

/// Run the Rust loader and, on success, the verifier over `data`.
///
/// Never panics: every failure path is a `Result`. Returns the first error the
/// loader/verifier produces, or `Ok(())`.
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
