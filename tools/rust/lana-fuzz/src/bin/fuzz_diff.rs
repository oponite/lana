//! Differential driver: run the same bytes through the Rust loader/verifier and
//! the C11 `lanavm verify` subprocess, and compare accept/reject + error code.
//!
//!   cargo run -p lana-fuzz --bin fuzz-diff -- [--lanavm PATH] FILE... | -
//!
//! The C11 binary defaults to `$LANAVM`, else `build/lanavm` relative to the
//! repo root (this crate lives at `tools/rust/lana-fuzz`, so the repo
//! root is three levels up). Each input is written to a temp file because
//! `lanavm verify` reads from a path; the C11 loader's `LANA_ERR_IO` path is
//! therefore never exercised (the Rust loader reads bytes directly and has no
//! IO error).
//!
//! Exits non-zero if any input diverges between the two implementations.

use std::io::Read;
use std::process::{Command, ExitCode};

fn main() -> ExitCode {
    let mut lanavm = std::env::var("LANAVM").unwrap_or_else(|_| {
        let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("../../../build/lanavm");
        path.to_string_lossy().into_owned()
    });

    let mut files: Vec<String> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--lanavm" {
            match args.next() {
                Some(path) => lanavm = path,
                None => {
                    eprintln!("--lanavm requires a path");
                    return ExitCode::from(2);
                }
            }
        } else {
            files.push(arg);
        }
    }

    if files.is_empty() {
        eprintln!("usage: fuzz-diff [--lanavm PATH] FILE... | -");
        return ExitCode::from(2);
    }
    if !std::path::Path::new(&lanavm).exists() {
        eprintln!("C11 lanavm not found at {lanavm} (build it first, or set LANAVM)");
        return ExitCode::from(2);
    }

    let mut mismatches = 0usize;
    let mut count = 0usize;
    for file in &files {
        count += 1;
        let bytes = match read_input(file) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("{file}: {e}");
                mismatches += 1;
                continue;
            }
        };

        let rust = match std::panic::catch_unwind(|| lana_fuzz::outcome(&bytes)) {
            Ok(o) => o,
            Err(_) => {
                println!("MISMATCH {file}: rust=PANIC c11=?");
                mismatches += 1;
                continue;
            }
        };

        let c11 = match c11_verify(&lanavm, &bytes) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("{file}: C11 driver error: {e}");
                mismatches += 1;
                continue;
            }
        };

        if rust.name() == c11 {
            println!("MATCH    {file}: {}", rust.name());
        } else {
            println!("MISMATCH {file}: rust={} c11={}", rust.name(), c11);
            mismatches += 1;
        }
    }

    println!();
    println!("{}/{} inputs match", count - mismatches, count);
    if mismatches == 0 { ExitCode::SUCCESS } else { ExitCode::from(1) }
}

fn read_input(arg: &str) -> Result<Vec<u8>, String> {
    if arg == "-" {
        let mut buf = Vec::new();
        std::io::stdin()
            .read_to_end(&mut buf)
            .map_err(|e| e.to_string())?;
        Ok(buf)
    } else {
        std::fs::read(arg).map_err(|e| e.to_string())
    }
}

/// Run the C11 `lanavm verify` subprocess over `bytes` and reduce the result to
/// the stable `LANA_OK` / `LANA_ERR_*` name.
fn c11_verify(lanavm: &str, bytes: &[u8]) -> Result<String, String> {
    let path = std::env::temp_dir().join(format!(
        "lana-fuzz-diff-{}-{:x}.labc",
        std::process::id(),
        // Distinguish multiple inputs in one process without a global counter.
        bytes.len() ^ (bytes.iter().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(*b as u64)) as usize)
    ));
    std::fs::write(&path, bytes).map_err(|e| e.to_string())?;

    let output = Command::new(lanavm)
        .arg("verify")
        .arg(&path)
        .output()
        .map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&path);

    if output.status.success() {
        return Ok("LANA_OK".to_string());
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    for token in stderr.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
        if token.starts_with("LANA_ERR_") {
            return Ok(token.to_string());
        }
    }
    Err(format!("unrecognized C11 error output: {stderr}"))
}
