//! Plain single-input driver for the Rust loader/verifier fuzz target.
//!
//! Reads one or more files (or stdin with `-`) and reports the accept/reject
//! outcome of the Rust loader + verifier. Usable without libFuzzer:
//!
//!   cargo run -p lana-fuzz --bin fuzz-driver -- path/to/chunk.labc
//!   cargo run -p lana-fuzz --bin fuzz-driver -- - < chunk.labc
//!
//! Exits non-zero if any input panics or cannot be read; the loader/verifier
//! themselves never panic on arbitrary input.

use std::io::Read;
use std::process::ExitCode;

use lana_fuzz::{outcome, Outcome};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: fuzz-driver FILE... | -");
        return ExitCode::from(2);
    }

    let mut failed = false;
    for arg in &args {
        let bytes = if arg == "-" {
            let mut buf = Vec::new();
            if std::io::stdin().read_to_end(&mut buf).is_err() {
                eprintln!("-: read error");
                failed = true;
                continue;
            }
            buf
        } else {
            match std::fs::read(arg) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("{arg}: {e}");
                    failed = true;
                    continue;
                }
            }
        };

        // Belt-and-suspenders: the loader/verifier return `Result`, but a panic
        // here must be reported as a failure rather than aborting the driver.
        let result = std::panic::catch_unwind(|| outcome(&bytes));
        match result {
            Ok(Outcome::Ok) => println!("{arg}: OK"),
            Ok(Outcome::Err(code)) => println!("{arg}: ERR {}", code.name()),
            Err(_) => {
                eprintln!("{arg}: PANIC");
                failed = true;
            }
        }
    }

    if failed { ExitCode::from(1) } else { ExitCode::SUCCESS }
}
