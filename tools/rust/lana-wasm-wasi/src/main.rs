//! WASI command entry for Lana: `lana-wasm-wasi <source> [input] [capabilities]`.
//!
//! A thin wrapper over `lana_wasm_wasi::run` that reads the source program,
//! optional input string, and optional capabilities JSON from the command
//! line and prints the JSON result to stdout. Exit code 0 on success, 1 when
//! the program reports an error (mirrors the `ok` field of the JSON).

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    // args[0] is the program name. The remaining arguments are the source
    // program, the optional input string, and the optional capabilities JSON.
    // A source prefixed with "HEXBLOB:" is taken as a hex-encoded, precompiled
    // LABC blob (see `run_bytecode_hex`) so the conformance can exercise host
    // calls the compiler cannot yet emit.
    if args.len() < 2 || args.len() > 4 {
        eprintln!(
            "usage: lana-wasm-wasi <source> [input] [capabilities]"
        );
        return ExitCode::from(2);
    }
    let source = &args[1];
    let input = if args.len() >= 3 { &args[2] } else { "" };
    let capabilities = if args.len() >= 4 { &args[3] } else { "" };
    let result = if let Some(hex) = source.strip_prefix("HEXBLOB:") {
        lana_wasm_wasi::run_bytecode_hex(hex, capabilities)
    } else {
        lana_wasm_wasi::run(source, input, capabilities)
    };
    println!("{result}");
    if result.starts_with("{\"ok\":true") {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
