//! Embed the self-hosted compiler bytecode (`lana-compiler.labc`) so the wasm
//! bundle can compile Lana source without a filesystem.
//!
//! The compiler is produced by the C build (`cmake --build build`). Its path is
//! resolved from `LANA_COMPILER_LABC`, falling back to the repo build directory
//! relative to this crate.

use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let compiler = std::env::var("LANA_COMPILER_LABC")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest.join("../../../build/lana-compiler.labc"));
    if !compiler.exists() {
        panic!(
            "lana-compiler.labc not found at {}; build the C side first \
             (cmake --build build) or set LANA_COMPILER_LABC",
            compiler.display()
        );
    }
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    std::fs::copy(&compiler, out_dir.join("lana-compiler.labc"))
        .expect("failed to copy lana-compiler.labc into OUT_DIR");
    println!("cargo:rerun-if-changed={}", compiler.display());
    println!("cargo:rerun-if-env-changed=LANA_COMPILER_LABC");
}
