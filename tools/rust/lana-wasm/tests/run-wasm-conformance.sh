#!/usr/bin/env bash
# Build the `lana-wasm` crate for wasm32-unknown-unknown, generate the nodejs
# bindings, and run the native-vs-WASM conformance assertions.
#
# Prerequisites (one-time):
#   rustup target add wasm32-unknown-unknown
#   cargo install wasm-bindgen-cli --version 0.2.100
#
# The wasm-bindgen CLI version must match the `wasm-bindgen` crate version
# pinned in tools/rust/lana-wasm/Cargo.toml.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"

WASM_BINDGEN="${WASM_BINDGEN:-wasm-bindgen}"
OUT_DIR="$REPO_ROOT/target/wasm-bindgen-nodejs"

RUSTC="$(rustup which rustc)"
export RUSTC
if [[ "$(uname -s)" == Darwin ]]; then
    # rust-lld's rpath can omit the toolchain's top-level LLVM library directory.
    export DYLD_FALLBACK_LIBRARY_PATH="$("$RUSTC" --print sysroot)/lib${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
fi
"$(rustup which cargo)" build --locked -p lana-wasm --target wasm32-unknown-unknown

mkdir -p "$OUT_DIR"
"$WASM_BINDGEN" --target nodejs --out-dir "$OUT_DIR" \
    "$REPO_ROOT/target/wasm32-unknown-unknown/debug/lana_wasm.wasm"

LANA_WASM_JS="$OUT_DIR/lana_wasm.js" node "$REPO_ROOT/tools/rust/lana-wasm/tests/conformance.mjs"
