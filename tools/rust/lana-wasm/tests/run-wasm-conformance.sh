#!/usr/bin/env bash
# Build the `lana-wasm` crate for wasm32-unknown-unknown, generate the nodejs
# bindings, and run the native-vs-WASM conformance assertions.
#
# Prerequisites (one-time):
#   rustup target add wasm32-unknown-unknown
#   cargo install wasm-bindgen-cli --version 0.2.100
#   cmake --build build          # produces build/lana-compiler.labc
#
# The wasm-bindgen CLI version must match the `wasm-bindgen` crate version
# pinned in tools/rust/lana-wasm/Cargo.toml.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"

WASM_BINDGEN="${WASM_BINDGEN:-wasm-bindgen}"
OUT_DIR="$REPO_ROOT/target/wasm-bindgen-nodejs"

if [[ -z "${LANA_COMPILER_LABC:-}" ]]; then
    export LANA_COMPILER_LABC="$REPO_ROOT/build/lana-compiler.labc"
fi
if [[ ! -f "$LANA_COMPILER_LABC" ]]; then
    echo "lana-compiler.labc not found at $LANA_COMPILER_LABC (cmake --build build first)" >&2
    exit 1
fi

cargo build -p lana-wasm --target wasm32-unknown-unknown

mkdir -p "$OUT_DIR"
"$WASM_BINDGEN" --target nodejs --out-dir "$OUT_DIR" \
    "$REPO_ROOT/target/wasm32-unknown-unknown/debug/lana_wasm.wasm"

LANA_WASM_JS="$OUT_DIR/lana_wasm.js" node "$REPO_ROOT/tools/rust/lana-wasm/tests/conformance.mjs"
