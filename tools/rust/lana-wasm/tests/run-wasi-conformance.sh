#!/usr/bin/env bash
# LIP-025 WASI conformance: build the `wasm32-wasip1` command entry and run the
# same class of assertions as the node conformance (compile -> gate -> run)
# under the `wasmtime` WASI runtime.
#
# Prerequisites (one-time):
#   rustup target add wasm32-wasip1
#   brew install wasmtime   (or cargo install wasmtime-cli)
#   cmake --build build     # produces build/lana-compiler.labc
#
# Missing prerequisites exit 77. CTest records a skip, not a passing test.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"

if ! command -v wasmtime >/dev/null 2>&1; then
    echo "SKIP: wasmtime not installed (brew install wasmtime); WASI conformance not run" >&2
    exit 77
fi

if [[ -z "${LANA_COMPILER_LABC:-}" ]]; then
    export LANA_COMPILER_LABC="$REPO_ROOT/build/lana-compiler.labc"
fi
if [[ ! -f "$LANA_COMPILER_LABC" ]]; then
    echo "lana-compiler.labc not found at $LANA_COMPILER_LABC (cmake --build build first)" >&2
    exit 1
fi

cargo build -p lana-wasm-wasi --target wasm32-wasip1 >/dev/null
WASM="$REPO_ROOT/target/wasm32-wasip1/debug/lana-wasm-wasi.wasm"

# run_prog SOURCE [INPUT] [CAPABILITIES]: print the program's JSON result.
# `wasmtime` receives the source as a single argument; newlines survive argv.
# The WASI command exits 1 when the program reports ok:false, but the JSON is
# always on stdout, so swallow the exit code here (set -e elsewhere).
run_prog() {
    set +e
    wasmtime "$WASM" "$1" "${2:-}" "${3:-}"
    set -e
}

assert_ok() { # SOURCE INPUT CAPABILITIES expected_result
    local out
    out="$(run_prog "$1" "$2" "$3")"
    if [[ "$out" != "$4" ]]; then
        echo "FAIL: wasi: expected '$4', got '$out'" >&2
        return 1
    fi
    echo "ok  wasi: $5"
    WASI_OK=$((WASI_OK + 1))
}

assert_fail() { # SOURCE INPUT CAPABILITIES message_substring
    local out
    out="$(run_prog "$1" "$2" "$3")"
    if [[ "$out" != '{"ok":false'* ]]; then
        echo "FAIL: wasi: expected ok:false, got '$out'" >&2
        return 1
    fi
    if [[ -n "$4" ]] && [[ "$out" != *"$4"* ]]; then
        echo "FAIL: wasi: expected message to contain '$4', got '$out'" >&2
        return 1
    fi
    echo "ok  wasi: $5"
    WASI_OK=$((WASI_OK + 1))
}

WASI_OK=0
pass() { echo "ok  wasi: $1"; WASI_OK=$((WASI_OK + 1)); }

# Byte-identical scalar result (LIP-025 §1: same source, same result as native).
out="$(run_prog 'return 42;
' '' '')"
[[ "$out" == '{"ok":true,"result":"42"}' ]] || { echo "FAIL: ret42 got $out" >&2; exit 1; }
pass "return 42 byte-identical"

# Input passed as the single program argument.
out="$(run_prog 'let a = args();
return a[0];
' 'hello' '')"
[[ "$out" == '{"ok":true,"result":"hello"}' ]] || { echo "FAIL: args got $out" >&2; exit 1; }
pass "input as single argument"

# Host-call gating: an FS call is unavailable by default...
out="$(run_prog 'let e = directory_list("/tmp");
return e;
' '' '')"
[[ "$out" == *'LANA_ERR_UNSUPPORTED_OPERATION'* ]] || { echo "FAIL: dir gated got $out" >&2; exit 1; }
pass "directory_list gated by default"

# ...and proceeds to the host when the capability is wired. In a sandboxed
# wasmtime without a mounted directory the host call fails with a network/IO
# error — NOT with the gate error — which is the LIP-025 property.
out="$(run_prog 'let e = directory_list("/tmp");
return e;
' '' '{"directory_list":true}')"
[[ "$out" != *'LANA_ERR_UNSUPPORTED_OPERATION'* ]] || { echo "FAIL: dir wired gate not lifted: $out" >&2; exit 1; }
pass "directory_list gate lifted when wired"

# Network host-call gating (LIP-025 §3): a precompiled HOST_CALL http_get or
# socket_connect chunk is unavailable by default. The compiler at this fork
# point cannot yet emit network host calls from source, so the conformance
# feeds them as hex-encoded LABC blobs built by hand.
HTTP_GET=4c41424302000000010000000000000001000000000000000100000000000000002708000000a0000000000000000000000001000000
SOCKET_CONNECT=4c41424302000000010000000000000001000000000000000100000000000000002708000000a2000000000000000000000001000000

out="$(run_prog "HEXBLOB:$HTTP_GET" '' '')"
[[ "$out" == *'LANA_ERR_UNSUPPORTED_OPERATION'* ]] && [[ "$out" == *'http_get'* ]] \
    || { echo "FAIL: wasi: http_get not gated: $out" >&2; exit 1; }
pass "http_get gated by default"
out="$(run_prog "HEXBLOB:$HTTP_GET" '' '{"http_get":true}')"
[[ "$out" != *'LANA_ERR_UNSUPPORTED_OPERATION'* ]] \
    || { echo "FAIL: wasi: http_get gate not lifted when wired: $out" >&2; exit 1; }
pass "http_get gate lifted when wired"
out="$(run_prog "HEXBLOB:$SOCKET_CONNECT" '' '')"
[[ "$out" == *'LANA_ERR_UNSUPPORTED_OPERATION'* ]] && [[ "$out" == *'socket_connect'* ]] \
    || { echo "FAIL: wasi: socket_connect not gated: $out" >&2; exit 1; }
pass "socket_connect gated by default"
# Determinism: same source + seed -> same result (LIP-025 §4).
prog='state a = state(p: 0.5, d: 0.0);
state b = state(p: 0.3, d: 0.0);
let c = append(a, b);
let s = sample(c);
return s;
'
first="$(run_prog "$prog" '' '{"seed":42}')"
second="$(run_prog "$prog" '' '{"seed":42}')"
[[ "$first" == "$second" ]] || { echo "FAIL: determinism $first vs $second" >&2; exit 1; }
[[ "$first" == '{"ok":true'* ]] || { echo "FAIL: determinism not ok: $first" >&2; exit 1; }
pass "deterministic under a fixed seed"

# Resource-limit enforcement: exceeding the instruction budget fails (LIP-025 §4).
out="$(run_prog 'let i = 0;
while (i < 1000000) { i = i + 1; }
return i;
' '' '{"instruction_limit":100}')"
[[ "$out" == '{"ok":false,"error":'* ]] || { echo "FAIL: instruction limit got $out" >&2; exit 1; }
pass "instruction-limit enforcement"

echo "wasi conformance: $WASI_OK assertions passed"
