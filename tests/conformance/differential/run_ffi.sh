#!/usr/bin/env bash
# Differential conformance spot-checks for the LIP-018 two-way FFI
# (ffi_declare, ffi_load, ffi_call).
#
# The fixtures exercise the deterministic paths that need no real shared
# library: signature parsing, capability denial, and argument type rejection.
# A real dlopen call is C-only (the Rust libloading path differs in error
# strings), so it is covered by the C unit test instead.
#
#   cargo build -p lana-cli
#   ./tests/differential/run_ffi.sh
#
# The C11 binary is expected at build/lanavm relative to the repo root.

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
C11="${C11:-$REPO_ROOT/build/lanavm}"
RUST="${RUST:-$REPO_ROOT/target/debug/lana-cli}"
FIXTURES="$REPO_ROOT/tests/conformance/differential/ffi"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

if [[ ! -x "$C11" ]]; then
    echo "C11 lanavm not found at $C11 (build it first)" >&2
    exit 1
fi
if [[ ! -x "$RUST" ]]; then
    echo "Rust lana-cli not found at $RUST (cargo build -p lana-cli first)" >&2
    exit 1
fi

failures=0
count=0
for fixture in "$FIXTURES"/*.lasm; do
    name="$(basename "$fixture" .lasm)"
    count=$((count + 1))

    if ! "$C11" asm "$fixture" -o "$WORK/$name.labc" >"$WORK/$name.asm.out" 2>&1; then
        echo "FAIL $name: assembly failed"
        cat "$WORK/$name.asm.out"
        failures=$((failures + 1))
        continue
    fi

    "$C11" run "$WORK/$name.labc" >"$WORK/$name.c11.out" 2>"$WORK/$name.c11.err"
    c11_exit=$?
    "$RUST" run "$WORK/$name.labc" >"$WORK/$name.rust.out" 2>"$WORK/$name.rust.err"
    rust_exit=$?

    ok=1
    if [[ $c11_exit -ne $rust_exit ]]; then
        echo "FAIL $name: exit $c11_exit (C11) != $rust_exit (Rust)"
        ok=0
    fi
    if ! cmp -s "$WORK/$name.c11.out" "$WORK/$name.rust.out"; then
        echo "FAIL $name: stdout differs"
        diff "$WORK/$name.c11.out" "$WORK/$name.rust.out" | head -20
        ok=0
    fi
    if ! cmp -s "$WORK/$name.c11.err" "$WORK/$name.rust.err"; then
        echo "FAIL $name: stderr differs"
        diff "$WORK/$name.c11.err" "$WORK/$name.rust.err" | head -20
        ok=0
    fi

    if [[ $ok -eq 1 ]]; then
        echo "ok   $name"
    else
        failures=$((failures + 1))
    fi
done

echo
echo "$((count - failures))/$count fixtures match"
[[ $failures -eq 0 ]]
