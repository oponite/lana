#!/usr/bin/env bash
# End-to-end durable-pipeline host-call check (Rust-only).
#
# Compiles `durable_pipeline.lana` with the self-hosted compiler, assembles the
# result with the Rust assembler, and runs it on the Rust VM. The store, policy,
# and ledger host calls are Rust-only (the C11 VM is frozen at 52 host calls),
# so this is not a differential check — it asserts the Rust pipeline succeeds.
#
#   cargo build -p lana-cli
#   ./tests/durable/run_durable.sh
#
# The compiler is expected at build/lana-compiler.labc relative to the repo
# root (re-assembled from compiler/bootstrap/compiler.lasm by the C11 build).

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
COMPILER="${COMPILER:-$REPO_ROOT/build/lana-compiler.labc}"
RUST="${RUST:-$REPO_ROOT/target/debug/lana-cli}"
FIXTURE="$REPO_ROOT/tests/conformance/durable/durable_pipeline.lana"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK" /tmp/lana_durable_pipeline' EXIT

if [[ ! -f "$COMPILER" ]]; then
    echo "compiler not found at $COMPILER (build it first)" >&2
    exit 1
fi
if [[ ! -x "$RUST" ]]; then
    echo "Rust lana-cli not found at $RUST (cargo build -p lana-cli first)" >&2
    exit 1
fi

# Compile the fixture with the self-hosted compiler (run on the Rust VM).
if ! "$RUST" run "$COMPILER" --memory-limit-mib 256 --instruction-limit 50000000 \
    -- "$FIXTURE" "$WORK/durable.lasm" >"$WORK/compile.out" 2>&1; then
    echo "FAIL: compilation failed"
    cat "$WORK/compile.out"
    exit 1
fi

# Assemble with the Rust assembler (the C11 assembler does not know the
# store/policy/ledger host-call names).
if ! "$RUST" asm "$WORK/durable.lasm" -o "$WORK/durable.labc" >"$WORK/asm.out" 2>&1; then
    echo "FAIL: assembly failed"
    cat "$WORK/asm.out"
    exit 1
fi

# Run on the Rust VM. The fixture asserts every durable-pipeline host call.
if ! "$RUST" run "$WORK/durable.labc" >"$WORK/run.out" 2>&1; then
    echo "FAIL: run failed"
    cat "$WORK/run.out"
    exit 1
fi

echo "ok   durable_pipeline"
