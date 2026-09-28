#!/usr/bin/env bash
# End-to-end durable-pipeline host-call check (Rust-only).
#
# Compiles `durable_pipeline.lana` with the self-hosted compiler, assembles the
# result with the Rust assembler, and runs it on the Rust VM. The store, policy,
# and ledger host calls are checked through the Rust pipeline.
#
#   python3 tools/build.py build
#   ./tests/conformance/durable/run_durable.sh
#
# The compiler is expected at target/lana/bin/lana-compiler.labc relative to the repo
# root (assembled from compiler/bootstrap/compiler.lasm by the Rust build).

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
COMPILER="${COMPILER:-$REPO_ROOT/target/lana/bin/lana-compiler.labc}"
RUST="${RUST:-$REPO_ROOT/target/lana/bin/lana}"
FIXTURE="$REPO_ROOT/tests/conformance/durable/durable_pipeline.lana"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK" /tmp/lana_durable_pipeline' EXIT

if [[ ! -f "$COMPILER" ]]; then
    echo "compiler not found at $COMPILER (build it first)" >&2
    exit 1
fi
if [[ ! -x "$RUST" ]]; then
    echo "Rust lana-cli not found at $RUST (python3 tools/build.py build first)" >&2
    exit 1
fi

# Compile the fixture with the self-hosted compiler (run on the Rust VM).
if ! "$RUST" run "$COMPILER" --memory-limit-mib 256 --instruction-limit 50000000 \
    -- "$FIXTURE" "$WORK/durable.lasm" >"$WORK/compile.out" 2>&1; then
    echo "FAIL: compilation failed"
    cat "$WORK/compile.out"
    exit 1
fi

# Assemble with the Rust assembler.
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
