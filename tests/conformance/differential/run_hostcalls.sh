#!/usr/bin/env bash
# Differential conformance spot-checks for the host-call layer (Phase 3).
#
# For each fixture in hostcalls/, assemble with the C11 assembler, run under
# both the C11 `lanavm` and the Rust `lana-cli`, and assert byte-identical
# stdout, stderr, and exit codes. A second pass runs both VMs with `--stats`
# and asserts identical instruction/state-transition/opcode accounting (the
# allocation and elapsed-ns fields are excluded: the Rust VM uses native
# ownership rather than the C11 mark-sweep GC, so byte accounting differs by
# design). The Rust CLI must be built first:
#
#   cargo build -p lana-cli
#   ./tests/differential/run_hostcalls.sh
#
# The C11 binary is expected at build/lanavm relative to the repo root.
# Each fixture runs in its own scratch directory so file-system host calls
# (write_text, directory_list, ...) are isolated and cleaned up.

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
C11="${C11:-$REPO_ROOT/build/lanavm}"
RUST="${RUST:-$REPO_ROOT/target/debug/lana-cli}"
FIXTURES="$REPO_ROOT/tests/conformance/differential/hostcalls"
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

# Strip the fields that legitimately differ between the two memory models:
# allocation count/bytes (GC vs native ownership) and elapsed wall-clock.
normalize_stats() {
    sed -E 's/"allocations":[0-9]+,//; s/"allocated_bytes":[0-9]+,//; s/"elapsed_ns":[0-9]+,//'
}

failures=0
count=0
for fixture in "$FIXTURES"/*.lasm; do
    name="$(basename "$fixture" .lasm)"
    count=$((count + 1))

    # Assemble with the C11 assembler so both VMs run identical bytecode.
    if ! "$C11" asm "$fixture" -o "$WORK/$name.labc" >"$WORK/$name.asm.out" 2>&1; then
        echo "FAIL $name: assembly failed"
        cat "$WORK/$name.asm.out"
        failures=$((failures + 1))
        continue
    fi

    # Optional per-fixture extra CLI args (e.g. --seed 7).
    args=()
    if [[ -f "$FIXTURES/$name.args" ]]; then
        read -r -a args < "$FIXTURES/$name.args"
    fi

    # Run each VM in its own scratch dir so file-system host calls are isolated.
    rundir="$WORK/run-$name"
    mkdir -p "$rundir"
    ( cd "$rundir" && "$C11" run "$WORK/$name.labc" "${args[@]}" >"$WORK/$name.c11.out" 2>"$WORK/$name.c11.err" )
    c11_exit=$?
    ( cd "$rundir" && "$RUST" run "$WORK/$name.labc" "${args[@]}" >"$WORK/$name.rust.out" 2>"$WORK/$name.rust.err" )
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

    # Resource accounting: instruction count, state transitions, and opcode
    # counts must match. Allocation bytes/count and elapsed time are excluded.
    ( cd "$rundir" && "$C11" run "$WORK/$name.labc" "${args[@]}" --stats >"$WORK/$name.c11.stats.out" 2>"$WORK/$name.c11.stats.err" )
    ( cd "$rundir" && "$RUST" run "$WORK/$name.labc" "${args[@]}" --stats >"$WORK/$name.rust.stats.out" 2>"$WORK/$name.rust.stats.err" )
    grep '^LANAVM_STATS ' "$WORK/$name.c11.stats.err" | normalize_stats >"$WORK/$name.c11.stats"
    grep '^LANAVM_STATS ' "$WORK/$name.rust.stats.err" | normalize_stats >"$WORK/$name.rust.stats"
    if ! cmp -s "$WORK/$name.c11.stats" "$WORK/$name.rust.stats"; then
        echo "FAIL $name: stats differ"
        diff "$WORK/$name.c11.stats" "$WORK/$name.rust.stats" | head -20
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
