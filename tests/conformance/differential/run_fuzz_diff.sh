#!/usr/bin/env bash
# Differential fuzz spot-check for the loader/verifier boundary (Phase 3).
#
# Runs the same arbitrary bytes through the Rust `lana-bytecode` loader +
# verifier and the C11 `lanavm verify` subcommand, and asserts identical
# accept/reject plus identical `LANA_ERR_*` error codes. The Rust driver must
# be built first:
#
#   cargo build -p lana-fuzz --bin fuzz-diff
#   ./tests/differential/run_fuzz_diff.sh
#
# The C11 binary is expected at build/lanavm relative to the repo root.
#
# A valid chunk with a trailing byte appended is included in the conformance
# set: both loaders reject trailing bytes (`fgetc(file) != EOF` in
# vm/c/bytecode.c, mirrored by the Rust loader's trailing-byte check).

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
C11="${C11:-$REPO_ROOT/build/lanavm}"
DRIVER="${DRIVER:-$REPO_ROOT/target/debug/fuzz-diff}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

if [[ ! -x "$C11" ]]; then
    echo "C11 lanavm not found at $C11 (build it first)" >&2
    exit 1
fi
if [[ ! -x "$DRIVER" ]]; then
    echo "Rust fuzz-diff not found at $DRIVER (cargo build -p lana-fuzz --bin fuzz-diff first)" >&2
    exit 1
fi

# --- Conformance inputs: both implementations must agree. --------------------
# Valid chunks (assembled by the C11 toolchain) plus malformed byte strings.
valid=("$REPO_ROOT/build/lana-compiler.labc" "$REPO_ROOT/build/compiler-selfcheck.labc")
for f in "${valid[@]}"; do
    [[ -f "$f" ]] || { echo "missing valid fixture $f" >&2; exit 1; }
done

printf '' > "$WORK/empty.labc"
printf 'not labc at all' > "$WORK/notlabc.labc"
printf 'LABC\x02\x00\x00\x00\xff\xff\xff\xff' > "$WORK/counts-overflow.labc"
printf 'LABC\x63\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00' > "$WORK/bad-version.labc"
printf 'LABC\x02\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00' > "$WORK/truncated-constant.labc"

# A valid chunk with a trailing byte appended: both loaders reject it.
cp "$REPO_ROOT/build/lana-compiler.labc" "$WORK/trailing.labc"
printf '\x00' >> "$WORK/trailing.labc"

conformance=("${valid[@]}" "$WORK/empty.labc" "$WORK/notlabc.labc" \
             "$WORK/counts-overflow.labc" "$WORK/bad-version.labc" \
             "$WORK/truncated-constant.labc" "$WORK/trailing.labc")

echo "== conformance (must match) =="
"$DRIVER" --lanavm "$C11" "${conformance[@]}"
conformance_ok=$?

echo
if [[ $conformance_ok -eq 0 ]]; then
    echo "all conformance inputs match"
    exit 0
fi
echo "FAIL: conformance inputs diverged" >&2
exit 1
