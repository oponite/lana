#!/usr/bin/env bash
set -euo pipefail
root="$(mktemp -d "${TMPDIR:-/tmp}/lana-trees.XXXXXX")"
trap 'rm -rf "$root"' EXIT
"$1" run "$2" -- "$root/store" save | grep -q TREE_SAVED
"$1" run "$2" -- "$root/store" read | grep -q TREE_READ
"$1" run "$2" -- "$root/store" second | grep -q TREE_SECOND
if "$1" run "$2" -- "$root/store" tamper > "$root/tamper.out" 2>&1; then
    echo "tampered tree report was saved" >&2
    exit 1
fi
grep -q LANA_ERR_SCHEMA "$root/tamper.out"
"$1" run "$2" -- "$root/store" read | grep -q TREE_READ
echo TREES_STORE_PASS
