#!/usr/bin/env bash
set -euo pipefail
root="$(mktemp -d "${TMPDIR:-/tmp}/lana-rules.XXXXXX")"
trap 'rm -rf "$root"' EXIT
"$1" run "$2" -- "$root/store" save | grep -q RULES_STORE_SAVED
"$1" run "$2" -- "$root/store" read | grep -q RULES_STORE_READ
"$1" run "$2" -- "$root/store" correct | grep -q RULES_STORE_CORRECTED
if "$1" run "$2" -- "$root/store" changed_retry > "$root/conflict.out" 2>&1; then
    echo "changed correction retry was accepted" >&2
    exit 1
fi
grep -q LANA_ERR_CONFLICT "$root/conflict.out"
"$1" run "$2" -- "$root/store" inactive | grep -q RULES_STORE_INACTIVE
"$1" run "$2" -- "$root/store" rollback | grep -q RULES_STORE_ROLLED_BACK
if "$1" run "$2" -- "$root/store" tamper > "$root/tamper.out" 2>&1; then
    echo "tampered rule report was saved" >&2
    exit 1
fi
grep -q LANA_ERR_SCHEMA "$root/tamper.out"
"$1" run "$2" -- "$root/store" read | grep -q RULES_STORE_READ
echo RULES_STORE_PASS
