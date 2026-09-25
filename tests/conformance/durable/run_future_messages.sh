#!/usr/bin/env bash
set -euo pipefail

cli="$1"
root="$2"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
export LANA_STDLIB_DIR="$root/stdlib"
export LANA_FUTURE_STORE="$work/store"

for phase in save check settle verify; do
    export LANA_FUTURE_PHASE="$phase"
    "$cli" run "$root/tests/conformance/durable/future_messages.lana"
done
echo "FUTURE_MESSAGES_PASS"
