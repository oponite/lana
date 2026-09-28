#!/usr/bin/env bash
set -euo pipefail
root="$(mktemp -d "${TMPDIR:-/tmp}/lana-dataset-apply.XXXXXX")"
trap 'rm -rf "$root"' EXIT
"$1" run "$2" -- "$root/store"
