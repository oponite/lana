#!/usr/bin/env bash
set -euo pipefail
lana="$1"
source="$2"
first="$(LANA_STDLIB_DIR="$(dirname "$source")/../../stdlib" "$lana" run "$source")"
second="$(LANA_STDLIB_DIR="$(dirname "$source")/../../stdlib" "$lana" run "$source")"
test "$first" = "$second"
printf '%s\n' "$first"
