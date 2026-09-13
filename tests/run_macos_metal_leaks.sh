#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != Darwin ]]; then
    echo "macOS Metal leak check requires macOS" >&2
    exit 2
fi

output="${1:?usage: run_macos_metal_leaks.sh OUTPUT_DIR}"
root="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$output"
output="$(cd "$output" && pwd)"
binary="$output/metal-lifetime-probe"

xcrun clang -g -O0 -Wall -Wextra -Wpedantic -Werror \
    -framework Foundation -framework Metal \
    "$root/tests/metal_lifetime_probe.m" -o "$binary"

run_leaks() {
    local name="$1"
    shift
    set +e
    MallocStackLogging=1 /usr/bin/leaks --atExit -- "$binary" "$@" \
        >"$output/$name.log" 2>&1
    local status=$?
    set -e
    printf '%s\n' "$status" >"$output/$name.status"
}

run_leaks clean --clean-control
test "$(<"$output/clean.status")" -eq 0
grep -Eq '0 leaks for 0 total leaked bytes' "$output/clean.log"

run_leaks deliberate --leak-control
test "$(<"$output/deliberate.status")" -ne 0
grep -Eq '[1-9][0-9]* leaks for [1-9][0-9]* total leaked bytes' "$output/deliberate.log"

for run in 1 2 3; do
    run_leaks "metal-$run"
    test "$(<"$output/metal-$run.status")" -eq 0
    grep -Eq '0 leaks for 0 total leaked bytes' "$output/metal-$run.log"
done
