#!/bin/zsh
set -euo pipefail

# Usage: zsh package.sh [BUILD_DIR] [OUTPUT_DIR]
# Package the installed prefix, including the Rust CLI and FFI library.
build_dir="${1:-build-release}"
version="$(tr -d '\n' < VERSION)"
output_dir="${2:-build/release-evidence/${version}/packages}"
if [[ ! -f "$build_dir/CMakeCache.txt" ]]; then
    echo "FAILED: configure and build '$build_dir' first" >&2
    exit 1
fi
mkdir -p "$output_dir"
prefix="$(mktemp -d "$output_dir/install.XXXXXX")"
cmake --install "$build_dir" --prefix "$prefix"
python3 -E tests/verify_install.py --prefix "$prefix"
archive="$output_dir/lana-${version}-$(uname -s | tr '[:upper:]' '[:lower:]').tar.gz"
if [[ -e "$archive" ]]; then
    echo "FAILED: archive already exists: $archive" >&2
    exit 1
fi
tar -czf "$archive" -C "$prefix" .
shasum -a 256 "$archive"
echo "Package created: $archive"
