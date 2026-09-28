#!/usr/bin/env sh
set -eu
# Usage: package.sh [INSTALLED_PREFIX] [ARCHIVE]
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
version=$(cat "$root/VERSION")
exec python3 "$root/tools/build.py" package \
    --from "${1:-$root/target/universal}" \
    --output "${2:-$root/dist/lana-$version-macos.tar.gz}"
