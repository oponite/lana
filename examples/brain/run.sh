#!/bin/sh
set -eu

example_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$example_dir/workshop.py" "$@"
