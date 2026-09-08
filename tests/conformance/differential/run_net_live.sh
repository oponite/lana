#!/usr/bin/env bash
set -euo pipefail
exec python3 -E "$(dirname "$0")/../../test_net_live.py" "$@"
