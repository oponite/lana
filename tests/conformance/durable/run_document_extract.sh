#!/usr/bin/env bash
set -euo pipefail
root="$(mktemp -d "${TMPDIR:-/tmp}/lana-document.XXXXXX")"
trap 'rm -rf "$root"' EXIT
printf '# Top\r\nalpha 🐈\r\n\n```txt\nx\n```\n' > "$root/document.md"
"$1" run "$2" -- "$root/document.md"
