#!/bin/sh
set -eu

example_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
lana_bin=${LANA_BIN:-lana}
brain_path=${1:-"$example_dir/brain.lbrn"}
export LANA_HF=${LANA_HF:-"$example_dir/../../tools/lana-hf/lana_hf.py"}

"$lana_bin" brain new "$brain_path" 3 2 2 7
"$lana_bin" brain train "$brain_path" 2 0.1 0 1
"$lana_bin" brain evaluate "$brain_path" 0 1
python3 "$LANA_HF" package "$brain_path" "${brain_path}.hf" "$example_dir/tokenizer.json"
python3 "$LANA_HF" unpackage "${brain_path}.hf" "$brain_path"
"$lana_bin" brain chat "$brain_path" "$example_dir/tokenizer.json" "hello lana"
"$lana_bin" brain inspect "$brain_path"
