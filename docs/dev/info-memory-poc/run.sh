#!/bin/sh
# Reproduce every result in this directory (see README.md). Memory figures are
# deterministic for a given seed; timings are machine-local evidence.
set -eu
cd "$(dirname "$0")"
root=../../..
cargo build --release --locked --manifest-path "$root/Cargo.toml" -p lana-runtime --example info_memory_bench
bench="$root/target/release/examples/info_memory_bench"
"$bench" --objects 1024 --len 4096 --repeats 3 --out main.json > main.md
for update in 100 500 900; do
    "$bench" --objects 256 --len 4096 --repeats 3 --update-per-mille "$update" \
        --out "update-$update.json" > "update-$update.md"
done
"$bench" --objects 16384 --len 64 --repeats 3 --out small-objects.json > small-objects.md
"$bench" --objects 256 --len 4096 --repeats 3 --seed 7 --out seed-7.json > seed-7.md
