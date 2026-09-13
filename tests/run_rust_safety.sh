#!/usr/bin/env bash
set -euo pipefail

# Usage: bash tests/run_rust_safety.sh asan|tsan|fuzz OUTPUT_DIR [CLANG]
# Requires rustup nightly and rust-src. Never disables leak detection.
mode="$1"
output="$2"
mkdir -p "$output"
output="$(cd "$output" && pwd)"
export RUSTC="$(rustup which --toolchain nightly rustc)"
export RUSTDOC="$(rustup which --toolchain nightly rustdoc)"
cargo_bin="$(rustup which --toolchain nightly cargo)"
target="$($RUSTC -vV | sed -n 's/^host: //p')"
export CARGO_TARGET_DIR="$output/target"
case "$mode" in
    asan|tsan)
        sanitizer=address
        extra=()
        test_args=()
        if [[ "$mode" == tsan ]]; then
            sanitizer=thread
            extra=(-Zbuild-std)
        elif [[ "$target" == *apple-darwin ]]; then
            # Apple framework lifetime is checked separately with the native
            # detector; LeakSanitizer remains enabled for every Lana-only test.
            test_args=(-- --skip resident_buffers_survive_pool_drain_and_repeated_commands \
                --skip gpu_matmul_2d_within_float32_tolerance)
        fi
        export RUSTFLAGS="-Zsanitizer=$sanitizer"
        export RUSTDOCFLAGS="$RUSTFLAGS"
        export ASAN_OPTIONS=detect_leaks=1
        export TSAN_OPTIONS=halt_on_error=1
        "$cargo_bin" test --locked --target "$target" "${extra[@]}" \
            -p lana-vm -p lana-runtime -p lana-ffi -p lana-cli "${test_args[@]}"
        if [[ "$mode" == asan && "$target" == *apple-darwin ]]; then
            bash "$(dirname "$0")/run_macos_metal_leaks.sh" "$output/metal-leaks"
        fi
        ;;
    fuzz)
        compiler="${3:-clang}"
        export RUSTFLAGS='-Zsanitizer=address -Cpasses=sancov-module -Cllvm-args=-sanitizer-coverage-level=4 -Cllvm-args=-sanitizer-coverage-inline-8bit-counters -Cllvm-args=-sanitizer-coverage-pc-table -Cllvm-args=-sanitizer-coverage-trace-compares'
        "$cargo_bin" build --locked --target "$target" -p lana-fuzz --lib
        if [[ "$target" == *apple-darwin ]]; then
            python3 "$(dirname "$0")/build_fuzzer_runtime.py" --cc "$compiler" --output "$output/runtime"
            "$compiler" -fsanitize=address "$output/runtime/libFuzzer.a" \
                "$CARGO_TARGET_DIR/$target/debug/liblana_fuzz.a" \
                -lc++ -lpthread -ldl -lm -o "$output/lana-rust-fuzz"
        else
            "$compiler" -fsanitize=fuzzer,address "$CARGO_TARGET_DIR/$target/debug/liblana_fuzz.a" \
                -lpthread -ldl -lm -o "$output/lana-rust-fuzz"
        fi
        mkdir -p "$output/corpus"
        export ASAN_OPTIONS=detect_leaks=1
        "$output/lana-rust-fuzz" -seed=42 -max_total_time=600 -timeout=5 \
            "-artifact_prefix=$output/" "$output/corpus"
        ;;
    *) echo "Expected asan, tsan, or fuzz" >&2; exit 2 ;;
esac
