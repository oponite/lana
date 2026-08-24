# Lana 1.0 Release Checklist

A release is ready only when all four gates pass from the exact candidate tree.
Record command output; do not substitute earlier results.

## 1. Build and self-hosting

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build --output-on-failure
build/lana version
git diff --check
```

Required result: all tests pass, including the twice-repeated byte-stable native
compiler bootstrap, generated-project workflow, imports, LSP, and debugger.
`lana version` must report `1.0.0` and LABC v1.

## 2. Memory and concurrency safety

```bash
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug \
  -DLANA_ENABLE_SANITIZERS=ON
cmake --build build-asan --parallel
ctest --test-dir build-asan --output-on-failure

cmake -S . -B build-tsan -DCMAKE_BUILD_TYPE=Debug \
  -DLANA_ENABLE_TSAN=ON
cmake --build build-tsan --parallel
ctest --test-dir build-tsan --output-on-failure
```

Required result: both complete suites pass with no sanitizer report.

## 3. Malformed-bytecode resilience

```bash
cmake -S . -B build-fuzz -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER="$(brew --prefix llvm)/bin/clang" \
  -DLANA_ENABLE_SANITIZERS=ON -DLANA_BUILD_FUZZERS=ON
cmake --build build-fuzz --target lana_bytecode_fuzz --parallel
build-fuzz/lana_bytecode_fuzz -max_total_time=600 -timeout=5
```

Required result: ten minutes complete without a crash, leak, timeout, failed
assertion, or sanitizer report. Preserve any crashing input as a regression.
On macOS, use a Clang installation that includes the libFuzzer runtime; current
Xcode command-line tools may omit it.

## 4. Universal clean install

```bash
cmake -S . -B build-universal -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES='arm64;x86_64'
cmake --build build-universal --parallel
prefix="$(mktemp -d /tmp/lana-install.XXXXXX)"
cmake --install build-universal --prefix "$prefix"
lipo "$prefix/bin/lana" -verify_arch arm64 x86_64
lipo "$prefix/bin/lanavm" -verify_arch arm64 x86_64
arch -arm64 "$prefix/bin/lana" version
arch -x86_64 "$prefix/bin/lana" version
"$prefix/bin/lana" run examples/belief.lana
```

Required result: both slices execute, the installed CLI finds its adjacent
`lana-compiler.labc`, and a source program runs without using the build tree.
Code signing and package-manager publication are distribution steps, not claims
made by the source release.
