# Working with Lana

This file is the fastest entrypoint for a person or an AI coding tool. Start a
session with:

> Read AGENTS.md, explain Lana's authority order, then help me build my first
> program. Ask before changing repository files.

## First program

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
build/lana new hello-lana
cd hello-lana
../build/lana run .
../build/lana test .
```

`lana new` creates a small module in `src/belief.lana`, imports it from
`src/main.lana`, and adds a test. Read those three files together: they show
state construction, module imports, measurement, and assertions.

## Authority order

Resolve disagreements in this order:

1. `papers/semantics.md` — mathematical meaning.
2. `spec/SPEC.md` — source syntax and programmer-visible behavior.
3. `spec/BYTECODE.md` — the one LABC v1 encoding.
4. `spec/VM.md` — runtime architecture and resource behavior.

New or changed source syntax must additionally satisfy `spec/SYNTAX.md` — the
syntax design principles (SYNTAX-1..12 + Acceptance Principle).

Do not invent semantics from an implementation detail. Change the highest
applicable authority first when intentionally changing the language.

## Repository map

- `vm/`: canonical Rust `lana-vm` + `lana-bytecode` crates, plus the frozen C11
  reference VM core (`vm/c/`, `vm/include/`).
- `runtime/`: canonical Rust `lana-runtime` + `lana-ffi` crates, plus the C11
  hardware boundary (`runtime/c/`, `runtime/include/`).
- `tools/`: Rust `lana-cli` + `lana-fuzz` crates, plus the C11 CLI, LSP, and
  project tooling (`tools/c/`, `tools/include/`).
- `compiler/*.lana`: self-hosted compiler source.
- `compiler/bootstrap/compiler.lasm`: checked, reproducible bootstrap artifact.
- `examples/`: runnable Lana and LABC examples.
- `tests/unit/`: runtime and public-API tests.
- `tests/regression/`: source-language pass/fail fixtures.
- `tests/conformance/`: differential conformance harness (C11 vs Rust).

## Daily commands

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build --output-on-failure
build/lana run examples/general.lana
build/lana check examples/belief.lana
git diff --check
```

Project workflow:

```bash
build/lana new my-program
build/lana build my-program
build/lana check my-program
build/lana test my-program
build/lana run my-program
```

Low-level bytecode workflow:

```bash
build/lanavm asm examples/belief.lasm -o build/belief.labc
build/lanavm verify build/belief.labc
build/lanavm dis build/belief.labc
build/lanavm run build/belief.labc --trace
```

## Release gates

A release is ready only when all five gates pass from the exact candidate tree.
Record command output; do not substitute earlier results.

### 1. Build and self-hosting

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build --output-on-failure
build/lana version
git diff --check
```

Required result: all tests pass, including the twice-repeated byte-stable native
compiler bootstrap, generated-project workflow, imports, LSP, and debugger.
`lana version` must report the version in `VERSION` and LABC v2.

### 2. Memory and concurrency safety

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

### 3. Malformed-bytecode resilience

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

### 4. Universal clean install

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

### 5. Optional integrations and release artifacts

```bash
python3 -m venv /tmp/lana-integrations-venv
/tmp/lana-integrations-venv/bin/python -m pip install -e 'integrations/python[test]'
/tmp/lana-integrations-venv/bin/python -m pytest -q integrations/python/tests

cmake -S . -B build-integrations -DCMAKE_BUILD_TYPE=Release \
  -DLANA_BUILD_INTEGRATIONS=ON
cmake --build build-integrations --parallel
ctest --test-dir build-integrations --output-on-failure
```

Required result: the Python bridge accepts Lana 1.x with LABC v2, the native
bridge reports ABI v1, and all integration tests pass.

The release workflow downloads the macOS archive into a clean directory, checks
its SHA-256 digest, extracts it, and runs both installed architecture slices
against a copied example. It also checks the source archive digest, builds it in
a clean directory, and runs the example before publication. Homebrew Core
submission is an external publication step; the release workflow publishes a
checksum-backed formula artifact. Signing and notarization remain deferred.

The compiler emits LABC v2; the dual-version loader accepts v1 and v2.
Pre-release bytecode and textual assembly are not accepted or converted; rebuild
them from source.

## Change rules

- Preserve unrelated dirty work and inspect a file before writing it.
- C is C11 with four-space indentation and `-Wall -Wextra -Wpedantic -Werror`.
- C names use `snake_case`; public types use `Lana...`; public functions use
  `lana_...`; constants use `LANA_...`.
- Lana uses lowercase `snake_case`, semicolons, and small explicit functions.
- Add C regressions in `tests/unit/`; add language fixtures in
  `tests/regression/` and register them in `cmake/CTestTargets.cmake`.
- Never weaken compiler limits: 256 MiB and 50,000,000 instructions. Exhaustion
  is an error and must not expose partial bytecode.
- Benchmark result snapshots are machine-local evidence, not conformance.
- Run `git diff --check`; there is no repository-wide formatter.

## Development policy

Language, compiler, bytecode, and VM development is active. Preserve the
authority order, compatibility expectations, correctness, security, and data
integrity. Add regression coverage for behavior changes.
