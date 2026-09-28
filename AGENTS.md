# Working with Lana

This file is the fastest entrypoint for a person or an AI coding tool. Start a
session with:

> Read AGENTS.md, explain Lana's authority order, then help me build my first
> program. Ask before changing repository files.

## First program

```bash
python3 tools/build.py build
target/lana/bin/lana new hello-lana
cd hello-lana
../target/lana/bin/lana run
../target/lana/bin/lana test
```

`lana new` creates a small module in `src/belief.lana`, imports it from
`src/main.lana`, and adds a test. Read those three files together: they show
state construction, module imports, measurement, and assertions.

## Authority order

Resolve disagreements in this order:

1. `papers/semantics.md` — mathematical meaning.
2. `spec/SPEC.md` — source syntax and programmer-visible behavior.
3. `spec/BYTECODE.md` — the one LABC v2 encoding.
4. `spec/VM.md` — runtime architecture and resource behavior.

New or changed source syntax must additionally satisfy `spec/SYNTAX.md` — the
syntax design principles (SYNTAX-1..12 + Acceptance Principle).

Do not invent semantics from an implementation detail. Change the highest
applicable authority first when intentionally changing the language.

## Repository map

- `vm/`: canonical Rust `lana-vm` and `lana-bytecode` crates.
- `runtime/`: canonical Rust `lana-runtime` crate.
- `tools/`: Rust `lana-cli`, `lana-fuzz`, and `lana-wasm` crates.
- `compiler/*.lana`: self-hosted compiler source.
- `compiler/bootstrap/compiler.lasm`: checked, reproducible bootstrap artifact.
- `examples/`: runnable Lana and LABC examples.
- `tests/regression/`: source-language pass/fail fixtures.
- `tests/conformance/`: published bytecode compatibility fixtures and behavior checks.

## Daily commands

```bash
python3 tools/build.py build
python3 tests/run.py --no-build
target/lana/bin/lana run examples/general.lana
git diff --check
```

Project workflow:

```bash
target/lana/bin/lana new my-program
cd my-program
../target/lana/bin/lana build
../target/lana/bin/lana test
../target/lana/bin/lana run
```

Low-level bytecode workflow:

```bash
target/lana/bin/lana asm examples/belief.lasm -o target/belief.labc
target/lana/bin/lana verify target/belief.labc
target/lana/bin/lana dis target/belief.labc
target/lana/bin/lana run target/belief.labc --trace
```

## Release gates

A release is ready only when all five gates pass from the exact candidate tree.
Record command output; do not substitute earlier results.

### 1. Build and self-hosting

```bash
python3 tools/build.py build
python3 tests/run.py --no-build
target/lana/bin/lana version
cargo test --locked --workspace --no-fail-fast
git diff --check
```

Required result: all tests pass, including the twice-repeated byte-stable native
compiler bootstrap, generated-project workflow, imports, LSP, and debugger.
`lana version` must report the version in `VERSION` and LABC v2.

### 2. Malformed-bytecode resilience

```bash
RUSTC="$(rustup which --toolchain nightly rustc)" rustup run nightly cargo fuzz run lana_bytecode --fuzz-dir fuzz -- -max_total_time=600 -timeout=5
```

Required result: ten minutes without a crash, leak, timeout, failed assertion,
or sanitizer report. Preserve crashing inputs as regressions.

### 3. Universal clean install

```bash
rustup target add aarch64-apple-darwin x86_64-apple-darwin
python3 tools/build.py universal
prefix="$(mktemp -d /tmp/lana-install.XXXXXX)"
python3 tools/build.py install --from target/universal --prefix "$prefix"
lipo "$prefix/bin/lana" -verify_arch arm64 x86_64
arch -arm64 "$prefix/bin/lana" version
arch -x86_64 "$prefix/bin/lana" version
"$prefix/bin/lana" run examples/belief.lana
```

Required result: both slices execute, the installed CLI finds its adjacent
`lana-compiler.labc`, and a source program runs without using the build tree.
Code signing and package-manager publication are distribution steps, not claims
made by the source release.

### 4. Optional integrations and release artifacts

```bash
python3 -m venv /tmp/lana-integrations-venv
/tmp/lana-integrations-venv/bin/python -m pip install -e 'integrations/python[test]' tokenizers safetensors numpy
/tmp/lana-integrations-venv/bin/python -m pytest -q integrations/python/tests
/tmp/lana-integrations-venv/bin/python -m unittest discover -s tools/lana-hf/tests -v

python3 tools/build.py build
python3 tests/run.py --no-build
```

Required result: the Python bridge accepts Lana 4.0 through the Rust CLI and
worker, and all integration tests pass.

The release workflow downloads the macOS archive into a clean directory, checks
its SHA-256 digest, extracts it, and runs both installed architecture slices
against a copied example. It also checks the source archive digest, builds it in
a clean directory, and runs the example before publication. Homebrew Core
submission is an external publication step; the release workflow publishes a
checksum-backed formula artifact. Signing and notarization remain deferred.

The compiler emits LABC v2 by default; the Rust loader accepts v1-v5, and
published v1-v2 bytecode is checked against frozen fixtures.
Pre-release bytecode and textual assembly are not accepted or converted; rebuild
them from source.

### 5. Performance

Compare Release builds on the same machine with the workloads and method in
`plans/rust-only-performance.md`. The 4.0 warm median must be no more than 5%
above the paired 3.0.2 median for repeated Python bytecode calls, repeated
Python source calls, Rust VM execution, and Rust compilation. Report Python
first-call time separately. Do not treat a historical snapshot from a different
load condition as an exact threshold.

## Change rules

- Preserve unrelated dirty work and inspect a file before writing it.
- Lana uses lowercase `snake_case`, semicolons, and small explicit functions.
- Add language fixtures in
  `tests/regression/` and register them in `tests/cases.json`.
- Never weaken compiler limits: 256 MiB and 50,000,000 instructions. Exhaustion
  is an error and must not expose partial bytecode.
- Benchmark result snapshots are machine-local evidence, not conformance.
- Run `git diff --check`; there is no repository-wide formatter.

## Development policy

Language, compiler, bytecode, and VM development is active. Preserve the
authority order, compatibility expectations, correctness, security, and data
integrity. Add regression coverage for behavior changes.
