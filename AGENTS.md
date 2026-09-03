s

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
2. `SPEC.md` — source syntax and programmer-visible behavior.
3. `BYTECODE.md` — the one LABC v1 encoding.
4. `VM.md` — runtime architecture and resource behavior.

Do not invent semantics from an implementation detail. Change the highest
applicable authority first when intentionally changing the language.

## Repository map

- `csrc/`: C11 runtime, VM, CLI, assembler, tooling, and project commands.
- `include/lana/`: public C API.
- `compiler/*.lana`: self-hosted compiler source.
- `compiler/bootstrap/compiler.lasm`: checked, reproducible bootstrap artifact.
- `examples/`: runnable Lana and LABC examples.
- `tests/c/`: runtime and public-API tests.
- `tests/native/`: source-language pass/fail fixtures.

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

Run `RELEASE_CHECKLIST.md` exactly. The native bootstrap CTest compiles the
compiler twice and requires byte-identical output. Sanitizer configurations are
mutually exclusive. The fuzz target is opt-in with `LANA_BUILD_FUZZERS=ON`.

LABC v1 is the only supported bytecode format. Pre-release bytecode and textual
assembly are not accepted or converted; rebuild them from source.

## Change rules

- Preserve unrelated dirty work and inspect a file before writing it.
- C is C11 with four-space indentation and `-Wall -Wextra -Wpedantic -Werror`.
- C names use `snake_case`; public types use `Lana...`; public functions use
  `lana_...`; constants use `LANA_...`.
- Lana uses lowercase `snake_case`, semicolons, and small explicit functions.
- Add C regressions in `tests/c/`; add language fixtures in `tests/native/` and
  register them in `CMakeLists.txt`.
- Never weaken compiler limits: 256 MiB and 50,000,000 instructions. Exhaustion
  is an error and must not expose partial bytecode.
- Benchmark result snapshots are machine-local evidence, not conformance.
- Run `git diff --check`; there is no repository-wide formatter.

## Development policy

Language, compiler, bytecode, and VM development is active. Preserve the
authority order, compatibility expectations, correctness, security, and data
integrity. Add regression coverage for behavior changes.
