# Contributing

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
```

The Rust runtime is a Cargo workspace at the repository root (crates under
`vm/rust/`, `runtime/rust/`, and `tools/rust/`):

```bash
cargo build -p lana-cli
```

## Test

```bash
ctest --test-dir build --output-on-failure
```

Published LABC v1-v2 bytecode and assembly fixtures are checked by
`lana_legacy_bytecode` in CTest. Rust loader fuzzing lives in `fuzz/`.

## Change process

1. Language, bytecode, or VM changes require an accepted LIP (`lip/`).
2. Bug fixes, documentation, and tooling do not.
3. Every source, bytecode, compiler, or VM change requires focused regression
   coverage.

## Code style

- Lana: the self-hosted compiler source under `compiler/`.
- Rust: the workspace crates under `vm/rust/`, `runtime/rust/`, and
  `tools/rust/`.

Prefer existing patterns before adding dependencies.

## References

- [Governance](GOVERNANCE.md)
- [Versioning](VERSIONING.md)
- [LIP process](lip/README.md)
