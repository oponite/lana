# Contributing

## Build

Build prerequisites are Rust/Cargo, Python 3.11 or newer, and a native linker.
The regression suite also uses Bash, curl, and OpenSSL. Supported release hosts
are Linux and macOS. The installed CLI does not require Python.

```bash
python3 tools/build.py build
```

The command puts the CLI, compiler bytecode, standard library, and license in
`target/lana`. Release is the default profile. `--profile debug` selects an
unoptimized CLI.

The Rust runtime is a Cargo workspace at the repository root (crates under
`vm/rust/`, `runtime/rust/`, and `tools/rust/`):

```bash
cargo build --locked --release -p lana-cli
```

CLI and WASM Cargo builds share `tools/compiler_build.rs`. This script assembles
and checks the compiler bootstrap in Cargo's `OUT_DIR`. A Cargo development
executable can find that compiler output without an existing build directory.

## Install

```bash
python3 tools/build.py install --from target/lana --prefix "$HOME/.local"
```

An installed CLI requires the adjacent `bin/lana-compiler.labc` and the standard
library in `share/lana/stdlib`. The installation also contains
`share/doc/lana/LICENSE`. `cargo install` alone does not install the compiler
and standard library.

## Test

```bash
python3 tests/run.py --no-build
cargo test --locked --workspace --no-fail-fast
```

`tests/cases.json` defines regression commands, expected results, and timeouts.
The runner uses isolated temporary directories and reports failures across the
suite. It requires all success markers and rejects signals as expected failures.
An empty filter fails.

```bash
python3 tests/run.py --list
python3 tests/run.py --no-build --filter 'bootstrap|local_install'
```

Add source regressions to `tests/regression/` and register them in
`tests/cases.json`. The bootstrap check compares two compiler outputs with the
checked assembly. Its limits remain 256 MiB and 50,000,000 instructions.

Published LABC v1-v2 bytecode and assembly fixtures are checked by
`lana_legacy_bytecode` in `tests/run.py`. Rust loader fuzzing lives in `fuzz/`.

## Focused feature checks

Run feature checks through the regression runner so it supplies their temporary
stores, command arguments, and host configuration:

```bash
python3 tests/run.py --no-build --filter 'native_future_messages|lana_execution_live'
python3 tests/run.py --no-build --filter native_dataset_uncertain_history
python3 tests/run.py --no-build --filter lana_brain_
python3 tests/run.py --no-build --filter 'lana_object_|native_object_'
python3 tests/run.py --no-build --filter 'lana_packages|lana_package_release'
cargo test --locked -p lana-vm v5_core_operation_matrix
cargo test --locked -p lana-vm measure
cargo test --locked -p lana-runtime
```

The Brain checks cover fitting, typed-memory restart, semantic retrieval,
evidence selection, and workshop reports. Dataset checks cover restart, failed
updates, and historical reads. The execution check creates a trusted local
HTTPS server and checks authorization and receipt failures.

Project and tooling checks cover generation, `fmt --check` without writes,
LSP queries, source breakpoints, instruction stepping, and JSON/DOT inspection.
Object checks cover source declarations, export, collection, and tooling.
Package checks cover cache integrity and the publication tag guard without
publishing. The manifest defines each command and its environment.
A focused check does not qualify a release. Use the gates in [AGENTS.md](AGENTS.md).

## Universal macOS build

```bash
rustup target add aarch64-apple-darwin x86_64-apple-darwin
python3 tools/build.py universal --output target/universal
python3 tools/build.py install --from target/universal --prefix /tmp/lana-install
```

The universal build requires matching compiler bytes and checks both architecture
slices. Both slices must run. An Apple Silicon host requires Rosetta for the
x86_64 check. Missing targets, linkers, lipo, or slice execution cause failure.

## Packaging and editor checks

```bash
python3 tools/build.py package --from target/lana --output /tmp/lana.tar.gz
sh package.sh target/universal /tmp/lana-macos-universal.tar.gz
python3 tests/test_editors_live.py neovim target/lana/bin/lana
(cd integrations/editors/vscode && npm install && npm test)
python3 tests/test_editors_live.py vscode target/lana/bin/lana
```

The packager checks installed assets and source execution before it replaces an
archive. It reports the archive SHA-256 in JSON. Missing assets or failed checks
preserve the previous archive. Release CI uses this packager.

The regression suite checks publication failures in a disposable Cargo target
directory. Test-only binaries never replace the normal installation.

Release CI checks archive digests and runs both installed macOS slices after
extraction. It also rebuilds source archives in a clean directory. Signing and
notarization remain deferred. Release acceptance requires the gates in
[AGENTS.md](AGENTS.md) and the [release checklist](docs/release-checklist.md).
Branch protection uses the check names in
[.github/BRANCH_PROTECTION.md](.github/BRANCH_PROTECTION.md).

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
