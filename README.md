# Lana

[A programming language for uncertainty computation.](https://oponite.github.io/data-articles/website/can-uncertainty-be-programmable.html)

## Install and Run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
cmake --install build --prefix "$HOME/.local"
"$HOME/.local/bin/lana" run examples/belief.lana
"$HOME/.local/bin/lana" check examples/belief.lana
```

The installed `lana` command uses the Rust VM and the self-hosted Lana compiler
bytecode. Python is not required. The C11 `lanavm` binary remains a frozen
reference for LABC v1 and v2 conformance.

Rust is the canonical runtime (`lana-bytecode`, `lana-vm`,
`lana-runtime`, `lana-ffi`, and `lana-cli` under `vm/rust/`, `runtime/rust/`,
and `tools/rust/`). The compiler emits LABC v2 through v5 as required by the
source program. The C11 VM remains the conformance reference.

For VM development:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Low-level tooling assembles, verifies, disassembles, traces, and executes LABC:

```bash
build/lanavm asm examples/belief.lasm -o build/belief.labc
build/lanavm dis build/belief.labc
build/lanavm run build/belief.labc --trace
```

## Integrations

The source-install integrations connect Lana 3.0.0 to JSON subprocess callers,
MCP hosts, Jupyter, VS Code, Neovim, and a narrow native C ABI without adding
dependencies to the normal Lana build. Start with
[`integrations/README.md`](integrations/README.md).

## Development Policy

The Lana language, compiler, bytecode, and VM are under active development.
Changes preserve the documented authority order, compatibility expectations,
correctness, security, and data integrity.

Language, bytecode, VM, and mathematical-contract changes require an accepted [LIP] (lip/README.md) before implementation; other work follows [GOVERNANCE.md] (GOVERNANCE.md).
