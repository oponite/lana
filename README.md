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

The installed `lana` command uses the Rust VM and self-hosted Lana compiler
bytecode. Python is optional. Published LABC v1-v2 behavior is checked against
frozen fixtures. The Rust loader accepts LABC v1-v5. The build no longer has
Lana-owned C sources; it can still use system libraries such as SQLite and
macOS Metal.

For VM development:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Low-level tooling assembles, verifies, disassembles, traces, and executes LABC:

```bash
build/lana asm examples/belief.lasm -o build/belief.labc
build/lana dis build/belief.labc
build/lana run build/belief.labc --trace
```

## Integrations

The optional integrations connect Lana 4.0 to JSON callers, MCP hosts,
Jupyter, VS Code, and Neovim through the Rust CLI and persistent worker.
See [`integrations/README.md`](integrations/README.md).

## Development Policy

The Lana language, compiler, bytecode, and VM are under active development.
Changes preserve the documented authority order, compatibility expectations,
correctness, security, and data integrity.

Language, bytecode, VM, and mathematical-contract changes require an accepted [LIP] (lip/README.md) before implementation; other work follows [GOVERNANCE.md] (GOVERNANCE.md).
