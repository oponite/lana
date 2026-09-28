# Lana

[A programming language for uncertainty computation.](https://oponite.github.io/data-articles/website/can-uncertainty-be-programmable.html)

## Install and Run

```bash
python3 tools/build.py build
python3 tools/build.py install --from target/lana --prefix "$HOME/.local"
"$HOME/.local/bin/lana" run examples/belief.lana
```

The installed `lana` command uses the Rust VM and self-hosted Lana compiler
bytecode. Python 3 is required for the build/install/test scripts; it is not
required to run the installed CLI. Published LABC v1-v2 behavior is checked against
frozen fixtures. The Rust runtime executes LABC v1-v5 and v6 immutable values
and task-local classes with checked construction, mutation, and graph transfer.
Source `value`, `class`, and `interface` declarations compile to v6, including
checked methods, private factories, and same-module `copies`/`replace`. The build no longer has
Lana-owned C sources; it can still use system libraries such as SQLite and
macOS Metal.

For VM development:

```bash
python3 tools/build.py build
python3 tests/run.py --no-build
cargo test --locked --workspace --no-fail-fast
```

See [Contributing](CONTRIBUTING.md) for build, install, test, universal macOS,
and packaging instructions.

Low-level tooling assembles, verifies, disassembles, traces, and executes LABC:

```bash
target/lana/bin/lana asm examples/belief.lasm -o target/belief.labc
target/lana/bin/lana dis target/belief.labc
target/lana/bin/lana run target/belief.labc --trace
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


Run source examples from the repository root:

```bash
LANA_STDLIB_DIR="$PWD/stdlib" target/lana/bin/lana run examples/belief.lana
```

Replace the source path with a linked example. Some examples require a store,
host configuration, or command arguments, as noted in their section.
An installed CLI finds its standard library automatically.

Use `lana new`, `build`, `run`, and `test` for ordinary projects.
[Contributing](../CONTRIBUTING.md) describes builds and developer checks.
The [support matrix](support-matrix.md) describes platforms and bytecode compatibility.