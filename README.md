# Lana

[An ecosystem for informational computing.](https://oponite.github.io/data-articles/website/can-uncertainty-be-programmable.html)

## Install

macOS (Apple Silicon and Intel) and Linux (x64 and ARM64, glibc 2.35+):

```sh
curl -fsSL https://github.com/oponite/lana/releases/latest/download/install.sh | sh
```

Windows (x64 and ARM64), in Windows PowerShell 5.1 or PowerShell 7:

```powershell
irm https://github.com/oponite/lana/releases/latest/download/install.ps1 | iex
```

The installers download prebuilt Lana, its compiler, and standard library;
check the archive checksum; and configure your PATH. No Rust, Python, or
administrator access is needed. Open a new terminal, then run `lana version`.
Rerun the same command to upgrade. Releases remain unsigned.

```sh
lana new hello-lana
cd hello-lana
lana run
lana test
```

Optional Homebrew: `brew trust --formula oponite/oponite/lana`, then
`brew install oponite/oponite/lana`.

See [installer behavior and recovery](docs/dev/INSTALLERS.md), the
[platform support matrix](docs/PRODUCT_SUPPORT.md), or
[Contributing](CONTRIBUTING.md) to build Lana itself.

## Integrations

The optional integrations connect Lana 4.1 to JSON callers, MCP hosts,
Jupyter, VS Code, and Neovim through the Rust CLI and persistent worker.
See [`integrations/README.md`](integrations/README.md).

## Live programs

Run `lana live examples/live.lana` to start a foreground session. The command
prints an opaque process-local handle. A live source calls
`live_register("source", information(possibility([1, 2, 3])))` to retain a
root, and can register pure derived values as well. Session commands are:

```text
inspect <handle> source
observe <handle> source {"possibility":[2,3]}
pause <handle>
observe <handle> source 2
resume <handle>
delete <handle>
quit
```

`load <path.lana|path.labc>` creates another handle. An observation updates
the selected root and affected pure values without rerunning top-level code.
Handles end with the host process. `lana run` remains one-shot.

## Development Policy

The Lana language, compiler, bytecode, and VM are under active development.
Changes preserve the documented authority order, compatibility expectations,
correctness, security, and data integrity.

Language, bytecode, VM, and mathematical-contract changes require an accepted [LIP](docs/lip/README.md) before implementation; other work follows [GOVERNANCE.md](GOVERNANCE.md).

Use `lana new`, `build`, `run`, and `test` for ordinary projects.
[Contributing](CONTRIBUTING.md) describes builds and developer checks.
The [support matrix](docs/PRODUCT_SUPPORT.md) describes platforms and bytecode compatibility.
