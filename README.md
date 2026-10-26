# Lana

[An ecosystem for informational computing.](https://oponite.github.io/data-articles/website/can-uncertainty-be-programmable.html)

## Install

### Homebrew (optional)

Install the latest public release from [nitebrews](https://github.com/oponite/homebrew-nitebrews):

```sh
brew install oponite/nitebrews/lana
lana version
```

Homebrew builds from source and installs the required build dependencies
(Rust and Python for current releases). The installed Lana command does not
require Python. The tap updates after each public stable release; upgrade with:

```sh
brew update
brew upgrade oponite/nitebrews/lana
```

Public stable releases from 2.0.0 onward are also available by exact version:

```sh
brew install oponite/nitebrews/lana@4.0.0
"$(brew --prefix oponite/nitebrews/lana@4.0.0)/bin/lana" version
```

Exact-version formulas remain at that version and are not linked into your
PATH, so they can coexist with the default `lana`. Older versions retain their
original language behavior and bugs. See the tap README for the version list.

### Build from source

You need Git, Rust/Cargo, Python 3.11 or newer, and a native linker to build Lana. The installed `lana` command does not require Python.

1. Download and install Lana:

```bash
git clone https://github.com/oponite/lana.git
cd lana
python3 tools/build.py install --prefix "$HOME/.local"
```

2. Add Lana to your shell's `PATH`, then check the installation:

```bash
export PATH="$HOME/.local/bin:$PATH"
lana version
```

3. Create and run your first program:

```bash
lana new hello-lana
cd hello-lana
lana run
lana test
```

See [Contributing](CONTRIBUTING.md) for build, install, test, universal macOS,
and packaging instructions.

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
