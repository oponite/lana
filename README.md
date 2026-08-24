# Lana

Lana is a general-purpose language with:

- an immutable density-operator primitive `STATE`, with an observable probability
  and normalized complex disposition
- immutable lazy `STATE_DIST` values that compose states without sampling or
  mutation until explicitly measured or sampled
- and ordinary numbers, booleans, strings, arrays, functions, tasks, and host
  calls.

```lana
state belief = state(p: 0.50, d_re: 0.30, d_im: 0.10);
state evidence = state(p: 0.90, d: 0.65);
let combined = append(belief, evidence);
transform combined with invert();
print(measure combined as probability);
```

`p` is the observable probability. The complex normalized disposition is
`d = d_re + i d_im` with `|d| <= 1`; the shorthand `d:` selects the real axis.
At `p = 0` or `p = 1`, the disposition is canonicalized to zero. `append()`
creates an immutable lazy distribution, and `sample()` returns a concrete state.

Lana is personal-first: it is the language I use to build programs. It is also
public because the language model and implementation are worth exploring. Issues
and questions are welcome, and supported 1.0 defects will receive ongoing triage.

## Install and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
cmake --install build --prefix "$HOME/.local"
"$HOME/.local/bin/lana" run examples/belief.lana
"$HOME/.local/bin/lana" check examples/belief.lana
```

The installed compile and run path consists of the C11 `lana`/`lanavm` binaries
and the self-hosted Lana compiler bytecode. Python is not required.

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

## Optional integrations

The source-install integrations connect Lana 1.0 to JSON subprocess callers,
MCP hosts, Jupyter, VS Code, Neovim, and a narrow native C ABI without adding
dependencies to the normal Lana build. Start with
[`integrations/README.md`](integrations/README.md).

## Language basics

State fields accept runtime expressions. Read `p`, `d_re`, and `d_im` directly.
Optional `timestamp`, `source`, `weight`, and `confidence` metadata stays outside
the mathematical state and is preserved by assignment, history, and transforms.

`measure value` defaults to the Bernoulli distribution. Use `as probability` for
its exact expected probability or `as sample` for one classical bit. These
measurements are read-only. `sample(dist)` is distinct: it samples a concrete
`STATE` from a `STATE_DIST`.

Concrete states also support exact named-basis measurement with
`in computational`, `in x`, or `in y`. Basis-qualified probability/distribution
measurement of a `STATE_DIST` is intentionally unsupported; use the explicit
`estimate_measure dist in x as probability with samples: N` or distribution form
for the documented Monte Carlo approximation.

`fork` runs a function in an isolated VM. Arguments and results are deep-copied,
including shared distribution DAGs and metadata; bytecode remains immutable and
shared. Task groups, cancellation, timeout joins, arrays, control flow, and typed
JSON/CSV data boundaries remain ordinary language features. Model fitting and
inference belong in external programs that exchange ordinary Lana values.

The authority order is:

1. [papers/semantics.md](papers/semantics.md) — sole mathematical authority.
2. [SPEC.md](SPEC.md) — source syntax and programmer-visible behavior.
3. [BYTECODE.md](BYTECODE.md) — the single LABC v1 encoding.
4. [VM.md](VM.md) — allocation, cloning, RNG, and budget architecture.

Benchmark programs are reproducible source evidence; generated reports and
machine-local result snapshots are not part of the source release.

## 1.0 maintenance policy

The Lana language, compiler, bytecode, and VM feature set is frozen through
December 31, 2026 while development moves to programs built with Lana. During
that period, documentation, support, and compatibility-preserving fixes for
correctness, security, data loss, or installation failures remain in scope.
