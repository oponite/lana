# Lana

> A programming language for uncertainty computation.

Most software treats a decision as a function that returns an answer. Lana
treats a decision as a *reasoned, auditable artifact*: uncertain inputs are
combined under an explicit mathematical contract, and the path from evidence to
outcome is recorded rather than hidden inside a black box.

The problem Lana addresses is concrete. Teams that make consequential decisions
under uncertainty — risk, forecasting, service health — usually reach for a model or a library. That works
until someone asks *why* a decision was made, or *what evidence* it rested on,
or *whether it can be reproduced*. A library gives you a number; it does not
give you a first-class, testable, versionable account of the reasoning.

Lana's answer is to make uncertain reasoning a language feature, not a library
call:

- **Explicit, explainable probability.** A `STATE` is an immutable
  density-operator primitive: an observable probability `p` plus a normalized
  complex disposition `d`. There is no hidden state, no implicit sampling.
- **Evidence combination with a mathematical contract.** `append()` combines
  independent evidence under a documented probabilistic rule. The result is a
  lazy `STATE_DIST` that composes without sampling until you explicitly
  `measure` or `sample` it.
- **Durable, auditable decisions.** A decision pipeline — store, policy,
  ledger, claims, and effects — records what was decided, on what evidence, and
  what effect was authorized. Decisions can be replayed and validated without
  re-executing their effects.
- **A language, not a library.** Reasoning lives in ordinary, testable source
  code with functions, types, and a compiler. It can be versioned, reviewed,
  and conformance-tested like any other code.

Lana is a public, actively developed language project. Issues, questions, and
focused pull requests are welcome. Changes to language behavior follow the
authority order below and include tests or documentation when applicable.

## Batteries Included

Lana comes with:

- an immutable density-operator primitive `STATE`, with an observable
  probability and normalized complex disposition
- immutable lazy `STATE_DIST` values that compose states without sampling or
  mutation until explicitly measured or sampled
- ordinary numbers, booleans, strings, arrays, functions, tasks, and host
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

## Reference applications

Four reference applications demonstrate the decision pipeline end to end, each
with fixtures:

- **Sensor fusion** — fuses two independent sensor readings into a single
  confidence, attaches provenance to each reading, and measures the fused
  probability.
- **Service health** — compares a live metric against a baseline, combines the
  two readings, and measures the resulting health probability.
- **Document router** — routes a document by provenance and sensitivity, then
  lets the policy decide whether the archive effect is authorized.
- **Advisory forecast** — reads a trend series, builds a forecast state from
  the latest point, and measures the forecast probability.

Each reads a fixture, computes a probability or route, and writes a
decision-request for the bridge pipeline. Sources live in
[`examples/reference-apps/`](examples/reference-apps/).

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

The canonical VM is the Rust runtime (crates `lana-bytecode`, `lana-vm`,
`lana-runtime`, `lana-ffi`, `lana-cli` under `vm/rust/`, `runtime/rust/`, and
`tools/rust/`). The C11 VM is retained as a frozen reference implementation for
conformance comparison.

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

## Three small examples

These examples introduce Lana in three steps:

1. ordinary computation
2. measuring one uncertain state
3. combining independent evidence

Run any tutorial example with:

```bash
build/lana run examples/tutorials/01_counter.lana
```

### 1. Ordinary computation

Lana supports familiar functions, variables, assignment, loops, and printing.

```lana
fn add(left, right) {
    return left + right;
}

let total = add(1, 2);
let step = 1;

while (total < 5) {
    total = total + step;
}

print(total);
```

Run it:

```bash
build/lana run examples/tutorials/01_counter.lana
```

Output:

```text
5
```

Source: [`examples/tutorials/01_counter.lana`](examples/tutorials/01_counter.lana)

### 2. Measure an uncertain state

A `state` represents an uncertain binary event. The `p` field is its observable
probability.

```lana
state belief = state(p: 0.75, d: 0.20);

let probability = measure belief as probability;
print(probability);
```

Run it:

```bash
build/lana run examples/tutorials/02_belief_measurement.lana
```

Output:

```text
0.75
```

Source: [`examples/tutorials/02_belief_measurement.lana`](examples/tutorials/02_belief_measurement.lana)

### 3. Combine independent evidence

`append()` combines two states using Lana's independent probabilistic-OR rule.

```lana
state first_signal = state(p: 0.40, d: 0.20);
state second_signal = state(p: 0.60, d: 0.30);

let combined = append(first_signal, second_signal);
let probability = measure combined as probability;
print(probability);
```

The combined probability is:

```text
1 - (1 - 0.40) * (1 - 0.60) = 0.76
```

Run it:

```bash
build/lana run examples/tutorials/03_combined_evidence.lana
```

Output:

```text
0.76
```

Source: [`examples/tutorials/03_combined_evidence.lana`](examples/tutorials/03_combined_evidence.lana)

## Repository

- `compiler/` — the self-hosted Lana compiler.
- `vm/` — canonical Rust `lana-vm` + `lana-bytecode`, plus the frozen C11
  reference VM core.
- `runtime/` — canonical Rust `lana-runtime` + `lana-ffi`, plus the C11
  hardware boundary.
- `tools/` — Rust `lana-cli` + `lana-fuzz` + `lana-wasm` (WebAssembly bindings),
  plus the C11 CLI, LSP, and project tooling.
- `spec/` — `SPEC.md`, `SYNTAX.md`, `BYTECODE.md`, `VM.md`.
- `papers/` — `semantics.md` (1.0) and `semantics-2.md` (2.0), the mathematical
  authorities.
- `lip/` — Lana Improvement Proposals.
- `stdlib/` — future Lana standard library.
- `tests/` — unit, regression, and conformance suites.
- `integrations/` — Python, editors, native ABI.

Project governance: [GOVERNANCE.md](GOVERNANCE.md), [VERSIONING.md](VERSIONING.md),
[CONTRIBUTING.md](CONTRIBUTING.md), [CHANGELOG.md](CHANGELOG.md).

## Optional integrations

The source-install integrations connect Lana 2.0.0 to JSON subprocess callers,
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

1. [papers/semantics.md](papers/semantics.md) — mathematical authority for the
   Lana 1.0 contract.
2. [papers/semantics-2.md](papers/semantics-2.md) — mathematical authority for
   the Lana 2.0 density-operator substrate and its operations.
3. [SPEC.md](spec/SPEC.md) — source syntax and programmer-visible behavior.
4. [BYTECODE.md](spec/BYTECODE.md) — the single LABC v2 encoding.
5. [VM.md](spec/VM.md) — allocation, cloning, RNG, and budget architecture.

New or changed source syntax must additionally satisfy
[SYNTAX.md](spec/SYNTAX.md) — the syntax design principles.

Benchmark programs are reproducible source evidence; generated reports and
machine-local result snapshots are not part of the source release.

## Development Policy

The Lana language, compiler, bytecode, and VM are under active development.
Changes preserve the documented authority order, compatibility expectations,
correctness, security, and data integrity. Language, bytecode, and VM changes
are proposed through the [LIP process](lip/README.md) and governed by
[GOVERNANCE.md](GOVERNANCE.md).
