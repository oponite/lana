# Lana 3.0 Source and Runtime Surface

## Provenance expressions

`evidence(value, "source")` and `assume(value, "proposition")` return the same
mathematical value with an immutable provenance root. `derivation(value)`
returns a canonical map/array record, while `explain(value)` returns its fixed,
deterministic text rendering. Labels are explicit strings and imply no
relationship or inference rule. Successful `observe` is the only current
operation that advances a derivation revision.

The mathematics of `STATE`, `STATE_DIST`, `MEASURE`, `TRANSFORM`, and `APPEND`
is defined only by `papers/semantics.md`. This document defines source syntax and
programmer-visible runtime behavior.

## Type, effect, and failure foundation

Lana distinguishes ordinary values from `Information<T>`,
`Claim<T, Proposition>`, `Sample<T>`, planned effects, task handles,
capabilities, and `Result<T, E>`. The initial source constructors and accessors
are ordinary calls:

```lana
let unknown = information(42);
let asserted = claim(true, "sensor is active");
let sampled = random();
let value = sample_value(sampled);
let metadata = sample_metadata(sampled);
let io = capability("io");
let deferred = planned_effect("io", io);
let decision = execute_effect(deferred);
let receipt = effect_status(deferred);
let success = result_ok(value);
```

`claim_value`, `claim_proposition`, `claim_status`, `execute_effect`,
`effect_status`, `result_is_ok`, `result_value`, and
`result_error_value` are explicit accessors. Claims require a literal explicit
proposition. Claim status exposes exactness, tolerance, and source validity as
separate fields. Stochastic reads require `sample_value` before exact use and carry
immutable metadata fields `source_dependency`, `rng_seed`, `task_lineage`,
`operation`, and `revision`.

The compiler tracks `pure`, `observation`, `stochastic`, `io`, `mutation`,
`task`, and `external_call` effects. Unresolved guards reject real-world
effects. Planned effects remain inert typed data until `execute_effect`; a
successful execution is cached by plan identity and committed revision, so
propagation and repeated access reuse its receipt. An unresolved payload is not
executable. Diagnostics use stable error
codes/kinds and source spans; runtime failures additionally expose causes,
resolution, exact-support, cancellation, and resource-limit context without a
partial result.

`information(value)` creates a process-local live dependency root. Pure
arithmetic, comparison, and unary Boolean operations over it produce live
derived Information values. A successful `observe` may select only an existing
alternative, publishes one atomic revision, and recomputes affected pure nodes.
Arrays, maps, field/index access, variables, and function calls retain those
links. Serialization materializes the current revision. Combining two finite
uncertain values requires the same dependency identity or an explicit joint;
Lana does not silently form a Cartesian product.

## STATE

```lana
state a = state(p: 0.4, d: 0.2);
state b = state(p: 0.4, d_re: 0.1, d_im: -0.3);
```

Exactly one disposition form is required: real-axis `d`, or both `d_re` and
`d_im`. Readable fields are `.p`, `.d_re`, and `.d_im`. The disposition must be
in the closed complex unit disk. `p` must be in `[0,1]`. Runtime expressions are
allowed; materially invalid results raise `LANA_ERR_INVALID_STATE`.

The implementation tolerance is `1e-12`. A probability is clamped only within
that tolerance. A disposition whose radius exceeds one only within tolerance is
normalized. Boundary probabilities force both disposition components to zero,
and exposed negative zero is normalized to positive zero.

Optional `timestamp`, `source`, `weight`, and `confidence` fields are metadata.
Construction, assignment, history, and transforms preserve them. States created
by an APPEND sampling kernel have empty metadata because Lana 2.0 defines no
metadata propagation rule for APPEND.

## Distributions and observation

```lana
let dist = append(a, b);
let concrete = sample(dist);
let bernoulli = measure(dist, result: "distribution");
let probability = measure(dist, result: "probability");
let bit = measure(dist, result: "sample");
```

`append()` accepts every `STATE`/`STATE_DIST` pair and returns an immutable lazy
`STATE_DIST`. `sample()` accepts only `STATE_DIST`, returns one concrete `STATE`,
and does not mutate the distribution. Measurement accepts a state or distribution
and is read-only. Its default mode is `distribution`; `probability` computes the
exact expected probability recursively, and `sample` draws one classical bit
from that exact Bernoulli mixture.

Unqualified `measure` is the exact computational-basis operation and continues
to compile to `MEASURE`. The computational basis is ordered as
`(|0>, |1>)`, so the returned distribution is always `distribution(1-p, p)`.

### Basis-aware measurement

Concrete states may be measured in one of three named ordered bases:

```lana
let px = measure(belief, basis: "x", result: "probability");
let dy = measure(belief, basis: "y", result: "distribution");
let bit = measure(belief, basis: "x", result: "sample");
```

The basis names and outcome ordering are:

```text
computational = (|0>, |1>)
x             = (|+>, |->)
y             = (|+y>, |-y>)
```

For a concrete `STATE`, define `q_B` as the exact probability of outcome `1`.
With `c = sqrt(p * (1 - p)) * (d_re + i * d_im)`, the values are
`q_computational = p`, `q_x = 1/2 - Re(c)`, and `q_y = 1/2 + Im(c)`.
Probability mode returns `q_B`, distribution mode returns
`distribution(1 - q_B, q_B)`, and sample mode draws `Bernoulli(q_B)`.
All three modes are exact and read-only. A qualified
measurement of `STATE_DIST` supports only `sample`: it samples one concrete
state from the distribution, then performs the selected exact basis
measurement. Qualified `probability` and `distribution` on `STATE_DIST`
return `LANA_ERR_UNSUPPORTED_EXACT_MEASUREMENT`; use unqualified measurement for
the existing exact computational-basis expectation.

### Monte Carlo STATE_DIST estimation

Approximate basis-aware expectation is explicit in the source language:

```lana
let p = estimate_measure dist in x as probability with samples: 10000;
let d = estimate_measure dist in y as distribution with samples: 10000;
```

`estimate_measure` accepts only a `STATE_DIST`, and `samples` must be a positive
integer literal. It samples a concrete state `N` times, averages the exact
outcome-1 probability `q_B` of each sampled state, and returns that average (or
`distribution(1-q_hat, q_hat)`). This is a deliberate Monte Carlo
approximation, not the exact mathematical probability and not a hidden runtime
optimization. The VM uses the configured seed and independent samples; larger
sample counts generally reduce error, but no confidence interval is returned.
Zero, negative, non-integer, or non-literal counts, unsupported modes, invalid
bases, and non-`STATE_DIST` inputs are rejected.

Equality compares canonical concrete states by exact binary64 equality of `p`,
`d_re`, and `d_im`. Metadata is not part of state equality. Equality or inequality
involving `STATE_DIST` raises `LANA_ERR_UNSUPPORTED_OPERATION`.

## Information and named joints

The source-level Information forms use LABC v5 when they require the balanced
Core surface. In Lana 4.0, source `sample` expressions also emit v5 so
unweighted Possibility cannot acquire legacy uniform-sampling behavior.
Published v1-v4 chunks retain their established behavior:

```lana
let product = joint independent { x: a, y: b };
let correlated = joint correlated (x, y) with support: [
    [0, 10, 0.25],
    [1, 11, 0.75]
];
let relation = joint conditional { x: a, y: kernel };
let joint_value = rename(product, "x", "subject");
let one_variable = project(joint_value, "subject");
let refined = condition(joint_value, a);
let sampled_assignment = sample(one_variable);
let definite = resolve(refined); // succeeds only for singleton support
```

`condition(info, evidence)` and `observe(info, evidence)` are the public
refinement forms. The historical three-argument named-joint condition spelling
is a compatibility alias. `std/core.distribution([[value, weight], ...])`
constructs finite weighted support; `possibility([value, ...])` remains
unweighted and cannot be sampled. For definite, Possibility, and Distribution
inputs, evidence is either one definite value or an unweighted
`possibility([value, ...])` subset. Impossible evidence fails with
`LANA_ERR_INVALID_CONDITIONING`. A Distribution keeps matching weights and
renormalizes them. The result retains its form even with one value; call
`resolve` to get that value. Joint evidence remains a map from names to exact
values. Paths reject both refinements. `condition` leaves the input and VM
revision unchanged; successful `observe` records one revision.

Source uses typed joint forms; descriptor strings are rejected. Each correlated
support row contains one value per declared variable followed by a positive
weight, and all row weights must sum to one within `1e-12`. Names are unique
and canonicalized in the runtime. `project` performs exact marginalization and
combines duplicate projected rows; `condition` is pure refinement and reports
`LANA_ERR_INVALID_CONDITIONING` for impossible evidence. `sample` is read-only
stochastic evaluation and chooses one complete correlated row. `resolve`
returns `LANA_ERR_UNRESOLVED_VALUE` unless the law has singleton support and does
not choose a representative. Conditional nodes declare their supported exact
operations; the initial opaque-kernel form returns `LANA_ERR_UNSUPPORTED_OPERATION`
for inference. Unsupported general inference and equality stay explicit errors.
Finite unresolved values and guarded execution are explicit:

```lana
let guard = possibility([true, false]);
let result = 0;
if (guard) { result = 10; } else { result = 20; }
let selected = sample(result);
let refined = observe(correlated, "x", 1);
```

Pure arithmetic, comparison, and function execution map over alternatives while
preserving a shared dependency identifier. An unresolved `if` executes both
branches into guarded environments and joins changed values as a path set.
Unresolved loop guards are rejected. Printing, host calls, task creation, and
observation inside an unresolved branch fail before producing an effect.
`condition` does not record an event; successful `observe` does. Path count is
bounded by the VM and exhaustion returns `LANA_ERR_PATH_LIMIT` without exposing a
partial result.

## Dataset selection

Dataset filters accept definite Boolean results only. Unresolved results and
sorting, grouping, or joining keys raise `LANA_ERR_UNRESOLVED_VALUE`; a definite
non-Boolean predicate raises `LANA_ERR_TYPE`. Nested uncertainty is rejected.
Keys are checked even for singleton inputs and empty joins. Use an explicit
supported `resolve` operation before selection; selection never samples or
implicitly resolves a value. Failure returns no partially materialized dataset.

## Native compiler and modules

The production pipeline is `Lana source -> Lana lexer -> fixed-layout typed
syntax -> semantic IR -> textual LABC -> Rust assembler/verifier -> Rust VM`. The
compiler sources live in `compiler/`; `compiler/bootstrap/compiler.lasm` is the
reproducible textual bootstrap artifact. A normal build assembles that artifact
and executes it in the Rust VM without invoking Python. Published LABC v1-v2
bytes remain covered by frozen compatibility fixtures.

Imports are relative `.lana` paths and must precede executable syntax. The
native loader canonicalizes paths, rejects cycles and imported-module top-level
statements, deduplicates modules, and resolves local and alias-qualified calls.
Compiler execution uses explicit 256 MiB memory and 50,000,000-instruction
policies; exhaustion is an error and never emits partial bytecode.

## Decision surface

`import "std/decision" as decision;` exposes ordinary module functions.
`recommend_information(policy, input, evaluation)`, `recommend_definite(...)`,
and `recommend_sampled(...)` wrap `policy_evaluate` and return an advisory map
with `mode` and `advisory: true`. A recommendation does not authorize execution.
The mode labels do not perform inference, conversion, or sampling.

`value_of_information(prior, candidates, actions, utility, costs)` scores
explicit finite relationships; missing relationships remain unscorable.
`decision_context(...)` records alternatives, utility, evidence, provenance,
assumptions, evaluation time, and recommendation. `review_outcome(context,
outcome)` reports decision quality separately from observed outcome quality.
`value_of_deliberation(value, direct_cost, delay_cost)` subtracts costs;
`validate_alternative_set(alternatives)` reports missing alternative categories.
Malformed probabilities, duplicate names, missing utility pairs, and non-finite
costs fail validation. See the runnable [operation index](../docs/operations.md).

Durable policy evaluation, ledger entries, and claims use the existing
`policy_evaluate`, `ledger_*`, and claim APIs. Advisory maps are not durable
claims or Execution authorizations.

## Future messages

`import "std/future_messages" as messages;` exposes `create(record)`,
`inspect(id)`, `check(context)`, `receive(after_id, limit)`, `acknowledge(id)`,
and `cancel(id)` on the Rust runtime. The application calls `store_open(path)`
first and owns the checks: on startup, relevant events, or a timer. Nothing runs
while the application is stopped. A missed event must be replayed explicitly.

`create` takes `{id, payload, condition, provenance_refs}`. The ID is 1–128
ASCII letters, digits, `.`, `_`, or `-`; provenance references are an array of
at most 16 nonempty strings of at most 128 UTF-8 bytes each. Payloads must be
plain, definite JSON-compatible snapshots, at most 64 KiB encoded. Process-local
handles, cyclic values, non-finite numbers, and values carrying live metadata
are rejected. The stored record has `schema_version: 1`, `created_at_utc` in
Unix seconds, and one of `pending`, `ready`, `acknowledged`, or `cancelled`.

`condition` contains `not_before_utc` (a finite nonnegative Unix-seconds
number), `event` (a nonempty name), or both. An event may have `comparisons`,
an array of up to 16 `{field, op, value}` maps. Operators are `eq`, `ne`,
`lt`, `le`, `gt`, `ge`; equality compares like-typed definite scalars, and
ordering compares numbers or strings of the same type. Comparisons are joined
with AND. Comparison strings are at most 4 KiB; event names, field names, and
context IDs are at most 128 UTF-8 bytes. A check receives `null` or
`{id, event, values}` with at most 64 context fields. The runtime snapshots
that context and captures one current time per check. Missing or unresolved
fields leave a message pending; bad
schemas or incompatible types return errors. No prose inference or implicit
sampling occurs.

`check` returns `{released_ids, pending}`, where each pending entry has `id`
and a reason. It validates all pending messages before changing any. Each
matching message then moves to `ready` in its own atomic store revision with a
receipt containing match time, context ID (or null), condition, and exact
revision as a decimal string. Ready persists through restarts and clock
rollback. `receive(after_id, limit)` accepts a limit of 1–100 and reads up to
that many ready records in ID order;
use `""` for the first page and the last returned ID for the next. Receiving
does not acknowledge. Acknowledgment and cancellation record their time and
revision. Repeated identical creates, checks, acknowledgments, and cancellations
do not create another transition. Cancellation cannot undo acknowledgment.
Creation also records its time and exact revision in `creation_receipt`.

At most 1,024 messages may be pending; a stored record is at most 128 KiB.
Limits fail before the affected transition is staged. A transition refuses
unrelated staged store writes. `LANA_ERR_IO` during a message commit means the
commit outcome is unknown: reopen the store and inspect the stable ID before
retrying. The inbox is local to the application's existing store access
boundary and sends no notification or external effect.

## Execution surface

Core inspection, policy decisions, and execution receipts are schema-1 Lana
maps. Each exposes `record_schema`, `id` when an identity exists, `kind`,
`transport_status`, `domain_status`, `payload`, `error`, `evidence`,
`assumptions`, `exactness`, and `metadata`; form-specific fields remain on the
same map. A plain value with no stable provenance has a null inspection ID.
The bridge has its own protocol schema and serializes these maps as JSON
without changing their record schema. Consumers check both versions.

`import "std/execution" as execution;` exposes four functions:

- `plan_webhook(path, payload)` creates a pure schema-1 plan map with
  `kind: "webhook"`, `path`, and `payload`. The path is origin-relative,
  without URL components or control characters;
  the payload must be definite JSON-compatible data.
- `capability()` obtains the host-configured opaque HTTPS capability.
- `authorize(capability, decision, plan)` binds an Authorize decision to this
  capability and the digest of this exact plan. The decision must be the
  unchanged map issued by `policy_evaluate` in the same VM; a source-created
  or modified Authorize map has no execution authority.
- `execute(capability, authorization, plan)` commits a Pending receipt before
  one POST attempt, then returns a schema-1 receipt record with `status` of
  Succeeded, Failed, or Unknown, plus its plan digest and authorization ID.

Host configuration owns the HTTPS origin, credential reference, and receipt
store. Source cannot choose arbitrary origins or headers. Invalid plans,
revoked tokens, mismatched authorization, duplicate execution IDs, and a store
with unrelated staged writes fail before sending. Redirects and retries are
disabled. A transport failure or timeout can mean Unknown: the remote server
may already have acted. This is at most one local send attempt per durable
execution ID, not exactly-once remote delivery.

If receipt persistence fails, the store closes and must be reopened and
reconciled by execution ID before any retry. An uncertain terminal write does
not erase the previously committed Pending receipt. Temporary credential files
are private from creation and removed on ordinary success and error paths.
Process termination can interrupt cleanup; credentials are never receipt data.

## Transforms

```lana
transform target with invert();
transform target with neutralize();
```

The statement replaces `target`. `invert()` maps `(p,d)` to
`(1-p, conjugate(d))`. `neutralize()` maps `(p,d)` to `(p,0)`. Both preserve
metadata and lift lazily over `STATE_DIST` because each registry entry supplies
a concrete rule, validity guarantee, and exact expected-probability rule.

A concrete transform can be deterministic, Borel-measurable, and
validity-preserving yet remain state-only. A distribution lift additionally
requires an exact rule for expected probability. Missing exact support produces
`LANA_ERR_UNSUPPORTED_OPERATION`; Lana never substitutes Monte Carlo estimation.

`apply`, `compose`, `collapse`, `reset_d`, and unlisted transform names are not
Lana operations and are rejected by the source compiler.

## Ordinary language and tasks

Numbers, booleans, strings, null, arrays, functions, control flow, host calls,
history, indexes, CLI commands, and task syntax retain their prior behavior.
Forked tasks own independent VM heaps, budgets, error states, and RNG streams.
Arguments and joined results are deep-copied; shared distribution subgraphs stay
shared inside the receiving VM and are never shared across VMs.

The filesystem host calls `directory_list(path)`, `directory_create(path)`,
`path_exists(path)`, and `write_text_atomic(path, text)` are effectful local
tooling boundaries. `directory_list` returns sorted maps with `name` and
`kind` (`file` or `directory`); it does not expose directory entries on a
failed read. `directory_create` is idempotent for an existing directory.
`path_exists` returns false only for a missing path and reports other lookup
failures. `write_text_atomic` publishes a complete replacement or reports an
I/O failure without replacing the destination. `hash_update(seed_hex, text)`
returns the exact lowercase 16-digit FNV-1a state for incremental deterministic
tooling hashes. Its internal `"xor"` mode combines two such states for
dependency lock identities; hash values remain strings so 64-bit results are
never rounded through Lana numbers.

Shared Information is created with `shared_information(value)`, which returns
only an admin capability. `shared_grant(admin, "read"|"observe"|"admin")`
creates a distinct token; `shared_revoke` requires admin authority. Reads use
`shared_snapshot`, `shared_at`, or `shared_wait`; observation uses
`shared_observe(capability, evidence, effective_time)`. Effective times are
exact nonnegative integers. `shared_identity` and `shared_revision` expose
process-local metadata. `inspect_information` returns an ordinary map and
reuses canonical derivation/runtime metadata rather than defining new inference.

A project is rooted by schema-1 `lana.toml`; `lana.lock` records the content
identity used by the build cache. `lana new`, `build`, `run`, `test`, `check`,
`fmt`, and `doc` use stable nonzero failure exits. Formatting is idempotent and
`fmt --check` performs no writes. `lana lsp` speaks JSON-RPC over standard I/O;
`lana debug` uses the same source-line mapping emitted by the compiler.
