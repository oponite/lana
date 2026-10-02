# Lana 4.1 Source and Runtime Surface

## Mathematical definition and language contract

Read `docs/papers/semantics.md` for mathematical meaning, then use this document
for source forms, effects, errors, and current availability. The section map
identifies where the language and implementation material belongs:

| Mathematical definition in `docs/papers/semantics.md` | Language contract here |
| --- | --- |
| 0.1 Information forms, joints, finite calculus, execution, and evidence | Type, effect, and failure foundation; Information and named joints; Finite information calls; Dataset selection and history calls |
| 0.4 bounded learning and 0.5 walk-forward evaluation | Bounded rules and tree calls; Walk-forward evaluation |
| 1 `STATE` and `STATE_DIST`; 6 valid boundaries | STATE; Distributions and observation |
| 2 `MEASURE`, including named bases and estimation | Distributions and observation; Basis-aware measurement; Monte Carlo STATE_DIST estimation |
| 3 `TRANSFORM` | Transforms |
| 4 `APPEND` and 5 composition, lifting, and sampling | Distributions and observation; Transforms |
| Appendix A equations | The corresponding operation sections above |
| Reactive Information | Type, effect, and failure foundation; Ordinary language and tasks |

The type, effect, failure, and object source contracts are defined here. Runtime
representation and resources belong to `VM.md`; instruction names and versioned
encoding belong to `BYTECODE.md`. A mathematical definition does not by itself
make a source feature available.

## Provenance expressions

`evidence(value, "source")` and `assume(value, "proposition")` return the same
mathematical value with an immutable provenance root. `derivation(value)`
returns a canonical map/array record, while `explain(value)` returns its fixed,
deterministic text rendering. Labels are explicit strings and imply no
relationship or inference rule. Successful `observe` is the only current
operation that advances a derivation revision.

## Immutable Information snapshots

`snapshot(info)` captures the current Information form, law, guards,
exactness, provenance and revision without resolution or observation.
It detaches live dependencies, preserves finite alternative alignment,
and deep-copies nested arrays and maps as immutable containers. Index
assignment, map writes, and array pushes on captured containers fail with
`LANA_ERR_UNSUPPORTED_OPERATION`. Effectful handles and executable values
are unsupported; depth over 64 and VM resource exhaustion fail without
returning a partial snapshot. Immutable Tensor storage can be shared while
preserving dtype and view metadata. Capturing does not advance an evidence revision.

## Additive object source contract

At module top level, `value`, `class`, and `interface` declarations use
the canonical grammar in `SYNTAX.md`. Top-level functions, ADTs, maps,
existing generics, `STATE`, and all five Information forms remain valid.
Nested object declarations, generic values, generic object methods,
generic interfaces, interface fields/bodies/static methods/inheritance,
and unmarked fields or methods are compile errors. Every value/class
field and method requires exactly one `public` or `private` marker.
Privacy belongs to the declaring type, not its source file or module.
Object type identity uses the portable module ID in `BYTECODE.md`,
so moving the same project tree between machines does not rename
its types. Two distinct modules with one portable ID fail linking.

A `value` has ordered typed fields with no defaults and no `mutable`.
`Name(args...)` supplies exactly one value per field in order. It is
available inside the value declaration; outside, it is available only
when every field is public. A public static factory may validate and
construct private fields with `Self(args...)`. Values may contain
`STATE`, `STATE_DIST`, and explicit immutable `snapshot(info)` values,
but no live Information or class reference, even through an interface.
`snapshot` freezes the current Information form, law/alternatives,
exactness, guards, provenance, and revision; it never calls `resolve`.
Values compare structurally only when every field has defined equality.
A `Tensor` field retains its dtype, dimensions, view, and immutable data.
A `Shape` field is a captured dimension array checked with the existing tensor
shape rules: at most 32 finite, nonnegative integer dimensions representable
by the runtime. Scalar shapes (`[]`) and zero dimensions are valid. Invalid
shapes fail before the value is published. Tensor field equality remains
unsupported; a Shape retains existing array equality behavior.

A `class` has identity. `new Name(args...)` passes arguments to its
declared `init`, never directly to fields. Outside code may use `new`
only with a public initializer. A class has at most one initializer.
With no initializer, `new Name()` is implicitly public only when all
fields have defaults; other argument lists or missing defaults fail.
Each pure field default is evaluated once per construction and cannot
read `self` or another field. The initializer assigns all remaining
fields before publication. Fixed fields are assigned exactly once
and cannot be reset after a default; mutable fields may be assigned
during initialization. Incomplete `self` cannot call an instance
method, escape, or be used externally. An initializer cannot return a
value or perform an external effect. A public static factory may call
its own private initializer through `new Self(...)`.

Class field access and assignment obey visibility and mutability.
Outside assignment requires a public mutable field and has the
`mutation` effect; private or fixed outside assignment fails at
compile time. New identity creation also has `mutation` effect and
cannot occur under an unresolved guard or pure Information lift.
Class equality uses identity. Class field assignment does not perform
`observe`. Direct class serialization is unsupported; explicit value
snapshot export cannot expose private fields automatically. `json_stringify`
accepts a value whose fields are all public and recursively JSON-serializable,
encoding `{"$lana_value":"<portable type identity>","fields":{...}}` with
sorted field names. Any private field, class reference, or unsupported payload
rejects the whole export. Export mathematical or Information payloads through
explicitly chosen JSON-compatible fields; this operation never measures or
resolves them. `json_parse` returns ordinary maps, not privileged object
instances; a user factory may validate them and construct a fresh identity.

`class Child copies Parent` may name one same-module class only.
The compiler copies fields, methods, and initializer at compile time
and creates an independent child type. `Child` is not usable where
`Parent` is required. A child may replace copied fields, methods, or
initializer, including type, visibility, mutability, default, effect,
or signature, only with leading `replace`. Missing `replace`,
unknown replacement, duplicate member, or incompatible copied body
fails at compile time. There is one final field per name. Copied
bodies, including static methods, are rechecked against that final
shape. Calls on `self` dispatch to the child's final methods.
`Self` in copied bodies and result types rebinds to the child; a
literal parent name still names the parent. Copied private members
belong to the child. No runtime parent object or implicit second
initializer call exists.

`interface` promises only public instance-method signatures with
explicit typed parameters, result type, and maximum effects. Values
and classes explicitly `implements` zero or more interfaces; a copied
child must repeat the promise and is checked after replacement.
Interface-typed values hold a value snapshot or the same class
reference within one task. Calling through the latter preserves
identity. Copying alone grants no interface conversion. An omitted
`effects()` list promises pure read-only behavior; an explicit list
uses existing effect names and may include `mutation`, `observation`,
`stochastic`, `io`, `task`, or `external_call`. Unknown or repeated
names fail. The inferred method effects must be a subset of each
promised effect set. Value methods cannot mutate their receiver or
perform effects through captured live references.

Instance methods start with `self` and call as
`receiver.method(args...)`. Static methods omit `self`, call as
`Name.method(args...)`, have no shared mutable type state, and never
overload. Every other parameter is typed; every non-`init` method
declares `-> Type`, using `-> null` when appropriate. A field,
instance method, and static method cannot share a name. Instance
method overloads may differ only by explicitly typed signatures
involving `Information<T>`, `STATE`, or `STATE_DIST`. The inner `T`
distinguishes Information overloads, but all five Information forms
select the same `Information<T>` overload. Selection uses declared
argument types, never runtime form, result type, sampling, or
implicit conversion. No matching or ambiguous overload is a
compile error. Interfaces check each overload separately.

A pure value method on `Information<Value>` lifts pointwise under the
existing pure-function rules and returns `Information<ResultType>`
while preserving alternatives, dependency, guards, weights,
exactness, and live revision links. Effectful calls require an
explicitly definite receiver; unresolved class identity cannot
receive field or method access. Two names in one task may alias a
class object. Cross-task transfer deep-copies class graphs with
fresh identities and preserves aliasing and cycles inside the
receiving task; live Information fields become immutable snapshots.
Transfer failure exposes no partial result.

Conformance includes valid construction with `STATE`, structural
versus identity equality, private factories, public mutable assignment,
one copied child with field/method replacement and `Self` rebinding,
interface dispatch, every allowed overload family, all five pure
Information lifts, task transfer with cycles, and failures for
privacy, ambiguity, incomplete initialization, nested declarations,
imported-parent copying, effect violations, unsupported serialization,
and malformed source spans. Existing source and v1-v5 fixtures retain
their behavior. OO programs require the v6 extension described in
`BYTECODE.md` and `VM.md`.

The mathematics of `STATE`, `STATE_DIST`, `MEASURE`, `TRANSFORM`, and `APPEND`
is defined only by `docs/papers/semantics.md`. This document defines source syntax and
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

A structured failure retains its code, kind, message, full source span,
operation, bounded cause chain, resolution reason and remaining-alternative
count, exact-support status, cancellation context, and resource-limit context.
Human rendering may change formatting but must preserve those fields' meaning.
`Information<T>`, `Claim<T, Proposition>`, `Sample<T>`,
`PlannedEffect<T, Effect>`, `TaskHandle<T, Capabilities>`,
`Capability<Name>`, and `Result<T, E>` remain distinct types; none unwraps
implicitly to `T`. Claims carry caller-supplied propositions, not propositions
inferred from `STATE` probabilities. Task join and timeout return typed results.

`information(value)` creates a process-local live dependency root. Pure
arithmetic, comparison, and unary Boolean operations over it produce live
derived Information values. A successful `observe` may select only an existing
alternative, publishes one atomic revision, and recomputes affected pure nodes.
Arrays, maps, field/index access, variables, and function calls retain those
links. Serialization materializes the current revision. Combining two finite
uncertain values requires the same dependency identity or an explicit joint;
Lana does not silently form a Cartesian product.
Each successful root observation publishes affected pure nodes together and
retains the previous revision as immutable history. Effect results are leaves
of the dependency graph. A planned effect executes at most once per
`(plan identity, committed revision)` and later reads reuse its receipt.

### Persistent live code

`live_register("name", information_value)` explicitly retains and names an
Information root or derived value in the current live program. The name must
be a nonempty string literal unique within the instance. The call returns the
same live value. It is an effectful registration, allowed during initial
evaluation, and does not itself observe or advance a revision. A host can
observe only registered roots and inspect any registered value. A derived
value cannot be used as an observation target. Programs without a live host
may use the call, but registrations end when that one-shot VM exits.

`lana live <source.lana|program.labc>` evaluates once and prints a process-local
handle. Its foreground session accepts `load`, `observe`, `inspect`, `pause`,
`resume`, `delete`, and `quit`, with each instance command naming its handle.
`lana run` keeps its one-shot behavior. The Rust `LiveHost` and Python `Lana`
live methods use the same runtime; handles cannot be transferred to another
host process or used after shutdown.

An observation supplies definite JSON evidence, an explicit unweighted
Possibility subset, or the existing tagged finite-value encoding for values
outside ordinary JSON. Invalid evidence returns an error and leaves the
revision unchanged. A successful event updates the root and affected pure
dependencies atomically, in deterministic graph order. Inspection and
`measure` are reads; they do not trigger an event or execute an effect.

An instance is `LIVE` during initial evaluation or event processing,
`QUIESCENT` when idle, `SUSPENDED` while accepting queued observations,
`FAILED` after a terminal VM failure, and `DELETED` after deletion. Pausing
queues observations FIFO under a 64 MiB queue cap and the instance memory
budget. Resume reports each queued event's result. Invalid evidence fails its
own event and leaves later events eligible. A terminal VM failure stops the
queue; inspection remains available until deletion. Unknown or deleted
handles return `LANA_ERR_NOT_FOUND`.

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
`append()` leaves both inputs unchanged. A transform produces a new state or
distribution value; only the `transform target with ...` statement replaces a
binding. No operation silently samples, collapses, or changes its source.

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

## Finite information calls

Import `std/core` and call `entropy(joint, names)`,
`conditional_entropy(joint, target_names, given_names)`,
`mutual_information(joint, left_names, right_names)`, or
`broja(joint, target_name, x_name, y_name)`. Name lists are nonempty,
unique, and present in the same finite joint; the groups may overlap.
Internally use canonical joint-name order. The first three calls return
finite numbers in bits with provenance, selected names, and the input
revision retained in their result metadata. `broja` requires three
distinct single names
and returns either
`{status:"converged", shared, unique_x, unique_y, synergy, total,
error_bound_bits, input_revision, target, x, y}` or
`{status:"unconverged", reason, marginal_residual,
mass_residual, error_bound_bits, input_revision, target, x, y}`.
The unconverged result contains no component or total fields. A missing
certificate sets `error_bound_bits` to null and cannot be reported as
convergence.

`kernel(input_domains, output_domain, chances_fn)` accepts a nonempty
ordered finite domain per input, a nonempty ordered output domain, and a
named top-level pure Lana function. Domain members are distinct definite
values under finite-joint equality. The function receives one Cartesian
input tuple as an array and returns exactly one
`[output, probability]` pair for each declared output in output-domain
order. It is called once for every tuple during construction, using
the existing effect check and function-reference lowering. Wrong type,
missing/duplicate output, thrown error, or attempted effect rejects
the entire kernel. The result stores immutable rows, never a live
callback. The returned `Kernel` is opaque: no general equality,
JSON serialization, or store persistence is defined in this slice.
`identity_kernel(domain)` produces the identity law.
`compose_kernels(later, earlier)` requires the earlier output domain
to equal the later single input domain; it returns their normalized
composition or fails on mismatch.

`network(root_joint, nodes)` accepts a finite named root joint and an
ordered list of `{name, parents, kernel}` records. Each new name is
unique; each parent names a root or node; its order and domain match
the kernel inputs exactly. Root domains are positive-mass support and
child domains are kernel output domains. Duplicate, missing,
incompatible, or cyclic declarations fail before publication.
The returned `Network` is likewise opaque and is not implicitly
serialized.
`infer(network, query_names, evidence_map)` returns a finite named
joint for a nonempty unique query group, conditioned on a map from
declared names to exact values. A queried variable in evidence has a
singleton result. Zero-probability evidence returns
`LANA_ERR_INVALID_CONDITIONING`. The call never guesses a missing
relationship or samples.

`forget_weights(distribution)` returns unweighted positive-mass
Possibility support with a provenance note that weights were discarded.
`assign_weights(possibility, rows)` requires exactly one
`[candidate, positive_weight]` row per candidate with total one within
`1e-12`. No default uniform law is supplied. Both calls are pure and
leave the source intact. The existing `distribution(rows)` constructor,
`resolve`, `sample`, `measure`, and `STATE` embeddings retain their
current meanings. Other cross-form conversions are unsupported.

Wrong argument type returns `LANA_ERR_TYPE`; invalid names, domain, or
topology return `LANA_ERR_INVALID_PARAMETERS`; invalid probability
returns `LANA_ERR_INVALID_DISTRIBUTION`; unsupported exact law or
conversion returns `LANA_ERR_UNSUPPORTED_OPERATION`. Every call
performs overflow-safe cardinality and budget checks before
materialization. Limit, OOM, callback error, cancellation, or numerical
failure publishes no partial result. A finite input that yields a
non-finite measure or a negative measure beyond its stated rounding
tolerance returns `LANA_ERR_INVALID_PARAMETERS` with
`reason: "numerical_failure"`. No call commits evidence or
authorizes an effect. These library calls use existing source grammar
and function-reference/host-call lowering; previously valid source
and published LABC keep their behavior.
`BYTECODE.md` reserves host names and IDs 189–199 for these calls;
`core_infer` is distinct from the existing ML `infer` host call.
Conformance cases include fair/certain entropy, independent and copied
bits, XOR BROJA synergy one bit, duplicate-source shared one bit,
identity and composed kernels, a two-step network query, effectful and
invalid callbacks, missing relations, impossible evidence, exhausted
budgets, explicit weight loss and assignment, and unchanged published
bytecode fixtures.

## Dataset selection

Dataset filters accept definite Boolean results only. Unresolved results and
sorting, grouping, or joining keys raise `LANA_ERR_UNRESOLVED_VALUE`; a definite
non-Boolean predicate raises `LANA_ERR_TYPE`. Nested uncertainty is rejected.
Keys are checked even for singleton inputs and empty joins. Use an explicit
supported `resolve` operation before selection; selection never samples or
implicitly resolves a value. Failure returns no partially materialized dataset.

## Dataset history calls (all six calls implemented)

Existing `dataset`, `filter`, `map`, `select`, `limit`, `sort`,
`group_by`, `aggregate`, `join`, `materialize`, and plan `explain`
retain their calls and definite-selection behavior. After
`store_open`, six ordinary library calls add versioned history:

| Call | Result |
| --- | --- |
| `dataset_source(source_id)` | Register an empty named source; identical registration is idempotent. |
| `dataset_query(id, source_ids, plan_name, calculation_version)` | Register one named pure plan and its initial snapshot. |
| `dataset_apply(source_id, expected_revision, batch_id, changes)` | Atomically publish source and dependent query revisions. |
| `dataset_snapshot(query_id, revision)` | Return an immutable historical snapshot; null revision means current. |
| `dataset_evidence(snapshot, output_id)` | Return one output row's typed derivation and source IDs. |
| `dataset_exclusions(snapshot)` | Return ordered excluded-source decisions. |

All IDs and `plan_name` are nonempty UTF-8 strings of at most 128
bytes. Source and query IDs have separate namespaces; row IDs are
unique per source and batch IDs unique per source. A source row is
an explicit row ID plus finite map payload, with no reserved
`source_id` column. Insertion order determines row order; correction
keeps its row position. Output row IDs are length-delimited paths
through source, join, or group operations, so distinct join matches
remain distinct and group keys use definite typed values rather than
display text. A persistent derivation ID is SHA-256 of query ID,
plan digest, source revision, operator path, and ordered input/output
row IDs. It is a stable evidence label, not a VM pointer.

A query plan is a named pure Lana function accepting source datasets
in the declared order and returning materialized rows. It may use
existing dataset operators and pure callbacks, but no I/O,
observation, mutation, stochastic, task, or unstable external read.
`calculation_version` is an explicit nonempty string. The compiled
plan digest is stored; reusing the version with a changed digest or
source list conflicts. Identical registration is idempotent. A new
version triggers full recomputation and atomic replacement.
The plan digest is lowercase SHA-256 over the exact verified linked
LABC chunk bytes followed by a u32 little-endian UTF-8 byte length
and `plan_name` bytes. This deliberately includes the plan's full
compiled program and imports: changing any of them requires a new
calculation version, even if the named function's result would
remain equal. After store reopen, saved snapshots remain readable
but executable query plans are unbound. The application must call
`dataset_query` again with the same ID, source list, name, version,
and digest to bind the current pure function; this matching bind
is read-only and idempotent. An update affecting any unbound query
fails `LANA_ERR_CONFLICT` before changing its source. A changed
plan or version must be explicitly registered and fully recomputed
before updates resume.
Persistent cells are definite ordinary values or explicit immutable
Information snapshots; a live root is unsupported and storing a row
does not implicitly observe, sample, resolve, or subscribe to it.
Captured finite Possibility, Distribution, and finite Joint cells use
`dataset_information_row_v1`: canonical tagged `value`, batch-local
`dependencies` and `relationships` bindings, per-value `evidence`, and a
complete topologically ordered `nodes` array. Bindings map canonical decimal
labels to canonical JSON `[source_id,batch_id,"dependency"|"joint",label]`
identities assigned at their first successful publication. The runtime retains
shared bindings across subsequent batches in that store session. Retrying a
batch compares its canonical values, evidence, and within-batch relationships,
then restores its original bindings; a conflicting existing binding fails.
Reconstructed independent laws in a new batch acquire new identities even if
their marginal values match. A query uses one fresh decoding context across
all source rows and sources, preserving aligned alternatives for each saved
identity. Historical result labels remain evidence, not live subscriptions.
Aligned dataset alternatives retain their order, including repeated outcomes
from distinct worlds; equal outcomes do not merge their probability mass or
erase their alignment. Definite rows retain their original encoding and retry digests.
Evidence stores each value's optional node index and container children in
canonical field order; nodes preserve revision, kind, operation, earlier input
indices, label, function, line, exactness, details, outcome, and reason. No
process-local derivation identity, pointer, executable closure, or autodiff
handle is serialized. Reload assigns fresh VM-local evidence identities.

At evaluation, every operator sees one global store revision.
Unresolved filter predicates or sort/group/join keys return
`LANA_ERR_UNRESOLVED_VALUE`; wrong definite predicate or key type
returns `LANA_ERR_TYPE`. `map` retains mapped Information, and
projection, sort, limit, group, and join preserve cell forms and
derivations. Join keeps the current right-overwrites-left column
rule, while evidence names both source rows and any dropped cell.
`aggregate` accepts `["count"]` or
`["sum"|"mean"|"min"|"max", column]` and follows the Core rules in
`docs/papers/semantics.md`. Unsupported unrelated uncertainty fails;
there is no hidden independence assumption.

A snapshot is a definite map with exactly `{schema_version:1,
format:"dataset_snapshot_v1", query_id, source_revision,
calculation_version, plan_digest, rows, row_ids, evidence,
exclusions}`. Rows and row IDs align by index. Each evidence entry
has `{output_id, operation, input_ids, cell_derivations, decisions}`.
`evidence` is a map with `nodes` and `rows`: `nodes` is the complete
topologically ordered derivation DAG, and `rows` is the result-aligned
array of those evidence entries. A row entry also has `derivation_id`.
Each node stores its stable ID, operation, ordered input derivation and
row IDs, output row ID, operator path, kind, exactness, outcome, reason,
label, and details. An exclusion entry has `{source_id, row_id,
operation, reason, input_derivation, predicate_derivation,
predicate_value}`; the predicate fields are null when absent. A joined
row exclusion lists each contributing source row in left-to-right
order. Lists use evaluation order and all referenced IDs resolve.
Filter-false, limit-excluded, and no-matching-key are distinct reasons.
`dataset_evidence` and `dataset_exclusions` describe this saved
revision; plan `explain` still describes the plan.
`dataset_snapshot` interprets a numeric revision as a global committed store
revision and returns the latest snapshot visible at that revision; null uses
the current revision. `dataset_evidence` returns the selected row evidence
fields plus `nodes` (its reachable topological derivation subgraph) and
`source_rows` (ordered `{source_id, row_id}` records). An unknown output ID
returns `LANA_ERR_NOT_FOUND`. `dataset_exclusions` returns the saved ordered
exclusion entries. These read calls validate the complete typed snapshot
without rebinding or running the query plan.

`changes` is an ordered array of exact maps: `{op:"add",id,row}`,
`{op:"correct",id,row}`, or `{op:"delete",id}`. `row` is a finite
definite map in this implemented slice. The call takes the same exact
integer numeric revision returned by `store_current_revision` and returns
the newly committed numeric revision. Revisions above the exact binary64
integer range are rejected. The implementation reruns every bound dependent
query against the candidate source revision and commits source, snapshots,
query records, and batch receipt together. An unbound dependent query
conflicts before staging.

The full contract for `changes` is an ordered list of `add(id,row)`,
`correct(id,row)`, or `delete(id)` for one source. Add requires an
absent ID; correction/deletion require an existing ID; a batch
cannot mention one row ID twice. A completed identical batch-ID
retry returns its prior receipt before expected-revision checking.
A changed payload under that ID conflicts. Otherwise
`expected_revision` must equal the global committed revision.
One successful batch recomputes every query naming the source
and publishes source plus all results in one store commit;
even a source without dependents advances its revision.
No background update runs while Lana is stopped.

Wrong call, row, or descriptor returns `LANA_ERR_TYPE` or
`LANA_ERR_INVALID_PARAMETERS`. Unknown source, query, output,
or revision returns `LANA_ERR_NOT_FOUND`. Stale revision,
conflicting ID, or changed same-version plan digest returns
`LANA_ERR_CONFLICT`. Unsupported Information combination returns
`LANA_ERR_UNSUPPORTED_OPERATION`; compacted history returns
`LANA_ERR_COMPACTED_HISTORY`. Resource exhaustion returns
`LANA_ERR_LIMIT` or `LANA_ERR_OOM`; store failure returns
`LANA_ERR_IO`. Failed registration, evaluation, update, or
reload exposes no partial source or query revision. The calls
use existing source syntax and `HOST_CALL`, with IDs specified
in `BYTECODE.md`; old source and bytecode behavior is unchanged.

## Bounded rules and tree calls

`std/rules` provides `learn(task, train, holdout, options)`,
`predict(rule, features)`, `save(store, task_id, expected_revision,
report)`, `add_counterexample(store, task_id, expected_revision,
example, new_holdout)`, `inspect(store, task_id, revision)`, and
`rollback(store, task_id, expected_revision, prior_revision)`.
`std/trees` provides `fit(task, train, holdout, options)`,
`predict(model, features)`, `explain(model, features)`,
`save(store, id, expected_revision, model)`, and
`load(store, id, revision)`. These are ordinary calls; no new Lana
grammar or opcode is required. Prediction never executes an action.

Every example is `{id, features, target}` with a unique nonempty
UTF-8 ID of at most 128 bytes. Task feature schema is ordered and
each entry is exactly `{name,kind,nullable,categories}` with kind
`number`, `boolean`, or `category`. `categories` is 1–64 ordered
distinct strings for a category feature and empty otherwise.
Feature values are finite numbers, Booleans, declared
category strings, or explicit null only where nullable. Targets are
definite Boolean or string labels for classification, finite numbers
for regression. Unknown fields, wrong types, non-finite numbers,
duplicate IDs, unseen categories, or overlapping training/holdout
IDs fail validation. Caps are 10,000 rows per input, 64 features,
and 128 labels. Empty training fails. A holdout has at least 20 rows
and one row per declared classification label before `validated`.

Every fit/learn result is schema-1
`{schema_version, status, task, options, train, holdout,
model_or_rule, training_report, validation_report,
mistakes, source_ids, calculation_version, limits_used}`.
`task`, `options`, `train`, and `holdout` are the fully validated
inputs in their original order, so `rules.save` and `trees.save`
can preserve them without reading a mutable caller value.
`training_report` is exactly
`{training_ids,mistakes,limits_used,search_status,rejections}`;
`validation_report` is exactly `{holdout_trace,metrics}`.
The trace and metric shapes are the saved `learned_task_v1` shapes
below. `source_ids` is training IDs followed by holdout IDs.
The outer `mistakes` and `limits_used` equal their training-report
copies. `search_status` is null and `rejections` empty for trees.
Saving derives a version's `report` from these fields without
rerunning fitting or changing the trace.
Status is `candidate`, `validated`,
`insufficient_evidence`, or `limit_exhausted`. Only
`validated` may replace an active version. Symbolic results also
have independent `search_status` of `exact_found`,
`exhausted`, or `limit_exhausted`. Reports include holdout
count, accuracy for classification or MAE/RMSE for regression,
per-label counts, prediction trace, data IDs, and calculation version.
No holdout target can influence candidate choice.
When the declared rule-search cap is reached, `status` and
`search_status` are `limit_exhausted`, `model_or_rule` is null,
and the report gives examined counts. This result cannot be saved
as an active version. Enclosing VM resource exhaustion is an error
with no result.

A rule task is exactly
`{id,feature_schema,target_labels,allowed,known_facts}`.
`target_labels` is exactly `[false,true]`; `known_facts` is an
ordered feature-value map or empty. Its options are exactly
`{max_candidates,max_predicate_visits}` with defaults 100,000
and 5,000,000 and positive integer values no higher than those
defaults. The ordered `allowed` operators are chosen from `eq`, `ne`,
`lt`, `le`, `gt`, `ge`, `and`, `or`, `not`, with at
least one comparison. `known_facts` optionally supplies definite
feature values attached to each example; a conflict fails.
Rule truth tables, bounded form, thresholds, and enumeration order
are specified in `docs/papers/semantics.md`. A candidate needing a null
feature records missing-row IDs and is discarded.
New learning records `calculation_version: "rules-v2"`. Existing
`rules-v1` reports retain their original enumeration and are accepted only
after replay against that version; corrections use the current version.
Search defaults to 100,000 candidate evaluations and 5,000,000 predicate visits;
options may only lower these caps. `predict` returns a Boolean,
matched clauses, input features, and active version, or
`unsupported` for a needed unknown/missing feature. Complete search
without zero training error is `exhausted`, not proof that no rule
exists. Symbolic `validated` additionally requires holdout
accuracy at least `0.90`.

`rules.save` stores task schema, search options, ordered training
and counterexample IDs/examples, untouched holdout IDs/examples,
candidate syntax, rejection summary, validation report, active
version, and parent version. It activates a first rule only if
validated. `add_counterexample` requires a new example ID and a
new holdout whose IDs have never appeared in training or any prior
validation. It reruns bounded search and commits a new version
atomically. An imperfect or insufficient replacement remains
inactive while the old active rule serves predictions. Exact retry
of example and holdout is idempotent; changed content conflicts.
`rollback` points to a prior validated version in one commit,
preserving later history. Stale revision, unknown version,
malformed record, or failed save leaves the active pointer.
After uncertain I/O, inspect the committed version before retry.
The `report` argument to `rules.save` is the complete `learn` result
above. `trees.save` likewise accepts the complete `fit` result as
its `model` argument; passing only the inner model is invalid.

Tree tasks are exactly `{id,feature_schema,problem,labels}`.
`problem` is `classification` or `regression`; `labels` is an
ordered array of 2–128 distinct Boolean/string labels for
classification and empty for regression. Tree options are exactly
`{family,seed,max_depth,min_leaf,ensemble_size}`:
`family` is `tree`, `forest`, or `boosted`; seed is unsigned u64,
depth is 1–16, minimum leaf 1–100, and ensemble size 1–100.
Defaults are `tree`, zero, six, two, and one for a tree
or 25 for an ensemble, respectively. For `tree`, ensemble size
must be one after defaulting.
CART, forest, and boosted objectives
and tie rules are in `docs/papers/semantics.md`. Multiclass boosting
returns `LANA_ERR_UNSUPPORTED_OPERATION`. Tree prediction
rejects unknown features/wrong kinds, uses stored missing branch,
and returns per-label forest vote fraction, one-tree leaf proportion,
or binary boosted logistic probability with
`calibration:"uncalibrated"`. `explain` lists every traversed
split/leaf; an ensemble lists each tree's contribution and
aggregation. Tree fit is `validated` only with the common
holdout sufficiency, otherwise `insufficient_evidence`.

The saved task state is canonical JSON with exactly
`{schema_version,format,kind,task_id,versions,active_version,receipts,digest}`.
It uses schema 1, format `learned_task_v1`, and kind `rule` or `tree`.
`versions` is an ascending array of immutable records, each exactly
`{version,parent_version,task,options,train,holdout,model,report}`.
Versions are one-based decimal u64 strings; the first parent is null
and later parents name an earlier version. `train` and `holdout` retain
complete ordered examples, not IDs alone. `active_version` is null or
names a validated version. Rollback changes only this pointer in one
new store revision. `receipts` is empty for trees; rule receipts are
sorted by counterexample ID and exactly
`{counterexample_id,payload_digest,version}`, where the payload digest
covers the canonical example and new holdout. Equal retries return
the recorded version; changed payload conflicts.

A rule `model` is exactly `{clauses}`: one or two ordered clauses,
each one to three ordered atoms. An atom is exactly
`{feature,operator,constant,negated}`; `constant` uses the canonical
tagged scalar encoding above. The outer list means OR, each clause
means AND, and `negated` applies to one comparison. This AST is the
saved rule; its display syntax is derived from it and never parsed
on load. A tree `model` is exactly
`{family,problem,labels,feature_schema,seed,base_score,learning_rate,trees}`.
`feature_schema` retains the complete declared schema, including features
unused by any split, so raw-model prediction validates inputs even for a
single leaf. It must agree with the saved task's schema. Older complete
reports and stored models that omit this field remain readable after full
refit validation using their saved task schema; raw models without a task
or feature schema are rejected.
`family` is `tree`, `forest`, or `boosted`; `problem` is
`classification` or `regression`. `labels` is the ordered declared
classification labels or empty for regression; `seed` is decimal u64.
`base_score` and `learning_rate` are null except for boosted models,
where they are binary64 bit strings (learning rate is 0.1).
Each tree is exactly `{nodes}` with root index zero and nodes in
preorder. A split is exactly
`{kind:"split",feature,operator,constant,missing_left,left,right,gain}`;
`operator` is `le` for numeric thresholds or `eq` for Boolean/category
values, `constant` is a tagged scalar, `gain` is binary64 bits,
and child indices are JSON integers greater than the parent index.
A leaf is exactly `{kind:"leaf",count,label_counts,value}`;
`count` and aligned `label_counts` are nonnegative JSON integers.
Classification tree/forest leaves have null `value` and nonempty
`label_counts`; regression and boosted leaves have empty
`label_counts` and finite binary64 `value` bits. Every node must be
reachable exactly once; duplicate, cyclic, or out-of-range edges fail.

Each version `report` is exactly
`{status,training_ids,holdout_trace,metrics,mistakes,calculation_version,limits_used,search_status,rejections}`.
`training_ids` and `mistakes` are ordered ID arrays.
`holdout_trace` aligns with `holdout` and each entry is exactly
`{id,target,prediction}`. Classification `metrics` is
`{count,correct,accuracy,per_label}`; regression `metrics` is
`{count,mae,rmse}`. `per_label` follows declared label order and
each entry is `{label,count,correct}`. `limits_used` is exactly
`{candidate_evaluations,predicate_visits,split_evaluations}`,
with unused counts zero. `search_status` is null for trees.
`rejections` is empty for trees and an ordered list of
`{candidate,reason,representative_id}` for rules; `candidate` is
the rule AST. All counts are bounded nonnegative JSON integers;
Missing-feature rejections contain one entry per affected training row,
with that row ID as `representative_id`. A `contradictory_clause` rejection
uses its own `candidate-N` ID; an `equivalent_truth_vector` rejection names
the first equivalent candidate. Reload replays the bounded search and checks
the complete report, including candidate ordering, rejections, and counts.
finite real metrics use binary64 bit strings, or null when no
metric is defined. No unknown fields are accepted.

Canonical JSON uses the sorted-key, UTF-8, no-whitespace rules above;
real values use 16-digit binary64 bit strings, versions and seed use
decimal strings, and the file ends with one LF. `digest` is lowercase
SHA-256 of the canonical JSON object excluding only `digest` and its
final LF. Loading verifies the digest, schema, complete examples,
report trace and metrics, rule bounds or tree node reachability,
finite values, references, and active validated status. Saves use
the existing store's atomic expected-revision commit and historical
reads. Numerical and work limits are in `VM.md`. Acceptance covers unsupplied-rule
discovery, correction, inactive/active replacement, reload and
rollback, hand-calculated tree/forest/boosted predictions,
missing/category branches, seeded replay, corrupt save,
insufficient holdout, resource exhaustion, and no action from a
prediction.

## Walk-forward evaluation

`std/evaluation.walk_forward(examples, trainer, options)` is a pure,
read-only evaluation call. An example is exactly `{id, observed_at,
target_at, features, target}`, with unique ID and finite integer UTC
times satisfying `observed_at <= target_at`. Input order is
ascending by observed time then ID. Each feature is
`{value, available_at}` with a definite JSON-compatible value and must be available by that row's
observation time. `trainer` is exactly
`{fit, predict, kind, labels, calculation_version}`: two named pure
function references, `kind` equal to `classification` or
`regression`, an ordered set of distinct definite Boolean/string
labels for classification (empty for regression), and a nonempty
calculation-version string. The source call supplies this trainer as an
inline map literal so the compiler can verify both named callbacks are
pure; the runtime validates the constructed map again. Every classification target must occur
in `labels`. `fit`
receives ordered training rows, internal validation rows, and the
fold seed as a canonical decimal u64 string and returns a versioned model map; `predict` receives that
model and one feature map. Classification prediction is exactly
`{label, probabilities}`, with a declared label and either null or
a map containing every declared label exactly once with finite
probabilities in [0,1] summing to one within `1e-12`. When present,
`label` is the maximum-probability label, breaking ties by declared
order. Regression prediction is exactly `{value}` with a finite
number. Both functions reject I/O,
observation, unseeded stochastic behavior, task, mutation, and
external effects. The evaluator does not infer a prediction API
from the model's type.

The model is exactly `{calculation_version, payload}`, where the
version matches the trainer and `payload` is an acyclic definite
value accepted by the tagged-value encoding above. Live roots,
handles, class references, callbacks, and non-finite numbers fail.
The evaluator computes `model_digest` as lowercase SHA-256 hex of
the canonical tagged JSON bytes of that model, without a trailing
newline. The caller cannot supply a digest. A replay with the
same fold inputs, trainer, seed, and calculation version must
produce the same digest and prediction trace.

Options `{initial_train, test_size, step_size, gap, seed}` default
to 100, 20, 20, 0, and 0. The first three are positive integers,
gap is nonnegative, and seed is unsigned u64: source accepts a nonnegative
exact integer Number through `2^53-1` or a canonical decimal string for
the full u64 range. Fold reports also use decimal seed strings. The fold and leakage
rule is in `docs/papers/semantics.md`. `seed + fold_index` and all
index arithmetic are checked for overflow before fitting. Eligible fitting rows are split
with `validation_count = max(20, ceil(eligible_count / 5))`;
there must be at least one remaining training row. Return
`{schema_version:1,status,folds,aggregate,repeated_test_ids,incomplete_tail_count,
calculation_version}`. A fold names exact train, validation, and
test IDs, time bounds, seed, model digest, predictions, observed
targets, and metrics. Classification fold and aggregate metrics
are exactly `{count,correct,accuracy,log_loss,log_loss_status}`.
`log_loss_status` is `finite`, `unavailable` when any prediction
lacks probabilities, or `infinite` when all maps exist but a
true-label probability is zero. `log_loss` is null unless status
is `finite`. Regression metrics are exactly `{count,mae,rmse}`.
Aggregate values pool every scored prediction in fold order;
overlapping test IDs are counted again and appear once in
`repeated_test_ids` at first repeat occurrence.
A fresh model is fit at every fold.
Insufficient complete folds or eligible training/validation rows
return `status:"insufficient_evidence"` with no score.
At most 10,000 examples and 100 folds are accepted; failed
validation or fitting returns no partial report. The result is
not saved unless the caller explicitly uses the store. Tests
include a future-feature leakage trap, gap, repeated test IDs,
deterministic fresh refits, and separate-process replay.

## Local SQLite and document inputs

`dataset_sqlite(path, sql, parameters, schema)` opens a local
SQLite database read-only, prepares exactly one read-only
row-returning statement, and evaluates it within one read
transaction. Reject writes, multiple statements, PRAGMA changes,
ATTACH, virtual-table side-effect calls, and user-defined
functions. Bind an ordered array of definite null, Boolean,
finite number, or string parameters; never interpolate values
into SQL text. Result order is SQLite's result order, so callers
use SQL `ORDER BY` for reproducible order.

The ordered schema entries are `{name, kind}` and match unique
result columns exactly. Kinds are `bool`, `number`, `string`,
`nullable_bool`, `nullable_number`, `nullable_string`, or
`information_json`. SQL NULL is missing only for nullable
columns, not Possibility. An integer outside `[-2^53,2^53]`,
non-finite real, BLOB, wrong kind, or invalid tagged Information
fails. `information_json` uses the canonical finite tagged-value
schema above and cannot carry a live root. One unique non-null
`id` string column (at most 128 UTF-8 bytes) identifies every
row. The result is `{rows, source_revision, evidence}`: `rows` is an
ordered array of maps keyed by the declared column names, and `evidence`
is a parallel array of maps `{path, sql_digest, parameter_digest,
source_revision, row_id, column_types}`. `column_types` is the ordered
array of declared kind names. An `information_json` cell is returned as
canonical tagged JSON text after validation; it is an immutable snapshot
encoding, not a live Information root. Source revision is SHA-256 of canonical column names,
schema, bound parameters, and complete ordered rows from that
transaction. Evidence names path, SQL digest, parameter digest,
source revision, row ID, and column types without storing
secrets or claiming that SQL rows define a joint law.
The revision preimage is the canonical JSON map
`{columns,parameters,rows,schema}` using the tagged-value encoding
and JSON byte rules above. `columns` is the ordered result-name
array, `schema` the ordered declared entries, `parameters` the
ordered bound tagged values, and `rows` the ordered arrays of
tagged cell values. Digests are lowercase SHA-256 hex. SQL and
parameter digests use the original UTF-8 SQL bytes and that same
canonical `parameters` array respectively.
Caps are 10,000 rows, 64 columns, 16 MiB SQL input, and 64 MiB
encoded output. Schema, duplicate ID, unsupported SQL/value,
lock, I/O, or limit failure returns no dataset. This is
unsupported on WASM and does not change one-row `adapter_fetch`.

`document_extract(path, format)` accepts only explicit `"text"`
or `"markdown"` UTF-8 files up to 16 MiB; invalid UTF-8 or NUL
fails. It returns schema-1 `{format, sha256, chunks, status}`.
Each chunk is `{text, start_line, end_line, start_byte,
end_byte, heading_path}` with one-based inclusive lines and
zero-based half-open original UTF-8 byte offsets. `heading_path`
is an ordered array of heading text from outermost to nearest
heading, empty for plain text or before any heading. Text splits
on blank lines. Markdown headings are one to six `#` followed
by a space; a new heading replaces the prior heading at its
level and removes deeper levels. Blank lines end a logical
chunk, while fenced code including fence lines stays one
logical chunk before the size cap. CRLF is one line break, but offsets refer
to original bytes. Content inside a chunk is never rewritten.
Split chunks above 4,096 bytes at the last complete UTF-8
character before the cap with contiguous offsets. Empty input
has no chunks and `status:"empty"`; nonempty valid input is
`status:"exact_text"`, which makes no truth claim. I/O,
decoding, size, or offset error returns no partial chunks.
No PDF, OCR, office, HTML, network, or model parser is implied.
The native runtime implements this call at host ID 219. The first
implementation caps extraction at 100,000 chunks within the VM's
resource budget; exceeding that cap returns `LANA_ERR_LIMIT` without a
partial result. WASM returns `LANA_ERR_UNSUPPORTED_OPERATION` because
it has no local file boundary for this call.
Acceptance compares every chunk with its exact source byte
slice across mixed line endings, Unicode, headings, fences,
long chunks, empty input, and invalid UTF-8.

## Hosted source packages

Hosted packages are source-only public GitHub repositories named by lowercase
`owner/repo@X.Y.Z`, with exact stable decimal versions. The tag is
`lana-vX.Y.Z`. The release contains `repo-X.Y.Z-lana.tar.gz` and
`SHA256SUMS` with exactly one line: 64 lowercase SHA-256 hex digits,
two ASCII spaces, the asset filename, and LF. Ranges, floating tags,
private repositories, credentials, registry search, and prereleases are
unsupported. This is separate from Lana's own binary release workflow.

`lana package pack DIRECTORY -o ARCHIVE` validates a package project and
returns its archive SHA-256. The archive has one top-level `repo-X.Y.Z/`
directory containing `lana.toml`, `src/`, and optionally `tests/`. The
manifest declares schema 1, `name = "repo"`, the exact version, and an
entry under `src/`. Published manifests may list exact hosted dependencies
under `[hosted_dependencies]` as `alias = "owner/repo@X.Y.Z"`. They may not
contain local or absolute path dependencies or build scripts. Pack includes
only regular files in that tree, rejects symlinks, absolute or `..` paths,
duplicate archive names, and out-of-tree files, and writes deterministic
tar.gz bytes: sorted entry names, normalized user/group IDs, permissions,
entry timestamps, and zero gzip timestamp. Repeated pack with the same
CLI version and identical input bytes produces identical archive bytes.
The tar stream uses POSIX ustar only, with explicit root, `src/`,
and optional `tests/` directory entries and all regular files sorted
by UTF-8 archive path bytes. Directory mode is `0755`, file mode
`0644`, UID/GID and mtime are zero, owner/group names are empty,
and there are exactly two zero blocks after the final entry.
Unrepresentable ustar paths fail. Gzip has no optional header fields,
mtime zero, and OS byte 255. The compressor settings are fixed for
one CLI version and recorded in its pack fixture; an archive from
another compressor need only pass the validated tar/gzip reader.
Pack does not publish.

The schema-1 writer uses flate2's Rust backend at compression level 6;
`packages::tests::deterministic_archive_and_strict_reader` pins its fixture
digest. Paths are portable UTF-8: no backslash, colon, control character,
empty/dot component, or trailing dot/space. Case-insensitive duplicate paths
are rejected. The reader accepts regular ustar files and directories only,
requires explicit parents, and rejects extra gzip members or trailing tar
data. Expanded size includes tar headers and padding; archive bookkeeping
is also capped at 2,001 entries.

`lana package add owner/repo@X.Y.Z` fetches that tag's asset and checksum
over HTTPS, verifies the compressed archive SHA-256 before extraction, then
checks all paths and manifest identity/version. Each package is limited to
1,000 files, 64 MiB compressed, and 256 MiB expanded. Resolve the full
dependency closure, at most 64 packages, rejecting cycles and different
versions of one `owner/repo`. No package source executes during add.

For the root project, `lana.lock` is the sole hosted-dependency authority;
`lana.toml` continues to own local-path dependencies. A schema-1 lock entry
records identity, version, tag, asset name, SHA-256, direct/transitive status,
and sorted direct dependencies. Packages live in
`.lana/packages/<sha256>/`. Replace the lockfile only after the complete
closure validates. An unused cache directory left by failure grants no
import authority. Repeating the identical add leaves the lock and cache
unchanged. Changed digest, release asset, manifest, missing dependency,
unsafe path, download, or write failure leaves the previous lock usable.
Post-replacement directory-sync failure reports uncertain durability and
requires a fresh-process lock reload before retry.

Local-only projects retain their existing build-generated lock format until
the first explicit hosted add. Subsequent builds preserve the hosted JSON
lock and include its bytes in the compiled-cache key along with the existing
local source/dependency plan. Add uses `curl` with bounded HTTPS-only redirects
and transfer time/size limits. HTTP origin overrides exist only in explicit
fixture builds. Cache archives are named `archive.tar.gz`; extraction retains
the archive's top-level directory. Symlinks in cache storage are rejected.

`lana.lock` is UTF-8 canonical JSON with exactly the top-level keys
`direct`, `packages`, and `schema_version` (integer 1). `direct` is a
sorted array of exact `owner/repo@X.Y.Z` strings. `packages` is sorted by
`owner/repo` and each entry is exactly
`{identity,version,tag,asset,sha256,direct,dependencies}`. `identity` is
`owner/repo`, `direct` is Boolean, and `dependencies` is a sorted array of
exact package strings. JSON object keys are sorted, there is no whitespace
or duplicate key, strings use UTF-8 with JSON escapes only where required,
and the file ends with one LF. A lock is invalid if a direct name is missing
from `packages`, closure references are missing, one identity has two
versions, or recorded names disagree with manifest dependencies.

The existing quoted import syntax resolves
`import "pkg/owner/repo/src/module.lana" as module;` only through the
matching exact lock entry and digest-verified local cache. Missing or
ambiguous entries and traversal fail. Existing imported-module restrictions
still apply. Build never fetches implicitly and rechecks cached digest.
The cache retains the original archive beside extracted source so the digest
can be checked on every build; extracted file paths and bytes must also match
the verified archive before use.
An explicit repository choice and the lock digest define trust; a checksum
from that same release checks corruption, not third-party code safety.

A package repository may publish through a protected `lana-vX.Y.Z` tag.
Its workflow checks identity, checksum, clean extraction, build, and tests
before a final release job with scoped `contents: write` permission.
It refuses to overwrite a tag or release with different assets. Publication
requires separate authorization. Acceptance includes local HTTP release
fixtures for success, bad digest, traversal, cycle, version conflict,
partial download, and changed-asset retry, plus a public-release download
and digest smoke test. Existing local-path project fixtures still pass.
These commands and source imports add no LABC opcode or bytecode version.

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
The compiler verifies complete output before replacing its destination.
Parse, verifier, cancellation, limit, and allocation failures report
`LANA_ERR_PARSE`, the applicable verifier error, `LANA_ERR_CANCELLED`,
`LANA_ERR_LIMIT`, or `LANA_ERR_OOM` with a source span or bytecode offset
where available. A pre-replacement `LANA_ERR_IO` leaves old output bytes
or absence unchanged. A post-replacement parent-directory sync failure
reports `LANA_ERR_IO`, `durability: "uncertain"`, and the path; a fresh
process must verify the surviving file before retry. A failed VM call
returns no partial public Value or committed derivation revision.

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
costs fail validation. See the [decision operations](../PRODUCT_SURFACE.md#decisions).

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

The Rust signed relationship-claim record has `kind: "relationship_claim"`,
`id: "relationship_claim/<claim_id>/<version>"`, and the same schema-1 envelope.
Its `payload` contains the fields signed by the existing LCP1 encoding:
`schema_version`, `claim_id`, `version`, `subject`, `scope`, `issuer`,
`issuer_key_id`, `authority_policy_version`, `origin`, `relationship`,
`parameters_hex`, `validity_start`, `validity_end`, and `lifecycle`.
`claim_id` and the validity bounds are decimal strings so JSON preserves all
64 bits; `relationship` and `lifecycle` use lowercase snake case. The record
adds lowercase hexadecimal `payload_digest` and `signature` fields. Parsing
this record does not grant authority: trust validation still checks the LCP1
payload, digest, Ed25519 signature, issuer, scope, lifecycle, and validity.
The LCP1 signed bytes and relationship-resolution encoding are unchanged.

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

Shared handles have process-local identity and a total commit revision. Read,
observe, and admin authority are separate revocable capabilities; a handle or
admin token alone grants no read or observation authority. Observations carry
integer effective time and definite evidence. A transaction replays them in
`(effective_time, sequence)` order and atomically publishes one complete
revision. Equal-time equal evidence is idempotent; equal-time conflicting
evidence fails. Late evidence replays later versions. Failed replay,
revocation, cancellation, or resource failure publishes nothing. A waiter for
`revision > after_revision` rechecks authority before returning an immutable
snapshot; samples never enter shared observation history.

A project is rooted by schema-1 `lana.toml`; `lana.lock` records the content
identity used by the build cache. `lana new`, `build`, `run`, `test`, `fmt`,
and `doc` use stable nonzero failure exits. Formatting is idempotent and
`fmt --check` performs no writes. `lana lsp` speaks JSON-RPC over standard I/O;
`lana debug` uses the same source-line mapping emitted by the compiler.


## Low-level HTTP hosts

`http_get(url, headers, timeout_ms)` and
`http_post(url, body, headers, timeout_ms)` require a live `net` read capability.
They return `Result<Information<Response>, string>`; transport/protocol failures
return an error result, while type, capability, memory and instruction failures
remain runtime errors. Response fields are `status`, `body`, `headers`, and
`trailers`. The body remains UTF-8 text with replacement for invalid sequences.
Header/trailer maps use lowercase names and ordered arrays of field values;
field octets map reversibly to Unicode U+0000..U+00FF. Trailers are separate
from initial headers and cannot replace framing, authentication, or body metadata.

Request headers have string values. Boolean `verify` controls certificate
verification and is never sent as a header; verification defaults to enabled.
Callers cannot set Host, Content-Length, Transfer-Encoding, Connection, Upgrade,
Trailer, TE, Proxy-Connection, or Expect. Invalid names, controls, and CR/LF
injection are rejected before opening a socket. URL credentials and malformed
ports/percent escapes are rejected; fragments are omitted from the request.

Responses follow HTTP/1.x framing: fixed lengths, close-delimited bodies, or
chunked bodies with extensions and trailers. Up to 16 informational responses
may precede the final response; protocol upgrades are unsupported. Bodyless
204/304 responses complete without waiting for connection close. Conflicting
lengths, simultaneous transfer encoding and length, unsupported transfer coding,
malformed status/fields/chunks, and premature EOF fail without publishing a
partial response. Header sections together are limited to 64 KiB; chunk-size
lines to 8 KiB. Declared bodies reserve within the VM heap before reading;
received bytes consume the instruction budget. Positive timeouts are in
milliseconds (rounded down, minimum one); nonpositive values select 5000 ms.
Nonfinite values are invalid. The timeout bounds each connect/write operation
and the entire response-read phase; synchronous platform DNS resolution is not
covered. Successful framed responses do not require TLS close-notify.


## Workspace editor queries

The language server reports the installed Lana version and uses UTF-16 LSP
positions. Compiler symbol queries preserve canonical source paths, declaration
identities, and lexer spans across modules and lexical scopes. Definition,
references, hover, completion, prepareRename, and rename use those symbols.
Open-buffer text overrides disk sources, including imported buffers and new files
whose parent directory exists. A query never writes an editor buffer or source.

Workspace rename checks every Lana source in the initialized workspace folders
and all open documents. It excludes build/dependency directories (`target`,
`.git`, `.lana`, `node_modules`, `.venv`) from source discovery; imported dependency
symbols remain navigable. Without workspace folders, open documents define the
editable scope. A proposed name must be a nonreserved identifier and must not
collide with another indexed declaration. The server rechecks all editable
sources with the complete proposed overlay before returning any edits. Invalid
or incomplete analysis, missing source spans, or any required dependency edit
fails the request without a partial WorkspaceEdit. Workspaces above 4096 source
files fail explicitly instead of returning a truncated index.
