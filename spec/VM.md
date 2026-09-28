# Runtime architecture

## Derivation ownership

Each VM owns immutable derivation nodes and a monotonic local sequence. Nodes
use deterministic task lineage plus local sequence as IDs. Values carry an
optional node reference; mathematical equality ignores it. Task transfer clones
the DAG with memoization while preserving origin IDs. Successful observation
increments `revision` after refinement validates. Failure records may be cited
by structured errors but are never published as values.

## Information runtime boundary

The VM owns all joint names, domain descriptors, independent marginals,
finite weighted rows, projections, and sampled arrays.
Joint values are immutable after construction; projection and conditioning
allocate new VM-owned views and never mutate the source. Task cloning deep
copies the named joint graph, so no joint pointer crosses VM heaps. Sampling
uses the child VM's deterministic RNG stream. A finite correlated sample draws/
one row and never samples columns independently. Resolution is exact and fails
with `LANA_ERR_UNRESOLVED_VALUE` unless the joint support is a singleton. Duplicate
names, unknown projection names, impossible evidence, invalid descriptors, and
unsupported exact operations fail before a partial result is exposed. Clone
memoization preserves shared joint graphs while preventing cross-VM pointers.

The VM represents finite possibilities with dependency identifiers. Pure
arithmetic, comparisons, and function bodies lift over those alternatives.
`PATH_SPLIT` snapshots the active frames for an unresolved Boolean and
executes both branches; `PATH_JOIN` restores and merges them into guarded
path values. Nested splits use a bounded execution stack. History-bearing
register merges, incompatible dependency joins, and unresolved loops are
explicitly unsupported. Printing, host calls, task creation, sampling, and
observation are rejected while a split branch is active, preventing partial or
duplicate effects. Successful observation increments the VM observation log
counter only after refinement succeeds.

Ordinary Information values carry a process-local reactive node. Root
nodes retain finite support and a dependency identity; binary, comparison, and
unary nodes retain their pure inputs, operation, exactness, current value, and
immutable revision history. Observation first stages the replacement and every
affected pure result. The VM publishes all staged values under one revision only
after the full traversal succeeds; invalid evidence, allocation failure, or an
unsupported relationship leaves the old revision intact. Different dependency
identities never imply independence and cannot be combined by ordinary lifted
operators. Containers and function values retain node pointers, while
serialization and cross-VM transfer materialize snapshots.

Claims retain the proposition separately from exactness, tolerance, and source
validity. Planned effects retain a stable process-local identity, captured
payload, and linked receipts. Execution requires a definite payload and invokes
the configured executor at most once for each `(identity, revision)`; later
reads return the captured result. The collector traces all reactive inputs,
history values, Claim records, plans, and receipts, so unreachable cycles and
obsolete graphs remain reclaimable.

Shared Information owns an isolated storage VM, immutable observation
evidence, and an immutable linked commit history. Candidate commits are built
off-lock from a stable observation-array copy, then published only if the base
commit, capability epoch, and observation count still match. A process-global
atomic counter supplies total commit order. Capability revocation advances the
epoch and wakes waiters so they can fail without reading a value. Each Lana VM
retains every shared identity referenced by one of its capability values; task
cloning retains the identity but never copies the live graph.

The collector classifies allocations as young, old, or stable shared. Minor
collections trace exact roots and remembered old objects, reclaim unreachable
young allocations, and promote survivors. Write barriers record old-to-young
edges and force young targets promoted before a stable shared owner can retain
them. Incremental marking drains a bounded worklist per step; the existing full
non-moving collection remains the severe-pressure and invariant fallback.

Collector scratch capacity is charged to the same checked heap. Ordinary
container pressure collection starts above three quarters of the memory limit;
class construction retains its earlier half-limit pressure check and synchronous
fallback. These scheduling thresholds do not raise either resource limit.

The Rust embedding result API returns `Result<RootedValue, LanaError>` because
root registration must obey the heap limit. A retained handle owns the
lifetime of reachable class storage beyond VM teardown; cloning the handle
preserves identity and shares its tracked root. Borrowed `Value` data and
internal graph edges do not create independent embedding roots. See
`docs/rust-embedding.md` for migration and the remaining collector limits.

`lana debug` installs an instruction hook before execution. Stops are keyed by
the deterministic source-line field stored in every LABC instruction and expose
the active function and frame count; step, continue, and quit never reinterpret
bytecode. The low-level trace uses the same instruction and line mapping.

The self-hosted compiler runs as ordinary verified bytecode with explicit
256 MiB memory and 50,000,000-instruction limits. `path_resolve` and the
effectful filesystem tooling calls are Rust host boundaries for OS facts and
atomic file publication; project policy remains ordinary Lana code. Lexing,
parsing, resolution, semantic IR lowering, LABC emission, import-cycle checks,
and call remapping execute in Lana. Clean builds assemble the checked textual
bootstrap artifact and require no Python runtime.

Compiler publication stages a completely emitted and verified chunk in an
owned sibling file, syncs the file, atomically replaces the requested output,
then syncs its parent directory. Parse, verifier, cancellation, memory, and
instruction-limit failures happen before replacement and report
`LANA_ERR_PARSE`, the applicable verifier error, `LANA_ERR_CANCELLED`,
`LANA_ERR_OOM`, or `LANA_ERR_LIMIT` with a source span or bytecode offset
where available. Pre-replacement I/O reports `LANA_ERR_IO` and preserves
the old output bytes or absence. A parent-directory sync failure after
replacement reports `LANA_ERR_IO` with
`durability: "uncertain"` and the destination path; a fresh process must
verify that path before retry. Internal allocation never publishes a
partial VM return Value, store revision, register update, or effect receipt.
Fault tests cover old and absent destinations at each stage and verify in
a new process that any surviving bytecode is complete.

## Finite information execution (three measures and conversions implemented)

The Rust VM evaluates entropy, conditional entropy, and mutual information
over declared finite joints in bits. It groups equal projected assignments,
charges enumeration against the task limits, and returns a number with a
derivation carrying selected names and the input revision. Invalid names,
unsupported laws, numerical failure, and exhausted budgets publish no result.
The Rust VM also supports explicit loss and assignment of finite weights,
validating complete unique candidate rows and recording the conversion in
the result derivation. The remaining operations below are pending implementation.

The Rust VM stores opaque immutable finite kernels. `identity_kernel` validates
a distinct definite domain and constructs its exact identity rows;
`compose_kernels` validates matching domains and multiplies normalized rows.
`kernel` invokes a named pure callback once per Cartesian tuple, validates
complete ordered rows, and stores normalized probabilities. `network` validates
finite root support, exact ordered parent domains, and acyclic dependencies,
then snapshots the root and kernels. `infer` materializes the bounded finite
law, applies exact named evidence, sums out other variables, and returns a
normalized finite named joint.

The Rust VM owns immutable finite-kernel and network values. A kernel
contains ordered definite input/output domains and complete normalized
rows, with no retained callback. A network contains one immutable root
joint, ordered child names/parent lists, and their kernel values.
Constructors validate all domains, row counts, normalization,
references, and acyclicity before publishing an opaque value.
Task transfer deep-copies these values with memoization; no VM-owned
pointer crosses heaps. Pure callbacks execute once per Cartesian tuple
under the active task's effect and instruction limits. A failed or
effectful callback publishes no kernel.

Before materializing any projected, product, coupled, or network law,
multiply cardinalities with overflow checks and charge all row
enumeration, callback evaluation, and BROJA optimizer work to the
active 50,000,000-instruction and 256 MiB memory budgets. Cancellation,
`LANA_ERR_LIMIT`, `LANA_ERR_OOM`, invalid row, unsupported exact law,
or numerical failure clears the public result. The BROJA solver certifies
cases where the original law meets the objective bound, both target-source
marginals factor exactly, or finite source couplings meet a checked convex
objective gap. For larger source domains it uses a bounded transport-dual
oracle; failure to certify within the budget returns `unconverged`.
BROJA may return
`converged` only with the certified marginal residual and objective
error bound in `papers/semantics.md`; otherwise it returns the
unconverged record with no component values. The same input and budget
produce the same row order, optimizer result, and diagnostic.

## Immutable Information capture

Host ID 220 implements `snapshot` with the existing deep-copy memo. Capture
reads each live input's current value, removes its live link, and preserves
the captured revision in an immutable derivation. Joint rows, finite weights,
Paths guards, and dependency labels survive the copy. Captured arrays and
maps reject writes at both opcode and host-call boundaries. Capture charges
VM work, retains heap limits, and rejects nesting beyond 64; a failed capture
publishes no result. This primitive is available before the object extension.

## Object runtime

The v6 descriptor/operand verifier, value construction/field reads, and pure
value instance/static methods are implemented. A static factory can construct
private fields, and an owning method can read them or call private methods.
Method calls check packed argument count, definite receiver identity, argument
types, visibility, and result type before publishing the return value. Frames
are charged against the VM memory budget. Frame ownership and instruction
ownership must agree; direct named/callback entry cannot acquire private access.

Before any instruction, the VM checks every method and its transitive call
graph, including unused methods. Declared effect masks must cover opcode and
host effects; unknown extension hosts require external-call permission. Runtime
checks retain every active member's effect promise across indirect callbacks.
Default bodies remain pure and initializers may mutate only their own candidate
fields or construct contained objects. Typed method arguments may carry live
Information. Pure value receiver maps preserve alternatives, weights, path guards,
and named Joint laws. Derived live results recompute transactionally with their
source revision. Additional mutable/live replay parameters fail explicitly.

Classes implement identity, defaults, checked initialization, public mutable
writes, and alias/cycle-preserving task transfer. Defaults run once and remain
pure; initializers may construct contained objects and assign their own fields.
Failure leaves the destination unchanged. Live Information fields become
snapshots on transfer; class writes do not advance Information revisions.
Class storage and registered container cycles are reclaimed during a task by tracing passes between
instructions, outside construction and transfer transactions. Tracing preserves
references held by frames, suspended computations, host callers, Information
graphs, and task results, including aliases and cycles. Collector scratch space
and traversal work use the task's memory and instruction budgets; exhaustion
fails before sweeping. Routine collection preserves progress and performs at
most 128 work units per safepoint. Full synchronous tracing remains the explicit,
severe-pressure, invariant and shutdown fallback. Mutation barriers invalidate
stale traces before sweeping. Registered immutable DAGs retire iteratively.
Unreachable class storage and returned call frames release their memory charges.
Weak references also break class cycles at teardown. Interface conversion retains the underlying
identity and dispatches by exact promised signature. Source declarations,
constructors, copied blueprints, typed methods and overloads compile to v6.
Defaults and initializers use resumable owned frames: debugger breakpoints and
steps reach their source lines without publishing an incomplete candidate.

Value construction checks runtime types and copies/freezes arrays, maps, and
ADTs using the existing snapshot mechanism. Scalars, STATE, STATE_DIST, nested
values, Tensors, Shapes, and explicitly captured finite Information are supported; live roots,
cycles, capabilities, executable payloads and unsupported payload types fail
before the destination changes. Tensor storage is already immutable and can be
shared. Shape fields reuse `tensor_shape_from_array` validation and freeze the
copied dimension array; shape-validation buffers use the existing VM heap budget.
Captured Information checks every alternative against its inner type; Joint
assignments require `map` or `Dynamic`. Construction traversals charge VM work
and reject nesting at 64. Equality checks every field before returning a result,
rejects unsupported field equality, retains existing array identity equality,
and bounds traversal to 100,000 nodes and depth 64. STATE_DIST nodes and already
frozen values may safely share immutable storage. Display exposes the nominal
name only. Explicit JSON export accepts public-field values with recursively
JSON-compatible payloads and rejects any private field. JSON loading yields
ordinary maps and cannot restore identity or private members.

The following rules describe the complete object runtime contract:

The Rust VM represents a value as an immutable descriptor reference plus
ordered deep field snapshots. A class object holds its descriptor,
task-local identity, field storage, and initialization state. Two
references inside one task may point to the same class object. A
private member is accessible only while the active frame belongs to
that descriptor. A copied method is compiled as a child-owned function;
there is no parent object or runtime superclass pointer.
`OO_AS_INTERFACE` creates a typed view over the same value snapshot
or class reference. The view grants only interface-promised methods,
with dynamic dispatch to the object's verified final method table.
It never grants field access or converts child to parent.

`VALUE_NEW` checks each positional argument type and deep immutability
before publishing the snapshot. Live Information and class references
are rejected, including nested or interface-wrapped references.
`OBJECT_NEW` allocates an unobservable candidate, evaluates each pure
default once, then calls the one declared initializer with the
supplied arguments. An immutable field can be assigned once; a
defaulted immutable field cannot be assigned again. A mutable field
may be assigned during initialization and afterward. The VM tracks
initialized fields and rejects `self` escape, instance calls, or
external use until all required fields are set. Failure frees the
candidate and exposes no identity, field write, or effect receipt.
Only successful construction publishes the object.

`OO_GET` and `OO_SET` check the receiver descriptor, field index,
visibility, initialization state, and declared type at runtime.
`OO_SET` requires a mutable field outside initialization and a definite receiver; it records
the mutation effect, not an Information observation. `OO_CALL` checks
receiver identity or value, arity, argument types, method visibility,
and effect allowance before dispatch. Interface dispatch checks that
the concrete descriptor explicitly implements the interface and that
the final method signature and inferred effects satisfy its promise.
`OO_STATIC_CALL` has no receiver or type-level mutable state.
An unresolved class reference or an unresolved execution branch
cannot call a mutating method or construct an object.

Task transfer clones reachable class graphs with a memo table, assigns
fresh identities in the receiving task, and preserves aliases and
cycles there. Values are copied as deep snapshots. A live Information
field is materialized as an immutable snapshot at the transfer
boundary. Unsupported transfer returns an error before publication.
Task-owned arenas retain class fields and interface references; value fields
are immutable snapshots. Direct class serialization is unsupported; bytecode
descriptors contain names, types, and function indices only.

The v6 verifier validates the descriptor and ownership rules in
`BYTECODE.md` before execution. Runtime checks remain mandatory for
dynamic receivers. Any malformed descriptor, type mismatch, illegal
private access, incomplete construction, effect violation, OOM,
cancellation, or unsupported lift clears the public result. Existing
v1-v5 chunks and their old object-free runtime behavior are unchanged.

## Dataset history runtime (six public calls implemented)

The `dataset_apply` VM boundary permits unresolved values only in the changes
argument; the source codec then requires explicit immutable capture of every
finite uncertain cell. It reuses the result snapshot's typed-value encoder,
retains complete source-cell evidence, and binds local labels to durable
source/batch identities. Batch receipts retain those bindings so retries after
restart can reuse a captured relationship in later mutations. Failed updates
do not install bindings. Each evaluation decodes all sources with one shared
relationship context, assigns fresh VM-local dependency/evidence IDs, freezes
nested containers, and rejects malformed evidence or incompatible aligned laws.
Definite source records and old receipts retain their original byte encoding.

`dataset_apply` validates all changes and affected query reruns before staging,
writes source, snapshots, query records, and batch receipt in one store revision, and returns the prior receipt
for an identical retry before checking the expected revision. A conflicting
retry, stale revision, failed rerun, or unbound dependent query publishes nothing.

The store owns named source rows, registered plan identity/digest,
typed snapshot records, persistent derivation nodes, batch receipts,
and historical revisions. A query captures one global store revision
before reading any source. All source rows and affected query reruns
are staged off the published revision and validated before one
atomic store commit. Registration computes its initial snapshot in
that commit. After restart, matching `dataset_query` calls must bind
the current verified chunk and pure function; historical reads need
no bind. A mutation touching an unbound dependent query fails
before staging. A change reruns every query naming the changed source;
row-level incremental execution is optional, but published rows,
order, typed values, evidence, and bytes must equal a clean full
rerun at that revision. Source insertion, left/right join, and
first-seen group order are stable. Correction keeps source position;
deletion removes obsolete output and evidence paths.

The named pure-plan runner enforces one shared 5,000,000-step dataset-work
budget across nested source materialization, callbacks, comparisons, and
aggregate cell visits. It also rejects more than 100,000 result rows before
returning a plan result. This also protects query registration and later updates.
The runtime's internal evaluation path reads every declared source record at
one captured store revision, validates canonical rows and unique row IDs,
and passes ordered source datasets to that runner. Evaluation publishes
nothing; registration commits a validated snapshot with its query record.
Named plans reject `array_push` even though older generic pure callbacks
permit it for local array construction; a plan cannot mutate an input array.
The evaluator also attaches ephemeral source-row derivations using
length-delimited source/row IDs. Filter, map, select, limit, sort, group,
aggregate, and join carry those row links; filter-false, limit-excluded,
and unmatched join sides collect ordered exclusion decisions. Source cells
inherit their source-row link, mapped cells without their own derivation
inherit the map-row link, and aggregate result cells carry an operation link.
Filter decisions retain the definite Boolean value. Named plans reject
non-map output rows and failed runs clear the decisions. The internal evaluator now
derives stable row paths from source IDs, joined paths, or canonical typed
group keys. A separate runtime pass assigns SHA-256 derivation IDs from the
query ID, exact plan digest, source revision, deterministic traversal path,
and ordered input/output row paths. It rejects duplicate or unattributed
output rows and caps the derivation graph at 100,000 nodes. The runtime can
encode the complete typed result and evidence as canonical snapshot bytes
and validate a reload in memory. `dataset_query` persists the initial snapshot
and query record in one store revision. Matching calls after reopen validate
and bind without a write; a changed plan or source list at the same calculation
version conflicts. `dataset_apply` reruns bound dependents at the candidate
revision and publishes every changed snapshot in one commit. Historical reads
follow the query record visible at the requested global revision, validate its
typed snapshot, and return compacted history explicitly. Row evidence is the
selected result row's reachable derivation subgraph and ordered source rows;
exclusions retain their saved order.

Saved `dataset_snapshot_v1` records use the canonical tagged-value
encoding in `SPEC.md`, including binary64 bit strings for numbers,
STATE components, and weights. The record contains plan and
calculation version, source/store revision, ordered rows, inclusion
and exclusion decisions, and a complete topologically ordered
derivation DAG. IDs and revision strings remain historical labels
after reload, not live subscriptions or pointers. Load validates
schema, UTF-8, unique IDs, references, acyclicity, finite laws,
probabilities, source/plan digest and revision, and canonical
re-encoding before publishing. Retained historical snapshots keep
their referenced source and derivation records; compacted history
returns its explicit error and never current rows as a substitute.
An I/O error during commit has uncertain outcome: reopen and inspect
the batch ID and revision before retrying.

Caps are 10,000 rows per source, 100,000 per result, 64 MiB per
encoded snapshot, 100,000 derivation nodes per snapshot, and 1,024
alternatives per finite cell. One full query rerun has a 5,000,000
dataset-work-step budget in addition to the enclosing VM budget;
each predicate/callback, row comparison, and aggregate cell visit
costs at least one step. Check counts and byte sizes for overflow
before allocation. A limit, unresolved selection, unsupported
relationship, failed callback, conflict, OOM, or corrupt reload
publishes neither partial rows nor a source revision. Existing
dataset calls remain in-memory unless the application explicitly
registers sources and queries through the new calls.

## Bounded learning runtime

The Rust runtime validates the complete example, feature, label, option,
and train/holdout split before search or fitting. Training and
validation arrays each have at most 10,000 rows, 64 features, and
128 labels. Rule search is capped at 100,000 candidate evaluations
and 5,000,000 predicate visits by default; caller options can only
lower these. Tree/forest/boosted fitting obeys the declared depth,
leaf, and ensemble limits and the enclosing VM instruction and
256 MiB memory budgets. Each example, candidate, split evaluation,
and prediction is charged to the active work budget. Reaching a
rule-search cap returns `limit_exhausted` with no selected rule and
cannot activate a version. Enclosing VM budget exhaustion, OOM,
invalid input, non-finite arithmetic, or cancellation returns an
error and publishes no model or active-version change.

Rule and tree artifacts use the exact `learned_task_v1` JSON schema
in `SPEC.md` and the existing revisioned store. Before publication
or reload, validate its digest, complete examples, rule AST or tree
reachability, finite thresholds/weights, feature domains, version
lineage, report metrics, and active validated status. Recompute
the validation trace from saved holdout rows and the saved model;
a mismatch is corruption. The holdout targets and each version's
calculation settings are immutable. The encoded task state is at
most 64 MiB; overflow or a larger record returns `LANA_ERR_LIMIT`
before publication.

A `save`, counterexample, or rollback operation checks the expected
store revision and stages the complete new record and active pointer
before one atomic commit. An invalid or insufficient candidate may
be retained as inactive while the old active version remains.
Identical counterexample retry returns the prior receipt; a
different payload with the same ID conflicts. On an I/O error with
uncertain commit outcome, reopen and inspect the version/receipt
before retry. Prediction and explanation are read-only and carry
`calibration:"uncalibrated"` where a probability is returned;
neither invokes an executor.

## Walk-forward and local-input execution

Walk-forward evaluation validates all timestamps, feature availability,
IDs, options, and trainer purity before running a fold. Every fold
starts a fresh trainer state and uses only its time-eligible training
and internal validation rows. It charges each fit and prediction to
the enclosing VM budget, caps input at 10,000 examples and 100 folds,
validates each complete definite model and prediction against
`SPEC.md`, computes the model digest from canonical tagged JSON,
and publishes one complete report only after all folds validate.
No model or report is saved implicitly. Failed trainer, non-finite
prediction, cancellation, or exhaustion exposes no partial report.

`dataset_sqlite` uses the installed `rusqlite` adapter with a read-only
database handle and one read transaction for the entire statement.
Prepare exactly one read-only row-returning statement, deny PRAGMA
changes, ATTACH, virtual-table side-effect calls, and user functions,
and bind parameters through SQLite's typed binding API. Validate
column names/types and every row before publishing. The source digest
covers canonical schema, bound values, and ordered rows from that same
transaction. Close or roll back on error. Caps are 10,000 rows, 64
columns, 16 MiB SQL input, and 64 MiB encoded output; integer
conversion outside binary64 exact range fails. WASM reports
`LANA_ERR_UNSUPPORTED_OPERATION`. The existing `adapter_fetch`
boundary is unchanged.

The native `document_extract` call reads at most 16 MiB from a local file and validates
UTF-8 and NUL absence before segmentation. All chunk offsets index
original bytes; CRLF counts as one line. Keep source text unchanged
inside each emitted span, omit blank separators, and check that
`source[start_byte:end_byte]` decodes to `text` for every chunk.
It also caps the chunk count at 100,000 and returns a limit error
before publishing any result. WASM reports unsupported operation.
Fenced blocks form one logical chunk before the 4,096-byte split;
oversize logical chunks split at UTF-8 boundaries into adjacent
subchunks. Errors return no partial chunk array. These local reads
do not grant a document truth or Information exactness claim.

## Hosted package execution

Pack uses the same deterministic archive writer for every invocation;
checked file names and sizes are charged before allocation. Add downloads
into an owned staging path, verifies the compressed digest before extraction,
and rejects unsafe tar headers, links, duplicate names, path traversal,
oversize files, and overlarge dependency closures before publishing the
lock. Extraction creates an owned cache directory keyed by the archive
SHA-256. A failed operation may leave an unused cache directory, but import
resolution requires a matching validated lock entry. Cache contents are
immutable after installation.

Lock replacement uses sibling staging, file sync, atomic rename, and parent
directory sync. A pre-rename failure preserves the old lock. A post-rename
directory-sync failure returns uncertain durability and the path; a fresh
process reloads and validates the lock before retry. Build reads the lock
and cache without network access, checks the archived digest again, and
resolves only the recorded package identity and source-relative path.
No fetched package code runs during add. A changed release asset or checksum
cannot silently replace a locked digest. Local HTTP fixture tests exercise
truncated downloads and each reject path, then restart to confirm the
previous lock still resolves.

## Brain file v2 (pending implementation)

`LBRN2` starts with five ASCII magic bytes, then little-endian u64
vocabulary, embedding width, Brain version, seed, and hidden-layer count.
Each layer has a positive u64 width, one activation byte
(`1 = relu`, `2 = gelu`), and seven zero reserved bytes. Parameter groups
follow as a u64 element count and that many finite little-endian F32 values.
Group order is embedding weights, each hidden layer's weights and bias in
layer order, output weights, then output bias. Expected shapes are the
row-major shapes in `SPEC.md` and are calculated before allocation.

The next fields are a u64 memory-entry count with repeated u64 byte length
and UTF-8 bytes, a u64 training-history count and that many finite F32
losses, a u64 replay-step count, then a u64 typed-memory JSON byte length
and canonical UTF-8 JSON bytes. Zero typed-memory length means no typed
records; otherwise the field is the complete schema-1 typed Information
memory record. The final 32 raw bytes are SHA-256 of every preceding byte.
The whole file is at most 256 MiB. Reject a wrong digest, reserved bit,
activation, count, shape, finite-value check, UTF-8 string, typed-memory
replay, truncation, trailing byte, or size overflow before exposing a Brain.
Keep `LBRN1` parsing and its five SafeTensors names unchanged.
Save `LBRN1` for a legacy one-layer ReLU model with no new layers or typed
memory; save `LBRN2` when either feature is present.

## Brain typed-memory replay (pending implementation)

Before publishing a loaded `LBRN2` Brain, validate its complete typed-memory
record, all IDs and references, finite-law normalization, derivation
acyclicity, and revision order. Reconstruct initial roots in an isolated VM;
replay observations in stored commit order through Core `observe`; then
compare the canonical current-value digest with `final_digest`. The loaded
record contains no process-local pointer. Historical derivation IDs remain
stable evidence labels and are not substituted for live VM identities.
Reject an unsupported law, impossible evidence, unknown relationship,
corrupt reference, mismatch, or budget exhaustion without exposing a
partial Brain. All changes to memory revision and the Brain file are
published only after this validation and the atomic-file boundary below.

## Brain fit publication (pending implementation)

`brain fit` loads a complete Brain and validates both JSONL files and all
options before updating a private model copy. Training, validation, and
checkpoint selection do not mutate the loaded file. The existing 256 MiB
Brain-file limit still applies. Integer products and offsets are checked
before allocation; exhaustion, cancellation, invalid data, or a non-finite
loss returns a structured error and exposes no partial model or success
report.

The selected checkpoint is serialized completely to an owned sibling file,
synced, atomically renamed over the requested Brain file, then followed by
parent-directory sync. A failure before rename preserves the old Brain bytes.
A directory-sync failure after rename reports uncertain durability with the
destination path; the caller must load and validate that path in a fresh
process before retrying. No success report is emitted for that uncertain
outcome. A staged file is removed after failure when possible.

## Brain evidence selector

The optional CLI selector uses the validated semantic index and a separate
schema-1 artifact. F32 logistic fitting never mutates Brain parameters or
typed memory. Vector training and ranking charge bounded work to the default
VM limit; cached vectors are checked before allocation. Required exact-target
and forecast/action evidence is retained before learned candidates, including
conflicting and unresolved targets. Contexts exceed neither 64 entries nor
4,096 tokenizer tokens; overflow fails rather than dropping evidence.

Training validates disjoint input pairs and the held-out activation gate
before publishing. An active model replaces `MODEL` through the shared atomic
writer; a rejected candidate and its report go to `MODEL.inactive.json` while
the prior model stays intact. Pre-rename failure leaves the old artifact;
post-rename sync failure reports uncertain durability and the destination.
Loaded selectors validate the exact schema, finite canonical F32 bits, pair
IDs, active status, and Brain/tokenizer/corpus digests before use.

## Brain recovery qualification (pending implementation)

Brain saves, package imports, and workshop reports validate full contents
before replacement and use owned sibling staging, file sync, atomic rename,
and parent-directory sync. A staged-write or pre-rename failure preserves
the old complete destination or its absence. After rename, a
directory-sync failure reports `durability: "uncertain"` and the path;
the new complete bytes may already be visible. A caller must inspect and
validate the destination in a fresh process before retrying and cannot
claim rollback. Fault tests stop immediately before staged-file sync,
immediately before rename, and immediately after rename before directory
sync. After each stop a fresh process must load the entire Brain or package
and find exactly the old or new complete revision and SHA-256, never mixed
bytes or a digest mismatch. Fault controls exist only in test builds.

## Sample records

Every stochastic source expression is lowered to a two-element runtime record:
the sampled value followed by an immutable metadata map. `sample_value` and
`sample_metadata` are explicit projections of that record. The metadata contains
`source_dependency`, the root `rng_seed`, deterministic `task_lineage`, the
sampling `operation`, and the current observation `revision`. The compiler-only
`sample_record(value, dependency, operation)` host primitive snapshots those VM
fields after the stochastic operation succeeds; it does not consume RNG state.
No exact operation implicitly projects the sampled value.

## Structured runtime failures

The VM owns one `LanaErrorInfo` record for the current failure. Stable code and
kind are accompanied by a message, full source span, operation, instruction and
opcode, bounded cause chain, and operation-specific context. Resolution errors
record the reason and remaining alternatives; unsupported exact operations
record requested and available support; cancellation records the task/reason;
resource-limit failures record the resource, limit, and observed amount.
Failure clears the public result before returning. Child-task errors are copied
without publishing a partial child value.

The canonical runtime is a Rust register VM. `STATE` values store canonical
binary64 `p`, `d_re`, and `d_im` inline with metadata. `STATE_DIST` is a pointer
to an immutable VM-owned node:

- `DIRAC` captures a full state value, including metadata.
- `APPEND` retains left and right nodes and may cache direct-state parameters.
- `TRANSFORM` retains a child and a registered v3 transform identifier.

Moves share nodes inside one VM. Task transfer deep-copies the DAG with a memo
table, copies captured metadata strings, preserves shared subgraphs, and never
shares nodes across VMs. APPEND-generated concrete states have empty metadata;
transforms preserve their sampled input metadata.

Unqualified computational-basis expected probability is evaluated exactly and
recursively: Dirac returns `p`, APPEND returns
`1-(1-E[left])*(1-E[right])`, and TRANSFORM invokes its registered exact
expectation rule. Every distribution-liftable transform must register a concrete
state function, validity guarantee, and exact expectation function.

Basis-aware concrete-state measurement reconstructs
`c = sqrt(p*(1-p)) * (d_re + i*d_im)`. With ordered bases
`computational=(|0>,|1>)`, `x=(|+>,|->)`, and
`y=(|+y>,|-y>)` where `|+y>=(|0>+i|1>)/sqrt(2)`, `q_B` is the probability of
outcome `1`: `p`, `1/2 - Re(c)`, and `1/2 + Im(c)`, respectively. Probability
mode returns `q_B`, distribution mode returns `distribution(1-q_B, q_B)`, and
sample mode draws `Bernoulli(q_B)`. Measurement is read-only.

Qualified basis-aware measurement of `STATE_DIST` supports exact sampling only:
the VM samples one concrete state, computes its exact outcome-1 probability
`q_B`, and uses the existing PCG32 binary draw. Qualified probability/distribution modes
return `LANA_ERR_UNSUPPORTED_EXACT_MEASUREMENT`.

`estimate_measure` is the explicit approximate path. For each of `N` trials the
VM consumes one instruction-budget unit, samples the existing distribution, and
computes the exact outcome-1 probability `q_B` of that sampled state. It returns the
arithmetic mean `q_hat`, or `distribution(1-q_hat, q_hat)`, only after all trials
complete. Cancellation, memory errors, invalid distribution nodes, and budget
exhaustion return an error and never expose a partial estimate. The VM's existing
seeded RNG is used, so the same chunk, inputs, sample count, and seed reproduce
the same estimate. This is regular Monte Carlo and is not presented as exact
mathematical evaluation; no confidence interval is returned.

Sampling preserves the binary tree. An APPEND node samples each child, computes
the conditional APPEND parameters, then uses PCG32 and a fixed Marsaglia-polar
normal-pair proposal with no cached spare. Unit-disk rejection implements the
truncated circular complex normal. Before every proposal the VM checks
cancellation and consumes one instruction-budget unit. Cancellation and budget
exhaustion return immediately with no fallback sample.

VM-lifetime allocations are released by `lana_vm_free`. Allocation failure returns
`LANA_ERR_OOM`. Malformed lazy nodes return `LANA_ERR_INVALID_DISTRIBUTION`; missing
exact transform support returns `LANA_ERR_UNSUPPORTED_EXACT_MEASUREMENT` during
expectation evaluation and prevents distribution lifting with
`LANA_ERR_UNSUPPORTED_OPERATION`.

Forked functions have independent registers, heap, instruction and memory
budgets, RNG stream, and error state. Bytecode/constants remain immutable and may
be shared. Task groups, cooperative cancellation, joins, tracing, and host calls
retain their existing architecture.
