# Runtime architecture

This document defines how the Rust VM owns data, runs work, and handles failure.
[The language specification](SPEC.md) defines source behavior. [The bytecode specification](BYTECODE.md) defines LABC encoding and verification.

## Rules shared by runtime operations

Each VM owns its registers, heap, instruction budget, memory budget, random stream, and error state. Bytecode and constants are immutable and may be shared between tasks. VM-owned pointers never cross task heaps: transfer copies reachable data and preserves aliases and cycles with a memo table.

Validate a complete result before publishing it. An invalid input, unsupported operation, cancellation, exhausted budget, allocation failure, or failed callback exposes no partial value, register update, store revision, or effect receipt. Operations charge their work and temporary memory to the active task. The compiler is limited to 256 MiB and 50,000,000 instructions.

An I/O error before atomic replacement leaves the old file or store state intact. If directory sync fails after replacement, report `LANA_ERR_IO` with `durability: "uncertain"` and the path. Reopen and inspect the result before retrying.

## Derivations and Information

Each VM owns immutable derivation nodes. Their IDs combine deterministic task lineage with a local sequence. Values may refer to a node, but mathematical equality ignores that reference. Task transfer copies the derivation graph and keeps origin IDs. Failed work may appear in a structured error. It does not become a value.

Finite Information, named Joints, projections, and kernels are VM-owned. Constructors validate names, domains, rows, weights, references, and normalization before publishing. Projection and conditioning make new values. They leave the source unchanged. Exact resolution requires one remaining result or returns `LANA_ERR_UNRESOLVED_VALUE`. A finite correlated sample draws one row, not one independent draw per column. A child task uses its own random stream.

A live Information root owns finite support and a dependency identity. Pure operations retain their inputs and recompute affected results when the root changes. Observation stages the entire update, then publishes it under one new revision. If any result fails, the old revision remains. Different dependency identities do not imply independence. Serialization and task transfer capture current values instead of live links.

Pure arithmetic, comparisons, and function bodies can lift over finite alternatives. `PATH_SPLIT` runs both sides of an unresolved Boolean. `PATH_JOIN` merges their guarded results. Nested splits are bounded. Unresolved loops, incompatible dependency joins, and merges of history-bearing registers are unsupported. A split branch cannot print, call a host, create a task, sample, or observe. This prevents duplicate or partial effects.

Claims keep their proposition, exactness, tolerance, and source validity separate. Plans keep a stable local identity, payload, and receipts. Execution requires a definite payload. For each `(identity, revision)`, the configured executor runs at most once. Later reads return the saved result.

Shared Information has an isolated storage VM and an immutable commit history. A candidate commit is built from a stable evidence copy outside the lock. It publishes only if the base commit, capability epoch, and observation count still match. A global atomic counter orders commits. Revocation advances the epoch and wakes waiters. Task transfer retains shared identity without copying the live graph.

## Finite laws and measures

The VM computes entropy, conditional entropy, and mutual information over declared finite Joints in bits. It groups equal projected assignments and records selected names and input revision in the derivation. Explicit weight conversion validates complete, unique rows and records the conversion.

Kernels contain ordered input and output domains with complete normalized rows. `identity_kernel` creates exact identity rows. `compose_kernels` requires matching domains. `kernel` runs one pure callback per Cartesian tuple and keeps no callback afterward. Networks hold an immutable root Joint and acyclic, ordered child dependencies. `infer` applies exact named evidence, sums out other variables, and returns a normalized finite Joint.

Before materializing a projected, product, coupled, or network law, check cardinality overflow and charge enumeration, callbacks, and BROJA work to the VM budgets. BROJA reports `converged` only with the certified residual and objective error bound in [the semantics](../papers/semantics.md). It otherwise returns `unconverged` without component values. The same input and budget produce the same row order and diagnostic.

## Immutable capture

Host ID 220 implements `snapshot`. It copies the current value of every live input, removes live links, and records the captured revision. Joint rows, weights, path guards, and dependency labels survive. Captured arrays and maps reject writes. Capture uses the VM budget, rejects nesting beyond 64, and publishes nothing on failure.

## State distributions and measurement

A `STATE` stores binary64 `p`, `d_re`, and `d_im` with metadata. A `STATE_DIST` is an immutable VM-owned tree: `DIRAC` holds a state, `APPEND` holds two children, and `TRANSFORM` holds a child and a registered transform. Moves share nodes inside one VM. Task transfer copies the tree and metadata without sharing VM pointers. APPEND-generated states have empty metadata. Transforms retain metadata from the sampled input.

Computational-basis expected probability is exact: `DIRAC` returns `p`. `APPEND` returns `1-(1-E[left])*(1-E[right])`. `TRANSFORM` uses its registered exact expectation rule. A distribution-liftable transform must register a concrete state function, validity guarantee, and exact expectation function.

For a concrete state, measurement reconstructs `c = sqrt(p*(1-p)) * (d_re + i*d_im)`. Outcome-1 probabilities are `p` in the computational basis, `1/2 - Re(c)` in the x basis, and `1/2 + Im(c)` in the y basis. Probability and distribution modes return exact results. Sample mode draws a bit. Measurement does not change the state.

Qualified measurement of a `STATE_DIST` supports sampling only. It first samples a concrete state, then draws a bit from that state's exact outcome probability. Probability or distribution mode returns `LANA_ERR_UNSUPPORTED_EXACT_MEASUREMENT`.

`estimate_measure` is the explicit approximate path. It samples a state and computes its exact outcome probability for each of `N` trials, charging one instruction per trial. It returns the mean, or a distribution made from that mean, only after all trials finish. The seeded random stream makes the result reproducible for the same chunk, inputs, count, and seed. It provides no confidence interval.

APPEND sampling keeps the tree. It samples both children, computes conditional APPEND parameters, and uses PCG32 with a fixed Marsaglia-polar normal-pair proposal and unit-disk rejection. Each proposal checks cancellation and costs one instruction. Failure gives no fallback sample. Malformed nodes return `LANA_ERR_INVALID_DISTRIBUTION`. Missing exact transform support returns the applicable unsupported-operation error.

## Objects, effects, and collection

The v6 verifier checks descriptors, ownership, and declared effects before execution, including unused methods and transitive calls. Unknown extension hosts require external-call permission. Runtime checks still enforce types, visibility, effects, and definite receivers at dynamic calls. Older v1-v5 chunks retain their object-free behavior.

A value owns an immutable descriptor and ordered deep field snapshots. `VALUE_NEW` checks types and immutability before publishing. It copies and freezes supported nested arrays, maps, ADTs, States, State distributions, Tensors, Shapes, and captured finite Information. Live roots, class references, cycles, capabilities, executable payloads, and unsupported payloads are rejected. Construction has a depth limit of 64. Equality has a depth limit of 64 and a 100,000-node traversal limit. Display shows only the nominal name. JSON export accepts public, JSON-compatible fields. JSON loading makes ordinary maps and cannot restore identity or private members.

A class object has a task-local identity, field storage, and initialization state. `OBJECT_NEW` keeps the candidate private while pure defaults run once and the initializer assigns required fields. An immutable field is assigned at most once. A defaulted immutable field cannot be assigned again. A mutable field can change after initialization. `self` cannot escape before construction finishes. Failure discards the candidate without exposing an identity, write, or receipt.

`OO_GET`, `OO_SET`, and `OO_CALL` check descriptor, member, visibility, initialization, type, arity, and effect allowance. A write outside initialization requires a mutable field and a definite receiver. It is an object mutation, not an Information observation. `OO_STATIC_CALL` has no receiver or type-level mutable state. A private member is available only to a frame owned by its descriptor. Copied methods belong to the child. There is no runtime superclass pointer.

`OO_AS_INTERFACE` keeps the same underlying value or class identity. It grants only promised methods, dispatches through the verified final method table, and grants no field access. Class transfer assigns fresh identities in the receiving task while preserving aliases and cycles. Live Information fields become immutable snapshots. Direct class serialization is unsupported.

The collector traces frames, suspended work, host roots, Information graphs, plans, receipts, and task results. It reclaims unreachable classes, cycles, and immutable graphs. Minor collections trace young objects and remembered old objects. Write barriers protect old-to-young and stable-shared edges. Routine incremental work is limited to 128 units per safepoint. Full synchronous tracing handles severe pressure, invariant fallback, and shutdown. Scratch space and traversal cost count against VM budgets. Container pressure starts above three quarters of the memory limit. Class construction also checks at half the limit. These thresholds do not raise either limit.

The Rust embedding API returns `Result<RootedValue, LanaError>`. A rooted handle keeps reachable class storage alive beyond VM teardown. Cloning the handle shares that root. Borrowed values and internal edges do not create embedding roots.

## Compiler, files, and debugging

The self-hosted compiler runs as verified Lana bytecode within the compiler limits. Lana code performs lexing, parsing, resolution, IR lowering, emission, import-cycle checks, and call remapping. Rust host calls provide OS paths and atomic file operations. A clean build assembles the checked textual bootstrap artifact and needs no Python runtime.

Compilation emits and verifies a complete chunk in a sibling file. It syncs that file, atomically replaces the destination, then syncs the parent directory. Parse, verification, cancellation, OOM, and limit errors occur before replacement and report the applicable code with a source span or bytecode offset when available. If the final directory sync fails, use the uncertain-durability rule above.

`lana debug` uses an instruction hook and the deterministic source line stored in each LABC instruction. Stops show the active function and frame count. Step, continue, quit, and low-level trace use that same mapping. Defaults and initializers use owned frames, so debugging cannot publish an incomplete object.

## Dataset history

The store owns source rows, plan identity and digest, typed snapshots, derivations, receipts, and revisions. A query reads every declared source at one captured store revision. Registration validates the initial result and commits its query record and snapshot together. After restart, a matching `dataset_query` binds the current verified chunk and pure function without writing. Historical reads need no bind. A changed plan or source list at the same calculation version conflicts.

`dataset_apply` allows unresolved values only in its changes argument. Each uncertain cell needs an explicit immutable capture. The codec keeps source evidence and binds local labels to durable source and batch identities. Receipts keep those bindings for retries after restart. An evaluation decodes sources in one relationship context, assigns fresh local IDs, freezes nested values, and rejects malformed or incompatible laws. Existing definite records keep their byte encoding.

Before a change commits, validate all changes and rerun every bound query that names the source. An unbound dependent query blocks the change. Commit sources, snapshots, query records, and receipt in one store revision. An identical batch retry returns its prior receipt before checking the expected revision. A conflicting retry, stale revision, or failed rerun publishes nothing. Insertions, joins, and first-seen groups have stable order. Correction keeps source position and deletion removes obsolete output and evidence.

Named plans use one 5,000,000-step work budget across source reads, callbacks, comparisons, and aggregate visits. They reject input mutation through `array_push`, non-map output rows, duplicate or unattributed rows, and more than 100,000 result rows. Each output row carries stable source and operation links. Filtered, excluded, and unmatched rows retain ordered exclusion decisions. SHA-256 derivation IDs bind query ID, exact plan digest, source revision, traversal path, and ordered row paths. A snapshot contains the complete typed result and evidence, which the runtime validates again on reload.

A saved `dataset_snapshot_v1` uses the canonical tagged-value encoding in [the source specification](SPEC.md). It records versions, revisions, ordered rows, inclusion and exclusion decisions, and a topologically ordered derivation graph. Reload checks schema, UTF-8, IDs, references, cycles, finite laws, digests, revisions, and canonical re-encoding before publication. Historical reads use the record visible at the requested revision. Compacted history returns its own error. It never substitutes current rows. If commit durability is uncertain, reopen and inspect the batch ID and revision before retrying.

| Dataset limit | Value |
| --- | ---: |
| Rows per source | 10,000 |
| Rows per result | 100,000 |
| Encoded snapshot | 64 MiB |
| Derivation nodes per snapshot | 100,000 |
| Alternatives per finite cell | 1,024 |
| Work steps per full query rerun | 5,000,000 |

These limits add to the enclosing VM budgets. Check counts and byte sizes for overflow before allocation. Calls remain in memory unless the application explicitly registers sources and queries.

## Bounded learning and evaluation

Validate examples, features, labels, options, and the train/holdout split before fitting. Training and validation each allow at most 10,000 rows, 64 features, and 128 labels. Rule search defaults to 100,000 candidates and 5,000,000 predicate visits. Caller options may only lower these caps. Trees, forests, and boosting also obey their declared depth, leaf, and ensemble limits. Work counts against the active budget. Reaching a rule-search cap returns `limit_exhausted` without selecting or activating a rule.

Rule and tree artifacts use `learned_task_v1` from [the source specification](SPEC.md) and the revisioned store. Before publication or reload, validate the digest, examples, model structure, finite values, domains, lineage, reports, and active status. Recompute the saved holdout trace. A mismatch is corruption. Holdout targets and version settings are immutable. Encoded task state is limited to 64 MiB.

Save, counterexample, and rollback operations stage the full record and active pointer for one atomic commit at the expected revision. Invalid candidates may remain inactive while the old version stays active. An identical counterexample retry returns its receipt. A changed payload with the same ID conflicts. After uncertain I/O, reopen and inspect the version or receipt. Prediction and explanation are read-only, never execute effects, and label returned probabilities `calibration:"uncalibrated"`.

Walk-forward evaluation validates timestamps, feature availability, IDs, options, and trainer purity. Each fold starts a fresh trainer and uses only time-eligible rows. It allows at most 10,000 examples and 100 folds. Fits and predictions count against VM limits. It publishes one complete report after all folds pass validation. It saves no model or report implicitly.

## Local inputs and hosted packages

`dataset_sqlite` uses one read-only transaction and one row-returning statement. It rejects PRAGMA changes, ATTACH, side-effecting virtual-table calls, and user functions. Bind typed parameters and validate every row. The digest covers schema, bindings, and ordered rows from that transaction. Limits are 10,000 rows, 64 columns, 16 MiB SQL input, and 64 MiB encoded output. Integers outside the exact binary64 range fail. WASM returns `LANA_ERR_UNSUPPORTED_OPERATION`.

`document_extract` reads at most 16 MiB of local UTF-8 without NUL bytes. Chunks preserve original byte spans. CRLF counts as one line. Fenced blocks form a chunk before splitting at 4,096 bytes on UTF-8 boundaries. Blank separators are omitted. It allows at most 100,000 chunks and returns no partial array. WASM reports unsupported operation. Reading a document does not establish its truth or Information exactness.

Package packing uses deterministic archives. `add` verifies the compressed digest before extraction and rejects links, unsafe headers, duplicate names, path traversal, oversize files, and excessive dependency closure. It stores an immutable cache by archive SHA-256. A failed add may leave unused cache data, but imports require a validated lock entry. No fetched code runs during add.

Lock replacement stages a sibling file, syncs it, renames it atomically, and syncs the directory. Build uses the lock and cache without network access. It checks the archive digest again and resolves only recorded package identities and paths. A changed release asset or checksum cannot replace a locked digest silently. Uncertain directory sync follows the shared retry rule above.

## Samples and failures

A stochastic source expression produces a two-element record: sampled value and immutable metadata. `sample_value` and `sample_metadata` read those parts explicitly. Metadata includes `source_dependency`, root `rng_seed`, `task_lineage`, sampling `operation`, and observation `revision`. The compiler-only `sample_record` host call captures these after sampling without consuming more random state. Exact operations never extract the sampled value implicitly.

The VM keeps one `LanaErrorInfo` for the current failure. It includes stable code and kind, message, source span, operation, instruction, opcode, a bounded cause chain, and relevant context. Resolution errors report remaining alternatives. Unsupported exact work reports available support. Cancellation and limits report their reason or resource. Failure clears the public result. Child-task errors are copied without exposing a partial child value.
