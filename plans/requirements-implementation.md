# Requirements implementation checkpoint

Started: 2026-09-26. Local, uncommitted work. Historical checkpoints below
record evidence for their own candidate, not current-tree qualification.

## Current status (2026-09-28)

Completion-matrix rows 2 (heap ownership), 3 (collector policy), and 8
(exact-tree qualification) are open. Rows 4–7 have earlier local evidence and
require final-candidate checks. Checkpoint 60 corrects checkpoint 59.
Historical “remaining” lists are superseded by later numbered checkpoints and
the current matrix; they are not additional authorization for deferred work.

The user authorized implementation of the active requirements in dependency-ordered
batches. Explicit exclusions remain excluded. The tree already contained changes
to the authority documents, Brain fit/layers, atomic publication, and unrelated
is-ought work when this implementation started. Those changes were preserved.
`REQUIREMENTS.md` has not been cleared.

## Implemented and checked in this pass

1. Extended the shared atomic-file test boundary to cover errors and process exit
   before file sync, before rename, and after rename. Fresh processes validate
   old/new complete files for LBRN1, dense LBRN2, and typed-memory LBRN2, with both
   existing and absent destinations: 36 cases.
2. Corrected the dense-layer xorshift stream, rejected oversized new architectures
   before parameter allocation, bounded architecture-file reads, rejected
   non-finite logits, and exposed changed parameter groups from CLI training.
3. Added strict LBRN2 SafeTensors/package support, exact snapshot/tensor comparison,
   digest checks, tokenizer checks, and uncertain post-replacement error reporting.
4. Added the canonical finite Information codec and durable Brain roots,
   observations, revisions, derivation records, and final replay digest. Replay
   constructs isolated Core roots and uses Core observation. Exposed
   `brain memory add|observe|inspect`; failed observations preserve file bytes.
5. Added exact fact/root question aliases and `brain chat ... --grounded`.
   Missing, unresolved, and ambiguous queries do not write. Successful answers
   include evidence references. Typed package validation invokes the Rust loader;
   standalone bridge commands need `lana` on PATH or `LANA_CLI`.
6. Added the workshop JSONL answer fixture and atomic `--report` output, including
   artifact/fixture/parameter digests and reload revisions. Deliberately wrong
   expected answers fail without replacing an earlier report.
7. Preserved compiler VM error codes outside parse/type diagnostics and surfaced
   uncertain durability and destination path on compiler/assembler output writes.
   Routed legacy Brain new/train/save/remember/chat writes through
   `save_with_status`, so post-replacement failures retain that status.
   Added separate-process package/unpackage exit checks around publication.
8. Added caller-declared finite forecasts to the versioned typed Brain record,
   with immutable probability vectors, evidence-reference validation, replayed
   creation/score revisions, exact idempotent retries, and multiclass Brier
   scoring. Exposed `brain forecast add|score` through atomic Brain saves.
9. Added bounded grounded context for exact root targets. Identical typed
   observations collapse into one working entry with all observation and
   derivation references; unrelated roots are omitted. More than 64 distinct
   selected entries returns `LANA_ERR_LIMIT` before chat history is saved.
10. Added an advisory Brain decision example using the existing
    `std/decision.value_of_information`, with a positive-net-value observation,
    a nonpositive one, and an unscorable missing relationship. Registered it
    as a CTest case; it performs no observation or effect.
11. Added separate-process compiler malformed-bytecode, cancellation,
    instruction-limit, and memory-limit checks using synthetic compiler inputs
    under the production 50,000,000-instruction and 256 MiB policies. Existing
    and absent output destinations remain unchanged; surviving output verifies
    in a fresh process. These prove those failure paths, not post-replacement
    durability.
12. Added paired ordinary/grounded workshop observations from copies of the
    same starting Brain artifact, with per-call elapsed time, selected-record
    count, equal parameter digests, and explicit `held_out: false` and
    `improvement_claim: false` labels. This is instrumentation for the later
    frozen held-out evaluation, not evidence of generalization.
13. Added a caller-declared two-label forecast and observed outcome to the
    workshop fixture. The report checks its hand-calculated Brier score and
    reports it separately from answer counts and next-token loss.
14. Included forecasts explicitly attached to a grounded target after its
    root/observation context and then their explicitly referenced evidence,
    within the 64-entry bound. Unreferenced roots and unrelated-target
    forecasts are omitted; the workshop checks the selected record count.
15. Added an opt-in publication-failure CLI build that forces failures before
    file sync, before replacement, and after replacement without changing the
    normal binary. A separate-process check covers compiler and Brain outputs
    with existing and absent destinations, checks error/durability/path status,
    and verifies complete survivors in fresh CLI processes.
16. Added a late-relevant-evidence check after 80 unrelated roots. Added a
    SHA-pinned held-out question-text fixture for the fixed Brain setup and
    separate ordinary/grounded answer scoring. Its scope is normalization and
    exact retrieval for known targets; no decision utility or broader
    generalization claim is made.
17. Added separate-process Python package/unpackage checks for directory-sync
    failure after a complete replacement. The CLI returns structured uncertain
    durability and destination path; fresh processes load the resulting package
    and Brain file for absent and existing Brain destinations.

Verification executed: focused Brain/codec/replay/recovery Rust tests, separate
process fit/workflow/memory/workshop checks, Python bridge tests, workspace Cargo
tests, CMake configure/build, and `git diff --check`. The Python reference-library
test skips when tokenizers/SafeTensors/NumPy are absent. Full CTest passed 76/76
in 99.36 seconds; details are in `build/Testing/Temporary/LastTest.log`.
The final derivation-count guard and expanded 36-case recovery matrix also passed
the focused Rust Brain checks after the broader workspace run.
After item 7, `cargo build --locked -p lana-cli`, separate-process compiler-output
and Brain workflow checks, Python bridge tests (13 run, one optional skip), and
`git diff --check` passed. The broad suite was not rerun after item 7.
After item 8, focused Brain memory Rust tests, the separate-process Brain
memory/package workflow, and the full Cargo workspace passed. CMake built.
CTest initially passed 74/76; two import-cycle tests exposed a source-span
regression in compiler assertions. After restoring that diagnostic path and
rebuilding, both failed tests passed. The other 74 were not rerun after that
diagnostic-only fix. Release gates were not run.
After item 9, focused Rust Brain memory tests and the separate-process Brain
memory workflow passed, along with `git diff --check`; the broad suite was not
repeated for this additional response-field change.
After item 10, the example ran through the CLI and its registered CTest case
passed. The full CTest suite was not repeated for this new example.
After item 11, the separate-process compiler-output check passed for
cancellation, both limits, and both destination states; `git diff --check`
remains clean.
After item 12, the separate-process workshop report check passed, including
preservation of an existing report on a deliberately wrong answer.
The registered CTest compiler-output and Brain-workshop checks passed after
the latest additions (2/2). The full CTest suite was not rerun in this pass.
After item 13, the registered CTest compiler-output and Brain-workshop checks
passed (2/2), including the forecast round-trip and an unchanged prior report
on a wrong answer. `git diff --check` passed. The full CTest suite was not
rerun after these fixture/report additions.
After item 14, focused Rust Brain memory tests and the separate-process
workshop report check passed. A focused Rust case also confirmed that 65
selected entries fail with `LANA_ERR_LIMIT` and leave memory bytes unchanged.
The full Cargo workspace and `git diff --check` then passed. After rebuilding
the CLI, the separate-process workshop check passed against the current binary.
The full CTest suite was not rerun.
After item 15, the opt-in CLI build and its 12 separate-process publication
cases passed. The normal CLI ignored the test-only failpoint and the existing
compiler-output process check passed. Focused atomic-file Rust tests passed 4/4
and `git diff --check` passed. The full CTest suite was not rerun.
After item 16, the focused late-evidence Rust test and the separate-process
Brain workshop check passed, including pinned-fixture rejection on tampering.
The held-out CLI command also passed. After rebuilding the CMake CLI, the full
CTest suite passed 77/77 in 100.96 seconds, and the full Cargo workspace tests
passed. A CTest attempt before the rebuild failed two Brain checks because it
used a stale CMake binary; both passed after rebuilding. Release gates were not
run.
After item 17, the focused package process case and the complete local
`tools/lana-hf` unittest suite passed (14 tests, one optional-library skip),
and `git diff --check` passed. The full CTest suite was run before item 17;
its bridge test was covered by the focused Python suite after that addition.

18. Implemented finite-joint entropy, conditional entropy, and mutual
   information as `std/core` calls. They combine equal projected assignments,
   accept weighted independent marginals, preserve derivation names/revision,
   and reject duplicate names or unsupported exact laws. Replaced the
   compiler's linear host-name chain with a one-time registry so the new
   bootstrap remains below the unchanged 50,000,000-instruction limit.
19. Implemented explicit `std/core` weight loss and assignment. The calls
    preserve candidate support and derivation, require complete unique
    positive weights totaling one, and leave their source unchanged. Bytecode
    host IDs 198–199 are active while pending IDs 192–197 remain rejected.
20. Added an opaque finite kernel value, `identity_kernel`, and
    `compose_kernels`. Domain validation rejects empty, duplicate, and
    unresolved candidates; composition requires identical ordered middle
    domains and normalizes each result row. Callback-based kernel construction
    and network inference are still pending.
21. Added `info.kernel(input_domains, output_domain, chances_fn)` with
    named pure callback lowering, ordered Cartesian evaluation, complete
    row validation, normalized immutable rows, and a VM effect guard.
    Invalid callbacks and rows fail before a kernel is published.
22. Added an opaque `info.network(root_joint, nodes)` constructor. It accepts
    finite root support, checks unique names, exact ordered parent domains,
    and cycles, and snapshots the root and kernels. `info.infer` remains pending.
23. Added `info.infer(network, query_names, evidence_map)` over declared
    finite laws. It supports correlated and independent roots, multiplies
    kernel rows in topological order, filters exact evidence, sums out other
    variables, and rejects impossible evidence.
24. Added bounded `info.broja` with analytical certificates for exact
    target-independent marginals and original-law objective gaps. XOR and
    duplicate-source fixtures converge; uncertified laws return an
    `unconverged` record without components. A general certified optimizer
    remains pending.
25. Added bounded binary-source BROJA optimization. It searches feasible
    per-target 2-by-2 coupling intervals and checks the convex objective
    gap, marginal residuals, and mass before publishing components. The AND
    fixture converges; a ternary-source fixture remains `unconverged`.
    Larger source domains still need a certified transport solver.
26. Extended the bounded solver to one binary and one larger finite source.
    Pairwise feasible transfers reduce the objective, while a greedy linear
    transport oracle supplies the convex error bound. Both 3-by-2 and 2-by-3
    source fixtures converge; a 3-by-3 fixture stays `unconverged`.
27. Added a bounded general finite BROJA solver with a transport-dual objective
    certificate. A 3-by-3 source fixture converges, while a harder 4-by-4
    fixture retains the `unconverged` fallback without components.
28. Rejected using the same physical file for Brain fit training and validation,
    including symlink and hard-link aliases on Unix. Expanded fit input-failure
    and opt-in publication-failure checks across LBRN1 and LBRN2, all three
    atomic-write stages, and fresh-process reload.
29. Added VM rollback regressions for a failed host call that filled its staged
    result, an impossible observation, and an out-of-memory planned effect.
    They check destination register, public result, revision, observation count,
    and receipt state at the failure boundary.
30. Added cooperative SIGINT/SIGTERM cancellation to Unix Brain fit. The CLI
    returns `LANA_ERR_CANCELLED` without a success report and leaves the saved
    model unchanged when interrupted before publication. Separate-process checks
    send both signals during a long fit and reload the unchanged model.

After item 18, the generated bootstrap was byte-stable across two builds,
the finite source fixtures passed, the full Cargo workspace passed, and
the full CTest suite passed 79/79. A final change made temporary measure
buffers use scoped VM reservations; focused VM tests and the registered
finite/bootstrap CTest cases passed after that change. `git diff --check`
passed. Release qualification was not run.
After item 19, the generated bootstrap was byte-stable across two builds
under the unchanged compiler limits, and the focused conversion and reserved
host-ID checks passed. The full Cargo workspace and stable-binary CTest suite
passed (81/81); `git diff --check` passed. An earlier CTest run failed only
because a concurrent CLI rebuild changed its executable during the 60-second
study case; the stable rerun passed. Release qualification was not run.
After item 20, the full Cargo workspace passed, the compiler bootstrap was
byte-stable under unchanged limits, and focused kernel/host-ID checks passed.
The stable-binary full CTest suite passed 83/83, and `git diff --check` passed.
Release qualification was not run.
After item 21, the generated bootstrap was byte-stable under unchanged limits,
the full Cargo workspace passed, the stable-binary CTest suite passed 86/86,
and `git diff --check` passed. Release qualification was not run.
After item 22, the generated bootstrap was byte-stable under unchanged limits,
the full Cargo workspace passed, the stable-binary CTest suite passed 89/89,
and `git diff --check` passed. Release qualification was not run.
After item 23, the generated bootstrap was byte-stable under unchanged limits,
the full Cargo workspace passed, the stable-binary CTest suite passed 92/92,
and `git diff --check` passed. Release qualification was not run.
After item 24, the generated bootstrap was byte-stable under unchanged limits,
the full Cargo workspace passed, the stable-binary CTest suite passed 96/96,
and `git diff --check` passed. Release qualification was not run.
After item 25, the full Cargo workspace passed, the stable-binary CTest suite
passed 97/97, and `git diff --check` passed. Compiler source and bootstrap were
unchanged in this batch; release qualification was not run.
After item 26, the full Cargo workspace passed, the stable-binary CTest suite
passed 99/99, and `git diff --check` passed. Compiler source and bootstrap were
unchanged in this batch; release qualification was not run.
After item 27, the full Cargo workspace passed, the stable-binary CTest suite
passed 100/100, and `git diff --check` passed. Release qualification was not run.
After item 28, the focused Brain fit and opt-in publication-failure process
checks passed. The full Cargo workspace and stable-binary CTest suite then
passed (100/100), and `git diff --check` passed. Release qualification was not run.
After item 29, focused VM rollback tests and the full Cargo workspace passed,
along with `git diff --check`. The CTest suite was unchanged and not rerun;
release qualification was not run.
After item 30, the separate-process SIGINT/SIGTERM fit checks, full Cargo
workspace, stable-binary CTest suite (100/100), and `git diff --check` passed.
Release qualification was not run.

## Current four-batch pass

1. Confirmed the existing dirty-tree baseline and remaining gaps without
   resetting or staging files. The full Cargo workspace and `git diff --check`
   passed before new work.
2. Extended the VM failure matrix to cancellation, instruction exhaustion,
   memory exhaustion, and malformed bytecode before publication. Each forced
   failure keeps the destination register, public result, revision, and
   observation count unchanged. The focused VM test and compiler-output,
   Brain-memory, and Brain-workshop CTest cases passed; `git diff --check`
   passed. Existing separate-process publication failpoint coverage remains
   the durable-file check. No release qualification was run.
3. Audited the existing fixed-weight grounded selector and resource boundaries;
   no duplicate implementation was needed. The selector retains late forecast
   evidence after unrelated roots, deduplicates equivalent observations with
   their IDs, and returns `LANA_ERR_LIMIT` before publication above 64 selected
   records. The Brain loader bounds files at 256 MiB; ordinary model chat
   bounds its token context at 4,096. Focused selector tests (3/3) and the
   separate-process Brain-memory workflow passed. Action decisions remain
   advisory through `std/decision`; the paired decision evaluation is next.
4. Extended the SHA-pinned held-out workshop fixture with one declared
   decision. The workshop runs `std/decision.value_of_information`, checks
   positive, nonpositive, and unscorable candidates, and reports each paired
   answer, retained evidence, time, selected-record count, unchanged parameter
   digest, and the utility of a fixed answer-to-action policy. In this fixture,
   ordinary chat falls back to `carry` (utility 0); grounded evidence supports
   `leave` (utility 4) in the declared dry state. The improvement claim is
   explicitly limited to unseen question text for fixed known targets and
   this one declared decision. No action is executed. The focused workshop,
   decision, and publication-failpoint process checks, full Cargo workspace,
   and `git diff --check` passed. CTest initially failed because the generated
   decision source could not find `std/decision` from CTest's build-directory
   working directory; setting its standard-library path fixed that failure,
   and the focused CTest rerun passed. The workshop now needs a source compiler
   for its decision case, so the documented command uses `build/lana` with its
   adjacent compiler artifact; that command passed. Release qualification was
   not run.

## Next batch: finite BROJA qualification

5. Added a full-support 2-by-4-by-4 finite law with an independently known
   optimum: a feasible coupling makes one source a permutation of the other,
   so the minimum joint target information equals either single-source mutual
   information. The new fixture checks the returned certificate against that
   bound and source-swap symmetry. The existing harder 4-by-4 fixture still
   reports `unconverged`, with feasible marginal and mass residuals and no
   uncertified components. Increasing the general solver's iteration bound
   did not improve its certificate, so that experiment was reverted. All nine
   registered BROJA CTest cases and `git diff --check` passed. No solver
   behavior or published bytecode changed; release qualification was not run.

## Remaining work and order

1. Core-routed dataset calculations, evidence records, durable snapshots, and
   atomic incremental updates. Reuse the finite codec, without silently dropping
   unsupported metadata or inventing relationships.
2. Symbolic rule search, corrections/activation/rollback, tree families, and
   walk-forward evaluation with independent holdouts.
3. Semantic retrieval and learned context selection with frozen Brain parameters
   and explicit held-out activation gates.
4. Read-only SQLite datasets, text/Markdown extraction, and exact-version GitHub
   Releases source packages, each as a separate bounded batch.
5. Additive value/class/interface compiler and VM work, preserving existing source
   and published bytecode compatibility and running bootstrap/conformance gates.
6. Exact-candidate release gates only after implementation and local acceptance:
   fuzz duration, clean universal install, integration environment, paired
   performance, and artifact checks. Publication is not part of this checkpoint.

The new tests prove the included finite fixtures, not general intelligence,
semantic retrieval, learned programs, arbitrary inference, or completion of the
remaining requirements. Do not remove a requirement based only on this checkpoint.

## Dataset aggregate boundary

6. Aggregate materialization now rejects malformed descriptors, non-finite
   numeric cells, arithmetic overflow, and empty min/max groups before
   publishing a result. It folds uncertain numeric cells through Core's pure
   lift, retaining same-root worlds and weights; unrelated roots fail with
   `LANA_ERR_UNSUPPORTED_OPERATION`. Inspect presents unique support values
   and summed weights while the internal worlds stay aligned for later pure
   operations. Dataset host calls accept unresolved cells; definite predicate
   and key checks remain at materialization. Rust and source regressions cover
   malformed descriptors, overflow, weighted sum/mean/min/max, same-root
   reuse, and unrelated-root rejection. The full Cargo workspace passed.
   The full CTest run passed 103/104: the is-ought study exposed duplicate
   values in inspection. After fixing inspection, the affected seven CTest
   cases passed, including the study. The other 97 were not rerun after that
   display-only fix. `git diff --check` passed. Dataset evidence records, snapshots, and updates remain
   pending; this batch does not qualify the release.

## Dataset source registration

7. Added `dataset_source(source_id)` at the reserved host ID 200. After
   `store_open`, it commits an empty named source with separate ordered
   `row_ids` and `rows` fields. Repeating the registration, including after
   reopening the store, does not advance the revision. Empty or over-128-byte
   IDs and a store with unrelated staged writes fail without a source commit.
   The bytecode verifier still rejects pending IDs 201–205. The compiler
   bootstrap was regenerated and twice-verified byte-stable under its fixed
   limits. The full Cargo workspace and 106/106 CTest cases passed, as did
   `git diff --check`. Query binding and source-row evidence are next;
   snapshots and updates remain pending. This is local implementation, not
   release qualification.

## Dataset plan execution boundary

8. Added VM execution of a named pure dataset plan with ordered arguments,
   shared instruction, memory, cancellation, and call-frame limits. The plan
   must return a row array; missing names, wrong arity, effects, and non-array
   results fail without publishing a result. Dataset operators are available
   inside the pure guard, while mutation and external host calls remain
   blocked. The existing one- and two-argument nested function paths now use
   the same runner. All 212 VM library tests and `git diff --check` passed.
   Store-host query registration, durable snapshots, evidence, and updates
   remain pending; no release qualification was run.

9. Changed the durable host-call extension to receive the executing VM. A
   runtime host can now evaluate a named pure plan in that same VM before a
   store commit; the callback is restored after nested execution. Updated the
   CLI, Python worker, REPL, and Wasm callers, and added a VM regression that
   executes a plan through the extension. The full Cargo workspace passed and
   `git diff --check` passed. Exact loaded-bytecode digest capture and dataset
   evidence instrumentation must precede public `dataset_query` registration;
   no query or snapshot is published by this bridge alone.

10. Moved LABC serialization into the bytecode crate so generated chunks have
    one encoder, while loaded CLI and Python-worker chunks retain their exact
    original bytes. Each store host now holds those bytes; the query-plan
    digest hashes them followed by the little-endian UTF-8 plan-name length
    and name. A fixed digest vector and encoder/loader round trip passed.
    The full Cargo workspace, native build, 106/106 CTest cases, and
    `git diff --check` passed. The digest is ready for query registration,
    which still needs row evidence and atomic snapshot publication.

11. Reused the finite tagged-value codec for definite dataset source rows and
    typed keys. Source-row maps now have a canonical encode/decode boundary:
    finite number bits and sorted map entries round-trip exactly; noncanonical,
    corrupt, cyclic, and unsupported values fail. Typed key bytes distinguish
    a number from its display-equivalent string and normalize negative zero.
    The focused codec regression, full Cargo workspace, and `git diff --check`
    passed. This codec is not yet connected to `dataset_apply`; explicit
    immutable Information cells and row lineage remain pending.

12. Added source-only `dataset_apply` at host ID 202 while ID 201 remains
    rejected. It accepts ordered add/correct/delete maps for a registered
    source, keeps correction position, validates canonical definite row
    payloads, caps source rows at 10,000, and commits the source plus a
    batch receipt in one store revision. Identical retries return the prior
    revision even after reopen; changed retries, stale revisions, duplicate
    IDs, unsupported cells, and an existing query registry record fail
    without a source commit. The compiler bootstrap was regenerated and
    twice-verified byte-stable. Full Cargo workspace and 107/107 CTest cases
    passed before a linear-time row-indexing adjustment; its focused Rust,
    source, and bootstrap checks passed afterward, along with
    `git diff --check`. Query registration, dependent reruns, evidence, and
    historical snapshots remain pending; no release qualification was run.

13. Added native `document_extract(path, format)` at host ID 219 for explicit
    UTF-8 text and Markdown. It retains original byte spans, CRLF line
    accounting, heading paths, fenced-code chunks, and UTF-8-safe 4,096-byte
    splits; it rejects NUL, invalid UTF-8, unsupported formats, input above
    16 MiB, and output above 100,000 chunks before publishing a result.
    WASM reports unsupported operation. The compiler bootstrap was regenerated
    and twice-verified byte-stable. The full Cargo workspace, 108/108 CTest
    cases, and `git diff --check` passed. At this checkpoint SQLite remained
    pending because its public result shape needed an explicit contract.

14. Added native `dataset_sqlite(path, sql, parameters, schema)` at host ID
    218. The source contract now fixes `{rows, source_revision, evidence}`;
    Information cells return validated canonical tagged JSON text instead of
    live roots. A read-only connection pins one snapshot, authorizes only
    ordinary-table reads and SQLite built-in functions other than
    `load_extension`/`sqlite_log`, rejects multi-statement and non-row SQL,
    binds typed parameters, validates exact schema and unique IDs, and
    checks typed integers, tagged Information, row/input/output caps, and
    complete ordered digest before exposing a result. WASM reports unsupported
    operation. A temporary-database source fixture checks independent revision
    bytes, row order, nulls, evidence, rejection cases, and a concurrent WAL
    writer with an uncommitted update. The compiler bootstrap is byte-stable;
    full Cargo workspace, 109/109 CTest cases, and `git diff --check` passed.
    Dataset query registration, dependent reruns, and durable historical
    snapshots remain pending; no release qualification was run.

15. Added the named pure dataset-plan work budget to the Rust VM. One run
    shares 5,000,000 steps across source rows, callbacks, sort/group/join
    comparisons, and aggregate cell visits; the plan also rejects over
    100,000 output rows. The limits are checked before lazy-source and join
    result allocation where possible, and exhaustion leaves the caller's
    output unchanged. Existing unregistered in-memory dataset calls retain
    their prior behavior. The focused VM regression and full Cargo workspace
    passed. The 109/109 CTest suite passed before a one-line join preallocation
    guard move; the focused final-tree VM test and 11 dataset/bootstrap CTest
    cases passed afterward, along with `git diff --check`. Public query registration,
    row evidence, atomic reruns, and durable snapshots remain pending.

16. Added a read-only runtime evaluation path for named pure dataset plans.
    It captures the current store revision once, loads every declared source
    from that revision, validates exact source records, unique row IDs, and
    canonical typed row bytes, and passes ordered source datasets to the VM.
    It retains the source/row ID lists for the later evidence pass and returns
    no partial result on corruption or plan failure. Repeated source IDs are
    allowed; source order survives store reopen. Named dataset plans now also
    reject `array_push` so a callback cannot mutate an input array. The
    focused corruption/reopen/purity regressions, full Cargo workspace,
    109/109 CTest cases before the purity guard, 11 final-tree dataset/bootstrap
    CTest cases, and `git diff --check` passed. Public query registration,
    row-level evidence, atomic dependent reruns, and durable snapshots remain
    pending; no release qualification was run.

17. Added ephemeral row-level provenance for named pure dataset plans. The
    runtime labels each decoded source row with length-delimited source/row
    IDs. Filter, map, select, limit, sort, group, aggregate, and join retain
    row derivation links; filter-false, limit-excluded, and unmatched join
    sides record ordered decisions. Failed plans clear decisions, and ordinary
    unregistered in-memory datasets keep their prior behavior. Focused VM and
    runtime regressions, the full Cargo workspace, and 109/109 CTest cases
    passed. Stable persistent derivation IDs, typed cell derivations, snapshot
    validation, public query registration, atomic reruns, and historical
    snapshot reads remain pending; no release qualification was run.

18. Source-row cells now inherit the source derivation without mutating a
    shared input map. Named-plan mapped cells without their own derivation
    inherit the map-row link, and aggregate result cells carry an operation
    link, including definite counts. Filter exclusions retain their Boolean
    result. Named plans reject non-map output rows and clear decisions on
    failure. Focused VM/runtime checks, full Cargo workspace, CMake build,
    and 109/109 CTest cases passed. This is still in-memory evidence:
    persistent IDs, complete typed cell serialization, validated snapshots,
    public query registration, atomic reruns, and historical reads remain
    pending; no release qualification was run.

19. The read-only named-plan evaluator now clears the VM's previous
    exclusion decisions before validating or loading sources. A failed
    source lookup cannot leave decisions from the preceding successful run
    available for misattribution. The focused runtime regression, full Cargo
    workspace, rebuilt CLI, 109/109 CTest cases, and `git diff --check` passed;
    public query registration and durable evidence remain pending.

20. Named dataset plan rows now carry deterministic source, join, and typed
    group-key paths through later operators. A runtime identity pass assigns
    SHA-256 derivation IDs from query ID, plan digest, source revision,
    deterministic traversal path, and ordered row paths. It checks duplicate
    or unattributed output rows and caps the graph at 100,000 nodes. Fresh-VM
    join/aggregate reruns and store reopen checks reproduce the same paths
    and IDs; changing query, digest, or revision changes derivation IDs.
    The full Cargo workspace, rebuilt CLI, 109/109 CTest cases, and
    `git diff --check` passed.
    Canonical typed snapshot bytes, reload validation, query registration,
    atomic reruns, and historical read APIs remain pending.

21. The read-only named-plan evaluator now encodes `dataset_snapshot_v1`
    bytes with ordered rows, typed finite Information cells, source-row
    exclusions, and a topologically ordered derivation DAG. Reload checks
    the schema, canonical bytes, row and node references, source revision,
    plan digest, and finite values before returning the record. Fresh join
    and aggregate runs round-trip through the codec; a source snapshot is
    byte-stable after store reopen. The full Cargo workspace, CMake build,
    and 109/109 CTest cases passed, followed by the focused final-tree
    join/aggregate round-trip test. Query registration, atomic persistence,
    and historical reads remain pending; no release qualification was run.

22. `dataset_query` now accepts a named pure plan through host ID 201,
    evaluates registered sources, and commits its query record and canonical
    initial snapshot in one store revision. A matching call after reopen
    validates the saved snapshot and binds without writing; reusing a
    calculation version with changed plan bytes or source list conflicts.
    Failed evaluation leaves no staged record. The compiler and verifier
    accept the call, and the checked bootstrap artifact was regenerated.
    The full Cargo workspace, CMake build, 110/110 CTest cases (including
    the repeated bootstrap check), a final populated-source round-trip
    test, and `git diff --check` passed. Source updates still refuse any
    registered query until atomic dependent reruns are implemented;
    historical snapshot/evidence calls remain pending. No release
    qualification was run.

23. `dataset_apply` now evaluates every registered query naming the changed
    source against the proposed source rows at the candidate global revision.
    It requires each affected query to be bound to the saved plan after
    reopen, then stages source, query records, canonical snapshots, and batch
    receipt for one store commit. A failed rerun or unbound query leaves the
    prior revision unchanged; successful retries return the saved receipt.
    The store checks the combined 256 MiB journal payload limit before
    writing. Two-query atomic rerun, fresh full-rerun byte equality,
    reopen/rebind, and failed-rerun recovery tests passed, as did the full
    Cargo workspace, CMake build, 110/110 CTest cases, and `git diff --check`.
    Public historical snapshot/evidence calls remain pending; no release
    qualification was run.

24. Added `dataset_snapshot`, `dataset_evidence`, and `dataset_exclusions`
    at host IDs 203–205. Snapshot reads resolve the query record visible at
    the requested global revision, validate canonical typed bytes, and work
    after reopen without rebinding. Evidence returns the chosen row's
    reachable derivation nodes and ordered source rows; exclusions return
    saved decisions. Focused tests cover old/current reads, a missing output,
    filtered exclusion, corrupt current snapshot, and compacted history.
    The source-level fixture exercises all three calls. The checked compiler
    bootstrap was regenerated; the full Cargo workspace, CMake build,
    110/110 CTest cases, and `git diff --check` passed. All six
    dataset-history entrypoints are present locally; broader requirements
    and release qualification remain open.

25. Reconciled the bytecode spec with the implemented SQLite host ID 218.
    Added the first bounded `std/rules` slice: `learn` and `predict` use
    reserved host IDs 206–207, charge native predicate work to the VM budget,
    validate finite task/examples/options, enumerate bounded Boolean rules,
    retain training mistakes, and report held-out accuracy, explicit search
    exhaustion, matching clauses, and needed-missing-input status. Focused
    fixtures cover an unsupplied conjunction, disjunction, contradictory
    examples, deterministic replay, both search caps, overlapping IDs, and a
    wrong-kind prediction. The checked compiler bootstrap was regenerated;
    the full Cargo workspace, CMake build, 114/114 CTest cases, and
    `git diff --check` passed. Rules 208–211, durable rule versions and
    corrections, trees, walk-forward, semantic retrieval/selection, hosted
    packages, the object model, and exact-candidate release qualification
    remain open.

26. Added the durable symbolic-rule calls `save`, `add_counterexample`,
    `inspect`, and `rollback` at host IDs 208–211. The record includes ordered
    examples and versions, tagged scalar values, binary64 metric bits, a
    SHA-256 digest, active pointer, and sorted retry receipts. Reload validates
    every version, held-out trace, parent link, active status, receipt payload,
    and digest before returning it. Correction requires fresh IDs, reruns the
    bounded learner, keeps an imperfect candidate inactive, and returns the
    recorded version on an exact retry; rollback retains history. `predict`
    accepts an inspected record and reports its active version. The
    fresh-process fixture covers save, reload, corrected activation, exact
    retry, changed-content conflict, inactive candidate, rollback, and
    tampered-report rejection. Focused tests, compiler bootstrap, the Cargo
    workspace, all 115 CTest cases, and `git diff --check` passed on this
    candidate. This does not complete the rule contract's full search/report
    validation, trees, walk-forward evaluation, Brain retrieval/selector,
    hosted packages, object model, or release gates.

27. Added the first complete `std/trees` call surface at host IDs 212–216:
    CART, seeded PCG32 forests, boosted regression, binary boosted
    classification, prediction, explanation, and versioned save/load.
    Fitting validates exact task/example/options shapes, keeps holdout rows
    out of split selection, records split and leaf nodes with bit-encoded
    gains/values, and charges native work to the VM budget. Saved tree records
    use tagged scalar examples and split constants, binary64 metric bits,
    canonical JSON, a digest, and historical revisions. Loading refits each
    version from its ordered inputs and rejects changed models or metrics.
    Focused fixtures cover midpoint, category and missing branches, forest
    seed replay, both boosted tasks, fresh-process save/reload and inactive
    second version, and malformed nodes. The checked compiler bootstrap,
    focused CTest cases, full Cargo workspace, all 121 CTest cases, and
    `git diff --check` passed on this candidate. Broader holdout and corruption
    matrices and exact-candidate release qualification remain open.

28. Added the `std/evaluation.walk_forward` host call at ID 217 with
    chronological folds, target-availability filtering, internal validation,
    fresh pure fit/predict callbacks, model digests, classification/regression
    metrics, overlapping-test disclosure, and bounded report construction.
    Source fixtures cover deterministic separate-process replay, gap and
    overlap handling, classification scoring, future-feature rejection,
    effectful callback rejection, insufficient evidence, and the 100-fold cap.
    The compiler bootstrap passes at the production 50,000,000-instruction
    limit. The full Rust workspace and all 127 CTest cases pass. Pruned
    completed compiler/publication, Brain fit/layers/memory/grounded/workshop/
    recovery, paired Brain evaluation, finite-information, SQLite, and
    document-extraction contracts from `REQUIREMENTS.md` after checking
    their existing regression coverage. Full-range u64 fold seeds use canonical
    decimal strings at the callback boundary; the verified walk-forward
    contract was also removed from `REQUIREMENTS.md`.

29. Added opt-in Brain semantic indexing and chat using frozen WordLevel
    embeddings over current facts, aliases, and definite typed memory. The
    index rejects stale Brain, tokenizer, or corpus data; labeled development
    and disjoint held-out fixtures gate exact answers, with abstention and
    source references otherwise. The CLI writes the index atomically and
    reports paired held-out outcomes. The focused index test, full Cargo
    workspace, 128/128 CTest cases, and `git diff --check` passed. Removed
    the verified semantic-retrieval contract from `REQUIREMENTS.md`; learned
    context selection and other active requirements remain there.

30. Added learned Brain context selection through `brain compress fit` and
    semantic chat's `--selector` option. Five deterministic F32 logistic
    epochs train separate weights over the frozen index; validation enforces
    disjoint questions/records, complete relevant-record retention, unchanged
    grounded outcomes, and the baseline context-size bound. Required exact
    targets and forecast/action references remain intact, even for conflicting
    or unresolved targets. Schema validation, stale-digest rejection, work and
    context caps, and atomic publication protect both the saved Brain and the
    prior active selector. A failed gate saves a separate inactive report.
    Focused tests cover replay, SGD, irrelevant duplicates, late references,
    source labels resembling fact IDs, semantic exactness, conflicts, limits,
    leakage, corrupt/stale artifacts, and publication failures before and after
    rename. The final CMake build, full Cargo workspace, 129/129 CTest cases,
    opt-in publication-failure checks, and `git diff --check` passed. Removed
    the completed selector contract from `REQUIREMENTS.md` and stopped after
    this implementation as requested. Release qualification remains separate.

31. Added exact GitHub Releases source packages: reproducible `package pack`,
    bounded HTTPS `package add`, strict manifest/archive validation, recursive
    exact-version closure resolution, canonical locks, and content-addressed
    cache installation. Builds preserve hosted locks, include them in cache
    keys, and verify archived and extracted bytes before compiler resolution.
    `pkg/owner/repo/src/...` imports require a verified lock entry; relative
    traversal and direct imports from unused cache directories are rejected.
    Existing local dependencies still affect the compiled cache. The package
    publication template and helper qualify a clean extraction, restrict
    write permission to publication, check protected-tag identity, and refuse
    changed existing assets. They were tested locally with mocked publication;
    no workflow was deployed and no release was published.
    The final CMake build, Cargo workspace, 131/131 CTest cases, local HTTP
    failure/atomic-publication fixtures, publication-guard tests, workflow YAML
    parse, and `git diff --check` passed. The public boundary smoke downloaded
    GitHub CLI v2.63.2's 13,069,957-byte Linux archive and verified its published
    SHA-256 `912fdb1ca29cb005fb746fc5d2b787a289078923a29d0f9ec19a0b00272ded00`.
    Removed the completed hosting contract from `REQUIREMENTS.md` and stopped
    after this implementation. Package-repository publication and Lana release
    qualification remain separate actions.

32. Closed the remaining grounded Brain uncertainty boundary. Unresolved root
    replies now retain their original law, observation context, and latest
    derivation reference without changing the null answer/target contract.
    Expanded the separate-process memory fixture to cover Possibility roots,
    equal but unrelated marginals, explicit singleton refinement, correlated
    joint resolution, missing targets, and failed-refinement byte preservation.
    `cargo build --locked -p lana-cli`, the complete Brain memory fixture, and
    `git diff --check` passed. Removed that completed requirements section.
    Next: dataset/rule/tree contract closure, then the object model and final
    qualification; no release claim is made by this checkpoint.

33. Completed the bounded-rule search/report boundary. Enumerate every allowed
    clause partition in atom-count/order, reject contradictory clauses over
    declared domains, retain every missing training ID, and preserve finite
    midpoints for opposite extreme numbers. `known_facts` now defaults to an
    empty map as specified. Report validation replays the bounded learner,
    rejecting forged search counts, rejection reports, and reordered rules;
    exhausted searches with no usable rule also round-trip. Two Rust regressions,
    both source learning fixtures, fresh-process rule save/correction/rollback
    fixture, and `git diff --check` passed. Next: tree and dataset acceptance.

34. Closed tree input-schema and evidence validation gaps. Raw models retain
    their full feature schema, including unused features and single-leaf trees;
    prediction rejects unknown names and wrong kinds. Regression metric and
    boosted-score overflow fails explicitly. Enabled serde_json float_roundtrip
    to preserve binary64 reports: a forest RMSE previously changed by one bit
    and caused freshly fitted reports to fail their own validation.
    Two matrix regressions exercise all six family/problem combinations,
    holdout independence, insufficient evidence, VM limits, one-round numeric
    results, and corruption with recomputed digests. All four source tree
    fixtures, the durable tree fixture, and focused Rust tests passed.
    Dataset audit found a substantive dependency: source-row encoding is still
    definite-only. Immutable Information snapshots must precede uncertain
    dataset-source persistence and final dataset acceptance. The revised order
    is snapshot semantics/runtime, source codec and replay, remaining object
    support, then exact-candidate qualification.

35. Added the shared `snapshot(info)` prerequisite through host ID 220, typed
    compiler lowering, and VM deep capture. It detaches live updates, preserves
    law/guards/exactness/provenance/revision, and freezes nested arrays and maps.
    Mutation and unsupported executable/effectful payloads fail explicitly.
    Source success/failure fixtures and a Joint/Paths/container/resource unit
    check pass. Bootstrap initially reached the existing 50,000,000-instruction
    limit; consolidating host names into the existing registry string reduced
    compiler work without changing the limit. A generated compiler rebuilt
    itself byte-identically. Dataset uncertain source encoding and the object
    model remain unfinished; this prerequisite alone does not close them.
    Integrated snapshot checks: Rust workspace and 133/133 CTest cases passed,
    including the twice-repeated native bootstrap under unchanged limits.
    Compatibility follow-up retains rules-v1 replay while new searches use
    rules-v2, and recovers older saved tree schemas from their validated tasks.
    Focused legacy save/load and all-family replay checks pass.
    Final compatibility candidate: CMake build, full locked Rust workspace,
    and 133/133 CTest cases passed (119.22 seconds), including source snapshot
    mutation rejection and bootstrap reproducibility. `git diff --check` passed.
    Logs: /tmp/lana-closure-final-build.log, /tmp/lana-closure-final-cargo.log,
    /tmp/lana-closure-final-ctest.log. This is partial execution of the full
    plan: uncertain dataset persistence, value/class/interface implementation,
    and all exact-candidate release gates still remain.


36. **Captured uncertain dataset source rows and history — complete.**
    Reused the canonical tagged cell encoder for finite Possibility,
    Distribution, and Joint captures, including nested containers and aligned
    worlds with repeated outcomes. Source rows preserve the complete captured
    derivation DAG, exactness and revision. Source/batch-scoped identities and
    saved receipt bindings preserve shared dependencies across rows, sources,
    subsequent batches and restart retries without correlating independent
    equal marginals. Historical result labels remain evidence only.
    Query reads share one decode context; containers reload frozen and nodes
    receive fresh VM-local identities. Live roots, corrupt payloads/evidence,
    incompatible relationships, and failed reruns publish nothing. Definite
    source records and existing receipt hashes retain their original encoding.
    Fixed snapshot cloning to preserve repeated references to a declared joint.
    The VM admits uncertainty only in dataset_apply's changes argument, where
    the source codec enforces explicit capture.

    Verification: 14 focused runtime dataset tests passed; CLI subprocesses
    covered initial publication, restart retry, later-batch reuse, independent
    root rejection, live-root rejection, correction, deletion and historical
    reads. Saved bytes match clean full reruns after add/correct/delete/reopen.
    Codec checks cover finite joints, nested captures, preserved world alignment,
    frozen containers, provenance and corrupt DAG/binding/value rejection.
    CMake build, full locked Rust workspace, and 134/134 CTest cases passed
    (115.78 seconds), including the twice-repeated native bootstrap. Published
    bytecode fixtures are unchanged. git diff --check passed.
    Logs: /tmp/lana-dataset-build.log, /tmp/lana-dataset-workspace.log,
    /tmp/lana-dataset-ctest.log. Removed completed dataset requirements.
    Next: additive value/class/interface implementation. Exact-candidate release
    qualification remains separate and has not been performed.


37. **Object descriptor/bytecode foundation — complete; object execution pending.**
    Added LABC v6 opcodes 83–89 without renumbering existing instructions.
    The loader, assembler and disassembler support the new encoding; v1-v5
    still reject those opcodes. Canonical typed descriptors validate field and
    method schemas, nominal references, overload families, interface signatures
    and declared effect-mask compatibility, initializer/default arities,
    constructor/register bounds, private access, and unique member ownership.
    Function-local control flow prevents jumps/fallthrough into owned bodies;
    direct CALL, LOAD_FUNCTION, fork, async, generator, lazy and bootstrap
    references cannot enter those bodies. Initializer functions cannot be
    aliased as ordinary methods/defaults. V6 string constants use strict UTF-8,
    and its assembly version directive precedes instructions/constants.

    This does not implement object values, class instances, method execution,
    source declarations, or inference of actual body effects. All v6 execution
    is blocked centrally before any instruction, including debugger steps and
    callback runners. Preserve that guard until dynamic receiver/type checks,
    construction/privacy rules and inferred effects are implemented. Named
    function entry through host APIs also needs runtime ownership checks.

    Verification: 27 bytecode tests passed, including malformed descriptors,
    privacy/ownership bypasses, interface signatures/masks, construction and
    static/interface operand checks, binary round trips and old-version rejection.
    VM tests prove run/debug-step refusal before the first register write.
    The CLI check assembles/verifies/disassembles v6 and rejects execution before
    an earlier PRINT can run. Final CMake build, locked Rust workspace and
    135/135 CTest tests passed (119.21 seconds), including frozen compatibility
    fixtures and twice-repeated self-hosting. git diff --check passed.
    Logs: /tmp/lana-objects-build.log, /tmp/lana-objects-workspace.log,
    /tmp/lana-objects-ctest.log. Removed completed bytecode-definition/static
    validation work from the remaining compatibility step in REQUIREMENTS.md;
    object-model acceptance remains open. Next: object construction and method
    runtime, then source/compiler/tooling integration and release qualification.

38. **Public immutable value runtime — complete; remaining object runtime pending.**
    Added a distinct nominal ValueKind with private Rust field storage.
    V6 VALUE_NEW and OO_GET now execute for public, method-free values, with
    runtime descriptor/type checks and publication only after all fields pass.
    Construction reuses snapshot copying/freezing, preserves STATE components
    and captured finite Information, rejects live references, cycles, effectful
    handles and executable payloads, and enforces work/depth/memory limits.
    Nested immutable values and STATE_DIST trees can share immutable storage.
    Equality checks all fields for supported equality, compares nominal identity
    and fields, preserves existing array identity equality, and rejects unsupported
    fields even when an earlier field differs. Display exposes only the type name.

    Class/interface/method-bearing chunks still fail before their first
    instruction, including debugger steps. Tensor/Shape fields, member-owned
    access/construction, method execution and effects, classes, interface dispatch,
    value export, and object source syntax/tooling remain pending. V6 now retains
    v5 Core sampling and observation rules instead of taking legacy branches.
    Updated support documentation and narrowed remaining runtime requirements;
    source-language acceptance criteria remain open.

    Focused verification: five immutable-value VM tests and CLI assembly,
    verification, execution, STATE round trip, wrong-type/receiver rejection,
    private construction rejection, old-version rejection, and fail-before-effect
    class rejection passed. Final build and locked Rust workspace passed,
    including 225 VM tests. Final CTest passed 135/135 tests (119.35 seconds),
    including self-hosting, frozen compatibility fixtures and the new CLI check.
    git diff --check passed.
    Logs: /tmp/lana-values-build.log, /tmp/lana-values-tests.log,
    /tmp/lana-values-workspace.log, /tmp/lana-values-ctest-final.log.
    Exact-candidate release qualification has not been performed.

39. **Tensor and Shape immutable fields — complete.**
    Public method-free value construction accepts Tensor payloads and validates
    Shape fields with the existing tensor_shape_from_array helper. Shapes are
    copied and frozen. Immutable Tensor storage is shared without changing dtype,
    shape, view strides, or complex data; snapshot(Tensor) uses the same boundary.
    Tensor field equality remains unsupported. Wrong payloads, invalid dimensions,
    and excessive rank fail before changing the destination. Removed Tensor/Shape
    field work from REQUIREMENTS.md; object source syntax remains pending.

    Verification: the focused constructor regression covers all five dtypes,
    transposed views, independent arithmetic results, frozen shape copies, invalid
    dimensions/rank/types, and unchanged destinations. CMake build and locked Rust
    workspace passed, including 226 VM tests. Five focused CTest checks passed
    (4.03 seconds): snapshots, mutation rejection, bootstrap/self-hosting and the
    CLI object round trip with invalid-shape rejection. git diff --check passed.
    Logs: /tmp/lana-value-fields-build.log, /tmp/lana-value-fields-workspace.log,
    /tmp/lana-value-fields-tests.log, /tmp/lana-value-fields-ctest.log.
    The full CTest suite and release qualification were not rerun for this slice.
    Next: member execution/effect checks, then class/interface runtime and object
    source/compiler/tooling integration.

40. **Checked pure value methods — complete; broader methods remain pending.**
    OO_CALL and OO_STATIC_CALL now dispatch value methods with packed-argument,
    nominal receiver, visibility, argument-type and result-type checks. Owning
    frames may construct/read private fields and invoke private methods. A failed
    call or result check does not publish a partial return. V6 method and ordinary
    call frames are charged to the VM memory budget; register sizing respects
    declared v6 frame bounds and does not treat descriptor constants as registers.
    Ordinary calls copy arguments directly into callee storage.

    The runtime preflight proves the supported zero-effect subset through every
    method and its transitive ordinary/object calls. Unsupported opcodes, hosts,
    and nonzero masks reject the chunk before any instruction, including debugger
    entry. Instruction ownership and frame ownership must agree, preventing named
    host/callback entry from acquiring member privileges. Recursive call graphs
    are checked without recursive verifier traversal.

    General effect inference, additional pure operations, live Information method
    arguments, uncertain-receiver lifting, class/interface runtime, and object
    source/compiler/tooling remain pending. Generic value export remains pending.
    Removed completed member-owned value construction/access and value-member
    entry protection from the remaining runtime requirements; source contracts
    remain open.

    Focused tests passed for private factories and nested private instance calls,
    wrong arity/receiver/argument/result types, unchanged return destinations,
    memory exhaustion, direct/transitive effect rejection before run/debug writes,
    and named-entry/instruction-location bypass prevention. CLI tests passed for
    factory/read round trips, effect rejection before PRINT and result-type errors.
    Final build and locked Rust workspace passed, including 229 VM tests.
    Full CTest passed 135/135 (115.89 seconds), including frozen compatibility,
    twice-repeated compiler bootstrap, project/LSP/debugger workflows and the
    extended object CLI test. git diff --check passed.
    Logs: /tmp/lana-value-methods-build.log, /tmp/lana-value-methods-tests.log,
    /tmp/lana-value-methods-workspace.log, /tmp/lana-value-methods-ctest.log.
    Release qualification was not performed.

41. **Class runtime foundation — complete; source and collection remain pending.**
    Implemented task-local identity, defaults, transactional initialization,
    fixed/public/private field checks, mutation, and task graph transfer preserving
    aliases/cycles with fresh identities. Live fields capture snapshots. Failed
    construction leaves destinations unchanged. Class storage uses weak references
    and task-lifetime arenas; incremental reclamation remains open.
    Verification: 233 VM tests and CLI OBJECT_BYTECODE_PASS, including construction
    failure, identity, cycle transfer, live capture, and teardown. Logs:
    /tmp/lana-classes-vm-final.log and /tmp/lana-classes-build-final.log.
    Next in the same authorized run: method effects and Information handling,
    interfaces, source compiler, tooling/export and full acceptance.

42. **Method runtime effects and Information mapping — complete.**
    V6 checks opcode/host effects through direct calls and enforces every active
    member's promise across indirect callbacks. Defaults/initializers retain their
    stricter rules. Typed methods accept live Information arguments. Pure value
    receiver maps reuse pointwise traversal for definite captures, possibilities,
    distributions and paths, and preserve named Joint coordinates/row weights.
    Live receiver results participate in transactional revision recomputation.
    Mutable/live extra inputs to uncertain receiver replay fail explicitly.
    Verification: full locked workspace passed (235 VM tests); final focused
    object tests passed 11/11 after added Joint/Paths and provenance checks.
    git diff --check passed. Logs: /tmp/lana-methods-workspace.log and
    /tmp/lana-methods-final.log. Next: interface runtime in this same run.

43. **Interface runtime — complete.**
    OO_AS_INTERFACE checks explicit implementation and preserves the underlying
    value/reference. OO_CALL selects a concrete method by the exact promised
    signature; descriptor verification checks visibility, results, every overload,
    and effect bounds. Interface-typed fields/arguments/results accept only explicit
    implementors. Construction completeness and task ownership remain enforced.
    Verification: 236 VM tests passed, including value/class interface dispatch,
    identity preservation, and failed conversion without publication. Release CLI
    rebuilt; OBJECT_BYTECODE_PASS passed. git diff --check passed.
    Logs: /tmp/lana-interfaces-vm.log, /tmp/lana-interfaces-build.log,
    /tmp/lana-interfaces-cli.log. Next: self-hosted source compiler.

44. **Self-hosted object source compiler — complete.**
    Added contextual declarations, typed members and top-level interoperation,
    constructors, field/method access, overload checks, interface promises,
    private/static factories, and same-module blueprint copying/replacement.
    Copied bodies and Self are rechecked against the final child shape. Imports
    retain original ownership. Object emission uses v6 with complete register
    bounds and portable module identities; old source keeps version selection.
    Compiler hot paths and byte classification were simplified to keep bootstrap
    within the unchanged 256 MiB / 50,000,000 instruction limits.
    Verification: OBJECT_SOURCE_PASS covers success and rejection cases, including
    STATE fields, imported interfaces, task cycles, overload replacement, Self
    parameters, privacy, effects and contextual identifiers. Portable identity
    relocation check passed. Compiler reproduces its output byte-for-byte.
    Six focused CTest checks passed before the final portable-ID refinement;
    final source tests and git diff --check passed afterward.
    Logs: /tmp/lana-object-source-ready.log, /tmp/lana-oo-source-ready.log,
    /tmp/lana-oo-portable-stable.log, /tmp/lana-oo-source-ctest.log.
    Next: tooling/export and full final acceptance in this same run.

45. **Object tooling/export and five-batch acceptance — complete.**
    Public-field value snapshots export through checked JSON; private fields,
    nested private snapshots, and direct class serialization fail without output.
    Reload yields ordinary maps; explicit construction creates a new identity.
    Constructor/default continuations use owned frames, preserve rollback, and
    expose source breakpoints without publishing incomplete candidates.
    LSP resolves local object types and members with distinct identities and
    correct rename spans; unsaved documents retain their import base. Formatter
    idempotence and behavior are tested. Cross-module member navigation/rename
    is not provided. Nested construction and Self field access/assignment pass.
    Full qualification exposed and fixed an emitter fast path bypassing legacy
    kernel/evaluation callbacks, plus the unsaved-document canonicalization bug.
    Failure fixtures now receive the same standard-library path as success cases.
    Verification: 138/138 CTest tests passed in 122.48 seconds, including old
    source, frozen legacy bytecode, object success/rejection, local install,
    LSP/debugger, and twice-repeated byte-stable compiler bootstrap. All workspace
    tests passed (237 VM tests); one existing ignored test remains ignored.
    Rust Release CLI built; Lana reports 4.0.0 / LABC v2 and emits v6 for objects.
    Compiler limits remain 256 MiB / 50,000,000 instructions. git diff --check
    passed. Logs: /tmp/lana-final-ctest.log, /tmp/lana-object-workspace-final.log,
    /tmp/lana-final-build.log, /tmp/lana-final-reproduced.log.
    Completed requirements removed from REQUIREMENTS.md. Class storage remains
    task-lifetime bounded; incremental class collection is still explicitly open.
    No publication, signing, universal release qualification, or performance
    release claim was made by this implementation run.

46. **Cargo build and CMake workflow migration — complete.**
    Cargo assembles the checked compiler artifact for the CLI and WASM.
    tools/build.py provides native builds, explicit prefix installation, and
    universal macOS builds. tests/run.py and tests/workflows.py replace CTest;
    tests/cases.json preserves all 138 original test names and predicates, with
    one additional runner regression. docs/build-migration.md maps every old
    job to its replacement. CI, release packaging, and current instructions use
    the replacements; CMakeLists.txt and cmake/ were deleted afterward.
    Verification: 139/139 tests passed after deletion in 129.23 seconds; Cargo
    workspace tests passed. A clean working-tree source copy without old build
    artifacts passed fresh build/install, standalone WASM tests, and bootstrap
    checks. The universal archive passed checksum/extraction checks and actual
    source execution under both arm64 and x86_64. Workflow YAML and generated
    formula syntax checks passed. Compiler limits remain unchanged.
    Logs: /tmp/lana-without-cmake-tests.log, /tmp/lana-cargo-workspace.log,
    /tmp/lana-clean-migration.log, /tmp/lana-cargo-universal2.log.
    Hosted CI, remote branch-protection changes, and publication were not run.

47. **During-task class memory reclamation — complete (2026-09-27).**
    Added synchronous tracing passes between instructions, outside constructor
    and transfer transactions. Arc ownership identifies roots held by frames,
    host callers, containers, suspended computations, Information history, and
    task results. Unreachable class/container cycles are broken without changing
    live identities. Concurrent task/shared state remains an external root.
    Locked graphs defer collection; scratch memory and traversal use existing
    task budgets, and exhaustion or cancellation fails before sweeping.
    Pressure before construction rechecks objects retained by an earlier pass.
    Collection does not promise bounded pauses; graph lookup is currently linear
    and charged to the instruction budget.

    Class storage and v6 call frames now own heap reservations, releasing their
    charges on reclamation, rollback, or return. Seven focused Rust regressions
    cover cycles, aliases, Information history, suspended computations,
    constructor suspension, task transfer with zero/two workers, failed
    collection, and repeated construction within a 64 KiB heap. The registered
    source regression creates 1,000 cyclic objects under a 1 MiB limit and checks
    a surviving object's identity, cyclic aliases, and method result.

    Final verification: Release build passed; the locked Cargo workspace passed
    (244 VM tests, one existing ignored workspace test); all 140/140 source/CLI
    acceptance checks passed in 121.92 seconds, including twice-repeated compiler
    bootstrap, frozen compatibility, local installation, LSP and debugger checks.
    git diff --check passed. Logs: /tmp/lana-class-gc-build-final.log,
    /tmp/lana-class-gc-workspace-final.log, /tmp/lana-class-gc-suite-final.log.
    Removed the completed collection requirement and stale object-model inventory
    row from REQUIREMENTS.md. Only explicit exclusions remain. Compiler limits
    remain 256 MiB / 50,000,000 instructions. Release qualification and publication
    were not performed.

48. **Standalone cycle reclamation and Neovim compatibility (2026-09-27).**
    Repaired the missing discovery of standalone array/map/set cycles. Budgeted
    weak registrations also seed VM-created reactive nodes, generators, futures,
    and planned effects. Clone placeholders register before recursive copying.
    The tracer uses a budgeted index instead of linear node lookup. Tracking
    charges disappear with their owners; pressure scheduling leaves headroom
    between full traces and preserves the compiler's existing limits.

    The original 10,000-array-cycle reproducer now completes under 1 MiB
    (387,264 live charged bytes at completion); its acyclic control returns to
    zero. New checks cover host aliases, standalone Information/receipt cycles,
    a 20,000-node cyclic graph, and 10,000 mixed container cycles under 1 MiB.
    These establish synchronous reclamation, not bounded pauses or shutdown
    reclamation of every graph.

    Neovim now accepts Lana 3.x/4.x with LABC v2-v6, matching VS Code. Its Lua
    regression covers supported and rejected versions, command failure, and
    cached probing. A real headless Neovim session initialized and stopped LSP.
    That check also exposed replies to LSP notifications; the server now ignores
    unhandled notifications, with a protocol regression in tests/workflows.py.

    Verification: Release build; locked Cargo workspace (247 VM tests, one
    existing ignored workspace smoke); 141/141 acceptance checks in 132.53 s;
    Lua compatibility check; headless Neovim lifecycle; git diff --check.
    Logs: /tmp/lana-collector-build-final.log,
    /tmp/lana-collector-workspace-verified.log,
    /tmp/lana-collector-suite-verified.log.

    Correction to checkpoint 47: the broader collector requirement was cleared
    prematurely. Generations, remembered edges, shared promotion, incremental
    slices, and the Release pause target remain active in REQUIREMENTS.md.
    Historical C measurements in plans/gc-performance.md are now labeled as
    historical; they do not qualify the Rust collector. No release or publication
    claim is made.

49. **Remaining-work inventory (2026-09-27).**
    Added the tracked `plans/requirements-completion.md` matrix and expanded the
    ignored/local REQUIREMENTS.md with every active sweep finding. The matrix
    records implementation boundaries, positive/negative acceptance, fixed Rust,
    HTTP and LSP contracts, and explicit exclusions. Final qualification must
    identify the whole dirty source tree, including required untracked files.
    No remaining implementation or release gate was cleared. `git diff --check`
    passed; REQUIREMENTS.md remains ignored.

50. **Retained Rust values and teardown — ownership sub-batch (2026-09-27).**
    Separated the synchronous graph tracer from instruction execution. Normal
    collection still charges the same task budgets. `Vm::result()` now returns
    `RootedValue`, retaining class storage and registrations beyond VM teardown.
    `Vm::retain_value()` provides checked host-call retention, rejecting foreign
    mutable heaps, incomplete construction, held locks and exhausted budgets.
    Migrated the REPL and task-result lifetime boundary. Added migration details
    in `docs/rust-embedding.md`.

    Teardown releases execution roots before tracing and transfers storage to
    retained result ownership. The last handle retries synchronous reclamation.
    Queued children are detached from their scheduler during owner teardown,
    breaking the no-worker scheduler/child ownership cycle.

    New Rust checks cover retained class fields/identity, last-root reclamation
    of class/container cycles, repeated VM teardown, explicit host-root rejection,
    scratch-OOM preservation/retry, and unjoined no-worker child heap release.
    Compiler limits remain 256 MiB / 50,000,000 instructions.

    Verification: Release build passed; locked Cargo workspace passed (252 VM
    tests; one existing ignored public-GitHub smoke); 141/141 acceptance checks
    passed in 128.25 s, including twice-repeated byte-stable bootstrap, frozen
    compatibility, project/import/LSP/debugger checks and local installation.
    Version output is Lana 4.0.0 / LABC v2. `git diff --check` passed.
    Logs: /tmp/lana-rooted-build-final2.log,
    /tmp/lana-rooted-workspace-qualified.log,
    /tmp/lana-rooted-acceptance-qualified.log.
    Ownership-code digests in /tmp/lana-rooted-code-snapshot.json stayed unchanged
    during the final checks. Git status has 206 entries (204 pre-existing plus
    the new completion matrix and Rust embedding guide); unrelated work remains.
    This is not completion of batch 2 or release qualification. Opaque managed
    handles/edge mutation barriers, reserved collector metadata, guaranteed
    cleanup under sustained scratch OOM/held locks, generational/incremental
    policy and every later completion-matrix batch remain active. No remaining
    requirement was removed, and no publication was performed.

51. **Actual WASM execution — complete local batch (2026-09-27).**
    Refined the remaining order to establish native/WASM verification before
    further collector changes; collector batches 2/3 remain active. Workspace
    feature inheritance now disables native defaults for portable consumers;
    the CLI explicitly enables TLS. Native TLS/dynamic loading/SQLite/ring stay
    out of the wasm32 dependency graph. Native encrypted execution configuration
    remains enabled; portable execution/config host entry points return explicit
    unsupported errors. Embedded compiler builds now include all seven stdlib
    modules in the existing virtual filesystem.

    The WASM runner selects Rustup's matching compiler/toolchain and repairs
    macOS rust-lld library lookup locally. Actual Node checks cover malformed
    source, every stdlib import, representative Core/rules/trees/evaluation
    programs, task joins, repeated cycles, unsupported host calls, memory limits,
    instruction limits, and recovery with a successful subsequent call. Clock and
    sleep no longer invent successful WASM behavior. Added an actual-WASM CI job;
    hosted CI itself was not run.

    Verification: /tmp/lana-wasm-full-conformance.log;
    /tmp/lana-wasm-native-workspace.log (252 VM tests; one existing ignored smoke);
    /tmp/lana-wasm-cli-build.log; /tmp/lana-wasm-native-live.log (native twice-stable
    bootstrap and live HTTPS execution, 2/2 in 35.53 s); dependency graph excludes
    ring/getrandom/rustls/libloading/rusqlite; git diff --check passed.
    This closes the WASM implementation row, not final-tree release qualification.
    Next independent batch: HTTP framing/headers, then remaining integrations and
    collector policy; final qualification remains dependent on every active row.

52. **HTTP framing and headers — complete local batch (2026-09-27).**
    Added a bounded incremental HTTP/1.x parser for fixed, EOF and chunked bodies,
    extensions, separate trailers, informational and bodyless responses. Lowercase
    response names retain ordered duplicate values. Existing request headers now
    reach the wire after pre-connect validation; verify remains TLS-only control.
    Rejects request injection, conflicting lengths, unsupported coding, malformed
    status/fields/chunks, premature EOF and oversized declared bodies. Received
    bytes consume instructions; buffers reserve against the VM heap. Framed reads
    finish without TLS EOF and share one read deadline. URL validation covers
    credentials, ports, IPv6, escapes, query-only paths and fragments.

    Seven Rust parser/live tests pass, including fragmentation, trusted local TLS,
    default untrusted rejection, explicit verify opt-out, timeout and heap failure.
    Six portable tests also pass with TLS disabled. The registered source fixture
    exercises POST headers, duplicate response fields, trailers and injection
    through the actual native compiler/CLI. Locked workspace passed (258 VM tests;
    one existing ignored public-GitHub smoke); Release build and actual Node WASM
    conformance passed. git diff --check passed. Logs: /tmp/lana-http-live.log,
    /tmp/lana-http-no-tls.log, /tmp/lana-http-source.log,
    /tmp/lana-http-workspace.log, /tmp/lana-http-build.log,
    /tmp/lana-http-wasm.log. Synchronous platform DNS remains outside socket
    timeouts, explicitly documented in spec/SPEC.md. This closes batch 5 only;
    workspace editors, packaging, collector ownership/policy and final qualification
    remain active. Next batch: compiler symbol identity and workspace editors.

53. **Workspace compiler symbols and editors — complete local batch (2026-09-27).**
    Compiler queries now retain canonical module paths, lexical declaration
    identities, typed/member references and lexer spans. Parameters, assignments,
    comprehensions and match bindings preserve their declaration locations.
    Imported functions/types and aliases resolve through module exports; ordinary
    compilation does not build the editor index. Replaced linear function-name
    lookup with the existing map and avoided eagerly formatting successful emitter
    assertions. A temporary reduced editor-free bootstrap stage enabled migration;
    it was never installed. The full final compiler then compiled itself and the
    regular acceptance gate repeated byte-stable bootstrap twice within unchanged
    256 MiB / 50,000,000-instruction limits.

    The LSP uses workspace sources plus open-buffer overlays, including unsaved new
    modules. Queries use UTF-16 positions and preserve client URIs. Rename validates
    names/collisions, rejects dependency changes and incomplete analysis, and
    recompiles the complete proposed overlay before returning an atomic edit.
    Server version comes from the package version. Python accepts supported LABC
    v2-v6. Added real Neovim and VS Code extension-host tests in disposable workspaces;
    a completion marker ensures a launcher exit cannot masquerade as a VS Code test.

    Verification: Release build; 142/142 acceptance tests in 126.51 s; locked Cargo
    workspace (258 VM tests, one existing ignored public-GitHub smoke); actual Node
    WASM conformance; object tooling; protocol tests for imports, shadowing, aliases,
    assignments, comprehensions, typed/object references, Unicode, overlays,
    dependency protection and invalid/incomplete rename; Neovim and VS Code live;
    all 58 Python tests with MCP/Jupyter extras installed; git diff --check.
    Logs: /tmp/lana-lsp-release-build.log, /tmp/lana-lsp-acceptance.log,
    /tmp/lana-lsp-workspace.log, /tmp/lana-lsp-wasm.log,
    /tmp/lana-lsp-roundtrip.log, /tmp/lana-lsp-objects.log,
    /tmp/lana-neovim-live-final.log, /tmp/lana-vscode-live-final.log,
    /tmp/lana-lsp-python-release.log. This closes batch 6, not collector policy or
    final qualification. Next: installed-prefix packaging, CI and current docs.

54. **Installed-prefix packaging, CI and current docs — complete local batch
    (2026-09-27).** Installs now include all stdlib assets and the license. The
    root packager delegates to tools/build.py, validates staged execution, prints
    a SHA-256 report, and atomically replaces the archive only after validation.
    Release CI uses the same prefix packager. The registered package check verifies
    checksum, safe extraction, source/stdlib execution outside the checkout,
    license inclusion, and preservation of an old archive when compiler/stdlib/
    license assets are missing.

    Registered publication-failure qualification builds its feature-enabled CLI
    in a disposable Cargo target, tests old/new/absent artifact outcomes, then
    proves the unchanged normal CLI ignores the failpoint switch. Added real
    Neovim/VS Code CI execution and required-check documentation; hosted CI and
    repository protection settings were not changed remotely. Python CI installs
    MCP/Jupyter extras so these tests do not silently skip.

    Replaced the orphan removed-benchmark import with a current paired Release
    harness and a registered method check (alternating order, 30 calls, five
    warmups, explicit backends, checked echo results, first-call reporting,
    compiler/binary digests, and the 5% threshold). Actual paired performance is
    still a final qualification gate. Removed stale active Brain/Core/object
    implementation labels against the passing source/workspace/integration
    coverage. Added nonignored docs/release-checklist.md and updated build/support
    guidance; historical C measurements remain clearly historical.

    Verification: /tmp/lana-binary-package.log,
    /tmp/lana-publication-isolated.log, /tmp/lana-packaging-build.log,
    /tmp/lana-packaging-gates.log (4/4 passed in 13.64 s); workflow YAML parsed,
    Python scripts compiled, shell syntax and git diff --check passed. No commit,
    tag or publication occurred. Remaining work: collector ownership/policy
    batches 2/3, followed by exact-tree qualification batch 8.

55. **Collector accounting and retained workspace — partial ownership batch
    (2026-09-27).** The first current paired run against a newly built, detached
    d102fda Release baseline found VM/compile regressions of 11.3%/13.1%.
    Safepoints were repeatedly locking the heap just to read usage. Usage now
    reads an atomic snapshot published before the allocator lock is released;
    actual allocation admission remains locked. A below-pressure check also
    avoids the cycle-registry lock. The concurrency test still admits only one
    full-budget allocation, and a new test proves usage reads do not wait for
    the allocator lock.

    Host retention and full collection now preserve their heap-accounted graph,
    index and worklist capacities. Every attempt clears graph references, including
    failure paths; an empty collected graph releases its workspace. Shutdown can
    reuse this space without new allocation. A new regression retains wide cyclic
    graphs with 1, 128 and 20,000 children, clones the root, fixes the heap limit at
    current usage, then drops the VM and both roots. Aliases remain usable, no new
    allocation is admitted, and all heap bytes are released. This does not prove
    first-use exhaustion, growth beyond admitted capacity, retained-lock cleanup,
    explicit edge/root tracking or incremental collection; batches 2/3 stay open.

    Verification: Release build; 145/145 source acceptance tests in 133.25 s,
    including twice byte-stable bootstrap; locked workspace (260 VM tests);
    focused 20,000-child zero-headroom regression; actual Node WASM execution;
    git diff --check. Logs: /tmp/lana-gc-reserve-build.log,
    /tmp/lana-gc-reserve-acceptance.log, /tmp/lana-gc-reserve-workspace.log,
    /tmp/lana-gc-reserve-wide.log and /tmp/lana-gc-reserve-wasm.log.
    The retained-workspace tree also passed all four paired performance workloads
    (30 alternating calls, five warmups discarded). Baseline/candidate warm
    medians in ms: Python bytecode 0.415/0.421, Python source 23.727/18.227,
    Rust VM 25.887/25.045, Rust compile 22.118/21.843. Python first calls were
    1.348/3.921 ms for bytecode and 23.477/23.086 ms for source, reported separately.
    Raw samples and binary/compiler digests are in
    /tmp/lana-gc-reserve-performance.json. This is local checkpoint evidence,
    not completion of the collector or exact-final-tree release gates.

56. **Class-arena capacity accounting — partial heap-ownership batch
    (2026-09-27).** The task-local `class_objects` arena now reserves its Vec
    capacity against the VM heap limit. The reservation follows the arena into
    task completion and retained-root ownership, and is released when collection
    empties the arena or the last retained graph is reclaimed. Existing
    constructor rollback and task-transfer paths use the same owned reservation.

    Verification: all seven focused class-collection tests and the retained
    scratch-OOM regression passed. The full `lana-vm` crate ran 260 tests; 259
    passed and `array_growth_failure_preserves_existing_items` failed because
    its expected array register was null. That test does not exercise class
    allocation; its failure needs separate diagnosis before crate-wide
    qualification. `git diff --check` passed. Explicit edge/root
    tracking, accounting for every collector metadata allocation, and cleanup
    under all locked/OOM/task failure paths remain open.

57. **Generational barrier regression — local collector checkpoint
    (2026-09-27).** Added a direct Rust regression that promotes an array,
    attaches a young child, verifies minor collection keeps the child, removes
    the edge, and verifies a later minor collection reclaims it. Corrected the
    array-growth OOM regression to create its initial array before exhausting
    the remaining heap, so collector scratch cannot mask the growth failure.

    Verification: locked `lana-vm` tests passed (261/261); `git diff --check`
    passed. This verifies the existing write barrier and collection behavior;
    it does not implement incremental slices. Full-graph routine tracing,
    bounded edge traversal and the Release p99 target remain open.

58. **Tracked embedding roots — in progress (2026-09-27).** `Vm::result()` now
    returns `Result<RootedValue, LanaError>` so root-registry growth is admitted
    against the VM heap. Cloned handles share one lease and one explicitly
    enumerated root entry; scalar results need no registry allocation. The CLI,
    REPL, WASM wrapper, task result, embedding guide, and VM spec use the new
    fallible boundary. The root registry uses the existing heap-accounted
    `Buffer` and the collector seeds marking from its entries.

    Verification: `cargo check --locked --workspace`; all 262 `lana-vm` tests;
    focused root/OOM tests; `git diff --check` passed. Incremental collection
    and exact-tree qualification remain open; this checkpoint is not completion.

59. **Heap roots, incremental collection, and exact-tree qualification —
    locally complete (2026-09-27).** `Vm::result()` now registers graph-bearing
    results in a heap-accounted root table shared by cloned `RootedValue` leases;
    scalar results do not consume root-table space. Collector work persists
    across routine safepoints and processes at most 128 work units per slice.
    Mutations to containers, roots, futures, generators, reactive state, and
    planned-effect receipts invalidate an in-progress trace before sweeping.
    The severe-pressure allocation fallback remains synchronous.

    Verification on this source tree: `python3 tools/build.py build`;
    `python3 tests/run.py --no-build` (145/145, including repeated stable
    compiler bootstrap); `target/lana/bin/lana version` (4.0.0, LABC v2);
    `cargo test --locked --workspace --no-fail-fast` (263 VM, 115 runtime,
    27 bytecode, and 7 WASM conformance tests passed); `git diff --check`;
    Release deep/wide 20,000-node collector test (128-work cap and p99 <10 ms);
    nightly bytecode fuzz (601 seconds, 185,727,021 executions, no finding);
    universal install and both architecture slices, plus example execution;
    Python integration tests and 14 Hugging Face tests; clean source-candidate
    archive extraction/build/install/example run.

    The paired 3.0.2 Release benchmark's first run exceeded the 5% bytecode
    threshold (0.413/0.445 ms baseline/candidate); its repeat passed all four
    workloads: Python bytecode 0.426/0.424 ms, Python source 23.866/17.971 ms,
    Rust VM 26.044/25.946 ms, and Rust compile 22.920/22.024 ms. First-call
    Python timings are recorded separately in
    `/tmp/lana-candidate-performance-repeat.json`.

    The final candidate content manifest and clean source-candidate archive
    are recorded under `/tmp/lana-exact-candidate-manifest.txt` and
    `/tmp/lana-source-candidate/lana-4.0.0-source.tar.gz`.
    This is local qualification only; nothing was committed, tagged, or published.

60. **Completion-claim correction (2026-09-28).** Reopened matrix rows 2,
    3, and 8 after inspecting the actual collector and its callers. Hash-table
    growth, workspace reset, arena compaction and registry pruning run whole
    passes outside the counted slice. Sweep ignores mutation-epoch changes.
    The shutdown exhaustion check raises the heap limit before final release;
    the deep stress check manually breaks the chain before teardown. These
    checks do not establish the complete ownership or bounded-pause contracts.
    Actual WASM/editor results from earlier candidates and a passing benchmark
    retry do not establish final-tree qualification or explain the failed run.
    Historical logs remain intact. REQUIREMENTS.md retains its active items.

61. **Retained-value API and checked limits (2026-09-28).** Removed public
    Deref access from RootedValue. Scalar inspection and child(index) expose
    tracked handles; independent child handles keep the same heap owner.
    Migrated CLI inspection and internal Rust callers. set_memory_limit now
    returns Result, rejects an undersized limit transactionally, and propagates
    failure through CLI/task creation. Updated embedding migration guidance.
    Verification: 264 VM tests and locked workspace check passed; logs
    /tmp/lana-heap-limit-tests.log, /tmp/lana-root-interface-tests.log,
    /tmp/lana-root-interface-check.log. This does not close ownership row 2.

62. **Root retirement and resumable cleanup (2026-09-28).** Root removal now
    tombstones its registry entry so its graph stays alive until tracing can
    protect disposal. Admission and shutdown use a separate workspace from
    an active incremental trace. Result admission reserves traversal/worklist
    capacities against the original heap. Natural final release of the tested
    retained graph requires no fresh allocation or increased heap limit.
    Routine cleanup now resumes class compaction, node/edge/index release and
    weak/owned registry pruning. Sweep uses try_lock and retries held locks.
    Mutation invalidates sweep as well as marking; young class writes also
    invalidate active traces. Added one-unit phase/root/mutation checks,
    nonblocking sweep-lock checks, independent child retention, and deep
    immutable retirement. Removed manual chain destruction and heap-limit
    restoration from the relevant acceptance tests.
    Verification: all 268 VM tests passed; /tmp/lana-natural-teardown-tests.log;
    git diff --check passed. Allocation-time admission for every graph kind,
    post-retention graph growth, exact generational barriers, hash/buffer growth,
    complete operation counting and final qualification remain open.

63. **Allocation-time scratch admission and individual metadata edges (2026-09-28).**
    Heap reservations now carry collector node/edge credits. Mutable container
    buffers admit their edge capacities before growth; class creation admits
    references and fields. Collector ownership and retained-root leases are
    charged to the same heap. Suspended generator/future registers, initial
    reactive history and effect receipts admit scratch before publication.
    Reactive history extensions are preflighted across all affected nodes
    before revision publication; effect receipt growth is admitted before append.
    Active traces use a separately prepared index when admission changes capacity.
    Incremental value tracing processes each runtime metadata link separately;
    class storage/reference registration also occupies separate steps.
    Added a one-unit multi-link regression. Updated memory-budget checks to
    include live collector metadata; class collection succeeds with no remaining
    headroom while instruction-limit/cancellation failures preserve its graph.
    Verification: 269 VM tests passed in /tmp/lana-component-tests.log.
    The earlier admission-only workspace check and compiler build passed in
    /tmp/lana-admission-workspace.log and /tmp/lana-admission-bootstrap.log.
    Locked workspace tests passed in /tmp/lana-admission-final-workspace.log;
    the public GitHub download/checksum smoke case remains explicitly ignored
    by the existing harness. All 16 Release collector checks, including the current pause
    assertion, passed in /tmp/lana-admission-release.log. These focused checks
    do not establish complete hash/payload operation counting. git diff --check
    passed. The final native rebuild/regression run is still pending.
    Immutable payload admission, unretained execution-root teardown, guard
    ownership, hash probe/growth counting, wide immutable destruction, exact
    generational barriers and final candidate qualification remain open.

64. **Weak-slot locks and compiler pressure regression (2026-09-28).**
    Weak pruning now releases the heap lock before dropping an upgraded slot
    and uses try_lock for slot metadata. Added a threaded locked-slot regression.
    Node registration refreshes a changed epoch before any edges have been
    traced; external-root bounds are refreshed too. Slot tracing protection
    begins when its edges are traced. Existing post-trace invalidation remains.
    Plain values take one primary-edge step; multi-link values remain resumable.
    Extended mutation/root coverage to cycle registration.
    The native suite initially passed 144/145: self-hosting hit the fixed
    instruction limit (/tmp/lana-admission-regressions.log). Sparse metadata
    tracing alone did not resolve it (/tmp/lana-sparse-bootstrap-regression.log).
    Diagnostic collection failed at 50,000,000 charged instructions after only
    19,145,765 bytecodes, with 137,142,714 live bytes; registration changes alone
    also failed (/tmp/lana-gc-debug-bootstrap.log and
    /tmp/lana-gc-registration-bootstrap.log). Temporary diagnostics were removed.
    Ordinary container pressure now starts at three quarters of the checked
    limit. The diagnostic bootstrap passed at 49,680,928 instructions with
    byte-identical assembly (/tmp/lana-gc-pressure-bootstrap.log).
    Retained the half-limit class pressure path after two small-heap class
    regressions exposed insufficient construction headroom. All 270 VM tests
    pass with both paths (/tmp/lana-pressure-object-tests.log). Native rebuilding
    and full qualification remain pending; rows 2, 3 and 8 remain open.

65. **Resumable hashing, pressure fallback and immutable registrations (2026-09-28).**
    Hash collisions now probe one slot per charged work unit; routine collection
    uses mutator-admitted index capacity. Added a 300-collision regression.
    Full fallback traces replacement pins before releasing an interrupted graph.
    Array construction retries once after full collection on unpublished OOM;
    the 1 MiB container regression passes. Removed duplicate array headers from
    the allocation helper. Scalar root disposal prunes string charges and the
    weak collector backlink. All 274 VM tests passed at that boundary
    (/tmp/lana-root-string-hash-tests.log).
    Registered eleven immutable payload families with weak slots and admitted
    collector scratch. Decoder Arc::get_mut calls failed because registration
    adds weak owners: five runtime failures are preserved in
    /tmp/lana-immutable-factory-tests.log. Decoders now move fresh payloads with
    Arc::try_unwrap, preserve stored bits/world identity, and register their
    finalized values. The repaired workspace passes
    (/tmp/lana-immutable-factory-repair-tests.log).

66. **Execution-root and sliced immutable retirement (2026-09-28).**
    VM shutdown traces managed graphs before releasing execution roots. Full
    collection releases parent pins before child pins using admitted scratch.
    A 20,000-node unretained managed chain drops with no heap headroom.
    Routine retirement counts dependency edges and releases wide immutable
    payloads one field per work unit using inline retirement state. Added a
    20,000-field one-unit regression. The deep-chain loop needed additional
    iterations for the new counted phases; its initial failure is recorded in
    /tmp/lana-wide-immutable-retirement-tests.log. The subsequent workspace passes:
    276 VM tests and 115 runtime tests
    (/tmp/lana-retirement-workspace-tests.log).
    Remembered sets now inspect individual owner edges, one edge per routine
    work unit. Foreign payloads without local age still require conservative
    tracing. Suspended frames, reactive nodes and effects now promote and mark
    their registered slots on mutation; 276 VM tests pass
    (/tmp/lana-suspended-age-tests.log).
    Payload byte ownership, foreign graph age, locked/reentrant teardown,
    interrupted immutable retirement and final exact-tree qualification remain
    open. Native and Release checks are running; earlier results cannot close
    matrix rows 2, 3 or 8.

67. **Payload reservations and owning collector registrations (2026-09-28).**
    Immutable headers and vector capacities now have lifetime reservations in
    their slots; removed their corresponding cumulative alloc_bytes charges.
    Added repeated payload collection and exhausted-budget rejection checks.
    Full fallback was growing its hash table before checking duplicate nodes at
    half load. A zero-headroom duplicate regression proves that lookup now
    reuses admitted capacity. The native suite before that fix passed 144/145:
    self-hosting passed, but the 1 MiB container host allocation failed
    (/tmp/lana-immutable-retirement-native-tests.log). The diagnostic identifies
    full-collection OOM (/tmp/lana-map-allocation-diagnostic.log). The repaired
    container case passes (/tmp/lana-map-hash-container.log); temporary diagnostics
    were removed. Workspace tests then passed, including 279 VM tests and 115
    runtime tests (/tmp/lana-payload-hash-workspace-tests.log).
    Weak registration alone did not protect graphs released by a mutator.
    Registered nodes now have a collector-owned reference, excluded from root
    counts and released only after successful marking. Unpublished decoder
    payloads use checked ownership transfer before final registration.
    Collection releases empty idle scratch when all admissions disappear.
    Four initial VM failures are recorded in /tmp/lana-owned-node-arena-tests.log;
    three assumed immediate destruction, and host array allocation needed the
    same unpublished-OOM retry as array bytecode and map creation. Updated the
    tests to assert major collection of promoted garbage and explicit set
    collection. Exhaustion checks collect existing garbage before fixing the
    limit. All 279 VM tests pass (/tmp/lana-owned-arena-idle-tests.log).
    Stable shared container and class writes promote targets before publishing
    them; 279 VM tests pass (/tmp/lana-stable-write-tests.log). A deep interrupted
    retirement regression is running. Foreign graph ownership/ages,
    locked/reentrant cleanup and final candidate qualification remain open.

68. **Foreign admission, deferred guards and scheduler unwinding (2026-09-28).**
    Retained foreign immutable graphs receive owned slots, payload reservations
    and actual edge admissions. The original manual retention walk did not save
    per-node edge ranges: the initial foreign deep-chain check failed with OOM,
    and its diagnostic unwind overflowed the stack
    (/tmp/lana-foreign-arena-admission-tests.log and
    /tmp/lana-foreign-edge-diagnostic.log). Saved those ranges before adoption;
    280 VM tests pass (/tmp/lana-foreign-edge-admission-repair-tests.log).
    Foreign slot pruning is resumable. Routine metadata reads and full sweep
    use nonblocking attempts. The workspace passes, including 280 VM and 115
    runtime tests (/tmp/lana-foreign-nonblocking-tests.log).
    A stable-write check accidentally entered index_get and panicked source
    conformance; the panic was hidden by waiting scheduler workers. Preserved
    /tmp/lana-arena-wasm-panic-diagnostic.log and
    /tmp/lana-owned-arena-conformance.sample. Moved the check to index_set.
    Source conformance and the workspace pass
    (/tmp/lana-host-write-arena-repair-tests.log). Added scheduler shutdown on
    unwind and a retained guard/deferred teardown regression. Unknown-host
    spelling initially made the panic regression fail during assembly; the
    corrected store_open fixture passes with all 282 VM tests
    (/tmp/lana-guard-scheduler-repair-tests.log).
    Successful extension graph outputs now receive checked admission before
    register publication, with an exhausted-budget no-publication regression.
    All 283 VM tests pass (/tmp/lana-host-output-regression-tests.log).
    Initial suspended-frame/reactive/effect payload capacities now have owned
    reservations. The workspace passes
    (/tmp/lana-suspended-payload-reservation-tests.log).
    Release collector checks pass 26 tests
    (/tmp/lana-owning-arena-release-tests.log), but final qualification remains
    open. Native self-hosting hit the unchanged instruction limit
    (/tmp/lana-owning-arena-native-regressions.log). Respecting the existing
    pressure growth trigger still failed
    (/tmp/lana-arena-growth-bootstrap.log). Full major collection now releases
    unaliased arrays of at most eight supported fields before graph tracing,
    charging the work and protecting child arrays through their owning slots.
    All 283 VM tests pass at that boundary
    (/tmp/lana-unaliased-array-tests.log); bootstrap requalification is running.
    No active requirement is removed or marked complete by this checkpoint.

69. Bootstrap budget investigation remains open. Temporary limit diagnostics
    measured 44,651,217 bytecodes at 50,000,001 total instructions, with
    216,621,232 live bytes and 110,991 registered slots
    (/tmp/lana-bootstrap-current-metrics.log). Diagnostics were removed.
    Container pressure fallback now respects the selected generation.
    Wider unaliased-array reclamation preserves live children and old-generation
    policy; all 28 collector checks pass
    (/tmp/lana-pressure-reaper-policy-tests.log). Earlier half-limit scheduling
    still exceeded the instruction limit (/tmp/lana-half-pressure-bootstrap.log).
    A separate pressure reclamation pass reached memory exhaustion during
    environment_clone (/tmp/lana-pressure-reaper-bootstrap.log); it is an
    unfinished implementation experiment, not qualification evidence.
    Extended the checked path to maps and moved container pressure start to
    allocation safepoints, avoiding premature incremental pins. That workspace
    passes 284 VM and 115 runtime tests
    (/tmp/lana-allocation-safepoint-workspace-tests.log), but bootstrap still
    exceeds its instruction budget. Dead registry pruning and idle scratch
    reduction have focused live-child/metadata coverage
    (/tmp/lana-reaper-scratch-retention-test.log); scratch reduction alone still
    fails bootstrap (/tmp/lana-reaper-scratch-bootstrap.log). The final pressure
    headroom policy remains under investigation.
    Resource limits and active requirement clauses remain unchanged.

70. **Bootstrap headroom through compiler work reduction (2026-09-28).**
    Cached the immutable parser token count, next-register value, argument counts
    and expression tag. Call packing now reuses an already contiguous argument
    range; other ranges retain the existing copy path. Added
    contiguous_call_arguments.lana for literals, reordered/repeated arguments,
    zero/single arguments, evaluation order and shared mutable identity.
    Regenerated the checked compiler with the existing 3.0.2 Rust VM under
    256 MiB and 50,000,000 instructions, then iterated to byte stability.
    The candidate VM produced two identical copies at 44,978,970 instructions
    with the experimental reclamation path
    (/tmp/lana-contiguous-self-hosting.log). Removed that entire path and its
    speculative headroom/idle-scratch policies. The existing traced collector
    also passes twice-byte-stable bootstrap
    (/tmp/lana-optimized-traced-bootstrap.log). No resource limit was raised.
    The workspace passes 283 VM and 115 runtime tests
    (/tmp/lana-optimized-traced-workspace-tests.log); the new fixture passes
    (/tmp/lana-contiguous-arguments-test.log). Temporary diagnostics are removed.
    Ownership/task audit and exact final qualification remain open.

71. **Task capture and result ownership audit (2026-09-28, in progress).**
    Bound cloned Set payloads to their managed slots in both clone paths.
    Task transfer now copies suspended generator/future registers and mutable
    graph captures in datasets, training results, posteriors, object values,
    claims and effect receipts. Generator/future memo entries precede captured
    edges; copied waiting futures register dependency wakeups in the receiving
    VM. A locked source frame fails without blocking.
    Child class storage stays with the retained heap owner, rather than moving
    into TaskState and invalidating independent result leases. Argument copying
    observes parent cancellation; cancellation or scheduler shutdown before
    queue insertion publishes no task and restores the live-task count.
    Focused suspended-capture, dependency, retained child-class and cancellation
    checks pass. The VM checkpoint passes 286 tests
    (/tmp/lana-task-ownership-checkpoint-tests.log); the two subsequently added
    checks pass independently (/tmp/lana-waiting-future-transfer-test.log and
    /tmp/lana-cancelled-task-transfer-test.log).
    Joined task-result graph tracing, internal versus external task result
    leases, service-held task lifetimes and bounded cleanup across child heaps
    still require implementation and qualification. Rows 2, 3 and 8 remain open;
    no active requirement clause was removed.

72. **Durable host result heap boundary (2026-09-28).**
    The rebuilt native suite passed compiler bootstrap but initially failed
    eight durable/runtime cases: decoded mutable records belonged to separate
    heaps and correctly failed host-result retention
    (/tmp/lana-task-transfer-native-tests.log, 138/146). Added Vm::import_value
    using the existing budgeted graph transfer and class rollback machinery.
    StoreHost imports successful outputs at its shared dispatch boundary before
    VM publication. Arbitrary extension outputs still undergo the existing
    ownership validation. Documented import versus retention for Rust callers.
    A real store_get callback check retains its decoded result after VM teardown.
    The workspace passes 289 VM and 116 runtime tests
    (/tmp/lana-host-import-workspace-tests.log). Native rerun evidence is in
    /tmp/lana-host-import-native-tests.log; it includes successful twice-stable
    bootstrap and durable future messages. Subsequent composite-future transfer
    checks cover future_all/future_race operation markers and dependency wakeups
    (/tmp/lana-composite-future-transfer-test.log). That later VM change requires
    the next native rebuild before exact candidate qualification.
    Task-result cycle tracing and final qualification remain open.

73. **Preserve issued authorization identities (2026-09-28).**
    The host import rerun passed 145/146 native cases
    (/tmp/lana-host-import-native-tests.log). The remaining execution-live
    failure exposed the policy decision's pointer-identity seal: copying its
    issued map invalidated authorization. Runtime dispatch now allocates on
    the active VM heap and preserves the exact issued policy map. Decoded
    durable records still use the explicit import boundary. The seal check
    remains unchanged. Latest workspace before this correction passes 290 VM
    and 116 runtime tests (/tmp/lana-task-host-import-workspace-tests.log).
    Correction verification and final qualification remain pending.

74. **Joined task cycle ownership (2026-09-28).**
    Task handles now have accounted managed headers and collector registrations.
    Joined results are traced as local edges. Unjoined results remain behind
    independent child-heap leases. Joining removes the task from the VM and
    scheduler service lists and applies the managed write barrier before result
    publication. Removed the unused TaskState class arena cache.
    Retained array/task cycles survive VM teardown and disappear after the last
    root drops; routine collection also reclaims them through one-unit slices.
    Managed task admission exposed geometric scratch-buffer growth at the
    original low-memory fork boundary. Buffer growth now retries the exact
    required capacity when doubling fails with OOM. The original fork test
    budget is unchanged; a direct growth check proves success and failed-growth
    preservation. All 293 VM tests pass
    (/tmp/lana-task-cycle-slice-tests.log), and the workspace passes 293 VM and
    116 runtime tests (/tmp/lana-managed-task-workspace-tests.log).
    Live authorization succeeds with both success and failure receipts
    (/tmp/lana-issued-execution-live-test.log), closing checkpoint 73's failure.
    The final removal of unused TaskState fields requires another rebuild.
    Full child-domain cleanup accounting and final collector/native/release
    qualification remain open. REQUIREMENTS.md is unchanged.

75. **Nested task owners and nonblocking root release (2026-09-28).**
    Retired child heaps collect through an intrusive thread-local owner list.
    Queue links are part of the accounted RootOwner header; queueing introduces
    no unbounded scratch vector. Nested child-lease drops enqueue cleanup instead
    of recursively invoking the next collector. A 20,000-owner task-domain chain
    is covered by the workspace check below.
    Root records now hold weak lease identities. Last-lease release uses try_lock
    rather than blocking on the root registry; collection recognizes dead leases
    even when their numeric tombstone could not be written. A held-registry,
    cross-thread release check passes (/tmp/lana-nonblocking-root-release-test.log).
    The earlier 2,000-owner focused check passes
    (/tmp/lana-nested-task-owners-test.log), and the interim VM suite passes 294
    tests (/tmp/lana-nonblocking-root-tests.log). Latest workspace/build evidence
    is /tmp/lana-task-root-domain-workspace-tests.log and
    /tmp/lana-task-root-domain-build.log. Full child-domain instruction accounting,
    interruption/locked cleanup audit and exact final qualification remain open.
    The workspace passes 295 VM and 116 runtime tests, including the 20,000-owner
    chain (/tmp/lana-task-root-domain-workspace-tests.log); the native build passes
    (/tmp/lana-task-root-domain-build.log). Extended root release to cover retired
    heaps as well as active VMs; both pass
    (/tmp/lana-active-retired-root-release-test.log). The retired tracer also uses
    try_lock for its registry access. This final adjustment requires a rebuild
    before exact-tree claims; ongoing native results identify the earlier build.

76. Scheduler task roots and bounded DAG transfer (2026-09-28).
    Scheduler service entries retain explicit RootedValue leases so a task
    creator's retired heap survives queued work. Completed task child handles
    preserve the original result owner independently of join's copied result.
    Focused checks pass in /tmp/lana-rooted-scheduler-owner-test.log and
    /tmp/lana-retained-task-child-test.log. The earlier native build passes
    146/146 in /tmp/lana-task-root-domain-native-tests.log; it predates these
    changes and does not qualify the current candidate.

    Distribution transfer now memoizes shared nodes, charges traversal work,
    and returns Limit at the existing 64-level copy boundary instead of
    recursing without a bound. Derivation transfer copies its DAG, preserves
    origin IDs and revisions, copies text through the existing checked string
    helper, and isolates gradient locks and backing buffers. A locked source
    gradient returns UnsupportedOperation. Focused checks pass in
    /tmp/lana-distribution-transfer-test.log and
    /tmp/lana-derivation-transfer-test.log.

    These transfer fixes do not establish lifetime-owned reservations or
    incremental retirement for distribution and derivation headers. Those two
    DAG families still need managed ownership; active join cleanup instruction
    accounting, interrupted/locked teardown, collector policy qualification,
    and all exact final gates remain open. REQUIREMENTS.md remains unchanged.

77. Managed distribution DAG ownership (2026-09-28).
    All five VM distribution constructors and transfer copies now use the
    existing managed payload arena with lifetime-owned header reservations and
    collector edge credits. Full and incremental traces visit distribution
    children; routine retirement removes one immutable child per work unit.
    Transfer now uses a checked heap Buffer for iterative postorder traversal,
    preserving shared nodes without the temporary 64-level restriction from
    checkpoint 76. Derivation transfer retains its current depth guard.

    The 20,000-node retained distribution check passes after copying the DAG,
    dropping the VM, and dropping the last host root: both end weak references
    expire and heap.live_bytes() returns zero. Evidence:
    /tmp/lana-deep-distribution-transfer-test.log. The earlier workspace passes
    300 VM and 116 runtime tests in
    /tmp/lana-managed-distribution-workspace-tests.log; its build passes in
    /tmp/lana-managed-distribution-build.log. Those runs preceded the iterative
    copy adjustment and must be rerun for the current tree. Derivation arena
    ownership, active join cleanup charging, interruption audits and final
    qualification remain open; REQUIREMENTS.md is unchanged.

78. Managed derivation DAG ownership (2026-09-28).
    Derivation headers, input capacities and gradient lock headers now reserve
    through the existing managed payload path. Constructors finalize AD and
    revision fields before publication; historical imports still assign fresh
    origins while task copies preserve origin IDs. The graph tracer includes
    derivation metadata as a fifth Value component, with matching admission
    credits for container buffers, class fields and host roots. Retirement
    drains one input or AD parent per routine work unit.

    Distribution and derivation transfer both use iterative heap-owned
    traversal stacks. The 20,000-node derivation check copies the graph,
    preserves the retained source through VM teardown, and releases both end
    weak references plus all heap reservations after the last host root drops.
    Evidence: /tmp/lana-deep-derivation-owner-test.log. The focused gradient
    isolation check passes in /tmp/lana-managed-derivation-test.log. Workspace
    and native build reruns are in progress; their evidence is not yet a gate
    claim. Active join cleanup charging, locked/interrupted teardown, foreign
    ownership/ages and final qualification remain open.

79. Explicit foreign graph import (2026-09-28).
    Retention now rejects every unregistered foreign graph node, including
    immutable DTOs and DAGs. Foreign callers must import a local copy before
    retaining it. Removed the second ownership arena and its adoption/pruning
    code. Retired child inspection applies the same ownership check. This
    avoids competing heap owners and ambiguous foreign generations; Rust
    embedding migration is documented in docs/rust-embedding.md.

    Live snapshot transfer now copies provenance parents and distribution
    nodes into the receiving heap. Transfer reuses immutable DAG/string memos
    across capture boundaries without cloning those memo tables. State source
    labels and map keys use the checked receiving string allocator. Rooted
    distribution inspection renders through inspect_state_dist(format),
    replacing the clonable Arc accessor.

    The foreign immutable check demonstrates rejection before import, valid
    independent retained copies and zero residual bytes in each released heap.
    /tmp/lana-local-graph-ownership-workspace-tests.log passes 302 VM and 116
    runtime tests. The additional live-distribution check passes in
    /tmp/lana-live-distribution-owner-test.log. Earlier native evidence passes
    146/146 in /tmp/lana-managed-dag-native-tests.log; the Release slice check
    passes in /tmp/lana-managed-dag-release-slices.log. Both precede the final
    foreign-boundary edits and must be rerun. An initial migration run aborted
    when an obsolete raw 20,000-node test graph was rejected and then dropped
    recursively; that check now constructs managed nodes. Constructor, host
    callback and metadata checks use local managed payloads or explicit import.
    Active join cleanup charging and exact final qualification remain open.

80. **Budgeted task retirement and nonblocking imports (2026-09-28).**

    Workers retire their VM before publishing task completion. Join copies the
    result into the receiving heap, then charges retired source-heap cleanup
    against the caller instruction budget. A failed cleanup retains the complete
    cached copy and queued source owner; retry drains cleanup before publication.
    The focused limit/retry check passes in
    /tmp/lana-join-cleanup-budget-test.log. No partial result is published.

    Deferred owner chains drain iteratively at thread exit. Both a 20,000-owner
    chain and cross-thread queue ownership reset pass in
    /tmp/lana-thread-retirement-tests.log. An initial thread-exit check exposed
    incorrect TLS destruction order; queue initialization now establishes the
    fallback before the primary queue. Checked derivation text allocation passes
    its low-memory rejection check in /tmp/lana-derivation-text-budget-test.log.

    Array, map, set and reactive imports use try_lock and preserve the source
    on rejection. Reactive snapshots capture value and revision under one lock.
    /tmp/lana-locked-imports-test.log passes. Full collection marks registered
    nodes as tracing and checks the mutation epoch before marking and sweeping.
    /tmp/lana-full-collector-mutation-test.log passes the mutation-after-mark
    check. Its first run exposed the missing full-collection tracing flag.

    /tmp/lana-checked-retirement-workspace-tests.log passes 308 VM and 116
    runtime checks before the final mutation fix. The new exact workspace,
    native build and bootstrap runs are in progress. Rows 2, 3 and 8 remain
    open until final qualification; REQUIREMENTS.md is unchanged.

81. **Fixed-budget compiler qualification repair (2026-09-28).**

    The first post-ownership native suite failed compiler self-hosting at the
    unchanged 50,000,000-instruction limit. The log is preserved in
    /tmp/lana-final-ownership-native-tests.log. A fixed-limit standalone VM
    diagnostic reached 39,835,690 bytecode instructions plus native work before
    exhaustion; /tmp/lana-small-call-diagnostic.log records that failure.

    Argument packing now returns directly for zero/one arguments and emits
    two-argument packs without temporary argument-register arrays. Resolver and
    emitter dispatch cache immutable AST tags. These changes reduce actual
    compiler work; collector tracing and compiler limits stay intact. Existing
    contiguous_call_arguments coverage includes zero/single, reordering, repeated
    arguments, evaluation order and shared identity.

    The bootstrap was seeded with the trusted 3.0.2 executable, then compiled
    twice by the current VM under 256 MiB and 50,000,000 instructions. Both runs
    finish at 49,241,462 instructions and produce byte-identical assembly in
    /tmp/lana-small-call-fixed-point.log. compiler/bootstrap/compiler.lasm now
    contains that artifact. The native suite and dependent gates must rerun
    against the rebuilt candidate before completion.

82. **Active requirements locally complete (2026-09-28).**

    The rebuilt candidate passes 146/146 native checks, twice-stable bootstrap,
    the locked workspace (309 VM and 116 runtime tests), Release deep/wide
    20,000-node slice/pause assertions, 601 seconds of bytecode fuzzing, both
    universal macOS slices, checksum-backed binary/source archive checks,
    Python/HF integrations, actual Node WASM and live Neovim/VS Code clients.
    Three declared paired Release benchmark rounds independently meet the
    5% threshold; the largest regression is 3.02%. First calls remain separate.

    plans/requirements-qualification.md records commands, artifact identities,
    results, logs and the preserved earlier failures. The final candidate
    manifest and source archive are under /tmp/lana-final-candidate-source/.
    Final documentation edits preserve compiled input hashes and are included
    in the fresh source extraction/build check. The existing public-release
    download smoke is excluded with its explicit publication boundary; all
    required local packaging checks run against the actual candidate.

    Matrix rows 2, 3 and 8 are now locally verified; rows 4–7 have current rerun
    evidence. REQUIREMENTS.md removes only these verified active items and
    retains its exclusions. spec/VM.md, Rust embedding guidance and collector
    performance status now describe the implemented policy. Nothing was
    committed, tagged, signed or published.
