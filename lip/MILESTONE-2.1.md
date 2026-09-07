# Lana 2.1.0 implementation milestone

Status: **Paused at a verified checkpoint; not release ready.**

Approved scope: LIPs 004–016 and 018–025. LIP-017 is deferred. This is an
open-ended, backward-compatible release, not a promise tied to a calendar date.
Implementation and acceptance are distinct: a LIP is Final only after its
specified behavior, errors, integrations, and conformance checks pass.

## Decisions

1. Preserve supported 2.0 source/API behavior, bridge ABI v1, local project
   workflows, and LABC v1/v2 loading. Introduce v3 only with the suspension
   implementation; do not bump the bytecode version merely for host calls.
2. Implement native language behavior in C11 and Rust. WASM has an explicit
   host-capability boundary; missing host facilities fail explicitly.
3. Ordinary tensor operations retain LIP-004's Tensor/scalar return types.
   LIP-008 applies to uncertain operands and ML results. Prediction extraction
   is explicit and retains provenance. Unknown correlation is not independence.
4. CPU tensors use binary64. GPU is flexible: explicit precision/backend
   selection, documented numerical tolerance, and recorded provenance. Never
   silently convert binary64 to float32. CPU/GPU bitwise equality and software
   binary64 Metal are not release requirements. CPU replay remains strict;
   approximate GPU computation is labeled and cannot claim strict CPU replay.
5. Separability requires an explicit bipartition. Return evidence-supported
   separable/entangled outcomes or inconclusive, never an unproved boolean.
6. Preserve distribution-valued APPEND and its explicit relationship rules.
   Do not invent N-qubit APPEND or discard a distribution during training.
7. Incremental dependencies include optimizer state and structural inputs;
   a zero gradient at one point does not prove independence.
8. Async replay consumes recorded external completions and responses.
   Cancellation prevents unstarted work; it cannot undo completed effects.
9. FFI loads libraries in an isolated worker, including library constructors.
   It does not automatically retry an external effect after worker failure.
10. Standard-library installation and local content locks do not require a
    package registry. Preserve local dependencies; do not build LIP-017 unless
    a concrete dependency establishes that it is necessary.

These are approved amendments to the implementation targets. The mathematical
and source authorities must be updated before changing corresponding behavior;
this checklist does not override them or claim the targets already work.

## Sequence and completion gates

| Batch | Boundary | Completion gate | Status |
|---|---|---|---|
| 1 | Baseline and contract reconciliation | Fresh C/Rust/bootstrap baseline; approved conflicts recorded | Baseline verified; per-feature authorities updated with their batches |
| 2 | LIP-004 CPU tensors | Shape/type/resource failures, full tensor surface, C/Rust conformance | Validation, axis reductions, indexing/slices, BLAS matmul dispatch, full scratch accounting, and matmul-specific unit/differential/regression coverage verified |
| 3 | Capabilities and Metal | Denial before access; explicit precision and actual device tests | Capability machinery (grant/revoke, revoked-token denial) verified; Metal still deferred |
| 4 | LABC v3, generator/async frames | v1/v2 compatibility, verifier rejection, stable bootstrap | Generator suspension (LIP-022 §2) verified: v3 opcodes rejected in v1/v2 chunks, byte-stable bootstrap; async frames deferred to batch 10 |
| 5 | LIPs 016/021/022/023 | Installed stdlib, immutable collections, lazy generators, text/codecs | Partial existing library; pending full contract |
| 6 | LIP-015 datasets | Lazy algebra, bounded execution, atomic commits, recovery, provenance | Pending; reuse existing store/adapters |
| 7 | LIPs 005/011 | Valid STATE operators, inconclusive separability, analytic/finite-difference gradients | Pending |
| 8 | LIPs 006/007/008/013 | Authorized training, uncertainty, STATE models, static shape errors | Pending |
| 9 | LIPs 009/010/014; complete 012 | Known posteriors, atomic updates, exact serialized resume | Pending |
| 10 | LIPs 019/024 | Real HTTP/TLS/concurrent I/O, cancellation and offline replay | Pending |
| 11 | LIP-018 | Real C ABI calls, denied load, worker crash containment | Pending |
| 12 | LIPs 020/025 | Persistent REPL, export, browser/WASI execution and host denial | Partial existing WASM binding; pending full contract |
| 13 | Release qualification | All exact-candidate release gates plus Rust/Metal/WASM evidence | Pending |

CPU tensor correctness precedes bytecode suspension because it is already an
accepted host-call surface with no v3 dependency. No independent LIP is marked
complete just because a prerequisite or a historical test passes.

## Requirement ledger

| LIP | Required behavior still to qualify |
|---|---|
| 004 | Construction, bounds, real/complex arithmetic, broadcasting, BLAS matmul, axis reductions, views/slices, explicit GPU |
| 005 | Density operators, POVMs, channels, observables, products, partial trace, measurement, expectation, mixture, distance, separability evidence |
| 006 | Explicit initial parameters, SGD/Adam, data streaming, authorized execution, step history, non-finite errors |
| 007 | Valid STATE tensor construction, distribution-preserving operations, derivatives and constrained training |
| 008 | Uncertain inputs and ML results, correlation-aware propagation, explicit approximations and prediction extraction |
| 009 | MCMC/VI/SMC, exact conditioning retained, diagnostics, posterior provenance and seeded replay |
| 010 | Structural/optimizer dependencies, sparse/dense updates, atomic revisions and immutable frozen updates |
| 011 | Pure grad/vjp, saved operands in derivations, accumulation, supported control flow and explicit rejection |
| 012 | Host-rooted resource authority, scoped grant/revoke, inert plans, cached receipts and data/weights/GPU enforcement |
| 013 | Tensor shape parameters, typed model/layer ADTs, Dense/Conv/Relu, connections, exhaustive matches and explicit Dynamic |
| 014 | Addressable steps, optimizer/RNG/data state, serialized restart and byte-identical CPU continuation |
| 015 | Lazy relational algebra, inspectable optimizer, stable aggregation order, MVCC, transactions, WAL recovery and adapters |
| 016 | Installed std imports and complete named modules; local content locks without registry |
| 017 | Deferred: package manager, remote dependency resolution, registry and publishing |
| 018 | Signature validation, supported bounded marshalling, process-isolated loading/calls, no automatic retry |
| 019 | Bounded TCP/HTTP/TLS, bind/listen/accept for minimal server, capabilities, recorded responses |
| 020 | Persistent compiled session, multiline input, error recovery, commands, replayable export without rerunning effects |
| 021 | Unicode text/case/normalization, linear-time regex, formatting, bounded failures |
| 022 | Immutable deterministic sets, suspended lazy generators, collectors and comprehensions | Generators (§2) implemented; sets, comprehensions, and iterators deferred to batch 5 |
| 023 | Result codecs, canonical JSON, integer fidelity, input completion, CSV/TOML, source provenance |
| 024 | Cold futures, run_async entry, ready ordering, recorded I/O replay, cancellation, shared budgets |
| 025 | Source/bytecode bindings, measurement/sampling, bundled compiler/stdlib, browser/WASI, explicit host facilities |

## Evidence

Baseline on 2026-09-06, before milestone code edits:

- `cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug`: passed.
- `cmake --build build --parallel 6`: passed.
- `ctest --test-dir build --output-on-failure -j 4`: 74/74 passed, including
  native compiler bootstrap and local install. Output: `build/Testing/Temporary/LastTest.log`.
- `cargo test --workspace`: passed: 18 bytecode, 3 FFI, 36 runtime, 87 VM,
  and 6 WASM-binding host tests. The WASM-binding tests run on the native host;
  they are not browser/WASI acceptance.

These are local baseline results, not sanitizer, GPU, cross-VM differential,
or release-candidate evidence. Keep VERSION at 2.0.1 until release preparation.

Checkpoint: 78/78 CTest, 155 Rust workspace tests, 14/14 core and
28/28 host-call differential fixtures passed. Six focused ASan/UBSan tensor
tests passed. Axis reductions, indexing/slices (views, negative positions,
clamped bounds, read-only views, strided arithmetic and reductions, position
lists and slice syntax in the compiler), and byte-stable compiler bootstrap
are verified. Views share source buffers: C marks the base chain iteratively
and only base tensors free data; Rust shares an Arc. The new source fixture
also passed through the Rust CLI, and the durable pipeline passed with the
regenerated bootstrap. Matmul now runs through one native dispatch point in
both VMs (Accelerate `dgemm`/`zgemm` on Apple, same naive loop elsewhere,
true leading dimensions for direct strided views, accounted packing
scratch), and all tensor shape/stride/scratch temporaries are VM-accounted;
differential fixtures are byte-identical through the BLAS path. Matmul-specific
unit/differential/regression coverage and the focused sanitizer pass are now
complete; the Rust portable fallback's complex interleaving was corrected to
match the C backend. Remaining for LIP-004: capabilities and Metal. Detailed
handoff and remaining implementation work:
[`../REMAINING_WORK.md`](../REMAINING_WORK.md).
