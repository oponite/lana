# Lana 2.1.0 — implementation handoff

Updated: 2026-09-07. **Milestone incomplete; no release is ready.**

This file was re-audited on 2026-09-07 against the code. The prior version
(2026-09-06) was stale: it listed LIPs 005–014 as pending, but those host calls
and compiler builtins are already implemented and conformance-tested. The
genuinely remaining work is much smaller — see "Remaining after that" below.
This file is not a Git stash or a claim that the milestone is complete.

## Approved scope

- Implement LIPs 004–016, 018–025, and 027 as an open-ended compatible 2.1.0 release.
- Defer LIP-017 (package manager/registry) unless a concrete dependency needs it.
- Keep C11 and Rust language behavior aligned and existing 2.0 APIs working.
- Ordinary tensor math stays ordinary. Uncertainty is required for uncertain
  operands and ML results, not every numerical tensor operation.
- GPU requirements were relaxed: explicit approximate precision and numerical
  tolerances are allowed; software binary64 Metal and CPU/GPU byte identity are
  no longer required. Do not silently downcast. No GPU implementation yet.
- Return inconclusive separability results where no supported proof exists.
- Follow the batched-implementation skill: complete and verify one boundary
  before beginning a dependent batch. Do not request implementation permission
  again for work already approved in this session.

Full decisions, sequence, and per-LIP requirements: [milestone ledger](lip/MILESTONE-2.1.md).

## LIP-027 (new, 2026-09-07)

[`lip/LIP-027.md`](lip/LIP-027.md) — mixed-precision tensor dtypes (fp32 / fp16 /
bf16) — is newly added and **in scope for 2.1.0**. It is the precision half of
the training-speed work; the fused-step half is `mLIP-001`. It is independent of
B5–B10 and can proceed in parallel with them. It makes no LABC change (dtype is a
host-call argument, not bytecode), so bytecode stays v2/v3. Slices S1–S8 below
are one verifiable boundary each; complete and verify one before starting the
next.

## Working tree

- Branch at checkpoint: `main`; HEAD: `5f054304d2bae7cf26c63a9a9d2a63b5a50d0748`.
- VERSION remains `2.0.1`; bytecode is v2 with v1/v2/v3 loading (v3 emitted
  only for generator programs).
- The user already had 128 staged files, including tensor/compiler/bootstrap
  changes, LIPs, stdlib modules, unrelated experiments, and clangd cache files.
  Preserve them. No stage/reset/stash/commit/push was performed in this session.
- This session's modifications are unstaged; its added files are untracked.
  `git diff` shows our tracked edits against the existing index, while
  `git diff --cached` is the pre-existing work. Do not broad-stage either.

## Completed code boundary

Tensor indexing and slicing (LIP-004):

1. Both VMs share the `index_get` host call for tensors: a number spec is one
   integer position, an array spec is a position list of numbers and
   two-element `[start, end]` slice pairs; fewer positions than the rank keep
   trailing axes whole; integer positions drop their axis, slices keep it.
2. Non-integer, non-finite, or non-number positions and malformed pairs are
   Type errors; out-of-range integer positions (after negative wrap) are Key
   errors; more positions than the rank is InvalidParameters. Slice bounds
   wrap negatives and clamp into `[0, dim]`, never range-error, and
   `start > end` is an empty axis.
3. Non-fully-integer selections return views that share the source buffer: C
   adds `offset`/`base` fields and marks the base chain iteratively (no
   recursion, so view chains cannot overflow the mark stack; base tensors
   alone free the shared data); Rust holds `data: Arc<Vec<f64>>` plus
   `offset`. A full integer selection returns a number, or the established
   rank-zero complex tensor for complex elements. Views are read-only:
   `index_set` on a tensor is a Type error.
4. Element-wise arithmetic, matmul, axis reductions, and full reductions now
   read through each tensor's offset and strides, so non-contiguous views
   compute correctly; empty full reductions reject mean/max/min and sum to 0.
5. Compiler: `t[1, -3]` position lists and `t[0:2, 1]` slices parse into a
   positions payload (new node kind 39 for slices); the resolver type-checks
   every position and rejects array slicing/multi-position indexing as
   compile errors; assignment still takes exactly one non-slice position.
   Single non-slice positions emit byte-identical code to the previous
   emitter (verified by cmp on existing programs). Bootstrap regenerated;
   self-compilation byte-stable.

Changed code: `vm/c/vm.c`, `vm/c/value.c`, `vm/include/tensor.h`,
`vm/rust/lana-vm/src/{tensor,value,vm}.rs`,
`compiler/{syntax,parser,resolver,emitter}.lana`, and
`compiler/bootstrap/compiler.lasm`.

Checks added: indexing coverage in `tests/unit/test_tensor.c` and the Rust
`tensor.rs` tests (including `Arc` buffer-sharing and strided traversal),
`tests/regression/tensor_index_pass.lana`, indexing diagnostic cases in
`tests/test_tensor_source.py`, and six `tensor_index*.lasm` host-call
differential fixtures.

Verification: CTest 78/78 (including the new native regression, bootstrap
self-check, and source-error cases), `cargo test --workspace` 155 passed,
core differential 14/14, host-call differential 28/28, Rust CLI source
regression passed, durable pipeline passed, focused ASan/UBSan tensor tests
6/6, `git diff --check` clean for both the unstaged edits and the
pre-existing index.

Axis reductions (previous batch):

Tensor validation in both VMs:

1. Enforce rank 32 and finite/nonnegative/integer/native-index dimensions.
2. Reject C allocation-byte overflow before a tensor buffer is used.
3. Reject cyclic and excessive-rank constructor input.
4. Reject empty mean/min/max, non-finite real/complex reduction components,
   and non-finite reduction results; preserve zero for empty sum.
5. Fix Rust `tensor_complex(a, a)` deadlock by avoiding simultaneous locks on
   component arrays.
6. Make shared unresolved-value traversal skip array/map back edges rather
   than recurse forever or deadlock. Rust now shares this check between Value
   and Vm and resolves reactive current values before checking contents.

Axis reductions and compiler support:

1. Both VMs implement optional-axis sum/mean/max/min, remove the selected
   dimension, support negative axes, and preserve complex sum/mean.
2. Empty reduced dimensions permit only sum; other empty output dimensions
   produce empty tensors. Invalid axes and non-finite values fail explicitly.
3. Source supports `sum(t, axis: 0)` and positional axis calls for all four
   reductions; misspelled, duplicate, misplaced, and unsupported labels fail.
4. Resolver marks axis results as tensors. Full sum/mean have unknown static
   type because element dtype is dynamic (number or rank-zero complex tensor).
5. Regenerated `compiler/bootstrap/compiler.lasm` with the existing compiler
   limits; checked byte-stable self-compilation and the normal bootstrap test.
6. Differential tests exposed a pre-existing C host-call output/input alias
   bug. Shared dispatch now computes into a local result and publishes only
   on success, matching Rust and preserving overlapping input registers.

Changed code: `vm/c/vm.c`, `vm/include/tensor.h`,
`vm/rust/lana-vm/src/{tensor,value,vm}.rs`, `compiler/parser.lana`,
`compiler/resolver.lana`, and `compiler/bootstrap/compiler.lasm`.

Checks added: `tests/unit/test_tensor.c`,
`tests/regression/tensor_boundaries_pass.lana`, `tests/test_tensor_source.py`,
Rust tensor tests, CTest registration, and three `tensor_axis*.lasm` host-call
differential fixtures. The diagnostic test also checks no output bytecode is
created after failed compilation.

Documents changed: `spec/SPEC.md` tensor boundaries, `lip/LIP-004.md` GPU
amendment, `lip/LIP-017.md` deferral, `lip/README.md`, and the new milestone ledger.
Other draft contradictions are recorded in the ledger but have **not** all been
rewritten in their LIP/spec files. Do that before implementing each affected feature.

## Verification after the code changes

| Check | Result |
|---|---|
| Debug C build | Passed |
| CTest including bootstrap, project workflow, local install | 77/77 passed |
| `cargo test --workspace` | 154 tests passed; one existing unused-field warning in store.rs |
| Core C/Rust differential fixtures | 14/14 match |
| Host-call C/Rust differential fixtures | 22/22 match |
| New source fixture run through Rust CLI | Passed, including aliased complex arrays |
| `git diff --check` and `git diff --cached --check` | Passed |
| Focused ASan/UBSan tensor checks | 5/5 passed |

The initial C cycle regression crashed in `value_is_unresolved` before reaching
the tensor constructor. The shared traversal fix above resolved it; the final
77-test C run includes that regression. No current Debug/Rust test failure remains.

These results do not establish full LIP-004 completion, GPU execution, WASM
browser execution, release sanitizers, fuzz resilience, or a release candidate.

## Completed code boundary (verified)

Native matmul backend and resource accounting (LIP-004 section 5):

1. Both VMs route `matmul` through one native dispatch point:
   `vm/c/backend.c` + `vm/include/backend.h`, mirrored by
   `vm/rust/lana-vm/src/backend.rs` (`lana_backend_gemm` /
   `backend_gemm`). Apple builds define `LANA_BLAS_ACCELERATE` and link the
   Accelerate framework, so `dgemm`/`zgemm` run the contraction; every other
   build runs the same naive loop through the same entry, so both VMs issue
   identical backend calls per platform. Accelerate's CBLAS uses the ATLAS
   argument order `(M, N, K)`, not netlib's `(M, K, N)` — the prototypes
   encode that, and a probe program verified dot, 2-d, strided, and
   column-vector forms.
2. Operand cores stay zero-copy when expressible: a direct (non-packed)
   operand passes its true row stride as the BLAS leading dimension
   (`lda`/`ldb` fields on `LanaGemmCall`); a promoted vector passes one row
   (`lda = K`) or its column row stride (`ldb`). Anything else (column
   stride != 1) is gathered into VM-accounted scratch per batch element, and
   each batch element issues one backend call into the contiguous result
   slice. Leading dimensions clamp to at least 1 so an empty contraction
   (`K = 0`) cannot trip BLAS parameter validation, which aborts the
   process on a bad call.
3. Complete scratch accounting in both VMs: C `tensor_shape_from_array` and
   `tensor_infer_shape`/`tensor_infer_shape_at` moved from `malloc`/`free`
   to `lana_vm_alloc` (GC memory; callers no longer free); Rust mirrors the
   same closures through `alloc` — shape arrays, batch/out shapes, batch
   strides, broadcast strides, index vectors, and the two matmul packing
   buffers. `tensor_shape_from_array`/`tensor_infer_shape` now take the
   allocator in both VMs.
4. CMake wires the backend identically into every runtime variant (Debug,
   Release, sanitizer, fuzz): `LANA_BLAS_LIBS`/`LANA_BLAS_DEFINES` come from
   one `APPLE` check, so no variant can pick a different backend and break
   differential byte identity.

Changed code: `vm/c/vm.c`, `vm/c/backend.c`, `vm/include/backend.h`,
`vm/rust/lana-vm/src/{backend,tensor,vm,lib}.rs`,
`cmake/{BuildTargets,CTestTargets}.cmake`.

Verification with the backend live in both VMs: CTest 78/78,
`cargo test --workspace` 155 passed, core differential 14/14, host-call
differential 28/28 (C/Rust byte-identical through BLAS), `git diff --check`
clean for both the unstaged edits and the pre-existing index.

## Completed code boundary (verified): matmul testing

The matmul backend and accounting were already implemented and verified; this
batch added the missing coverage and fixed one latent bug:

1. C unit tests in `tests/unit/test_tensor.c` for `LANA_HOST_TENSOR_MATMUL`:
   dot, both vector-matrix promotions, 2-d, batched with broadcast, complex
   `zgemm`, direct-strided views (`lda` = row stride), packed non-contiguous
   cores, empty contraction (K = 0), and the error cases (mismatched complex,
   rank-zero operand, incompatible inner dims, incompatible batch dims).
2. Rust mirror tests in `tensor.rs`: the same cases plus a `view` helper that
   shares the base `Arc` buffer with custom strides/offset.
3. `tensor_matmul.lasm`, `tensor_matmul_complex.lasm`, and
   `tensor_matmul_invalid.lasm` differential fixtures, plus
   `tests/regression/tensor_matmul_pass.lana` registered in CTest.
4. Fixed a latent bug in the Rust portable fallback (`backend.rs`): the
   complex interleaving factor `mult` was declared but never applied to the
   pointer arithmetic, so complex matmul was wrong on non-Apple platforms
   (where Accelerate is not linked). The macOS differential suite could not
   catch it because Accelerate handles complex there. The fallback now
   multiplies element indices by `mult`, matching the C backend.

Verification: CTest 79/79 (including the new `native_tensor_matmul_pass`),
`cargo test --workspace` 164 passed, core differential 14/14, host-call
differential 31/31 (C/Rust byte-identical through BLAS), focused ASan/UBSan
tensor tests and full sanitizer CTest 79/79, `git diff --check` clean for both
the unstaged edits and the pre-existing index.

Axis source syntax is deliberately limited to the four reduction builtins;
general named function arguments are not implemented. Dtype-aware static tensor
typing remains later work.

## Completed code boundary (verified): capability grant/revoke (LIP-012)

The capability machinery LIP-012 specifies is implemented and verified; the
GPU/Metal half of batch 3 remains deferred ("No GPU implementation yet").

1. `capability(name)` now compiles to the existing `shared_information` host
   call (id 38), producing an admin `LanaCapabilityToken` instead of a plain
   string literal. This resolves the LIP-012 §1/§2 draft contradiction.
2. Two new host calls: `grant` (id 74) maps `"use"`→`READ` and `"admin"`→`ADMIN`
   and delegates to `lana_shared_capability_grant`; `revoke` (id 75) is a
   single-argument invalidation via the new `lana_shared_capability_invalidate`
   (locks the shared mutex, sets `revoked`, bumps `capability_epoch`, broadcasts
   the condition — no admin check). Rust-only store/policy/ledger host-call IDs
   shifted 74–84 → 76–86 to make room.
3. `execute_effect` gains a revoked-token check: if the plan payload is a
   capability (or an array/map containing one), a revoked token returns
   `LANA_ERR_CLAIM_REVOKED` before the executor runs. Rust `CapabilityToken`
   `revoked` is now an `AtomicBool` so revocation actually takes effect under
   `Arc`.
4. Compiler: `capability` returns `type_shared_capability("string", "admin")`;
   `grant`/`revoke` are type-checked builtins (grant requires an admin
   capability and a `"use"`/`"admin"` literal; revoke requires a shared
   capability). Bootstrap regenerated and byte-stable.

Changed code: `vm/include/bytecode.h`, `vm/c/assembler.c`, `vm/c/vm.c`,
`runtime/c/shared.c`, `runtime/include/shared.h`,
`vm/rust/lana-vm/src/{value,vm}.rs`,
`vm/rust/lana-bytecode/src/{assembler,verifier}.rs`,
`compiler/{emitter,resolver}.lana`, `compiler/bootstrap/compiler.lasm`.

Checks added: `tests/unit/test_shared.c` grant/revoke/invalidate cases, Rust
`capability_grant_and_revoke`/`revoked_capability_denies_snapshot` tests,
`tests/conformance/differential/hostcalls/capability_grant_revoke.lasm`,
`tests/regression/m12_capability_pass.lana` and
`m12_capability_revoked.lana`, and `tests/test_capability_source.py` diagnostic
cases.

Verification: CTest 82/82, `cargo test --workspace` 103 VM tests (plus the rest
of the workspace), core differential 14/14, host-call differential 32/32,
`git diff --check` clean.

## Completed code boundary (verified): generator suspension (LIP-022 §2)

Batch 4's first verifiable boundary — suspendable/resumable generator frames —
is implemented and verified. Async (LIP-024) and the rest of LIP-022 (sets,
comprehensions, iterators) remain deferred to batches 10 and 5 respectively.

1. Bytecode advances to **LABC v3** only for programs containing generators.
   `OP_GENERATOR` (73), `OP_YIELD` (74), and `OP_NEXT` (75) are appended before
   `OP_COUNT`; the verifier's version-aware opcode ceiling rejects the three
   v3-only opcodes in v1/v2 chunks. The on-disk format is unchanged (magic +
   version + counts + data), so v3 is a version bump plus new opcodes.
2. `VAL_GENERATOR` (30) is a heap-allocated suspended frame snapshot (function,
   saved `ip`, GC-traced registers, `exhausted` flag). `OP_GENERATOR` allocates
   it and copies the `arity` arguments without running the body; `OP_YIELD`
   saves the frame and returns `result_ok(value)`; `OP_NEXT` restores the frame
   and runs to the next `YIELD`/`RETURN`, or returns `result_error("exhausted")`
   once exhausted. Both C and Rust VMs dispatch the three opcodes identically.
3. Compiler: `yield` is a keyword/statement; a function containing `yield` is a
   generator whose call returns `Generator<T>`; `next(it)` returns
   `Result<T, E>`. `emit_modules` emits `.version 3` when any function is a
   generator, else `.version 2` (bootstrap stays byte-stable). Generator
   functions reserve register 0 for the generator value, so their locals start
   at register 1 even at arity 0.

Changed code: `vm/include/bytecode.h`, `vm/c/{assembler,bytecode,vm,value}.c`,
`vm/include/value.h`, `vm/rust/lana-bytecode/src/{opcode,assembler,verifier,loader}.rs`,
`vm/rust/lana-vm/src/{value,vm}.rs`, `compiler/{syntax,parser,resolver,ir,emitter}.lana`,
`compiler/bootstrap/compiler.lasm`.

Checks added: `tests/unit/test_runtime.c` `test_generator_suspension` (laziness,
yield, exhaustion, v3 accept / v2 reject), Rust `generator_yield_returns_result_ok` /
`generator_exhaustion_returns_result_error` / `next_on_non_generator_returns_type_error`,
`tests/conformance/differential/core/generator.lasm`,
`tests/regression/m22_generator_pass.lana` and `m22_generator_exhausted.lana`,
and `tests/test_generator_source.py` diagnostic cases.

Verification: CTest 85/85, `cargo test --workspace` (106 VM tests plus the rest
of the workspace), core differential 15/15, host-call differential 32/32,
focused ASan/UBSan run on the generator tests, byte-stable bootstrap
self-compilation, `git diff --check` clean.

## Completed code boundary (verified): LIP-022 rest — sets, iterators, comprehensions

Batch 5 closes LIP-022. Three boundaries, each verified before the next began.

### Immutable sets (LIP-022 §1)

1. New `VAL_SET` value type (C `LanaSet { count, capacity, items }`; Rust
   `ValueKind::Set(Arc<Mutex<Vec<Value>>>)`), traced like `LanaArray`. Membership
   uses the existing `values_equal`; the set is a linear array (O(n)), consistent
   with the existing `LanaMap`.
2. Six host calls appended after `LANA_HOST_REVOKE` (75): `set_new` (76),
   `set_add` (77), `set_contains` (78), `set_union` (79), `set_intersect` (80),
   `set_difference` (81); `LANA_HOST_COUNT` → 82. Rust-only store/policy/ledger
   IDs shifted 76–86 → 82–92. `set_add`/union/intersect/difference are pure
   (return a new set); `set_add` rejects `STATE`/`STATE_DIST`/`Information` with
   `LANA_ERR_TYPE` via a `value_is_set_member` helper.
3. Compiler type-checks the six builtins (`type_set(element_type)`); stdlib
   `set_add` is now pure. Bootstrap regenerated and byte-stable.

### Iterators (LIP-022 §4)

1. `for x in iterable` loop form (node type 15). Array iterable lowers to an
   index loop; generator iterable lowers to a `next`-driven loop. The resolver
   embeds `is_generator` in the node; the emitter re-resolves the body with the
   loop variable bound.
2. `iter`/`enumerate`/`zip` are stdlib generator functions (lazy, no function
   argument). `map`/`filter`/`reduce` are compiler builtins that capture a
   function NAME at compile time (Lana has no first-class function values), and
   lower to eager inline loops — a deliberate deviation from the outline's
   "lazy", since the emitter cannot synthesize generator functions.

### Comprehensions (LIP-022 §3)

1. `[e for x in xs]`, `[e for x in xs if p]`, `{e for x in xs}`,
   `{k: v for k, v in es}` parse into a uniform node type 44
   (`[44, kind, key_expr, value_expr, key_binding, value_binding, iterable,
   filter, line, column]`). The resolver type-checks the iterable (array or
   generator), binds the loop variable(s), and embeds `is_generator` via
   `array_push` (after line/column, matching the map/filter/reduce pattern).
2. The emitter lowers each form to an inline loop: list → `array_new`/`array_push`,
   set → `set_new`/`set_add` (pure, `MOVE` the new set back), map →
   `map_new`/`map_set` (mutates in place; keys must be strings). Map
   comprehensions destructure the pair into key/value registers.

Changed code: `vm/include/value.h`, `vm/include/bytecode.h`, `vm/c/{vm,value}.c`,
`vm/rust/lana-vm/src/{value,vm}.rs`, `vm/rust/lana-bytecode/src/assembler.rs`,
`compiler/{syntax,parser,resolver,ir,emitter}.lana`,
`compiler/bootstrap/compiler.lasm`, `stdlib/collections.lana`.

Checks added: `tests/unit/test_runtime.c` set cases, Rust `vm.rs` set tests,
`tests/conformance/differential/core/set_ops.lasm`,
`tests/regression/m22_{set,iter,map_filter_reduce,comprehension}_pass.lana` and
`m22_set_state_rejected.lana`, `tests/test_{set,iter,comprehension}_source.py`
diagnostic cases, and CTest registration.

Verification: CTest 93/93, `cargo test --workspace` (all pass), core
differential 16/16, host-call differential 32/32, byte-stable bootstrap
self-compilation, Rust CLI runs the comprehension regression, `git diff --check`
clean.

## Completed code boundary (verified): installed std imports (LIP-016)

The `import "std/…"` path now resolves against an installed stdlib directory,
not just the source tree.

1. New `getenv` host call (id 82, shared C+Rust) returns the named environment
   variable's value or `""` when unset. Rust-only store/policy/ledger IDs shifted
   82–92 → 83–93; the Rust verifier's `LANA_HOST_COUNT` corrected to 94.
2. Compiler `load_module` checks the `std/` prefix before `path_resolve`, so
   `import "std/collections"` resolves to `<stdlib_dir>/collections.lana` instead
   of failing with `LANA_ERR_IO`. The stdlib dir comes from `LANA_STDLIB_DIR`,
   falling back to a cwd-relative `stdlib/`.
3. `lana_compiler_run` (C) and the Rust CLI's `run_compiler_program` derive the
   stdlib dir from the compiler binary's install prefix (`<prefix>/share/lana/stdlib`)
   and set `LANA_STDLIB_DIR` when it exists and is not already set. CMake installs
   `stdlib/` to `${CMAKE_INSTALL_DATADIR}/lana/stdlib`.
4. Bootstrap regenerated byte-stable (two-step, since the compiler now uses
   `getenv` itself).

Changed code: `vm/include/bytecode.h`, `vm/c/{assembler,vm}.c`,
`vm/rust/lana-vm/src/vm.rs`, `vm/rust/lana-bytecode/src/{assembler,verifier}.rs`,
`compiler/{resolver,emitter,main}.lana`, `compiler/bootstrap/compiler.lasm`,
`tools/c/compiler_service.c`, `tools/rust/lana-cli/src/main.rs`,
`cmake/{BuildTargets,CTestTargets}.cmake`.

Checks added: `tests/unit/test_runtime.c` `test_getenv_host_call` (set and unset
env vars), Rust `getenv_returns_value_and_empty_for_unset`,
`tests/regression/std_import_pass.lana` (collections/string/math), and CTest
registration (with `LANA_STDLIB_DIR` pointed at the source stdlib).

Verification: CTest 94/94, `cargo test --workspace` (all pass), core
differential 16/16, host-call differential 32/32, byte-stable bootstrap
self-compilation, `git diff --check` clean.

## Completed code boundary (verified): named stdlib modules + content lock (LIP-016)

The remaining LIP-016 named modules are implemented as pure Lana over the
existing host-call surface, plus two new host calls and the lana.lock content
identity.

1. Two new shared host calls: `random_seed` (id 83, reseeds the VM RNG via
   `lana_vm_seed`/`self.seed`) and `floor` (id 84, `floor(x)`). Rust-only
   store/policy/ledger IDs shifted 83–93 → 85–95; the Rust verifier's
   `LANA_HOST_COUNT` corrected to 96.
2. `std/random` — `random_seed`, `random_float` (`sample_value(random())`),
   `random_int(lo, hi)`, `random_choice(xs)`; all draw functions are
   `stochastic`.
3. `std/iterators` — re-exports `iter`/`enumerate`/`zip` from `std/collections`;
   `map`/`filter`/`reduce` remain compiler builtins (Lana has no first-class
   function values, so they cannot be stdlib functions).
4. `std/datetime` — `now` and `add_duration`; `format_timestamp`/`parse_timestamp`
   are deferred (calendar math needs integer division, which Lana lacks).
5. `std/testing` — added `assert_false`; `assert_raises`/`test` are deferred
   (no try/catch and no first-class functions).
6. `lana.lock` content identity: `project_hash_plan` now hashes the installed
   stdlib directory (`LANA_STDLIB_DIR`, falling back to `stdlib/`) into the
   `content` hash, so a build pins the exact stdlib it was written for.

Changed code: `vm/include/bytecode.h`, `vm/c/{assembler,vm}.c`,
`vm/rust/lana-vm/src/vm.rs`, `vm/rust/lana-bytecode/src/{assembler,verifier}.rs`,
`compiler/{resolver,emitter,main}.lana`, `compiler/bootstrap/compiler.lasm`,
`stdlib/{random,iterators,datetime,testing,self_test}.lana`,
`cmake/CTestTargets.cmake`.

Checks added: `tests/unit/test_runtime.c` `test_random_seed_and_floor` (and
contiguity checks for the two new IDs), Rust `floor_rounds_toward_negative_infinity`
and `random_seed_returns_null_and_random_is_in_range`,
`tests/regression/std_modules_pass.lana` (registered in CTest), and the
`stdlib/self_test.lana` additions.

Verification: CTest 95/95, `cargo test --workspace` (all pass), core
differential 16/16, host-call differential 32/32, byte-stable bootstrap
self-compilation, `git diff --check` clean.

## Completed code boundary (verified): CSV/TOML text codecs (LIP-023 §5)

`std/csv` and `std/toml` are implemented as pure Lana over the host-call
surface, plus two new shared host calls.

1. `string_to_number` (id 85) now returns `Result<number, string>` — a tagged
   pair `[true, n]` / `[false, "invalid number"]` — instead of crashing on bad
   input, so `toml.parse` can reject malformed numbers without aborting.
2. `type_of` (id 86) returns a type-name string for all 22 `ValueType` values,
   so `toml.stringify` can distinguish string/number/bool/map values at runtime.
   Rust-only store/policy/ledger IDs shifted 86–96 → 87–97; the Rust verifier's
   `LANA_HOST_COUNT` corrected to 98.
3. `std/csv` — `parse`/`stringify` text codecs (RFC-4180 quoting) plus the
   existing `read`/`write` file host calls. `parse` returns
   `Result<array<array<string>>, E>`.
4. `std/toml` — `parse`/`stringify` for the schema-1 `lana.toml` dialect (flat
   `key = value`, `[section]` headers, string/number/bool values, `#` comments).
   `parse` returns `Result<map, E>`; `stringify` is deterministic.

Changed code: `vm/include/bytecode.h`, `vm/c/{assembler,vm}.c`,
`vm/rust/lana-vm/src/vm.rs`, `vm/rust/lana-bytecode/src/{assembler,verifier}.rs`,
`compiler/{resolver,emitter}.lana`, `compiler/bootstrap/compiler.lasm`,
`stdlib/{csv,toml,self_test}.lana`, `cmake/CTestTargets.cmake`.

Checks added: `tests/unit/test_runtime.c` `test_string_to_number_and_type_of`,
Rust `string_to_number_returns_result_and_type_of_names`,
`tests/regression/m23_csv_toml_pass.lana` (registered in CTest), and a
`string_to_number_type_of.lasm` differential fixture.

Verification: CTest 96/96, `cargo test --workspace` (all pass), core
differential 16/16, host-call differential 33/33, byte-stable bootstrap
self-compilation, `git diff --check` clean.

## Completed code boundary (verified): JSON text codec (LIP-023 §1–3)

`std/json` is now a proper text codec matching CSV/TOML. The `json_parse` host
call returns `Result<Value, E>` instead of crashing on malformed input, and
`json_stringify` is deterministic.

1. `json_parse` returns a tagged pair `[true, value]` / `[false, "invalid JSON
   at byte N"]`; the error carries the byte offset. `lana_json_parse_offset`
   (C) exposes the offset; the runtime `lana_json_parse` (used by store/ledger/
   adapters) is unchanged and now a thin wrapper.
2. `json_stringify` emits object keys in sorted byte order (LIP-023 §1).
3. Number fidelity (LIP-023 §3): an integer literal whose magnitude exceeds
   2^53 is preserved as a string rather than silently rounded to binary64.
4. Duplicate object keys: last occurrence wins (LIP-023 §2), matching the
   existing `lana_map_set`/`Map::set` overwrite semantics.

Changed code: `runtime/c/data.c`, `runtime/include/data.h`, `vm/c/vm.c`,
`vm/rust/lana-vm/src/vm.rs`, `runtime/rust/lana-runtime/src/data.rs`,
`compiler/resolver.lana`, `compiler/bootstrap/compiler.lasm`,
`stdlib/json.lana`, `cmake/CTestTargets.cmake`.

Checks added: `tests/unit/test_runtime.c`
`test_json_large_integer_duplicate_keys_and_sorted_stringify`, Rust VM
`json_parse_returns_result_and_preserves_large_integers` and
`json_stringify_sorts_keys_and_duplicate_keys_last_wins`, Rust runtime
`json_preserves_large_integer_as_string`/`json_duplicate_keys_last_wins`/
`json_stringify_sorts_keys`, `tests/regression/m23_json_pass.lana` (registered
in CTest), and an expanded `json_ops.lasm` differential fixture.

Verification: CTest 97/97, `cargo test --workspace` (all pass), core
differential 16/16, host-call differential 33/33, byte-stable bootstrap
self-compilation, `git diff --check` clean.

## Completed code boundary (verified): JSON provenance (LIP-023 §4)

`json_parse` now returns `Result<Information<Value>, E>`: the parsed value is
rooted as an `Information` value and its derivation records the source-text
identity (SHA-256), so a decision that consumed a parsed record can be audited
and replayed against the exact input.

1. `json_parse` roots the parsed value via `lana_vm_reactive_root` (C) /
   `reactive_root` (Rust) and attaches a derivation with `kind = evidence`,
   `operation = "json_parse"`, `label = <sha256 hex of the source text>`,
   `details = "root"`, `exactness = exact`.
2. The resolver type is `Result<Information<Value>, E>`. The Information
   wrapping is transparent to existing operations (`map_get`, `index_get`,
   `json_stringify`, `==`, `print`) because the reactive/derivation fields are
   orthogonal to the value's type and contents.
3. `inspect_information` on a parsed value reports `reactive: true` and a
   `derivation` whose `source.label` is the text identity.

Changed code: `vm/c/vm.c`, `vm/rust/lana-vm/src/vm.rs`,
`vm/rust/lana-vm/src/sha256.rs` (new), `vm/rust/lana-vm/src/lib.rs`,
`compiler/resolver.lana`, `compiler/bootstrap/compiler.lasm`,
`tests/regression/m23_json_pass.lana`.

Checks added: `tests/regression/m23_json_pass.lana` provenance assertions, Rust
VM `json_parse_roots_information_with_text_identity`, and the `sha256` FIPS
vectors in `vm/rust/lana-vm/src/sha256.rs`.

Verification: CTest 97/97, `cargo test --workspace` (all pass), core
differential 16/16, host-call differential 33/33, byte-stable bootstrap
self-compilation, `git diff --check` clean.

## Completed code boundary (verified): formatting (LIP-021 §3)

`format` and `format_number` are two new shared host calls (ids 87–88), the
first sub-boundary of LIP-021 text processing.

1. `format(fmt, ...)` is variadic: the first argument is the format string and
   the rest are substituted into `{}` placeholders left-to-right. Numbers use
   `%.17g` (round-trip, matching `number_to_string`), strings are copied
   verbatim, bools render `true`/`false`, null renders `null`, and arrays/maps
   render via `json_stringify`. A placeholder/argument count mismatch is a
   `LANA_ERR_FORMAT` error, not a crash.
2. `format_number(value)` renders `%.17g` (full binary64 round-trip);
   `format_number(value, precision)` renders `%.*f` (fixed, `precision` decimal
   places). Precision is a positional integer ≥ 0 (the spec's `precision:`
   named-argument syntax is deferred — Lana's named-arg parser only supports
   `axis:` today).
3. Rust-only store/policy/ledger IDs shifted 87–97 → 89–99; the Rust verifier's
   `LANA_HOST_COUNT` corrected to 100.

Changed code: `vm/include/bytecode.h`, `vm/c/{assembler,vm}.c`,
`vm/rust/lana-vm/src/vm.rs`, `vm/rust/lana-bytecode/src/{assembler,verifier}.rs`,
`compiler/{resolver,emitter}.lana`, `compiler/bootstrap/compiler.lasm`,
`cmake/CTestTargets.cmake`.

Checks added: `tests/unit/test_runtime.c` `test_format_host_calls` (number/
string/bool/null substitution, round-trip, fixed precision, count-mismatch
error), Rust `format_and_format_number`, `format_ops.lasm` differential fixture,
and `tests/regression/m21_format_pass.lana` (registered in CTest).

Verification: CTest 98/98, `cargo test --workspace` (all pass), core
differential 16/16, host-call differential 34/34, byte-stable bootstrap
self-compilation, `git diff --check` clean.

## Completed code boundary (verified): Unicode core (LIP-021 §1/§4)

Four new shared host calls (ids 89–92) implement code-point iteration and
simple case mapping, the second sub-boundary of LIP-021 text processing.

1. `char_length(s)` returns the number of UTF-8 code points (not bytes).
2. `string_codepoint_slice(s, start, end)` returns a code-point-indexed
   substring (start/end are code-point offsets).
3. `to_upper(s)` / `to_lower(s)` apply simple (1:1) case mapping over code
   points. This is *simple* case mapping, not full case folding: characters
   whose case mapping is multi-character (e.g. U+00DF ß → "SS") are left
   unchanged. `casefold(s)` is a separate future API.
4. All four reject invalid UTF-8 (overlong, surrogate, out-of-range, truncated)
   with `LANA_ERR_SCHEMA` at the host boundary.

The case-mapping table is generated once by `tools/gen_unicode_tables.py` from
UnicodeData.txt 15.1.0 (1450 upper + 1433 lower simple mappings) and consumed
identically by both VMs: `vm/include/unicode_case.h` (C, binary search) and
`vm/rust/lana-vm/src/unicode_case.rs` (Rust, `binary_search_by_key`). No runtime
UCD dependency. NFC/NFD/NFKC/NFKD normalization is deferred to a dedicated
Unicode-normalization LIP.

Rust-only store/policy/ledger IDs shifted 89–99 → 93–103; the Rust verifier's
`LANA_HOST_COUNT` corrected to 104.

Changed code: `vm/include/bytecode.h`, `vm/c/{assembler,vm}.c`,
`vm/rust/lana-vm/src/{vm,lib}.rs`, `vm/rust/lana-bytecode/src/{assembler,verifier}.rs`,
`compiler/{resolver,emitter}.lana`, `compiler/bootstrap/compiler.lasm`,
`cmake/CTestTargets.cmake`, `stdlib/{unicode.lana,README.md}`,
`tools/gen_unicode_tables.py` (new), `vm/include/unicode_case.h` (generated),
`vm/rust/lana-vm/src/unicode_case.rs` (generated).

Checks added: `tests/unit/test_runtime.c` `test_unicode_host_calls` (golden
vectors + invalid-UTF-8 rejection), Rust `unicode_code_points_and_case_mapping`,
`unicode_ops.lasm` differential fixture, and `tests/regression/m21_unicode_pass.lana`
(registered in CTest).

Verification: CTest 99/99, `cargo test --workspace` (all pass, 122 VM tests),
core differential 16/16, host-call differential 35/35, byte-stable bootstrap
self-compilation, focused ASan/UBSan on the new host-call tests, `git diff
--check` clean.

## Completed code boundary (verified): regex engine (LIP-021 §2)

Four new shared host calls (ids 93–96) implement a Thompson NFA regular
expression engine, the third and final sub-boundary of LIP-021 text processing.

1. `regex_compile(pattern)` → `Result<Regex, E>` (a tagged pair `[true, re]` /
   `[false, error]`). Invalid patterns (unterminated group/class, unmatched
   `)`, invalid range, trailing backslash) return a compile-error string.
2. `regex_match(re, text)` → `Result<Match, E>` (anchored full match).
3. `regex_search(re, text)` → `Result<Match, E>` (unanchored leftmost match).
4. `regex_replace(re, text, replacement)` → string (all non-overlapping matches).

`Match` is a map `{start, end, text}` (byte offsets + matched substring); no
capture groups in the core subset. The engine is written from scratch in both
VMs (no `re2`/`regex` crate — they would diverge) and is byte-identical:
parse → AST → Thompson NFA (epsilon + character transitions), then linear-time
simulation over the active-state set (O(text × states), never exponential).
Supported syntax: literals, `.`, `*`, `+`, `?`, `|`, `[...]` (ranges +
negation), `(...)` (grouping only), `^`/`$` (anchors). Semantics are greedy
(leftmost-longest), matching RE2-style per the LIP-021 spec. No
backreferences/lookahead/named groups (they break the linear-time guarantee).

`VAL_REGEX` (C `LanaRegex`, Rust `ValueKind::Regex`) holds the compiled NFA as
an opaque GC leaf (no `Value` references inside, so tracing is a no-op; the NFA
is freed on `lana_value_free`/`Drop`).

Rust-only store/policy/ledger IDs shifted 93–99 → 97–103; the Rust verifier's
`LANA_HOST_COUNT` corrected to 108.

Changed code: `vm/include/{value,bytecode}.h`, `vm/c/{value,assembler,vm}.c`,
`vm/rust/lana-vm/src/{value,vm}.rs`, `vm/rust/lana-bytecode/src/{value,assembler,verifier}.rs`,
`compiler/{resolver,emitter}.lana`, `compiler/bootstrap/compiler.lasm`,
`cmake/CTestTargets.cmake`, `stdlib/regex.lana` (new).

Checks added: `tests/unit/test_runtime.c` `test_regex_host_calls` (compile/match/
search/replace + invalid-pattern rejection), Rust `regex_compile_match_search_replace`,
`regex_ops.lasm` differential fixture, and `tests/regression/m21_regex_pass.lana`
(registered in CTest).

Verification: CTest 100/100, `cargo test --workspace` (all pass, 123 VM tests),
core differential 16/16, host-call differential 36/36, byte-stable bootstrap
self-compilation, focused ASan/UBSan on the new host-call tests, `git diff
--check` clean.

## Completed code boundary (verified): explicit Metal execution (LIP-004 §5)

The GPU half of batch 3 is implemented and verified. GPU execution is explicit
and approximate: `matmul` stays CPU binary64, and a new `gpu_matmul(a, b,
precision)` host call is the opt-in float32 Metal path.

1. New shared host call `gpu_matmul` (id 97, after `regex_replace`). `precision`
   is a string; only `"float32"` is accepted (anything else is `LANA_ERR_TYPE`).
   Complex operands are rejected (`LANA_ERR_TYPE`). The result tensor's
   derivation records `kind = approximation`, `exactness = approximate`,
   `operation = "gpu_matmul"`, `details = "backend=metal precision=float32"`,
   so an approximate GPU result never claims bitwise CPU replay.
2. Metal backend: `vm/metal/matmul.metal` is a single deterministic float32
   sgemm compute kernel (one thread per output element, fixed reduction order).
   `vm/c/metal.m` exposes `lana_metal_sgemm(m, k, n, a, b, c)` over `float*`
   (lazily cached device/pipeline, per-call queue/buffers); the Rust
   `vm/rust/lana-vm/src/metal.rs` mirrors it via the `metal` crate and embeds
   the same `.metal` source with `include_str!`, so both VMs run the identical
   kernel. CMake compiles the `.m` with the Objective-C compiler and links
   `-framework Metal -framework Foundation -lobjc` (Apple only). No Metal
   device (or a failed Metal call) maps to `LANA_ERR_UNSUPPORTED_OPERATION`.
3. Dispatch: `tensor_gpu_matmul` mirrors `tensor_matmul`'s shape/batch/broadcast
   logic but downcasts each packed `double` operand to `float` (round-to-nearest),
   calls `lana_metal_sgemm`, and upcasts the `float` result to `double` (exact).
   The downcast is explicit and deterministic in both VMs; binary64 is never
   silently truncated.
4. Compiler: `gpu_matmul` is a pure host call returning `type_ordinary("tensor")`.
   Bootstrap regenerated byte-stable.

Changed code: `vm/metal/matmul.metal` (new), `vm/c/metal.m` (new),
`vm/include/metal.h` (new), `vm/include/bytecode.h`, `vm/c/{assembler,vm}.c`,
`vm/rust/lana-vm/src/{metal,lib,tensor,vm}.rs`,
`vm/rust/lana-bytecode/src/{assembler,verifier}.rs`,
`compiler/{resolver,emitter}.lana`, `compiler/bootstrap/compiler.lasm`,
`cmake/{BuildTargets,CTestTargets}.cmake`, `vm/rust/lana-vm/Cargo.toml`.

Checks added: `tests/unit/test_tensor.c` `gpu_matmul` (real 2x2 on the device
within float32 tolerance, derivation fields, bad-precision and complex error
cases; skips gracefully when no Metal device), Rust `tensor.rs`
`gpu_matmul_2d_within_float32_tolerance`/`gpu_matmul_rejects_complex`,
`tests/conformance/differential/hostcalls/gpu_matmul.lasm`, and
`tests/regression/m4_gpu_matmul_pass.lana` (registered in CTest).

Two latent bugs fixed along the way:

- The Metal shader used `get_global_id(0/1)` — an OpenCL builtin, not Metal —
  so the library failed to compile and `lana_metal_sgemm` always returned
  false. Replaced with the Metal `uint2 gid [[thread_position_in_grid]]`
  attribute in both the `.metal` file and the C embedded string.
- The `tensor_matmul*.lasm` differential fixtures (from the earlier matmul
  batch) were false passes: `array_push R4 2 R4` reads the pushed value from
  register R5 (the register after the array), but the fixtures loaded it into
  R2, and the two matmul operands were not in consecutive registers. Both VMs
  failed identically with `LANA_ERR_TYPE`, so the differential check reported a
  match without ever exercising matmul. Rewrote the three fixtures (and the new
  `gpu_matmul.lasm`) with correct value registers and consecutive tensor
  operands; they now produce the expected results (dot 32, 2x2 `[[19,22],[43,50]]`,
  vector-matrix `[22,28]`, matrix-vector `[14,32]`, complex `-I`, and the
  incompatible-inner-dims `LANA_ERR_INVALID_PARAMETERS`).

Verification: CTest 101/101, `cargo test --workspace` (all pass, 125 VM tests),
core differential 16/16, host-call differential 37/37 (C/Rust byte-identical
through the same Metal device), byte-stable bootstrap self-compilation, focused
ASan/UBSan on the tensor tests (including `gpu_matmul`), `git diff --check`
clean for the unstaged edits.

## Remaining after that (re-audited 2026-09-07)

The 2026-09-06 list overstated the remaining work. LIPs 004–014, 016, 021, 022,
and 023 are implemented and conformance-tested (differential fixtures exist for
005/006/007/008/009/010/011/014). What is genuinely left:

1. **LIP-016 deferred stdlib** — `format_timestamp`/`parse_timestamp` (now
   unblocked: `idiv`/`mod`/`ceil` added to `stdlib/math.lana` in B1).
   `assert_raises`/`test` stay deferred (need try/catch + first-class functions);
   stdlib `map`/`filter`/`reduce` stay compiler builtins.
2. **LIP-015 datasets** — lazy bounded datasets and the transactional
   store/ledger exist; the relational algebra (join/scan/aggregate/select),
   inspectable optimizer, stable aggregation order, MVCC, and WAL recovery are
   missing.
3. **LIP-018 FFI** — adapter plugins load via `dlopen`, but process isolation,
   signature validation, and bounded marshalling are missing.
4. **LIP-019 networking** — unbuilt: no TCP/HTTP/TLS, no bind/listen/accept
   server, no recorded responses.
5. **LIP-020 REPL/export** — unbuilt: no persistent compiled session, multiline
   input, error recovery, or replayable export.
6. **LIP-024 async** — v3 generator suspension is done; async frames
   (`run_async`, cold futures, ready ordering, recorded I/O replay,
   cancellation, shared budgets) are not.
7. **LIP-025 WASM** — the `lana-wasm` crate exists but its conformance tests run
   on the native host; browser/WASI execution and host-capability acceptance are
   not proven.
8. **Release qualification** — all exact-candidate gates (C/Rust, full
   sanitizers, fuzzing, universal install, integrations, hardware GPU, WASM,
   archives).
9. **LIP-027 matmul backend speedup (B16 follow-up)** — the fp32-accumulation
   path (f32/f16/bf16) uses a naive scalar loop, ~33x slower than the f64
   `cblas_dgemm` path (measured 2026-09-07: 128x128x200, f64 0.03s vs f32/f16/
   bf16 ~1.0s, C11 VM; consistent on Rust VM). Root cause: f64 lowers to
   Accelerate BLAS, the low-precision dtypes to a scalar loop. Fix: lower f32
   to `cblas_sgemm` on macOS (naive loop elsewhere) and pack f16/bf16 to
   `float` scratch before `cblas_sgemm` (PyTorch-style), preserving fp32
   accumulation and C/Rust byte-identity (both VMs select the same backend per
   platform). Recorded in `PERFORMANCE.md` §6; the fp16/bf16 path should not be
   relied on for speed until this lands.

Items 4–6 are unbuilt; 2, 3, 7 are partial; 1 and 8 are small/qualification;
9 is a performance regression on a shipped LIP-027 path.

## Re-planned batches (2026-09-07)

Each batch is one verifiable boundary; complete and verify before starting the
next. Ordered by dependency.

| Batch | Boundary | Status |
|---|---|---|
| B1 | Integer division + `ceil`/`mod` (stdlib over `floor`) | ✅ Done 2026-09-07; CTest 123/123 |
| B2 | LIP-016: `format_timestamp`/`parse_timestamp` | ✅ Done 2026-09-07; CTest 123/123, C+Rust VMs |
| B3 | LIP-024: async frames (`run_async`, cold futures, ready ordering) | ✅ Done 2026-09-07; CTest 129/129, C+Rust byte-identical |
| B4 | LIP-024: recorded I/O replay, cancellation, shared budgets | ✅ Done 2026-09-07; CTest 129/129, C+Rust byte-identical |
| B5 | LIP-015: relational algebra (join/scan/aggregate/select) + inspectable optimizer | ✅ Done 2026-09-07; CTest 133/133, C+Rust byte-identical |
| B6 | LIP-015: MVCC, transactions, WAL recovery, adapters | ✅ Done 2026-09-07; CTest 133/133, run_durable 4/4, C+Rust byte-identical |
| B7 | LIP-018: process-isolated loading, signature validation, bounded marshalling | ✅ Done 2026-09-07; CTest 134/134, run_ffi 3/3, C+Rust byte-identical |
| B8 | LIP-019: bounded TCP/HTTP/TLS + minimal server + recorded responses | Pending |
| B9 | LIP-020: persistent REPL + multiline input + error recovery + replayable export | Pending |
| B10 | LIP-025: browser/WASI execution + host-capability acceptance | Pending |
| B11 | Release qualification: all exact-candidate gates | Pending |
| B12 | LIP-027 S1: `dtype` field + `dtype:` on constructors + `dtype(t)` + element-width allocation | ✅ Done 2026-09-07; `LANA_HOST_TENSOR_DTYPE`, `tensor_new_dtype`, element-width buffers, CTest 140/140 |
| B13 | LIP-027 S2: `cast(t, dtype)` host call | ✅ Done 2026-09-07; `LANA_HOST_TENSOR_CAST`, same-dtype no-op, complex out of scope, CTest 140/140 |
| B14 | LIP-027 S3: element-wise same-dtype rule (mismatch → `LANA_ERR_TYPE`) | ✅ Done 2026-09-07; `a->dtype != b->dtype → LANA_ERR_TYPE` (vm.c tensor_elementwise, tensor.rs:427), CTest 140/140 |
| B15 | LIP-027 S4: matmul mixed dtype + `out_dtype:` + fp32 accumulation | ✅ Done 2026-09-07; `tensor_matmul_dtype_pass`, `tensor_matmul_fp32_accum_pass`, CTest 140/140 |
| B16 | LIP-027 S5: backend `sgemm` (f32) + f16/bf16 native loops, both VMs | ✅ Done 2026-09-07; CTest 140/140, C+Rust byte-identical. ⚠️ fp32 path is a naive loop, ~33x slower than f64 BLAS — see item 9 |
| B17 | LIP-027 S6: reductions preserve dtype + fp32 accumulation | ✅ Done 2026-09-07; axis reductions preserve dtype (incl. complex axis reduce), full reductions return a number, fp32 accum for f16/bf16, CTest 140/140 |
| B18 | LIP-027 S7: element-width accounting + provenance dtype + replay | ✅ Done 2026-09-07; `memory_accounting` (f16 quarter-width), provenance dtype, replay per dtype, CTest 140/140 |
| B19 | LIP-027 S8: tests + differential fixtures + `PERFORMANCE.md` measurement | ✅ Done 2026-09-07; CTest 140/140, hostcalls 61/61, `PERFORMANCE.md` §6 records the fp16/bf16 regression |

B2 is small (unblocked by B1). B3–B10 are the real remaining features. B11 is
the release gate. B12–B19 (LIP-027) are independent of B5–B10 and can run in
parallel; B11 must incorporate them once they land. `assert_raises`/`test` and
stdlib `map`/`filter`/`reduce` remain deferred (blocked on try/catch +
first-class functions, which are not in scope).

## Implementation cautions

- The generic unresolved-value walk is still recursive. C ancestor checks have
  O(depth^2) cost; extremely deep non-tensor graphs are not newly qualified.
- LIP-011 grad/vjp is implemented and conformance-tested; the saved-operand
  coverage is limited to the supported control-flow subset, so gradients over
  unsupported paths are rejected, not silently wrong.
- Existing WASM host tests are native tests, not browser/WASI execution proof.
- Never identify zero instantaneous gradient with absent dependency; optimizer
  momentum and the structural graph matter.
- No public registry is needed for installed `std/` imports or existing local
  project dependency locks.

## Resume commands

```bash
git status --short
git diff --check
git diff --cached --check
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel 6
ctest --test-dir build --output-on-failure -j 4
cargo test --workspace
cargo build -p lana-cli
bash tests/conformance/differential/run_core.sh
bash tests/conformance/differential/run_hostcalls.sh
LANA_COMPILER_LABC="$PWD/build/lana-compiler.labc" target/debug/lana-cli run tests/regression/tensor_boundaries_pass.lana
```
