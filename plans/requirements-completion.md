# Remaining requirements completion matrix

This matrix records the authorized active completion plan. The 2026-09-28
ownership and collector audit reopened rows 2, 3 and 8. Checkpoints 61–82 address
those findings and repeat the required local gates. Final evidence is recorded
in plans/requirements-qualification.md and its external content manifest.
Historical checkpoints establish only their stated bounded behavior.

Authority: `papers/semantics.md` → `spec/SPEC.md` → `spec/BYTECODE.md` →
`spec/VM.md`; source syntax additionally satisfies `spec/SYNTAX.md`.
Compiler limits remain 256 MiB and 50,000,000 instructions.

| Batch | Active requirement and implementation boundary | Success evidence required | Failure evidence required | Status |
| --- | --- | --- | --- | --- |
| 1 | Inventory and fixed contracts; this matrix and local REQUIREMENTS.md | Every sweep finding assigned to an implementation or qualification row | Explicit exclusions remain excluded; stale completion claims corrected | Recorded |
| 2 | Heap-owned non-moving handles and external roots; VM heap, values, classes, tasks, Rust callers | Aliases/identity survive execution and VM teardown; last external root releases cycles; all graph-bearing values traced | Locked/reentrant callbacks, constructor rollback, cancelled/OOM work and transfers preserve live values and bounds | Verified locally; tracked handles, managed DAG/task ownership, budgeted join retirement and failure checks; checkpoint 82 |
| 3 | Generational and incremental policy; spec/VM.md and plans/gc-performance.md | Young/old/stable-shared, promotion, remembered edges, bounded roots/edges/mark/sweep; max 128 work units per routine slice; Release p99 ≤10 ms for deep and wide 20,000-node graphs | Mutations and new roots between slices; sharing/tasks; fallback; low-memory bootstrap | Verified locally; generations, barriers, interruption, 128-unit slices and Release p99 acceptance; checkpoint 82 |
| 4 | Actual wasm32 execution; Cargo features, runtime host boundaries, lana-wasm | Rustup target build and matching wasm-bindgen; Node conformance with embedded stdlib, tasks and repeated calls | Malformed source, resource limits, unsupported host operations return errors without traps; native TLS/config still pass | Verified locally; checkpoints 51 and 82 |
| 5 | HTTP/1.1 framing and headers; VM network hosts | Local HTTP and trusted HTTPS, fragmented reads, chunk extensions/trailers, interim/bodyless responses, ordered duplicate headers, outbound headers | Injection, ambiguous lengths, invalid status/framing, truncation, timeout and resource-limit rejection | Verified locally; checkpoints 52 and 82 |
| 6 | Compiler symbol identity and workspace LSP; compiler/main.lana, resolver.lana, CLI, integrations | Cross-module definition/references/rename, shadowing, aliases, unsaved overlays, Unicode positions; real Neovim, VS Code and Python checks; bootstrap | Invalid names/collisions, incomplete analysis and dependency-source rename cause no partial edits | Verified locally; checkpoints 53 and 82 |
| 7 | Installed-prefix packaging, CI and current docs | Checksum/extract/run outside repo using packaged compiler+stdlib+license; actual WASM/editor/failpoint jobs; current benchmark test | Missing asset/package failure preserves old output; fault-injection artifacts isolated from normal builds | Verified locally; checkpoints 54 and 82 |
| 8 | Exact final tree qualification; AGENTS.md five gates | Content manifest including required untracked files; Release acceptance and locked workspace; twice stable bootstrap; frozen fixtures; ten-minute fuzz; both macOS slices; clean archives; Python/HF; paired 3.0.2 performance | No unexplained required skip, failed gate or stale evidence; changed content invalidates affected evidence | Verified locally; exact content manifest and all required local gates; checkpoint 82. Historical failed performance remains preserved for its superseded candidate |

## Fixed interface decisions

1. Rust embedding may migrate to tracked handles and explicit host roots. A
   retained host value survives its VM; document migration for all changed
   public APIs. Lana source, bytecode and Python/JSON interfaces are preserved.
2. Existing low-level HTTP signatures and text bodies stay unchanged. Response
   `headers` and `trailers` map lowercase names to ordered arrays of strings.
   Request header values are strings. Boolean `verify` is TLS control, never a
   wire header. Callers cannot override transport framing.
3. LSP uses canonical source paths, declaration identities and source spans.
   Workspace edits cover all editable references or fail without partial edits.
   Dependencies remain navigable but their source cannot be renamed.

## Classification of other sweep findings

| Finding | Classification and disposition |
| --- | --- |
| Python gate accepts only LABC v2-v5; editors accept v2-v6 | Active compatibility gap, batch 6 |
| CLI LSP version is hardcoded; prepareRename advertised without implementation | Active protocol gaps, batch 6 |
| HTTP low-level headers ignored; response headers empty | Active existing API gaps, batch 5 |
| Native rlib tests stand in for actual WASM; native ring dependencies leak into target | Active build/execution gap, batch 4 |
| tests/test_publication_failpoints.py is not wired to regular verification | Active qualification gap, batch 7 |
| tests/test_benchmark.py imports removed benchmark/run_benchmark.py | Active orphan verification gap, batch 7; use current harness |
| Root package.sh omits installed stdlib/license | Active packaging gap, batch 7 |
| docs/RELEASE_CHECKLIST.md ignored by basename pattern and uses CTest | Active documentation/qualification gap, batch 7; use a nonignored canonical path |
| Brain fit/layers/memory/forecasts/workshop and finite Core/object pending labels | Existing implementations and focused tests are recorded in requirements-implementation.md; reconcile each stale active label against code/tests in batch 7 |
| Old C collector/compiler benchmark results | Historical evidence only, never current Rust qualification |
| Older ADT/data/ML proposal labels superseded by current contracts | Historical/superseded; reconcile authority links, do not implement a second contract |
| Cross-variable guarded Paths correlation | Research proposal, outside active requirements |
| Continuous laws, sampled estimation, unnamed inference/ML/data families | Explicit exclusions |
| Recursive learned rules/programs, recursive ADTs/GADTs and unspecified conversions | Deferred/research, outside this plan |
| Arbitrary connectors/custom source headers/credential storage/retries/MCP/SFTP/CSV | Explicit exclusions; existing low-level HTTP headers remain in batch 5 |
| Exactly-once external delivery | Explicitly deferred without an external proving protocol |
| Information deltas in semantics-2.1 draft | Non-normative research; no committed feature |
| Signing/notarization, Homebrew submission and publication | Distribution boundary, excluded |

## Evidence policy

Each batch checkpoint records changed behavior, commands, results and remaining
work in `requirements-implementation.md`. Final evidence identifies the whole
working source by content, including required untracked files. HEAD alone cannot
identify this dirty candidate. Historical test logs are not rerun evidence.
Do not commit, tag or publish as part of this plan.
