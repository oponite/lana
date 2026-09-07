# mLIP-001: Fused training step kernels

- **Status:** Draft
- **Author:** Vijay
- **Date:** 2026-09-07

## Purpose

A training step (LIP-006) is currently a sequence of tensor host calls
(LIP-004) — forward ops, backward ops, optimizer update — each crossing the
VM boundary, allocating, and GC-accounting. For a step with thousands of ops,
that per-op dispatch overhead dominates the math for small tensors. This mLIP
fuses the forward + backward + update of a step into a single native routine
that runs without returning to the interpreter between ops.

## Scope

Implementation only. **No contract change**: the language surface (`train`,
`grad`, `vjp`, tensor ops), the LABC encoding, VM semantics, and the
mathematical objects are all unchanged. A fused step produces the same values
as the unfused path. Bounded by **LIP-004** (tensor), **LIP-006** (train),
**LIP-011** (autodiff), and **LIP-027** (dtypes).

## Implementation plan

1. **Identify the step graph.** In the `train` host call, collect the forward
   ops, the backward ops (LIP-011 reverse-mode rules), and the optimizer update
   into one ordered op graph per step.
2. **Emit a fused routine.** Lower that graph to a single native routine (a
   sequence of backend calls with no interpreter round-trip between them),
   reusing the LIP-004 dispatch point so C/Rust byte identity holds.
3. **Arena-allocate the tape and activations.** Allocate the step's tape and
   activation buffers in one arena; free the whole arena at step end instead of
   per-object GC. This is the memory half of the win.
4. **Preserve step-boundary provenance.** Materialize the LIP-006 step record
   (parameters, optimizer state, provenance) at each step boundary exactly as
   today. Only the *inside* of a step is fused; the audit contract is intact.
5. **Keep the unfused path.** Retain the existing per-op path as the fallback
   (e.g. for `grad`/`vjp` used outside `train`), so behavior is unchanged
   everywhere else.

## Verification

- **Correctness:** a fused step produces results identical to the unfused path
  within fp tolerance, for the existing LIP-006 training tests (convex loss
  reduction, determinism, provenance).
- **No behavior change:** the full existing test suite still passes; the
  unfused path is unchanged.
- **Effect:** measure host-call round-trips eliminated per step and the
  wall-clock change on a small training run, recorded in `PERFORMANCE.md`.
- **Replay:** step-granularity replay (LIP-006/014) still holds.

## Out of scope

- **No dtype work** — that is LIP-027.
- **No relaxation of byte-identity** — changing the determinism/replay
  contract is a full LIP, not an mLIP.
- **No GPU kernels** — a GPU backend is a separate effort; this mLIP fuses on
  the existing CPU dispatch point.
- **No change to the `train` surface** — `train` keeps its signature and
  semantics.
