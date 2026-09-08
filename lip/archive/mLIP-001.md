# mLIP-001: Autodiff provenance allocation optimization

- **Status:** Final
- **Author:** Vijay
- **Date:** 2026-09-07

## Purpose

A training step (LIP-006) records many internal autodiff derivations. Their
metadata strings were copied into GC storage for every node even though they
are immutable for the lifetime of the VM. This mLIP reuses those strings in C
and shared `Arc<str>` values in Rust.

## Disposition

Implemented after profiling. On the representative small-tensor training run,
GC safepoints consumed 29.4% of wall samples. Reusing immutable autodiff
provenance strings at the shared derivation recorder reduced the warmed median
from 2.78 s to 2.35 s (15.5%).

## Scope

Implementation only. **No contract change**: the language surface (`train`,
`grad`, `vjp`, tensor ops), the LABC encoding, VM semantics, and the
mathematical objects are all unchanged. Bounded by **LIP-004** (tensor), **LIP-006** (train),
**LIP-011** (autodiff), and **LIP-027** (dtypes).

## Implementation plan

1. Detect internal autodiff derivations at the shared recorder.
2. In C, retain pointers to immutable VM-lifetime strings.
3. In Rust, intern the repeated constants with `OnceLock<Arc<str>>`.
4. Leave all other derivations and step-boundary provenance unchanged.

## Verification

- **Correctness:** existing LIP-006 training and provenance tests pass.
- **No behavior change:** C and Rust retain identical derivation content.
- **Effect:** the wall-clock change is recorded in `PERFORMANCE.md`.
- **Replay:** step-granularity replay (LIP-006/014) still holds.

## Out of scope

- **No dtype work** — that is LIP-027.
- **No relaxation of byte-identity** — changing the determinism/replay
  contract is a full LIP, not an mLIP.
- **No fused graph, arena, or GPU kernels** — add one only if a new profile
  shows this smaller fix is insufficient.
- **No change to the `train` surface** — `train` keeps its signature and
  semantics.
