# Lana Improvement Proposals (LIPs)

A LIP is a written proposal for a change to Lana's language, bytecode, or VM
semantics. It is the mechanism by which a design target becomes an
implementation-ready contract.

## When a LIP is required

A LIP is required for any change to:

- source syntax or programmer-visible behavior (`docs/spec/SPEC.md`),
- the LABC encoding (`docs/spec/BYTECODE.md`),
- runtime architecture or resource behavior (`docs/spec/VM.md`),
- the mathematical objects in `docs/papers/semantics.md`.

A LIP is **not** required for bug fixes, documentation, tooling, or changes that
do not alter the contracts above.

## Process

1. **Propose** — open a numbered `LIP-NNN.md` with status `Draft`.
2. **Discuss** — refine the motivation, specification, and rationale.
3. **Accept / Reject** — the maintainer (BDFL) records the decision in the
   status line. Acceptance requires the change to be consistent with the
   authority order (`docs/papers/semantics.md` → `docs/spec/SPEC.md` → `docs/spec/BYTECODE.md`
   → `docs/spec/VM.md`).
4. **Implement** — once accepted, the change is implemented and the LIP status
   moves to `Final`.

## Status lifecycle

`Draft` → `Accepted` → `Final`, or `Draft` → `Rejected`.

## Format

Copy [`TEMPLATE.md`](TEMPLATE.md). Every LIP carries: title, status, author,
date, motivation, specification, rationale, compatibility, and test coverage.

## Archived proposals

Older proposals and finalized LIPs are retained in [archive/](archive/README.md).
Archiving preserves their recorded status; it does not mark a deferred proposal
as implemented or establish current conformance.
