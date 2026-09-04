# Lana Improvement Proposals (LIPs)

A LIP is a written proposal for a change to Lana's language, bytecode, or VM
semantics. It is the mechanism by which a design target becomes an
implementation-ready contract.

## When a LIP is required

A LIP is required for any change to:

- source syntax or programmer-visible behavior (`spec/SPEC.md`),
- the LABC encoding (`spec/BYTECODE.md`),
- runtime architecture or resource behavior (`spec/VM.md`),
- the mathematical objects in `papers/semantics.md`.

A LIP is **not** required for bug fixes, documentation, tooling, or changes that
do not alter the contracts above.

## Process

1. **Propose** — open a numbered `LIP-NNN.md` with status `Draft`.
2. **Discuss** — refine the motivation, specification, and rationale.
3. **Accept / Reject** — the maintainer (BDFL) records the decision in the
   status line. Acceptance requires the change to be consistent with the
   authority order (`papers/semantics.md` → `spec/SPEC.md` → `spec/BYTECODE.md`
   → `spec/VM.md`).
4. **Implement** — once accepted, the change is implemented and the LIP status
   moves to `Final`.

## Status lifecycle

`Draft` → `Accepted` → `Final`, or `Draft` → `Rejected`.

## Format

Copy [`TEMPLATE.md`](TEMPLATE.md). Every LIP carries: title, status, author,
date, motivation, specification, rationale, compatibility, and test coverage.
