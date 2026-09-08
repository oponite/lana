# Lana Improvement Proposals (LIPs)

A LIP is a written proposal for a change to Lana's language, bytecode, or VM
semantics. It is the mechanism by which a design target becomes an
implementation-ready contract.

Closed proposals are retained in [`archive/`](archive/README.md). Active or
deferred proposals remain in this directory.

[`LIP-026.md`](archive/LIP-026.md) is the design charter: fifteen normative design
requirements plus the ten-question design test that every surface change must
answer. All LIPs carry the resulting "Design test" section (see
[`TEMPLATE.md`](TEMPLATE.md)).

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
   authority order (`papers/semantics.md` → `LIP-026` → `spec/SPEC.md` →
   `spec/BYTECODE.md` → `spec/VM.md`).
4. **Implement** — once accepted, the change is implemented and the LIP status
   moves to `Final` and the document moves to `archive/`.

## Status lifecycle

`Draft` → `Accepted` → `Final`, or `Draft` → `Rejected`.

## Format

Copy [`TEMPLATE.md`](TEMPLATE.md). Every LIP carries: title, status, author,
date, motivation, specification, rationale, design test, compatibility, and
test coverage.

## Mini LIPs (mLIPs)

An **mLIP** is a written proposal for implementation work that makes **no
contract change** — no source syntax, bytecode, VM semantics, or math objects.
It is the vehicle for runtime/backend engineering (fused kernels, arena
allocation, native loops) that is bounded by an existing LIP but does not alter
it. Copy [`mLIP-TEMPLATE.md`](mLIP-TEMPLATE.md). Every mLIP carries: title,
status, author, date, purpose, scope, implementation plan, verification, and
out of scope. An mLIP that turns out to need a contract change is promoted to a
full LIP.
