# Governance

## Authority order

Resolve disagreements in this order:

1. `papers/semantics.md` — mathematical meaning.
2. `spec/SPEC.md` — source syntax and programmer-visible behavior.
3. `spec/BYTECODE.md` — the LABC encoding.
4. `spec/VM.md` — runtime architecture and resource behavior.

Do not invent semantics from an implementation detail. When intentionally
changing the language, change the highest applicable authority first.

## Decision model

Lana is maintained by a single maintainer (BDFL). The maintainer is the final
authority on all decisions.

Language, bytecode, and VM changes are proposed and accepted through the LIP
process (`lip/`). A change to the contracts above requires an accepted LIP
before implementation.

Bug fixes, documentation, and tooling do not require a LIP.

## Compatibility promises

- LABC v1 and v2 chunks are both accepted by the dual-version loader.
- Resource limits are 256 MiB and 50,000,000 instructions.
- `UNKNOWN` is preserved; no operation substitutes a default relationship,
  probability, or policy outcome.
- The 1.x authority documents remain authoritative for unchanged behavior.
