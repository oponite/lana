# Migrating to Lana 3.0.1

No source rewrite is required from 3.0.0 to 3.0.1.

- Use `lana` for source programs and LABC v3-v5. It is the canonical Rust CLI
  and VM.
- Keep native ABI-v1 consumers on `liblana_bridge` and precompiled LABC v2, or
  move them to the JSON bridge. `lana-ffi` is not a drop-in ABI-v1 replacement.
- Rebuild bytecode from source when moving between paths. Lana provides no
  bytecode converter.
- Keep C11 conformance consumers on `lanavm` with LABC v1-v2 only.

C removal is not part of 3.0.1. It is a 4.0 migration only after ABI-v1 is
replaced or retired, consumers have migrated, v1-v2 conformance is retained,
and a release candidate passes without the C runtime.
