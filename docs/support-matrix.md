# Lana 3.0.1 support matrix

| Interface | Status | Bytecode contract |
| --- | --- | --- |
| `lana` Rust CLI and VM | Canonical production runtime | Loads LABC v1-v5; compiler defaults to v2 and emits later versions when needed. |
| `lanavm` C11 binary | Frozen conformance reference | Loads LABC v1-v2 only. |
| JSON, MCP, Jupyter, and editor adapters | Supported through the Rust CLI | Source programs and LABC v2-v5 supported by the Rust path. |
| `liblana_bridge` native ABI v1 | Frozen C bridge | Tested precompiled LABC v2 path. |
| `lana-ffi` | Documented Rust subset | Not an ABI-v1 replacement. |

`lana version` reports `LABC v2` because v2 is the default assembly format. It
does not limit the Rust loader's v1-v5 compatibility range.

The C11 reference remains in 3.x because the native ABI-v1 bridge still links
it. Do not delete C until a 4.0 candidate has a Rust replacement or retirement
plan for ABI v1, migrated C consumers, preserved v1-v2 conformance coverage,
and a passing release candidate with no C runtime.
