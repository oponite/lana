# Lana 4.0 support matrix

| Interface | Status | Bytecode contract |
| --- | --- | --- |
| `lana` Rust CLI and VM | Canonical production runtime | Loads LABC v1-v5; compiler emits the lowest supported version. |
| JSON, MCP, Jupyter, and editor adapters | Supported through the Rust CLI | Source and LABC v1-v5 through Rust. |
| Python `Lana` worker | Supported for repeated calls | One fresh VM per request; no automatic effect retry. |
| Lana native-library calls | Supported for bounded scalar signatures | Requires the `ffi` capability on native targets. |
| C VM, C ABI, and standalone evidence HTTP server | Retired | Use Rust CLI or JSON bridge. |

`lana version` reports LABC v2 as the default assembly format; this does not
limit v1-v5 loader compatibility. Published v1-v2 goldens preserve coverage.
The wasm32 cross-build is not qualified in this candidate because its existing
`ring` and `getrandom` dependencies fail for that target.
