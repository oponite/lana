# Lana 4.0 support matrix

| Interface | Status | Bytecode contract |
| --- | --- | --- |
| `lana` Rust CLI and VM | Canonical production runtime | Executes LABC v1-v5 and v6 immutable values and task-local classes with checked construction and graph transfer; checked methods, interface dispatch, and object source compilation are supported. Compiler emits the lowest supported version. |
| JSON, MCP, Jupyter, and editor adapters | Supported through the Rust CLI | Source and LABC v1-v6 through Rust; JSON export still rejects unsupported opaque values. |
| Python `Lana` worker | Supported for repeated calls | One fresh VM per request; no automatic effect retry. |
| Lana native-library calls | Supported for bounded scalar signatures | Requires the `ffi` capability on native targets. |
| `lana-wasm` | Source compilation and execution checked in Node on wasm32-unknown-unknown | Embedded compiler and stdlib; portable computation and inline task execution. Native filesystem, sockets, TLS, clock/waits, dynamic libraries and durable host services return unsupported errors. |
| C VM, C ABI, and standalone evidence HTTP server | Retired | Use Rust CLI or JSON bridge. |

`lana version` reports LABC v2 as the default assembly format; this does not
limit v1-v6 loader compatibility. Published v1-v2 goldens preserve coverage.
Run `bash tools/rust/lana-wasm/tests/run-wasm-conformance.sh` for actual WASM
execution. It selects the Rustup compiler and requires the wasm32 target and
wasm-bindgen CLI 0.2.100. Native rlib tests alone do not qualify this interface.
