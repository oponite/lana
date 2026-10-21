# Lana 4.1.1 support matrix

## Release installation

| Platform | Release bundle | Native release check |
| --- | --- | --- |
| macOS Apple Silicon and Intel | Universal tar/gzip archive and shell installer | Both slices execute outside the checkout. |
| Windows x64 | MSVC ZIP archive and PowerShell installer | Windows x64 build, compiler/project tests, and installation. |
| Windows ARM64 | Native MSVC ZIP archive and PowerShell installer | Windows ARM64 build, compiler/project tests, and installation. |
| Linux x64 and ARM64 (glibc 2.35+) | Native tar/gzip archives and shell installer | Ubuntu 22.04 x64/ARM64 build, compiler/project tests, and installation. |

Installer availability starts with release 4.1.1. Linux and Windows support
requires passing the native jobs for the exact candidate; adding a workflow
alone does not establish qualification. Linux binaries do not support musl. These
checks cover the CLI/compiler/project installation workflow, not every optional
integration or GPU backend. Signing and notarization remain deferred.

## Runtime interfaces

| Interface | Status | Bytecode contract |
| --- | --- | --- |
| `lana` Rust CLI and VM | Canonical production runtime | Executes LABC v1-v5 and v6 immutable values and task-local classes with checked construction and graph transfer; checked methods, interface dispatch, and object source compilation are supported. Compiler emits the lowest supported version. |
| JSON, MCP, Jupyter, and editor adapters | Supported through the Rust CLI | Source and LABC v1-v6 through Rust; JSON export still rejects unsupported opaque values. |
| Python `Lana` worker | Supported for repeated calls | One fresh VM for each `run`; live methods retain a process-local VM and named Information graph until deletion or worker shutdown. No automatic effect retry. |
| Rust `LiveHost` and `lana live` | Supported for process-local sessions | Initial code runs once; host observations update registered roots and pure descendants transactionally. Handles have no durable identity. |
| Lana native-library calls | Supported for bounded scalar signatures | Requires the `ffi` capability on native targets. |
| `lana-wasm` | Source compilation and execution checked in Node on wasm32-unknown-unknown | Embedded compiler and stdlib; portable computation and inline task execution. Native filesystem, sockets, TLS, clock/waits, dynamic libraries and durable host services return unsupported errors. |
| C VM, C ABI, and standalone evidence HTTP server | Retired | Use Rust CLI or JSON bridge. |

`lana version` reports LABC v2 as the default assembly format; this does not
limit v1-v6 loader compatibility. Published v1-v2 goldens preserve coverage.
Run `bash tools/rust/lana-wasm/tests/run-wasm-conformance.sh` for actual WASM
execution. It selects the Rustup compiler and requires the wasm32 target and
wasm-bindgen CLI 0.2.100. Native rlib tests alone do not qualify this interface.
