# Migrating to Lana 4.0

Lana 4.0 distributes one Rust `lana` executable and its adjacent
`lana-compiler.labc`. The C `lanavm`, C headers, `liblanaruntime`, and native
ABI-v1 bridge are retired. Rebuild pre-release bytecode from source. Published
LABC v1-v5 remains loadable by the Rust CLI.

C ABI callers should use the JSON bridge or the persistent Python `Lana`
worker. Each worker request receives a fresh VM; a process timeout does not
retry an effectful request. Python callers should close `Lana` when done.
C-specific bridge wrappers should be replaced with the JSON response wrapper
in `integrations/lana/bridge.lana`; old wrapper bytecode still runs, but its
result shape is unchanged rather than translated by the retired C facade.

SQLite read-only and localhost HTTP_JSON data adapters run through the Rust
runtime. Lana calls to native libraries remain available through the existing
native-library mechanism. The standalone C evidence HTTP service is retired.
