# Migrating to Lana 4.2.0

Published bytecode and store encoding are unchanged. Decimal exponent literals
(`1e-5`) are new syntax. `sample` on paths from an unweighted guard returns
`LANA_ERR_UNSUPPORTED_OPERATION`; weight the guard to sample its branches.
Oversized bootstrap requests and bytecode files now fail with structured resource errors.
After a store error carrying `durability: "uncertain"`, reopen and inspect the
revision or receipt before retrying. The old handle is closed. Worker errors
preserve the optional `durability` and `path` fields. No effect is retried
automatically. Store journal files must be regular files with a single hard link; symlinks
and shared hard links are rejected.

Benchmark report schema 2 replaces `rust_vm` with `precompiled_cli`. It includes
CLI startup, bytecode loading, and execution, and excludes source compilation.
Historical schema-1 results are not directly comparable for that workload.

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
