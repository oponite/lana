# Versioning

Lana has three independent version axes.

## Language version (semver)

`MAJOR.MINOR.PATCH`, applied to the language and its source contract.

- **MAJOR** — a breaking change to source syntax or programmer-visible
  behavior.
- **MINOR** — a new, backward-compatible feature.
- **PATCH** — a bug fix with no contract change.

The 1.x line targets LABC v1. Lana 2.0 introduces LABC v2.

## LABC version (integer)

The bytecode format version, independent of the language version. It is bumped
only when the encoding changes. The loader accepts LABC v1, v2, v3, and v4
chunks. The compiler emits v2. The `LABC` magic is unchanged. The version
field is `2` for new artifacts.

## Runtime version

The canonical runtime is the Rust implementation; the Rust assembler, verifier,
VM, runtime, FFI, and CLI are the sole actively-developed implementation. The
C11 VM is a frozen reference implementation retained for differential
conformance comparison; it is versioned under the language version but is no
longer actively developed.

## Compatibility

There is no pre-release bytecode compatibility. A published LABC version is a
stable contract; a breaking change requires a new LABC version, not a silent
reinterpretation of an existing one.
