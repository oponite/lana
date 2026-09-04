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
only when the encoding changes. The dual-version loader accepts both LABC v1 and
v2 chunks; the compiler emits v2. The `LABC` magic is unchanged; the version
field is `2` for new artifacts.

## Runtime version

The canonical runtime is the Rust implementation. The C11 VM is a frozen
reference implementation retained for conformance comparison; it is not
independently versioned.

## Compatibility

There is no pre-release bytecode compatibility. A published LABC version is a
stable contract; a breaking change requires a new LABC version, not a silent
reinterpretation of an existing one.
