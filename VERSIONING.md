# Versioning

Lana has three independent version axes.

## Language version (semver)

`MAJOR.MINOR.PATCH`, applied to the language and its source contract.

- **MAJOR** — a breaking change to source syntax or programmer-visible
  behavior.
- **MINOR** — a new, backward-compatible feature.
- **PATCH** — a bug fix with no contract change.

The 1.x line targets LABC v1. Lana 2.0 introduced LABC v2. Lana 3.0 adds the
Rust-owned Core surface in LABC v5; its Rust loader accepts v1-v5, while the
frozen C reference backend accepts v1-v2.

## LABC version (integer)

The bytecode format version, independent of the language version. It is bumped
only when the encoding changes. The Rust loader accepts LABC v1-v5; the C
reference loader accepts v1-v2. The
`LABC` magic is unchanged. The compiler emits the lowest version that supports
the source form: v2-v4 for established forms and v5 for Core distribution and
map refinement forms.

## Runtime version

The canonical runtime is the Rust implementation. The C11 VM is a frozen
reference implementation retained for conformance comparison; it is not
independently versioned.

## Compatibility

There is no pre-release bytecode compatibility. A published LABC version is a
stable contract; a breaking change requires a new LABC version, not a silent
reinterpretation of an existing one.
