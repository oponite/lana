# Versioning

Lana has three independent version axes.

## Language version (semver)

`MAJOR.MINOR.PATCH`, applied to the language and supported integration contract.

- **MAJOR** — a breaking change to source syntax, programmer-visible behavior,
  or supported integration surfaces.
- **MINOR** — a new, backward-compatible feature.
- **PATCH** — a bug fix with no contract change.

The 1.x line targets LABC v1. Lana 2.0 introduced LABC v2. Lana 3.0 added the
Rust-owned Core surface in LABC v5. Lana 4.0 retires the C reference backend.

## LABC version (integer)

The bytecode format version, independent of the language version. It is bumped
only when the encoding changes. The Rust loader accepts LABC v1-v6. Frozen
v1-v2 goldens preserve legacy coverage. The `LABC` magic is unchanged. The
compiler emits the lowest version that supports
the source form: v2-v4 for established forms, v5 for Core distribution and
map refinement, and v6 for object declarations.

## Runtime version

The canonical runtime is the Rust implementation. Lana 4.0 retires the C VM.

## Compatibility

There is no pre-release bytecode compatibility. A published LABC version is a
stable contract; a breaking change requires a new LABC version, not a silent
reinterpretation of an existing one.
