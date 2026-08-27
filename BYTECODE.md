# Lana Bytecode

Lana 1.1 uses one binary format: **LABC v1**. `LABC` is the four-byte file
magic. The next 32-bit little-endian field is always `1`. The loader rejects
every other magic or version before execution.

This is a clean compatibility boundary. Lana 1.1 neither reads nor converts
artifacts made by pre-release toolchains. Recompile source with Lana 1.1.

## Layout

The header is `LABC`, version, constant count, function count, instruction
count, and entry offset. Every instruction stores an opcode byte followed by
four 32-bit operands (`a`, `b`, `c`, and `imm`) and a 32-bit source line.
Numbers are IEEE-754 binary64; struct memory and runtime pointers are never
serialized.

## Instruction set

LABC v1 includes the complete Lana 1.1 runtime surface: state construction and
transformation, lazy state distributions, basis measurement and estimation,
arrays and maps, functions, tasks, host boundaries, Information values,
provenance, claims, planned effects, and shared Information capabilities.

Opcodes have stable numeric values within Lana 1.1. Their names and operands
are defined by `include/lana/bytecode.h`; the verifier checks register ranges,
constant types, function metadata, jump targets, host-call IDs, and every
instruction-specific operand rule before the VM executes a chunk.

## Assembly

Textual assembly uses `.lasm` and must begin with `.version 1` when a version
directive is supplied. The assembler emits LABC v1 only. The native compiler
and its bootstrap artifact follow the same rule.

```text
.version 1
STATE_NEW R0 0.5 0.2 0.0
MEASURE R0 R1 probability
HALT
```

The authority order is `papers/semantics.md`, `SPEC.md`, this document, then
`VM.md`.
