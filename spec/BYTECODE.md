# Lana Bytecode

Lana 2.0 uses one binary format: **LABC v2**. `LABC` is the four-byte file
magic. The next 32-bit little-endian field is `2` for new artifacts. The
dual-version loader accepts both v1 and v2 chunks and rejects every other magic
or version before execution.

This is a clean compatibility boundary. Lana 2.0 neither reads nor converts
artifacts made by pre-release toolchains. Recompile source with Lana 2.0.

## Layout

The header is `LABC`, version, constant count, function count, instruction
count, and entry offset. Every instruction stores an opcode byte followed by
four 32-bit operands (`a`, `b`, `c`, and `imm`) and a 32-bit source line.
Numbers are IEEE-754 binary64; struct memory and runtime pointers are never
serialized.

## Instruction set

LABC v2 includes the complete Lana 2.0 runtime surface: state construction and
transformation, lazy state distributions, basis measurement and estimation,
arrays and maps, functions, tasks, host boundaries, Information values,
provenance, claims, planned effects, and shared Information capabilities.

Opcodes have stable numeric values within Lana 2.0. Their names and operands
are defined by `vm/include/bytecode.h`; the verifier checks register ranges,
constant types, function metadata, jump targets, host-call IDs, and every
instruction-specific operand rule before the VM executes a chunk.

## Assembly

Textual assembly uses `.lasm` and may begin with `.version 1` or `.version 2`
when a version directive is supplied. The assembler emits LABC v2 by default.
The native compiler and its bootstrap artifact follow the same rule.

```text
.version 2
STATE_NEW R0 0.5 0.2 0.0
MEASURE R0 R1 probability
HALT
```

The authority order is `../papers/semantics.md`, `SPEC.md`, this document, then
`VM.md`.
