# Lana Bytecode

Lana 3.0 uses the `LABC` binary format. `LABC` is the four-byte file magic.
The next 32-bit little-endian field selects bytecode v1 through v5. The
canonical Rust loader accepts these versions and rejects every other magic or
version before execution. The frozen C11 reference accepts v1 and v2 only.

This is a clean compatibility boundary. Lana does not convert pre-release
artifacts. Recompile source with Lana 3.0.

## Layout

The header is `LABC`, version, constant count, function count, instruction
count, and entry offset. Every instruction stores an opcode byte followed by
four 32-bit operands (`a`, `b`, `c`, and `imm`) and a 32-bit source line.
Numbers are IEEE-754 binary64; struct memory and runtime pointers are never
serialized.

## Instruction set

LABC v2 includes the Lana 2.0 runtime surface. LABC v3 and v4 add the
documented autodiff, generator, and async operations. LABC v5 adds the Core
information operations. The source compiler selects the lowest required
version for each program.

Opcodes have stable numeric values within each LABC version. Their names and operands
are defined by `vm/include/bytecode.h`; the verifier checks register ranges,
constant types, function metadata, jump targets, host-call IDs, and every
instruction-specific operand rule before the VM executes a chunk.

## Assembly

Textual assembly uses `.lasm` and may begin with `.version 1` through
`.version 5`. The assembler emits LABC v2 by default. The source compiler
selects a later version when a program needs later operations.

```text
.version 2
STATE_NEW R0 0.5 0.2 0.0
MEASURE R0 R1 probability
HALT
```

The authority order is `../papers/semantics.md`, `SPEC.md`, this document, then
`VM.md`.
