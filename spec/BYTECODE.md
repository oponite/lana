# Lana Bytecode

<<<<<<< Updated upstream
Lana 2.0 uses one binary format: **LABC v2**. `LABC` is the four-byte file
magic. The next 32-bit little-endian field is `2` for new artifacts. The
dual-version loader accepts both v1 and v2 chunks and rejects every other magic
or version before execution.
=======
Lana 3.0 adds **LABC v5**. `LABC` is the four-byte file magic. The Rust loader
accepts v1 through v5 and rejects every other magic or version before
execution. C remains the frozen v1-v4 reference backend.

LABC v3 is introduced only for generator suspension (LIP-022 §2): a program
whose functions contain `yield` is emitted as v3, and the three v3-only opcodes
(`GENERATOR`, `YIELD`, `NEXT`) are rejected by the verifier in v1/v2 chunks.
The on-disk format is unchanged — v3 is a version-number bump plus new opcodes,
with no conditional parsing.

LABC v4 is introduced only for async/await (LIP-024): a program whose functions
contain `await` is emitted as v4, and the three v4-only opcodes (`ASYNC`,
`AWAIT`, `RUN_ASYNC`) are rejected by the verifier in v1/v2/v3 chunks. As with
v3, the on-disk format is unchanged — v4 is a version-number bump plus new
opcodes, with no conditional parsing.
>>>>>>> Stashed changes

LABC v5 adds `DISTRIBUTION_BUILD <support> <dest>`, `JOINT_CONDITION_MAP`, and
`OBSERVE_MAP`. `support` is a nonempty
array of `[value, weight]` rows with exact definite values, finite positive
weights, no duplicate values, and total weight within `1e-12` of one.
`INFO_SAMPLE` rejects an unweighted `Possibility`; only `Distribution` has a
sampling law.

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

<<<<<<< Updated upstream
Textual assembly uses `.lasm` and may begin with `.version 1` or `.version 2`
when a version directive is supplied. The assembler emits LABC v2 by default.
The native compiler and its bootstrap artifact follow the same rule.
=======
Textual assembly uses `.lasm` and the Rust assembler accepts `.version 1`
through `.version 5`. The assembler emits LABC v2 by default. The C assembler
accepts only v1-v4. The native compiler emits the lowest required version:
v2-v4 for established forms and v5 for the Rust-owned 3.0 Core runtime.
>>>>>>> Stashed changes

```text
.version 2
STATE_NEW R0 0.5 0.2 0.0
MEASURE R0 R1 probability
HALT
```

The authority order is `../papers/semantics.md`, `SPEC.md`, this document, then
`VM.md`.
