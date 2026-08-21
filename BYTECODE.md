# SSBC Bytecode

## SSBC v5 Information extension

SSBC versions 1 through 4 retain their existing opcode numbers and meanings.
Version 5 appends the following instructions:

| Instruction | Meaning |
| --- | --- |
| `JOINT_BUILD_V5 Rdst Rbase N descriptor` | Build an immutable independent product or declared conditional node from `N` values. Descriptors are compiler-generated metadata, not source syntax. |
| `JOINT_BUILD_FINITE_V5 Rrows Rdst names` | Build an exact finite weighted joint law. Each nested row contains the named values followed by its weight. |
| `JOINT_PROJECT_V5 Rsrc Rdst names` | Build an immutable projection view over the named subset. |
| `JOINT_CONDITION_V5 Rsrc Rdst name Revidence` | Refine a joint by exact evidence for one named member. Impossible evidence returns `SS_ERR_INVALID_CONDITIONING`. |
| `JOINT_SAMPLE_V5 Rsrc Rdst` | Sample definite members into an array without mutating the joint. |
| `RESOLVE_V5 Rsrc Rdst` | Resolve only singleton support; otherwise return `SS_ERR_UNRESOLVED_VALUE`. |
| `JOINT_RENAME_V5 Rsrc Rdst old new` | Return a renamed immutable joint, rejecting unknown or duplicate names. |
| `POSSIBILITY_BUILD_V5 Rarray Rdst` | Build an immutable non-probabilistic finite possibility and dependency identity. |
| `PATH_SPLIT_V5 Rcondition false-label` | Execute both branches for an unresolved Boolean, subject to the path limit. |
| `PATH_JOIN_V5` | Merge the matching guarded branch environments without selecting a result. |
| `OBSERVE_V5 Rsrc Rdst name Revidence` | Apply and record external evidence only after successful refinement. |
| `INFO_SAMPLE_V5 Rsrc Rdst` | Stochastically read one definite value from supported Information without mutation. |

The verifier checks version, register ranges, descriptor constants, and all
operand modes before execution. The old `COMPOSE_JOINT` remains a v1 opcode
with its historical two-state behavior and is not reinterpreted as a general
joint instruction.

SSBC serializes explicit little-endian fields. Every instruction keeps the
existing layout: one byte opcode, four 32-bit operands (`a`, `b`, `c`, `imm`),
and a 32-bit source line. Numbers are IEEE-754 binary64. Struct memory is never
serialized directly.

Version gates are frozen:

- v1 accepts opcodes `0..32` through `HALT`.
- v2 accepts opcodes `0..41` through `HOST_CALL`.
- v3 accepts opcodes `0..47` through `SAMPLE_STATE_DIST`.
- v4 accepts opcodes `0..50` through the basis-aware measurement additions.
- v5 accepts opcodes `0..62` through the current Information extension.
- Versions below 1 or above 5 are rejected.

The v1/v2 instruction meanings and numbers are unchanged. They are legacy
compatibility operations, including their signed-real state model.

## SSBC v3 additions

| Opcode | Name                  | Logical operands                                                  |
| -----: | --------------------- | ----------------------------------------------------------------- |
|     42 | `STATE_NEW_V3`      | destination,`p` constant, `d_re` constant, `d_im` constant  |
|     43 | `STATE_BUILD_V3`    | `p` register, `d_re` register, `d_im` register, destination |
|     44 | `TRANSFORM_V3`      | destination, source, transform identifier, zero                   |
|     45 | `MEASURE_V3`        | source, destination, mode identifier, zero                        |
|     46 | `APPEND`            | left, right, destination, zero                                    |
|     47 | `SAMPLE_STATE_DIST` | source, destination, zero, zero                                   |

The v3 prefix has 48 opcodes. The logical fourth operand uses physical `SSInstruction.imm`;
the serialized instruction layout did not change.

Transform identifiers are `INVERT=0` and `NEUTRALIZE=1`. Measurement identifiers
are `PROBABILITY=0`, `DISTRIBUTION=1`, and `SAMPLE=2`.

`STATE_NEW_V3` assembly accepts numeric literals, which are interned, or `K<n>`
references to existing numeric constants. Disassembly prints both constant
indexes and values. `STATE_BUILD_V3` validates registers at verification time and
values at execution time. Constant construction is validated and canonicalized
through the shared state helper during verification.

`APPEND` runtime operands must be `STATE` or `STATE_DIST`; sampling requires
`STATE_DIST`. Unused v3 operands must be zero. Existing Lana 1.0 source
operations continue to emit these v3 state instructions; basis-aware source
operations emit the v4 instructions below. The assembler still accepts legacy
mnemonics so frozen v1/v2 programs remain operational.

## SSBC v4 additions

The v4 instructions append to the v3 opcode range; all v1-v3 opcode values and
meanings remain unchanged.

| Opcode | Name | Logical operands |
| -----: | --- | --- |
| 48 | `MEASURE_BASIS_V4` | source, destination, basis ID, measurement mode |
| 49 | `ESTIMATE_MEASURE_PROBABILITY_V4` | source, destination, basis ID, positive sample count |
| 50 | `ESTIMATE_MEASURE_DISTRIBUTION_V4` | source, destination, basis ID, positive sample count |

The physical fields are `a=source`, `b=destination`, `c=basis`, and `imm=mode`
for `MEASURE_BASIS_V4`. For either estimate opcode, `imm` is the sample count.
Basis IDs are `computational=0`, `x=1`, and `y=2`. Measurement mode IDs remain
`probability=0`, `distribution=1`, and `sample=2`. The verifier rejects unknown
bases or modes, zero sample counts, and v4 opcodes in v1-v3 chunks.

The assembler forms are:

```text
MEASURE_BASIS_V4 Rsource Rdestination x probability
ESTIMATE_MEASURE_PROBABILITY_V4 Rsource Rdestination x 10000
ESTIMATE_MEASURE_DISTRIBUTION_V4 Rsource Rdestination y 10000
```

Disassembly prints the named basis, mode, and sample count. The serialized
instruction layout remains unchanged.

## SSBC v5 opcode range

The v5 extension uses opcodes 51 through 62 and raises `OP_COUNT` to 63.
Opcodes 51 through 55 retain the originally assigned v5 joint operations;
opcodes 56 through 62 append finite joints, renaming, possibilities, paths,
observation, and general Information sampling. Their logical operands and
semicolon-delimited textual descriptor form are defined at the start of this
document. v5 chunks may contain all prior instructions and are the only chunks
allowed to contain the Information family.

Compiler-only host primitives append to the host-call ID table without changing
`HOST_CALL` itself. `path_resolve(base, relative)` canonicalizes relative module
paths (or `base` itself when `relative` is empty); malformed IDs are rejected by
the verifier before execution.
