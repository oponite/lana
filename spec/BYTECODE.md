# Lana Bytecode

Lana uses the `LABC` binary format. `LABC` is the four-byte file magic.
The next 32-bit little-endian field selects bytecode v1 through v5. The
canonical Rust loader accepts these versions and rejects every other magic or
version before execution. Lana 4.0 has one supported loader: the Rust loader.

This is a clean compatibility boundary. Lana does not convert pre-release
artifacts. Recompile source with Lana.

## Layout

The header is `LABC`, version, constant count, function count, instruction
count, and entry offset. Every instruction stores an opcode byte followed by
four 32-bit operands (`a`, `b`, `c`, and `imm`) and a 32-bit source line.
Numbers are IEEE-754 binary64; struct memory and runtime pointers are never
serialized.

## Instruction set

LABC v2 includes the Lana 2.0 runtime surface. LABC v3 and v4 add the
documented autodiff, generator, and async operations. LABC v5 adds the Core
information operations. Lana 4.0 source `sample` expressions require v5;
published older chunks keep their original version and behavior. The source
compiler selects the lowest required version for each program.

Opcodes have stable numeric values within each LABC version. The following
ordered lists define their numeric values: the first name in each row has the
first number, and each following name increments by one. Values 0-72 are the
published v1-v2 set; later versions append without renumbering them.

| Values | Opcodes in numeric order |
| --- | --- |
| 0-31 | NOP, LOAD_CONST, MOVE, STATE_NEW, STATE_BUILD, TRANSFORM, MEASURE, APPEND, SAMPLE_STATE_DIST, MEASURE_BASIS, ESTIMATE_MEASURE_PROBABILITY, ESTIMATE_MEASURE_DISTRIBUTION, GET_FIELD, GET_INDEX, SET_INDEX, HISTORY_CONFIG, PREVIOUS, CHANGE, VELOCITY, BINARY, UNARY, COMPARE, JUMP, JUMP_IF_TRUE, JUMP_IF_FALSE, ARRAY_NEW, ARRAY_GET, ARRAY_SET, CALL, RETURN, PRINT, HALT |
| 32-39 | FORK, JOIN, JOIN_TIMEOUT, JOIN_ALL, CANCEL, TASKGROUP_ENTER, TASKGROUP_EXIT, HOST_CALL |
| 40-55 | JOINT_BUILD, JOINT_PROJECT, JOINT_CONDITION, JOINT_SAMPLE, RESOLVE, JOINT_BUILD_FINITE, JOINT_RENAME, POSSIBILITY_BUILD, PATH_SPLIT, PATH_JOIN, OBSERVE, INFO_SAMPLE, EVIDENCE, ASSUME, DERIVATION, EXPLAIN |
| 56-72 | MIX, MAP, SUPPORT, EXPECT, VALIDATE, REVISION, ATTENUATE, TRACE_DISTANCE, APPEND_REDUNDANT, APPEND_FULL_REDUNDANCY, APPEND_COMPLEMENTARY, ADT_BUILD, ADT_CASE, ADT_GET, LAZY, FORCE, BOOTSTRAP |
| 73-79 | LOAD_FUNCTION, GENERATOR, YIELD, NEXT, ASYNC, AWAIT, RUN_ASYNC |
| 80-82 | DISTRIBUTION_BUILD, JOINT_CONDITION_MAP, OBSERVE_MAP |

In v5, `JOINT_CONDITION_MAP` and `OBSERVE_MAP` are the two-argument Core
refinement instructions. Joint values take a named evidence map. Definite,
Possibility, and Distribution values take an exact value or unweighted
Possibility subset. Paths reject refinement. The historical three-argument
joint instructions retain their older contract.

Each instruction has `a`, `b`, `c`, and `imm` operands as described above;
unused operands are `0xffffffff`. The verifier checks register ranges,
constant types, function metadata, jump targets, host-call IDs, and every
instruction-specific operand rule before the VM executes a chunk.
Rust host-call ID 188 is the `future_message` local-inbox bridge.
It adds no opcode or bytecode version.

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
