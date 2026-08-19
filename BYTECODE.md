# SSBC Bytecode

SSBC v1 is Lana's portable register bytecode for ordinary values and the native
`STATE(p, d)` primitive. The version was frozen after the C VM, assembler,
verifier, serializer, loader, disassembler, trace mode, structured-state
semantics, sanitizer tests, and end-to-end source compiler passed acceptance.

Serialized files use explicit little-endian fields and never copy raw C struct
layout. The header contains magic `SSBC`, version, constant count, function
count, instruction count, and entry point. Numbers use IEEE-754 binary64.

Every instruction stores an opcode, four 32-bit operands, and a source line.
`STATE_NEW` stores one `p` constant and one `d` constant. No complementary
probability is stored. The verifier checks opcodes, registers, constant types,
function ranges, and jump targets before execution.

Changes to instruction meanings or encoding require a new bytecode version.

## Bytecode v2

SSBC v2 appends opcodes without renumbering any v1 opcode. It adds runtime state
construction, host calls, and structured tasks:

```text
STATE_BUILD
FORK
JOIN
JOIN_TIMEOUT
JOIN_ALL
CANCEL
TASKGROUP_ENTER
TASKGROUP_EXIT
HOST_CALL
```

The loader accepts v1 and v2. The verifier rejects v2-only instructions in a
v1 file. Product version and bytecode version are independent.
