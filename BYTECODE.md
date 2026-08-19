# SSBC Bytecode v1

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
