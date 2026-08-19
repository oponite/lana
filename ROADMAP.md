# Lana Roadmap

## Foundation

Implement and validate the C semantic runtime, register VM, assembler,
portable bytecode, disassembler, trace mode, and sanitizer tests.

The obsolete Python v0 VM is retained under `reference/python_v0/` for
historical comparison only.

## Language

Compile the new brace syntax with Python tooling into verified SSBC executed by
the C VM.

## After Bytecode v1

Bytecode v1 is frozen. Add broader collections, standard-library I/O, fuzzing,
and performance work without changing its existing instruction meanings.
JIT compilation, concurrency, and a REPL are deliberately deferred.
