# Lana Bytecode

Lana 2.0 uses one binary format: **LABC v2**. `LABC` is the four-byte file
magic. The next 32-bit little-endian field is `2` for new artifacts. The
dual-version loader accepts v1, v2, v3, and v4 chunks and rejects every other
magic or version before execution.

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

LABC v3 adds suspendable generator frames (LIP-022 §2):

- `GENERATOR <func> <arg_reg> <arity> <dest>` — allocate a `VAL_GENERATOR`
  value that owns a suspended frame snapshot (function, saved `ip`, a copy of
  the `arity` arguments, and an `exhausted` flag). The body does not run.
- `YIELD <gen_reg> <value_reg>` — save the current frame's `ip` and registers
  into the generator, pop the frame, and return `result_ok(value)` to the
  `next` call site.
- `NEXT <gen_reg> <dest>` — if the generator is exhausted, produce
  `result_error("exhausted")`; otherwise push a frame, restore the saved `ip`
  and registers, and run until the next `YIELD` or `RETURN`.

`next` returns the existing `Result` tagged-pair encoding (`[bool tag, value]`):
`result_ok(v)` is `[true, v]`, `result_error(e)` is `[false, e]`.

LABC v4 adds suspendable async frames (LIP-024 §4):

- `ASYNC <func> <arg_reg> <arity> <dest>` — allocate a `VAL_FUTURE` value that
  owns a suspended frame snapshot (function, saved `ip`, a copy of the `arity`
  arguments, and an `exhausted` flag). The body does not run.
- `AWAIT <future_reg> <dest>` — if the awaited future is already complete, store
  its result in `dest` and continue; otherwise save the current frame's `ip`
  and registers into the future, pop the frame, and re-queue the awaited future
  on the event loop. On resume the `AWAIT` re-executes and observes the result.
- `RUN_ASYNC <future_reg> <dest>` — run the single-threaded cooperative event
  loop to completion on the given future, then store its result in `dest`.

The four async host calls are `run_async`, `future_all`, `future_race`, and
`sleep`. `future_all` and `future_race` take an array of futures and return a
composite future; `sleep` takes a millisecond duration and returns a composite
future that completes after that duration, yielding `null`. The event loop
schedules ready futures FIFO by creation order, so resumption order (and thus
results) is deterministic for a given computation.

Opcodes have stable numeric values within Lana 2.0. Their names and operands
are defined by `vm/include/bytecode.h`; the verifier checks register ranges,
constant types, function metadata, jump targets, host-call IDs, and every
instruction-specific operand rule before the VM executes a chunk. The verifier
also enforces a version-aware opcode ceiling: v1/v2 chunks reject the v3-only
opcodes, and v1/v2/v3 chunks reject the v4-only opcodes.

## Assembly

Textual assembly uses `.lasm` and may begin with `.version 1`, `.version 2`,
`.version 3`, or `.version 4` when a version directive is supplied. The
assembler emits LABC v2 by default. The native compiler emits v3 only when the
program contains generators and v4 only when it contains async functions; its
own bootstrap artifact stays v2 and byte-stable.

```text
.version 2
STATE_NEW R0 0.5 0.2 0.0
MEASURE R0 R1 probability
HALT
```

The authority order is `../papers/semantics.md`, `SPEC.md`, this document, then
`VM.md`.
