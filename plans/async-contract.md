# LIP-024 Async/Await — Shared Contract

This document is the authoritative contract for adding async/await to Lana. Every
implementer (compiler, C11 VM, Rust VM) implements against this exact contract.
It is the foundational step; do not deviate from the names, tags, opcodes, or
semantics below without updating this document first.

## 1. Bytecode version

- New LABC version constant: `LABC_VERSION_4 = 4u`.
- A chunk that uses any of the new opcodes (`OP_ASYNC`, `OP_AWAIT`,
  `OP_RUN_ASYNC`) MUST be emitted with `.version 4`.
- Version 4 is a superset of version 3: it accepts every opcode version 3
  accepts, plus the three async opcodes. Version 1 and 2 reject all generator
  and async opcodes.
- `max_opcode_for_version` returns `OP_COUNT` for both v3 and v4; it returns
  `OP_GENERATOR` for v1/v2.
- Every version-acceptance check (assembler `.version` directive, chunk
  verifier, chunk reader) MUST accept `LABC_VERSION_4` in addition to
  `LABC_VERSION`, `LABC_VERSION_1`, and `LABC_VERSION_3`.

## 2. New opcodes

Three opcodes are added immediately before `OP_COUNT` in the `OpCode` enum
(C11 `vm/include/bytecode.h`, Rust `vm/rust/lana-bytecode/src/opcode.rs`).
Their discriminants are stable and MUST match between the two implementations.

| Opcode | Operands | Semantics |
|--------|----------|-----------|
| `OP_ASYNC` | `<target> <arg0> <argcount> <dest>` | Create a **cold** future by calling async function `<target>` with `<argcount>` arguments starting at register `<arg0>`, storing the future in `<dest>`. The body is NOT executed (mirrors `OP_GENERATOR`). |
| `OP_AWAIT` | `<future> <dest>` | Suspend the current async frame until the future in `<future>` completes, then store its result in `<dest>` and yield control to the event loop. |
| `OP_RUN_ASYNC` | `<future> <dest>` | Run the event loop to completion on the future in `<future>`, then store its result in `<dest>`. |

Register encoding (matches the generator family):

- `OP_ASYNC`: `a = <dest>`, `b = <target function index>`, `c = <arg0>`,
  `imm = <argcount>`. `<argcount>` must satisfy
  `0 <= argcount <= LANA_MAX_REGISTERS` and `arg0 + argcount <= LANA_MAX_REGISTERS`.
- `OP_AWAIT`: `a = <future>`, `b = <dest>`.
- `OP_RUN_ASYNC`: `a = <future>`, `b = <dest>`.

Assembler mnemonics: `ASYNC <target> <arg0> <argcount> <dest>`,
`AWAIT <future> <dest>`, `RUN_ASYNC <future> <dest>`.

## 3. AST node types

### 3.1 Async function

An async function is a `function_node` with an `is_async` flag, a sibling to
`is_generator`.

- `function_node` currently is `[1, name, type_params, parameters, body, line, column]`
  with `is_generator` stamped at `node[6]` by the resolver.
- The `is_async` flag is stamped at `node[7]`.
- `node[6]` remains `is_generator`; `node[7]` is `is_async`. A function may be
  both a generator and async only if the language later permits it; for LIP-024
  the resolver rejects a function that is both.

### 3.2 Await expression

`await` is a new expression node with `node[0] == 45`, named `await_node`.

- Constructor: `fn await_node(value, line, column) { return [45, value, line, column]; }`
- `node[1]` is the awaited expression (which must type as a future).

## 4. Host call signatures

Four new host calls are added to the `LanaHostCallId` enum (C11
`vm/include/bytecode.h`, Rust mirror). Signatures:

| Host call | Signature | Returns |
|-----------|-----------|---------|
| `run_async` | `run_async(future) -> value` | The future's result, after running the event loop to completion. |
| `future_all` | `future_all(futures) -> future` | A future that completes when all input futures complete, yielding an array of their results in input order. |
| `future_race` | `future_race(futures) -> future` | A future that completes with the first input future to complete, yielding that future's result. |
| `sleep` | `sleep(ms) -> future` | A future that completes after `ms` milliseconds, yielding `null`. |

## 5. LanaFuture value type

A new value tag `VAL_FUTURE` mirrors `VAL_GENERATOR`.

```c
typedef struct {
    LanaFunction *function;   /* the async function being run */
    uint32_t ip;              /* next instruction to execute */
    Value *registers;         /* the async frame's registers */
    uint32_t register_count;  /* number of registers in the frame */
    bool exhausted;           /* true once the future has completed */
    bool ready;               /* true when the future is runnable (not suspended on await) */
} LanaFuture;
```

- A future is **cold** on creation: `OP_ASYNC` allocates the frame but does not
  execute the body.
- `exhausted` is set when the future's body returns; the result is retained in
  `registers[0]` (or a designated result register) so repeated reads are stable.
- `ready` is false while the future is suspended on an `OP_AWAIT`; the event
  loop only schedules futures with `ready == true`.

## 6. Event loop contract

- **Single-threaded cooperative scheduler.** There is exactly one event loop per
  `run_async`/`OP_RUN_ASYNC` invocation (or per top-level async entry).
- **Deterministic ready ordering: FIFO by creation order.** Futures become
  runnable in the order they were created; the loop always picks the oldest
  runnable future next. This guarantees reproducible scheduling across runs.
- **Cold futures.** Calling an async function returns a future without executing
  the body. The body runs only when the event loop schedules the future.
- **`await` suspends.** `OP_AWAIT` suspends the current async frame (sets
  `ready = false`), records the resumption point, and yields control to the
  loop. When the awaited future completes, the awaiting frame is marked
  `ready = true` and re-queued at the tail of the FIFO.
- **Completion.** When a future's body returns, its result is stored, `exhausted`
  is set, and any futures awaiting it are made ready.
- **`run_async` / `OP_RUN_ASYNC`** drives the loop until the target future is
  `exhausted`, then returns its result. Nested `run_async` calls are permitted
  and simply run a nested loop to completion.

## 7. Compiler contract

### 7.1 Parser

- Recognizes the `async fn` keyword pair and produces a `function_node` with the
  `is_async` flag set (stamped at `node[7]`).
- Recognizes the `await` keyword as an expression and produces an `await_node`
  (`node[0] == 45`).

### 7.2 Resolver

- Types async functions as the `io` effect (they may perform I/O and
  suspension).
- Validates that `await` appears only inside an async function body; an `await`
  outside an async function is a compile error.
- Types the operand of `await` as a future; the result of `await` is the
  future's element/result type.
- Stamps `is_async` at `node[7]` on the resolved `function_node`.
- Rejects a function that is both a generator and async.

### 7.3 Emitter

- Emits `OP_ASYNC` when creating a future from an async function call.
- Emits `OP_AWAIT` for an `await` expression.
- Emits `OP_RUN_ASYNC` for a top-level `run_async` call.
- Bumps the emitted `.version` to `4` when any module contains an async
  function (`has_async`), mirroring how `has_generators` bumps to `3`. The
  version-selection logic becomes: if `has_async` -> `.version 4`; else if
  `has_generators` or `has_ad` -> `.version 3`; else `.version 2`.

## 8. Files touched by this contract

- `vm/include/bytecode.h` — `LABC_VERSION_4`, three opcodes, `VAL_FUTURE` /
  `LanaFuture` (value.h), host call ids.
- `vm/c/bytecode.c` — opcode name table, `max_opcode_for_version`, version
  checks, operand verification, disassembler.
- `vm/c/assembler.c` — `.version` acceptance, `ASYNC`/`AWAIT`/`RUN_ASYNC`
  mnemonics.
- `vm/rust/lana-bytecode/src/opcode.rs` — `LABC_VERSION_4`, three opcodes, names.
- `vm/rust/lana-bytecode/src/verifier.rs` — version check, `max_opcode_for_version`,
  operand verification.
- `vm/rust/lana-bytecode/src/assembler.rs` — `.version` acceptance, mnemonics.
- `vm/rust/lana-bytecode/src/disassembler.rs` — disassembly of the three opcodes.
- `compiler/syntax.lana` — `await_node` constructor (`node[0] == 45`).
- `compiler/parser.lana` — `async fn` and `await` parsing.
- `compiler/resolver.lana` — `io` effect typing, `await` validation, `is_async`
  stamping.
- `compiler/emitter.lana` — `OP_ASYNC`/`OP_AWAIT`/`OP_RUN_ASYNC` emission and
  `.version 4` bump.
