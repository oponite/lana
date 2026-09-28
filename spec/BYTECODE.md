# Lana Bytecode

Lana uses the `LABC` binary format. `LABC` is the four-byte file magic.
The next 32-bit little-endian field selects bytecode v1 through v6. The
canonical Rust loader accepts these versions and rejects every other magic or
version. Execution supports v1-v5 and the v6 object instructions described below. Lana 4.0 has one supported loader: the Rust loader.

This is a clean compatibility boundary. Lana does not convert pre-release
artifacts. Recompile source with Lana.

The pending Brain `fit`, `LBRN2`, typed-memory, grounded-chat, forecast,
semantic-retrieval, selector, and package contracts are CLI/runtime file
surfaces. They do not change LABC encoding, opcode numbering, or published
bytecode compatibility.

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
| 83-89 (v6; partial execution) | VALUE_NEW, OBJECT_NEW, OO_GET, OO_SET, OO_CALL, OO_STATIC_CALL, OO_AS_INTERFACE |

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

The finite-information `std/core` calls reserve Rust host-call
IDs 189–199 in this order: `core_entropy`,
`core_conditional_entropy`, `core_mutual_information`, `core_broja`,
`core_kernel`, `core_identity_kernel`, `core_compose_kernels`,
`core_network`, `core_infer`, `core_forget_weights`, and
`core_assign_weights`. The `core_` prefix keeps `core_infer` distinct
from the existing ML `infer` host call. The source compiler lowers the
`std/core` members to these host names using the existing `HOST_CALL`
instruction and function-reference form. The Rust runtime implements
IDs 189–199, all accepted by the verifier. Existing chunks and IDs 0–188 retain their
behavior. No new opcode or LABC layout is required.

The dataset-history calls reserve Rust host-call IDs 200–205
for `dataset_source`, `dataset_query`, `dataset_apply`,
`dataset_snapshot`, `dataset_evidence`, and
`dataset_exclusions` in that order. They use the existing
`HOST_CALL` instruction. IDs 200–205 implement the six dataset-history
calls. Existing dataset IDs 130–140 and their results are
unchanged. The new calls do not require a bytecode version bump.

The `std/rules` calls reserve host IDs 206–211 in
`learn`, `predict`, `save`, `add_counterexample`,
`inspect`, `rollback` order, with `rules_` host-name prefixes.
The `std/trees` calls use IDs 212–216 in
`fit`, `predict`, `explain`, `save`, `load` order, with
`trees_` prefixes. They use existing `HOST_CALL` and store
protocols; IDs 206–211 implement the six rule calls, and IDs 212–216
implement the five tree calls. ID 217 implements walk-forward evaluation. No
published opcode or host ID is renumbered.

`evaluation_walk_forward`, `dataset_sqlite`, and `document_extract`
use host IDs 217–219 respectively. They use the existing
`HOST_CALL` instruction and add no source grammar or opcode.

`snapshot` uses host ID 220. It captures immutable Information without
resolving or observing it and does not require LABC v6.

## LABC v6 object encoding

The Rust loader, assembler, disassembler, descriptor checks and static access
checks support this encoding. All seven object instructions execute with
runtime ownership, type, initialization and effect checks. Interface calls
select the explicitly implemented concrete signature and preserve identity.
Malformed descriptors and effect violations fail before execution, including
debugger entry. Object source emits v6; other source retains its lowest required
version. See `VM.md` for runtime bounds.

LABC v6 keeps the existing header, constant/function/instruction sections,
and 21-byte instruction layout. It appends opcodes 83–89; v1-v5 reject
them. The assembler accepts `.version 6`, and the source compiler chooses
v6 only for an object-model program. All old opcode and host-call numbers
retain their meaning. In v6, UTF-8 string constants used as object
descriptors are decoded strictly; malformed UTF-8 is `LANA_ERR_FORMAT`.

In v6, JSON string constants declaring `schema_version`, `qualified_name`,
and a `kind` of `value`, `class`, or `interface` are descriptor declarations.
Other JSON strings remain ordinary constants.
A descriptor is a canonical UTF-8 JSON string constant with exactly
`{schema_version, kind, qualified_name, fields, methods, implements}`.
`schema_version` is integer 1; `kind` is `value`, `class`, or
`interface`. `qualified_name` is a portable module ID followed by `/`
and the declared type name. Project file IDs start with
`project/` followed by their UTF-8 paths relative to the
project root. Standalone file IDs start with `file/` followed
by their paths relative to the root source's directory. Imported
file IDs use resolved relative paths with `/` separators and no
`.` components; paths outside that root retain leading `../`
components. Standard-library IDs start with `std/` and hosted IDs
with `pkg/owner/repo/`. Compiler path resolution still deduplicates
physical modules; it must reject two distinct physical modules
that produce the same portable ID. Absolute machine paths and
discovery-order indices never enter an ID. Its identity
within the linked chunk is its constant index, never a runtime pointer.
`fields` preserves declaration order and each entry is exactly
`{name, type, visibility, mutable, default_function}`.
`methods` preserves final declaration order and each entry is exactly
`{name, visibility, static, parameter_types, result_type,
effect_mask, function_index, is_init}`. Types use the compiler's
canonical source type spelling. A pure method has mask zero; bits
0–5 grant `observation`, `stochastic`, `io`, `mutation`,
`task`, and `external_call` respectively. `implements` is an ordered
list of qualified interface names. Visibility is `public` or
`private`. Function indices
and effect masks are unsigned u32 JSON integers. `parameter_types` excludes
`self`; an instance body (including `init`) has one extra receiver argument.
An initializer has `result_type: null`; other methods have a type string. A missing function
or default is JSON null. A default function has zero arguments, returns
the declared field type, and has effect mask zero. An interface has no
fields or function bodies;
a value has no mutable field or default. A class descriptor is the
fully flattened final blueprint, including copied/replaced members.
JSON has sorted object keys, no extra whitespace, UTF-8 strings,
and no duplicate keys or unknown fields. All descriptor references
must resolve inside the linked chunk.

Operands use `0xffffffff` for unused slots:

| Opcode | `a` | `b` | `c` | `imm` |
| --- | --- | --- | --- | --- |
| `VALUE_NEW` | destination register | first argument register | value descriptor constant | argument count |
| `OBJECT_NEW` | destination register | first argument register | class descriptor constant | argument count |
| `OO_GET` | destination register | receiver register | defining descriptor constant | field index |
| `OO_SET` | receiver register | value register | defining descriptor constant | field index |
| `OO_CALL` | destination register | packed argument-array register, receiver first | declared descriptor constant | method index |
| `OO_STATIC_CALL` | destination register | packed argument-array register | class/value descriptor constant | method index |
| `OO_AS_INTERFACE` | destination register | source register | interface descriptor constant | `0xffffffff` |

Assembly writes each object instruction as `MNEMONIC Ra Rb descriptor_hex index`.
The descriptor operand is its canonical UTF-8 JSON encoded as hexadecimal,
like `LOAD_STRING`; identical descriptor strings share one constant. `-`
encodes an unused argument register or index (`0xffffffff`). An otherwise
unreferenced interface descriptor can be emitted with `LOAD_STRING`.
The `.version 6` directive must precede instructions and constants.
Version 6 requires strict UTF-8 string constants, distinct function entries,
and function-local jumps; functions cannot fall through into another body.
Entry cannot point to a member/default body, and direct call-like operations
(including generator, async, fork, lazy and bootstrap) cannot reference those bodies.
These checks do not alter v1-v5 verification.

Argument registers for constructors are consecutive from `b` and
verified against the function's register count; when count is zero,
`b` is `0xffffffff`. `VALUE_NEW` outside its owner requires every
field public. `OBJECT_NEW` outside its owner requires a public
initializer, including the implicit zero-argument initializer.
`OO_SET` has no result.
`OO_CALL` and `OO_STATIC_CALL` use a definite array built before the
call; the VM validates its length and declared argument types. An
interface descriptor in `OO_CALL` dispatches by exact name and
parameter signature to the receiver's explicitly implemented
interface, retaining the class object's identity. No instruction
serializes a pointer.

The verifier checks descriptor JSON/schema, uniqueness and canonical
order, type references, implementation signatures/effects, method
function indices and arities, field/default rules, operand ranges,
privacy ownership, and legal opcode version before execution. Each
method/default body has one descriptor owner and a distinct function index;
an initializer body cannot also serve as an ordinary method or default. Direct `CALL` or
`LOAD_FUNCTION` to a method/default function is rejected; it can run
only through these checked operations. Dynamic receiver type and
initialization state are checked again at runtime. A malformed v6
descriptor or illegal private access fails before any instruction
executes; no v1-v5 verifier rule is weakened.

## Assembly

Textual assembly uses `.lasm` and may begin with `.version 1` through
`.version 6`. The assembler
emits LABC v2 by default. The source compiler selects a later version when
a program needs later operations.

```text
.version 2
STATE_NEW R0 0.5 0.2 0.0
MEASURE R0 R1 probability
HALT
```

The authority order is `../papers/semantics.md`, `SPEC.md`, this document, then
`VM.md`.
