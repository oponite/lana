# Lana 2.0 — ADTs, pattern matching, generics, exhaustiveness, uncertainty

> Follow-up spec for deferred item 1. Source syntax + opcode encoding + runtime
> behavior. Sits below `papers/semantics-2.md` and above `spec/SPEC.md` /
> `spec/BYTECODE.md` / `spec/VM.md` in the authority order. All new syntax
> satisfies `spec/SYNTAX.md` (SYNTAX-1..12 + Acceptance Principle).

## Scope

Four features, in implementation order:

1. ADT declarations (sum types).
2. Pattern matching (`match`).
3. Generics (parametric polymorphism).
4. Exhaustiveness + explicit uncertainty cases.

## 1. ADT declarations

### 1.1 Source syntax

```
type Result<T, E> = Ok(T) | Error(E);
type Maybe<T> = Some(T) | None;
```

- `type Name<params> = Variant1 | Variant2 | ...;`
- A variant is `Name` (no fields) or `Name(T1, T2, ...)` (one or more fields).
- At least one variant is required.
- `type` is a new keyword; `Name` and variant names are identifiers.

### 1.2 Runtime representation

- A new `ValueType` tag `VAL_ADT`.
- Payload: `{ uint32_t variant; Value *fields; size_t field_count; }`.
- `variant` is a per-type index (`0, 1, 2, ...`) assigned in declaration order.
- `unknown` is a reserved variant index (`0xFFFFFFFF`) available to every ADT.

## 2. Construction

```
let x = Ok(5);
let y = None;
let z = unknown;
```

- `Variant(args)` constructs a value with that variant tag and its fields.
- `unknown` is a keyword that constructs the unknown value (reserved tag).

## 3. Pattern matching

### 3.1 Source syntax

```
match x {
  Ok(value) => print(value)
  Error(reason) => print(reason)
  _ => print("other")
}
```

- Arms are `Pattern => expression`.
- Patterns: `Variant(bindings...)`, `_` (wildcard), `unknown`.
- Bindings name the variant's fields in order.

### 3.2 Exhaustiveness

- A `match` over an ADT must cover every declared variant, or carry a `_`
  wildcard.
- Missing a declared variant is a **compile error**.

### 3.3 `unknown`

- `unknown` is an implicit variant of every ADT.
- Exhaustiveness requires handling `unknown` (or `_`).
- Matching `unknown` never substitutes a default value; it is handled
  explicitly.

## 4. Generics

- `<T, E>` type parameters on type and function declarations.
- `fn identity<T>(x: T) -> T { x }`.
- Type parameters are checked for safe use; an unsound instantiation is a
  compile error.

## 5. Opcode encoding

New opcodes (appended after `OP_APPEND_COMPLEMENTARY` = 66):

| Opcode | `a` | `b` | `c` | `imm` |
|---|---|---|---|---|
| `ADT_BUILD` (67) | dest | base register (first field) | field count | variant tag (constant index) |
| `ADT_CASE` (68) | value | variant tag (constant index) | — | target offset |
| `ADT_GET` (69) | dest | value | field index | — |

`OP_COUNT` becomes 70.

## 6. Runtime behavior

- `ADT_BUILD`: allocate the fields array, set the variant tag, store a `VAL_ADT`
  value in `a`.
- `ADT_CASE`: if the value's variant equals the tag, jump to `imm`; otherwise fall
  through to the next instruction.
- `ADT_GET`: extract field `c` from the value's fields into `a`.

## 7. Errors

| Condition | Error |
|---|---|
| `match` not exhaustive | compile error |
| `ADT_GET` field out of range | `InvalidParameters` |
| `ADT_CASE` / `ADT_GET` on a non-ADT value | `Type` |

## 8. Deferred

- Recursive ADTs (self-referential types).
- Higher-kinded types.
- GADTs.
