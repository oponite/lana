# Lana Syntax Design Principles

This document is the design authority for Lana source syntax. `SPEC.md` is the
1.x contract — the "what" of existing syntax. This document is the "how to
design new syntax": the principles every new or changed language construct MUST
satisfy. New syntax must satisfy both.

## SYNTAX-1 — Tired-Friendly Readability

Lana syntax MUST remain understandable under low attention and fatigue.

A developer SHOULD be able to infer the meaning of a statement by reading it
left-to-right without mentally decoding punctuation or unusual symbols.

## SYNTAX-2 — One Canonical Form

Each language operation MUST have one canonical syntax.

Lana MUST NOT support multiple equivalent forms for the same operation unless
compatibility requires it.

Example:

```lana
measure x
```

rather than also supporting:

```lana
measure(x)
x.measure()
MEASURE(x)
```

## SYNTAX-3 — Explicit Semantic Operations

Operations with non-obvious or probabilistic semantics MUST use explicit
keywords.

Lana MUST NOT overload familiar operators such as `+`, `|`, or `*` to silently
represent operations such as APPEND, TRANSFORM, or MEASURE.

Example:

```lana
append a with b as independent
```

not:

```lana
a + b
```

## SYNTAX-4 — Predictable Grammar

Related operations SHOULD follow a consistent grammatical structure.

Preferred pattern:

```text
operation → target → modifier
```

Examples:

```lana
measure x
transform x with invert
append a with b as redundant 0.8
```

A developer who learns one operation SHOULD be able to reasonably predict the
syntax of another.

## SYNTAX-5 — Keywords Over Novel Symbols

Lana SHOULD prefer descriptive keywords over language-specific symbolic syntax.

Novel operators MUST only be introduced when they provide substantial clarity
or expressive value over equivalent keyword syntax.

## SYNTAX-6 — No Unsafe Implicit Defaults

Optional syntax MUST NOT introduce mathematically or semantically meaningful
assumptions unless the default is unambiguous and safe.

If APPEND relationship semantics cannot be safely inferred, the relationship
mode MUST be explicitly supplied.

Example:

```lana
append a with b as independent
```

An ambiguous:

```lana
append a with b
```

MUST produce a compile-time error rather than silently assuming a relationship.

## SYNTAX-7 — Invalid States Should Be Grammatically Difficult

Finite semantic choices SHOULD be represented directly by grammar or typed
constructs rather than arbitrary strings.

Example:

```lana
append a with b as synergistic 0.4
```

The parser SHOULD reject unknown relationship modes before runtime.

## SYNTAX-8 — Minimal Structural Punctuation

Lana SHOULD minimize punctuation required purely for syntax.

A construct MUST NOT require multiple redundant structural markers when one is
sufficient.

Avoid unnecessary combinations such as parentheses, colons, and semicolons for
a single construct.

## SYNTAX-9 — Visually Explicit Type Transitions

When an operation changes the semantic type of a value, syntax SHOULD make the
transition easy to recognize.

Example:

```lana
state_dist possibilities =
    append a with b as independent

state scenario =
    sample possibilities

bit result =
    measure scenario
```

Important transitions such as:

```text
STATE → STATE_DIST → STATE → BIT
```

SHOULD remain obvious when reading source code.

## SYNTAX-10 — Compiler-Assisted Recovery

Syntax errors MUST explain how to correct the statement whenever the intended
construct can reasonably be identified.

Errors SHOULD include:

- what is missing or invalid,
- the relevant source location,
- a valid example,
- available alternatives when finite.

Example:

```text
APPEND requires a relationship mode.

    append a with b


Try:
    append a with b as independent

Available modes:
    independent
    redundant <strength>
    complementary
    synergistic <strength>
```

## SYNTAX-11 — Guessability

Common Lana syntax SHOULD be approximately guessable by a developer who
understands the operation but has forgotten its exact grammar.

The language SHOULD optimize for semantic obviousness over terseness.

## SYNTAX-12 — Readability Over Character Count

Lana MUST NOT prefer shorter syntax when the shorter form materially reduces
semantic clarity.

A few additional keywords are acceptable when they make behavior explicit.

## Acceptance Principle

A syntax feature SHOULD pass all five checks:

1. **Scan** — meaning is obvious when visually scanning the statement.
2. **Guess** — a developer can approximately reconstruct the syntax from memory.
3. **Recover** — an incorrect guess produces a useful corrective error.
4. **Safe** — omitted syntax cannot silently change important semantics.
5. **Consistent** — the syntax follows patterns already established elsewhere
   in Lana.
