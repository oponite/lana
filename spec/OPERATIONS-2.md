# Lana 2.0 Operations — `attenuate`, `trace_distance`, relationship-aware `append`

> Follow-up spec for `papers/semantics-2.md` §9.2, §9.4, §9.5.
> Source syntax + opcode encoding + runtime behavior.

## Authority

This spec is the committed, implementation-ready follow-up to the mathematics
in `papers/semantics-2.md` §9. It sits below `papers/semantics-2.md` and above
`spec/SPEC.md` / `spec/BYTECODE.md` / `spec/VM.md` in the authority order. All
new syntax satisfies `spec/SYNTAX.md` (SYNTAX-1..12 + Acceptance Principle).

## Scope

Three operations, in implementation order:

1. `attenuate` — fade the off-diagonal (disposition) of a state (§9.2).
2. `trace_distance` — distance between two concrete states (§9.4).
3. relationship-aware `append` — combine two states under an explicit
   relationship (§9.5).

Deferred (require further mathematics, not in this increment):

- `STATE_DIST` inputs for all three operations (the coupling π for distributed
  inputs is unspecified).
- the `synergistic` relationship mode (requires an explicit base relationship
  and an interaction event).
- N > 1 substrates (only the N = 1 specialization is specified here).

## 1. `attenuate`

### 1.1 Mathematics (§9.2)

```
attenuate(ρ, f) = f·ρ + (1-f)·diag(ρ),   f ∈ [0,1]
```

At N = 1, for a concrete state `(p, d)`:

```
attenuate((p, d), f) = (p, f·d)
```

The observable probability `p` is invariant; only the disposition is scaled by
`f`. `f = 1` is the identity; `f = 0` collapses to the diagonal (disposition
zero). Equivalently `attenuate(ρ, f) = mix(ρ, neutralize(ρ), f)`.

### 1.2 Source syntax

```
attenuate <source> by <factor>
```

An expression returning a new `STATE` or `STATE_DIST`. `<source>` is an
expression evaluating to a `STATE` or `STATE_DIST`; `<factor>` is a runtime
expression evaluating to a number. The source is not mutated.

```lana
state belief = state(p: 0.5, d: 0.6);
let faded = attenuate belief by 0.5;   // faded is (0.5, 0.3); belief unchanged
```

### 1.3 Opcode encoding

New opcode `OP_ATTENUATE` (appended after `OP_REVISION`, before `OP_COUNT`):

```
ATTENUATE a b c
```

- `a` — destination register (`STATE` or `STATE_DIST`).
- `b` — source register (`STATE` or `STATE_DIST`).
- `c` — factor register (number).
- `imm` — unused (`LANA_NO_OPERAND`).

### 1.4 Runtime behavior

- `b` must be a `STATE` or `STATE_DIST`; otherwise `LanaError::Type`.
- `c` must be a finite number in `[0,1]`; otherwise
  `LanaError::InvalidParameters`.
- `STATE` input: write `(p, f·d)` to `a`. `p` is copied unchanged; `d_re` and
  `d_im` are each multiplied by `f`. The result is re-validated as a state
  (`LanaError::InvalidState` on failure).
- `STATE_DIST` input: lift lazily. `a` is a new `ATTENUATE` node wrapping `b`
  with factor `f`. Expected probability is invariant (identity); sampling
  applies `(p, d) → (p, f·d)` to each sampled concrete state.

### 1.5 Errors

| Condition | Error |
|---|---|
| `b` not `STATE`/`STATE_DIST` | `Type` |
| `c` not finite or outside `[0,1]` | `InvalidParameters` |
| result not a valid state | `InvalidState` |

## 2. `trace_distance`

### 2.1 Mathematics (§9.4)

```
trace_distance(ρ, σ) = ½‖ρ - σ‖₁
```

At N = 1, for concrete states `a = (p_a, d_a)`, `b = (p_b, d_b)`, with
`c_a = √(p_a(1-p_a))·d_a` and `c_b = √(p_b(1-p_b))·d_b`:

```
trace_distance(a, b) = √( (p_a - p_b)² + |c_a - c_b|² )
```

where `|c_a - c_b|² = (Re c_a - Re c_b)² + (Im c_a - Im c_b)²`. The result is a
number in `[0,1]`.

### 2.2 Source syntax

```
trace_distance <left> <right>
```

An expression returning a number. `<left>` and `<right>` are expressions
evaluating to concrete `STATE` values.

```lana
let d = trace_distance a b;
```

### 2.3 Opcode encoding

New opcode `OP_TRACE_DISTANCE`:

```
TRACE_DISTANCE a b c
```

- `a` — destination register (number).
- `b` — left state register (`STATE`).
- `c` — right state register (`STATE`).
- `imm` — unused.

### 2.4 Runtime behavior

- `b` and `c` must be concrete `STATE`; otherwise `LanaError::Type`.
- `STATE_DIST` inputs are rejected with `LanaError::UnsupportedOperation`
  (distributed trace distance requires the coupling π, deferred).
- Write the N = 1 formula above to `a`.

### 2.5 Errors

| Condition | Error |
|---|---|
| `b` or `c` not `STATE` | `Type` |
| `b` or `c` is `STATE_DIST` | `UnsupportedOperation` |

## 3. Relationship-aware `append`

### 3.1 Mathematics (§9.5)

For concrete states `a = (p_a, d_a)`, `b = (p_b, d_b)`, the overlap `q = P(A∧B)`
is determined by the relationship mode, then `p_C = p_a + p_b - q`. The
disposition is composed exactly as in the independent case: `m_C = (d_a+d_b)/2`,
`σ_C = |d_a-d_b|/2`, and the cached `Append` node stores `(p_C, m_C, σ_C)`.

Modes (`q` as a function of the independent overlap `I = p_a·p_b`):

| Mode | `q` | strength |
|---|---|---|
| `independent` | `I` | — |
| `redundant` | `(1-r)·I + r·U`, `U = min(p_a, p_b)` | `r ∈ [0,1)` |
| `full_redundancy` | `p_a` (requires `p_a = p_b`) | — |
| `complementary` | `(1-k)·I + k·L`, `L = max(0, p_a+p_b-1)` | `k ∈ [0,1]` |

`synergistic` is deferred (requires an explicit base relationship and an
interaction event).

### 3.2 Source syntax

```
append <left> with <right> as <mode> [<strength>]
```

An expression returning a `STATE_DIST`. `<mode>` is one of `independent`,
`redundant`, `full_redundancy`, `complementary`. `<strength>` is required for
`redundant` and `complementary`, forbidden for `independent` and
`full_redundancy`.

```lana
let c = append a with b as independent;
let c = append a with b as redundant 0.8;
let c = append a with b as full_redundancy;
let c = append a with b as complementary 0.3;
```

The `append(a, b)` function-call form is deprecated. It remains accepted for
Lana 1.x compatibility and is equivalent to `append a with b as independent`,
but new code must use the keyword form.

### 3.3 Opcode encoding

New opcodes (appended after `OP_TRACE_DISTANCE`):

```
APPEND_REDUNDANT a b c imm
APPEND_FULL_REDUNDANCY a b c
APPEND_COMPLEMENTARY a b c imm
```

The operand layout follows the existing `OP_APPEND` convention (`a`=left,
`b`=right, `c`=destination), with `imm` carrying the strength:

- `a` — left state register (`STATE`).
- `b` — right state register (`STATE`).
- `c` — destination register (`STATE_DIST`).
- `imm` — strength register (number) for `redundant`/`complementary`; unused for
  `full_redundancy`.

`independent` reuses the existing `OP_APPEND` (no new opcode).

### 3.4 Runtime behavior

- `b` and `c` must be concrete `STATE`; otherwise `LanaError::Type`.
- `STATE_DIST` inputs are rejected with `LanaError::UnsupportedOperation`
  (coupling π deferred).
- `redundant`: strength `r` must be finite and in `[0,1)`; otherwise
  `LanaError::InvalidParameters`.
- `complementary`: strength `k` must be finite and in `[0,1]`; otherwise
  `LanaError::InvalidParameters`.
- `full_redundancy`: requires `p_a = p_b`; otherwise
  `LanaError::InvalidParameters`.
- Compute `q` per the mode table, `p_C = p_a + p_b - q`, then compose the
  disposition as in `OP_APPEND` (reusing `append_parameters` with the
  relationship-derived `p_C`). The result is a `STATE_DIST` whose `Append` node
  caches `(p_C, m_C, σ_C)`.

### 3.5 Errors

| Condition | Error |
|---|---|
| `b` or `c` not `STATE` | `Type` |
| `b` or `c` is `STATE_DIST` | `UnsupportedOperation` |
| strength out of range / non-finite | `InvalidParameters` |
| `full_redundancy` with `p_a ≠ p_b` | `InvalidParameters` |

## 4. Opcode table (additions)

| Opcode | `a` | `b` | `c` | `imm` |
|---|---|---|---|---|
| `ATTENUATE` | dest | source | factor | — |
| `TRACE_DISTANCE` | dest | left | right | — |
| `APPEND_REDUNDANT` | left | right | dest | strength |
| `APPEND_FULL_REDUNDANCY` | left | right | dest | — |
| `APPEND_COMPLEMENTARY` | left | right | dest | strength |

## 5. Deferred

- `STATE_DIST` inputs for all three operations (coupling π).
- `synergistic` relationship mode (base relationship + interaction event).
- N > 1 substrates.
