# Lana 2.0 — Lazy bounded datasets, declared independence and correlation

> Follow-up spec for deferred item 3. Sits below `papers/semantics-2.md` §9.8
> and above `spec/SPEC.md` / `spec/BYTECODE.md` / `spec/VM.md` in the authority
> order. All new syntax satisfies `spec/SYNTAX.md` (SYNTAX-1..12 + Acceptance
> Principle).

## Scope

Three features, in implementation order:

1. Lazy bounded datasets (a generator, infinite allowed, bounded materialization).
2. Declared independence (product measure).
3. Declared correlation (a `d` coefficient, sugar over the finite joint law).

## 1. Lazy datasets

### 1.1 Source syntax

```
let ds = lazy(fn, 1000);   // generator fn: ℕ → Value, materialization bound 1000
let x  = force(ds, 3);     // materialize element 3
let xs = force_all(ds);    // materialize 0..k up to the bound
```

- `lazy(fn, bound)` constructs a lazy dataset from a generator function and a
  materialization bound.
- `force(ds, i)` materializes element `i` on demand.
- `force_all(ds)` materializes elements `0..k` until the bound is reached.

### 1.2 Runtime representation

- New `ValueType` tag `VAL_LAZY`.
- Payload: `{ uint32_t function; size_t bound; }`.
- `function` is the generator (a `VAL_FUNCTION` index); `bound` is the
  materialization bound.

### 1.3 Bounded materialization

"Bounded" bounds materialization, not cardinality. An infinite lazy dataset is
valid; it occupies constant space until a value is forced. `force` and
`force_all` count forced values against `bound`; exceeding the bound is an
error and exposes no partial result.

## 2. Declared independence

`independent(x, y)` denotes the product measure `μ_x ⊗ μ_y`. It is the existing
`independent` joint kind (`LanaJointKind::INDEPENDENT` /
`LanaJointKind::Independent`). The declaration is an authoritative assertion by
the declarer; the system does not verify it and never infers independence from
array shape, naming, similarity, or correlation. If the assertion is false, the
declarer is responsible for the resulting error.

## 3. Declared correlation

### 3.1 Source syntax

```
correlated(x, y, d)              // binary coefficient form (new)
correlated(x, y) with support : J // finite-law form (existing)
```

- `correlated(x, y, d)` is binary sugar: `x` and `y` are `STATE` values, `d` is
  the correlation coefficient.
- The parser disambiguates by arity: three arguments is the coefficient form;
  two names followed by `with support` is the finite-law form.

### 3.2 Lowering

`correlated(x, y, d)` lowers to the existing finite joint law over the 2×2
space, with off-diagonal

```math
c = d \sqrt{p(1-p)},
\qquad |d| \le 1,
```

reusing the normalized off-diagonal of `STATE` (`semantics-2.md` §1.5). `d = 0`
is independence; `|d| = 1` is full correlation. The finite-law form
`correlated(x, y) with support : J` remains authoritative for the general
(N > 1) case.

## 4. Opcode encoding

No new opcodes. `lazy`, `force`, and `force_all` are host calls;
`correlated(x, y, d)` is compiler sugar over the existing joint construction.
`force` invokes the generator through the VM call path; if the host-call
boundary cannot invoke a Lana function, a dedicated `FORCE` opcode is the
fallback (implementation note, not a contract change).

## 5. Errors

| Condition | Error |
|---|---|
| `force`/`force_all` exceeds the materialization bound | `LANA_ERR_LIMIT` |
| `force` on a non-lazy value | `LANA_ERR_TYPE` |
| `correlated(x, y, d)` with `|d| > 1` | `LANA_ERR_INVALID_PARAMETERS` |
| `correlated(x, y, d)` with non-`STATE` operands | `LANA_ERR_TYPE` |

## 6. Deferred

- Non-binary correlation (N > 1 coefficient form).
- Automatic violation detection for a false independence/correlation
  declaration (the declarer is responsible; detection is a future feature).
- Lazy datasets over non-`ℕ` index domains.
