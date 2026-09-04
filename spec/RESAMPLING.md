# Lana 2.0 — Deterministic resampling and structured statistical results

> Follow-up spec for deferred item 4. Sits below `papers/semantics-2.md` §9.9
> and above `spec/SPEC.md` / `spec/BYTECODE.md` / `spec/VM.md` in the authority
> order. All new syntax satisfies `spec/SYNTAX.md` (SYNTAX-1..12 + Acceptance
> Principle).

## Scope

One feature: deterministic bootstrap resampling that returns a structured
statistical result (point estimate + 95% confidence interval), never a bare
number.

## 1. Source syntax

```
let r = bootstrap(data, statistic);        // system-chosen B
let r = bootstrap(data, statistic, 2000);  // user-chosen B
```

- `data` is an array of observations.
- `statistic` is a function `T(x_1, ..., x_n) -> number`.
- `B` is the resample count; omitted means the system default.

## 2. Semantics

Given `n` observations and a statistic `θ̂ = T(x_1, ..., x_n)`, bootstrap draws
`n` observations with replacement for each of `B` resamples and computes
`θ*_b = T(x*_b)`. The point estimate is `θ̂`; the 95% confidence interval is the
2.5% and 97.5% percentiles of `{θ*_b}`. The draws are seeded and deterministic.

## 3. Structured result

`bootstrap` returns a map (existing `VAL_MAP`) with the keys:

| key | type | meaning |
|---|---|---|
| `estimate` | number | point estimate `θ̂` |
| `ci_low` | number | 2.5% percentile |
| `ci_high` | number | 97.5% percentile |
| `method` | string | `"sampled"` |
| `procedure` | string | `"bootstrap"` |
| `sample_count` | number | `B` |
| `seed` | number | RNG seed |

The result never collapses to a bare number; the method, procedure, count, and
seed are preserved. A dedicated `VAL_STATISTICAL` value type is a possible
future refinement; the map form is the initial contract.

## 4. Resample count

`B` has a system default chosen for stability (the system increases `B` until
the confidence interval stabilizes within a tolerance) and may be overridden by
the user. The chosen `B` is recorded in `sample_count`.

## 5. Opcode encoding

No new opcodes. `bootstrap` is a host call that invokes the statistic function
through the VM call path; if the host-call boundary cannot invoke a Lana
function, a dedicated opcode is the fallback (implementation note, not a
contract change).

## 6. Errors

| Condition | Error |
|---|---|
| `data` is not an array | `LANA_ERR_TYPE` |
| `statistic` is not a function | `LANA_ERR_TYPE` |
| `B` is not a positive integer | `LANA_ERR_INVALID_PARAMETERS` |
| resampling exhausts the instruction/memory limit | `LANA_ERR_LIMIT` |

## 7. Deferred

- Non-bootstrap resampling procedures (jackknife, permutation).
- Confidence levels other than 95%.
- A dedicated `VAL_STATISTICAL` value type.
