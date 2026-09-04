# Lana 2.0 — Surprisal and uncertainty-aware ML

> Follow-up spec for deferred item 5. Sits below `papers/semantics-2.md` §9.10
> and above `spec/SPEC.md` / `spec/BYTECODE.md` / `spec/VM.md` in the authority
> order. All new syntax satisfies `spec/SYNTAX.md` (SYNTAX-1..12 + Acceptance
> Principle).

## Scope

Three features, in implementation order:

1. `surprisal(P)` — natural-log surprisal in nats.
2. Uncertainty-aware ML — every ML operation carries its uncertainty.
3. Surprisal action policy — report by default, act only on user prompt.

## 1. Surprisal

### 1.1 Source syntax

```
let s = surprisal(0.25);   // -ln(0.25) ≈ 1.386 nats
```

### 1.2 Semantics

```math
\operatorname{surprisal}(P) = -\ln(P).
```

`surprisal(0)` is `+∞` and is reported as such, never as a finite substitute.
The event, outcome, model identity, model version, precision, zero-probability
behavior, and provenance are specified with the value (carried in the value's
derivation, per `spec/EVIDENCE.md`).

### 1.3 Opcode encoding

No new opcodes. `surprisal` is a host call.

## 2. Uncertainty-aware ML

Every ML operation returns its prediction together with its uncertainty; no ML
operation returns a bare prediction. The uncertainty is carried as part of the
result and is never discarded or coerced into an ordinary value. This is a
type discipline, not a new primitive: an ML operation's result is a structured
value (a map or ADT) whose fields include the prediction and its uncertainty.

## 3. Surprisal action policy

Surprisal is reported by default. Routing, flagging, or acting on a surprisal
signal occurs only on an explicit user prompt; the prompt is the policy that
authorizes the action. Surprisal never authorizes an effect on its own.

## 4. Errors

| Condition | Error |
|---|---|
| `surprisal` on a non-number | `LANA_ERR_TYPE` |
| `surprisal` on a negative probability | `LANA_ERR_INVALID_PARAMETERS` |

## 5. Deferred

- A concrete ML operation surface (the item specifies the discipline, not the
  operations).
- Surprisal routing/flagging/acting beyond the report-by-default policy.
