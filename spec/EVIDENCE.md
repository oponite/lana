# Lana 2.0 — Evidence lifecycle, causal links, provenance, deterministic replay

> Follow-up spec for deferred item 2. Sits below `papers/semantics-2.md` §9.7
> and above `spec/SPEC.md` / `spec/BYTECODE.md` / `spec/VM.md` in the authority
> order. All new syntax satisfies `spec/SYNTAX.md` (SYNTAX-1..12 + Acceptance
> Principle).

## Scope

Four features, in implementation order:

1. Evidence record `(value, status, producer, time)`.
2. Status combination (least-certain wins).
3. Causal links (declared conditional relationships).
4. Provenance paths + deterministic replay.

No new opcode and no new value type are introduced. The evidence record is a
**derived view** over the existing derivation DAG (`LanaDerivation` /
`Derivation`), which every value already carries. This spec fixes the mapping
and the combination rule; the runtime already threads the underlying records.

## 1. Evidence record

### 1.1 Status derivation

The evidence status is a derived label over the existing derivation record,
not a new field. Given a derivation `d`, the status is:

| status   | `kind`        | `exactness`   | `outcome`   |
|----------|---------------|---------------|-------------|
| observed | `OBSERVATION` | `EXACT`       | `SUCCESS`   |
| exact    | `OPERATION` or `EVIDENCE` | `EXACT` | `SUCCESS` |
| modeled  | `ASSUMPTION`  | `APPROXIMATE` | `SUCCESS`   |
| sampled  | `SAMPLE`      | `SAMPLE`      | `SUCCESS`   |
| unknown  | (any)         | (any)         | `UNRESOLVED`|

Certainty order (least to most certain):

```text
unknown < sampled < modeled < exact < observed
```

The status is a credibility label, never derived from a `STATE` probability,
similarity, naming, or correlation.

### 1.2 Producer and time

- **producer** = `operation` (the operation that produced the value), with
  `label` and `function` retained as secondary identity.
- **time** = `(task_lineage, local_sequence, revision)`.

These are already fields of `LanaDerivation` / `Derivation`; no new storage.

## 2. Status combination

When an operation combines two evidence values (e.g. `APPEND`), the result's
value follows that operation's law, but the result's status is the
least-certain input:

```text
status(e1 ⊕ e2) = min(status(e1), status(e2))
```

under the certainty order above. Concretely: `exact ⊕ sampled = sampled`,
`modeled ⊕ observed = modeled`, `exact ⊕ unknown = unknown`. The result's
producer and time are the operation's own; the input producers and times are
retained in the derivation's `inputs`, not overwritten.

The runtime applies this rule when it constructs the result derivation of a
combining operation: it computes the two input statuses and sets the result's
`kind`/`exactness`/`outcome` to the least-certain input's.

## 3. Causal links

A causal link `A → B` is a declared conditional relationship: the law of `B`
is conditioned on `A`, `B ~ P(B | A)`. It reuses the existing machinery, not a
new edge kind:

- **joint form**: `conditional(x | y)` (`LanaJointKind::CONDITIONAL` /
  `LanaJointKind::Conditional`).
- **reactive form**: `LanaRelationshipKind::EXPLICIT_JOINT` /
  `RelationshipKind::ExplicitJoint`.

A causal link is distinct from two other edges, which are **not** probabilistic:

- a **dependency edge** — "B was computed from A" — composition/execution
  structure (`LanaReactive` with `EXACT` or `SAME_DEPENDENCY` relationship);
- a **cause chain** — the sequence of operations that led to a failure
  (`LanaErrorInfo.causes` / `cause_chain_truncated`).

Only the causal link asserts that A's probability changes B's probability. A
bare dependency edge never implies a causal link, and a causal link must be
declared, never inferred from dataflow, naming, or correlation.

## 4. Provenance paths

A provenance path is a path in the derivation DAG from a root to a node. The
DAG is the existing `inputs` graph of `LanaDerivation` / `Derivation`; each
node's provenance is the set of `(root, path)` pairs that produced it.
Operations preserve provenance: an exact operation retains exact status, a
sampled operation retains sample status, and an explicit approximation retains
approximate status. Unwrapping a value does not erase its derivation; it only
makes the value available to an exact operation.

## 5. Deterministic replay

Replay is byte-identical: given the same `(seed, inputs, version)`, the output
bytes are identical. The runtime contract is:

- fixed evaluation order;
- IEEE 754 binary64 with no fused-multiply-add reordering;
- a seeded deterministic RNG (the existing PCG32 stream);
- deterministic iteration order over maps and sets.

The floating-point and randomness policies of `semantics.md` (1.0) §7.3 and
§7.4 remain authoritative; replay is their composition contract. A version
change may change bytes; a seed or input change may change bytes; nothing else
may.

## 6. Runtime representation

No new value type. The mapping is:

| Spec concept | C field | Rust field |
|---|---|---|
| status | derived from `LanaDerivation.kind` + `.exactness` + `.outcome` | `Derivation.kind` + `.exactness` + `.outcome` |
| producer | `LanaDerivation.operation` | `Derivation.operation` |
| time | `task_lineage`, `local_sequence`, `revision` | same |
| provenance path | `LanaDerivation.inputs[]` | `Derivation.inputs` |
| causal link | `LanaRelationshipKind::EXPLICIT_JOINT` | `RelationshipKind::ExplicitJoint` |

The one new runtime helper is a status classifier:

```c
LanaEvidenceStatus lana_derivation_status(const LanaDerivation *d);
```

returning one of `UNKNOWN`, `SAMPLED`, `MODELED`, `EXACT`, `OBSERVED`, and a
`lana_evidence_status_name()` for stable rendering. The Rust mirror is
`Derivation::status()` / `status_name()`.

## 7. Errors

| Condition | Error |
|---|---|
| combining evidence with an undeclared relationship | `SS_ERR_UNSUPPORTED_OPERATION` (existing) |
| causal link inferred from a bare dependency edge | not permitted; must be declared |

## 8. Deferred

- Non-binary causal links (N > 1 conditional relationships).
- Automatic violation detection for a false independence/correlation
  declaration (the declarer is responsible; detection is a future feature).
- Cross-process provenance (provenance is process-local today).
