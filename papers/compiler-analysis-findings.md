# Compiler analysis layer — semantic findings

Status: working notes from the self-hosted compiler's analysis layer (CFG →
SSA → dataflow → optimization). These record what the compiler's own stress
test of Lana's information semantics established, and what we chose to build
on top of it. They are findings + design notes, not a spec change.

## 1. `Paths<T>` does not preserve cross-variable correlation

Spec §11 asks whether `Paths<T>` — Lana's guarded-alternatives information
form — can carry correlated facts of the shape

```
(c → x=10, y=1)  /  (!c → x=20, y=2)
```

The answer is **no**. The runtime builds a `PathSet` **per register**, not per
execution path:

- `PathAlternative` is `{ guard: bool, weight: f64, result: Value }`
  (`vm/rust/lana-vm/src/value.rs:263-268`). The `guard` is a bare boolean — it
  records *which branch* this alternative came from, not a shared row identity.
- `path_join` walks every register independently and, wherever the true/false
  values differ, constructs an independent two-alternative `PathSet`
  (`vm/rust/lana-vm/src/vm.rs:7870-7910`). `x` and `y` therefore become two
  unrelated marginals `{true→10, false→20}` and `{true→1, false→2}`.

There is no object that ties "`x=10` and `y=1` share one execution path" to "`x=20`
and `y=2` share the other". Consequently a consumer cannot soundly derive the
correlated fact `x/y = 10` (the two marginals each admit four pairwise
combinations, not two rows). `Paths` preserves *per-value* alternatives, not
*per-path* correlations.

The correlation-preserving primitive is **`Joint`**: a finite law is a vector of
whole rows, `JointRow { values: Vec<Value>, weight }`
(`vm/rust/lana-vm/src/value.rs:120-125`, `131-138`). A joint carries every
coordinate of a path in one row, so `(x=10,y=1)` and `(x=20,y=2)` are two rows
that cannot be cross-multiplied apart.

## 2. Consequence for the optimizer

Guarded-branch constant propagation that needs correlation (the `z = x/y = 10`
example) must therefore **not** be built on `Paths`. We model it as a
compiler-internal "correlated rows" fact — a small set of whole rows, each row a
map from SSA name to abstract value — which is structurally the same thing
`Joint` is. This is internal to the compiler; the VM is not touched.

Two independent concerns stay separate:

- **SSA** answers "which definition is this use?" (dominance, phi placement).
- **Correlated facts** answer "what do we know about these values, together?",
  using whole-row facts that `Joint` already expresses.

## 3. Proposed smallest extension (recorded, not implemented)

If `Paths` is ever asked to carry correlation natively, the smallest change is
one of:

1. Let a path alternative carry a **whole row** (a `Joint` over named
   variables) instead of a single `Value` — i.e. promote `PathAlternative.result`
   from `Value` to a named row.
2. Add a first-class **guarded joint**: a `Joint` indexed by a guard condition,
   so the two rows above are literally `Joint` rows annotated with `c` / `!c`.

Either keeps the "path set is execution structure, not a joint law" distinction
in place (LIP-030 §Rationale) while giving the optimizer the one thing it is
missing: a shared row identity across variables.

## 4. Other lattice observations

- `Definite` cleanly separates "a known constant" from "one SSA definition".
  That is the distinction the optimizer's must-analysis hangs on: `Definite`
  is a value fact, SSA is a name fact.
- `unknown` / `unreachable` have **no** first-class type in the source ADT.
  They are compiler-internal lattice states (the dataflow engine's top and an
  explicit reachability flag), never the ADT `unknown`. Conflating them with the
  ADT would let a "no information" fact masquerade as "one of the ADT variants",
  which is exactly what a must-analysis must reject.
