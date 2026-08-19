# Lana Runtime Semantics

Numbers, booleans, strings, arrays, functions, and null remain ordinary values.
Alongside them, Lana has one additional native primitive:

```text
STATE = (p, d)
```

`p` is the current observable probability/state value and must satisfy
`0 <= p <= 1`. `d` is directional dependency: its sign selects aligned or
inverse influence, and its magnitude selects influence strength. It must satisfy
`-1 < d < 1`. Positive `d` pulls a target toward `p`; negative `d` pulls it
toward `1 - p`; zero exerts no influence.
The primitive stores no complementary probability. When a binary distribution
is requested, the VM derives `P(1) = p` and `P(0) = 1 - p`.

For `apply A -> B`, first select A's influence target and strength:

```text
q_A = A.p       when A.d >= 0
q_A = 1 - A.p   when A.d < 0
s_A = abs(A.d)

B.p' = clip(B.p + s_A * (q_A - B.p), 0, 1)
```

Therefore `A = (0.9, -0.8)` pulls B toward `0.1` with strength `0.8`.
Negative `d` never means unbounded extrapolation away from A's probability.

It preserves B's `d` and indexes. States have value semantics: moving,
assigning, or passing a state copies it. Only the named target of a mutating
operation changes.

Measurements are registered runtime operations. `probability` returns `p` and
is the default observation. `distribution` derives the binary pair. `sample`
emits a seeded PCG32 draw without mutation. `collapse` emits a draw and replaces
the state with `(sample, 0)`.

Indexes (`timestamp`, `source`, `weight`, `confidence`) are metadata around the
primitive. History belongs to a register slot, is disabled by default, and may
retain the latest N versions or versions within a timestamp duration.

Merge averages two states. Update applies each original state to the other
simultaneously. Joint creates a pair container and assigns it no additional
probability mathematics.
