# Five Information Forms for Explicit Uncertainty

**Author:** Vijay Sam Neelakantan

---

## Abstract

This paper introduces an explicit system to trace the origins and nature of quantifiable uncertainty in five distinct, formally defined forms of information — `Definite`, `Possibility`, `Distribution`, `Joint`, and `Paths`. Every operation within this system either preserves or strictly decreases information content relative to its full input; an operation will inherently never create or invent new information.

The one operation that constructs a probability law, `assign_weights`, does so only by consuming explicit caller-supplied weights.

The results are established for finite domains and finite path sets, and are illustrated by recorded executions of an
implementation (Lana 4.1.0).

---

## 1. Introduction

### 1.1 Motivation: Possibility does not imply probability

Computer programs that represent uncertainty treat possible values as though they had assigned probabilities implicitly; a program is capable of knowing from a candidate integer set of {2, 3} but is unable to inherently know of a likelihood for either candidate. Equal probability is an *additional assumption*, not a consequence of the candidate set. Under that assumption the
probability that the value exceeds 2 is one half; without it, the probability
is unspecified.

This motivates the research question: **which distinctions must a programming
model preserve so that possibility is never silently treated as probability?**

### 1.2 Contribution

This paper defines five information forms and proves two things:

1. The five forms are ordered by *information content* in a strict partial order.

2. Every operation in the model either preserves that order or strictly decreases it (that is, no operation silently *invents* information).

The argument is restricted to finite domains; runtime floating-point behavior
and continuous domains are outside the stated scope.

---

## 2. Formal definitions

Throughout, let $X$ be a finite, nonempty set (the domain of possible values).

### 2.1 The five forms

Define the set of information values over $X$ as the disjoint union

$$
\operatorname{Information}(X)
=
\operatorname{Definite}(X)
\;\sqcup\;
\operatorname{Possibility}(X)
\;\sqcup\;
\operatorname{Distribution}(X)
\;\sqcup\;
\operatorname{Joint}(X)
\;\sqcup\;
\operatorname{Paths}(X).
$$

Each alternative is defined precisely below.

**Definition 1 (Definite).**
$\operatorname{Definite}(x)$ for a single value $x \in X$. The form carries
exactly one known value.

**Definition 2 (Possibility).**
$\operatorname{Possibility}(S)$ for a finite, nonempty set $S \subseteq X$ of
distinct candidates. The form supplies no probabilities.

**Definition 3 (Distribution).**
$\operatorname{Distribution}(\mu)$ for a probability law $\mu$ over $X$ with
finite support: a sequence $(x_i, w_i)_{i=1}^{m}$ of distinct values with
$w_i > 0$ and $\sum_i w_i = 1$.

**Definition 4 (Joint).**
$\operatorname{Joint}(J)$ for one probability law over assignments to a finite
nonempty set of distinct variable names, with assignments respecting their
domains and positive row weights summing to one. Formally
$J = \{(a_i, w_i)\}_{i=1}^m$ with $a_i \in \prod_{n \in N} X_n$, $w_i > 0$,
$\sum_i w_i = 1$.

**Definition 5 (Paths).**
$\operatorname{Paths}(P)$ for guarded execution alternatives. Each path is the
tuple
$(\text{condition}, \text{environment delta}, \text{result information},
\text{pending effects}, \text{optional weight})$.

> **Validity rules for Paths.** A path set is *valid* when the following two
> conditions hold.
>
> 1. **Compatibility.** All alternatives agree on every variable not reassigned
>    by the branch: two paths may differ only on the variables that the branch
>    condition itself distinguishes. Formally, the environments of any two
>    alternatives agree on every name that is not the branch variable (or a name
>    assigned within the guarded alternatives).
> 2. **Weight source.** A path's optional weight is valid only if it originates
>    from an explicitly weighted input (a weighted guard). An unweighted guard
>    yields paths with *no* weight; sampling such a path set is rejected rather
>    than silently supplied a law (cf. §5.4).

### 2.2 The five forms are distinct

The five forms are *different kinds* of information, not special cases of one
another. In particular, a definite value is a **fact** ("the value is exactly
$x$"), whereas a distribution is a **belief** ("a probability law over $X$").
A fact and a belief are not the same kind of information; a fact carries no
uncertainty at all, while a distribution is *defined* by how it encodes
uncertainty. We therefore never identify $\operatorname{Definite}(x)$ with the
degenerate law $\delta_x$: the latter is a distribution one may *construct*
from a definite value, not the same object.

The following *conversions* exist, and each is a genuine operation that
changes the form (never an identity):

- **fact-to-belief.** A definite value may be converted to a distribution via
  the Dirac mass: $\operatorname{Definite}(x) \mapsto
  \operatorname{Distribution}(\delta_x)$, where $\delta_x$ puts all mass on
  $x$. This is a deliberate conversion; it does not assert that the two forms
  were equal.
- **belief-to-candidates.** A distribution may be converted to a possibility
  by forgetting its weights: $\operatorname{Distribution}(\mu) \mapsto
  \operatorname{Possibility}(\operatorname{supp}(\mu))$.
- **candidates-to-belief** requires an explicit supplied law; there is no
  canonical conversion, and none is performed implicitly.

### 2.3 Information-content order

**Definition 6 (information-content order).** For two forms $A, B$, write
$A \succ B$ ("$A$ carries strictly more information than $B$") when a value of
form $A$ determines, by a total conversion, a unique value of form $B$, but a
value of form $B$ does not determine a value of form $A$ without additional
explicit input.

Concretely, the conversions of §2.2 give the strict chain

$$
\operatorname{Distribution} \;\succ\; \operatorname{Possibility}.
$$

The form $\operatorname{Definite}$ is **incomparable** with
$\operatorname{Distribution}$ and $\operatorname{Possibility}$: it is the form
with no uncertainty at all, and it is not a special case of either. A fact
carries no belief and no candidate set, and a belief or candidate set is not a
fact. Thus $\operatorname{Definite}$ sits off the
$\operatorname{Distribution} \succ \operatorname{Possibility}$ ladder, related
to the others only by the explicit conversions of §2.2 (fact-to-belief) and by
the partial extraction operation `resolve` (belief/possibility-to-fact on a
singleton).

For the remaining forms, the order is as follows:

- **Joint over Distribution.** A joint law over a variable set $N$ determines
  every marginal distribution by projection, but distinct joints can share the
  same marginals (Lemma 4). Hence $\operatorname{Joint} \succ
  \operatorname{Distribution}$ when compared on a common variable, and
  projection is the conversion that realizes it.
- **Paths.** A path set determines, for each path, a result together with its
  condition; it is not a probability law unless an optional weight is present.
  A weighted path set determines a distribution over results by summing the
  supplied weights; an unweighted path set determines only a possibility over
  results. Paths therefore sit above $\operatorname{Possibility}$, and a
  weighted path set sits above $\operatorname{Distribution}$, but only through
  the weights that were *explicitly* carried into the path — never through
  weights fabricated during execution.

---

## 3. Operations

Each operation has an exact domain, codomain, rule, and failure condition. The
table states the intended mathematical behavior within the finite scope.

| Operation | Type / domain | Codomain | Rule |
| --- | --- | --- | --- |
| `pure` | $\operatorname{Information}(X)$ | $\operatorname{Information}(Y)$ | apply a deterministic function, combine equal outputs |
| `condition` | $\operatorname{Information}(X) \times E$ | $\operatorname{Information}(X)$ | retain evidence-matching values, renormalize |
| `observe` | $\operatorname{Information}(X) \times E$ | $\operatorname{Information}(X)$ | pure conditioning on evidence already present in the input; retain matching values, renormalize |
| `project` | $\operatorname{Joint}(X)$ | $\operatorname{Joint}(X_M)$ | marginalize onto retained names |
| `sample` | $\operatorname{Distribution}(X) \mid \operatorname{Joint} \mid \operatorname{Paths}$ | $\operatorname{Definite}(X)$ | draw one outcome |
| `resolve` | $\operatorname{Information}(X)$ | $\operatorname{Definite}(X)$ | succeed only if already definite/singleton |
| `forget_weights` | $\operatorname{Distribution}(X)$ | $\operatorname{Possibility}(X)$ | return support, drop weights |
| `assign_weights` | $\operatorname{Possibility}(X) \times \text{weights}$ | $\operatorname{Distribution}(X)$ | require explicit positive weights summing to one |

**Failure conditions.** Each operation fails deterministically, without a
partial result, when its precondition is unmet:

- **`pure`.** The deterministic function must be total on the input; an
  undefined application fails rather than producing a partial value.
- **`condition`.** Impossible evidence (retained mass zero) fails with an
  invalid-conditioning error. `Paths` reject conditioning until a path-evidence
  model is defined.
- **`observe`.** As pure conditioning, the same invalid-evidence failure
  applies. `Paths` reject observation.
- **`project`.** An unknown or duplicated requested name, or an unsupported
  exact marginalization, fails.
- **`sample`.** Sampling an unweighted possibility or an unweighted path set
  fails with an unsupported-operation error (recorded in §5.4); a weighted law
  is required. A joint samples one weighted row, never per-variable.
- **`resolve`.** A non-singleton input fails with an unresolved-value error; it
  never selects an arbitrary candidate.
- **`forget_weights`.** Defined on distributions; the result records that the
  law was discarded.
- **`assign_weights`.** Requires exactly one caller-supplied positive weight
  per candidate, no extra candidate, and a total within tolerance of one;
  otherwise it fails.

**Probability vs. bookkeeping.** The optional weight on a `Path` is a
probability law only when it originates from an explicitly weighted input; the
internal weights the runtime may use for branch bookkeeping are *not* exposed
as a probability law (see §6).

---

## 4. Main results

### 4.1 The information-content order is a strict partial order

**Theorem 1 (order).**
The relation $\succ$ of Definition 6, closed under transitivity, is a strict
partial order on the five forms; in particular it is irreflexive, transitive,
and antisymmetric. No form is equivalent to a strictly higher form.

*Proof.* The relation $\succ$ is defined by the conversions of §2.2 together
with projection and weight-summing. We close it under transitivity and verify
the three partial-order axioms.

*Irreflexivity.* We must show no form $A$ satisfies $A \succ A$. Each defining
conversion changes the form: fact-to-belief maps $\operatorname{Definite}$ to
$\operatorname{Distribution}$, belief-to-candidates maps
$\operatorname{Distribution}$ to $\operatorname{Possibility}$, projection maps
a joint to a marginal over a strict subset of names, and weight-summing maps a
weighted path set to a distribution. No defining conversion is an endomap of
its own form, and no composition of strictly form-changing conversions is an
identity. Hence $A \succ A$ never holds.

*Transitivity.* By construction: $\succ$ is *defined* as the transitive closure
of the generating conversions. Thus if $A \succ B$ and $B \succ C$ then
$A \succ C$ holds by definition.

*Antisymmetry.* We must show that $A \succ B$ and $B \succ A$ cannot both hold
for distinct forms. Suppose both held. Then a value of form $A$ determines a
unique value of form $B$, and a value of form $B$ determines a unique value of
form $A$; composing, every value of form $A$ determines a unique value of form
$A$ through a non-identity conversion. But each conversion is lossy or
one-way:

- belief-to-candidates is not injective: distinct laws may share a support
  (Lemma 2), so it cannot be part of a two-way cycle with
  candidates-to-belief, which is not even total (it requires explicit input);
- fact-to-belief is injective but its only would-be inverse is
  $\operatorname{resolve}$, which is partial (defined only on singletons) and
  therefore not a total conversion back to $\operatorname{Definite}$;
- projection is not injective: distinct joints share marginals (Lemma 4).

Hence no two distinct forms are mutually convertible by total conversions, so
antisymmetry holds. $\Box$

### 4.2 Operations preserve or decrease information content

**Theorem 2 (monotonicity).**
For every operation $f$ in the table of §3, and every input $I$ in its
domain, the output satisfies $f(I) \preceq I$, where $\preceq$ is the reflexive
closure of $\succ$. Equivalently, no operation strictly increases information
content; each operation either preserves the form or converts downward.

*Proof.* By cases over the operation table. For each operation we show the
output is determined by a subset of the input, so it cannot carry strictly more
information than the input.

**`pure`.** Applying a deterministic function to a value preserves its form
(the function is applied to the carried value or to each guarded result). When
distinct inputs map to the same output, the operation *combines* them, which is
a quotient: the output is a function of the input, so $f(I) \preceq I$. A
deterministic function never fabricates a weight that was not already present,
nor a candidate that was not already present.

**`condition`.** Retaining evidence-matching values discards the non-matching
values; renormalizing rescales the *already-present* weights by their total.
The output support is a subset of the input support, and every output weight
is a function of input weights. Hence $f(I) \preceq I$.

**`observe`.** Observation is pure conditioning on evidence already present in
the input: it retains the evidence-matching values and renormalizes their
*already-present* weights by the retained total. No new information enters
from outside the model, so the output is a function of the input alone, and
$f(I) \preceq I$. (The model deliberately provides no operation that imports
external evidence into the value's form; external observation is outside the
scope of this paper.)

**`project`.** Marginalization onto a strict subset of variable names sums the
existing weights over the discarded names. The output law is a function of the
input law, and the discarded relationship is lost. Hence $\operatorname{Joint}
\succ \operatorname{Joint}(X_M)$ whenever $M \subsetneq N$ (a strict decrease),
and equals the input when $M = N$.

**`sample`.** A sample returns one definite value drawn from the input law.
The sampled value is a single point, hence a $\operatorname{Definite}$; the
input law is not carried forward, and no probability is fabricated. Since
$\operatorname{Definite}$ is incomparable with $\operatorname{Distribution}$
(§2.3), sampling is a lossy form *change* rather than a move along the
$\operatorname{Distribution} \succ \operatorname{Possibility}$ ladder: from one
sample one cannot reconstruct the source law (Lemma 2-style non-injectivity).
Thus sampling never increases information content, and $f(I) \preceq I$ holds.

**`resolve`.** Resolution succeeds only when the input is already definite or a
singleton (Lemma 1). It extracts the unique value; it never selects among
unresolved alternatives. Thus $f(I) \preceq I$, and for a non-singleton input
the operation fails rather than fabricating a choice.

**`forget_weights`.** This is precisely the belief-to-candidates conversion of
§2.2: the output support is a function of the input, and the law is lost
(Lemma 2). Hence $f(I) \preceq I$ strictly.

**`assign_weights`.** This is the candidates-to-belief conversion. It *does*
produce a form that is strictly higher in the chain
($\operatorname{Distribution} \succ \operatorname{Possibility}$), but it does
so only by consuming *explicit caller-supplied weights*, which are additional
input, not something derived from the possibility alone. The theorem's
statement is about operations as functions of their full input tuple: the
output is a function of $(\text{possibility}, \text{weights})$, so it does not
increase information relative to that full input. This is the one operation
whose codomain is strictly higher than its *first* argument, and it is exactly
the operation by which a probability law enters the system — a fact the model
makes explicit rather than implicit. $\Box$

**Remark.** Theorem 2 states the monotonicity property *relative to the full
input*. The operation `assign_weights` is the single upward conversion, and it
is upward only because its input already includes the weights. The branching
behavior noted in §6 (equal weights assigned to unweighted Boolean candidates
during `path_split`) is a runtime behavior *outside* the operation table of §3,
and is therefore not covered by this theorem; it is addressed in the
limitations.

### 4.3 Supporting lemmas

**Lemma 1 (resolution respects uniqueness).**
If $\operatorname{resolve}(I)$ succeeds, then $I$ is already definite or a
singleton, so the result is unique.

*Proof.* By the definition of `resolve` in §3: it succeeds only when the input
is $\operatorname{Definite}(x)$ or a representation whose support is a single
value $x$. In either case there is exactly one value to extract, so the result
is uniquely determined. For any input with two or more unresolved values,
`resolve` fails rather than choosing. Hence success implies uniqueness. $\Box$

**Lemma 2 (weight removal preserves support, loses law).**
$\operatorname{forget_weights}(\operatorname{Distribution}(\mu)) =
\operatorname{Possibility}(\operatorname{supp}(\mu))$, and in general $\mu$
cannot be recovered from $\operatorname{supp}(\mu)$.

*Proof.* The first claim is immediate from the definition of
`forget_weights`: it returns exactly the positive-mass support as an unweighted
possibility.

For non-recoverability, exhibit two distinct laws on $X=\{2,3\}$ with the same
support $\{2,3\}$: let $\mu(2)=\mu(3)=\tfrac12$, and let
$\nu(2)=\tfrac13,\ \nu(3)=\tfrac23$. Both have
$\operatorname{supp}=\{2,3\}$, yet $\mu \neq \nu$. Since the support is
identical for infinitely many distinct laws, the support alone cannot determine
the law. $\Box$

**Lemma 3 (projection preserves mass).**
Projection maps a normalized joint law to a normalized marginal law.

*Proof.* Let $J=\{(a_i,w_i)\}_{i=1}^m$ be a joint law with $\sum_i w_i = 1$.
Projection onto a set $M \subseteq N$ maps each assignment $a_i$ to its
restriction $a_i|_M$ and sums the weights of equal projections. The total mass
of the projected law is the sum over projected assignments of the summed
weights, which equals the sum over *all* original rows, i.e. $\sum_i w_i = 1$.
Hence the projection is normalized. $\Box$

**Lemma 4 (marginals do not determine the joint).**
Two distinct joint laws can share the same marginals.

*Proof.* Let $X=Y=\{0,1\}$ and define two joint laws on $(X,Y)$:

$$
J_1(0,0)=\tfrac12,\ J_1(1,1)=\tfrac12,\ J_1(0,1)=J_1(1,0)=0,
$$

$$
J_2(0,1)=\tfrac12,\ J_2(1,0)=\tfrac12,\ J_2(0,0)=J_2(1,1)=0.
$$

Both give each of $X$ and $Y$ the marginal law $\tfrac12$ on each value. Yet
$J_1$ has $P(X=Y)=1$ while $J_2$ has $P(X=Y)=0$, so $J_1 \neq J_2$. Hence the
marginals do not determine the joint law. $\Box$

---

## 5. Worked examples

All recorded runs use **Lana 4.1.0 (LABC v2, Rust VM, native compiler)**, obtained
by `target/lana/bin/lana version`.

### 5.1 One value, different information

Let $X=\{2,3\}$. The four information forms distinguish four different things
one might know about a value $x$:

| Input information | What follows about the event $x>2$ |
| --- | --- |
| $\operatorname{Definite}(2)$ | The event is false, certainly. |
| $\operatorname{Possibility}(\{2,3\})$ | The event may be false or true; its probability is unspecified. |
| $\operatorname{Distribution}$ with $\mu(2)=0.7$, $\mu(3)=0.3$ | The event has probability $0.3$. |
| The same distribution after conditioning on $x=3$ | The support is $\{3\}$ with probability $1$; resolution returns $3$. |

The first and second rows differ precisely because a candidate set supplies no
probability, while a definite value supplies certainty. The model does not
derive the third row from the second without an explicit law.

### 5.2 Same marginals, different relationships

Each of two variables $x,y$ has probability $\tfrac12$ of $2$ and $\tfrac12$ of
$3$. These are the marginals. Two joints share them but differ in whether the
variables agree:

| Joint | Supported assignments and probabilities | $P(x=y)$ |
| --- | --- | --- |
| Equal values | $(2,2): 0.5$, $(3,3): 0.5$ | $1$ |
| Opposite values | $(2,3): 0.5$, $(3,2): 0.5$ | $0$ |

This is Lemma 4 in concrete form: the marginals do not determine the joint.

### 5.3 Guarded execution

```text
x belongs to {2, 3}, with no probabilities supplied
if x equals 2: result = 10
else:          result = 20
```

The resulting alternatives retain the conditions $x=2$ and $x=3$ and produce
results $10$ and $20$, but supply no probability for either result. The
conditions are preserved, not collapsed into a single outcome. When the same
computation is run with an *explicitly weighted* guard, the weights are carried
through; with an *unweighted* guard, sampling the resulting path set is
rejected (see §5.4, rows 3–4).

### 5.4 Implementation evidence (recorded runs)

| Fixture | Behavior under test | Recorded result |
| --- | --- | --- |
| `tests/regression/core_weight_conversion.lana` | `forget_weights` then `assign_weights` round-trip support, record weight loss | `CORE_WEIGHT_CONVERSION_PASS` |
| `tests/regression/core_possibility_sample_rejected.lana` | sampling an unweighted possibility is rejected | `error[LANA_ERR_UNSUPPORTED_OPERATION]: possibility has no weights; use distribution(...) before sample (instruction 4, opcode INFO_SAMPLE)` |
| `tests/regression/core_paths_unweighted.lana` | sampling an unweighted guarded result is rejected | `error[unsupported/LANA_ERR_UNSUPPORTED_OPERATION] ... (operation INFO_SAMPLE) (instruction 14, opcode INFO_SAMPLE)` |
| `tests/regression/core_paths_weighted.lana` | a weighted guard samples successfully | `CORE_PATHS_WEIGHTED_PASS` |
| `tests/regression/core_paths_unweighted_transformed.lana` | a pure function over an unweighted guarded result still rejects sampling | `error[unsupported/LANA_ERR_UNSUPPORTED_OPERATION] ... (operation INFO_SAMPLE) (instruction 16, opcode INFO_SAMPLE)` |

The unweighted-path rejection (rows 3 and 5) is direct evidence for the model's
central distinction: an unweighted guard produces paths that carry conditions
and results but no probability, and the runtime refuses to sample them rather
than silently supplying a law. Row 5 shows that the rejection survives a pure
transformation (`result + 1` applied to each guarded result): the operation
maps over the paths but does not fabricate the missing weights. A weighted
guard (row 4) carries its weights through and samples normally. Row 2 records
the analogous rejection for a bare `Possibility`, with an explicit error string
directing the programmer to `distribution(...)`.

---

## 6. Limitations

- **Finite scope.** All results are for finite domains and finite path sets;
  continuous domains and general inference require separate arguments.
- **Branch splitting (`path_split`).** Source inspection notes that, internally,
  the VM may assign equal weights to unweighted Boolean candidates when
  splitting a branch. This internal bookkeeping is **not** exposed as a
  probability law: the recorded runs of §5.4 show that sampling an unweighted
  guarded result is *rejected* with an unsupported-operation error, and that
  the rejection survives a pure transformation. We therefore state the
  monotonicity result (Theorem 2) over the operation table of §3, and report
  branch splitting as the boundary at which the model's internal execution
  bookkeeping and its *observable* information semantics diverge. A formal
  account of this boundary is future work.
- **Floating point.** Runtime weights are valid only after the normalization
  and tolerance checks; exact mathematical examples do not cover
  floating-point behavior.

---

## 7. Conclusion

We defined five distinct information forms — `Definite`, `Possibility`,
`Distribution`, `Joint`, and `Paths` — and showed they are related by a strict
partial order of information content (Theorem 1), in which every operation
either preserves or strictly decreases information content relative to its full
input (Theorem 2). The single upward conversion, `assign_weights`, is upward
only because its input already includes the explicit weights; no operation
derives a probability law from a bare candidate set.

The benefit of the model is that the boundary between possibility and
probability is made *explicit and enforced*: an unweighted candidate set cannot
be silently sampled as if it carried a law, as the recorded rejection in §5.4
demonstrates. The most important remaining question is the runtime branching
boundary of §6 — equal weights assigned to unweighted Boolean candidates during
branch splitting — together with the extension of the finite-domain results to
continuous domains.

---

## References

> **Status: citations not yet verified.** The four classic references
> (Plotkin, Goodman et al., Pearl, King) are well-established works but are
> listed here from the draft without confirmed bibliographic details. The two
> 2026 entries (`Gürtler & Kaminski`, `Liell-Cock & Staton`) could not be
> independently confirmed and must be replaced with real, verifiable sources or
> removed before submission.

- Plotkin (1976), *Call-by-name, call-by-value and lazy evaluation*.
- Goodman et al. (2008), *Probabilistic Programming*.
- Pearl (1986), *Probabilistic Reasoning in Intelligent Systems*.
- King (1976), *Symbolic Execution and Program Testing*.
