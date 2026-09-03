# 1.0 - Lana Mathematical Semantics

## 0. Scope and Semantic Authority

This document is the normative Lana 1.0 definition of `STATE`, `STATE_DIST`, and the core state operations `MEASURE`, `TRANSFORM`, and `APPEND`. It also specifies how those values and operations compose, their mathematical guarantees and boundary behavior, and the minimum obligations of a conforming runtime.

The other Lana documents have narrower roles:

1. `VM.md` describes execution architecture.
2. `BYTECODE.md` describes LABC representation and instruction encoding.
3. `README.md` describes the user-facing language and development experience.
4. `SPEC.md` and other documents may describe implementations or language facilities built around these semantics, but they may not redefine the mathematical objects or operations defined here.

If an implementation or another document disagrees with this document, this document is authoritative and the implementation is non-conforming until the disagreement is resolved.

This document intentionally does not define general parser grammar, variable syntax except in non-normative examples, string or array implementation, standard-library internals, CLI or project initialization, bytecode binary layout, register allocation, memory management, or threading implementation. Those concerns cannot change the mathematical value semantics defined here.

## 0.1 Information model

### Immutable derivations

Provenance is an immutable derivation DAG attached to a value and is not part
of that value's mathematics. Mathematical equality ignores derivation identity.
`evidence(v, label)` and `assume(v, proposition)` create explicit roots; labels
carry no hidden inference rule. Every successful Information operation creates
a node only after its mathematical result has validated. `condition`, sampling,
and explicit approximation do not advance the revision. A successfully
committed `observe` advances the VM revision exactly once; failed observation
publishes no value and advances no revision.

Each node has the canonical fields `id = [task_lineage, local_sequence]`,
`revision`, `kind`, `operation`, ordered `inputs`, `source`, `exactness`,
operation-specific `details`, `outcome`, and `reason`. IDs are deterministic
within a seeded execution and contain no addresses, allocation order, wall
clock time, or generated prose. Task cloning preserves origin IDs and clones
the DAG with memoization. `derivation(v)` returns this record as ordinary maps
and arrays. `explain(v)` uses a fixed renderer over the same fields.

Sampling records the seed lineage and remains a read. `estimate_measure`
records explicit approximation. Failed observation and resolution may attach an
unpublished failure-node ID to the structured error, but never expose a partial
value.

Lana represents information separately from the concrete value it may describe.
For a measurable value domain $X$, define

```math
\operatorname{Information}(X)=
\operatorname{Definite}(X)
\;|\;\operatorname{Possibility}(X)
\;|\;\operatorname{Distribution}(X)
\;|\;\operatorname{Joint}(X)
\;|\;\operatorname{Paths}(X).
```

The alternatives have different meanings and must not be silently coerced:

- `Definite(x)` contains exactly one known value $x\in X$.
- `Possibility(S)` contains a valid set or relation of candidates $S\subseteq X$
  without assigning probabilities to those candidates.
- `Distribution(\mu)` contains a probability measure $\mu$ over $X$.
- `Joint` contains one law or supported lazy law over named variables in a
  product domain. It is not a collection of unrelated marginal fields.
- `Paths` contains guarded alternatives produced by uncertain execution. A
  path is not another representation of a joint law.

A quantum state is a future specialized information domain with additional
physical constraints. Lana does not interpret ordinary unresolved or
probabilistic information as quantum information.

Every information value has a domain, a validity condition, and a declared
evaluation boundary. An operation that cannot be performed exactly must return
an explicit unsupported-operation error. Sampling, finite precision, and
Monte Carlo estimation are not implicit replacements for exact semantics.

For each value type $T$, let $(X_T,\Sigma_T)$ be its measurable domain. A
definite value denotes the Dirac measure $\delta_x$ when embedded in a
distribution. A finite possibility is a nonempty finite subset of $X_T$. A
finite distribution is a sequence $(x_i,w_i)_{i=1}^m$ with distinct supported
values, $w_i>0$, and $\sum_iw_i=1$. Runtime floating-point weights are valid
only after the normalization and tolerance checks in Section 7.3; zero,
negative, non-finite, and materially non-normalized weights are invalid.

### 0.1.1 Information operations

The operations below are semantically distinct:

```math
\begin{aligned}
\operatorname{project}(I,N)&:\operatorname{Joint}(X)\to\operatorname{Joint}(X_N),\\
\operatorname{condition}(I,e)&:\operatorname{Information}(X)\to\operatorname{Information}(X),\\
\operatorname{observe}(I,e)&:\operatorname{Information}(X)\to\operatorname{Information}(X),\\
\operatorname{sample}(I)&:\bigl(\operatorname{Distribution}(X)\;|\;\operatorname{Joint}(X_N)\;|\;\operatorname{Paths}(X)\bigr)\to\operatorname{Definite}(X),\\
\operatorname{resolve}(I)&:\operatorname{Information}(X)\to\operatorname{Definite}(X).
\end{aligned}
```

`project` (or marginalize) is pure and returns a new view or law over the
requested named variables. An unknown variable, duplicate requested name, or
unsupported exact marginalization is an error. `condition` is pure refinement;
impossible evidence returns `LANA_ERR_INVALID_CONDITIONING` and does not mutate
the input. `observe` is an external information event: it records or consumes
evidence in the execution context and returns the refined information or a
definite observed result. It is effectful even when its returned value is
immutable. `sample` is stochastic and read-only; it consumes the configured
random stream and returns one definite result. `resolve` is exact and succeeds
only for already definite information or a singleton after refinement. It may
not choose an arbitrary representative; a non-singleton returns
`LANA_ERR_UNRESOLVED_VALUE`.

For a joint assignment, sampling has the corresponding product-space form:

```math
\operatorname{sample}(J):\operatorname{Joint}(X_N)
\to\operatorname{Definite}\left(\prod_{n\in N}X_n\right).
```

The returned assignment is definite for every named variable. Sampling a
joint must not mutate the joint law or any of its projections.

General equality for distributions, joints, and path sets is unsupported unless
a later canonical representation makes it decidable. The existing concrete
`STATE` and `STATE_DIST` equality rules remain unchanged.

### 0.1.2 Joint information

A joint value has a finite canonical ordered set of unique variable names, a
domain descriptor for each variable, and a joint law or explicitly supported
lazy expression over their product space. The following constructions are
distinct:

1. `independent(x, y)` denotes the product of the supplied marginal laws.
2. `correlated(x, y)` denotes a law whose relationship is part of the joint
   object.
3. `conditional(x | y)` denotes a law or relation conditioned on named
   variables.

Combining two marginals does not imply correlation. Dependency edges used by a
runtime are derived composition or execution structure; they are not, by
themselves, the mathematical joint law. Cyclic relation definitions, duplicate
names, empty joints, and unsupported exact inference are invalid or unsupported
and must fail deterministically. Joint construction, projection, and
conditioning are immutable: the source joint remains unchanged.

For the first exact correlated representation, a finite joint law is

```math
J=\{(a_i,w_i)\}_{i=1}^m,
\qquad
a_i\in\prod_{n\in N}X_n,
\qquad
w_i>0,
\qquad
\sum_iw_i=1.
```

Equal assignments are canonicalized by summing their weights. Projection to
$M\subseteq N$ maps every $a_i$ to $a_i|_M$, combines equal projected
assignments, and preserves total mass. Conditioning on evidence $E$ retains
the rows satisfying $E$ and divides their weights by
$J(E)=\sum_{a_i\models E}w_i$. If $J(E)=0$, conditioning returns
`LANA_ERR_INVALID_CONDITIONING`. Sampling performs one weighted row selection;
it never samples each variable independently.

An independent product may remain lazy. Its law is the product measure
$\bigotimes_{n\in N}\mu_n$ and therefore does not assert correlation. A
relation-backed or conditional node must declare which of project, condition,
sample, and resolve it supports. An undeclared operation returns
`SS_ERR_UNSUPPORTED_OPERATION`; it may not fall back to enumeration, sampling,
or an independence assumption.

Existing `STATE` values embed as `Definite(STATE)`. Existing `STATE_DIST`
values embed as `Distribution(STATE)`. These embeddings do not change the
observable behavior of `MEASURE`, `TRANSFORM`, `APPEND`, or
`SAMPLE_STATE_DIST`; their existing sections below remain authoritative.

More explicitly, for the embeddings $\eta_S(s)=\operatorname{Definite}(s)$
and $\eta_D(\mu)=\operatorname{Distribution}(\mu)$:

```math
\begin{aligned}
\operatorname{MEASURE}(\eta_S(s))&=\operatorname{MEASURE}(s),\\
\operatorname{TRANSFORM}(\eta_S(s),\Phi)&=\eta_S(\operatorname{TRANSFORM}(s,\Phi)),\\
\operatorname{APPEND}(\eta_S(a),\eta_S(b))&=\eta_D(\operatorname{APPEND}(a,b)),\\
\operatorname{sample}(\eta_D(\mu))&=\eta_S(\operatorname{SAMPLE\_STATE\_DIST}(\mu)).
\end{aligned}
```

These are compatibility identities, not alternate implementations. In
particular, embedding cannot add normalization, sampling, collapse, or a new
equality rule to an existing operation.

### 0.1.3 Information-aware execution

An execution path is the immutable tuple

```text
(path condition, environment delta, result information,
 pending effects, optional probability weight)
```

An unresolved Boolean used as an `if` guard creates guarded alternatives. Path
conditions remain separate from dependencies inside a joint law. Pure functions
map over each guarded path and preserve its condition and information. Joining
paths preserves conditions and results; it does not silently select or average
one alternative.

External I/O, printing, task creation, and host calls require definite inputs
unless an explicit per-path effect policy is present. The initial execution
semantics reject unresolved loop guards. Path count, memory, instruction, and
sampling limits are hard limits. Budget exhaustion, cancellation, or an
unsupported path join returns an error and exposes no partial result.

The semantics in this section define the boundary for future quantum-compatible
lowering. The current LABC and C VM implementation may support only a declared
subset; it must report unsupported operations rather than silently approximate
or collapse information.

## 0.2 Static uncertainty, effect, and failure foundation

The source type categories are disjoint tagged constructors:

```text
T
Information<T>
Claim<T, Proposition>
Sample<T>
PlannedEffect<T, Effect>
TaskHandle<T, Capabilities>
Capability<Name>
Result<T, E>
```

`Information<T>` means that a value is not yet definite. `Claim<T, P>` contains
both a value and the programmer-supplied proposition `P`; Lana never derives a
proposition from a bare `STATE` probability. `information(v)` and
`claim(v, "P")` are therefore distinct constructions. `resolve(i)`,
`claim_value(c)`, and `claim_proposition(c)` are explicit eliminations. A
wrapped value cannot be consumed where exact `T` is required.

Every stochastic source expression returns `Sample<T>`. Its runtime record is
the pair `(value, metadata)`, where metadata contains the source dependency,
root RNG seed, task lineage, operation, and revision. `sample_value(s)` is the
only ordinary-value elimination and `sample_metadata(s)` exposes the immutable
metadata record. Unwrapping does not erase the record or change provenance; it
only makes the sampled value available to an exact operation.

Effects form finite sets over `pure`, `observation`, `stochastic`, `io`,
`mutation`, `task`, and `external_call`. Pure contributes no real-world effect.
Observation and stochastic effects remain distinct. An unresolved guard may
map pure computation over alternatives, but it cannot perform I/O, mutation,
task creation, observation, or an external call. A `PlannedEffect<T, E>` is
ordinary inert data describing a future effect; constructing or storing it does
not execute `E`, and it cannot be consumed as `T`. Capabilities are explicit
typed values and do not arise from possession of data or from probability.

`Result<T, E>` has disjoint success and error alternatives. Construction and
elimination are explicit; using a whole result as `T` is a type error. Task join
and timeout produce typed result foundations rather than silently exposing a
partial task result.

Failures have a stable code and kind, message, full source span, operation,
bounded causal chain, resolution reason and remaining-alternative count,
exact-support status, cancellation context, and resource-limit context. A
failure result contains no partial computational value. Human rendering may add
formatting, but it cannot discard these structured fields or change their
meaning.

## 1. Mathematical Domains

### 1.1 STATE

Let the computational basis be the fixed ordered basis

```math
(|0\rangle,|1\rangle).
```

Matrix index $0$ therefore corresponds to $|0\rangle$, and matrix index $1$ corresponds to $|1\rangle$. In particular, $\rho_{00}$ is the upper-left entry and $\rho_{11}$ is the lower-right entry.

The domain of concrete Lana `STATE` values is

```math
\mathcal S
=
\left\{
\rho\in\mathcal L(\mathbb C^2)
\;\middle|\;
\rho=\rho^\dagger,\;
\rho\succeq0,\;
\operatorname{Tr}(\rho)=1
\right\}.
```

Here $\mathcal L(\mathbb C^2)$ denotes the linear operators on $\mathbb C^2$, $\rho^\dagger$ is the conjugate transpose, $\rho\succeq0$ means that $\rho$ is positive semidefinite, and $\operatorname{Tr}$ is the matrix trace. For distributional constructions, $\mathcal S$ carries the Borel measurable structure inherited from this finite-dimensional operator space.

Every $\rho\in\mathcal S$ has the canonical representation

```math
\rho=
\begin{pmatrix}
1-p & c \\
c^* & p
\end{pmatrix},
```

where

```math
0\le p\le1,
\qquad
c\in\mathbb C,
\qquad
|c|^2\le p(1-p).
```

Here

```math
p=\rho_{11},
\qquad
1-p=\rho_{00},
\qquad
c=\rho_{01},
\qquad
c^*=\rho_{10}.
```

These conditions are the global mathematical invariant for every concrete Lana `STATE`. The complex number $c$ is the canonical mathematical internal parameter.

The notation $c^*$ denotes the complex conjugate of $c$.

For $0<p<1$, define the normalized disposition

```math
d=
\frac{c}{\sqrt{p(1-p)}}.
```

The positivity condition implies

```math
d\in\mathbb C,
\qquad
|d|\le1.
```

At the probability boundaries

```math
p\in\{0,1\},
```

positive semidefiniteness forces $c=0$, and Lana defines

```math
d=0
```

by convention. Thus $c$ is canonical, while $d$ is a normalized representation derived from $(p,c)$ with an explicit boundary convention. No signed-real ordering or unrelated inequality on $d$ is part of Lana 1.0 semantics.

### 1.2 STATE_DIST

For a measurable space $E$, write $\operatorname{Dist}(E)$ for the set of probability distributions, equivalently probability measures, on $E$.

The mathematical type `STATE_DIST` is

```math
\operatorname{STATE_DIST}
=
\operatorname{Dist}(\mathcal S).
```

For an ordinary state $\rho\in\mathcal S$, the Dirac distribution

```math
\delta_\rho\in\operatorname{Dist}(\mathcal S)
```

assigns probability $1$ to $\rho$ and probability $0$ to every set that does not contain $\rho$. It embeds one concrete `STATE` as a degenerate `STATE_DIST`.

### 1.3 Core Type Signatures

The core operations have the signatures below. The parameter $\Phi$ ranges over the valid transforms defined in Section 3.

```math
\operatorname{MEASURE}:
\mathcal S
\rightarrow
\operatorname{Dist}(\{0,1\}),
```

```math
\operatorname{TRANSFORM}_\Phi:
\mathcal S
\rightarrow
\mathcal S,
```

and

```math
\operatorname{APPEND}:
\mathcal S\times\mathcal S
\rightarrow
\operatorname{STATE_DIST}.
```

Section 5 defines the lifted forms acting on `STATE_DIST` values.

## 2. MEASURE

### 2.1 Definition

For

```math
\rho=
\begin{pmatrix}
1-p & c \\
c^* & p
\end{pmatrix}
\in\mathcal S,
```

define computational-basis measurement by

```math
\operatorname{MEASURE}(\rho)
=
\operatorname{Bernoulli}(p).
```

Therefore

```math
P(1)=p,
\qquad
P(0)=1-p.
```

`MEASURE` returns a probability distribution. It does not automatically sample a classical scalar; sampling is a separate evaluation action defined in Section 5.6.

### 2.2 Interpretation

The diagonal entries of $\rho$ determine computational-basis outcome probabilities. The internal parameters $c$ and $d$ do not affect this measurement distribution.

`MEASURE` is read-only. It changes neither $p$, $c$, nor $d$, and it does not replace or collapse its input state.

### 2.3 Properties

`MEASURE` is well-defined for every concrete `STATE`, preserves its input, and depends only on the canonical probability $p$.

### 2.4 Proofs

**Theorem 2.1 (Measurement Well-Definedness).** For every $\rho\in\mathcal S$,

```math
\operatorname{MEASURE}(\rho)
\in
\operatorname{Dist}(\{0,1\}).
```

**Proof.** The `STATE` invariant gives $0\le p\le1$. Hence $p$ and $1-p$ are nonnegative and sum to $1$, so they define a Bernoulli probability distribution. $\square$

**Theorem 2.2 (Measurement State Preservation).** Applying `MEASURE` leaves its input state unchanged.

**Proof.** By definition, `MEASURE` reads $p$ and returns $\operatorname{Bernoulli}(p)$. It performs no state-producing or state-replacement step, so the input remains $\rho$. $\square$

**Theorem 2.3 (Internal-Parameter Independence).** If $\rho_1,\rho_2\in\mathcal S$ have the same $p$ and any valid internal parameters $c_1,c_2$, then

```math
\operatorname{MEASURE}(\rho_1)
=
\operatorname{MEASURE}(\rho_2).
```

**Proof.** Both sides equal $\operatorname{Bernoulli}(p)$ by Definition 2.1; $c$ and $d$ do not occur in that definition. $\square$

### 2.4 Named basis measurement

Lana defines these ordered binary bases:

```math
B_{\mathrm{computational}}=(|0\rangle,|1\rangle),
\qquad
B_x=(|+\rangle,|-\rangle),
```

where $|+\rangle=(|0\rangle+|1\rangle)/\sqrt2$, and

```math
B_y=(|+y\rangle,|-y\rangle),
\qquad
|+y\rangle=(|0\rangle+i|1\rangle)/\sqrt2.
```

Outcome $0$ always refers to the first vector in the named ordered basis. For
$\rho$ with canonical off-diagonal value $c=\rho_{01}$, define $q_B(\rho)$ as
the exact probability of outcome $1$ by

```math
q_{\mathrm{computational}}(\rho)=p,
\qquad
q_x(\rho)=\frac12-\operatorname{Re}(c),
\qquad
q_y(\rho)=\frac12-\operatorname{Im}(c).
```

The basis-aware result is always the existing binary distribution shape
$\operatorname{distribution}(1-q_B,q_B)$. Measurement is read-only and does
not collapse or otherwise mutate the input.

For a concrete `STATE`, probability, distribution, and classical sample modes
use this exact $q_B$. For a `STATE_DIST`, exact basis-aware sampling first draws
a concrete state $\rho\sim\mu$, then draws the binary outcome from
$\operatorname{Bernoulli}(q_B(\rho))$. Exact basis-aware probability and
distribution modes over `STATE_DIST` are intentionally unsupported until a
separate exact basis-expectation evaluator is defined.

### 2.5 Explicit Monte Carlo estimation

Lana provides an explicit approximate operation for basis-aware probability and
distribution measurement of a `STATE_DIST`. Given $N>0$ independent samples
$\rho_1,\ldots,\rho_N\sim\mu$, define

```math
\widehat q_{B,N}(\mu)=\frac1N\sum_{i=1}^N q_B(\rho_i).
```

`estimate_measure` returns $\widehat q_{B,N}$ in probability mode and
$\operatorname{distribution}(1-\widehat q_{B,N},\widehat q_{B,N})$ in
distribution mode. This regular sample-mean Monte Carlo estimator is a
deliberate language semantic, not an invisible runtime optimization. It is not
the exact mathematical probability $\int q_B(\rho)\,d\mu(\rho)$; accuracy is
sample-count dependent. No confidence interval is part of the result. The
runtime may use the existing seeded RNG and must not return a partial estimate
when cancellation or a resource limit interrupts the trials.

## 3. TRANSFORM

### 3.1 Definition

A valid concrete Lana transform is a deterministic, Borel-measurable function

```math
\Phi:\mathcal S\rightarrow\mathcal S.
```

For $\rho\in\mathcal S$, define

```math
\operatorname{TRANSFORM}_\Phi(\rho)
=
\Phi(\rho).
```

Mathematically, a transform returns a new `STATE`; Section 5.7 distinguishes this value semantics from language-level assignment.

### 3.2 Well-Formedness

If

```math
\Phi(\rho)=
\rho'=
\begin{pmatrix}
1-p' & c' \\
{c'}^* & p'
\end{pmatrix},
```

then validity requires

```math
0\le p'\le1,
\qquad
c'\in\mathbb C,
\qquad
|c'|^2\le p'(1-p').
```

A rule that is nondeterministic, not Borel-measurable, or violates these
conditions for any input in its declared domain is not a valid concrete
transform on that domain.

### 3.3 Composition

For valid transforms

```math
\Phi,\Psi:\mathcal S\rightarrow\mathcal S,
```

define ordinary function composition by

```math
(\Psi\circ\Phi)(\rho)
=
\Psi(\Phi(\rho)).
```

Define the identity transform by

```math
I(\rho)=\rho.
```

### 3.4 Properties

Valid transforms preserve `STATE` well-formedness, are closed under composition, compose associatively, and include the identity. Individual transforms need not be invertible.

Consequently, the set of valid Lana transforms forms a **monoid under function composition**, but not necessarily a group.

### 3.5 Proofs

**Theorem 3.1 (Transform Closure).** For a valid transform $\Phi$,

```math
\rho\in\mathcal S
\implies
\Phi(\rho)\in\mathcal S.
```

**Proof.** This is the codomain obligation in the definition $\Phi:\mathcal S\rightarrow\mathcal S$. $\square$

**Theorem 3.2 (Closure Under Composition).** If $\Phi$ and $\Psi$ are valid transforms, then $\Psi\circ\Phi$ is a valid transform.

**Proof.** For any $\rho\in\mathcal S$, validity of $\Phi$ gives $\Phi(\rho)\in\mathcal S$. Validity of $\Psi$ then gives $\Psi(\Phi(\rho))\in\mathcal S$. The composition of deterministic functions is deterministic, and the composition of Borel-measurable functions is Borel-measurable, so $\Psi\circ\Phi:\mathcal S\rightarrow\mathcal S$ is valid. $\square$

**Theorem 3.3 (Associativity of Transform Composition).** For valid transforms $\Phi$, $\Psi$, and $\Theta$,

```math
\Theta\circ(\Psi\circ\Phi)
=
(\Theta\circ\Psi)\circ\Phi.
```

**Proof.** For every $\rho\in\mathcal S$, both sides evaluate to $\Theta(\Psi(\Phi(\rho)))$. Therefore the functions are equal. $\square$

**Theorem 3.4 (Identity).** For every valid transform $\Phi$,

```math
I\circ\Phi
=
\Phi\circ I
=
\Phi.
```

**Proof.** The function $I$ is deterministic and maps every $\rho\in\mathcal S$ to the same element $\rho\in\mathcal S$, so it is valid. For every $\rho\in\mathcal S$, $I(\Phi(\rho))=\Phi(\rho)$ and $\Phi(I(\rho))=\Phi(\rho)$. $\square$

### 3.6 Interpretation

Lana 1.0 uses an abstract state semantics: every deterministic,
Borel-measurable endofunction on $\mathcal S$ is eligible to be a transform,
provided its output remains in $\mathcal S$.

If `STATE` is interpreted as a physical quantum state, physically realizable deterministic transformations satisfy stronger requirements, normally arising from completely positive trace-preserving maps on operators. Lana 1.0 does **not** require every abstract transform to be CPTP or physically realizable.

### 3.7 Lana 1.0 Registered Transforms

Lana 1.0 defines

```math
\operatorname{INVERT}(p,d)=(1-p,\overline d)
```

and

```math
\operatorname{NEUTRALIZE}(p,d)=(p,0).
```

Both rules are deterministic and continuous, hence Borel-measurable.
Conjugation preserves $|d|$, so `INVERT` preserves the unit-disk and boundary
invariants. `NEUTRALIZE` maps every disposition to the disk origin and
preserves $p$. Their exact expected-probability rules are $q\mapsto1-q$ and
$q\mapsto q$, respectively.

## 4. APPEND

### 4.1 Definition

For ordinary inputs $A,B\in\mathcal S$, `APPEND(A,B)` combines the represented binary events using Lana's independent probabilistic-OR modeling rule and returns a probability distribution over concrete output states.

Invoking `APPEND(A,B)` asserts independence of the two represented binary events **for that operation only**. Lana does not infer that all distinct `STATE` values are independent, and Lana 1.0 does not define `APPEND` for correlated events.

The internal distribution defined in Section 4.4 is a Lana modeling rule, not a consequence of probability theory or quantum mechanics.

### 4.2 Observable Probability

Let the input states have observable probabilities $p_A$ and $p_B$. Define

```math
p_C
=
p_A+p_B-p_Ap_B,
```

or equivalently

```math
p_C
=
1-(1-p_A)(1-p_B).
```

The output `STATE_DIST` has this fixed observable probability when both inputs are concrete `STATE` values.

### 4.3 Normalized Internal Values

Define the complex unit disk by

```math
\mathbb D
=
\{z\in\mathbb C:|z|\le1\}.
```

Let $d_A$ and $d_B$ be the normalized dispositions derived from the input states by Section 1.1. Thus

```math
d_A,d_B\in\mathbb D.
```

Define

```math
m_C
=
\frac{d_A+d_B}{2}
```

and

```math
\sigma_C
=
\frac{|d_A-d_B|}{2}.
```

Because $\mathbb D$ is convex, $m_C\in\mathbb D$, and $\sigma_C\ge0$.

### 4.4 Internal Distribution

For

```math
0<p_C<1,
\qquad
\sigma_C>0,
```

define the normalization constant

```math
Z(m,\sigma)
=
\int_{\mathbb D}
\exp\left(
-\frac{|z-m|^2}{2\sigma^2}
\right)
\,d^2z.
```

For $m\in\mathbb D$ and $\sigma>0$, the integrand is positive and continuous on the compact disk, so $0<Z(m,\sigma)<\infty$.

The normalized output disposition $d_C$ then has the truncated and renormalized circular complex-normal density

```math
f_C(d)
=
\frac{1}{Z(m_C,\sigma_C)}
\exp\left(
-\frac{|d-m_C|^2}{2\sigma_C^2}
\right),
\qquad
d\in\mathbb D,
```

with respect to two-dimensional Lebesgue measure $d^2d$ on the complex plane. By construction,

```math
\int_{\mathbb D}f_C(d)\,d^2d=1.
```

If $0<p_C<1$ and $\sigma_C=0$, the internal distribution is the Dirac distribution concentrated at

```math
d_C=m_C.
```

If

```math
p_C\in\{0,1\},
```

the internal distribution is the Dirac distribution concentrated at

```math
d_C=0.
```

These cases collectively define a probability distribution for $d_C$ without treating a degenerate case as an ordinary density $f_C$.

### 4.5 STATE_DIST Result

For every possible $d_C$ from Section 4.4, construct

```math
c_C
=
d_C\sqrt{p_C(1-p_C)}
```

and

```math
\rho_C
=
\begin{pmatrix}
1-p_C & c_C \\
c_C^* & p_C
\end{pmatrix}.
```

The ordinary `APPEND` result is the induced probability distribution of $\rho_C$:

```math
\operatorname{APPEND}(A,B)
\in
\operatorname{Dist}(\mathcal S).
```

### 4.6 Chaining

For concrete independent input events, the observable component chains as

```math
p_{AB}=1-(1-p_A)(1-p_B)
```

and

```math
p_{(AB)C}
=
p_{A(BC)}
=
1-(1-p_A)(1-p_B)(1-p_C).
```

More generally, for $n\ge1$ inputs satisfying the required independence assumptions,

```math
p_{\operatorname{APPEND}}
=
1-\prod_{i=1}^{n}(1-p_i).
```

This associativity applies only to the observable probability. The internal `STATE_DIST` construction is evaluated as a binary tree and is **not assumed associative**. Binary grouping therefore matters for the internal output distribution.

### 4.7 Properties

For ordinary `STATE` inputs, `APPEND` has bounded observable probability, produces only valid concrete states, and is commutative. Its observable probability is associative, while its internal distribution has no associativity guarantee.

### 4.8 Proofs

**Theorem 4.1 (APPEND Probability Bounds).** If $p_A,p_B\in[0,1]$, then $p_C\in[0,1]$.

**Proof.** Since $1-p_A,1-p_B\in[0,1]$, their product lies in $[0,1]$. Therefore $1-(1-p_A)(1-p_B)\in[0,1]$. $\square$

**Theorem 4.2 (APPEND STATE Validity).** Every concrete state in the output distribution is an element of $\mathcal S$.

**Proof.** Section 4.4 places every possible $d_C$ in $\mathbb D$, including the degenerate cases, so $|d_C|\le1$. Therefore

```math
|c_C|^2
=
|d_C|^2p_C(1-p_C)
\le
p_C(1-p_C).
```

Together with Theorem 4.1, the canonical matrix in Section 4.5 is Hermitian, positive semidefinite, and has trace $1$. Hence $\rho_C\in\mathcal S$. $\square$

**Theorem 4.3 (APPEND Commutativity).** For ordinary `STATE` inputs,

```math
\operatorname{APPEND}(A,B)
=
\operatorname{APPEND}(B,A).
```

**Proof.** The expression $1-(1-p_A)(1-p_B)$ is symmetric in $A$ and $B$. Also

```math
\frac{d_A+d_B}{2}
=
\frac{d_B+d_A}{2}
```

and

```math
\frac{|d_A-d_B|}{2}
=
\frac{|d_B-d_A|}{2}.
```

Thus $p_C$, $m_C$, and $\sigma_C$ are unchanged when the inputs are exchanged, so every branch of the internal-distribution definition and the induced output distribution is unchanged. $\square$

**Theorem 4.4 (Observable Associativity).** For three concrete inputs whose represented events satisfy the required independence assumptions,

```math
p_{(AB)C}
=
p_{A(BC)}
=
1-(1-p_A)(1-p_B)(1-p_C).
```

**Proof.** Applying $p_{XY}=1-(1-p_X)(1-p_Y)$ twice gives the displayed expression under either grouping. Induction on the number of inputs gives

```math
1-\prod_{i=1}^{n}(1-p_i).
```

$\square$

**Explicit limitation (internal non-associativity).** Lana 1.0 supplies no theorem or modeling rule equating the internal distributions of `APPEND(APPEND(A,B),C)` and `APPEND(A,APPEND(B,C))`. They may differ.

## 5. Composition and Evaluation Semantics

### 5.1 Evaluation Order

Nested core expressions follow the language's deterministic expression evaluation order. This document requires that the selected order be honored; parser and VM mechanics belong in their respective specifications.

For a binary `APPEND` tree, grouping is semantically meaningful. For example,

```text
APPEND(APPEND(A, B), C)
```

means:

1. evaluate `APPEND(A,B)`;
2. obtain a `STATE_DIST`;
3. apply lifted `APPEND` to that distribution and `C`.

It must not be silently rewritten as

```text
APPEND(A, APPEND(B, C))
```

because the internal distributions are not assumed equal.

### 5.2 Degenerate Embedding

Define

```math
\eta:\mathcal S\rightarrow\operatorname{Dist}(\mathcal S)
```

by

```math
\eta(\rho)=\delta_\rho.
```

This explicit embedding allows ordinary `STATE` and `STATE_DIST` values to participate uniformly in the lifted operations below.

### 5.3 Lifting Deterministic TRANSFORM

Not every valid concrete transform is runtime-admissible for distribution
lifting. A lifted transform must supply all three of the following:

1. a concrete `STATE` to `STATE` rule;
2. a proof that the rule preserves `STATE` validity; and
3. an exact rule for expected probability under its pushforward.

No conforming runtime may replace the third obligation with Monte Carlo
expectation estimation. A transform without that exact rule remains valid on
concrete `STATE` values but is unsupported on `STATE_DIST`.

For an admissible transform $\Phi:\mathcal S\rightarrow\mathcal S$, a distribution $\mu\in\operatorname{Dist}(\mathcal S)$, and every measurable set $E\subseteq\mathcal S$, define the pushforward distribution $\Phi_*\mu$ by

```math
(\Phi_*\mu)(E)
=
\mu(\Phi^{-1}(E))
```

Operationally, sample $\rho\sim\mu$ and return $\Phi(\rho)$. This defines

```math
\widehat{\operatorname{TRANSFORM}}_\Phi:
\operatorname{STATE_DIST}
\rightarrow
\operatorname{STATE_DIST},
\qquad
\widehat{\operatorname{TRANSFORM}}_\Phi(\mu)=\Phi_*\mu.
```

This lifting introduces no randomness beyond the randomness already represented by $\mu$.

### 5.4 Lifting APPEND

Let $K(A,B)$ denote the ordinary `APPEND(A,B)` output distribution defined by Sections 4.1–4.5.

Let $\mu,\nu\in\operatorname{Dist}(\mathcal S)$. For independent component draws

```math
\rho_A\sim\mu,
\qquad
\rho_B\sim\nu,
```

for every measurable set $E\subseteq\mathcal S$, define lifted `APPEND` as the distribution $\lambda$ satisfying

```math
\lambda(E)
=
\int_{\mathcal S}\int_{\mathcal S}
K(\rho_A,\rho_B)(E)
\,d\mu(\rho_A)\,d\nu(\rho_B)
```

Thus the fully distributed form has signature

```math
\widehat{\operatorname{APPEND}}:
\operatorname{Dist}(\mathcal S)\times\operatorname{Dist}(\mathcal S)
\rightarrow
\operatorname{Dist}(\mathcal S).
```

Equivalently: independently draw one component state from each input distribution, then apply the ordinary `APPEND` rule, including its conditional internal-distribution draw. The independent-input requirement is part of this lifted definition.

Using the embedding $\eta$ where necessary, the supported forms are

```math
\operatorname{STATE}\times\operatorname{STATE},
```

```math
\operatorname{STATE_DIST}\times\operatorname{STATE},
```

```math
\operatorname{STATE}\times\operatorname{STATE_DIST},
```

and

```math
\operatorname{STATE_DIST}\times\operatorname{STATE_DIST}.
```

Every form yields `STATE_DIST`. For distributed inputs, $p_C$ need not be fixed globally; it is computed conditionally from each sampled component pair.

### 5.5 Measurement of STATE_DIST

For $\mu\in\operatorname{STATE_DIST}$, let $X\in\{0,1\}$ denote the classical computational-basis outcome. Define `MEASURE` as the mixture distribution

```math
P(X=x)
=
\int_{\mathcal S}
P(X=x\mid\rho)
\,d\mu(\rho),
\qquad
x\in\{0,1\}.
```

This lifted operation has signature

```math
\widehat{\operatorname{MEASURE}}:
\operatorname{STATE_DIST}
\rightarrow
\operatorname{Dist}(\{0,1\}).
```

Writing $p(\rho)=\rho_{11}$, computational-basis measurement gives

```math
P(X=1)
=
\int_{\mathcal S}p(\rho)\,d\mu(\rho)
```

and

```math
P(X=0)
=
1-P(X=1).
```

This operation returns a distribution unless classical sampling is explicitly requested.

### 5.6 Sampling

For $\mu\in\operatorname{STATE_DIST}$,

```math
\operatorname{SAMPLE}(\mu)\sim\mu
```

denotes a stochastic evaluation that returns one concrete `STATE`. Sampling does not mutate its source distribution.

For a measurement result,

```math
X\sim\operatorname{Bernoulli}(p)
```

returns one classical binary scalar. This classical draw is distinct from `MEASURE`, which constructs the Bernoulli distribution, and from `SAMPLE` on `STATE_DIST`, which returns a concrete `STATE`.

### 5.7 Mutation Semantics

The core mathematical operations have value semantics:

- `MEASURE` is read-only.
- Sampling is read-only unless a separate, explicitly mutating language operation is defined elsewhere.
- `TRANSFORM` mathematically returns a new `STATE`.
- Language-level assignment or replacement determines whether a program binds that returned state to a variable.
- `APPEND` mutates neither input.

VM register behavior cannot redefine these mathematical value semantics.

## 6. Boundary and Error Semantics

### 6.1 STATE Construction

A proposed canonical state is invalid if

```math
p\notin[0,1]
```

or

```math
|c|^2>p(1-p).
```

A runtime must reject invalid construction rather than silently reinterpret it as another state.

### 6.2 Probability Boundaries

At

```math
p=0
\quad\text{or}\quad
p=1,
```

positive semidefiniteness forces

```math
c=0.
```

The normalized disposition is defined by convention as

```math
d=0.
```

### 6.3 APPEND Degeneracy

At

```math
p_C\in\{0,1\},
```

`APPEND` produces a degenerate internal distribution concentrated at

```math
d_C=0.
```

When $0<p_C<1$ and

```math
\sigma_C=0,
```

the internal distribution is concentrated at

```math
d_C=m_C.
```

### 6.4 Invalid TRANSFORM Output

If a transform produces a value that violates the `STATE` invariants, that rule is not a valid Lana transform for the input in question. A conforming runtime must reject or trap the result rather than expose it as a valid `STATE`.

### 6.5 Invalid Operation Types

Core operations are defined only over the domains declared in Sections 1 and 5. Unsupported type combinations are type errors or runtime errors. No conversion is permitted unless it is explicitly defined in this document, such as the degenerate embedding $\eta$.

### 6.6 Numerical Failure

The mathematical semantics use exact real and complex arithmetic. Finite-precision approximation does not change mathematical validity; implementation tolerances and required rejection behavior are specified in Section 7.3.

## 7. Implementation Correspondence

This section states only the minimum implementation obligations needed to realize the mathematics. VM architecture and bytecode encoding remain outside this document.

### 7.1 Canonical Runtime Information

A concrete `STATE` implementation must retain enough information to reconstruct the canonical mathematical pair $(p,c)$ and therefore the canonical matrix $\rho$.

No particular struct, register representation, redundant storage of $p$ and $1-p$, or storage of derived $d$ is required. If a runtime stores $d$ instead of $c$, it must preserve complex values and reconstruct

```math
c=d\sqrt{p(1-p)}
```

with the boundary convention from Section 1.1.

### 7.2 STATE_DIST Runtime Representation

A runtime does not enumerate the generally infinite set of possible states in a `STATE_DIST`. It represents each runtime-constructible distribution as a finite lazy expression describing how values are generated. The mathematical domain $\operatorname{Dist}(\mathcal S)$ is not a requirement to enumerate or provide constructors for every probability measure on $\mathcal S$.

Conceptually:

```text
STATE_DIST {
    distribution_kind
    parameters
    inputs
}
```

For ordinary `APPEND` inputs, a node may contain:

```text
p_C
m_C
sigma_C
unit_disk_truncation
input expressions
```

Degenerate nodes must encode the boundary or zero-spread rule rather than attempt to evaluate the nondegenerate density with $\sigma_C=0$.

Chained operations may form finite lazy expression trees:

```text
AppendDist(
    AppendDist(A, B),
    C
)
```

Evaluation, measurement, or sampling recursively evaluates only the portions required to produce the requested result. A finite expression represents a continuous probability distribution; it is not an enumeration of its support.

### 7.3 Floating-Point Policy

The mathematical definitions use exact arithmetic, but conforming implementations may use finite precision.

For Lana 1.0:

- probabilities must use sufficient precision to preserve the documented domain;
- tiny violations near $0$ and $1$ may be clamped only within a documented implementation tolerance;
- an implementation-defined small $\varepsilon\ge0$ may be used for the check

```math
|c|^2
\le
p(1-p)+\varepsilon;
```

- materially invalid values must not be silently repaired; and
- accepted near-boundary values must be canonicalized to a mathematically valid `STATE` before exposure as a concrete value.

The conforming C runtime selects $\varepsilon=10^{-12}$. This tolerance does
not materially enlarge $\mathcal S$; every accepted near-boundary value is
canonicalized before exposure.

### 7.4 Randomness

Stochastic evaluation requires a pseudo-random source. A conforming implementation may provide seeded reproducibility. RNG algorithm selection, stream ownership, and serialization belong in VM or runtime documentation, not in the mathematical definition.

### 7.5 Semantic Conformance

An implementation conforms when its observable results agree with this document, modulo documented floating-point approximation and pseudo-random sampling. Specifically:

- valid `STATE` construction corresponds to an element of $\mathcal S$;
- `MEASURE` produces the prescribed Bernoulli distribution or mixture;
- basis-aware concrete measurement produces the prescribed $q_B$ distribution;
- explicit `estimate_measure` produces the documented Monte Carlo estimator and
  does not claim exact `STATE_DIST` basis expectation;
- `TRANSFORM` produces the state or pushforward prescribed by $\Phi$;
- `APPEND` produces the prescribed `STATE_DIST`;
- sampling follows the represented distribution;
- supported joint operations preserve names, immutability, and their
  explicit error behavior;
- operation evaluation order matches Section 5; and
- invalid states and operations follow Section 6.

### 7.6 Encoding Boundary

Lana 1.0 implements the density-operator representation and core operations
defined above through one encoding, LABC v1. Other bytecode formats and
operations are outside this semantics and are rejected.

## 8. Post-freeze mathematical extensions

This section defines the mathematical targets for Lana 2.0 after the 1.0
feature freeze. It does not change the Lana 1.0 source language.
It does not change LABC v1, the compiler, or the VM. An implementation MUST
NOT expose an operation in this section as a 1.x compatibility feature.

### 8.1 State coordinates and neutralization

For $\rho\in\mathcal S$, let

$$
\operatorname{coord}(\rho)=(p,d)
$$

denote its canonical probability and normalized disposition coordinates from
Section 1.1. This notation does not identify the pair $(p,d)$ with the density
matrix $\rho$.

Define neutralization by

$$
\operatorname{neutralize}(\rho)=
\begin{pmatrix}
1-p & 0\\
0 & p
\end{pmatrix}.
$$

Equivalently, neutralization maps canonical coordinates as

$$
(p,d)\mapsto(p,0).
$$

`neutralize` preserves $p$, sets $c$ and $d$ to zero, and returns a valid
`STATE`. It is deterministic and does not mutate its input. Its derivation
records the operation and input. It is idempotent:

$$
\operatorname{neutralize}(\operatorname{neutralize}(\rho))
=\operatorname{neutralize}(\rho).
$$

### 8.2 Attenuation

For $\rho\in\mathcal S$ with $\operatorname{coord}(\rho)=(p,d)$ and
$f\in[0,1]$, define attenuation by the coordinate map

$$
\operatorname{coord}(\operatorname{attenuate}(\rho,f))=(p,fd).
$$

The result MUST be canonicalized as a valid `STATE`. The operation has the
following laws:

$$
\operatorname{attenuate}(\rho,1)=\rho,
$$

$$
\operatorname{attenuate}(\operatorname{attenuate}(\rho,f_1),f_2)
=\operatorname{attenuate}(\rho,f_1f_2).
$$

At $f=0$, the result is `neutralize(\rho)`. Factors outside $[0,1]$ are
errors. Non-finite factors are errors. Unsupported input representations are
errors. This section does not
define lifting over `STATE_DIST`. Do not use an implicit expected-value rule.

### 8.3 Convex state mixing

For $a,b\in\mathcal S$ and $w\in[0,1]$, define

$$
\operatorname{mix}(a,b,w)=wa+(1-w)b.
$$

This is a convex density-operator mixture. It is not evidence combination.
The result is in $\mathcal S$. The inputs remain unchanged. The operation is
read-only, except for its derivation record. The record contains both inputs,
$w$, exactness, the operation, and the revision. Define a metadata policy.
Do not copy metadata without that policy.

The operation is weight-symmetric:

$$
\operatorname{mix}(a,b,w)=\operatorname{mix}(b,a,1-w).
$$

It is idempotent:

$$
\operatorname{mix}(a,a,w)=a.
$$

`STATE_DIST` inputs are not supported. Define dependency and correlation rules
before you define their lifting semantics.

### 8.4 Trace distance

For $a,b\in\mathcal S$, define

$$
\operatorname{trace\_distance}(a,b)=\frac12\lVert a-b\rVert_1,
$$

where $\lVert X\rVert_1=\operatorname{Tr}(\sqrt{X^\dagger X})$. The result is
real and lies in $[0,1]$. The operation is read-only. It produces provenance.

The following laws MUST hold:

$$
\operatorname{trace\_distance}(a,b)=\operatorname{trace\_distance}(b,a),
$$

$$
\operatorname{trace\_distance}(a,b)=0\iff a=b,
$$

$$
\operatorname{trace\_distance}(a,c)
\le\operatorname{trace\_distance}(a,b)+\operatorname{trace\_distance}(b,c).
$$

Distribution comparison is deferred. The name MUST remain `trace_distance`.
The ambiguous name `distance` is not part of the semantic interface.

### 8.5 Relationship-aware APPEND

Section 4 defines only the independent APPEND operation in Lana 1.0. This
relationship-aware extension is post-freeze Lana 2.0 semantics.

These modes define overlap between two binary events. They do not define a
general evidence-fusion operator.

For binary events $A,B$ with observable probabilities $p_A,p_B$, let

$$
p_C=P(A\lor B),\qquad q=P(A\land B).
$$

Every non-synergistic relationship uses

$$
p_C=p_A+p_B-q.
$$

The valid overlap bounds are

$$
L=\max(0,p_A+p_B-1),\qquad U=\min(p_A,p_B),
$$

with

$$
L\le q\le U,
$$

and independent overlap $I=p_Ap_B$.

#### Independent

$$
q=I=p_Ap_B,
$$

so

$$
p_C=1-(1-p_A)(1-p_B).
$$

This is exactly the Lana 1.0 rule in Section 4.

#### Redundant event overlap

`REDUNDANT(r)` means overlap greater than independent overlap. For
$r\in[0,1)$,

$$
q=(1-r)I+rU,
\qquad
p_C=p_A+p_B-q.
$$

At $r=0$, this is independent. For $0<r<1$, this is partial redundant event
overlap. Partial redundant overlap has no associativity rule.

#### Full redundancy

`FULL_REDUNDANCY` applies only when

$$
p_A=p_B,
\qquad
q=p_A.
$$

Under these conditions, $A=B$ almost surely. `FULL_REDUNDANCY` is commutative
and idempotent. Its observable probability is associative only when every
binary node has an applicable full-redundancy relationship. It is not a
parameter value of `REDUNDANT(r)`.

#### Complementary event overlap

`COMPLEMENTARY(k)` means overlap less than independent overlap. For
$k\in[0,1]$,

$$
q=(1-k)I+kL,
\qquad
p_C=p_A+p_B-q.
$$

At $k=0$ this is independent. At $k=1$,

$$
q=L,\qquad p_C=\min(1,p_A+p_B).
$$

At $k=1$, this is maximum feasible separation. Only maximum separation has
observable-probability associativity. Partial complementary overlap has no
associativity rule.

#### Unified parameter

A signed parameter $\lambda\in[-1,1)$ can represent independent, partial
redundant, and complementary event overlap:

$$
q=
\begin{cases}
I+\lambda(U-I), & \lambda\ge0,\\
I+(-\lambda)(L-I), & \lambda<0.
\end{cases}
$$

$\lambda=-1$ is maximum feasible separation. $\lambda=0$ is independence.
$0<\lambda<1$ is partial redundant overlap. `FULL_REDUNDANCY` is separate.
$\lambda$ MUST NOT be described as Pearson correlation.

#### Synergistic

Synergy does not preserve the event meaning $C=A\lor B$. It requires an
explicit base relationship, an interaction event $S_{AB}$, and
$\eta\in[0,1]$. Let

$$
C_{\mathrm{base}}=A\lor B,
\qquad
p_{\mathrm{base}}=P(C_{\mathrm{base}}=1).
$$

The interaction event has these conditional probabilities:

$$
P(S_{AB}=1\mid C_{\mathrm{base}}=1)=0,
\qquad
P(S_{AB}=1\mid C_{\mathrm{base}}=0)=\eta.
$$

Define the synergistic output event as

$$
C=C_{\mathrm{base}}\lor S_{AB}.
$$

Then

$$
p_C=p_{\mathrm{base}}+\eta(1-p_{\mathrm{base}})
=1-(1-\eta)(1-p_{\mathrm{base}}).
$$

The runtime MUST NOT infer $\eta$. Missing or unresolved base data is an
error. Strength values below zero, above one, or non-finite are errors. At
$\eta=0$, the result canonicalizes to the base relationship. It is not an
active synergistic relationship. At $\eta=1$, $P(C=1)=1$. If
$p_{\mathrm{base}}=1$, set $S_{AB}=0$ almost surely. The conditional
probability on $C_{\mathrm{base}}=0$ is then not evaluated.

#### Dispositions and output

Relationship mode changes observable overlap only. It does not change the
existing internal disposition construction:

$$
m_C=\frac{d_A+d_B}{2},\qquad
\sigma_C=\frac{|d_A-d_B|}{2}.
$$

For each possible $d_C$, construct

$$
c_C=d_C\sqrt{p_C(1-p_C)},\qquad
\rho_C=
\begin{pmatrix}
1-p_C & c_C\\
c_C^* & p_C
\end{pmatrix}.
$$

The result is a `STATE_DIST`. Each concrete state satisfies the existing STATE
invariants. Relationship modes MUST NOT add disposition reinforcement,
cancellation, attenuation, or interaction cross-terms. Add them only with a
separate semantic definition.

Independent, redundant, complementary, and full-redundancy modes are
commutative. A synergistic mode is commutative only when its base relationship
is commutative. The complete `STATE_DIST` operation has no general
associativity or inverse law.

#### Distributed inputs

Let $\mu_A,\mu_B\in\operatorname{Dist}(\mathcal S)$ be the input state
distributions. A relationship-aware operation requires an outer state coupling

$$
\pi\in\operatorname{Coupling}(\mu_A,\mu_B),
$$

whose marginals are $\mu_A$ and $\mu_B$. This coupling selects which concrete
states occur together. Marginal distributions alone MUST NOT determine $\pi$.

For each coupled pair $(\rho_A,\rho_B)$, the selected relationship $R$ defines
an inner event overlap

$$
q_R(\rho_A,\rho_B)
=P(A=1,B=1\mid\rho_A,\rho_B,R).
$$

For each pair, $q_R$ MUST satisfy the overlap bounds from this section. For a
synergistic relationship, $q_R$ defines the base event $C_{\mathrm{base}}$.
The interaction event then changes the output from $C_{\mathrm{base}}$ to $C$.

The outer coupling $\pi$ and the inner overlap $q_R$ are different objects.
Neither determines the other.

Let $K_R(\rho_A,\rho_B;E)$ be the APPEND output kernel for a measurable set
$E\subseteq\mathcal S$. The kernel uses $q_R$ to calculate the base output
probability. A synergistic kernel then applies the interaction-event rule. The
kernel uses the disposition construction in this section. The relationship-aware
result is the distribution $\nu$ defined by

$$
\nu(E)=
\int_{\mathcal S\times\mathcal S}
K_R(\rho_A,\rho_B;E)\,d\pi(\rho_A,\rho_B).
$$

$K_R$ MUST be a Markov kernel. For each input pair, it is a probability
distribution over output states. For each measurable $E$, it is measurable in
the input pair. The overlap function $q_R$ MUST also be measurable.

Every relationship-aware APPEND over `STATE_DIST` inputs requires an explicit
or trusted coupling. If no applicable coupling exists, the operation is
unresolved and MUST NOT construct a product coupling.

A concrete `STATE` embeds as its Dirac distribution. This gives the unique
outer coupling when both inputs are concrete, or when one input is concrete.

#### Chaining

Every binary APPEND node requires a relationship that applies to that node's
two inputs. A relationship used to construct an intermediate APPEND result
MUST NOT become the relationship between that result and a later input.

For distributed inputs, every later binary node also requires its own explicit
or trusted coupling. Pairwise relationships or couplings of earlier inputs do
not determine the relationship or coupling for that later node.

#### Relationship resolution boundary

Mathematics is selected in this order:

1. Explicit caller relationship.
2. Trusted, subject-bound relationship metadata.
3. Unresolved relationship.

Do not infer a relationship from similarity, naming, provenance strings,
observed correlation, model output, or agent judgment alone. Unresolved,
conflicting, invalid, or unavailable relationships MUST NOT use
relationship-specific probability mathematics.

### 8.6 Statistical and surprisal semantics

Future statistical operations MUST state `exact`, `modelled`, or `sampled`.
A statistical result MUST preserve its method and assumptions. It MUST preserve
the sample count and seed when applicable. It MUST preserve its uncertainty
interval when available, provenance, and exactness status. Sampling never
replaces an exact operation without an explicit rule.

For a model event with probability $P$, surprisal is

$$
\operatorname{surprisal}(P)=-\log(P).
$$

The semantic definition MUST specify the logarithm base and units. It MUST
specify the event, outcome, model identity, model version, precision,
zero-probability behavior, and provenance. Surprisal is an analysis or routing
signal. It cannot authorize an effect without an explicit policy.

## Appendix A — Canonical Equation Reference

This appendix contains the canonical equations for the Lana 1.0 contract and
the post-freeze 2.0 extensions in Section 8. The numbered sections are
authoritative and include additional definitions for composition, lifting,
errors, and evaluation.

### STATE

```math
\rho=
\begin{pmatrix}
1-p & c \\
c^* & p
\end{pmatrix}
```

```math
0\le p\le1
```

```math
|c|^2\le p(1-p)
```

For $0<p<1$,

```math
d=
\frac{c}{\sqrt{p(1-p)}},
\qquad
|d|\le1.
```

For $p\in\{0,1\}$,

```math
c=0,
\qquad
d=0.
```

### MEASURE

```math
\operatorname{MEASURE}(\rho)
=
\operatorname{Bernoulli}(p).
```

### TRANSFORM

```math
\Phi:\mathcal S\rightarrow\mathcal S.
```

### APPEND

```math
p_C
=
1-(1-p_A)(1-p_B)
```

```math
m_C
=
\frac{d_A+d_B}{2}
```

```math
\sigma_C
=
\frac{|d_A-d_B|}{2}
```

For $0<p_C<1$ and $\sigma_C>0$,

```math
f_C(d)
=
\frac{1}{Z(m_C,\sigma_C)}
\exp\left(
-\frac{|d-m_C|^2}{2\sigma_C^2}
\right),
\qquad
d\in\mathbb D,
```

where

```math
\mathbb D=\{z\in\mathbb C:|z|\le1\}
```

and

```math
Z(m,\sigma)
=
\int_{\mathbb D}
\exp\left(
-\frac{|z-m|^2}{2\sigma^2}
\right)
\,d^2z.
```

```math
\int_{\mathbb D}f_C(d)\,d^2d=1.
```

The degenerate cases are

```math
d_C=
\begin{cases}
m_C, & 0<p_C<1\ \text{and}\ \sigma_C=0, \\
0, & p_C\in\{0,1\}.
\end{cases}
```

For every possible output disposition,

```math
c_C
=
d_C\sqrt{p_C(1-p_C)}.
```

### STATE_DIST

```math
\operatorname{STATE_DIST}
=
\operatorname{Dist}(\mathcal S).
```

## Reactive ordinary Information

For a process-local Information root $I$ with finite support $S_I$, a valid
observation $e$ must satisfy $e \in S_I$. A committed observation refines the
support to the singleton $\{e\}$; it never adds an alternative. If $e \notin
S_I$, the operation fails and no revision is published.

For an exact pure function $f$ and related uncertain inputs, lifting is the
pointwise image of their declared relationship. Reusing one dependency is
therefore zipped, not independent:

```math
f(I,I)=\{f(x,x):x\in S_I\}.
```

Two distinct dependency identities have no implicit joint law. Their ordinary
binary combination is undefined until a joint or conditional relationship is
declared; the runtime must not substitute $S_A \times S_B$. Exact operations
retain exact status, samples retain sample status, and explicit approximations
retain approximate status.

A pure dependency graph denotes functions of roots. Publishing revision $r+1$
atomically replaces every affected node with its value under the committed root
assignment and retains the value from revision $r$ as immutable history. Effect
results are leaves, not graph operations. A planned effect may execute at most
once for each process-local pair `(plan identity, committed revision)`; all
subsequent propagation reads its receipt.

## Shared Information

A live shared handle has a stable process-local identity $h$ and a totally
ordered commit revision $r_h$. Authority is explicit: read, observe, and admin
tokens are distinct, revocable capabilities. Possession of $h$ or an admin
token does not imply read or observation authority.

An observation is the pair $(t,e)$ of integer effective time and definite
evidence. A transaction replays the ordered sequence by $(t, sequence)$ into an
isolated dependency graph, then atomically publishes the complete candidate at
a unique process revision. Readers observe the complete previous commit or the
complete new commit. Equal-time equal evidence is idempotent; equal-time unequal
evidence is a conflict. Late evidence is inserted at its effective time and all
later versions are replayed. Contradiction, revocation, cancellation, resource
failure, or propagation failure discards the candidate and wakes no waiter.

Subscription is a predicate wait on `revision > after_revision`. A wake is not
a value publication: the waiter rechecks authority and the committed predicate
under the shared mutex before cloning an immutable task-local snapshot. Samples
remain private reads and never enter shared observation history.

## Appendix B — Semantic Operation ↔ Runtime Mapping

This table records current instruction availability without making opcode encoding normative. Detailed instruction behavior and encoding remain in `BYTECODE.md`.

| Semantic concept               | Runtime responsibility                                                            | Current direct opcode                       |
| ------------------------------ | --------------------------------------------------------------------------------- | ------------------------------------------- |
| `STATE` construction         | Validate and create a concrete canonical`STATE`                                 | `STATE_NEW` / `STATE_BUILD`             |
| `MEASURE`                    | Produce the computational-basis distribution without mutation                     | `MEASURE`                                 |
| Classical measurement sampling | Draw a binary scalar using the runtime RNG                                        | `MEASURE` sample mode                     |
| `TRANSFORM`                  | Execute an admissible$\Phi$ and reject invalid output                           | `TRANSFORM`                               |
| `APPEND`                     | Construct the prescribed lazy`STATE_DIST`                                       | `APPEND`                                  |
| `STATE_DIST` sampling        | Evaluate and sample a lazy distribution                                           | `SAMPLE_STATE_DIST`                       |
| Named n-ary joint construction | Validate names and build an immutable joint law/view                              | `JOINT_BUILD`                             |
| Joint projection               | Return an immutable named projection                                              | `JOINT_PROJECT`                           |
| Joint conditioning             | Refine by exact evidence or return an explicit error                              | `JOINT_CONDITION`                         |
| Joint sampling                 | Return definite member values without mutation                                    | `JOINT_SAMPLE`                            |
| Singleton resolution           | Return a definite value only for a singleton                                      | `RESOLVE`                                 |
| Finite possibility             | Validate nonempty support and preserve dependency                                 | `POSSIBILITY_BUILD`                       |
| Guarded branch execution       | Snapshot, execute, and join bounded alternatives                                  | `PATH_SPLIT` / `PATH_JOIN`              |
| Observation                    | Refine and record evidence after success                                          | `OBSERVE`                                 |
| General supported sampling     | Return one definite read-only result                                              | `INFO_SAMPLE`                             |
| Reactive ordinary Information  | Retain pure dependencies and atomically publish affected values                   | Information host metadata over pure opcodes |
| Runtime Claim                  | Retain value, proposition, exactness, tolerance, and source validity              | Claim host calls                            |
| Definite planned effect        | Execute once per process-local plan identity and revision, then reuse its receipt | Planned-effect host calls                   |
| Shared Information             | Capability-checked effective-time replay and atomic process revision              | Shared Information host calls               |
| Information inspection         | Render runtime state from derivation and reactive metadata                        | `information_inspect` host call           |

No opcode is inferred from a semantic operation merely because the operation is normative.
