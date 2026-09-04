# 2.0 - Lana Mathematical Semantics: The Density-Operator Substrate

## 0. Scope and relationship to 1.0

This document is the normative Lana 2.0 definition of the generalized state
object and its operations. It **replaces** the Lana 1.0 `STATE`/`STATE_DIST`
core with an N-qubit density-operator substrate. It is a new substrate, not an
extension: the 1.0 core is recovered as the N=1 special case, but the source
language, bytecode, and VM are redefined around this document.

The authority order is unchanged: this document, then `SPEC.md`, `BYTECODE.md`,
`VM.md`. Where this document and `semantics.md` (1.0) disagree, this document is
authoritative for Lana 2.0.

Decisions fixed by this substrate:

1. A state holds **N qubits**, with N a runtime value, `1 <= N <= 10` for all of
   Lana 2.x.
2. **Entanglement** is supported.
3. **Two composition operations** exist: tensor product (side-by-side) and the
   classical merge (same-size evidence combination, the 1.0 `APPEND` rule).
4. **Measurement is user-defined** (POVM).
5. **Evolution is user-defined** (quantum channels), including noise.
6. **Complex amplitudes everywhere**; classical probability is the diagonal
   special case.

## 1. The state space

### 1.1 Qubit register

For a register of N qubits, the Hilbert space is

$$
\mathcal H_N = (\mathbb C^2)^{\otimes N},
\qquad
\dim \mathcal H_N = d = 2^N.
$$

The computational basis is the ordered set

$$
\{|b\rangle : b \in \{0,1\}^N\},
$$

indexed by bit strings of length N. Matrix index `b` corresponds to the basis
vector `|b\rangle`. N is a runtime value; the substrate is the family of spaces
`\mathcal H_N` for `1 <= N <= 10`.

### 1.2 Density operator

The domain of concrete Lana 2.0 states on N qubits is

$$
\mathcal S_N =
\left\{
\rho \in \mathcal L(\mathcal H_N)
\;\middle|\;
\rho = \rho^\dagger,\;
\rho \succeq 0,\;
\operatorname{Tr}(\rho) = 1
\right\}.
$$

A density operator is Hermitian, positive semidefinite, and unit-trace. This
single object is the universal state: every other state form is a special case
or an operation on it.

### 1.3 Pure states

A pure state is a rank-1 density operator

$$
\rho = |\psi\rangle\langle\psi|,
\qquad
|\psi\rangle \in \mathcal H_N,
\qquad
\langle\psi|\psi\rangle = 1.
$$

Pure-state languages (state-vector models) embed here.

### 1.4 Classical states

A classical state is a density operator diagonal in the computational basis:

$$
\rho = \sum_{b \in \{0,1\}^N} p_b\, |b\rangle\langle b|,
\qquad
p_b \ge 0,
\qquad
\sum_b p_b = 1.
$$

The vector `(p_b)` is a probability distribution over bit strings. Classical
probabilistic languages embed here. A classical state has zero coherence: all
off-diagonal entries vanish.

### 1.5 The N=1 recovery

For N=1, write `\rho` in the computational basis `(|0\rangle, |1\rangle)`:

$$
\rho =
\begin{pmatrix}
1-p & c \\
c^* & p
\end{pmatrix},
\qquad
0 \le p \le 1,
\qquad
|c|^2 \le p(1-p).
$$

This is exactly the Lana 1.0 `STATE` of `semantics.md` Section 1.1, with
`p = \rho_{11}` and `c = \rho_{01}`. The normalized disposition is

$$
d = \frac{c}{\sqrt{p(1-p)}}
$$

for `0 < p < 1`, and `d = 0` by convention at `p \in \{0,1\}`. The 1.0
`STATE` is therefore the N=1 density operator; the 1.0 `STATE_DIST` is a
probability distribution over `\mathcal S_1`.

## 2. Composition

### 2.1 Tensor product

For independent systems on registers A and B,

$$
\rho_{AB} = \rho_A \otimes \rho_B
\qquad\in \mathcal S_{N_A + N_B}.
$$

This is the "side-by-side" composition: it produces a larger state whose qubit
count is the sum. It is the general composition of independent systems and is
the substrate's primary combine operation.

### 2.2 Partial trace

For a state on registers A and B, the reduced state of A is

$$
\rho_A = \operatorname{Tr}_B(\rho_{AB}).
$$

Partial trace is the marginalization operation: "look at one subsystem of many."
It is the substrate's projection operation.

### 2.3 Classical merge (APPEND)

The 1.0 `APPEND` rule is retained as a **distinct, named operation** on the
N=1 (or diagonal) case. It combines two beliefs about the *same* binary event
using the independent probabilistic-OR rule

$$
p_C = 1 - (1-p_A)(1-p_B),
$$

and does **not** change the qubit count. It is not the tensor product, and the
two must not be conflated. Its full 1.0 definition (internal disposition
distribution, chaining, lifting) remains as in `semantics.md` Section 4, now
scoped to the N=1 case.

## 3. Measurement

### 3.1 POVM

A positive operator-valued measure (POVM) on `\mathcal H_N` is a finite set

$$
\{E_i\}_{i=1}^m,
\qquad
E_i \succeq 0,
\qquad
\sum_{i=1}^m E_i = I.
$$

For a state `\rho \in \mathcal S_N`, the outcome probability is

$$
p(i) = \operatorname{Tr}(\rho E_i).
$$

A POVM is a first-class value: users construct and pass their own. The outcome
distribution is `(p(1), \ldots, p(m))`.

### 3.2 Projective measurement (special case)

A projective measurement (PVM) is the special case where the `E_i` are
orthogonal projectors:

$$
E_i = |\phi_i\rangle\langle\phi_i|,
\qquad
\langle\phi_i|\phi_j\rangle = \delta_{ij},
\qquad
\sum_i |\phi_i\rangle\langle\phi_i| = I.
$$

### 3.3 Born rule

For a pure state `\rho = |\psi\rangle\langle\psi|`, the POVM outcome probability
reduces to the Born rule:

$$
p(i) = \langle\psi| E_i |\psi\rangle.
$$

### 3.4 The 1.0 measurement recovery

The 1.0 computational-basis measurement is the N=1 projective measurement with
`E_1 = |1\rangle\langle1|` and `E_0 = |0\rangle\langle0|`, giving
`p(1) = \rho_{11} = p`. The 1.0 named bases (computational, x, y) are the N=1
projective measurements in those bases.

## 4. Evolution

### 4.1 Quantum channel (CPTP)

A quantum channel is a completely positive, trace-preserving (CPTP) map

$$
\Phi : \mathcal L(\mathcal H_N) \to \mathcal L(\mathcal H_N).
$$

Every channel has a Kraus representation

$$
\Phi(\rho) = \sum_k K_k \rho K_k^\dagger,
\qquad
\sum_k K_k^\dagger K_k = I.
$$

A channel is a first-class value: users construct and pass their own. This is
the substrate's transform operation, and it generalizes the 1.0 `TRANSFORM`.

### 4.2 Unitary evolution (special case)

Unitary evolution is the single-Kraus-operator channel

$$
\Phi(\rho) = U \rho U^\dagger,
\qquad
U^\dagger U = I.
$$

### 4.3 Noise

Noise and decoherence are non-unitary channels. Standard examples are the
amplitude-damping and phase-damping channels, each expressible in Kraus form.
Noise is not a separate primitive; it is the class of channels that are not
unitary.

### 4.4 The 1.0 TRANSFORM recovery

The 1.0 `TRANSFORM` is the N=1 case of a channel. The 1.0 registered transforms
(`INVERT`, `NEUTRALIZE`) are specific N=1 channels. The 1.0 abstract-transform
monoid (`semantics.md` Section 3) is subsumed: every channel is a valid
transform, but not every abstract 1.0 transform is a channel.

## 5. Entanglement

### 5.1 Separability

A bipartite state `\rho_{AB} \in \mathcal S_{N_A + N_B}` is **separable** if it
can be written as a convex combination of product states:

$$
\rho_{AB} = \sum_i p_i\, \rho_A^{(i)} \otimes \rho_B^{(i)},
\qquad
p_i \ge 0,
\qquad
\sum_i p_i = 1.
$$

### 5.2 Entanglement

A state is **entangled** if it is not separable. Entanglement is the genuinely
new capability of the N-qubit substrate: two qubits whose joint state cannot be
described as independent or classically correlated subsystems.

## 6. Observables and the Lana surface

### 6.1 Observable

An observable is a Hermitian operator `A = A^\dagger` on `\mathcal H_N`. Its
expectation in a state `\rho` is

$$
\langle A \rangle_\rho = \operatorname{Tr}(\rho A).
$$

This generalizes the 1.0 binary `p` to arbitrary observables.

### 6.2 The 1.0 projections

For N=1, the 1.0 surface maps as:

- `p` (observable probability) = `\rho_{11} = \operatorname{Tr}(\rho |1\rangle\langle1|)`.
- `d = d_{re} + i d_{im}` = the normalized coherence `c / \sqrt{p(1-p)}`.
- `measure as probability` = `\operatorname{Tr}(\rho E)` for the relevant POVM element.
- `sample` = a draw from the outcome distribution `p(i) = \operatorname{Tr}(\rho E_i)`.

## 7. Boundary and error semantics

A proposed state is invalid if it is not Hermitian, not positive semidefinite,
or not unit-trace (within the floating-point tolerance of `semantics.md`
Section 7.3). A POVM is invalid if any `E_i` is not positive semidefinite or
the sum is not the identity. A channel is invalid if its Kraus operators do not
satisfy `\sum_k K_k^\dagger K_k = I`. Invalid constructions are rejected, never
silently repaired.

## 8. Implementation correspondence

For Lana 2.x, `N <= 10`, so `d = 2^N <= 1024`. A dense `d \times d` complex
matrix is the default representation; structured (sparse, stabilizer, MPS)
representations are a later optimization, not a semantic requirement. The
substrate is realized as a library over the general-purpose core (values,
arrays, maps, functions, calls, host calls), not as dedicated opcodes. The
mathematical objects defined here are independent of that representation.

## 9. Operations

This section defines the 2.0 operations, moved from `semantics.md` Section 8
and generalized to the N-qubit substrate where applicable. Sections 9.1–9.4 are
the N-qubit forms; 9.5 and 9.6 remain N=1 (or statistical) and are moved
verbatim. Cross-references to "Section 4" inside 9.5 refer to `semantics.md`.

### 9.1 Neutralize (full dephasing)

For `\rho \in \mathcal S_N`, the completely dephasing map in the computational
basis is

$$
\operatorname{neutralize}(\rho)
=
\sum_{b \in \{0,1\}^N} |b\rangle\langle b|\, \rho\, |b\rangle\langle b|
=
\operatorname{diag}(\rho).
$$

It zeroes every off-diagonal element and keeps the diagonal. It is a channel
(Kraus operators are the projectors `|b\rangle\langle b|`), it is idempotent,
and it maps every state to a classical state. At N=1 it recovers the 1.0
`NEUTRALIZE` transform: `(p, d) \mapsto (p, 0)`.

### 9.2 Attenuate (partial dephasing)

For `\rho \in \mathcal S_N` and `f \in [0,1]`,

$$
\operatorname{attenuate}(\rho, f)
=
f\,\rho + (1-f)\,\operatorname{diag}(\rho).
$$

It multiplies every off-diagonal element by `f` and keeps the diagonal. It is a
channel (the dephasing channel). The laws are

$$
\operatorname{attenuate}(\rho, 1) = \rho,
$$

$$
\operatorname{attenuate}(\operatorname{attenuate}(\rho, f_1), f_2)
=
\operatorname{attenuate}(\rho, f_1 f_2).
$$

At `f = 0` the result is `neutralize(\rho)`. Factors outside `[0,1]` and
non-finite factors are errors. At N=1 it recovers the 1.0 attenuation:
`(p, d) \mapsto (p, f d)`.

### 9.3 Convex state mixing

For `a, b \in \mathcal S_N` and `w \in [0,1]`,

$$
\operatorname{mix}(a, b, w)
=
w\,a + (1-w)\,b.
$$

This is a convex density-operator mixture, not evidence combination. The result
is in `\mathcal S_N`. It is weight-symmetric and idempotent:

$$
\operatorname{mix}(a, b, w) = \operatorname{mix}(b, a, 1-w),
$$

$$
\operatorname{mix}(a, a, w) = a.
$$

### 9.4 Trace distance

For `a, b \in \mathcal S_N`,

$$
\operatorname{trace\_distance}(a, b)
=
\frac12 \lVert a - b \rVert_1,
\qquad
\lVert X \rVert_1 = \operatorname{Tr}\!\left(\sqrt{X^\dagger X}\right).
$$

The result is real and lies in `[0,1]`. The laws are symmetry, zero iff equal,
and the triangle inequality:

$$
\operatorname{trace\_distance}(a, b) = \operatorname{trace\_distance}(b, a),
$$

$$
\operatorname{trace\_distance}(a, b) = 0 \iff a = b,
$$

$$
\operatorname{trace\_distance}(a, c)
\le
\operatorname{trace\_distance}(a, b) + \operatorname{trace\_distance}(b, c).
$$

### 9.5 Relationship-aware APPEND (N=1)

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

### 9.6 Statistical and surprisal semantics

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
signal. It cannot authorize an effect without an explicit policy. The base and
units are fixed in §9.10.1.

### 9.7 Evidence lifecycle, causal links, provenance, and deterministic replay

#### 9.7.1 Evidence record

An evidence value is the immutable tuple

```text
(value, status, producer, time)
```

where `status` is one of the disjoint labels

```text
unknown, sampled, modeled, exact, observed
```

ordered by certainty

```text
unknown < sampled < modeled < exact < observed
```

`producer` identifies the source (a root identity, operation, or external
observer); `time` is the integer effective time. The status is a credibility
label, not a probability: it records how the value was obtained, and it is
never derived from a `STATE` probability, similarity, naming, or correlation.

#### 9.7.2 Status combination

Combining two evidence values under an operation (e.g. `APPEND`) combines
their values per that operation's law, but the resulting status is the
least-certain input:

```math
\operatorname{status}(e_1 \oplus e_2)
= \min(\operatorname{status}(e_1), \operatorname{status}(e_2))
```

under the certainty order above. Combining evidence can add information
content but cannot upgrade the weakest input: `exact ⊕ sampled = sampled`,
`modeled ⊕ observed = modeled`, `exact ⊕ unknown = unknown`. The producer and
time of the result are the operation's own producer and time; the input
producers and times are retained in provenance (§9.7.4), not overwritten.

#### 9.7.3 Causal links

A causal link `A → B` is a declared conditional relationship: the law of `B`
is conditioned on `A`,

```math
B \sim P(B \mid A).
```

It reuses the `conditional(x | y)` joint construction of `semantics.md` (1.0)
§0.1.2; it does not introduce a parallel system. A causal link is distinct
from two other edges:

- a **dependency edge** — "B was computed from A" — is composition or
  execution structure and makes no probabilistic claim (`semantics.md` (1.0)
  §0.1.2);
- a **cause chain** — the sequence of operations that led to a failure — is a
  failure-record field (`semantics.md` (1.0) §0.2), not a probabilistic
  relationship.

Only the causal link asserts that A's probability changes B's probability. A
bare dependency edge never implies a causal link, and a causal link must be
declared, never inferred from dataflow, naming, or correlation.

#### 9.7.4 Provenance paths

A provenance path is a path in the derivation DAG from a root to a node. Each
node's provenance is the set of `(root, path)` pairs that produced it.
Operations preserve provenance: an exact operation retains exact status, a
sampled operation retains sample status, and an explicit approximation retains
approximate status. Unwrapping a value does not erase its provenance; it only
makes the value available to an exact operation.

#### 9.7.5 Deterministic replay

Replay is byte-identical: given the same `(seed, inputs, version)`, the output
bytes are identical. This requires a fixed evaluation order, IEEE 754 binary64
with no fused-multiply-add reordering, a seeded deterministic RNG, and a
deterministic iteration order over maps and sets. The floating-point policy
and randomness policy of `semantics.md` (1.0) §7.3 and §7.4 remain
authoritative; replay is their composition contract. A version change may
change bytes; a seed or input change may change bytes; nothing else may.

### 9.8 Lazy bounded datasets and declared relationships

#### 9.8.1 Lazy dataset

A lazy dataset is a computable function

```math
f : \mathbb{N} \to \operatorname{Value}
```

(or an equivalent generator) that produces values on demand. It occupies
constant space until a value is forced. A lazy dataset may be infinite.

#### 9.8.2 Bounded materialization

"Bounded" bounds materialization, not cardinality: the number of values forced
at once, or per step, is bounded. An infinite lazy dataset is valid; forcing is
bounded. Exhausting the materialization bound is an error and exposes no
partial result.

#### 9.8.3 Declared independence

`independent(x, y)` denotes the product measure

```math
\mu_x \otimes \mu_y.
```

The declaration is an authoritative assertion by the declarer. The system does
not verify it and never infers independence from array shape, naming,
similarity, or correlation. If the assertion is false, the declarer is
responsible for the resulting error; the system honored the declaration as
instructed.

#### 9.8.4 Declared correlation

`correlated(x, y)` declares a relationship with coefficient `d`, reusing the
normalized off-diagonal of `STATE`:

```math
d = \frac{c}{\sqrt{p(1-p)}}, \qquad |d| \le 1.
```

`d = 0` is independence; `|d| = 1` is full correlation. The first version
covers the binary (N=1) case where `d` is already defined; non-binary
generalizations are deferred.

### 9.9 Deterministic resampling and structured statistical results

#### 9.9.1 Bootstrap

Given `n` observations `x_1, ..., x_n` and a statistic
`θ̂ = T(x_1, ..., x_n)`, bootstrap resampling draws `n` observations with
replacement for each of `B` resamples:

```math
x^*_b = (x_{i_1}, \dots, x_{i_n}),
\qquad i_j \sim \operatorname{Uniform}\{1,\dots,n\},
```

and computes `θ*_b = T(x*_b)`. The bootstrap distribution is the empirical
distribution of `{θ*_b}`. The draws are seeded and deterministic.

#### 9.9.2 Structured result

A structured statistical result is the tuple

```text
(estimate, ci_low, ci_high, method, procedure, sample_count, seed)
```

where `estimate` is the point estimate, `ci_low`/`ci_high` are the 95%
confidence interval (the 2.5% and 97.5% percentiles of `{θ*_b}`), `method` is
one of `exact`, `modeled`, `sampled` (bootstrap is `sampled`), `procedure` is
`bootstrap`, `sample_count` is `B`, and `seed` is the RNG seed. A result never
collapses to a bare number; the method, procedure, count, and seed are
preserved.

#### 9.9.3 Resample count

The resample count `B` has a system default chosen for stability (the system
increases `B` until the confidence interval stabilizes within a tolerance) and
may be overridden by the user. The chosen `B` is recorded in the result.

### 9.10 Surprisal and uncertainty-aware ML

#### 9.10.1 Surprisal

Surprisal uses the natural logarithm, in nats:

```math
\operatorname{surprisal}(P) = -\ln(P).
```

The event, outcome, model identity, model version, precision,
zero-probability behavior, and provenance are specified with the value.
`surprisal(0)` is `+∞` and is reported as such, never as a finite substitute.

#### 9.10.2 Uncertainty-aware ML

Every ML operation returns its prediction together with its uncertainty; no ML
operation returns a bare prediction. The uncertainty is carried as part of the
result and is never discarded or coerced into an ordinary value.

#### 9.10.3 Surprisal action policy

Surprisal is reported by default. Routing, flagging, or acting on a surprisal
signal occurs only on an explicit user prompt; the prompt is the policy that
authorizes the action. Surprisal never authorizes an effect on its own.
