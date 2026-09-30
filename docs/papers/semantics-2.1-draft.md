# Lana 2.1 Draft: Information Deltas and Relationships

## 0. Status and scope

This document is a non-normative draft for Lana 2.1. `semantics.md` remains
the mathematical authority. This draft does not replace `semantics.md` or
`semantics-2.md`. It does not define implemented source syntax, bytecode, or VM
behavior.

The proposed foundation is:

```math
\boxed{\text{information}=\text{a structured delta between two states}}.
```

Sections 0.1 through 0.4 specify the proposed meaning of a delta and its
preservation requirement. These definitions are a research proposal.

For probabilistic interactions, this draft uses:

```math
\boxed{\text{interaction}=\text{joint law}+\text{transformation rule}}.
```

The joint law describes how input information occurs together. The
transformation rule describes what the interaction produces.

This draft builds on Lana's existing `Joint`, coupling, and relationship-aware
`APPEND` definitions. It adds proposed measures for dependence, information
gain, and information available only from inputs considered collectively.

### 0.1 States before information

A state is a valid description of a declared domain. The description specifies
values, possible configurations, or probabilities, together with the
relationships that belong to that state. This definition does not require a
prior definition of information.

Let $S_A$ be the reference state and $S_B$ the destination state. Their domains
must be compatible, or an explicit mapping must connect them. The reference
and destination specify the direction of comparison. They do not imply time
order or causation.

The general term *state* in this proposal is distinct from the existing Lana
`STATE` type. A concrete `STATE` retains its established mathematical validity
conditions. The programmer supplies the interpretation of each modeled value
or event.

### 0.2 Structured deltas and reconstruction

Let $R_{AB}$ specify the relationship between the two states. The proposed
information object is:

```math
\Delta_{AB}=\operatorname{delta}(S_A,S_B;R_{AB}).
```

The relationship is an explicit input. The two endpoint descriptions do not
necessarily determine it. For probabilistic states, this relationship can
include a joint law. For possibility states, it can include an allowed
relation without probabilities.

The proposed preservation requirement is:

```math
\boxed{
\operatorname{reconstruct}(S_A,\Delta_{AB})=(S_B,R_{AB})
}.
```

The reference state must remain available and unambiguous. Together, the
reference and delta must recover the destination and the declared
relationship exactly. A delta without its reference does not promise this
recovery.

If the destination describes a distribution, reconstruction recovers that
distribution. It does not reveal an unobserved outcome. Reconstruction recovers
a mathematical description. It does not imply that a physical process or a
stochastic transformation has an inverse.

A direct representation can retain the destination and relationship alongside
the reference. This satisfies the requirement but establishes no compression
result. The delta notation also differs from $\Delta_P$ in Section 4.1, which
denotes a set of joint laws.

### 0.3 Why probability differences are insufficient

Let $X$ and $Y$ each have probability $1/2$ of being 1. Consider two
relationships:

1. The second bit copies the first, so $Y=X$.
2. The second bit is independent of the first.

Their joint laws differ:

| $X$ | $Y$ | Copying | Independence |
|---:|---:|---:|---:|
| 0 | 0 | $1/2$ | $1/4$ |
| 0 | 1 | $0$ | $1/4$ |
| 1 | 0 | $0$ | $1/4$ |
| 1 | 1 | $1/2$ | $1/4$ |

Both cases have the same endpoint distributions. Thus, the difference between
those distributions is zero in both cases. The first relationship lets one
bit determine the other. The second relationship does not.

A delta that preserves the relationship must distinguish these cases. A
numerical difference between endpoint probabilities cannot satisfy that
requirement alone. Zero difference between the distributions does not identify
an identity transformation.

### 0.4 Meaning of preserving all deltas

For the modeled pair, preservation includes both endpoint descriptions and
every relationship declared as part of $R_{AB}$. It does not include facts
outside the declared model. The reconstruction requirement makes this scope
explicit.

For a fixed reference, an exact encoding must satisfy:

```math
\operatorname{decode}_{S_A}
\bigl(\operatorname{encode}_{S_A}(\Delta_{AB})\bigr)
=\Delta_{AB}.
```

Here, equality means the same reference, destination, and declared
relationship. It does not require identical storage layouts. This proposed
equality does not add a general equality operation to the current runtime.

Preservation of every delta is sufficient by definition. The research problem
is to find representations and composition rules that satisfy this requirement
with less storage or computation. A size claim must account for retained
references and shared relationship data.

### 0.5 Context from existing semantics

The following eight passages locate the existing definitions relevant to the
delta proposal. The first three passages belong to the mathematical authority.
The remaining passages belong to this draft.

| Passage | Relevance to deltas |
|---|---|
| [Information model](semantics.md#01-information-model), Section 0.1 | Defines `Definite`, `Possibility`, `Distribution`, `Joint`, and `Paths`. Derivation history remains separate from mathematical equality. |
| [Information operations](semantics.md#011-information-operations), Section 0.1.1 | Distinguishes projection, conditioning, observation, sampling, and resolution. Section 8.1 records their preservation boundaries. |
| [Joint information](semantics.md#012-joint-information), Section 0.1.2 | Stores relationships in a joint law. Marginal values and runtime dependency edges do not supply that law. |
| [Status and scope](#0-status-and-scope), Section 0 | Separates the proposed delta foundation from the existing joint-law and transformation machinery. |
| [Connection between values](#1-connection-between-two-information-values), Section 1 | Shows which binary relationships require an explicit overlap parameter. |
| [Composition](#5-algebra-of-interaction-rules), Section 5 | Describes kernel composition, information loss, and the limits of pairwise relationships. |
| [Application to STATE](#6-application-to-lana-state), Section 6 | Distinguishes outer state coupling, inner event relationships, and output transformations. |
| [Information forms](#7-other-lana-information-forms) and [proposed operations](#8-proposed-initial-semantic-surface), Sections 7 and 8 | Keep probability laws, possibility relations, and guarded paths distinct. |

## 1. Connection between two information values

Let $X$ and $Y$ be the particular properties or outcomes represented by two
information values. For ordinary binary events, define

```math
p_A=P(X=1),
\qquad
p_B=P(Y=1),
```

and introduce their overlap

```math
q=P(X=1,Y=1).
```

The joint law is then

```math
J_{XY}=
\begin{pmatrix}
1-p_A-p_B+q & p_B-q\\
p_A-q & q
\end{pmatrix}.
```

Rows correspond to $X=0,1$. Columns correspond to $Y=0,1$. Each entry is the
probability of that pair occurring.

The relationship is valid exactly when

```math
\boxed{\max(0,p_A+p_B-1)\le q\le\min(p_A,p_B)}.
```

This joint law distinguishes the following connection properties:

| Property | Mathematical condition |
|---|---|
| Independent | $q=p_Ap_B$ |
| Occur together more often than independence predicts | $q>p_Ap_B$ |
| Occur together less often than independence predicts | $q<p_Ap_B$ |
| Mutually exclusive | $q=0$ |
| Whenever $X$ happens, $Y$ happens | $q=p_A$ |
| Always agree | $p_A=p_B=q$ |

The validity bounds determine whether a listed condition is possible for the
given marginals.

The relationship cannot generally be recovered from $p_A$ and $p_B$. It must
be supplied explicitly or estimated from evidence. Lana must preserve the
difference between a declared relationship and an estimated relationship.

## 2. The result of an interaction

Introduce an output variable $Z$ and a Markov kernel

```math
K(z\mid x,y).
```

For every input pair, the kernel must satisfy

```math
K(z\mid x,y)\ge0,
\qquad
\sum_zK(z\mid x,y)=1.
```

The interaction is the joint law

```math
\boxed{
P(X=x,Y=y,Z=z)=J_{XY}(x,y)K(z\mid x,y)
}.
```

Its output distribution is

```math
P(Z=z)=\sum_{x,y}J_{XY}(x,y)K(z\mid x,y).
```

This definition includes deterministic and probabilistic interactions. For
deterministic Boolean operations:

```math
\begin{aligned}
Z=X\land Y
&:\quad P(Z=1)=q,\\
Z=X\lor Y
&:\quad P(Z=1)=p_A+p_B-q,\\
Z=X\mathbin{\mathrm{XOR}}Y
&:\quad P(Z=1)=p_A+p_B-2q.
\end{aligned}
```

An interaction should retain $P(X,Y,Z)$ when later explanation,
conditioning, or projection needs the original inputs. Returning only $P(Z)$
discards that information.

When $X$ and $Y$ are evidence about a target $T$, evidence combination is the
conditional distribution

```math
P(T\mid X=x,Y=y)
```

derived from a declared joint model. Marginal probabilities alone do not
define this conditional distribution.

## 3. How much one value reveals about the other

### 3.1 Revelation from a particular observation

For an observation $X=x$ with $P(X=x)>0$, define

```math
P(Y=y\mid X=x)
=
\frac{J_{XY}(x,y)}{P(X=x)}.
```

If $P(X=x)=0$, the observation is impossible under the model and conditioning
must return an explicit error.

### 3.2 Average revelation

Define mutual information by

```math
\boxed{
I(X;Y)=
\sum_{\{(x,y):J_{XY}(x,y)>0\}}
J_{XY}(x,y)
\log_2
\frac{J_{XY}(x,y)}{P(X=x)P(Y=y)}
}.
```

The unit is bits. Mutual information is the average reduction in uncertainty
about one variable from observing the other.

For two fair binary variables:

1. If they are independent, knowing $X$ reveals $0$ bits about $Y$.
2. If they are always equal, knowing $X$ reveals $1$ bit about $Y$.
3. If they are always opposite, knowing $X$ also reveals $1$ bit about $Y$.

Agreement and information are therefore different properties. Perfect
disagreement can be perfectly informative.

## 4. Information that appears only collectively

Collective information requires a named target $T$. It asks how much $X$ and
$Y$ reveal about $T$ together that neither reveals alone.

Let $X$ and $Y$ be independent fair bits and define

```math
T=X\mathbin{\mathrm{XOR}}Y.
```

The joint law is

| $X$ | $Y$ | $T$ | Probability |
|---:|---:|---:|---:|
| 0 | 0 | 0 | $1/4$ |
| 0 | 1 | 1 | $1/4$ |
| 1 | 0 | 1 | $1/4$ |
| 1 | 1 | 0 | $1/4$ |

It satisfies

```math
I(T;X)=0,
\qquad
I(T;Y)=0,
\qquad
I(T;X,Y)=1.
```

Neither input reveals the target individually. Together they determine it.
This is informational synergy.

### 4.1 Proposed two-source information decomposition

For a finite joint law $P(X,Y,T)$, define

```math
a=I(T;X),
\qquad
b=I(T;Y),
\qquad
c=I(T;X,Y).
```

Let $\Delta_P$ be the set of all valid joint laws $Q(X,Y,T)$ that preserve
the two source-target marginals:

```math
\Delta_P=
\left\{
Q:
Q_{XT}=P_{XT},
\quad
Q_{YT}=P_{YT}
\right\}.
```

Define

```math
M=
\min_{Q\in\Delta_P}I_Q(T;X,Y).
```

Then define the four information components:

```math
\begin{aligned}
\operatorname{shared}(T;X,Y)
&=a+b-M,\\
\operatorname{unique}_X(T;X\setminus Y)
&=M-b,\\
\operatorname{unique}_Y(T;Y\setminus X)
&=M-a,\\
\operatorname{synergy}(T;X,Y)
&=c-M.
\end{aligned}
```

They satisfy

```math
I(T;X,Y)
=
\operatorname{shared}
+\operatorname{unique}_X
+\operatorname{unique}_Y
+\operatorname{synergy}.
```

Under the selected BROJA definition, the four quantities are nonnegative. The
choice of information decomposition is part of the semantics: other published
decompositions can assign different values.

Lana's existing `SYNERGISTIC` APPEND relationship is a different concept. It
adds an explicit interaction event to a base OR event. It does not measure
informational synergy about a target. The language and runtime must keep these
terms distinguishable.

## 5. Algebra of interaction rules

For kernels $K:Y\mid X$ and $L:Z\mid Y$, define sequential composition by

```math
\boxed{
(L\circ K)(z\mid x)
=
\sum_yL(z\mid y)K(y\mid x)
}.
```

This composition has the following laws:

1. **Closure.** Composing valid kernels produces a valid kernel.
2. **Identity.** The kernel $\operatorname{id}_X(x'\mid x)=1$ when $x'=x$
   and $0$ otherwise changes nothing under composition.
3. **Associativity.** For compatible kernels,
   $(M\circ L)\circ K=M\circ(L\circ K)$.
4. **Parallel independent composition.** Independent kernels combine as
   $(K\otimes L)(y,v\mid x,u)=K(y\mid x)L(v\mid u)$.
5. **No general inverse.** A kernel may discard information.

These laws make finite Lana information transformations a category of
stochastic maps. Copying, discarding, and deterministic functions can be
represented as special kernels.

Associative kernel composition does not make every binary merge associative.
Changing which inputs an operation consumes, changing their relationship, or
discarding an intermediate joint can change the result.

For collective events involving additional inputs, retain an n-variable joint
law such as $P(X,Y,W,T)$. Pairwise relationships do not generally determine
this law. In the XOR example, every pair is independent even though the triple
obeys a strict deterministic constraint.

### 5.1 Information cannot increase under processing

If $Z$ is computed solely from $(X,Y)$ by a kernel and receives no additional
evidence about $T$, then

```math
\boxed{I(T;Z)\le I(T;X,Y)}.
```

A computation may expose information already available collectively. It does
not create information about an external target from no additional evidence.

### 5.2 Proposed delta composition requirements

Delta composition requires its own preservation scope. A result that recovers
the endpoint states and their relationship does not necessarily recover every
intermediate state.

For example, the numerical path $0\to1\to0$ has net change zero. The path
$0\to0\to0$ also has net change zero. Their intermediate states differ. A
requirement to preserve the full path must retain that distinction.

For uncertain states, adjacent pairwise joints do not generally determine the
joint law of a whole path. A Markov assumption can supply a composition rule,
but the model must declare that assumption.

The delta proposal requires answers to these questions before it defines a
composition law:

1. Which domains and reference states permit composition?
2. Does composition preserve the endpoint relationship or the full path law?
3. Which additional relationships must accompany the input deltas?
4. Under which assumptions are identity and associativity valid?
5. Which deltas permit reversal, and which transformations discard information?

Preservation of the full path requires access to the intermediate states and
their declared joint relationships. Existing kernel associativity alone does
not establish these delta laws.

## 6. Application to Lana `STATE`

Interactions between Lana states require two distinct mathematical levels.

### 6.1 Outer state coupling

For state distributions $\mu_A,\mu_B\in\mathrm{Dist}(\mathcal S)$, define an
explicit coupling

```math
\pi\in\operatorname{Coupling}(\mu_A,\mu_B).
```

The coupling describes which concrete state values occur together. Its
marginals must be $\mu_A$ and $\mu_B$.

### 6.2 Inner event relationship

For each coupled pair $(\rho_A,\rho_B)$, define an event relationship

```math
J_{XY\mid\rho_A,\rho_B}
```

or its binary overlap

```math
q_R(\rho_A,\rho_B)
=
P(X=1,Y=1\mid\rho_A,\rho_B,R).
```

The outer coupling and inner event relationship are different objects. Neither
determines the other. A joint containing two fixed `STATE` values does not, by
itself, specify a relationship between their measurement outcomes.

### 6.3 State-valued output

For a relationship-specific Markov kernel

```math
K_R(\rho_A,\rho_B;E),
```

the output state distribution is

```math
\boxed{
\nu(E)=
\int_{\mathcal S\times\mathcal S}
K_R(\rho_A,\rho_B;E)
\,d\pi(\rho_A,\rho_B)
}.
```

For every input pair, $K_R$ must return a probability distribution over valid
Lana states. For every measurable output set $E$, it must be measurable in the
input pair.

An event overlap $q$ determines an observable output probability for a chosen
Boolean event operation. It does not determine the output disposition $d_C$.
Any operation that changes disposition must define that change separately and
preserve all `STATE` validity conditions.

## 7. Other Lana information forms

The same structural distinction applies beyond probability distributions:

1. `Definite` interactions use deterministic functions or Dirac kernels.
2. `Distribution` interactions use joint probability laws and Markov kernels.
3. `Joint` stores the declared relationship among named variables.
4. `Possibility` stores allowed combinations without probabilities.
5. `Paths` stores guarded execution alternatives and remains distinct from a
   joint law.

For `Possibility`, composition follows the allowed relation between values.
Mutual information and quantities measured in bits are undefined until a
probability law is supplied.

## 8. Proposed initial semantic surface

The proposed calculus requires these definitions:

1. State domains and validity conditions independent of the information
   definition.
2. Structured deltas with reference states, destination states, and declared
   relationships.
3. Exact reconstruction and an explicit preservation scope for delta
   composition.
4. Finite named joint laws.
5. Explicit deterministic and Markov transformation rules.
6. Projection and conditioning.
7. Entropy, conditional entropy, and mutual information over finite joints.
8. One explicitly selected two-source information decomposition.
9. Composition, identity, parallel composition, and data-processing laws.
10. Explicit conversion boundaries for `STATE`, `STATE_DIST`, `Possibility`,
    and other `Information` forms.

The programmer supplies the meanings of variables, declared relationships,
and evidence. The calculus determines the mathematical consequences of that
model.

### 8.1 Existing operation boundaries

Existing operations act on the current information forms. The delta proposal
must account for their distinct behavior:

| Operation | Existing meaning | Requirement for a delta representation |
|---|---|---|
| `project` | Returns the requested variables and their law. | Projection can discard distinctions. Preservation of the original pair requires access to the omitted content. |
| `condition` | Refines a value without an observation event. | The result alone need not recover the original law. A delta must retain the reference and conditioning relationship. |
| `observe` | Refines information and records an external evidence event. | Mathematical reconstruction must not repeat the external effect. The committed event remains distinct from the returned value. |
| `sample` | Returns one definite outcome without changing the input law. | An outcome does not replace its source law. The declared relationship between the sample and source remains explicit. |
| `resolve` | Succeeds only for definite or singleton information. | A delta cannot supply an arbitrary outcome for an unresolved value. |

Provenance records derivation history under the current semantics. A provenance
edge alone does not establish a joint law, causation, or a transformation rule.
The delta proposal does not silently change this boundary.

### 8.2 Original mathematics target

The objective is an original mathematical result about information deltas.
Joint laws, Markov kernels, mutual information, and the selected BROJA
decomposition are existing mathematics. The delta definition and reconstruction
requirement alone do not establish novelty.

The research question is:

> For a specified class of computations, what is the smallest delta representation
> that preserves the modeled relationships, including information available only
> from combined inputs?

The initial research steps are:

1. Specify a finite state domain, admissible relationships, equality, and permitted operations.
2. Construct a candidate delta representation and a composition rule.
3. Search for counterexamples, including copying, independence, XOR, and paths with equal net changes.
4. Prove the conditions for exact recovery and the limits on representation size.
5. Compare the result with prior work on sufficient statistics, stochastic transformations, and information decomposition.

A successful result can be a new composition theorem, a sharp size bound, or
an impossibility theorem. A new theorem can use established mathematical
objects. Definitions, numerical examples, and literature comparisons remain
distinct from a proof and a supported novelty claim.

## 9. Validation status

### 9.1 Existing relationship examples

The binary relationship equations were checked numerically across 4,851 valid
binary-law cases. The checks covered normalization, marginal preservation,
AND, OR, XOR, and nonnegative mutual information.

The following reference cases were also checked:

1. Independent fair bits have zero mutual information.
2. Perfectly equal fair bits have one bit of mutual information.
3. Perfectly opposite fair bits have one bit of mutual information.
4. The XOR example gives zero information from either source alone and one bit
   from both sources together.
5. The proposed decomposition gives the expected results for XOR, duplicate
   sources, one unique source, and two independent unique sources.
6. Finite kernel composition preserved normalization, identity, and
   associativity in the checked examples.

These checks detect arithmetic errors in the examples. They are not a proof of
the complete proposal and do not establish implementation or conformance.

### 9.2 Delta proposal

Exact rational arithmetic verifies the copying and independence tables in
Section 0.3. Both tables normalize, share the same endpoint distributions, and
have different joint laws. The paths in Section 5.2 have equal net changes and
different intermediate states.

The reconstruction and encoding equations specify proposed requirements.
This draft supplies no general delta encoder, decoder, composition algorithm,
minimality proof, or novelty result. The checks of existing relationship
examples do not establish those results.

## 10. References

1. `semantics.md`, Sections 0.1 and 0.1.2: Lana `Information` and `Joint`.
2. `semantics-2.md`, Section 9.5: relationship-aware `APPEND`, couplings, and
   output kernels.
3. Polyanskiy, Y. and Wu, Y., *Information Theory*, Chapter 2: mutual
   information, conditional mutual information, and data processing.
4. Fritz, T., *A synthetic approach to Markov kernels, conditional
   independence and theorems on sufficient statistics*, 2020.
5. Williams, P. L. and Beer, R. D., *Nonnegative Decomposition of Multivariate
   Information*, 2010.
6. Bertschinger, N., Rauh, J., Olbrich, E., Jost, J., and Ay, N., *Quantifying
   Unique Information*, 2014.
7. Fullwood, J. and Parzygnat, A. J., [*The information loss of a stochastic
   map*](https://arxiv.org/abs/2107.01975), 2021. Related work on information
   measures for stochastic transformations, not a validation of the delta proposal.
