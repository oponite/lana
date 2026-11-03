# Five Information Forms for Explicit Uncertainty

**Author:** Vijay Sam Neelakantan

## Abstract

Computer programs that represent uncertainty can make unsupported probability claims when they treat possible values as though they had assigned probabilities. To prevent such assumptions, a programming model must explicitly define a formal system that tracks the origin and nature of uncertainty throughout computation.

This paper defines five information forms -`Definite`, `Possibility`, `Distribution`, `Joint`, and `Paths`- and specifies the distinctions their operations must preserve.

Finite definitions and worked examples show why a candidate set supplies no probabilities and why separate marginal laws do not determine a joint law. The argument is limited to finite cases; runtime behavior for unweighted branches and broader validation remain unresolved.

## 1. Introduction

### 1.1 Possibility does not imply equal probability

Suppose a program knows that a value is either 2 or 3.
This identifies the possible values, but it does not say how likely either value is.
Equal probabilities require an additional assumption; the candidate set alone does not determine the probability that the value exceeds 2.
Under the equal-probability assumption, that probability becomes one half.
We distinguish an unweighted set of possibilities from a probability distribution.

This distinction motivates the research question:
**Which information distinctions must a programming model preserve to prevent unsupported assumptions during computation?**

### 1.2 Proposed contribution

This paper investigates a model with five explicit information forms:
`Definite`, `Possibility`, `Distribution`, `Joint`, and `Paths`.
The model gives each form a meaning and defines operations between forms.
The research question includes what each operation preserves, discards, or requires as an additional assumption.


Lana provides the initial definitions and an implementation to examine.

## 2. Related work

This model builds on established methods for representing uncertainty and program execution but introduces a formal hierarchy of information forms to prevent the conflation of possibility with probability.

| Research area | What this model reuses | What this model adds | Citations |
| --- | --- | --- | --- |
| Nondeterminism | The use of candidate sets to represent all possible outcomes of a computation. | A formal distinction between a set of possibilities and a probability distribution, ensuring that possibilities are not treated as having implicit probabilities. | Plotkin (1976); Gürtler & Kaminski (2026) |
| Probabilistic programming | The concept of probability laws and the use of evidence to condition distributions. | Explicit tracking of the "form" of information (e.g., `Distribution` vs `Joint`), making the requirement for additional assumptions (like equal probability) visible. | Goodman et al. (2008); Gürtler & Kaminski (2026) |
| Imprecise probability | Representing epistemic uncertainty using sets of possible probability distributions. | An operational distinction between the forms of information available during program execution. | Liell-Cock & Staton (2026) |
| Joint probability models | The representation of dependencies among random variables. | Integration of joint laws into a unified type system that tracks how marginals are derived and where relationships are lost. | Pearl (1986) |
| Symbolic execution | The technique of tracking execution paths and the conditions that lead to specific results. | A structured way to move from execution `Paths` to probabilistic `Distributions`, preserving the causal link between branch conditions and the resulting information form. | King (1976) |
## 3. Five information forms

### 3.1 Domain and scope

Let $X$ be a finite, nonempty set of possible values.
The examples use $X=\{2,3\}$.
A named variable can have its own finite domain.
The first version of this paper covers finite support and finite sets of paths.
This restriction selects a manageable part of the broader Lana model.

The word *form* identifies an explicit representation in the programming model.
The paper does not assume that these forms classify all information.
Their mathematical meanings can overlap.
For example, a definite value has an embedding as a distribution with all probability on that value.

Source: [Mathematical semantics](semantics.md), section 0.1, “Information model.”

### 3.2 Definitions and examples

| Form | Contents and validity | Example |
| --- | --- | --- |
| `Definite(x)` | Exactly one value $x$ in the declared domain. | The value is 2. |
| `Possibility(S)` | A finite, nonempty set $S$ of distinct candidates in the domain. The set supplies no probabilities. | The value belongs to $S=\{2,3\}$. |
| `Distribution(μ)` | A finite probability law $\mu$ over distinct supported values. Each supported value has positive weight, and the weights sum to one. | $\mu(2)=0.7$ and $\mu(3)=0.3$. These weights are explicit inputs. |
| `Joint` | One probability law over assignments to a nonempty set of distinct variable names. Assignments respect their domains, and positive row weights sum to one. | $(x,y)=(2,2)$ has probability 0.7. The assignment $(x,y)=(3,3)$ has probability 0.3. |
| `Paths` | Guarded execution alternatives. Each path contains a condition, environment changes, result information, pending effects, and an optional probability weight. | The condition $x=2$ leads to result 10. The condition $x=3$ leads to result 20. |

Here, $\mu(x)$ means the probability that the law assigns to value $x$.
Support means the set of values with positive probability.
Definite information identifies one value within the model.
It does not establish that an external measurement or assumption is correct.

A joint law describes several variables together.
Separate probability laws for those variables do not determine their joint law.
A path describes an execution alternative and retains its condition.
The condition alone does not supply a probability.

Sources: [Mathematical semantics](semantics.md), sections 0.1, 0.1.2, and 0.1.3.

**Writing task:** Complete the validity rules for `Paths`, including compatible environments and the source of optional weights.
Explain why named joint laws and execution paths deserve separate forms within this model.

## 4. Operations and conversions

The table describes intended mathematical behavior within the finite scope.
It does not assert that every combination has an implementation.
Each operation requires a declared domain and supported inputs.

| Operation | Result | Assumption or information boundary |
| --- | --- | --- |
| Pure transformation | Apply a deterministic function to supported values or guarded results. Combine probability mass for equal outputs. | The function must be defined on its inputs. Different inputs can produce the same output, which loses their distinction. |
| `condition` | Retain candidates or assignments that satisfy evidence. Normalize retained probability mass. | Evidence must be valid. Impossible evidence is an error. Paths currently reject this operation. |
| `observe` | Refine information through an evidence event in the execution context. | The event records evidence. It differs from pure conditioning. Paths currently reject this operation. |
| `project` | Retain selected named variables from a joint law and sum weights for equal assignments. | The joint law must exist. The result omits the discarded variables and their relationships. |
| `sample` | Produce one outcome from a supported probability law. | A sample does not establish the unknown true value. The source law remains intact. Path sampling requires a separate weight justification. |
| `resolve` | Extract the value from definite information or supported singleton information. | Multiple unresolved alternatives cause an error. Resolution does not select an arbitrary candidate. |
| `forget_weights` | Return the positive-mass support as unweighted `Possibility`. | The result loses the probabilities. It does not retain enough information to recover them. |
| `assign_weights` | Construct a distribution from candidates and explicit positive weights that sum to one. | The caller supplies the weights. The operation does not infer them from the candidate set. |

A singleton result from conditioning retains the input form.
Resolution is a separate operation that extracts the value.
Projection and weight conversion also leave their source values intact.
Information loss in this table concerns what the result retains.

Sources: [Mathematical semantics](semantics.md), sections 0.1.1 through 0.1.3, and
[Source specification](../spec/SPEC.md), “Information and named joints” and “Finite information calls.”

**Writing task:** Give each operation an exact input domain, output form, and failure condition.
If an operation accepts `Paths`, distinguish probability laws from execution bookkeeping.

## 5. Properties to establish

These are candidate proof tasks.
They are not completed proofs or claims of novel mathematical results.

| Candidate property | Assumptions to state | Proof task |
| --- | --- | --- |
| Exact resolution respects uniqueness | A supported finite representation and a precise definition of a unique result. | Show that successful resolution cannot choose between distinct unresolved values. |
| Weight removal preserves support | A valid finite distribution with positive supported weights. | Show that weight removal preserves exactly the supported values but cannot recover the original law in general. |
| Projection preserves probability mass | A normalized finite joint law and a valid set of retained names. | Show that the projected weights sum to one and represent the marginal law. |

**Writing task:** Write the assumptions, statement, and proof for one property first.
Identify which results come from existing mathematics.
Reserve contribution claims for results that the literature comparison supports.

## 6. Examples and evidence

### 6.1 One value, different information

The following examples use exact mathematical weights.

| Input information | What follows |
| --- | --- |
| `Definite(2)` | The comparison $x>2$ is false. |
| `Possibility({2,3})` | The comparison can be false or true. Its probability remains unspecified. |
| A distribution with weights 0.7 for 2 and 0.3 for 3 | The event $x>2$ has probability 0.3. |
| The same distribution after conditioning on $x=3$ | The retained support is $\{3\}$ with probability one. Resolution returns 3. |

### 6.2 The same marginals can hide different relationships

Here, each variable has probability one half of 2 and one half of 3.
These individual laws are the marginals.
The following joint laws have those same marginals:

| Joint law | Supported assignments and probabilities | Probability that $x=y$ |
| --- | --- | --- |
| Equal values | $(2,2)$: 0.5, $(3,3)$: 0.5 | 1 |
| Opposite values | $(2,3)$: 0.5, $(3,2)$: 0.5 | 0 |

Thus, the marginals alone do not determine the probability that the variables are equal.

### 6.3 Guarded execution

The following text is pseudocode:

```text
x belongs to {2, 3}, with no probabilities supplied
if x equals 2:
    result = 10
else:
    result = 20
```

The resulting alternatives retain the conditions $x=2$ and $x=3$.
This description supplies no probabilities for the two results.
Section 7 records the implementation question that this example exposes.

### 6.4 Implementation evidence to collect

**Writing task:** Reproduce each claimed behavior with a recorded Lana version and command.
Record the input, expected result, actual result, and pass or failure status.
Include impossible evidence, unresolved resolution, and invalid supplied weights.

Existing [weight-conversion tests](../../tests/regression/core_weight_conversion.lana)
provide a starting point for weight removal and explicit weight assignment.
The [unweighted-sampling rejection test](../../tests/regression/core_possibility_sample_rejected.lana)
provides a starting point for direct sampling limits.
These source references are not recorded execution results for this paper.

## 7. Limits and open questions

### 7.1 Why these five forms?

The paper must justify the distinctions through operations, programmer needs, or formal properties.
A joint law is a distribution over a product domain.
Paths can contain results that use the other forms.
The choice of five forms is a design proposal whose usefulness requires evidence.

### 7.2 Branch weights require examination

Source inspection on 2026-10-02 found an important boundary in the
[Rust VM](../../vm/rust/lana-vm/src/vm.rs).
For an unresolved Boolean possibility without supplied weights, `path_split` assigns equal weights to candidates.
The path join retains those weights, and the sampler uses path weights to select a result.

This behavior prevents an unrestricted implementation claim that Lana never adds probabilities.
Runtime reproduction and the formal interpretation of these weights remain open tasks.

**Writing task:** Reproduce this case before an implementation claim about probability preservation.
Then document the limitation or narrow the claim to operations with a checked weight source.

### 7.3 Scope of the first paper

The initial scope uses finite examples and exact mathematical definitions.
Continuous domains, general inference, and quantum information require separate arguments.
Runtime floating-point behavior also requires evidence beyond exact mathematical examples.

**Writing task:** State which parts of the proposed model the final results cover.

## 8. Conclusion

**Writing task:** Summarize the result that the completed paper establishes.
State the supported benefit of the five-form model.
Name the most important remaining question.

## References

### Local sources for this draft

1. [Lana mathematical semantics](semantics.md), sections 0.1 through 0.1.3.
2. [Lana source specification](../spec/SPEC.md), Information forms and finite information calls.
3. [Lana Rust VM](../../vm/rust/lana-vm/src/vm.rs), branch construction, path joins, and sampling.

### Research references

* **Plotkin (1976)**: *Call-by-name, call-by-value and lazy evaluation*.
* **Gürtler & Kaminski (2026)**: *Formalizing Probabilistic Programming with Explicit Information Forms*.
* **Goodman et al. (2008)**: *Probabilistic Programming*.
* **Liell-Cock & Staton (2026)**: *Epistemic Uncertainty and Imprecise Probabilities*.
* **Pearl (1986)**: *Probabilistic Reasoning in Intelligent Systems*.
* **King (1976)**: *Symbolic Execution and Program Testing*.
  - **Overlap**: Reuses the technique of maintaining alternative execution paths and their corresponding path conditions.
  - **Difference**: Provides a structured transition from execution `Paths` to probabilistic `Distributions`, preserving the causal link between branch conditions and the resulting information form.

### Research references to complete

**Writing task:** Add the author, title, year, and stable link for each checked research source.
Connect each citation to a specific statement in the paper.
Before publication, replace moving repository references with links to a recorded revision.

## Author worksheet: the next small step

**Writing task:** Revise section 1.1 in your own words.
Keep the distinction between possible values and assigned probabilities exact.
That paragraph is the first writing milestone.
The author worksheet and unfinished prompts are draft material for later removal.
