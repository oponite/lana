# Lana is–ought study

Status: non-normative research and executable experiment. This document does
not change Lana's mathematical or language contracts.

## Result

The original control passes 222 study cases over three scenarios and
12 factual states. Each scenario's unconstrained reference model contains all
81 assignments of three obligation statuses to four factual states.

The expanded protection study also passes 80,351 finite cases under 32 declared
standards, with separate permission and duty results. Its definitions, incident
coding, failure checks, and counts appear in the expanded protocol below.

Lana preserves competing conclusions and can resolve agreement under supplied
rules. In the unconstrained model, the same facts admit a required action, an
optional action, and a forbidden action. Restricting that model to an explicit
rule table determines an answer; removing the restriction restores alternatives.

These results establish conditional computation in this finite model. They do
not establish a moral obligation from descriptive premises alone. They also do
not prove that every possible philosophical account or future Lana proposal
must fail. A contrary interpretation is a counterexample only if it is
admissible under the particular semantics being evaluated.

The test runner reports engineering checks separately from the philosophical
status. `philosophical_status: not_established` records the study's limited
scope; it is not an automated judge of every possible moral argument. A green
CTest result is never an `is_ought_solved` verdict.

## Reproduce the experiment

From the repository root:

```sh
cmake -S . -B build-is-ought -DCMAKE_BUILD_TYPE=Debug
cmake --build build-is-ought --parallel
ctest --test-dir build-is-ought \
  -R 'native_(is_ought_study|core_refinement_.*|decision_.*|m10_inspector_pass)' \
  --output-on-failure
git diff --check
```

The CMake Rust CLI target currently invokes Cargo's release profile even with
this CMake configuration. Record the actual build output rather than calling
the executable a Rust debug build.

To run only the study against that explicit CLI/compiler pair:

```sh
LANA_COMPILER_LABC="$PWD/build-is-ought/lana-compiler.labc" \
  python3 tests/test_is_ought.py build-is-ought/lana-rust build-is-ought/is-ought
```

The driver uses only Python's standard library. It compiles the original
`tests/regression/is_ought.lana` and expanded
`tests/regression/is_ought_protection.lana` fixtures. Requests run in fresh
processes with seed 7; the expanded study batches up to 16 definite cases per
request. Compilation keeps the existing resource limits. No LLM runs inside
the experiment or grades the answers. The fixtures and source coding are
researcher-authored and remain open to review. No test carries out the modeled
actions.

Artifacts in `build-is-ought/is-ought/`:

| File | Contents |
|---|---|
| `results.json` | Engineering result, philosophical scope, limitations, source revision, dirty-state fingerprint, executable/compiler and fixture hashes |
| `commands.json` | Commands, full inputs, expected results, exit codes, stdout and stderr |
| `models.json` | All reference assignments, explicit rule tables, and contrary witnesses for every factual state |
| `is-ought.labc` | The compiled experimental fixture |

The source fingerprint includes the tracked diff hash and hashes of untracked,
non-ignored files. The executable hashes identify what was actually run;
fingerprints alone do not prove a binary was built from the recorded source.
Use the fresh-build command and preserve its output for that claim. Results
are local evidence, not release qualification.

## Scenarios and explicit assumptions

Each scenario has the four states `(a,b) = 00, 01, 10, 11`. Actions are assumed
feasible. Factual input also carries `not_a`, which must be the complement of
`a`; this gives an explicit inconsistent-premise rejection test.

| Scenario | Meaning of `a` | Meaning of `b` | Action |
|---|---|---|---|
| Sharing | Alice holds two cookies rather than one | Bob holds one cookie rather than none | Give Bob one cookie |
| Promise | The specified promise words were spoken | The specified release words were spoken | Perform the stated task |
| Harm | The intervention prevents the modeled injury | Intervention costs one token rather than zero | Intervene |

The harm response is stipulated causal knowledge for the toy model, not a
causal inference from observational correlations. Cookie possession does not
silently include an ownership entitlement. The promise case records utterances,
not the already normative proposition that a morally binding duty exists.
Release words can be spoken even when no promise words were spoken.

The control rules are deliberately supplied normative premises:

| Scenario | First rule | Second rule |
|---|---|---|
| Sharing | Require giving if Bob has none | Require giving if Bob has none and Alice has a spare |
| Promise | Require performance if promise words were spoken and release words were not | Speech alone makes performance optional |
| Harm | Require intervention if it prevents injury | Require intervention if it prevents injury and costs nothing |

Every other case is explicitly optional under these rules. A separate control
rule forbids the action in every state. These are test assumptions, not endorsed
ethical principles. Treating an action as optional is itself a normative
assignment, not the absence of information.

The Python oracle contains literal tables in world order `00,01,10,11`:

| Scenario | First | Second |
|---|---|---|
| Sharing | required, optional, required, optional | optional, optional, required, optional |
| Promise | optional, optional, required, optional | optional, optional, optional, optional |
| Harm | optional, optional, required, required | optional, optional, required, optional |

Lana independently computes those controls with explicit conditions. The
oracle does not call the Lana evaluator to manufacture expected answers.

## The logical test

For one scenario, let `W` be its four factual states. A normative interpretation
is a function `n: W -> {required, optional, forbidden}`. The baseline admits all
`3^4 = 81` such functions. The statuses are mutually exclusive within each
interpretation. Disagreement between interpretations is retained; contradictory
constraints on one interpretation admit no models.

For a fixed factual state `w`, the baseline contains an interpretation assigning
`required` to `w` and another assigning `optional` to `w`. Both agree on all the
descriptive input. Therefore those descriptive premises do not entail
`required` **in this baseline semantics**. The driver saves witnesses for all
three statuses and checks a contrary pair through Lana for each factual state.

This is not a proof that normative interpretations must be independent of facts
in every philosophical theory. Their independence is a declared baseline
assumption. A stronger theory may restrict the interpretations. Its proponent
must state and justify the restrictions, and a challenge must respect them.

The restricted control fixes `n` to the first rule's complete table. Exactly
one interpretation remains. This verifies conditional reasoning, but it does
not derive the rule table's moral authority: that table was supplied. Removing
the table constraint restores all 81 interpretations. No models, including
the case of contradictory constraints, is an invalid study case rather than
vacuous proof of every duty.

## What each information form establishes

| Form | Executed check | Interpretation limit |
|---|---|---|
| Definite | Resolve each individual supplied rule, including prohibition | One settled output is not a certificate of moral truth |
| Possibility | Retain the candidate statuses; add a contrary prohibition | Membership records alternatives, not their moral authority |
| Distribution | Reverse 0.99/0.01 weights while retaining both outcomes | High probability does not permit exact resolution of disagreement |
| Joint | Condition a declared cookie/rule/duty law, then project duty | The authored rows already encode the normative relationship |
| Paths | Evaluate both rule branches; resolve agreement and reject disagreement | Consensus is relative to the included rules |

Equal distribution outcomes are explicitly combined before construction:
Lana requires distinct weighted support. The joint probe covers the two sharing
states where Bob has none. The other form/rule checks cover all 12 factual
states; this is not an exhaustive cross-product of all forms and all scenarios.

Seeded sampling selects a supported value, preserves the original alternatives,
and exposes sample metadata. Repeating the seed reproduces the result. Selecting
one candidate is not an entailment that it is required.

Candidate order and agent labels are swapped independently in invariance checks.
Probability weights follow their rules when reordered. These changes do not
affect the role-based conclusions. Invalid status tables, contradictory
facts, undeclared rules, invalid weights, and unknown modes/scenarios must fail
with the intended error code and diagnostic text.

The representation probe constructs three claims with a supplied `true` value:
“Alice ought to share,” “Alice ought not share,” and “Alice has ninety cookies.”
All retain their supplied value and proposition. The last description conflicts
with this probe's one-cookie input. Lana also represents the descriptive cookie
count `1 or 2` as Possibility. These checks demonstrate construction and metadata,
not a truth-checking service or a moral contradiction accepted as a theorem.

## Assumption and implementation audit

The inference commitments are explicit: use the supplied truth tables, project
the selected factual state, preserve supported alternatives, and resolve only
agreement. Consensus is never treated as a freestanding source of moral force.
Names, string labels, numeric weights, and the word `required` introduce no
additional inference rule.

The fixture records the actual result snapshot separately from annotations.
`evidence` labels the stipulated facts and the computed result; `assume` marks
the presence of supplied rules. A decision context records the specific rule
names or model restrictions. Those annotations are constructed after evaluation
and **do not certify complete automatic dependencies**. The raw snapshot is
retained so annotations cannot conceal the original provenance. Inspecting an
annotation's `modeled`/`approximate` status does not measure moral confidence.

Observed limits in this candidate:

1. The inspector exposes the number of guarded paths but not their result
   values. The study checks the two definite controls, path count, invariance,
   and resolution behavior. For distinct candidates `l,r`, it also resolves
   `(result == l) == (result != r)` to true: every path result belongs to that
   two-value set. This is an exact finite membership check, not an exported
   proof of every guard or provenance edge.
2. Direct `sample(weighted)` in the fixture's function is lowered to
   `SAMPLE_STATE_DIST` by the bundled compiler and fails with `LANA_ERR_TYPE`.
   The supported sampling check explicitly conditions on the full existing
   support before sampling, preserving both alternatives and selecting the
   Core operation. The direct call remains a separate capability probe in the
   artifacts. The driver accepts either a valid supported sample or this precise
   observed lowering failure, so it does not make the bug a required language
   behavior. Unexpected failures still fail the study.
3. The experiment supplies formal normative interpretations. It cannot
   automatically certify that their philosophical reading is correct, or that
   every value assumption in a new argument has been disclosed.

These limitations are reported without changing the compiler or VM.

## Evaluation of the proposed bridge

### Original proposal: is as Definite, ought as any information form

At a fixed modeled world, descriptive values can be definite. But a constructor
describes information structure, and a claim's content can be descriptive or
normative independently of that structure. Classifying a value does not supply
an inference rule between descriptive truth and moral obligation. The two
contrary interpretations remain available in the baseline after that
classification. The original mapping therefore does not establish the bridge
under the tested semantics.

### Application-level proposal: explicit conditional obligation

Use `Ought_S(action | facts)` as research notation for an obligation relative to
a declared, nonempty set of standards `S`. Each standard supplies a normative
status in the factual situation. Return the supported statuses and the standards
used; produce a settled status only when they agree. Uncertainty about the
standards can coexist with agreement about a particular action.

This is a formal interpretation for an application using existing Information
values and decision contexts, not a new Lana keyword or moral primitive. The
fixture implements finite instances. It requires no change to the mathematical
authority and no LIP. The interpretation remains conditional on why `S` is the
appropriate set and why its standards bind the agent.

A stronger proposal would have to supply a bridge `B` such that descriptive
premises `F` entail a moral obligation under `B`. Its review must establish that
`F` is consistent, the inference rules are sound for the stated meaning of
obligation, and `B` is independently justified rather than a concealed normative
premise or a renaming of the desired conclusion. Countermodels must satisfy
`B`; freely varying duties while ignoring `B` would not refute it. Excluding
them simply by declaring the desired moral verdict would not justify `B`.

This study supplies no such independently justified stronger bridge. Any future
proposal changing Lana's language contract must use a Draft LIP and the existing
authority order before implementation.

## Research comparison

- [Judea Pearl, *From Conditional Oughts to Qualitative Decision Theory*,
  UAI 1993](https://arxiv.org/pdf/1303.1455), introduction and section 3:
  the framework combines beliefs, causal relationships, and supplied utility
  rankings. It supports conditional action evaluation. It does not derive those
  utility rankings' moral authority from the descriptive inputs. Our explicit
  controls are analogous in their conditional scope; this fixture does not
  implement Pearl's causal calculus or claim the exact Lana design has precedent.
- [John R. Searle, *How to Derive “Ought” From “Is”*, 1964](https://www.finophd.eu/wp-content/uploads/2018/02/Searle_1964-oughtandis.pdf),
  pp. 44–46: the argument moves from an utterance to a promise, an undertaken
  obligation, and an ought, with conditions on the speech act and qualifications
  about other circumstances. Searle defends these connecting steps. Our audit
  keeps the utterance facts separate from the premise giving them binding force;
  executing that premise does not adjudicate his philosophical defense or its
  objections. The promise fixture tests this distinction, not every condition
  in his argument.

The computational evidence supports explicit conditional reasoning. Establishing
an unconditional moral obligation still requires the missing argument for the
bridge. A finite test can expose a faulty proposed derivation; successful finite
tests alone cannot establish its universal soundness.

## Expanded protection and assistance protocol

This extension operationalizes part of the proposed definition, “an action made
in awareness, respect, and acknowledgement of the human condition.” That phrase
alone does not determine consent conditions, acceptable risks, duties, or the
rules below. The experiment exposes those choices as **supplied moral
premises**. It does not certify someone's awareness, intentions, or character.

### Finite definitions

A case contains one to four people and one to four mutually exclusive, complete
action plans. `wait` is an explicit alternative in ordinary fixtures. Each
person has a causal role: source of the modeled threat, protected person, or
uninvolved person. “Source” does not independently establish moral aggression.
The model assumes a protective setting; it does not settle whether a threat is
itself justified defense, legitimate restraint, or part of reciprocal aggression.
Those questions require additional facts and standards before applying it to a
new dispute. No input named `unjustified_threat`, `necessary`, `proportionate`,
or a desired output status is accepted.

1. **Consent:** capacity, relevant information, freedom from coercion, and
   agreement covering this action must all hold; withdrawal must be absent.
   An explicit refusal conflicts with a claim of valid consent. For a protected
   person without valid consent, the emergency exception requires consent to be
   unavailable and no explicit refusal. This is a declared rule, including its
   strict refusal boundary.
2. **Adequate help:** the plan is feasible, protects the threatened interest,
   can take effect by the deadline, and meets the supplied success threshold.
   Probabilities are inputs; the evaluator does not estimate them.
3. **Proportionate protection:** nonconsensual impact on each affected person
   must fit the table below. There is no sum of benefits, harms, or lives across
   people. Irreversible impact additionally requires a grave threat with threat
   probability at least 90. Force must end by the modeled deadline.
4. **Necessary protection:** there is an active qualifying threat and the plan
   is adequate; no feasible, eligible, adequate alternative is no worse on every
   person's impact, duration, and burden to the helper while being strictly
   better on at least one. A lower-impact option that increases risk to the
   helper is not automatically preferable.
   Incomparable alternatives remain possible. This is a finite dominance test,
   conditional on a sufficiently complete option list, not a claim that every
   imaginable alternative has been considered.
5. **Permitted action:** it is feasible, does not deliberately use an uninvolved
   person as a means, and each nonzero nonconsensual impact meets the protective
   exception, proportionality, and necessity tests. Nonconsensual interference
   with an uninvolved person is excluded. Valid consent removes those protective
   exception requirements for that person; it does not remove the ban on using
   uninvolved people.
6. **Conditional duty to help:** the duty standard is enabled, the qualifying
   threat is serious or grave, and at least one permitted adequate helping plan
   is within the burden ceiling. Then an acceptable plan must provide adequate
   help. Several acceptable helping methods can each remain optional. Under the
   permission-only standard, no general helping duty is added.

| Grade | Meaning | Maximum nonconsensual impact, irreversible force forbidden | Maximum impact, irreversible force allowed |
|---|---|---:|---:|
| 0 | No threatened harm | 0 | 0 |
| 1 | Minor, reversible harm | 1 | 1 |
| 2 | Serious, recoverable harm | 2 | 2 |
| 3 | Death or permanent harm | 2 | 3 |

Impact and burden use these four categories. They are separate inputs: impact
tracks intrusion or anticipated harm to affected people; burden tracks cost or
risk to the actor. Assigning a grade is a substantive modeling choice. The
study does not claim an empirically calibrated scale, nor does respecting each
person follow merely from avoiding a utility sum.

The **32 standards** are the Cartesian product of:

- General assistance duty: disabled or enabled.
- Irreversible protective force: forbidden or conditionally permitted.
- Threat threshold: 50 or 90.
- Success threshold: 50 or 90.
- Maximum burden triggering a duty: grade 0 or grade 1.

These are competing experimental premises, not 32 discovered moral truths.
Other explicit commitments appear as `require` and `forbid` constraints over
complete plans. They are declared normative inputs, never observations.
Conflicting commitments can leave no acceptable plan: the result is `conflict`
with no action statuses and `must_help: null`, never vacuous obligation.
Necessity compares eligible alternatives before these additional commitments;
this convention can expose a conflict instead of silently discarding an
otherwise adequate alternative.

For a nonempty acceptable set, an action is **required** if every acceptable
plan chooses it, **optional** if some do, and **forbidden** if none do. `may`
means required or optional; `must` means required. `must_help` means every
acceptable plan provides adequate help. It does not identify one mandatory
method.

### Checks and failure criteria

`tests/is_ought_protection.py` implements a declarative reference model: it
enumerates Boolean plan selections and filters them against constraints. Lana
constructs the accepted set procedurally. They share the stated definitions,
but neither implementation calls the other. This checks implementation
agreement; shared conceptual mistakes remain possible. Twenty-four fictional
fixtures also have literal expected acceptable sets under `standard_25`,
independently of either algorithm's output.

The deterministic matrix includes:

1. All 32 consent combinations; all 32 threat/impact/table combinations;
   64 necessity combinations; 256 duty combinations; 144 role/consent/refusal/
   availability/impact/threat combinations; and a fixed witness that greater
   burden to the helper prevents an alternative from dominating.
2. Every acceptable-set mask for one through four alternatives, including
   empty sets; threshold endpoints and adjacent values; and the separate
   irreversible-force boundary at 89, 90, and 91.
3. Twelve fictional contrasting pairs under all 32 standards: active danger,
   free consent, refusal/incapacity, severity, lesser alternatives, timely help,
   excessive duration, burden, competing commitments, multiple helping methods,
   uninvolved people, and mistaken threat reports.
4. Renamed people, changed irrelevant context, reordered actions, and reordered
   people with their corresponding fields; removing a duty, increasing burden,
   adding an infeasible alternative, and adding an adequate lesser intervention.
5. Definite, Possibility, Distribution, Joint, and Paths representations of
   statuses computed from complete cases. This explicitly enumerates uncertain
   worlds/standards before constructing the Information law. It does not claim
   every host operation accepts unresolved structured inputs directly.
6. Exact permission/obligation separation, contradictory statuses, unanimous
   statuses, reversed weights, singleton resolution, and conditioning. Pure
   conditioning preserves the original law. The existing focused refinement
   tests separately cover observation revisions and source mutation.
7. Rejection of malformed data, invalid references, contradictory consent,
   unknown forms/actions, empty laws, invalid selections, invalid weights,
   conflicting models inside a law, and undeclared judgment fields.
8. Six compiled evaluator mutations: ignore consent, omit necessity, omit
   proportionality, declare every helping method mandatory, collapse unknown
   duty to false, or substitute a majority for unanimity. Each must compile and
   run successfully and then disagree with a recorded witness. Compilation or
   runtime failure does not count as detecting the modeled reasoning error.

Every definite result must match the reference model. Every uncertainty probe
must preserve the expected support and reject resolution when alternatives
remain. Paths do not export support, so the fixture uses an exact membership
assertion and tests resolution and form. Distribution comparison currently
produces Boolean possibilities rather than a weighted Boolean law; the status
law's original weights are checked separately. Joint rows contain status and
its derived `may`/`must` values before projection. Annotations are explicit
records, not a certification that Lana automatically discovered every premise.

### Historical incident sensitivity cases

`tests/fixtures/is_ought_incidents.json` contains source URLs, retrieval dates,
HTML hashes, paragraph locators, decision points, coding notes, complete model
inputs, and alternative domains. The tests require coverage of every top-level
world field and a matching alternative domain for every field labeled unknown.
This checks ledger consistency, not the correctness of the researcher's reading.
No web access is required to rerun the experiment.

- **Wesley Autrey, 2007-01-02:** decision after failed attempts to return the
  fallen man to the platform, before holding him between the rails. The
  [Carnegie Hero Fund account](https://www.carnegiehero.org/hero-search/wesley-james-autrey/)
  supports the approaching train and failed earlier attempts. Unknown consent,
  intervention impact, forecast success, burden, and a hypothetical lesser
  alternative remain explicit alternatives.
- **John Catania, 2022-11-22:** decision after an ignored warning and another
  blade strike, before pushing the attacker away. The
  [Carnegie Hero Fund account](https://www.carnegiehero.org/carnegie-medal-presentation-to-john-catania/)
  provides the sequence. Injury outcomes do not determine prior risk or success
  probabilities. The modeled lesser option is untried, not the already failed
  warning.
- **Jean Charles de Menezes, 2005-07-22:** decision immediately before the
  shooting. The primary ECHR judgment,
  [*Armani Da Silva v. United Kingdom*, paragraphs 29–41, 52, 66, and 127](https://lagen.nu/dom/echr/001-161975),
  supplies the record, accessed through its lagen.nu reproduction. Earlier
  identification and communication are uncertain; later evidence establishes
  the absence of the hypothesized terrorist threat. The judgment's procedural
  legal conclusion is not used as a moral answer label.

Each incident has separate `available_then` and `later_record` views. The
former is a retrospective reconstruction of available evidence, not direct
access to the actor's mental state. In the two rescues, later successful outcomes
are retained as context and do not collapse unknown forecasts. In the mistaken
shooting, the later reconstruction removes the hypothesized threat; that fact
must not be imported into the earlier view.

Values 0, 50, and 90 stand for probability intervals `[0,50)`, `[50,90)`, and
`[90,100]`. For this evaluator they cover all different threshold outcomes.
They are **not historical probability estimates**, and interpretations have no
probability weights. Fixed timing, refusal, roles, omitted people, and the
bounded alternative menu are disclosed modeling conventions, not observations.
The sensitivity sweep is exhaustive only over the listed domains. A settled
answer means agreement over those domains and standards, never certainty about
the complete incident or the actor's blameworthiness.

### Expanded artifacts and interpretation

Alongside the original control artifacts, the same output directory contains:

| File | Contents |
|---|---|
| `protection.json` | Counts, policies, goldens, uncertainty snapshots, mutation witnesses, incident supports and conditional status |
| `protection-cases.jsonl` | Each complete case, independent expected result, and actual Lana result |
| `protection.labc` | Compiled evaluator, whose hash is recorded |
| Six named `.lana` / `.labc` pairs | Reproducible faulty evaluator copies used by mutation checks |
| `commands.json` | Commands, exit codes, stdout and stderr for both studies |

`results.json` keeps the original 222-case count and adds a separate protection
summary. Hashes identify the CLI/compiler and both new evaluator sources, the
compiled evaluator, and incident fixture. All artifacts are machine-local
working-tree evidence; they do not establish a release or publication.

Passing this study supports the claim: **Lana can compute and preserve the
consequences of explicitly chosen protection and assistance rules, including
uncertainty about whether help is permitted or required.** The study can falsify
specific implementation claims and display the assumptions responsible for a
conclusion. It cannot establish that its moral premises are binding, that they
fully define the human condition, or that descriptive facts alone entail them.

### Recorded expanded results

The expanded evaluator passes **80,351 complete finite cases**, comprising 5,855
synthetic, boundary, and invariance cases plus 74,496 incident interpretations
across the 32 standards. These are deterministic grid points, not independent
observations or 74,496 historical incidents. There are additionally **27
Information probes, 28 invalid-input checks, and six detected executable
mutants**. The original 222 control cases remain separate and pass.

| Incident | Evidence view | Complete interpretations before the 32-standard sweep | Supported action statuses across the sweep |
|---|---|---:|---|
| Autrey | Available then | 864 | Forbidden, optional, required |
| Autrey | Later record | 864 | Forbidden, optional, required |
| Catania | Available then | 216 | Forbidden, optional, required |
| Catania | Later record | 216 | Forbidden, optional, required |
| de Menezes | Available then | 144 | Forbidden, optional, required |
| de Menezes | Later record | 24 | Forbidden |

All six historical views have nonempty acceptable sets in the enumerated cases;
synthetic conflicting-commitment and empty-set cases exercise conflict handling
separately. The historical rescues remain unsettled under the declared unknowns.
The later mistaken-threat reconstruction rules out the modeled nonconsensual
shooting under every tested standard. This is a conditional result about the
model inputs, not a retrospective assessment of what anyone should have known.

Qualification on 2026-09-25: fresh CMake configure/build passed; the focused six
CTests passed; the final full CTest run passed **74/74**, including the final
helper-burden comparison, in 102.91 seconds. `git diff --check` passed. The
original and expanded study ran in 65.52 seconds within that final suite.
Commands, output, timings, and before/after candidate hashes are saved under
`build-is-ought/qualification-v2/`. The final full run supersedes the earlier
focused run for the helper-burden change. No release qualification or publication
is claimed.
