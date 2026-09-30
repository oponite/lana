# Supported Product Surface

This guide summarizes Lana's supported capabilities and their essential limits.
The [source specification](spec/SPEC.md) defines API contracts and input validation.

Module aliases in the tables correspond to these imports:

```lana
import "std/core" as info;
import "std/decision" as decision;
import "std/execution" as execution;
import "std/future_messages" as messages;
```

## Information and State

Information represents definite values, unweighted possibilities, weighted distributions, and declared relationships. Inspection and derivation records expose status, evidence, and assumptions.

```lana
let source = information(possibility([1, 2, 3]));
let narrowed = condition(source, possibility([2, 3]));
let observed = observe(source, 3);
assert(resolve(source) == 3, "observed result");
assert(inspect_information(narrowed).remaining_alternatives == 2, "pure refinement");
```

`condition` creates a derived value and leaves the source unchanged.
`observe` records evidence on the live source. In this example, `narrowed`
still contains two alternatives after the observation.
`resolve` requires one remaining result. Sampling does not establish certainty.
`snapshot` captures immutable data without a live subscription.

Separate possibilities do not imply independence. Operations preserve declared shared dependencies. State distributions do not automatically become Core distributions.
Opaque support does not promise arbitrary exact inference.

State construction, mixtures, lazy distributions, transforms, and measurement support belief calculations. Transforms replace their targets.
Computational-basis measurement supports exact probability and distribution results.
Other bases support exact results for concrete States, but only sampling for State distributions.
`estimate_measure` uses an explicit positive integer sample count and returns no confidence interval.


| Operation | What it does | Output |
| --- | --- | --- |
| `information(value)`<br>`possibility(values)`<br>`info.distribution(rows)` | Declares definite values, alternatives, or weights. `information` creates a live root. | Information, Possibility, or Distribution |
| `condition(info, evidence)` | `condition` narrows a derived value without changing its source. | Refined Information |
| `observe(info, evidence)` | `observe` narrows the live source and records evidence. | Refined Information |
| `project(joint, names)` | `project` selects named variables from a Joint and computes their exact marginal. | Projected Joint |
| `sample(info)`<br>`sample(state_dist)` | `sample` draws from supported weighted Information or a State distribution. | `Sample<T>` or concrete State |
| `sample_value(sample)` | `sample_value` extracts the draw without establishing certainty. | Concrete sampled value |
| `resolve(info)` | `resolve` extracts the only remaining result. | Concrete value |
| `snapshot(value)` | `snapshot` captures current data without a live subscription. | Immutable captured value |
| `inspect_information(value)`<br>`lana inspect FILE --format json` | Reads current status and support without changing the value. | Inspection record or CLI JSON |
| `evidence(value, label)`<br>`assume(value, label)` | `evidence` and `assume` attach evidence or assumption references. | Annotated value |
| `derivation(value)`<br>`explain(value)` | `derivation` reads provenance. `explain` describes the value. | Derivation record or explanation string |
| `a + b` | Arithmetic preserves declared shared dependencies. | Value or lifted Information |
| `state(p: p, d: d)`<br>`state(p: p, d_re: re, d_im: im)` | `state` constructs a belief from probability and disposition. | Concrete State |
| `append(a, b)`<br>`mix(a, b, weight)` | `append` builds a lazy distribution. `mix` computes a weighted mixture. | State distribution (`append`) or concrete State (`mix`) |
| `measure(value, result: mode)`<br>`measure(value, basis: name, result: mode)` | `measure` returns an exact result or draws a bit, according to the mode and supported basis. | Probability, distribution, or sampled bit |
| `estimate_measure dist in x as probability with samples: N` | `estimate_measure` uses an explicit sample count. | Estimated probability without a confidence interval |
| `transform target with invert();`<br>`transform target with neutralize();` | `invert` or `neutralize` replaces the target. | Updated State or State distribution, with no return value |

References: [Information](spec/SPEC.md#information-and-named-joints), [measurement](spec/SPEC.md#basis-aware-measurement), and [State example](../examples/basic-programs/belief.lana).

## Decisions

The `std/decision` module provides advisory recommendations, observation choices, decision context, outcome reviews, and deliberation costs.
Recommendation modes label inputs. They do not convert inputs or sample them.
Decision quality and outcome quality remain separate.

```bash
target/lana/bin/lana run examples/basic-programs/decision.lana
```

Recommendations do not authorize execution.

### Advisory decisions

| Decision operation | Advisory result | Evaluation and authority boundary | Invalid decision inputs |
| --- | --- | --- | --- |
| `decision.recommend_information(policy, input, evaluation)` ([reference](../tests/regression/decision_recommendation_pass.lana)) | Advisory decision map | Evaluates policy and labels Information mode. Does not authorize execution | Invalid policy or evaluation data |
| `decision.recommend_definite(policy, input, evaluation)` | Advisory decision map | Labels definite mode. Does not convert the input or authorize execution | Invalid policy or evaluation data |
| `decision.recommend_sampled(policy, input, evaluation)` | Advisory decision map | Labels sampled mode. Does not sample or authorize execution | Invalid policy or evaluation data |
| `decision.value_of_information(prior, candidates, actions, utility, costs)` ([reference](../examples/basic-programs/decision.lana)) | Baseline, ranked candidates, recommendations | Scores explicit relationships. Pure advice, without execution | Invalid probabilities, duplicate names, missing utility pairs, non-finite or negative costs |
| `decision.decision_context(alternatives, utility, evidence_ids, provenance_ids, assumptions, evaluated_at, recommendation)` | Context map | Captures decision inputs without persistence or execution | Wrong input types; non-finite evaluation time |
| `decision.review_outcome(context, outcome)` ([reference](../tests/regression/decision_review_pass.lana)) | Decision and outcome quality map | Reviews decision quality separately from outcome quality. Pure advice | Inputs other than maps; malformed context fields |
| `decision.value_of_deliberation(value, direct_cost, delay_cost)` | Costs and net-value map | Subtracts direct and delay costs. Pure advice | Non-finite inputs; negative costs |
| `decision.validate_alternative_set(alternatives)` | Missing categories and completeness | Reports missing alternative categories without execution | Malformed alternatives, IDs, or tags |

### Policy records and ledger

Policy records and ledger entries use the open store. `policy_store_decision` stages a write. `store_commit()` makes that write durable.

| Record operation | Record result | Policy evaluation or store change | Validation and store failures |
| --- | --- | --- | --- |
| `policy_evaluate(policy, input, evaluation)` ([reference](../tests/regression/policy_library_pass.lana)) | Host-issued decision record | Evaluates policy; does not send | Invalid policy or evaluation data |
| `policy_store_decision(decision)` | `null` | Stages decision in the open store | Closed store; malformed decision |
| `ledger_append(event)` ([reference](../tests/conformance/durable/durable_pipeline.lana)) | Stored event map | Commits a ledger entry | Closed store; malformed event; conflicting revision |
| `ledger_query(query)` | Event array | Reads the open store | Closed store; malformed query |

Reference: [decision contract](spec/SPEC.md#decision-surface).

## Execution

A webhook requires a host-configured capability and an unchanged host-issued Authorize decision bound to the exact plan.
Execution commits a Pending receipt before one HTTPS POST attempt.
Receipts report `Succeeded`, `Failed`, or `Unknown`. Timeouts can mean `Unknown`.
Execution does not automatically retry or promise exactly-once remote delivery.
Signed claims require signature, issuer, scope, lifecycle, and validity checks.
Parsing a claim does not establish trust.

Future-message transitions and execution reject unrelated staged store writes.
If a commit outcome is uncertain, reopen the store and inspect the stable ID before retrying.

### Authorized webhook execution

| Execution operation | Plan, authority, or receipt | Authorization and send boundary | Invalid plans or execution failures |
| --- | --- | --- | --- |
| `execution.plan_webhook(path, payload)` ([reference](../tests/regression/execution_plan_pass.lana)) | Webhook plan map | No capability required to plan. Sends no request | Absolute URL, query, fragment, control characters, or unresolved payload |
| `execution.capability()` | Opaque execution capability | Obtains the opaque capability from host configuration. Sends no request | Missing or invalid host configuration |
| `execution.authorize(capability, decision, plan)` | Opaque authorization token | Requires the unchanged host-issued Authorize decision. Binds the capability and exact plan without sending | Source-created or changed decision; decision without host-issued authorization |
| `execution.execute(capability, authorization, plan)` ([reference](../tests/regression/execution_live_success_pass.lana)) | `Succeeded`, `Failed`, or `Unknown` receipt | Requires matching authorization. Commits Pending before one HTTPS POST attempt | Revoked/mismatched authorization; duplicate ID; unrelated staged writes; redirects |

### Future-message inbox

The `std/future_messages` module maintains a durable local inbox.
The application calls `messages.check`. No daemon sends notifications.
Receiving a message does not acknowledge it.
Identical message replays do not create duplicate transitions.

| Inbox operation | Inbox result | Durable transition and replay behavior | Invalid messages or transitions |
| --- | --- | --- | --- |
| `messages.create(record)` ([reference](../tests/conformance/durable/future_messages.lana)) | Message record | Commits a pending message. Identical replay creates no new transition | Invalid ID, payload, condition, or bounds; conflicting ID reuse |
| `messages.inspect(id)` | Message record | Reads the stored state. No transition | Missing or malformed ID; corrupt record |
| `messages.check(context)` | `{released_ids, pending}` | Commits matching pending messages as ready. Repeated checks create no duplicate transition | Invalid context/schema; incompatible comparison types |
| `messages.receive(after_id, limit)` | Ready-message array | Reads a page of ready messages. Does not acknowledge them | Invalid cursor; limit outside `1–100` |
| `messages.acknowledge(id)` | Message record | Commits acknowledgment. Identical replay creates no new transition | Invalid ID; message not ready or acknowledged |
| `messages.cancel(id)` | Message record | Commits cancellation. Identical replay creates no new transition | Invalid ID; already acknowledged message |

References: [execution](spec/SPEC.md#execution-surface) and [future messages](spec/SPEC.md#future-messages).

## Datasets

Datasets support deferred filters, sorting, grouping, joins, and aggregation.
`materialize` runs a query plan.

```lana
fn keep(row) { return row.key > 1; }
let rows = dataset([{key: 1}, {key: 2}]);
assert(array_length(materialize(filter(rows, keep))) == 1, "definite filter");
```

Filters require definite Booleans. Sorting, grouping, and joining require resolved keys, including nested values. Selection never samples or silently resolves uncertainty.
Aggregation preserves declared relationships between uncertain inputs.

Durable datasets support source revisions, atomic updates, historical snapshots, and captured evidence for included and excluded rows.
Stored uncertain cells require `snapshot(info)` so later observations cannot change historical rows.
Shared captured laws retain their relationship across save and reopen.
Identical marginals do not establish a shared relationship.
Stale revisions and conflicting retries fail.


### Query plans and definite selection

| Dataset operation | Dataset or plan result | Evaluation and uncertainty rule | Invalid rows, keys, or plans |
| --- | --- | --- | --- |
| `dataset(rows)` | Dataset | Constructs a dataset from supported rows | Unsupported rows |
| `filter(data, predicate)` ([reference](../tests/regression/dataset_definite_pass.lana)) | Dataset plan | Defers selection. The predicate must return a definite Boolean | Predicate result other than a definite Boolean |
| `sort(data, key)` | Dataset plan | Defers sorting. Keys must be resolved, including nested values | Unresolved keys, including nested values |
| `group_by(data, key)` | Dataset plan | Defers grouping. Keys must be resolved, including nested values | Unresolved keys, including nested values |
| `join(left, right, key)` | Dataset plan | Defers row matching. Keys must be resolved, including nested values | Unresolved keys, including nested values |
| `aggregate(groups, specification)` | Dataset plan | Defers aggregation and retains declared relationships between uncertain inputs | Invalid aggregation; unrelated uncertain inputs |
| `materialize(plan)` | Row array | Runs the plan. Selection and keys must be definite | Invalid plan; unresolved selection/key; exhausted limits |

### Durable revisions and evidence

| History operation | Revision or historical result | Persistence and evidence contract | Revision conflicts or store failures |
| --- | --- | --- | --- |
| `dataset_source(source_id)` | `null` | Commits an empty source. Identical replay is idempotent | Invalid ID; closed or dirty store |
| `dataset_query(id, source_ids, plan_name, calculation_version)` | `null` | Registers a pure plan and initial snapshot. A matching bind after reopen is read-only | Missing sources; effectful plan; changed plan under the same version |
| `dataset_apply(source_id, expected_revision, batch_id, changes)` ([reference](../tests/regression/dataset_uncertain_history.lana)) | Published revision number | Atomically commits source and dependent query revisions. Uncertain cells require immutable snapshots | Stale revision; conflicting retry; live Information cells; failed query plan |
| `dataset_snapshot(query_id, revision)` | Immutable snapshot map | Reads immutable saved history. A null revision selects the current revision | Missing query/revision; closed or dirty store; corrupt snapshot |
| `dataset_evidence(snapshot, output_id)` | Row derivation and source IDs | Reads the captured derivation and source IDs for an output row | Invalid snapshot; unknown output ID |
| `dataset_exclusions(snapshot)` | Excluded-source decisions | Reads captured decisions about excluded sources | Invalid snapshot |

References: [selection](spec/SPEC.md#dataset-selection), [history](spec/SPEC.md#dataset-history-calls-all-six-calls-implemented), and [history example](../tests/regression/dataset_uncertain_history.lana).

## Objects

`value` provides immutable snapshots. `class` provides mutable identity within a task.
Fields and methods require explicit visibility.

```lana
value Export {
    public count: number;
}
class Counter {
    private mutable count: number;
    public fn init(self, count: number) { self.count = count; }
    public fn export(self) -> Export { return Export(self.count); }
}
let counter = new Counter(7);
let exported = counter.export();
print(json_stringify(exported));
```

JSON export rejects private value fields, class references, and unsupported payloads.
Export never measures or resolves uncertainty. Reloaded JSON contains ordinary maps.
Generic object declarations, generic methods, and general-purpose overload families are unsupported.
Member navigation and rename work within a module, but not across modules.
Collection has no maximum-pause guarantee and remains subject to VM resource limits.

References: [object contract](spec/SPEC.md#additive-object-source-contract) and
[export and reload example](../tests/regression/object_snapshot.lana).

## Source packages

Source packages support explicit installation, archive creation, and offline imports from locked dependencies.

```bash
lana package add owner/repo@1.2.3
lana build
```

`owner/repo@1.2.3` is a placeholder for a published library and version.
Installation records exact versions and checksums in `lana.lock`.
Missing cached packages or changed package bytes fail instead of silently fetching replacements.
Package archives reject local-path dependencies and build hooks.

References: [package guide](dev/SOURCE_PACKAGES.md) and
[package contract](spec/SPEC.md#hosted-source-packages).
