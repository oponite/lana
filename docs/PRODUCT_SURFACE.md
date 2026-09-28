# Supported Product Surface (with limits)

## Information and State

Use Information when a value can have several possible results. Use `condition`
to narrow a value without changing its source. Use `observe` to record evidence
on the source, then `resolve` when only one result remains:

```lana
let source = information(possibility([1, 2, 3]));
let narrowed = condition(source, possibility([2, 3]));
let observed = observe(source, 3);
assert(resolve(source) == 3, "observed result");
assert(inspect_information(narrowed).remaining_alternatives == 2, "pure refinement");
```

| Operation | Example | Limits and failures |
| --- | --- | --- |
| Construct Information and calculate with it | [Reactive values](../tests/regression/m6_reactive_pass.lana) | Combining unrelated uncertain values requires a declared relationship. Separate possibilities do not imply independence. |
| Condition, observe, and resolve | [Finite refinement](../tests/regression/core_refinement_finite.lana) | Evidence must be a definite value, a possibility subset, or a named Joint map. Evidence with no matching result fails. Paths reject refinement. `resolve` fails when several results remain. |
| Construct a weighted distribution and sample it | [Weighted sampling](../tests/regression/core_distribution.lana) | Candidates must be distinct, with finite positive weights that total one. Empty support fails. Unweighted Possibility cannot be sampled. `sample_value` explicitly unwraps a sample. |
| Refine and project a Joint | [Named evidence](../tests/regression/core_refinement_map.lana) | Evidence names declared variables and definite values. Projection needs a Joint that supports exact operations. Opaque laws fail instead of estimating an answer. |
| Inspect evidence, assumptions, and derivations | [Information inspection](../tests/regression/information.lana) | Inspection reports recorded evidence. An assumption label does not prove a relationship or authorize an action. |
| Capture Information with `snapshot(info)` | [Detached snapshot](../tests/regression/information_snapshot_pass.lana) | Later observations do not change the snapshot. Nested arrays and maps reject mutation. Effectful handles cannot be captured. A snapshot does not resolve uncertainty. |
| Construct, measure, append, and transform State | [General example](../examples/general.lana), [Belief example](../examples/belief.lana) | Invalid probabilities and dispositions fail. State distributions do not automatically convert to Core distributions. |
| Mix States | [State mixture](../tests/regression/mix_pass.lana) | Invalid weights and unsupported input types fail. |
| Measure in a named basis | [Concrete measurement](../tests/regression/measure_canonical.lana) | Concrete States support computational, x, and y bases. A qualified State-distribution measurement supports sampling only. Probability or distribution requests fail. |
| Inspect a State distribution | [Distribution inspection](../tests/regression/inspect_state_dist.lana) | Inspection shows its structure. It does not compute every possible exact inference. |

For approximate State-distribution measurement, use `estimate_measure` with an
explicit positive integer sample count. It returns an estimate without a confidence interval.
The [measurement contract](../spec/SPEC.md#basis-aware-measurement) defines exact and approximate operations.

## Decisions and execution

Recommendations are advisory. Sending a webhook requires a separate execution
capability and host-issued authorization for the unchanged plan.

| Operation | Example | Limits and failures |
| --- | --- | --- |
| Compare decision recommendations | [Recommendation modes](../tests/regression/decision_recommendation_pass.lana) | Mode names do not sample or convert the input. A recommendation does not execute an action. |
| Calculate the value of more information | [Observation choice](../examples/brain/decision.lana), [Policy examples](../tests/regression/policy_library_pass.lana) | Missing relationships leave the calculation unscorable. Invalid probabilities and utility tables fail. |
| Review context, outcomes, and alternatives | [Decision review](../tests/regression/decision_review_pass.lana) | Missing context or categories are reported. An observed outcome does not establish that the original decision was good. |
| Store policies, ledger entries, and claims | [Sensor fusion application](../examples/reference-apps/sensor_fusion.lana) | Invalid policy data, closed stores, conflicting revisions, and unverifiable claims fail. Parsing a signed claim does not establish trust. |
| Save and release future messages | [Time and event messages](../tests/conformance/durable/future_messages.lana) | The application must call `check`; Lana has no notification daemon. Ready messages stay local until acknowledged or cancelled. After uncertain commit I/O, reopen the store and inspect the message ID before retrying. |
| Create a webhook plan | [Relative path and payload](../tests/regression/execution_plan_pass.lana) | Absolute URLs, query strings, fragments, control characters, and unresolved payloads fail. |
| Authorize and send a webhook | [Authorized execution](../tests/regression/execution_live_success_pass.lana) | Requires host configuration. A changed plan, revoked token, duplicate action ID, or unrelated staged store write blocks sending. Redirects fail. A timeout returns `Unknown`; Lana does not retry automatically. |

The future-message example requires `LANA_FUTURE_STORE` and
`LANA_FUTURE_PHASE`; its phases are `save`, `check`, `settle`, and `verify`.
The [future-message contract](../spec/SPEC.md#future-messages) describes application calls.
The [execution contract](../spec/SPEC.md#execution-surface) describes host setup and receipt outcomes.
Information inspection, Decisions, and execution receipts retain structured
status and evidence. Their record format is separate from the bridge protocol.

## Datasets

Selection requires definite answers. Stored uncertain cells require immutable
snapshots so later observations cannot silently change historical rows.

| Operation | Example | Limits and failures |
| --- | --- | --- |
| Filter, sort, group, and join | [Definite selection](../tests/regression/dataset_definite_pass.lana) | Filters must return definite Booleans. Sort, group, and join keys must be definite, including nested values. Unresolved results or keys fail. |
| Save uncertain cells and read history | [Updates and historical reads](../tests/regression/dataset_uncertain_history.lana) | The example requires a store path and phase argument. Use `snapshot(info)` for uncertain source cells. Aggregates preserve declared shared relationships. Live Information, unrelated aggregates, failed plans, stale revisions, and conflicting retries publish no update. |

The [dataset history contract](../spec/SPEC.md#dataset-history-calls-all-six-calls-implemented)
describes source registration, atomic updates, and historical reads.

## Brain

The [Brain workshop](../examples/brain/README.md) creates, trains, saves, reloads,
and queries a small local Brain. It includes typed memory, aliases, and forecasts:

```bash
LANA_BIN=target/lana/bin/lana examples/brain/run.sh
```

| Operation | Usage | Limits and failures |
| --- | --- | --- |
| Create, train, fit, evaluate, save, reload, and chat | [Workshop instructions](../examples/brain/README.md) | Dense layers use ReLU or GELU. The bridge accepts WordLevel tokenizers with no pre-tokenizer or WhitespaceSplit, and no other processing. Non-finite training and malformed files fail. Fit saves its best validation checkpoint. Unix SIGINT/SIGTERM cancels training and preserves the saved model. |
| Store typed memory and ask grounded questions | [Workshop memory examples](../examples/brain/workshop.py), [Memory commands](../spec/SPEC.md#brain-typed-memory) | Memory supports definite values, finite possibilities, distributions, and named Joints. Impossible evidence and corrupt history fail. Grounded questions require explicit aliases. Missing, unresolved, or ambiguous aliases return no exact answer and make no write. |
| Retrieve related saved evidence | [Semantic retrieval commands](../spec/SPEC.md#optional-brain-semantic-retrieval) | Semantic retrieval is opt-in. Exact answers require calibration and a passing held-out gate. Weak, unknown, or conflicting matches return no exact answer. A changed Brain, tokenizer, or corpus requires an index rebuild. |
| Select evidence for working context | [Evidence selector commands](../spec/SPEC.md#optional-brain-evidence-selector) | The selector requires an index and separate training data. It preserves original references, exact-target evidence, and forecast evidence. Failed validation keeps the active model. Stale or inactive models fail. |

After saving a Brain and tokenizer, build and query an index with these commands.
`FILE`, `TOKENIZER`, and `QUESTION` are placeholders:

```bash
lana brain index FILE TOKENIZER
lana brain chat FILE TOKENIZER QUESTION --semantic
```

An index without calibration cannot return exact semantic answers. To calibrate,
add `--calibrate DEVELOPMENT.jsonl HOLDOUT.jsonl` to the index command.
The [selector contract](../spec/SPEC.md#optional-brain-evidence-selector)
describes training and the `--selector MODEL` query option.

## Objects

Use `value` for an immutable snapshot and `class` for mutable identity within a task.
Export selected fields into a public value before JSON serialization:

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

The [complete export example](../tests/regression/object_snapshot.lana) also reloads
JSON and constructs a fresh identity. Reloaded JSON contains ordinary maps.
Export rejects private value fields, class references, and unsupported JSON
payloads. Export never measures or resolves uncertainty automatically.

Fields and methods require explicit visibility. Generic object declarations,
generic object methods, and general-purpose overload families are unsupported.
Member navigation and rename work within a module; cross-module member navigation
and rename are unavailable. Collection has no maximum-pause guarantee and remains
subject to VM memory and instruction limits.

The [object contract](../spec/SPEC.md#additive-object-source-contract) defines
visibility, interfaces, `copies`, `replace`, methods, and task transfer.

## Source packages

The [source package guide](source-packages.md) describes explicit `package add`,
`package pack`, and offline imports from locked dependencies. Missing cached
packages or changed package bytes fail instead of silently fetching replacements.
