# Runnable operations and failure boundaries

Use the Rust CLI: `build/lana-rust run <source>`. Set `LANA_STDLIB_DIR` to
`stdlib` when running an uninstalled CLI. The table links actual source or a
runnable boundary test; it does not promise unimplemented API names.

| Surface | Example/check | Failure boundary |
| --- | --- | --- |
| Core information construction, observation, arithmetic | [reactive example](../tests/regression/m6_reactive_pass.lana) | Combining unrelated uncertainty requires an explicit relationship. |
| Possibility, finite Distribution, Joint, Paths; project, condition, observe, sample, resolve, inspect | `cargo test -p lana-vm v5_core_operation_matrix` | The 30-cell test checks allowed operations and exact rejection codes. Refinement/project dispatch requires Joint; reactive roots additionally support observation. Possibility sampling is unsupported in v5; non-singleton resolution fails. |
| Weighted distribution and sample unwrapping | [distribution](../tests/regression/core_distribution.lana) | Empty, duplicate, non-finite, nonpositive, or non-normalized weights fail. |
| Joint map refinement | [condition and observe](../tests/regression/core_refinement_map.lana) | Impossible evidence fails; exact support is required. |
| Evidence, assumptions, derivation, explanation | [information](../tests/regression/information.lana); Rust provenance tests | Labels do not invent relationships or authorize effects. |
| State construction, measure, append, transform | [general](../examples/general.lana), [belief](../examples/belief.lana) | Invalid probabilities/dispositions fail. State distributions do not implicitly become Core distributions. |
| State mixture | [mix](../tests/regression/mix_pass.lana) | Invalid weights or unsupported inputs fail. |
| State basis measurement and estimation | `cargo test -p lana-vm measure`; [source contract](../spec/SPEC.md#basis-aware-measurement) | Exact non-computational measurement of a State distribution is unsupported; approximation requires explicit positive sample count. |
| State distribution inspection | [inspection](../tests/regression/inspect_state_dist.lana) | Inspection exposes lazy structure; it does not establish arbitrary exact inference. |
| Decision recommendation modes | [recommendations](../tests/regression/decision_recommendation_pass.lana) | Mode labels do not sample or convert input; recommendations remain advisory. |
| Decision value of information | [policy library](../tests/regression/policy_library_pass.lana) | Missing relationships remain unscorable; malformed probabilities and utility tables fail. |
| Decision context, outcome review, deliberation, alternatives | [review](../tests/regression/decision_review_pass.lana) | Missing context/categories are reported; observed outcome is separate from decision quality. |
| Durable policy, ledger, claims | `cargo test -p lana-runtime`; [reference applications](../examples/reference-apps/sensor_fusion.lana) | Invalid policy data, closed stores, conflicting revisions and unverifiable claims fail. |
| Execution plan | [webhook plan](../tests/regression/execution_plan_pass.lana) | Absolute URLs, query/fragment components, control characters, and unresolved payloads fail. |
| Execution capability, authorize, execute | [trusted local HTTPS check](../tests/conformance/run_execution_live.sh) | Wrong digest, revoked token, duplicate ID, or staged store writes cannot send. Redirects fail; timeout is Unknown. No automatic retries. |
| Dataset filter, sort, group, join | [definite selection](../tests/regression/dataset_definite_pass.lana) | Unresolved predicates and keys fail explicitly, including nested uncertainty. |
| Brain create, train, evaluate, package, reload, chat, inspect | [workshop](../examples/brain/README.md) | Fixed reference architecture; strict tokenizer subset; non-finite training and malformed files fail. Typed Information memory is pending. |
| Rust record FFI | [C consumer](../tests/unit/test_rust_records.c) | Only owned JSON parse/serialize/free subset; no complete Decision or Execution FFI. |

Normal development uses `lana new`, `check`, `build`, `run`, and `test`.
Registered CTest cases cover project generation, LSP protocol/roundtrip and
source debugger behavior. See [support boundaries](support-matrix.md) for the
Rust/C distinction. Passing a focused operation check is not release qualification.
