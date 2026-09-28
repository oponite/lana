# Brain workshop

This is the complete local reference workflow. It uses a WordLevel Hugging
Face `tokenizer.json`, a three-token CPU Brain, and no network access.

```bash
LANA_BIN=target/lana/bin/lana examples/brain/run.sh
```

The command creates and trains a Brain, stores a fact and a typed finite root,
observes evidence, and adds exact question aliases. It exports and reloads the
strict package, then checks exact, unsupported, and ambiguous answers against
[questions.jsonl](questions.jsonl). The default run removes its temporary directory.
Pass a file path to keep its outputs; its `.hf` package destination must be absent.
Failed training or bridge validation preserves the prior Brain file. Save uses
atomic replacement; a sync failure after replacement means durability is
uncertain and requires inspection before retrying.

Use `--report report.json` to save artifact and fixture digests, parameter
digest, memory revisions, each answer's evidence, and checked outcome counts.
A wrong answer or reload mismatch fails without replacing a prior report.
Next-token loss is reported separately from answer correctness. This fixture
checks explicit evidence lookup; it does not prove general intelligence.
The report also compares ordinary and grounded CLI calls from copies of the
same starting Brain file. It records per-call time and equal parameter digests.
These workshop questions are known to the setup, so the comparison is labeled
`held_out: false` and makes no improvement claim.
Pass `--held-out` to use the pinned [heldout_questions.jsonl](heldout_questions.jsonl)
fixture. Its question strings are absent from setup, while the target aliases
and facts are fixed before evaluation. The report scores ordinary and grounded
answers separately. Its weather case also runs the declared
`std/decision.value_of_information` calculation and compares realized utility
under one fixed answer-to-action policy. Any improvement claim is limited to
these fixed targets, unseen question strings, and one declared decision;
it is not a generalization or autonomous-action claim.
The weather case declares a two-label forecast and a later observed outcome;
its Brier score is reported as `forecast_scores`, separate from answer counts
and next-token loss.

The bridge supports WordLevel with no pre-tokenizer or WhitespaceSplit and no
other tokenizer processing. Unsupported pipelines fail explicitly. Typed memory
supports Definite, finite Possibility, Distribution, and named Joint laws.
Grounded chat requires an explicit full-question alias and does not guess keys.
`brain forecast add` records caller-declared finite probabilities and evidence;
`brain forecast score` records an observed outcome and its Brier score. Neither
command infers probabilities from token logits. Run
`target/lana/bin/lana run examples/brain/decision.lana` to check an advisory observation
choice using the existing `std/decision.value_of_information` function.

The runtime checks embedding, hidden, and output gradients against central
finite differences with `epsilon = 1e-3`, relative tolerance `1e-3`, and
absolute tolerance `1e-4`.
