# Brain workshop

This is the complete local reference workflow. It uses a WordLevel Hugging
Face `tokenizer.json`, a three-token CPU Brain, and no network access.

```bash
LANA_BIN=target/debug/lana examples/brain/run.sh
```

The command creates a brain, trains it, evaluates it, exports and reimports
its strict safetensors package, runs one memory-backed chat turn, and inspects
the reloaded record. The default run uses and removes a temporary directory.
Pass a file path to keep its outputs; its `.hf` package destination must be absent.
Failed training or bridge validation preserves the prior Brain file. Save uses
atomic replacement; a sync failure after replacement means durability is
uncertain and requires inspection before retrying.

The bridge supports WordLevel with no pre-tokenizer or WhitespaceSplit and no
other tokenizer processing. Unsupported pipelines fail explicitly. Memory is a
durable text/history record, not typed Information memory or learned facts.

The runtime checks embedding, hidden, and output gradients against central
finite differences with `epsilon = 1e-3`, relative tolerance `1e-3`, and
absolute tolerance `1e-4`.
