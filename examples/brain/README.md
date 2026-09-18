# Brain workshop

This is the complete local reference workflow. It uses a WordLevel Hugging
Face `tokenizer.json`, a three-token CPU Brain, and no network access.

```bash
LANA_BIN=target/debug/lana examples/brain/run.sh
```

The command creates a brain, trains it, evaluates it, exports and reimports
its strict safetensors package, runs one memory-backed chat turn, and inspects
the reloaded record. Failed training, bridge conversion, or save leaves the
prior Brain file unchanged.

The initial bridge supports WordLevel tokenizers only. Other tokenizer models
return an explicit unsupported result rather than changing token IDs.

The runtime checks embedding, hidden, and output gradients against central
finite differences with `epsilon = 1e-3`, relative tolerance `1e-3`, and
absolute tolerance `1e-4`.
