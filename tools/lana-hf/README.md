# lana-hf

Local-only bridge for the fixed Lana reference Brain format. No runtime Python
packages are required. It is not a Transformers model loader.

```bash
python3 tools/lana-hf/lana_hf.py tokenize tokenizer.json "hello lana"
python3 tools/lana-hf/lana_hf.py detokenize tokenizer.json '[1,2]'
```

Supported tokenizers are WordLevel with either no pre-tokenizer (whole input)
or WhitespaceSplit. Normalization, added tokens, post-processing, decoder,
padding, truncation, and other models are rejected. Vocabulary IDs must fit the
brain. Detokenization joins tokens with spaces; it does not reconstruct original
whitespace. Official tokenizers and SafeTensors compatibility checks run with:

```bash
python3 -m unittest discover -s tools/lana-hf/tests -v
```

Install `tokenizers`, `safetensors`, and `numpy` in a test environment to enable
the reference-library test; without them that test is explicitly skipped.

Packages contain config.json, generation_config.json, tokenizer.json, README.md,
model.safetensors, and the Lana brain snapshot. Tensor names, shapes, F32 dtype,
contiguous non-overlapping offsets, finite values, and model dimensions are
validated before replacing a brain. Files are bounded to 256 MiB.

Package destinations must be absent. Use one writer per package destination.
Files use exclusive temporary siblings and atomic replacement. A failure before
replacement preserves the old file. A directory-sync failure after replacement
means the complete file is visible but crash durability is uncertain; inspect it
before retrying. Abrupt process termination may leave an owned temporary file.
