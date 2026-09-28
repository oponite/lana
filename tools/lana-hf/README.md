# lana-hf

Local-only bridge for Lana LBRN1 and dense-layer LBRN2 Brain files. No runtime Python
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

LBRN2 packages use `lana-brain-hf-v2`, with explicit layer descriptions and
SHA-256 digests for the Brain, weights, and tokenizer. Import requires the
complete snapshot and checks that its weights match the SafeTensors bytes.
Standalone SafeTensors import requires an existing Brain template and preserves
its memory and training history. Typed Information memory is validated by the
Rust loader, including Core observation replay. For standalone bridge commands,
put `lana` on PATH or set `LANA_CLI` to its path. `lana brain save` and `load`
set that path automatically.

Package destinations must be absent. Use one writer per package destination.
Files use exclusive temporary siblings and atomic replacement. A failure before
replacement preserves the old file. A directory-sync failure after replacement
means the complete file is visible but crash durability is uncertain; inspect it
before retrying. Abrupt process termination may leave an owned temporary file.
