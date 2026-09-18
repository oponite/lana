# lana-hf

Local-only Hugging Face tokenizer bridge. It supports WordLevel `tokenizer.json`
files without a Python dependency and rejects other tokenizer models explicitly.

```bash
python3 tools/lana-hf/lana_hf.py tokenize tokenizer.json "hello lana"
python3 tools/lana-hf/lana_hf.py detokenize tokenizer.json '[1,2]'
```
