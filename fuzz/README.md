# Rust bytecode fuzzing

```bash
cd fuzz
rustup run nightly cargo fuzz run lana_bytecode -- -max_total_time=600 -timeout=5
```

The target calls the Rust loader and verifier through `lana_fuzz::check`.
`seeds/` contains minimal LABC v1-v5 assembly sources. Regenerate their corpus
entries with `cargo run -p lana-cli -- asm fuzz/seeds/vN.lasm -o fuzz/corpus/lana_bytecode/vN`.
Keep minimized crashing inputs in `corpus/lana_bytecode/`; CI uploads transient
crashes from `artifacts/` for promotion into that corpus.
