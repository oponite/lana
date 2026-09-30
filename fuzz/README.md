# Bytecode fuzz testing

This test feeds generated and mutated bytecode into the Lana Rust loader and
verifier. It checks that Lana rejects malformed files without crashes, hangs,
or memory errors. The test does not execute the bytecode in the VM.

## Run the test

The commands require Rust and `rustup`. Run them from the Lana repository root.

1. Install nightly Rust:

   ```bash
   rustup toolchain install nightly --profile minimal
   ```

2. Install the fuzz testing tool used by CI:

   ```bash
   rustup run nightly cargo install cargo-fuzz --version 0.13.2 --locked
   ```

3. Run the ten-minute test:

   ```bash
   RUSTC="$(rustup which --toolchain nightly rustc)" rustup run nightly cargo fuzz run lana_bytecode --fuzz-dir fuzz -- -max_total_time=600 -timeout=5
   ```

The `RUSTC` setting explicitly selects the nightly compiler.
The `-max_total_time=600` flag limits fuzz testing to ten minutes, after compilation.
The `-timeout=5` flag limits each input to five seconds.

## Read the results

Rejection of malformed bytecode is expected. The target accepts normal loader
and verifier errors, then continues with other inputs.

A successful run finishes without a crash, leak, timeout, failed assertion, or
sanitizer report. A sanitizer report identifies a possible memory error.
A successful run does not prove that all possible inputs are safe.

If the test reports a problem, inspect the saved input in `artifacts/lana_bytecode/`.

## Folder contents

The folder contains the test code and saved inputs:

| Path | Purpose |
| --- | --- |
| `fuzz_targets/lana_bytecode.rs` | Passes each input to `lana_fuzz::check`, which calls the loader and verifier. |
| `seeds/` | Contains minimal assembly examples for LABC versions 1–5. |
| `corpus/lana_bytecode/` | Contains bytecode inputs that the fuzzer reuses and mutates. |
| `artifacts/lana_bytecode/` | Contains inputs saved after crashes, timeouts, or other detected problems. |
| `target/` | Contains generated build files. |
| `Cargo.toml` and `Cargo.lock` | Define the fuzz target and its dependencies. |

## Maintain the saved inputs

To regenerate a seed, assemble its source from the Lana repository root.
For example, regenerate the LABC v2 seed:

```bash
cargo run -p lana-cli -- asm fuzz/seeds/v2.lasm -o fuzz/corpus/lana_bytecode/v2
```

The same command applies to versions 1–5 with the corresponding filenames.

After investigation, keep minimized crashing inputs in `corpus/lana_bytecode/`
as regression inputs. CI uploads files from `artifacts/` after a failed fuzz run.
