# Requirements local qualification — 2026-09-28

## Candidate identity

This report qualifies the dirty working source, including required untracked
files. HEAD alone does not identify the candidate. The final content manifest,
source archive and checksum sidecar are under `/tmp/lana-final-candidate-source/`.
The clean binary archive and checksum are under `/tmp/lana-qualified-binary/`.
The 229 pre-existing staged paths remain staged; no commit or publication was made.

| Artifact | SHA-256 |
| --- | --- |
| Native Release CLI | 997d37848dc4330980e0ebe89118571c6790f9dcaa534f6416d846a9a32cc6b0 |
| Installed compiler | 0a244b2db57927ec8022ca747e42dc07860e3678fd00f7c9b51ba82eecccf75a |
| Universal binary archive | 571bd501813f11878bce21643c524645581b793828ae7373ee3bf24417f48c00 |

Final documentation updates do not change compiled inputs. The final archive
contains those updates and is extracted and built in a new directory. Its
manifest records every included file; checksum verification precedes extraction.

## Required gates

| Check | Result | Evidence under /tmp |
| --- | --- | --- |
| Native Release build and version | 4.0.0, LABC v2 | lana-final-small-call-build.log |
| Source/CLI acceptance | 146/146; twice byte-stable native compiler bootstrap | lana-final-small-call-native-tests.log |
| Locked workspace | 309 VM, 116 runtime, 27 bytecode, 7 WASM checks; remaining CLI/doc checks pass | lana-final-small-call-workspace-tests.log |
| Release routine collector | Deep/wide 20,000-node graphs; ≤128 work units and p99 ≤10 ms assertions pass | lana-final-small-call-release-slices.log |
| Bytecode fuzz | 198,006,516 executions in 601 seconds; no finding | lana-final-ownership-fuzz.log |
| Universal build and clean install | arm64/x86_64 execute with adjacent compiler | lana-final-small-call-universal.log |
| Checksum binary archive extraction | Both slices run the copied source outside the checkout; stdlib/license included | lana-qualified-binary.log |
| Clean source archive build/run | Fresh extraction and build; installed source example runs | lana-final-candidate-source-build.log |
| Python integration | 58 passed | lana-final-small-call-python.log |
| Hugging Face integration | 14 passed | lana-final-small-call-hf.log |
| Actual wasm32 execution | Node source, stdlib, tasks, repeated calls, host errors and limits pass | lana-final-small-call-wasm.log |
| Actual editor clients | Neovim and VS Code live checks pass; VS Code unit checks pass | lana-final-small-call-neovim.log, lana-final-small-call-vscode.log, lana-final-small-call-vscode-unit.log |
| Packaging failure isolation | Included in the complete native suite | lana-final-small-call-native-tests.log |
| Whitespace errors | git diff --check passes | Final local check |

The workspace's existing ignored `public_release_smoke` downloads a published
GitHub release. Publication is outside this plan; the required local universal
and checksum/archive checks run directly against this candidate.

Compiler limits remain 256 MiB and 50,000,000 instructions. Current self-hosting
finishes twice at 49,241,462 instructions with byte-identical assembly;
`/tmp/lana-small-call-fixed-point.log` records both runs. An earlier candidate
failed bootstrap at the fixed limit. Direct small-call argument packing and
cached AST tags remove actual compiler work; no collector bypass or raised
limit was introduced. The failed log remains preserved.

## Commands

```sh
python3 tools/build.py build
python3 tests/run.py --no-build
target/lana/bin/lana version
cargo test --locked --workspace --no-fail-fast
cargo test --release --locked -p lana-vm routine_collection_uses_bounded_slices_and_restarts_after_mutation
RUSTC="$(rustup which --toolchain nightly rustc)" rustup run nightly cargo fuzz run lana_bytecode --fuzz-dir fuzz -- -max_total_time=600 -timeout=5
python3 tools/build.py universal
python3 tools/build.py install --from target/universal --prefix /tmp/lana-qualified-binary/install
lipo /tmp/lana-qualified-binary/install/bin/lana -verify_arch arm64 x86_64
python3 tools/build.py package --from /tmp/lana-qualified-binary/install --output /tmp/lana-qualified-binary/lana-4.0.0-macos.tar.gz
/tmp/lana-integrations-venv/bin/python -m pytest -q integrations/python/tests
/tmp/lana-integrations-venv/bin/python -m unittest discover -s tools/lana-hf/tests -v
bash tools/rust/lana-wasm/tests/run-wasm-conformance.sh
(cd integrations/editors/vscode && npm test)
python3 tests/test_editors_live.py neovim target/lana/bin/lana
python3 tests/test_editors_live.py vscode target/lana/bin/lana
git diff --check
```

After checksum verification and extraction, each installed architecture runs
`version` and the copied `belief.lana` outside the repository. The source archive
build runs `python3 tools/build.py build` in the extracted directory and executes
its installed `target/lana/bin/lana` against that directory's example.

The performance command is:

```sh
python3 tools/benchmark.py --baseline-root /tmp/lana-requirements-baseline-d102fda --baseline-cli /tmp/lana-requirements-baseline-build/cargo-target/release/lana --baseline-library /tmp/lana-requirements-baseline-build/liblana_bridge.3.0.2.dylib --candidate-cli target/lana/bin/lana --compiler target/lana/bin/lana-compiler.labc --output /tmp/lana-final-small-call-performance.json
```

The two additional declared rounds use separate `-2.json` and `-3.json` outputs.

## Paired Release performance

The unchanged harness uses 30 calls, five warmups, alternating baseline and
candidate calls, one compiler artifact, and each version's Python bridge.
The detached 3.0.2 baseline is `/tmp/lana-requirements-baseline-d102fda`.
All three declared rounds pass independently. No build or fuzzer runs during
the performance rounds.

| Workload | Round 1 old/new warm median, ms | Round 2 ratio | Round 3 ratio |
| --- | --- | --- | --- |
| Python bytecode | 0.422 / 0.430 | 1.0302 | 1.0059 |
| Python source | 23.315 / 18.824 | 0.7598 | 0.7829 |
| Rust VM | 25.021 / 24.057 | 0.9915 | 0.9657 |
| Rust compilation | 21.168 / 20.855 | 0.9547 | 0.9588 |

First Python calls are reported separately: bytecode 1.663 / 3.655 ms and source
24.724 / 23.078 ms, baseline / candidate. Raw samples and binary/compiler
identities are in `lana-final-small-call-performance.json`,
`lana-final-small-call-performance-2.json` and
`lana-final-small-call-performance-3.json` under `/tmp`.

The older `/tmp/lana-candidate-performance.json` failed its bytecode threshold
by 7.75%. Its binary and compiler identities differ from this candidate. That
failure remains historical evidence; a passing retry does not explain its
cause. This candidate has three independent passing rounds, and no causal
claim is made about the old run.

## Completion boundary

The active heap ownership, collector policy and qualification rows are closed
by this evidence and checkpoints 61–82. REQUIREMENTS.md removes verified active
items and preserves deferred/research exclusions. Signing, notarization,
Homebrew submission and publication remain outside the authorized plan.
