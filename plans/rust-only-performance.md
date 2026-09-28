# Rust-only 4.0 performance gate

Run `python3 tools/benchmark.py --help` for the current paired harness. It takes
explicit baseline root/CLI/native-library paths, uses one compiler artifact for
both CLIs, alternates paired calls, records all samples and SHA-256 identities,
and exits nonzero when any warm median exceeds 1.05 times the baseline. The Python
paths use their respective native bridge and Rust worker; fallback is an error.
The old C11 and current Rust echo wrappers must return the same checked input.

The tables below are historical evidence, not qualification of the current tree.

The pre-migration snapshot below is machine-local evidence, not conformance.
It was measured on 2026-09-24 from worktree HEAD
`6aacbde2f67f2dd5adfa42da22943f33b0457732` with other local changes
already present. Both CLIs were Release builds. Each median excludes five
warm-up calls from 30 sequential runs.

| Workload | 3.0.2 warm median |
| --- | ---: |
| Python `Lana.run_labc` through `liblana_bridge`, v2 echo | 0.397 ms |
| Python `Lana.run`, C bridge plus Rust compile, echo source | 17.783 ms |
| Rust `lana run` of `isa_ops_pass.lana` | 24.006 ms |
| Rust `lana compile` of `echo_bridge_c11.lana` | 15.976 ms |

The Python source wrapper changed from `echo_bridge_c11.lana` to
`echo_bridge.lana`; both echo the same JSON input, but their parsers differ.
The 4.0 candidate was measured with the same build profile and 30-call method:

| Workload | 4.0 warm median | First call | 5% threshold |
| --- | ---: | ---: | ---: |
| Python `Lana.run_labc`, Rust worker | 0.393 ms | 4.075 ms | 0.417 ms |
| Python `Lana.run`, Rust worker | 15.042 ms | 17.544 ms | 18.672 ms |
| Rust `lana run`, ISA fixture | 23.974 ms | 29.245 ms | 25.206 ms |
| Rust `lana compile`, echo source | 17.050 ms | 17.617 ms | 16.775 ms |

The isolated compile median exceeded the historical snapshot by 6.7% while a
fuzzer was running. After the fuzzer completed, the final idle paired check
used 30 calls per path with five warmups discarded:

| Workload | 3.0.2 median | 4.0 median | Result |
| --- | ---: | ---: | --- |
| Python bytecode echo | 0.412 ms | 0.387 ms | Pass |
| Python source echo | 18.017 ms | 14.938 ms | Pass |
| Rust ISA run | 22.953 ms | 22.974 ms | Pass |
| Rust source compile | 17.020 ms | 16.983 ms | Pass |

The CLI workloads alternated old and new binaries and used the same compiler
artifact. Both Python workloads used their respective 3.0.2 and 4.0 bridge
implementations. Every final median is within 5% of the paired old median.
First Python calls were 1.129 ms old versus 3.914 ms new for bytecode, and
22.278 ms old versus 15.576 ms new for source. The new bytecode first call
includes worker startup; it is outside the warm-call threshold.
