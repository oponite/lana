# Performance

Machine-local measurements of the C11 VM. These are evidence, not release
certification: no cache flush, no core pinning, and some results are single
samples. Treat overlapping trial ranges as inconclusive.

## 1. State-core comparison

Measured with `scripts/version_metrics.py`; canonical run at
`build/version-metrics-20260905-allocation/`. One unit is one completed task
instance. CPU cycles are process-wide hardware counts (`proc_pid_rusage`
`ri_cycles`), including kernel work and counter-read overhead. Five trials,
seeds 42–46, rotating version order.

Units/cycle (higher is better):

| Task | 1.0.0 | 2.0.0 | 2.0.0 vs 1.0.0 |
|---|---:|---:|---:|
| construction | 0.000376658 | 0.000338857 | −10.0% |
| transform | 0.000356376 | 0.000318221 | −10.7% |
| append | 0.000215733 | 0.000204806 | −5.1% |
| measurement | 0.000220111 | 0.000198988 | −9.6% |
| sampling | 0.000128037 | 0.000123100 | −3.9% |
| combined | 0.000206152 | 0.000192865 | −6.4% |

Allocation efficiency (combined task): 2.0.0 allocates 2.03 objects/unit vs
1.0.0's 4.03, but retains 588.8 bytes/unit vs 492.8 — fewer, larger
allocations. This is allocation efficiency, not memory-transfer efficiency.

## 2. DRAM transfer

`build/dram-transfer-20260905.md`. Fixed workload of 20 million combined
semantic units, raw uncore PMU counters (`UNC_M_RD_DATA`/`UNC_M_WR_DATA`,
32-byte transfers), baseline-subtracted. Linux, one serial sample per version.

2.0.0 transferred 36.84% fewer DRAM bytes and achieved 58.32% more semantic
units per transferred bit than 1.0.0. Background host traffic was nonzero; the
baseline subtraction is the principal uncertainty.

## 3. Compiler bootstrap

`plans/compiler-performance.md`. Self-hosted compiler, Release build, five
alternating runs after one warm build.

- Release median: 0.543 s
- Instructions: 19.5 M
- Allocations: 619 K
- Allocated bytes: 93.1 MB

## 4. GC pause

`plans/gc-performance.md`. A task safepoint performs one young-generation
collection or one bounded incremental mark slice (default 128 objects). The
acceptance target is a 10 ms p99 safepoint on the 20,000-node collector stress
graph.

The bounded 128-object incremental slice measured a maximum of 0.181 ms over the
20,000-node Debug stress graph, below the 10 ms target.

## 5. Dispatch-fix experiments

This session's measurements of the 2.0.1 dispatch changes on the `combined`
task:

- Safepoint throttle (skip the whole safepoint every 256 instructions, ceiling
  probe): −15.4%
- `opcode_counts` removal: −2.1%
- Combined: −16.4%

The shipped 2.0.1 changes keep GC timing semantics intact and target most of
this ceiling without the naive throttle.

## 6. fp16/bf16 matmul

LIP-027 B16 native fp32-accumulation loops for f16/bf16 inputs, measured on a
representative training workload: 1000 iterations of a 128x128 matmul (a small
linear-layer forward pass, iterated), `ones` inputs, C11 VM (`build/lana run`).
Wall-clock, best-of-5 median, no cache flush, single machine (macOS, Apple
Silicon). Cross-checked on the Rust VM.

| dtype | median time (s) | vs f64 |
|---|---:|---:|
| f64 | 0.09 | 1.0x |
| f16 | 5.10 | 0.018x (56.7x slower) |
| bf16 | 5.09 | 0.018x (56.6x slower) |

The fp16/bf16 native loops are ~57x slower than the f64 path, not faster. The
expected memory-bound speedup (half the bytes of f64) did not materialize; the
f16/bf16 path is dominated by per-element software conversion between the
low-precision storage type and the fp32 accumulator, which is far more
expensive than the native f64 multiply-add loop. The regression is consistent
across both VMs (Rust VM at n=200: f64 0.09s vs f16 3.81s, ~42x slower). The
fp16/bf16 native path should not be relied on for speed until this is
addressed.
