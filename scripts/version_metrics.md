# Compare Lana state-core versions

Run on an arm64 Mac with Xcode, CMake, Python 3, and AC power:

```sh
python3 scripts/version_metrics.py --self-test
python3 scripts/version_metrics.py --output build/version-metrics-new
```

The output directory must not exist. Without `--output`, the runner creates a
temporary directory. A run builds the three pinned releases and records 90
trials: six tasks, three versions, five trials each. Allow a few minutes.

The runner uses `git archive` on the pinned commits. It does not change branches,
tags, release sources, or existing files. The remote 1.1.0 release commit is
`9f307c40f532aa57e63ebf73def6f77f88078f42`; the local tag can differ. No 1.2
revision is included. This compares the releases' C implementations, not the
current Rust implementation.

## What is measured

One semantic unit is one completed instance of a task. The tasks are state
construction, alternating invert/neutralize transforms, lazy append, alternating
state/distribution probability measurement, sampling, and
construct–append–invert–measure. Each task includes the input construction and
result storage it needs. Scores are per task, not isolated opcode costs.

The frozen corpus has 256 distinct input pairs. The arithmetic permutation uses
seed 42, covers probabilities 0 and 1, and includes zero, real, and complex
dispositions. Python computes expected results independently from the semantic
formulas. Every output is checked, with absolute tolerance `1e-12`. Sampling
uses `sample_value(sample(...))` and checks validity and probability; a separate
symmetric-kernel smoke check uses 4,096 samples. A deliberately wrong expected
result must fail before measurements start.

```text
S = completed instances of this task
C = processor cycles
B = 8 × (CPU-to-main-memory bytes read + bytes written)

speed             = S / C
memory efficiency = S / B
```

Cycle counts use macOS `proc_pid_rusage(RUSAGE_INFO_V4).ri_cycles`, with the
same delta approach used in [Apple's counter tests](https://github.com/apple-oss-distributions/xnu/blob/main/tests/recount/recount_perf_tests.c).
These are process hardware counts, not elapsed time multiplied by frequency.
They include kernel work and counter-read overhead. Zero or unavailable counts
leave speed unscored; tasks/second is reported separately.

Memory traffic is **unavailable** in this implementation. The installed CPU
Counters template was probed, but an attributable CPU-to-DRAM byte counter was
not validated. Allocation sizes, cache misses, and system/GPU bandwidth do not
substitute for it. Add a memory score only after validating that boundary and
attribution on the measurement hardware.

The runner also reports two Lana-native allocation measures. Allocation
efficiency is completed semantic units divided by `allocation_count`, a
cumulative count of VM-managed allocations. Retained-heap efficiency is units
divided by `allocated_bytes` after execution. GC decrements `allocated_bytes`,
so it is live heap at that point, not total bytes allocated, peak heap, or DRAM
traffic. These measures are retained for every task/version beside the transfer
metric, which remains unavailable.

## Read the results

`report.md` contains medians, ranges, and changes against 1.0.0 and the preceding
version. `trials.csv` contains raw counts and durations. `manifest.json` records
commits, toolchain, power settings, corpus/source/bytecode hashes, and repetitions.
Build logs, generated sources, expected outputs, and sampling checks are retained.

Intervals include VM initialization, execution, and cleanup; loading bytecode,
validation, and output are excluded. Validation runs between execution and
cleanup and can warm caches. There is no cache flush, core pinning, or frequency
lock. Background work and core scheduling can affect results. Compare the trial
ranges before interpreting small differences. No overall weighted score or
release-readiness claim is produced.
