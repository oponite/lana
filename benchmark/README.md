# Lana falsification benchmark

This benchmark compares the same deterministic sequential prediction policy in
three forms: plain Python, a competent Python `State` class, and Lana SSBC
executed by the C VM. The checked-in CSV is generated without observing the
binary outcome when creating evidence, so no implementation receives target
leakage.

Run everything from the repository root:

```bash
python3 benchmark/run_benchmark.py
```

Generated source, bytecode, raw measurements, and the report are written below
`benchmark/results/`. Generated code and data rows are excluded from core-code
LOC. The report also includes end-to-end tooling LOC so that exclusion cannot
hide Lana's current lack of file I/O and dynamic state construction.

The synthetic evidence stream is domain-neutral. It contains regime changes,
noise, stale observations, conflicting sources, and heterogeneous weights and
confidence. Synthetic results test semantics and engineering complexity; they
do not establish real-world predictive validity.
