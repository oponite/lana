# Lana benchmark harness

Run all three matched benchmark pairs from the repository root:

```bash
python benchmarks/run.py --iterations 1000
```

Use `--json` for records suitable for later analysis. The harness compiles each
`.ss` file into Lana bytecode and runs that bytecode in a fresh VM per timed
iteration. Each conventional implementation encodes the same fixed initial
states and update sequence.

Reported fields include median runtime, one-run peak Python allocation, source
lines, explicit variables/update rules, state transitions, deterministic output
stability, and convergence status. The structural fields are transparent
proxies, not claims that Lana is universally more concise or faster. These
small cases intentionally establish a reproducible baseline before broader,
parameterized workload studies.
