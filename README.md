# Lana

Lana is a general-purpose programming language that treats uncertain, evolving
information as a first-class state rather than forcing it into definite values.

Its underlying belief is that information is not always best represented as a
definite value. It can be represented as a probability plus a built-in
disposition toward future change.

Alongside ordinary primitives such as numbers, booleans, strings, arrays, and
functions, Lana has a native `STATE = (p, d)` primitive. `p` is its current
observable probability/state value; `d` is its signed directional dependency—
the built-in disposition controlling the direction and strength of future
influence. Positive `d` aligns influence with `p`; negative `d` makes the
influence contrarian by targeting `1 - p`; `|d|` is the strength.

```lana
state belief = state { p: 0.50, d: 0.30 };
state evidence = state { p: 0.90, d: 0.65 };
apply evidence -> belief;
print(measure belief as probability);
```

Python provides source tooling. The canonical register VM and structured-state
semantics are written in C11. ARM64 and x86_64 assembly probes validate the
low-level `APPLY` arithmetic without becoming a second semantic authority.

## Build and run

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
PYTHONPATH=src python3 -m lana_tooling.cli run examples/belief.lana --trace
```

Low-level bytecode tools work without Python:

```bash
build/ssvm asm examples/belief.ssa -o build/belief.ssb
build/ssvm dis build/belief.ssb
build/ssvm run build/belief.ssb --trace
```

## Language basics

Lana source uses braces and semicolons. Python tooling compiles it to portable
SSBC bytecode; the C VM is the only execution engine.

`STATE = (p, d)` is additional to ordinary values, not a replacement for them.
Initial scalar syntax includes `let`, assignment, arithmetic, comparisons,
arrays, `if`, `while`, functions, and `return`. Structured-state operations
compile to dedicated opcodes.

Optional `timestamp`, `source`, `weight`, and `confidence` indexes stay outside
the primitive pair. Configure bounded history with
`history belief latest 32;` or a duration. `previous(belief)`,
`change(belief)`, and `velocity(belief)` expose recorded transitions.

See [SPEC.md](SPEC.md) for the normative state and operation semantics.

## Falsification benchmark

Run the domain-neutral Lana-versus-Python benchmark with:

```bash
python3 benchmark/run_benchmark.py
python3 benchmark/validate_results.py
python3 benchmark/build_report.py
```

The benchmark compares plain Python, a Python `State` class, and native Lana
bytecode on identical sequential evidence. It includes `d` ablations, stress
tests, maintainability changes, runtime/VM counters, and a blinded human-test
harness. See [benchmark/README.md](benchmark/README.md) for the methodology.
