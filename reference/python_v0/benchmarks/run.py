#!/usr/bin/env python3
"""Run matched Lana bytecode and ordinary-Python microbenchmarks.

This harness is intentionally a correctness-first comparison aid, not evidence
of a performance advantage. Each pair has the same fixed initial states and
update rule. It reports runtime and peak Python allocation alongside simple
source-structure proxies; it does not claim that those proxies are universal
measures of program complexity.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from structured_state.compiler.compiler import compile_source
from structured_state.vm.machine import VM


CASES = {
    "evolving_belief": {"variables": 2, "update_rules": 2, "state_transitions": 2},
    "conflicting_evidence": {"variables": 4, "update_rules": 1, "state_transitions": 1},
    "information_network": {"variables": 3, "update_rules": 3, "state_transitions": 3},
}


@dataclass(frozen=True)
class Measurement:
    result: float
    median_ms: float
    peak_bytes: int
    lines: int
    explicit_variables: int
    explicit_update_rules: int
    state_transitions: int
    behavioral_stability: float
    convergence: str


def source_lines(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#"))


def load_baseline(path: Path) -> Callable[[], float]:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load baseline {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run


def measure(run_once: Callable[[], float], *, iterations: int, lines: int, metadata: dict[str, int]) -> Measurement:
    # One traced allocation run captures a bounded peak; timed iterations use no tracing.
    tracemalloc.start()
    first = run_once()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results = [first]
    durations: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        results.append(run_once())
        durations.append((time.perf_counter_ns() - started) / 1_000_000)
    if any(result != first for result in results):
        raise AssertionError("deterministic benchmark produced inconsistent results")
    return Measurement(
        result=float(first),
        median_ms=statistics.median(durations),
        peak_bytes=peak,
        lines=lines,
        explicit_variables=metadata["variables"],
        explicit_update_rules=metadata["update_rules"],
        state_transitions=metadata["state_transitions"],
        behavioral_stability=statistics.pstdev(results),
        convergence="not_applicable: fixed finite transition sequence",
    )


def run_case(name: str, iterations: int) -> dict[str, object]:
    structured_path = ROOT / "benchmarks" / "structured" / f"{name}.ss"
    conventional_path = ROOT / "benchmarks" / "conventional" / f"{name}.py"
    program = compile_source(structured_path.read_text())
    lana = measure(lambda: VM(program).run(), iterations=iterations, lines=source_lines(structured_path), metadata=CASES[name])
    conventional = measure(load_baseline(conventional_path), iterations=iterations, lines=source_lines(conventional_path), metadata=CASES[name])
    if abs(lana.result - conventional.result) > 1e-12:
        raise AssertionError(f"{name}: Lana={lana.result} conventional={conventional.result}")
    return {"case": name, "lana": asdict(lana), "conventional": asdict(conventional)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run correctness-matched Lana benchmark pairs")
    parser.add_argument("--iterations", type=int, default=1_000)
    parser.add_argument("--case", choices=tuple(CASES), action="append")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    results = [run_case(name, args.iterations) for name in (args.case or CASES)]
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for item in results:
            print(f"{item['case']}: result={item['lana']['result']:.12f}; Lana median={item['lana']['median_ms']:.4f}ms; Python median={item['conventional']['median_ms']:.4f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
