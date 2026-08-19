"""Isolated Python runtime worker used for timing and allocation measurements."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys
import time
import tracemalloc


ROOT = Path(__file__).parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_steps(path: Path, layer: str, size: int) -> list[list[dict]]:
    grouped: dict[int, list[dict]] = {}
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            if raw["layer"] != layer:
                continue
            event = {
                "t": int(raw["t"]), "outcome": int(raw["outcome"]),
                "latent_p": float(raw["latent_p"]), "regime_switch": int(raw["regime_switch"]),
                "source": raw["source"], "signal_p": float(raw["signal_p"]),
                "weight": float(raw["weight"]), "confidence": float(raw["confidence"]),
                "age": float(raw["age"]), "d": float(raw["d"]),
            }
            grouped.setdefault(event["t"], []).append(event)
    base = [grouped[index] for index in sorted(grouped)]
    result: list[list[dict]] = []
    for index in range(size):
        copied = [dict(event, t=index) for event in base[index % len(base)]]
        result.append(copied)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("implementation", choices=("plain", "state_class"))
    parser.add_argument("--layer", default="B")
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--mode", default="dynamic")
    args = parser.parse_args()
    implementation = load_module(args.implementation, ROOT / "python" / f"{args.implementation}.py")
    steps = load_steps(ROOT / "data" / "events.csv", args.layer, args.size)
    started = time.perf_counter_ns()
    predictions = implementation.predict(steps, args.layer, args.mode)
    elapsed_ns = time.perf_counter_ns() - started
    tracemalloc.start()
    implementation.predict(steps, args.layer, args.mode)
    snapshot = tracemalloc.take_snapshot()
    _, peak = tracemalloc.get_traced_memory()
    allocations = sum(stat.count for stat in snapshot.statistics("filename"))
    tracemalloc.stop()
    print(json.dumps({
        "elapsed_ns": elapsed_ns, "peak_tracemalloc_bytes": peak,
        "live_tracemalloc_allocations": allocations, "final_p": predictions[-1],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
