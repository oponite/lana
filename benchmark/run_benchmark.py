"""Execute the Lana falsification benchmark and save inspectable evidence."""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import time

from analyze_complexity import collect as collect_complexity
from generate_data import write_csv


ROOT = Path(__file__).parents[1]
BENCHMARK = Path(__file__).parent
RESULTS = BENCHMARK / "results"
DATA = BENCHMARK / "data" / "events.csv"
DEFAULT_VM = ROOT / "build" / "lanavm"
VM = DEFAULT_VM


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PLAIN = load_module("benchmark_plain", BENCHMARK / "python" / "plain.py")
STATE_CLASS = load_module("benchmark_state_class", BENCHMARK / "python" / "state_class.py")
LANA_GENERATOR = load_module("benchmark_lana_generator", BENCHMARK / "lana" / "generate.py")


def load_steps(layer: str, limit: int | None = None) -> list[list[dict]]:
    grouped: dict[int, list[dict]] = {}
    with DATA.open(newline="") as handle:
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
    steps = [grouped[index] for index in sorted(grouped)]
    return steps if limit is None else steps[:limit]


def repeated_steps(layer: str, size: int) -> list[list[dict]]:
    base = load_steps(layer)
    return [[dict(event, t=index) for event in base[index % len(base)]] for index in range(size)]


def parse_vm_stats(stderr: str) -> dict:
    match = re.search(r"LANAVM_STATS (\{.*\})", stderr)
    if match is None:
        raise RuntimeError(f"VM stats missing from stderr: {stderr[-500:]}")
    return json.loads(match.group(1))


def assemble(assembly: Path, bytecode: Path) -> float:
    started = time.perf_counter_ns()
    subprocess.run([VM, "asm", assembly, "-o", bytecode], check=True, capture_output=True, text=True)
    return (time.perf_counter_ns() - started) / 1_000_000.0


def run_lana(steps: list[list[dict]], layer: str, mode: str, name: str,
             emit_predictions: bool = True) -> tuple[list[float], dict, float, Path]:
    generated = RESULTS / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    assembly = generated / f"{name}.lasm"
    bytecode = generated / f"{name}.labc"
    assembly.write_text(LANA_GENERATOR.generate(steps, layer, mode, emit_predictions=emit_predictions))
    compile_ms = assemble(assembly, bytecode)
    completed = subprocess.run([VM, "run", bytecode, "--stats"], check=True, capture_output=True, text=True)
    predictions = [float(line) for line in completed.stdout.splitlines() if line.strip()]
    return predictions, parse_vm_stats(completed.stderr), compile_ms, bytecode


def probabilistic_metrics(predictions: list[float], steps: list[list[dict]]) -> dict[str, float]:
    outcomes = [events[0]["outcome"] for events in steps]
    clipped = [max(1e-12, min(1.0 - 1e-12, value)) for value in predictions]
    n = len(outcomes)
    brier = sum((prediction - outcome) ** 2 for prediction, outcome in zip(predictions, outcomes)) / n
    log_loss = -sum(outcome * math.log(prediction) + (1 - outcome) * math.log(1 - prediction)
                    for prediction, outcome in zip(clipped, outcomes)) / n
    accuracy = sum((prediction >= 0.5) == bool(outcome) for prediction, outcome in zip(predictions, outcomes)) / n
    bins: list[list[tuple[float, int]]] = [[] for _ in range(10)]
    for prediction, outcome in zip(predictions, outcomes):
        bins[min(9, int(prediction * 10))].append((prediction, outcome))
    ece = sum(len(bucket) / n * abs(statistics.fmean(item[0] for item in bucket) - statistics.fmean(item[1] for item in bucket))
              for bucket in bins if bucket)
    volatility = statistics.fmean(abs(right - left) for left, right in zip(predictions, predictions[1:]))

    def latency(indices: list[int]) -> float:
        values = []
        for index in indices:
            desired = steps[index][0]["latent_p"] >= 0.5
            for offset, prediction in enumerate(predictions[index:index + 40]):
                if (prediction >= 0.5) == desired:
                    values.append(offset)
                    break
            else:
                values.append(40)
        return statistics.median(values) if values else 0.0

    switches = [index for index, events in enumerate(steps) if events[0]["regime_switch"]]
    conflicts = []
    for index in range(1, len(steps)):
        prior = any(event["source"] == "gamma" and event["confidence"] > 0.9 for event in steps[index - 1])
        current = any(event["source"] == "gamma" and event["confidence"] > 0.9 for event in steps[index])
        if prior and not current:
            conflicts.append(index)
    return {
        "accuracy": accuracy, "brier": brier, "log_loss": log_loss,
        "calibration_error": ece, "volatility": volatility,
        "regime_response_steps": latency(switches), "misleading_recovery_steps": latency(conflicts),
    }


def command_measure(command: list[str]) -> dict[str, float | int | dict]:
    wrapped = ["/usr/bin/time", "-l", *map(str, command)] if Path("/usr/bin/time").exists() else list(map(str, command))
    started = time.perf_counter_ns()
    completed = subprocess.run(wrapped, check=True, capture_output=True, text=True)
    wall_ns = time.perf_counter_ns() - started
    resident = re.search(r"(\d+)\s+maximum resident set size", completed.stderr)
    return {
        "wall_ns": wall_ns,
        "max_rss_bytes": int(resident.group(1)) if resident else 0,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def runtime_metrics() -> list[dict]:
    rows: list[dict] = []
    # Unrolled 10,000-step LABC exceeds the v1 loader's deliberate 100,000
    # constant safety ceiling. Four thousand steps is the largest rounded
    # comparison safely below that ceiling; the million-update loop stress test
    # covers long-running execution separately.
    for size_name, size in (("small", 100), ("large", 4_000)):
        steps = repeated_steps("B", size)
        _, vm_stats, compile_ms, bytecode = run_lana(steps, "B", "dynamic", f"runtime_{size_name}", False)
        commands = {
            "Python Plain": [sys.executable, BENCHMARK / "worker.py", "plain", "--size", str(size)],
            "Python State Class": [sys.executable, BENCHMARK / "worker.py", "state_class", "--size", str(size)],
            "Lana": [VM, "run", bytecode, "--stats"],
        }
        for implementation, command in commands.items():
            samples = [command_measure(command) for _ in range(6)]
            cold = samples[0]
            repeated = samples[1:]
            row = {
                "implementation": implementation, "size": size_name, "transitions": size,
                "cold_wall_ms": cold["wall_ns"] / 1_000_000.0,
                "repeated_wall_ms_median": statistics.median(sample["wall_ns"] for sample in repeated) / 1_000_000.0,
                "max_rss_bytes_median": int(statistics.median(sample["max_rss_bytes"] for sample in repeated)),
                "compile_ms": compile_ms if implementation == "Lana" else 0.0,
            }
            if implementation == "Lana":
                parsed_stats = [parse_vm_stats(str(sample["stderr"])) for sample in repeated]
                stats = parsed_stats[-1]
                row.update({
                    "instruction_count": stats["instructions"], "allocation_count": stats["allocations"],
                    "vm_allocated_bytes": stats["allocated_bytes"], "state_transitions": stats["state_transitions"],
                    "core_elapsed_ms": statistics.median(item["elapsed_ns"] for item in parsed_stats) / 1_000_000.0,
                })
            else:
                internals = [json.loads(str(sample["stdout"])) for sample in repeated]
                internal = internals[-1]
                row.update({
                    "instruction_count": None,
                    "allocation_count": int(statistics.median(item["live_tracemalloc_allocations"] for item in internals)),
                    "vm_allocated_bytes": int(statistics.median(item["peak_tracemalloc_bytes"] for item in internals)),
                    "state_transitions": None,
                    "core_elapsed_ms": statistics.median(item["elapsed_ns"] for item in internals) / 1_000_000.0,
                })
            rows.append(row)
    return rows


def stress_metrics() -> list[dict]:
    cases = {
        "confirming": [(0.9, 0.6)] * 100,
        "contradictory": [(0.1 if index % 2 == 0 else 0.9, 0.8) for index in range(100)],
        "d_near_negative_one": [(0.9, -0.999)] * 20,
        "d_zero": [(0.9, 0.0)] * 100,
        "d_near_positive_one": [(0.9, 0.999)] * 20,
        "p_near_zero": [(0.001, 0.5)] * 40,
        "p_near_one": [(0.999, 0.5)] * 40,
    }
    rng = random.Random(991)
    cases["weak_noise"] = [(max(0.001, min(0.999, rng.gauss(0.5, 0.2))), 0.05) for _ in range(1_000)]
    rows = []
    generated = RESULTS / "generated"
    for name, sequence in cases.items():
        target = 0.5
        observed = [target]
        lines = ["STATE_NEW R0 0.5 0.2"]
        for p, d in sequence:
            influence_target = 1.0 - p if d < 0.0 else p
            target = max(0.0, min(1.0, target + abs(d) * (influence_target - target)))
            observed.append(target)
            lines.extend((f"STATE_NEW R1 {p:.17g} {d:.17g}", "APPLY R1 R0"))
        lines.extend(("MEASURE R0 probability R2", "PRINT R2", "HALT"))
        assembly = generated / f"stress_{name}.lasm"
        bytecode = generated / f"stress_{name}.labc"
        assembly.write_text("\n".join(lines) + "\n")
        assemble(assembly, bytecode)
        completed = subprocess.run([VM, "run", bytecode, "--stats"], check=True, capture_output=True, text=True)
        vm_final = float(completed.stdout.strip())
        rows.append({
            "case": name, "transitions": len(sequence), "python_final_p": target,
            "lana_final_p": vm_final, "parity_error": abs(target - vm_final),
            "min_p": min(observed), "max_p": max(observed),
        })
    long_assembly = generated / "stress_million.lasm"
    long_bytecode = generated / "stress_million.labc"
    long_assembly.write_text(LANA_GENERATOR.stress_program(1_000_000, 0.73, 0.02, 0.5))
    assemble(long_assembly, long_bytecode)
    measured = command_measure([VM, "run", long_bytecode, "--stats"])
    stats = parse_vm_stats(str(measured["stderr"]))
    rows.append({
        "case": "million_updates", "transitions": 1_000_000,
        "python_final_p": 0.73, "lana_final_p": float(str(measured["stdout"]).strip()),
        "parity_error": abs(float(str(measured["stdout"]).strip()) - 0.73),
        "min_p": 0.5, "max_p": 0.73, "wall_ms": measured["wall_ns"] / 1_000_000.0,
        "instruction_count": stats["instructions"],
    })
    return rows


def maintenance_metrics() -> list[dict]:
    rows = []
    for change in ("stale", "sources"):
        for implementation, suffix in (("Lana", "lana"), ("Python Plain", "plain"), ("Python State Class", "state_class")):
            before = (BENCHMARK / "maintainability" / change / f"{suffix}_before.{ 'lana' if suffix == 'lana' else 'py'}").read_text().splitlines()
            after = (BENCHMARK / "maintainability" / change / f"{suffix}_after.{ 'lana' if suffix == 'lana' else 'py'}").read_text().splitlines()
            diff = list(difflib.ndiff(before, after))
            added = sum(line.startswith("+ ") for line in diff)
            removed = sum(line.startswith("- ") for line in diff)
            before_names = set(re.findall(r"\b[A-Za-z_]\w*\b", "\n".join(before)))
            after_names = set(re.findall(r"\b[A-Za-z_]\w*\b", "\n".join(after)))
            rows.append({
                "change": change, "implementation": implementation,
                "lines_added": added, "lines_removed": removed, "lines_changed": added + removed,
                "files_changed": 1, "new_names": len(after_names - before_names), "tests_modified": 0,
            })
    return rows


def write_csv_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(field for field in row if field not in fields)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    global VM
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="skip runtime and million-transition measurements")
    parser.add_argument("--vm", type=Path, default=DEFAULT_VM,
                        help="path to lanavm (default: build/lanavm)")
    args = parser.parse_args()
    VM = args.vm.resolve()
    if not VM.exists():
        raise SystemExit(f"{VM} missing; build lanavm or pass --vm PATH")
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_csv(DATA)
    behavior_rows: list[dict] = []
    prediction_rows: list[dict] = []
    parity: dict[str, float] = {}
    vm_layer_stats: dict[str, dict] = {}
    compile_times: dict[str, float] = {}
    representative_bytecode: Path | None = None
    for layer in ("A", "B", "C"):
        steps = load_steps(layer, 96)
        for mode in ("dynamic", "fixed", "zero"):
            plain_predictions = PLAIN.predict(steps, layer, mode)
            class_predictions = STATE_CLASS.predict(steps, layer, mode)
            lana_predictions, stats, compile_ms, bytecode = run_lana(steps, layer, mode, f"layer_{layer}_{mode}")
            parity[f"{layer}_{mode}_plain_class"] = max(abs(left - right) for left, right in zip(plain_predictions, class_predictions))
            parity[f"{layer}_{mode}_plain_lana"] = max(abs(left - right) for left, right in zip(plain_predictions, lana_predictions))
            compile_times[f"{layer}_{mode}"] = compile_ms
            if mode == "dynamic":
                vm_layer_stats[layer] = stats
                if layer == "B":
                    representative_bytecode = bytecode
            for implementation, predictions in (("Python Plain", plain_predictions), ("Python State Class", class_predictions), ("Lana", lana_predictions)):
                behavior_rows.append({"layer": layer, "mode": mode, "implementation": implementation, **probabilistic_metrics(predictions, steps)})
                prediction_rows.extend({
                    "layer": layer, "mode": mode, "implementation": implementation,
                    "t": events[0]["t"], "prediction": prediction,
                    "outcome": events[0]["outcome"], "latent_p": events[0]["latent_p"],
                    "regime_switch": events[0]["regime_switch"],
                } for events, prediction in zip(steps, predictions))
        p_only = PLAIN.predict(steps, layer, "p_only")
        behavior_rows.append({"layer": layer, "mode": "p_only", "implementation": "Python p-only", **probabilistic_metrics(p_only, steps)})
        prediction_rows.extend({
            "layer": layer, "mode": "p_only", "implementation": "Python p-only",
            "t": events[0]["t"], "prediction": prediction,
            "outcome": events[0]["outcome"], "latent_p": events[0]["latent_p"],
            "regime_switch": events[0]["regime_switch"],
        } for events, prediction in zip(steps, p_only))
    if representative_bytecode is not None:
        disassembly = subprocess.run([VM, "dis", representative_bytecode], check=True, capture_output=True, text=True).stdout
        (RESULTS / "representative.dis.txt").write_text(disassembly)
    complexity = collect_complexity(BENCHMARK)
    complexity["Lana"]["end_to_end_tooling_loc"] = sum(
        1 for source in (BENCHMARK / "lana" / "generate.py").read_text().splitlines()
        if source.strip() and not source.lstrip().startswith("#")
    )
    complexity["Python Plain"]["end_to_end_tooling_loc"] = complexity["Python Plain"]["loc"]
    complexity["Python State Class"]["end_to_end_tooling_loc"] = complexity["Python State Class"]["loc"]
    runtime = [] if args.quick else runtime_metrics()
    stress = [] if args.quick else stress_metrics()
    maintenance = maintenance_metrics()
    data_hash = hashlib.sha256(DATA.read_bytes()).hexdigest()
    evidence = {
        "data": {"path": str(DATA.relative_to(ROOT)), "sha256": data_hash, "rows": sum(1 for _ in DATA.open()) - 1, "synthetic": True, "seed": 1729},
        "evaluation_steps_per_layer": 96, "parity": parity, "behavior": behavior_rows,
        "complexity": complexity, "runtime": runtime, "stress": stress,
        "maintenance": maintenance, "vm_layer_stats": vm_layer_stats,
        "compile_times_ms": compile_times,
        "definitions": {
            "SMCR": "Python explicit state-management syntax occurrences divided by Lana native state-operation occurrences",
            "LCR": "Python nonblank noncomment LOC divided by Lana nonblank noncomment LOC",
            "runtime": "cold first process and median of five subsequent processes; source compilation excluded",
            "allocation": "VM arena allocations versus Python live tracemalloc blocks; directional only, not directly equivalent",
        },
        "observed_limits": {
            "unrolled_large_case": "10,000 steps exceeded the LABC v1 loader limit of 100,000 constants; runtime comparison uses 4,000 steps",
        },
    }
    (RESULTS / "evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    write_csv_rows(RESULTS / "behavior.csv", behavior_rows)
    write_csv_rows(RESULTS / "predictions.csv", prediction_rows)
    write_csv_rows(RESULTS / "runtime.csv", runtime)
    write_csv_rows(RESULTS / "stress.csv", stress)
    write_csv_rows(RESULTS / "maintainability.csv", maintenance)
    print(json.dumps({
        "evidence": str((RESULTS / "evidence.json").relative_to(ROOT)),
        "max_behavior_parity_error": max(parity.values()),
        "behavior_rows": len(behavior_rows), "runtime_rows": len(runtime), "stress_rows": len(stress),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
