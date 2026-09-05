#!/usr/bin/env python3
"""Compare pinned Lana releases on macOS. No third-party Python packages."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = {
    "1.0.0": "20d512bb8c71c28ccbf75818322168853a62699b",
    "1.1.0": "9f307c40f532aa57e63ebf73def6f77f88078f42",
    "2.0.0": "39f802e6eb7ac5303a256cc5361610bc50e10296",
}
TASKS = ("construction", "transform", "append", "measurement", "sampling", "combined")


def run(args, *, log=None, timeout=600):
    result = subprocess.run([str(a) for a in args], text=True, capture_output=True, timeout=timeout)
    if log:
        Path(log).write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed ({result.returncode}): {(result.stderr or result.stdout)[-3000:]}")
    return result.stdout


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs(i):
    def state(j):
        p = ((j + 42) % 256) / 255
        re = 0 if j < 64 else (j - 128) / 256
        im = 0 if j < 128 else (127 - j) / 256
        return [p, re if 0 < p < 1 else 0, im if 0 < p < 1 else 0]
    return [state(i), state((i + 97) % 256)]


def expected(task, a, b, i):
    p, re, im = a
    joined = 1 - (1 - p) * (1 - b[0])
    if task == "construction":
        return a
    if task == "transform":
        return [1 - p, re, -im] if i % 2 == 0 else [p, 0, 0]
    if task == "measurement":
        return [p if i % 2 == 0 else joined, 0, 0]
    return [1 - joined if task == "combined" else joined, 0, 0]


def source(task, smoke=False):
    # Formula corpus avoids large constant arrays exceeding older compiler registers.
    prefix = """fn make_state(j) {
    let k = j + 42; if (k >= 256) { k = k - 256; }
    let p = k / 255;
    let re = 0; let im = 0;
    if (j >= 64) { re = (j - 128) / 256; }
    if (j >= 128) { im = (127 - j) / 256; }
    state result = state(p: p, d_re: re, d_im: im);
    return result;
}
let results = array_new(0);
let i = 0;
let even = 0;
while (i < 256) {
    let a = make_state(i);
"""
    if smoke:
        prefix = prefix.replace("let a = make_state(i);", "state a = state(p: 0.5, d: -0.5);")
    if task not in ("construction", "transform"):
        prefix += "    let k = i + 97; if (k >= 256) { k = k - 256; }\n"
        prefix += "    state b = state(p: 0.5, d: 0.5);\n" if smoke else "    let b = make_state(k);\n"
    body = {
        "construction": "array_push(results, a);",
        "transform": "if (even == 0) { transform a with invert(); } else { transform a with neutralize(); } array_push(results, a);",
        "append": "array_push(results, append(a, b));",
        "measurement": "let p = 0; if (even == 0) { p = measure a as probability; } else { let d = append(a, b); p = measure d as probability; } array_push(results, p);",
        "sampling": "let d = append(a, b); let draw = sample(d); array_push(results, sample_value(draw));",
        "combined": "let d = append(a, b); transform d with invert(); let p = measure d as probability; array_push(results, p);",
    }[task]
    return prefix + "    " + body + "\n    i = i + 1; even = 1 - even;\n}\nreturn results;\n"


def scores(result):
    units = result["units"]
    if not isinstance(units, int) or units <= 0 or result["elapsed_ns"] <= 0:
        raise ValueError("incomplete trial")
    def ratio(denominator):
        return units / denominator if denominator and math.isfinite(denominator) and denominator > 0 else None
    return {"units_per_cycle": ratio(result.get("cycles")),
            "units_per_bit": ratio(result.get("memory_bits")),
            "units_per_allocation": ratio(result.get("allocations")),
            "units_per_retained_byte": ratio(result.get("retained_bytes")),
            "units_per_second": units * 1e9 / result["elapsed_ns"]}


def execute(directory, task, batches, seed):
    result = json.loads(run([directory / "driver", directory / f"{task}.labc",
                            directory.parent.parent / "corpus" / f"{task}.txt",
                            batches, seed, int(task in ("sampling", "smoke"))], timeout=120))
    if result["units"] != 256 * batches:
        raise ValueError("incomplete trial")
    result["memory_bits"] = None
    result.update(scores(result))
    return result


def prepare(out):
    corpus = out / "corpus"
    corpus.mkdir()
    pairs = [inputs(i) for i in range(256)]
    (corpus / "inputs.json").write_text(json.dumps(pairs, indent=2) + "\n")
    for task in (*TASKS, "smoke"):
        (corpus / f"{task}.lana").write_text(source("sampling" if task == "smoke" else task, task == "smoke"))
        values = [[0.75, 0, 0]] * 256 if task == "smoke" else [expected(task, *pair, i) for i, pair in enumerate(pairs)]
        value_type = 4 if task in ("construction", "transform", "sampling", "smoke") else 11 if task == "append" else 1
        (corpus / f"{task}.txt").write_text("".join(" ".join(format(x, ".17g") for x in row) + f" {value_type}\n" for row in values))
    return corpus


def build(out, version, commit, corpus):
    directory = out / "versions" / version
    directory.mkdir(parents=True)
    checkout = directory / "source"
    checkout.mkdir()
    # git archive reads pinned objects without changing tags, branches or the worktree.
    archive = directory / "source.tar"
    run(["git", "-C", ROOT, "archive", commit, "-o", archive])
    run(["tar", "-xf", archive, "-C", checkout])
    if (checkout / "VERSION").read_text().strip() != version:
        raise ValueError("pinned VERSION mismatch")
    build_dir = directory / "build"
    run(["cmake", "-S", checkout, "-B", build_dir, "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_OSX_ARCHITECTURES=arm64", "-DCMAKE_C_COMPILER=" + run(["xcrun", "--find", "clang"]).strip()], log=directory / "configure.log")
    run(["cmake", "--build", build_dir, "--target", "lana", "--parallel", "4"], log=directory / "build.log")
    include = ["-DLANA_OLD_HEADERS", "-I", checkout / "include"] if version != "2.0.0" else ["-I", checkout / "vm/include", "-I", checkout / "runtime/include", "-I", checkout / "tools/include"]
    run(["xcrun", "clang", "-std=c11", "-D_DARWIN_C_SOURCE", "-O3", "-DNDEBUG", "-Wall", "-Wextra", "-Wpedantic", "-Werror", *include,
         ROOT / "scripts/version_metrics_driver.c", build_dir / "liblanaruntime.a", "-lm", "-lpthread", "-o", directory / "driver"], log=directory / "driver-build.log")
    for task in (*TASKS, "smoke"):
        run([build_dir / "lana", "compile", corpus / f"{task}.lana", "-o", directory / f"{task}.labc"], log=directory / f"{task}-compile.log")
    return directory


def report(out, rows):
    text = ["# Lana state-core comparison", "", "One unit is one completed task instance; tasks are scored separately.", "",
            "CPU cycles: macOS proc_pid_rusage RUSAGE_INFO_V4 ri_cycles, process-wide hardware counts, including kernel work and counter-read overhead. No frequency conversion or sampling estimate.",
            "Transfer efficiency: unavailable. No validated CPU-to-DRAM read/write byte counter was found; no proxy is substituted.",
            "Allocation efficiency: completed semantic units divided by VM allocation count. Retained-heap efficiency: completed units divided by VM live heap bytes after execution. Neither is memory-transfer efficiency.",
            "Intervals: VM initialization + execution, then cleanup. Bytecode loading, output validation and printing are excluded. Input generation, result storage and loop overhead are included.", "",
            "| Task | Version | Correct | Units/cycle median [min, max] | vs 1.0 | vs previous | Tasks/s median |",
            "|---|---|---|---|---|---|---|"]
    for task in TASKS:
        baseline = previous = None
        for version in VERSIONS:
            trials = [r for r in rows if r["task"] == task and r["version"] == version]
            values = [r["units_per_cycle"] for r in trials if r["units_per_cycle"] is not None]
            value = statistics.median(values) if len(values) == 5 else None
            if version == "1.0.0": baseline = value
            delta = lambda other: f"{100*(value/other-1):+.1f}%" if value and other else "unavailable"
            display = f"{value:.6g} [{min(values):.6g}, {max(values):.6g}]" if value else "unavailable"
            text.append(f"| {task} | {version} | pass | {display} | {delta(baseline)} | {delta(previous)} | {statistics.median(r['units_per_second'] for r in trials):,.0f} |")
            previous = value
    text += ["", "| Task | Version | Allocations/unit median | Units/allocation median | Retained bytes/unit median | Units/retained byte median |",
             "|---|---|---:|---:|---:|---:|"]
    for task in TASKS:
        for version in VERSIONS:
            trials = [r for r in rows if r["task"] == task and r["version"] == version]
            per_unit = lambda key: statistics.median(float(r[key]) / float(r["units"]) for r in trials)
            score = lambda key: statistics.median(float(r[key]) for r in trials)
            text.append(f"| {task} | {version} | {per_unit('allocations'):.6g} | {score('units_per_allocation'):.6g} | {per_unit('retained_bytes'):.6g} | {score('units_per_retained_byte'):.6g} |")
    text += ["", "Five trials, seeds 42–46, rotating version order; one warm-up before each task/version's trials. Repetitions calibrated once on 1.0.0 and frozen. Normal warm-cache behavior; no cache flush or core pinning. Treat overlapping trial ranges as inconclusive.",
             "", "Sampling checks: every sampled state's probability and validity, same-seed repeatability within each release, and 4,096 draws from a symmetric APPEND kernel with each disposition-coordinate mean within 0.06 of zero. This is a smoke check, not a proof of the full distribution.",
             "", "Timing includes cleanup after validation; append validation can warm distribution caches before cleanup. This protocol is identical across releases. No weighted overall score; these are machine-local measurements, not release certification."]
    (out / "report.md").write_text("\n".join(text) + "\n")


def self_test():
    assert scores({"units": 1000, "elapsed_ns": 1000000000, "cycles": 20000, "memory_bits": 80000,
                   "allocations": 250, "retained_bytes": 500}) == {"units_per_cycle": .05, "units_per_bit": .0125,
                                                                         "units_per_allocation": 4, "units_per_retained_byte": 2,
                                                                         "units_per_second": 1000}
    for missing in (None, 0, -1):
        assert scores({"units": 1, "elapsed_ns": 1, "cycles": missing, "memory_bits": missing,
                       "allocations": missing, "retained_bytes": missing})["units_per_cycle"] is None
    try:
        scores({"units": 0, "elapsed_ns": 1})
        raise AssertionError("accepted incomplete work")
    except ValueError:
        pass
    assert len({tuple(inputs(i)[0]) for i in range(256)}) == 256
    assert expected("combined", [.5, 0, 0], [.5, 0, 0], 0) == [.25, 0, 0]
    print("self-test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new output directory (default: temporary directory)")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("this runner targets native arm64 macOS")
    if "AC Power" not in run(["pmset", "-g", "batt"]):
        parser.error("connect AC power before comparing releases")
    out = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix="lana-metrics-"))
    if args.output:
        out.mkdir(parents=True, exist_ok=False)
    print(f"Results: {out}", flush=True)
    corpus = prepare(out)
    manifest = {"versions": VERSIONS, "machine": run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip(),
                "os": platform.platform(), "clang": run(["xcrun", "clang", "--version"]),
                "cmake": run(["cmake", "--version"]), "power": run(["pmset", "-g", "custom"]),
                "battery": run(["pmset", "-g", "batt"]), "corpus_sha256": digest(corpus / "inputs.json"),
                "sources": {p.name: digest(p) for p in corpus.glob("*.lana")},
                "driver_sha256": digest(ROOT / "scripts/version_metrics_driver.c"),
                "cycle_counter": "proc_pid_rusage(RUSAGE_INFO_V4).ri_cycles; process user+kernel; unsampled",
                "memory_counter": "unavailable; CPU-to-DRAM bytes not validated"}
    directories = {}
    for version, commit in VERSIONS.items():
        print(f"Building {version} at {commit[:7]}", flush=True)
        directories[version] = build(out, version, commit, corpus)
        for task in TASKS:
            execute(directories[version], task, 1, 42)
        first = execute(directories[version], "sampling", 2, 42)
        second = execute(directories[version], "sampling", 2, 42)
        if first["last"] != second["last"] or first["mean_re"] != second["mean_re"] or first["mean_im"] != second["mean_im"]:
            raise ValueError("same-seed sampling is not repeatable")
        smoke = execute(directories[version], "smoke", 16, 42)
        if abs(smoke["mean_re"]) > .06 or abs(smoke["mean_im"]) > .06:
            raise ValueError("sampling statistical smoke check failed")
        (directories[version] / "sampling-check.json").write_text(json.dumps(smoke, indent=2) + "\n")
        # Real driver negative check: wrong expected results must reject the batch.
        bad = corpus / "incorrect.txt"
        bad.write_text("2 0 0 4\n" * 256)
        negative = subprocess.run([str(directories[version] / "driver"), str(directories[version] / "construction.labc"), str(bad), "1", "42", "0"], capture_output=True)
        if negative.returncode != 1:
            raise ValueError("driver did not reject incorrect output")
    counts = {}
    rows = []
    with (out / "trials.csv").open("w", newline="") as output:
        fields = ["task", "version", "seed", "batches", "units", "elapsed_ns", "cycles", "memory_bits", "allocations", "retained_bytes", "units_per_cycle", "units_per_bit", "units_per_allocation", "units_per_retained_byte", "units_per_second"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for task in TASKS:
            calibration = execute(directories["1.0.0"], task, 4, 42)
            counts[task] = max(1, min(10000, math.ceil(4 * 500000000 / calibration["elapsed_ns"])))
            for directory in directories.values():
                execute(directory, task, 1, 42)
            for trial in range(5):
                order = list(VERSIONS)
                order = order[trial % 3:] + order[:trial % 3]
                for version in order:
                    result = execute(directories[version], task, counts[task], 42 + trial)
                    row = {"task": task, "version": version, "seed": 42 + trial, "batches": counts[task], **{k: result[k] for k in fields[4:]}}
                    rows.append(row); writer.writerow(row); output.flush()
            print(f"Completed {task}: {counts[task]*256:,} units per trial", flush=True)
    manifest["batches"] = counts
    manifest["power_after"] = run(["pmset", "-g", "custom"])
    manifest["battery_after"] = run(["pmset", "-g", "batt"])
    manifest["bytecode_sha256"] = {f"{v}/{p.name}": digest(p) for v, d in directories.items() for p in d.glob("*.labc")}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report(out, rows)
    print(out / "report.md", flush=True)


if __name__ == "__main__":
    main()
