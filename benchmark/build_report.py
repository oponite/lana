"""Build the canonical portable report artifact from validated evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3


RESULTS = Path(__file__).parent / "results"


def sql_literal(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def sql_dataset(source_id: str, label: str, rows: list[dict], executed_at: str) -> tuple[dict, list[dict]]:
    columns = list(rows[0])
    selects = []
    for row_index, row in enumerate(rows):
        values = []
        for column in columns:
            alias = f' AS "{column}"' if row_index == 0 else ""
            values.append(sql_literal(row[column]) + alias)
        selects.append("SELECT " + ", ".join(values))
    query = "\nUNION ALL\n".join(selects)
    connection = sqlite3.connect(":memory:")
    try:
        cursor = connection.execute(query)
        executed = [dict(zip(columns, values)) for values in cursor.fetchall()]
    finally:
        connection.close()
    source = {
        "id": source_id, "label": label,
        "query": {
            "engine": "SQLite", "language": "SQL", "sql": query,
            "description": f"Materialize the validated {label.lower()} rows used in the report.",
            "executed_at": executed_at,
            "filters": ["Deterministic benchmark snapshot; no live filters"],
            "metric_definitions": ["Values are copied from validated benchmark/results/evidence.json"],
            "tables_used": [],
        },
    }
    return source, executed


def mean_metric(behavior: list[dict], implementation: str, metric: str) -> float:
    rows = [row for row in behavior if row["implementation"] == implementation and row["mode"] == "dynamic"]
    return sum(row[metric] for row in rows) / len(rows)


def large_runtime(runtime: list[dict], implementation: str) -> dict:
    return next(row for row in runtime if row["implementation"] == implementation and row["size"] == "large")


def main() -> None:
    evidence = json.loads((RESULTS / "evidence.json").read_text())
    validation = json.loads((RESULTS / "validation.json").read_text())
    complexity = evidence["complexity"]
    behavior = evidence["behavior"]
    runtime = evidence["runtime"]
    generated_at = datetime.now(timezone.utc).isoformat()

    labels = ("Python Plain", "Python State Class", "Lana")
    complexity_rows = [{
        "implementation": label,
        "core_loc": complexity[label]["loc"],
        "end_to_end_loc": complexity[label]["end_to_end_tooling_loc"],
        "tokens": complexity[label]["tokens"],
        "cyclomatic": complexity[label]["cyclomatic"],
        "state_operations": complexity[label]["state_management_operations"],
    } for label in labels]
    metric_rows = []
    metric_specs = (
        ("Core policy LOC", "loc", "number"),
        ("Core policy tokens", "tokens", "number"),
        ("End-to-end tooling LOC", "end_to_end_tooling_loc", "number"),
        ("Explicit state variables", "explicit_state_variables", "number"),
        ("Update functions", "update_functions", "number"),
        ("Cyclomatic complexity", "cyclomatic", "number"),
        ("State-management operations", "state_management_operations", "number"),
    )
    for label, field, _ in metric_specs:
        metric_rows.append({
            "metric": label,
            "python_plain": complexity["Python Plain"][field],
            "python_state": complexity["Python State Class"][field],
            "lana": complexity["Lana"][field],
        })
    metric_rows.extend((
        {
            "metric": "Mean Brier score, dynamic",
            "python_plain": round(mean_metric(behavior, "Python Plain", "brier"), 6),
            "python_state": round(mean_metric(behavior, "Python State Class", "brier"), 6),
            "lana": round(mean_metric(behavior, "Lana", "brier"), 6),
        },
        {
            "metric": "Mean log loss, dynamic",
            "python_plain": round(mean_metric(behavior, "Python Plain", "log_loss"), 6),
            "python_state": round(mean_metric(behavior, "Python State Class", "log_loss"), 6),
            "lana": round(mean_metric(behavior, "Lana", "log_loss"), 6),
        },
        {
            "metric": "4k-step core runtime, ms",
            "python_plain": round(large_runtime(runtime, "Python Plain")["core_elapsed_ms"], 3),
            "python_state": round(large_runtime(runtime, "Python State Class")["core_elapsed_ms"], 3),
            "lana": round(large_runtime(runtime, "Lana")["core_elapsed_ms"], 3),
        },
        {
            "metric": "4k-step max RSS, MiB",
            "python_plain": round(large_runtime(runtime, "Python Plain")["max_rss_bytes_median"] / 1048576, 2),
            "python_state": round(large_runtime(runtime, "Python State Class")["max_rss_bytes_median"] / 1048576, 2),
            "lana": round(large_runtime(runtime, "Lana")["max_rss_bytes_median"] / 1048576, 2),
        },
    ))
    ablation_rows = []
    for layer in "ABC":
        dynamic = next(row for row in behavior if row["layer"] == layer and row["implementation"] == "Lana" and row["mode"] == "dynamic")
        fixed = next(row for row in behavior if row["layer"] == layer and row["implementation"] == "Lana" and row["mode"] == "fixed")
        zero = next(row for row in behavior if row["layer"] == layer and row["implementation"] == "Lana" and row["mode"] == "zero")
        p_only = next(row for row in behavior if row["layer"] == layer and row["mode"] == "p_only")
        ablation_rows.append({
            "layer": layer,
            "dynamic_brier": round(dynamic["brier"], 6),
            "fixed_d_brier": round(fixed["brier"], 6),
            "p_only_brier": round(p_only["brier"], 6),
            "d_zero_brier": round(zero["brier"], 6),
            "dynamic_accuracy": round(dynamic["accuracy"], 4),
            "p_only_accuracy": round(p_only["accuracy"], 4),
            "dynamic_response_steps": dynamic["regime_response_steps"],
            "p_only_response_steps": p_only["regime_response_steps"],
        })
    runtime_rows = [{
        "implementation": row["implementation"], "size": row["size"],
        "transitions": row["transitions"],
        "core_ms": round(row.get("core_elapsed_ms", 0.0), 3),
        "process_ms": round(row["repeated_wall_ms_median"], 3),
        "max_rss_mib": round(row["max_rss_bytes_median"] / 1048576, 2),
        "instruction_count": row.get("instruction_count") or 0,
        "allocation_count": row["allocation_count"],
    } for row in runtime]
    vm_rows = [{
        "layer": layer, "instructions": stats["instructions"],
        "state_transitions": stats["state_transitions"],
        "apply": stats["opcodes"]["APPLY"],
        "apply_many": stats["opcodes"]["APPLY_MANY"],
        "transform": stats["opcodes"]["TRANSFORM"],
        "measure": stats["opcodes"]["MEASURE"],
        "branches": stats["opcodes"]["JUMP_IF_FALSE"],
    } for layer, stats in evidence["vm_layer_stats"].items()]
    stress_rows = [{
        "case": row["case"], "transitions": row["transitions"],
        "final_p": round(row["lana_final_p"], 12),
        "min_p": round(row["min_p"], 12), "max_p": round(row["max_p"], 12),
        "parity_error": row["parity_error"],
    } for row in evidence["stress"]]
    maintenance_rows = evidence["maintenance"]

    source_evidence = {"id": "evidence", "label": "Benchmark evidence", "path": "evidence.json"}
    source_predictions = {"id": "predictions", "label": "Saved prediction rows", "path": "predictions.csv"}
    dataset_inputs = {
        "complexity": complexity_rows, "metrics": metric_rows, "ablation": ablation_rows,
        "runtime": runtime_rows, "vm": vm_rows, "stress": stress_rows,
        "maintenance": maintenance_rows,
    }
    dataset_sources: list[dict] = []
    datasets: dict[str, list[dict]] = {}
    for dataset_id, rows in dataset_inputs.items():
        source, executed = sql_dataset(f"source_{dataset_id}", dataset_id.replace("_", " ").title(), rows, generated_at)
        dataset_sources.append(source)
        datasets[dataset_id] = executed
    sources = [source_evidence, source_predictions, *dataset_sources]

    title = "Lana vs Conventional Python: Falsification Benchmark"
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {title}"},
        {"id": "summary", "type": "markdown", "sourceId": "evidence", "body":
         "## Technical summary\n\n"
         "**Verdict: weak evidence of a Lana advantage.** Lana compressed the hand-authored core policy to 43 LOC versus 72 for a Python `State` class and 98 for plain Python. Its control surface was smaller, native VM operations preserved exact behavior, and the precompiled C VM ran this workload faster with lower process memory. However, the Python class required only 72 end-to-end LOC versus 91 for Lana's current code generator and changed more cheaply in both maintainability tests.\n\n"
         "`d` earned only provisional support: it improved Brier score and regime response in Layers A and B, but the p-only alternative slightly beat it on Layer C Brier, log loss, accuracy, and calibration. The language idea survives as an abstraction experiment, not yet as a demonstrated advantage over a good Python class."},
        {"id": "compression_intro", "type": "markdown", "sourceId": "evidence", "body":
         "## Native state compresses the core, not the complete workflow\n\n"
         "The core-policy LOC compression ratio is 2.28× versus plain Python and 1.67× versus the Python class. The syntax-count SMCR is 1.31× and 1.36× respectively. Those ratios capture visible coordination work; they do not measure semantic difficulty or generated-code cost. Once the required Python code generator is included, Lana uses 91 LOC—19 more than the class baseline."},
        {"id": "complexity_chart", "type": "chart", "chartId": "complexity_chart"},
        {"id": "metrics_intro", "type": "markdown", "body":
         "The exact comparison below combines code, control, behavior, runtime, and memory. Lower is better for every row except none; behavioral parity makes the dynamic Brier and log-loss rows identical by design."},
        {"id": "metrics_table", "type": "table", "tableId": "metrics_table"},
        {"id": "ablation_intro", "type": "markdown", "sourceId": "evidence", "body":
         "## `d` helps adaptation, but does not consistently improve forecasts\n\n"
         "Dynamic `d` beat a fixed-rate p-only updater on Brier score in Layers A and B and crossed regime thresholds sooner in every layer. In Layer C, p-only was marginally better on Brier and log loss and materially better on calibration. Setting `d = 0` froze `apply`, confirming that current `apply` semantics make nonzero influence essential—but that is a semantic consequence, not independent evidence that `d` belongs in the primitive."},
        {"id": "ablation_table", "type": "table", "tableId": "ablation_table"},
        {"id": "runtime_intro", "type": "markdown", "sourceId": "evidence", "body":
         "## The precompiled C VM is faster and leaner on this workload\n\n"
         "At 4,000 steps, Lana used less process memory, started faster, and executed the core faster than both Python implementations. This is a narrow VM result: source compilation is excluded, Python reads CSV while Lana loads precompiled bytecode, and the Python baselines were not optimized with native extensions. The 10,000-step unrolled case hit SSBC v1's 100,000-constant loader ceiling, so the large comparison uses 4,000 steps; a compact loop still completed one million `APPLY` transitions safely."},
        {"id": "runtime_table", "type": "table", "tableId": "runtime_table"},
        {"id": "vm_intro", "type": "markdown", "sourceId": "evidence", "body":
         "## VM instructions make semantic compression inspectable\n\n"
         "`APPLY`, `APPLY_MANY`, `TRANSFORM`, and `MEASURE` remain visible as native bytecode. Layer B required 96 aggregate operations instead of exposing the weighting loop in source. The VM's state-transition counter is much higher than prediction steps because every state creation, metadata write, transform, and update stores a validated state."},
        {"id": "vm_table", "type": "table", "tableId": "vm_table"},
        {"id": "stress_intro", "type": "markdown", "sourceId": "evidence", "body":
         "## Stress tests find stable arithmetic and one remaining semantic hazard\n\n"
         "The VM matched the independent Python formula within floating-point print precision across all stress cases and one million updates. Negative `d` now converges toward the source's inverse probability instead of extrapolating to a boundary. Repeated high-`d` contradictions still oscillate rather than settling, so aggregation and conflict behavior need explicit guidance."},
        {"id": "stress_table", "type": "table", "tableId": "stress_table"},
        {"id": "maintenance_intro", "type": "markdown", "sourceId": "evidence", "body":
         "## Maintainability does not beat the Python class\n\n"
         "The stale-evidence change added six Lana lines, four plain-Python lines, and three class-based lines. Adding three reliable sources changed six lines in all three versions. Native operations made the result readable, but did not reduce change propagation in these tests."},
        {"id": "maintenance_table", "type": "table", "tableId": "maintenance_table"},
        {"id": "scope", "type": "markdown", "body":
         "## Scope, data, and metric definitions\n\n"
         "The benchmark uses 96 sequential predictions per layer from one deterministic synthetic CSV (seed 1729). Evidence signals are generated from a hidden regime probability before outcomes are drawn; they never use the realized target. Each full-state implementation receives identical rows and update order. Brier score is mean squared probabilistic error; log loss uses 1e-12 clipping; calibration error uses ten equal-width bins. LCR is Python LOC divided by Lana LOC. SMCR is Python explicit state-management syntax occurrences divided by Lana native state-operation occurrences."},
        {"id": "methodology", "type": "markdown", "body":
         "## Methodology\n\n"
         "Layer A uses one belief and one stream. Layer B adds three weighted, confidence-indexed, stale and conflicting sources plus bounded history. Layer C adds four interacting states, conditional influence, aggregation, decay, and repeated observations. Plain Python uses dictionaries and functions; the class baseline uses dataclasses and methods; Lana programs are generated from the same CSV because v1 lacks file I/O and dynamic state construction. Generated data and generated Lana programs are excluded from core LOC but the generator is included in end-to-end LOC. Cold runtime is the first process; repeated runtime is the median of five subsequent processes."},
        {"id": "limitations", "type": "markdown", "body":
         "## Limitations, uncertainty, and robustness checks\n\n"
         "This is synthetic evidence, not real-world predictive validation. Ninety-six steps per layer make calibration estimates noisy. SMCR is a transparent syntax count, not a validated cognitive-complexity scale. RSS and allocation mechanisms differ across runtimes. Python reads CSV while Lana loads precompiled bytecode, so process-wall timing is informative but not a pure engine comparison; core timing is the better runtime comparison. No human participants were available, so comprehension results are intentionally absent. The blinded harness is ready for later use."},
        {"id": "next", "type": "markdown", "body":
         "## Recommended next steps\n\n"
         "1. Add property tests for the locked inverse-`d` rule and define how mixed-sign aggregation should resolve conflict.\n2. Add native data iteration/dynamic state construction, then rerun LOC and maintainability without a Python code generator.\n3. Tune `d` only on a calibration window and evaluate on a held-out stream; compare against learned fixed-alpha and Bayesian/log-odds baselines.\n4. Run the blinded comprehension test with at least 12 programmers and stratify by Python and language-design experience."},
        {"id": "questions", "type": "markdown", "body":
         "## Further questions\n\n"
         "Should mixed positive and negative influences cancel, average their effective targets, or preserve separate conflict? Should repeated contradictory evidence converge toward uncertainty rather than oscillate? Can a library-level Python protocol reproduce Lana's readability once the benchmark includes real ingestion and persistence?"},
    ]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1, "surface": "report", "title": title,
            "description": "A rigorous falsification benchmark of Lana STATE(p,d) against conventional Python.",
            "generatedAt": generated_at, "blocks": blocks, "sources": sources,
            "charts": [{
                "id": "complexity_chart", "title": "Hand-authored policy size",
                "subtitle": "Nonblank, noncomment core LOC; generated data and code excluded",
                "showDescription": True, "type": "bar", "intent": "comparison",
                "question": "How much hand-authored code expresses the three benchmark layers?",
                "rationale": "A bar chart directly compares absolute policy LOC across three implementations.",
                "dataset": "complexity", "sourceId": "source_complexity", "valueFormat": "number",
                "encodings": {
                    "x": {"field": "implementation", "type": "nominal", "label": "Implementation"},
                    "y": {"field": "core_loc", "type": "quantitative", "label": "Core LOC"},
                    "tooltip": [
                        {"field": "end_to_end_loc", "type": "quantitative", "label": "End-to-end LOC"},
                        {"field": "cyclomatic", "type": "quantitative", "label": "Cyclomatic complexity"},
                        {"field": "state_operations", "type": "quantitative", "label": "State operations"},
                    ],
                },
            }],
            "tables": [
                {"id": "metrics_table", "title": "Cross-implementation metrics", "subtitle": "Same 96-step windows per layer; lower values are preferable", "showDescription": True, "dataset": "metrics", "sourceId": "source_metrics", "defaultSort": {"field": "metric", "direction": "asc"}, "density": "dense", "columns": [
                    {"field": "metric", "label": "Metric", "type": "text"}, {"field": "python_plain", "label": "Python Plain"}, {"field": "python_state", "label": "Python State Class"}, {"field": "lana", "label": "Lana"},
                ]},
                {"id": "ablation_table", "title": "Ablation results", "subtitle": "Brier score and regime response across 96 predictions per layer", "showDescription": True, "dataset": "ablation", "sourceId": "source_ablation", "defaultSort": {"field": "layer", "direction": "asc"}, "columns": [
                    {"field": "layer", "label": "Layer", "type": "text"}, {"field": "dynamic_brier", "label": "Dynamic d Brier"}, {"field": "fixed_d_brier", "label": "Fixed d Brier"}, {"field": "p_only_brier", "label": "p-only Brier"}, {"field": "d_zero_brier", "label": "d=0 Brier"}, {"field": "dynamic_accuracy", "label": "Dynamic accuracy"}, {"field": "p_only_accuracy", "label": "p-only accuracy"}, {"field": "dynamic_response_steps", "label": "Dynamic response"}, {"field": "p_only_response_steps", "label": "p-only response"},
                ]},
                {"id": "runtime_table", "title": "Runtime and memory", "subtitle": "Cold/repeated process runs; core timing excludes input loading", "showDescription": True, "dataset": "runtime", "sourceId": "source_runtime", "defaultSort": {"field": "transitions", "direction": "asc"}, "density": "dense", "columns": [
                    {"field": "implementation", "label": "Implementation", "type": "text"}, {"field": "size", "label": "Size", "type": "text"}, {"field": "transitions", "label": "Steps"}, {"field": "core_ms", "label": "Core ms"}, {"field": "process_ms", "label": "Process ms"}, {"field": "max_rss_mib", "label": "Max RSS MiB"}, {"field": "instruction_count", "label": "VM instructions"}, {"field": "allocation_count", "label": "Allocations"},
                ]},
                {"id": "vm_table", "title": "VM operation counts", "subtitle": "Dynamic-d runs across 96 prediction steps per layer", "showDescription": True, "dataset": "vm", "sourceId": "source_vm", "defaultSort": {"field": "layer", "direction": "asc"}, "density": "dense", "columns": [
                    {"field": "layer", "label": "Layer", "type": "text"}, {"field": "instructions", "label": "Instructions"}, {"field": "state_transitions", "label": "State stores"}, {"field": "apply", "label": "APPLY"}, {"field": "apply_many", "label": "APPLY_MANY"}, {"field": "transform", "label": "TRANSFORM"}, {"field": "measure", "label": "MEASURE"}, {"field": "branches", "label": "Branches"},
                ]},
                {"id": "stress_table", "title": "Stress-test outcomes", "subtitle": "Independent Python formula and C VM agree within floating-point print precision", "showDescription": True, "dataset": "stress", "sourceId": "source_stress", "defaultSort": {"field": "transitions", "direction": "desc"}, "density": "dense", "columns": [
                    {"field": "case", "label": "Case", "type": "text"}, {"field": "transitions", "label": "Transitions"}, {"field": "final_p", "label": "Final p"}, {"field": "min_p", "label": "Minimum p"}, {"field": "max_p", "label": "Maximum p"}, {"field": "parity_error", "label": "Parity error"},
                ]},
                {"id": "maintenance_table", "title": "Requirement-change cost", "subtitle": "Actual before/after snippets for stale evidence and three sources", "showDescription": True, "dataset": "maintenance", "sourceId": "source_maintenance", "defaultSort": {"field": "change", "direction": "asc"}, "columns": [
                    {"field": "change", "label": "Change", "type": "text"}, {"field": "implementation", "label": "Implementation", "type": "text"}, {"field": "lines_added", "label": "Added"}, {"field": "lines_removed", "label": "Removed"}, {"field": "lines_changed", "label": "Changed"}, {"field": "files_changed", "label": "Files"}, {"field": "new_names", "label": "New names"}, {"field": "tests_modified", "label": "Tests modified"},
                ]},
            ],
        },
        "snapshot": {
            "version": 1, "generatedAt": generated_at, "status": "ready",
            "datasets": datasets,
        },
        "sources": sources,
        "package_info": {
            "validation_status": validation["status"],
            "evidence_sha256": evidence["data"]["sha256"],
            "report_notes": [
                "Technical audience structure used.",
                "One comparison chart selected; exact multi-metric results use audit tables.",
                "No human comprehension results fabricated.",
            ],
        },
    }
    (RESULTS / "artifact.json").write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"artifact": str(RESULTS / "artifact.json"), "verdict": "weak evidence"}))


if __name__ == "__main__":
    main()
