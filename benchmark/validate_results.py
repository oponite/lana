"""Independently recompute headline metrics from saved prediction rows."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).parent
RESULTS = ROOT / "results"


def main() -> None:
    evidence = json.loads((RESULTS / "evidence.json").read_text())
    reported = {
        (row["layer"], row["mode"], row["implementation"]): row
        for row in evidence["behavior"]
    }
    groups: dict[tuple[str, str, str], list[tuple[float, int]]] = {}
    with (RESULTS / "predictions.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = row["layer"], row["mode"], row["implementation"]
            groups.setdefault(key, []).append((float(row["prediction"]), int(row["outcome"])))
    differences: list[float] = []
    for key, values in groups.items():
        n = len(values)
        brier = sum((prediction - outcome) ** 2 for prediction, outcome in values) / n
        accuracy = sum((prediction >= 0.5) == bool(outcome) for prediction, outcome in values) / n
        log_loss = -sum(
            outcome * math.log(max(1e-12, min(1.0 - 1e-12, prediction))) +
            (1 - outcome) * math.log(max(1e-12, min(1.0 - 1e-12, 1.0 - prediction)))
            for prediction, outcome in values
        ) / n
        differences.extend((
            abs(brier - reported[key]["brier"]),
            abs(accuracy - reported[key]["accuracy"]),
            abs(log_loss - reported[key]["log_loss"]),
        ))
    max_parity = max(evidence["parity"].values())
    max_stress = max(row["parity_error"] for row in evidence["stress"])
    receipt = {
        "status": "passed" if max(differences) < 1e-15 and max_parity < 1e-12 and max_stress < 1e-12 else "failed",
        "groups_recomputed": len(groups), "prediction_rows": sum(map(len, groups.values())),
        "max_metric_recompute_error": max(differences),
        "max_implementation_parity_error": max_parity,
        "max_stress_parity_error": max_stress,
        "checks": ["accuracy", "Brier score", "log loss", "implementation parity", "stress parity"],
    }
    (RESULTS / "validation.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    if receipt["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
