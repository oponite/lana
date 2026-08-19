"""Generate deterministic, domain-neutral sequential evidence."""

from __future__ import annotations

import csv
from pathlib import Path
import random


FIELDS = [
    "layer", "t", "outcome", "latent_p", "regime_switch", "source",
    "signal_p", "weight", "confidence", "age", "d",
]


def clamp(value: float) -> float:
    return max(0.001, min(0.999, value))


def generate(steps: int = 240, seed: int = 1729) -> list[dict[str, object]]:
    rng = random.Random(seed)
    regimes = [0.18, 0.78, 0.35, 0.85, 0.25, 0.68]
    profiles = {
        "alpha": {"noise": 0.11, "bias": 0.00, "weight": 1.00, "confidence": 0.84, "d": 0.42},
        "beta": {"noise": 0.19, "bias": 0.04, "weight": 0.75, "confidence": 0.68, "d": 0.30},
        "gamma": {"noise": 0.15, "bias": -0.03, "weight": 0.90, "confidence": 0.78, "d": 0.36},
    }
    rows: list[dict[str, object]] = []
    segment = max(1, steps // len(regimes))
    for layer in ("A", "B", "C"):
        for t in range(steps):
            regime_index = min(t // segment, len(regimes) - 1)
            latent_p = regimes[regime_index]
            outcome = int(rng.random() < latent_p)
            sources = ("alpha",) if layer == "A" else tuple(profiles)
            for source_index, source in enumerate(sources):
                profile = profiles[source]
                signal = latent_p + float(profile["bias"]) + rng.gauss(0.0, float(profile["noise"]))
                # Deliberate conflicts are properties of the evidence process,
                # not the observed target.
                if source == "gamma" and (t % 13 in {5, 6, 7}):
                    signal = 1.0 - signal
                age = 0.0
                if source == "beta" and t % 11 == 0:
                    age = 50.0
                elif source == "gamma" and t % 17 == 0:
                    age = 80.0
                elif source_index > 0:
                    age = float((t + source_index * 3) % 18)
                confidence = float(profile["confidence"])
                if source == "gamma" and t % 13 in {5, 6, 7}:
                    confidence = 0.93  # conflicting high-confidence evidence
                rows.append({
                    "layer": layer,
                    "t": t,
                    "outcome": outcome,
                    "latent_p": f"{latent_p:.8f}",
                    "regime_switch": int(t > 0 and t % segment == 0),
                    "source": source,
                    "signal_p": f"{clamp(signal):.12f}",
                    "weight": f"{float(profile['weight']):.8f}",
                    "confidence": f"{confidence:.8f}",
                    "age": f"{age:.8f}",
                    "d": f"{float(profile['d']):.8f}",
                })
    return rows


def write_csv(path: Path, steps: int = 240, seed: int = 1729) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(generate(steps=steps, seed=seed))


if __name__ == "__main__":
    write_csv(Path(__file__).parent / "data" / "events.csv")
