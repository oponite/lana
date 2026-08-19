"""Competent conventional Python using dictionaries, floats, lists, and functions."""

from __future__ import annotations


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def _decay(state: dict, rate: float) -> None:
    state["p"] += rate * (0.5 - state["p"])
    state["d"] *= 1.0 - rate


def _apply(source: dict, target: dict) -> None:
    influence_target = 1.0 - source["p"] if source["d"] < 0.0 else source["p"]
    target["p"] = _clip(target["p"] + abs(source["d"]) * (influence_target - target["p"]))


def _evidence(event: dict, mode: str) -> dict:
    dependency = event["d"] if mode == "dynamic" else 0.28 if mode == "fixed" else 0.0
    state = {
        "p": event["signal_p"], "d": dependency, "timestamp": event["t"],
        "source": event["source"], "weight": event["weight"],
        "confidence": event["confidence"],
    }
    _decay(state, min(event["age"] / 120.0, 0.8))
    return state


def _aggregate(states: list[dict]) -> dict:
    weights = [state["weight"] * state["confidence"] for state in states]
    total = sum(weights)
    return {
        "p": sum(state["p"] * weight for state, weight in zip(states, weights)) / total,
        "d": sum(state["d"] * weight for state, weight in zip(states, weights)) / total,
    }


def predict(steps: list[list[dict]], layer: str, mode: str = "dynamic") -> list[float]:
    if mode == "p_only":
        return predict_p_only(steps, layer)
    predictions: list[float] = []
    if layer == "A":
        belief = {"p": 0.5, "d": 0.20}
        for events in steps:
            _apply(_evidence(events[0], mode), belief)
            predictions.append(belief["p"])
        return predictions
    if layer == "B":
        belief = {"p": 0.5, "d": 0.20, "history": [0.5]}
        for events in steps:
            _decay(belief, 0.015)
            _apply(_aggregate([_evidence(event, mode) for event in events]), belief)
            belief["history"].append(belief["p"])
            belief["history"] = belief["history"][-64:]
            predictions.append(belief["p"])
        return predictions
    node_d = (0.35, 0.25, 0.30, 0.15) if mode == "dynamic" else ((0.28,) * 4 if mode == "fixed" else (0.0,) * 4)
    a, b, c, target = ({"p": 0.5, "d": dependency} for dependency in node_d)
    target["history"] = [0.5]
    for events in steps:
        for state in (a, b, c, target):
            _decay(state, 0.01)
        first, second, conditional = [_evidence(event, mode) for event in events]
        _apply(first, a)
        _apply(second, b)
        a.update(weight=events[0]["weight"], confidence=events[0]["confidence"])
        b.update(weight=events[1]["weight"], confidence=events[1]["confidence"])
        _apply(_aggregate([a, b]), c)
        _apply(c, target)
        if c["p"] > 0.55:
            _apply(conditional, target)
        target["history"].append(target["p"])
        target["history"] = target["history"][-64:]
        predictions.append(target["p"])
    return predictions


def predict_p_only(steps: list[list[dict]], layer: str, alpha: float = 0.28) -> list[float]:
    """Strong ablation: ordinary probabilities with an explicit fixed update rate."""
    predictions: list[float] = []

    def update(probability: float, evidence: float, rate: float = alpha) -> float:
        return _clip(probability + rate * (evidence - probability))

    if layer == "A":
        belief = 0.5
        for events in steps:
            belief = update(belief, events[0]["signal_p"])
            predictions.append(belief)
        return predictions
    if layer == "B":
        belief = 0.5
        for events in steps:
            belief += 0.015 * (0.5 - belief)
            evidence = [_evidence(event, "fixed") for event in events]
            belief = update(belief, _aggregate(evidence)["p"])
            predictions.append(belief)
        return predictions
    a = b = c = target = 0.5
    for events in steps:
        a += 0.01 * (0.5 - a)
        b += 0.01 * (0.5 - b)
        c += 0.01 * (0.5 - c)
        target += 0.01 * (0.5 - target)
        a = update(a, events[0]["signal_p"])
        b = update(b, events[1]["signal_p"])
        weights = [events[0]["weight"] * events[0]["confidence"], events[1]["weight"] * events[1]["confidence"]]
        aggregate = (a * weights[0] + b * weights[1]) / sum(weights)
        c = update(c, aggregate)
        target = update(target, c, 0.15)
        if c > 0.55:
            target = update(target, events[2]["signal_p"])
        predictions.append(target)
    return predictions
