"""Conventional Python baseline using a compact, well-designed State class."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class State:
    p: float
    influence: float
    timestamp: float | None = None
    source: str | None = None
    weight: float = 1.0
    confidence: float = 1.0
    history: deque[float] = field(default_factory=lambda: deque(maxlen=64))

    def __post_init__(self) -> None:
        self.validate()
        self.history.append(self.p)

    def validate(self) -> None:
        if not 0.0 <= self.p <= 1.0 or not -1.0 < self.influence < 1.0:
            raise ValueError("invalid state")

    def apply_to(self, target: "State") -> None:
        influence_target = 1.0 - self.p if self.influence < 0.0 else self.p
        target.p = max(0.0, min(1.0, target.p + abs(self.influence) * (influence_target - target.p)))
        target.validate()
        target.history.append(target.p)

    def decay(self, rate: float) -> None:
        self.p += rate * (0.5 - self.p)
        self.influence *= 1.0 - rate
        self.validate()
        self.history.append(self.p)

    @classmethod
    def aggregate(cls, states: list["State"]) -> "State":
        weights = [state.weight * state.confidence for state in states]
        total = sum(weights)
        return cls(
            sum(state.p * weight for state, weight in zip(states, weights)) / total,
            sum(state.influence * weight for state, weight in zip(states, weights)) / total,
        )


def evidence(event: dict, mode: str) -> State:
    influence = event["d"] if mode == "dynamic" else 0.28 if mode == "fixed" else 0.0
    state = State(event["signal_p"], influence, event["t"], event["source"], event["weight"], event["confidence"])
    state.decay(min(event["age"] / 120.0, 0.8))
    return state


def predict(steps: list[list[dict]], layer: str, mode: str = "dynamic") -> list[float]:
    predictions: list[float] = []
    if layer == "A":
        belief = State(0.5, 0.20)
        for events in steps:
            evidence(events[0], mode).apply_to(belief)
            predictions.append(belief.p)
        return predictions
    if layer == "B":
        belief = State(0.5, 0.20)
        for events in steps:
            belief.decay(0.015)
            State.aggregate([evidence(event, mode) for event in events]).apply_to(belief)
            predictions.append(belief.p)
        return predictions
    node_d = (0.35, 0.25, 0.30, 0.15) if mode == "dynamic" else ((0.28,) * 4 if mode == "fixed" else (0.0,) * 4)
    a, b, c, target = (State(0.5, influence) for influence in node_d)
    for events in steps:
        for state in (a, b, c, target):
            state.decay(0.01)
        first, second, conditional = [evidence(event, mode) for event in events]
        first.apply_to(a)
        second.apply_to(b)
        a.weight, a.confidence = events[0]["weight"], events[0]["confidence"]
        b.weight, b.confidence = events[1]["weight"], events[1]["confidence"]
        State.aggregate([a, b]).apply_to(c)
        c.apply_to(target)
        if c.p > 0.55:
            conditional.apply_to(target)
        predictions.append(target.p)
    return predictions
