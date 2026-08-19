from __future__ import annotations

from structured_state.vm.machine import clip
from structured_state.vm.values import State


def _unit(name: str, value: float) -> float:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def decay(state: State, rate: float) -> State:
    _unit("rate", rate)
    return state.with_values(p1=state.p1 + rate * (0.5 - state.p1), d=state.d * (1 - rate))


def reinforce(state: State, strength: float) -> State:
    _unit("strength", strength)
    return state.with_values(p1=clip(0.5 + (1 + strength) * (state.p1 - 0.5)))


def invert(state: State) -> State:
    return state.with_values(p1=1 - state.p1, d=-state.d)


def neutralize(state: State) -> State:
    return state.with_values(p1=0.5, d=0.0)


def shift(state: State, amount: float) -> State:
    return state.with_values(p1=clip(state.p1 + amount))


def blend(state: State, other: State, weight: float) -> State:
    _unit("weight", weight)
    return state.with_values(p1=(1 - weight) * state.p1 + weight * other.p1, d=(1 - weight) * state.d + weight * other.d)


def clamp(state: State, low: float, high: float) -> State:
    if not 0 <= low <= high <= 1:
        raise ValueError("clamp bounds must satisfy 0 <= low <= high <= 1")
    return state.with_values(p1=clip(state.p1, low, high))


def reset_dependency(state: State) -> State:
    return state.with_values(d=0.0)


def set_dependency(state: State, dependency: float) -> State:
    return state.with_values(d=dependency)


def confidence_weight(state: State) -> State:
    return state.with_values(d=state.d * (state.indexes.confidence if state.indexes.confidence is not None else 1.0))


def time_decay(state: State, now: float, rate: float) -> State:
    _unit("rate", rate)
    age = 0 if state.indexes.timestamp is None else max(0, now - state.indexes.timestamp)
    result = decay(state, clip(rate * age))
    return result.with_values(indexes=result.indexes.set("timestamp", now))


TRANSFORMS = {name: value for name, value in globals().items() if callable(value) and not name.startswith("_") and name not in {"State"}}
