from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from structured_state.errors import InvalidDependencyError, InvalidProbabilityError, InvalidStateError

EPSILON = 1e-12


def _number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidStateError(f"{name} must be a number")
    return float(value)


@dataclass(frozen=True, slots=True)
class Indexes:
    timestamp: float | None = None
    source: str | None = None
    weight: float | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.timestamp is not None:
            object.__setattr__(self, "timestamp", _number("timestamp", self.timestamp))
        if self.source is not None and not isinstance(self.source, str):
            raise InvalidStateError("source must be a string")
        if self.weight is not None:
            weight = _number("weight", self.weight)
            if weight < 0:
                raise InvalidStateError("weight must be non-negative")
            object.__setattr__(self, "weight", weight)
        if self.confidence is not None:
            confidence = _number("confidence", self.confidence)
            if not 0 <= confidence <= 1:
                raise InvalidStateError("confidence must be between 0 and 1")
            object.__setattr__(self, "confidence", confidence)

    def get(self, name: str) -> Any:
        if name not in {"timestamp", "source", "weight", "confidence"}:
            raise InvalidStateError(f"unknown state index '{name}'")
        return getattr(self, name)

    def set(self, name: str, value: Any) -> "Indexes":
        if name not in {"timestamp", "source", "weight", "confidence"}:
            raise InvalidStateError(f"unknown state index '{name}'")
        return replace(self, **{name: value})


@dataclass(frozen=True, slots=True)
class State:
    """Canonical Lana state. p0 is retained, not merely inferred externally."""

    p0: float | None = None
    p1: float | None = None
    d: float = 0.0
    indexes: Indexes = field(default_factory=Indexes)

    def __post_init__(self) -> None:
        if self.p0 is None and self.p1 is None:
            raise InvalidProbabilityError("state requires p0 or p1")
        p0 = None if self.p0 is None else _number("p0", self.p0)
        p1 = None if self.p1 is None else _number("p1", self.p1)
        if p1 is None:
            p1 = 1.0 - p0  # type: ignore[operator]
        if p0 is None:
            p0 = 1.0 - p1
        if not 0 <= p0 <= 1 or not 0 <= p1 <= 1:
            raise InvalidProbabilityError("p0 and p1 must be between 0 and 1")
        if abs((p0 + p1) - 1.0) > EPSILON:
            raise InvalidProbabilityError("p0 + p1 must equal 1")
        dependency = _number("d", self.d)
        if not -1 < dependency < 1:
            raise InvalidDependencyError("d must be strictly between -1 and 1")
        if not isinstance(self.indexes, Indexes):
            raise InvalidStateError("indexes must be Indexes")
        object.__setattr__(self, "p1", p1)
        object.__setattr__(self, "p0", 1.0 - p1)
        object.__setattr__(self, "d", dependency)

    def with_values(self, *, p1: float | None = None, d: float | None = None, indexes: Indexes | None = None) -> "State":
        return State(p1=self.p1 if p1 is None else p1, d=self.d if d is None else d, indexes=self.indexes if indexes is None else indexes)


@dataclass(frozen=True, slots=True)
class Distribution:
    p0: float
    p1: float


@dataclass(frozen=True, slots=True)
class JointState:
    left: State
    right: State


@dataclass(frozen=True, slots=True)
class StateDelta:
    p1: float
