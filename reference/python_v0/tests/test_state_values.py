import pytest

from structured_state.errors import InvalidDependencyError, InvalidProbabilityError, InvalidStateError
from structured_state.vm.values import Indexes, State


def test_state_derives_p0_from_p1_and_retains_indexes() -> None:
    state = State(p1=0.62, d=0.35, indexes=Indexes(timestamp=10, source="sensor", weight=2, confidence=0.8))

    assert state.p0 == pytest.approx(0.38)
    assert state.p1 == pytest.approx(0.62)
    assert state.indexes.source == "sensor"
    assert state.indexes.confidence == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"p0": 0.3, "p1": 0.8, "d": 0.1}, InvalidProbabilityError),
        ({"p1": 1.1, "d": 0.1}, InvalidProbabilityError),
        ({"p1": 0.5, "d": 1.0}, InvalidDependencyError),
        ({"p1": 0.5, "d": -1.0}, InvalidDependencyError),
    ],
)
def test_state_rejects_invalid_core_values(kwargs: dict[str, float], error: type[Exception]) -> None:
    with pytest.raises(error):
        State(**kwargs)


def test_indexes_validate_their_declared_domains() -> None:
    with pytest.raises(InvalidStateError, match="non-negative"):
        Indexes(weight=-0.1)
    with pytest.raises(InvalidStateError, match="between 0 and 1"):
        Indexes(confidence=1.1)
    with pytest.raises(InvalidStateError, match="unknown state index"):
        Indexes().get("region")
