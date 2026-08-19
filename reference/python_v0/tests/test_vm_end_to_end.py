import pytest

from structured_state.compiler.compiler import compile_source
from structured_state.vm.machine import VM
from structured_state.vm.values import Distribution


def run(source: str, *, seed: int | None = None):
    return VM(compile_source(source), seed=seed).run()


def test_compiled_apply_updates_only_target_observable_probability() -> None:
    result = run(
        """
        state source = state(p1: 0.8, d: 0.6)
        state target = state(p1: 0.4, d: 0.1)
        apply source -> target
        return measure target using distribution
        """
    )

    assert result == Distribution(p0=pytest.approx(0.36), p1=pytest.approx(0.64))


def test_distribution_measurement_preserves_state() -> None:
    result = run(
        """
        state belief = state(p1: 0.7, d: 0.2)
        let observed = measure belief using distribution
        return measure belief using distribution
        """
    )

    assert result == Distribution(p0=pytest.approx(0.3), p1=pytest.approx(0.7))


def test_collapse_measurement_replaces_state_with_observed_outcome() -> None:
    result = run(
        """
        state belief = state(p1: 0.5, d: 0.2)
        let observed = measure belief using collapse
        return measure belief using distribution
        """,
        seed=0,
    )

    assert result == Distribution(p0=1.0, p1=0.0)


def test_conditional_apply_compiles_and_skips_false_condition() -> None:
    result = run(
        """
        state source = state(p1: 0.9, d: 0.5)
        state target = state(p1: 0.2, d: 0.1)
        apply source -> target when false
        return measure target using expectation
        """
    )

    assert result == pytest.approx(0.2)


def test_program_compiles_to_native_state_instructions_and_runs() -> None:
    program = compile_source(
        """
        state belief = state(p1: 0.5, d: 0.2)
        transform belief with decay(0.1)
        return measure belief using expectation
        """
    )

    assert [instruction.op.value for instruction in program.instructions] == [
        "LOAD_CONST", "LOAD_CONST", "STATE_NEW", "LOAD_CONST", "TRANSFORM", "MEASURE", "RETURN", "HALT"
    ]
    assert VM(program).run() == pytest.approx(0.5)
