from pathlib import Path
import importlib.util
import sys


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_python_baselines_have_identical_dynamic_behavior() -> None:
    runner = load("benchmark_runner_test", ROOT / "benchmark" / "run_benchmark.py")
    steps = runner.load_steps("B", 24)

    plain = runner.PLAIN.predict(steps, "B", "dynamic")
    state_class = runner.STATE_CLASS.predict(steps, "B", "dynamic")

    assert max(abs(left - right) for left, right in zip(plain, state_class)) < 1e-12


def test_lana_benchmark_generation_uses_native_operations() -> None:
    runner = load("benchmark_runner_generation_test", ROOT / "benchmark" / "run_benchmark.py")
    assembly = runner.LANA_GENERATOR.generate(runner.load_steps("C", 2), "C")

    assert "APPLY_MANY" in assembly
    assert "TRANSFORM" in assembly
    assert "HISTORY" in assembly
    assert "MEASURE" in assembly
    assert "JUMP_IF_FALSE" in assembly
