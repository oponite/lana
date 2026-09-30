from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from lana_integrations import BridgeRunner, Lana
from lana_integrations.bridge import _USIZE_MAX


ROOT = Path(__file__).resolve().parents[3]
BRIDGE = "./bridge.lana"


@pytest.fixture
def bridge_module(tmp_path: Path) -> None:
    (tmp_path / "bridge.lana").write_text((ROOT / "integrations/lana/bridge.lana").read_text())


@pytest.fixture
def random_program(tmp_path: Path, built_lana: Path, bridge_module) -> tuple[Path, Path]:
    source = tmp_path / "random.lana"
    source.write_text(f'import "{BRIDGE}" as bridge;\nbridge.write_response(sample_value(random()));\n')
    bytecode = tmp_path / "random.labc"
    subprocess.run([built_lana, "compile", source, "-o", bytecode], check=True)
    return source, bytecode


def test_controls_and_per_call_isolation(built_lana: Path, random_program) -> None:
    with Lana(built_lana) as lana:
        for operation, path in zip((lana.run, lana.run_labc), random_program):
            baseline = operation(path, {}).value
            seeded = operation(path, {}, seed=2**64 - 1, workers=2, max_tasks=3,
                               memory_limit_mib=256, instruction_limit=50_000_000)
            assert seeded.ok, seeded.error
            assert operation(path, {}, seed=2**64 - 1).value == seeded.value
            assert BridgeRunner(built_lana).run(random_program[0], {}, seed=2**64 - 1)["result"] == seeded.value
            assert operation(path, {}, seed=17).value != seeded.value
            assert operation(path, {}).value == baseline
            limited = operation(path, {}, instruction_limit=1)
            assert limited.status == "failed"
            assert limited.error["code"] == "LANA_ERR_LIMIT"
            assert operation(path, {}).ok


def test_memory_exhaustion(built_lana: Path, tmp_path: Path, bridge_module) -> None:
    source = tmp_path / "memory.lana"
    source.write_text(f'import "{BRIDGE}" as bridge;\nlet values = [];\n'
                      'let i = 0;\nwhile (i < 100000) { array_push(values, i); i = i + 1; }\n'
                      'bridge.write_response(array_length(values));\n')
    bytecode = tmp_path / "memory.labc"
    subprocess.run([built_lana, "compile", source, "-o", bytecode], check=True)
    with Lana(built_lana) as lana:
        for operation, path in ((lana.run, source), (lana.run_labc, bytecode)):
            result = operation(path, {}, memory_limit_mib=1)
            assert result.status == "failed"
            assert result.error["code"] == "LANA_ERR_OOM"
            assert operation(path, {}, memory_limit_mib=64).value == 100000


def test_scheduler_limit(built_lana: Path, tmp_path: Path, bridge_module) -> None:
    source = tmp_path / "tasks.lana"
    source.write_text(f'import "{BRIDGE}" as bridge;\nfn worker() {{ return 7; }}\n'
                      'let a = fork worker();\nlet b = fork worker();\n'
                      'bridge.write_response(join(a) + join(b));\n')
    bytecode = tmp_path / "tasks.labc"
    subprocess.run([built_lana, "compile", source, "-o", bytecode], check=True)
    with Lana(built_lana) as lana:
        for operation, path in ((lana.run, source), (lana.run_labc, bytecode)):
            failed = operation(path, {}, workers=1, max_tasks=1)
            assert failed.status == "failed", failed
            assert operation(path, {}, workers=2, max_tasks=8).value == 14


@pytest.mark.parametrize("name,maximum", [
    ("seed", 2**64 - 1), ("instruction_limit", 2**64 - 1),
    ("memory_limit_mib", _USIZE_MAX // (1024 * 1024)),
    ("workers", _USIZE_MAX), ("max_tasks", _USIZE_MAX),
])
def test_invalid_controls_before_execution(fake_lana: Path, program: Path, name: str, maximum: int) -> None:
    with Lana(fake_lana) as lana:
        for value in (True, False, 0, -1, 1.5, "1", maximum + 1):
            for operation in (lana.run, lana.run_labc):
                with pytest.raises(ValueError):
                    operation(program, {}, **{name: value})
            with pytest.raises(ValueError):
                BridgeRunner(fake_lana).run(program, {}, **{name: value})


@pytest.mark.parametrize("timeout", [True, 0, -1, float("nan"), float("inf"), -float("inf"), 10**1000])
def test_invalid_timeouts(fake_lana: Path, program: Path, timeout) -> None:
    with pytest.raises(ValueError):
        Lana(fake_lana, timeout_seconds=timeout)
    with pytest.raises(ValueError):
        BridgeRunner(fake_lana, timeout_seconds=timeout)
    with Lana(fake_lana) as lana:
        for operation in (lana.run, lana.run_labc):
            with pytest.raises(ValueError):
                operation(program, {}, timeout_seconds=timeout)
    with pytest.raises(ValueError):
        BridgeRunner(fake_lana).run(program, {}, timeout_seconds=timeout)


def test_legacy_source_options_forwarded(fake_lana: Path, program: Path, monkeypatch) -> None:
    seen = {}
    def run(self, path, value, **options):
        seen.update(options)
        return {"ok": True, "result": value}
    monkeypatch.setattr(BridgeRunner, "run", run)
    with Lana(fake_lana) as lana:
        result = lana.run(program, 7, seed=9, workers=2, max_tasks=4,
                          memory_limit_mib=8, instruction_limit=100, timeout_seconds=3)
        assert result.value == 7
        assert seen == dict(seed=9, workers=2, max_tasks=4, memory_limit_mib=8,
                            instruction_limit=100, timeout_seconds=3)
        assert lana.run_labc(program, {}).status == "unavailable"


def test_worker_rejects_nonfinite_input(built_lana: Path, random_program) -> None:
    with Lana(built_lana) as lana:
        for value in (float("nan"), float("inf"), -float("inf")):
            for operation, path in zip((lana.run, lana.run_labc), random_program):
                assert operation(path, {"invalid": value}).status == "failed"
        assert lana.run(random_program[0], {}).ok


def test_timeout_stops_worker_without_replaying_effect(built_lana: Path, tmp_path: Path, bridge_module) -> None:
    source = tmp_path / "effect.lana"
    marker = tmp_path / "effects.txt"
    source.write_text(f'import "{BRIDGE}" as bridge;\nlet request = bridge.read_request();\n'
                      'let path = request["marker"];\n'
                      'write_text(path, string_concat(read_text(path), "x"));\n'
                      'while (true) {}\n')
    bytecode = tmp_path / "effect.labc"
    subprocess.run([built_lana, "compile", source, "-o", bytecode], check=True)
    with Lana(built_lana) as lana:
        for operation, path in ((lana.run, source), (lana.run_labc, bytecode)):
            marker.write_text("")
            result = operation(path, {"marker": str(marker)}, instruction_limit=2**64 - 1,
                               timeout_seconds=1)
            assert result.status == "failed"
            assert "timed out" in result.error["message"]
            assert marker.read_text() == "x"
            assert lana._worker._process.poll() is not None
            # The failed request is never retried, even on the next call.
            assert lana.run(ROOT / "integrations/lana/echo_bridge.lana", {}).status == "failed"
            assert marker.read_text() == "x"
            lana.close()


def test_malformed_worker_controls(built_lana: Path) -> None:
    request = {"schema": 1, "op": "run", "path": str(ROOT / "integrations/lana/echo_bridge.lana"), "input": {}}
    bad = [{**request, name: value} for name in ("seed", "memory_limit_mib", "instruction_limit", "workers", "max_tasks")
           for value in (None, True, 0, -1, 1.5, "1", 2**64)]
    bad.extend([{**request, "memory_limit_mib": _USIZE_MAX // (1024 * 1024) + 1},
                {**request, "schema": 1.5}])
    completed = subprocess.run([built_lana, "bridge-worker"],
                               input="".join(json.dumps(value) + "\n" for value in [*bad, request]),
                               capture_output=True, text=True, timeout=10, check=True)
    replies = [json.loads(line) for line in completed.stdout.splitlines()]
    assert len(replies) == len(bad) + 1
    assert all(not reply["ok"] and reply["phase"] == "protocol" for reply in replies[:-1])
    assert replies[-1]["ok"]
