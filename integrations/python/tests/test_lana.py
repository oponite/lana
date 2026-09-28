from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from lana_integrations.lana import Lana, LanaResult, UNRESOLVED_MARKER


def test_run_round_trip_via_subprocess(fake_lana: Path, program: Path) -> None:
    result = Lana(executable=fake_lana).run(program, {"message": "hello"})
    assert isinstance(result, LanaResult)
    assert result.status == "ok"
    assert result.value == {"message": "hello"}
    assert result.backend == "subprocess"


def test_run_failure_is_distinct(fake_lana: Path, tmp_path: Path) -> None:
    fail = tmp_path / "fail.lana"
    fail.write_text("", encoding="utf-8")
    result = Lana(executable=fake_lana).run(fail, {})
    assert result.status == "failed"
    assert result.error is not None


def test_unresolved_result_is_not_coerced(fake_lana: Path, program: Path) -> None:
    raw = {UNRESOLVED_MARKER: True, "kind": "Information"}
    result = Lana(executable=fake_lana).run(program, raw)
    assert result.status == "unresolved"
    assert result.value == raw


def test_unavailable_when_no_runtime(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    result = Lana(executable=missing).run(tmp_path / "p.lana", {})
    assert result.status == "unavailable"
    assert result.error is not None
    assert result.error["code"] == "LANA_UNAVAILABLE"


def test_rust_worker_repeated_calls(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    executable = next((path for path in (root / "target/lana/bin/lana", root / "target/release/lana")
                       if path.is_file() and "Lana 4.0" in subprocess.check_output([path, "version"], text=True)), None)
    if executable is None:
        pytest.skip("requires the Rust-only build")
    program = root / "integrations" / "lana" / "echo_bridge.lana"
    bytecode = tmp_path / "echo.labc"
    subprocess.run([executable, "compile", program, "-o", bytecode], check=True)
    record = {"record_schema": 1, "id": "relationship_claim/9007199254740993/1",
              "kind": "relationship_claim", "payload": {"claim_id": "9007199254740993"},
              "payload_digest": "ab" * 32, "signature": "cd" * 64}
    with Lana(executable=executable) as lana:
        assert lana.backend == "worker"
        for value in ({"first": 1}, {"second": False}, record):
            for operation, path in ((lana.run, program), (lana.run_labc, bytecode)):
                result = operation(path, value)
                assert result.status == "ok", result.error
                assert result.value == value
                assert result.backend == "worker"
        bad = tmp_path / "bad.labc"
        bad.write_bytes(b"bad bytecode")
        failed = lana.run_labc(bad, {})
        assert failed.status == "failed"
        assert failed.error is not None
        assert failed.error["code"].startswith("LANA_ERR_")
