from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from lana_integrations.lana import Lana, LanaResult, UNRESOLVED_MARKER


def test_run_round_trip_via_subprocess(fake_lana: Path, program: Path) -> None:
    result = Lana(executable=fake_lana).run(program, {"message": "hello"})
    assert isinstance(result, LanaResult)
    assert result.status == "ok"
    assert result.value == {"message": "hello"}
    assert result.backend == "subprocess"


def test_check_via_subprocess(fake_lana: Path, program: Path) -> None:
    result = Lana(executable=fake_lana).check(program)
    assert result.status == "ok"
    assert result.backend == "subprocess"


def test_check_failure_is_distinct(fake_lana: Path, tmp_path: Path) -> None:
    bad = tmp_path / "bad.lana"
    bad.write_text("", encoding="utf-8")
    result = Lana(executable=fake_lana).check(bad)
    assert result.status == "failed"
    assert result.error is not None
    assert result.value is None


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


def test_native_round_trip_when_library_present(tmp_path: Path) -> None:
    library = os.environ.get("LANA_BRIDGE_TEST_LIBRARY")
    bytecode = os.environ.get("LANA_BRIDGE_TEST_BYTECODE")
    if not library or not bytecode:
        pytest.skip("native bridge test artifacts were not provided")
    result = Lana(library=library).run_labc(bytecode, {"ctypes": True})
    assert result.status == "ok"
    assert result.value == {"ctypes": True}
    assert result.backend == "native"
