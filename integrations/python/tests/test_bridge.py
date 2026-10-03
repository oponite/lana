from __future__ import annotations

from pathlib import Path

import pytest

from lana_integrations.bridge import BridgeRunner, LanaCompatibilityError
from lana_integrations.evidence import EvidenceValidationError


def test_round_trip_and_separate_logs(fake_lana: Path, program: Path) -> None:
    envelope = BridgeRunner(fake_lana).run(program, {"message": "hello", "value": 3})
    assert envelope["ok"] is True
    assert envelope["result"] == {"message": "hello", "value": 3}
    assert envelope["stdout"] == "program log\n"
    assert envelope["execution"]["lana_version"] == "3.0.0"


@pytest.mark.parametrize(
    ("name", "code"),
    [("missing.lana", "LANA_RESPONSE_MISSING"), ("invalid.lana", "LANA_RESPONSE_INVALID")],
)
def test_protocol_failures(fake_lana: Path, tmp_path: Path, name: str, code: str) -> None:
    program = tmp_path / name
    program.write_text("", encoding="utf-8")
    envelope = BridgeRunner(fake_lana).run(program, None)
    assert envelope["ok"] is False
    assert envelope["phase"] == "protocol"
    assert envelope["error"]["code"] == code


def test_runtime_failure(fake_lana: Path, tmp_path: Path) -> None:
    fail = tmp_path / "fail.lana"
    fail.write_text("", encoding="utf-8")
    assert BridgeRunner(fake_lana).run(fail, {})["error"]["code"] == "LANA_RUN_FAILED"


def test_check_entry_points_are_removed() -> None:
    from lana_integrations.cli import main
    from lana_integrations.lana import Lana

    assert not hasattr(BridgeRunner, "check")
    assert not hasattr(Lana, "check")
    with pytest.raises(SystemExit) as error:
        main(["check", "program.lana"])
    assert error.value.code == 2


def test_timeout(fake_lana: Path, tmp_path: Path) -> None:
    program = tmp_path / "sleep.lana"
    program.write_text("", encoding="utf-8")
    envelope = BridgeRunner(fake_lana, timeout_seconds=0.05).run(program, {})
    assert envelope["phase"] == "timeout"
    assert envelope["error"]["code"] == "LANA_BRIDGE_TIMEOUT"


def test_plain_execution(fake_lana: Path, program: Path) -> None:
    envelope = BridgeRunner(fake_lana).run_plain(program)
    assert envelope["ok"] is True
    assert envelope["stdout"] == "plain output\n"


def test_evidence_input_is_validated_before_execution(
    fake_lana: Path, program: Path
) -> None:
    with pytest.raises(EvidenceValidationError, match="status"):
        BridgeRunner(fake_lana).run_evidence(program, {"schema": 1, "status": "maybe"})


def test_rejects_incompatible_version(tmp_path: Path) -> None:
    executable = tmp_path / "lana"
    executable.write_text("#!/bin/sh\necho 'Lana 3.0.0 (LABC v1)'\n", encoding="utf-8")
    executable.chmod(0o755)
    with pytest.raises(LanaCompatibilityError):
        BridgeRunner(executable)


@pytest.mark.parametrize("version", ["3.0.0", "3.1.0", "3.2.0", "4.0.0", "4.1.0"])
@pytest.mark.parametrize("labc", [2, 3, 4, 5, 6])
def test_accepts_compatible_labc_versions(tmp_path: Path, version: str, labc: int) -> None:
    executable = tmp_path / "lana"
    executable.write_text(
        f"#!/bin/sh\necho 'Lana {version} (LABC v{labc}, fake)'\n", encoding="utf-8"
    )
    executable.chmod(0o755)
    runner = BridgeRunner(executable)
    assert runner.version == version
    assert runner.labc_version == labc


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_rejects_nonfinite_json(fake_lana: Path, program: Path, value: float) -> None:
    with pytest.raises(ValueError):
        BridgeRunner(fake_lana).run(program, {"value": value})
