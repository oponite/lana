from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from lana_integrations.http_client import LanaHttpError, _raise_for_payload, fetch_evidence


def test_error_payload_raises_typed_error() -> None:
    with pytest.raises(LanaHttpError) as excinfo:
        _raise_for_payload(
            {"schema": 1, "ok": False, "error": {"code": "LANA_ERR_NOT_FOUND", "message": "nope"}},
            404,
        )
    assert excinfo.value.code == "LANA_ERR_NOT_FOUND"
    assert excinfo.value.status == 404


def test_ok_payload_returns() -> None:
    _raise_for_payload({"schema": 1, "ok": True, "evidence": {"p": 0.9}}, 200)


def _spawn_service(data_dir: Path) -> tuple[subprocess.Popen, int]:
    service = os.environ.get("LANA_HTTP_SERVICE")
    if not service:
        pytest.skip("LANA_HTTP_SERVICE not set")
    proc = subprocess.Popen(
        [service, "--port", "0", "--data", str(data_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert proc.stdout is not None
    line = proc.stdout.readline().strip()
    port = int(line)
    return proc, port


def test_fetch_evidence_live() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        (data_dir / "sensor-1.json").write_text('{"p":0.9,"source":"sensor-1"}', encoding="utf-8")
        proc, port = _spawn_service(data_dir)
        try:
            base = f"http://127.0.0.1:{port}"
            evidence = fetch_evidence(base, "sensor-1")
            assert evidence["p"] == 0.9
            with pytest.raises(LanaHttpError) as excinfo:
                fetch_evidence(base, "missing")
            assert excinfo.value.code == "LANA_ERR_NOT_FOUND"
        finally:
            proc.terminate()
            proc.wait(timeout=5)
