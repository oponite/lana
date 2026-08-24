from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from lana_integrations.native import NativeBridge


def test_ctypes_native_round_trip(tmp_path: Path) -> None:
    library = os.environ.get("LANA_BRIDGE_TEST_LIBRARY")
    bytecode = os.environ.get("LANA_BRIDGE_TEST_BYTECODE")
    if not library or not bytecode:
        pytest.skip("native bridge test artifacts were not provided")
    request = tmp_path / "request.json"
    response = tmp_path / "response.json"
    request.write_text(json.dumps({"ctypes": True}), encoding="utf-8")
    envelope = NativeBridge(library).run_labc(bytecode, request, response)
    assert envelope["ok"] is True
    assert envelope["result"] == {"ctypes": True}
    assert envelope["native_status"] == 0
