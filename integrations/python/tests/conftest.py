from __future__ import annotations

from pathlib import Path
import os

import pytest

from lana_integrations.bridge import BridgeRunner


@pytest.fixture(scope="session")
def built_lana() -> Path:
    root = Path(__file__).resolve().parents[3]
    explicit = os.environ.get("LANA_EXECUTABLE")
    if explicit is not None:
        return Path(BridgeRunner(explicit).executable)
    for path in (root / "target/lana/bin/lana", root / "target/release/lana"):
        if path.is_file():
            return Path(BridgeRunner(path).executable)
    pytest.skip("requires a built Lana executable or LANA_EXECUTABLE")


FAKE_LANA = r'''#!/usr/bin/env python3
import json
from pathlib import Path
import sys
import time

command = sys.argv[1]
if command == "version":
    print("Lana 3.0.0 (LABC v5, fake)")
    raise SystemExit(0)
program = Path(sys.argv[2])
if command != "run":
    raise SystemExit(2)
if program.name.startswith("sleep"):
    time.sleep(2)
if program.name.startswith("fail"):
    print("fake runtime failure", file=sys.stderr)
    raise SystemExit(1)
if "--" not in sys.argv:
    print("plain output")
    raise SystemExit(0)
separator = sys.argv.index("--")
request = Path(sys.argv[separator + 1])
response = Path(sys.argv[separator + 2])
print("program log")
if program.name.startswith("missing"):
    raise SystemExit(0)
if program.name.startswith("invalid"):
    response.write_text("not json")
else:
    response.write_text(json.dumps(json.loads(request.read_text())))
'''


@pytest.fixture
def fake_lana(tmp_path: Path) -> Path:
    executable = tmp_path / "lana"
    executable.write_text(FAKE_LANA, encoding="utf-8")
    executable.chmod(0o755)
    return executable


@pytest.fixture
def program(tmp_path: Path) -> Path:
    source = tmp_path / "program.lana"
    source.write_text("print(1);\n", encoding="utf-8")
    return source
