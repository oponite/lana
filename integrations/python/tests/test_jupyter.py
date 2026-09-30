from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from lana_integrations.jupyter import load_ipython_extension, unload_ipython_extension


IPython = pytest.importorskip("IPython")


def test_cell_magic_executes_with_selected_lana(
    fake_lana: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from IPython.terminal.interactiveshell import TerminalInteractiveShell

    monkeypatch.setenv("LANA_EXECUTABLE", str(fake_lana))
    shell = TerminalInteractiveShell.instance()
    unload_ipython_extension(shell)
    load_ipython_extension(shell)

    shell.run_cell_magic("lana", "", "print(1);\n")

    assert shell.user_ns["_lana_result"]["ok"] is True
    assert shell.user_ns["_lana_result"]["stdout"] == "plain output\n"
    unload_ipython_extension(shell)


def test_real_notebook_plain_and_json(built_lana: Path, tmp_path: Path, monkeypatch) -> None:
    from IPython.terminal.interactiveshell import TerminalInteractiveShell
    root = Path(__file__).resolve().parents[3]
    monkeypatch.setenv("LANA_EXECUTABLE", str(built_lana))
    monkeypatch.chdir(tmp_path)
    shell = TerminalInteractiveShell.instance()
    shell.run_line_magic("reload_ext", "lana_integrations.jupyter")
    try:
        shell.run_cell_magic("lana", "--seed 17", 'print("hello notebook");\n')
        assert shell.user_ns["_lana_result"]["ok"]
        assert shell.user_ns["_lana_result"]["stdout"] == "hello notebook\n"
        request = tmp_path / "input.json"
        request.write_text(json.dumps({"message": "hello", "value": False}))
        bridge = root / "integrations/lana/bridge.lana"
        (tmp_path / "bridge.lana").write_text(bridge.read_text())
        shell.run_cell_magic("lana", "--input input.json --instruction-limit 50000000",
                             'import "./bridge.lana" as bridge;\nbridge.write_response(bridge.read_request());\n')
        assert shell.user_ns["_lana_result"]["result"] == {"message": "hello", "value": False}
        assert not list(tmp_path.glob(".lana-cell-*"))
    finally:
        shell.run_line_magic("unload_ext", "lana_integrations.jupyter")


def test_removed_check_flag_does_not_execute_cell(
    fake_lana: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from IPython.terminal.interactiveshell import TerminalInteractiveShell
    from lana_integrations.bridge import BridgeRunner

    monkeypatch.setenv("LANA_EXECUTABLE", str(fake_lana))
    monkeypatch.chdir(tmp_path)
    shell = TerminalInteractiveShell.instance()
    unload_ipython_extension(shell)
    load_ipython_extension(shell)

    def unexpected_execution(*args, **kwargs):
        pytest.fail("removed flag executed a cell")

    monkeypatch.setattr(BridgeRunner, "run_plain", unexpected_execution)
    monkeypatch.setattr(BridgeRunner, "run", unexpected_execution)
    with pytest.raises((SystemExit, argparse.ArgumentError)) as error:
        shell.run_cell_magic("lana", "--check", "print(1);\n")
    if isinstance(error.value, SystemExit):
        assert error.value.code == 2
    else:
        assert "unrecognized arguments: --check" in str(error.value)
    assert "_lana_result" not in shell.user_ns
    assert not list(tmp_path.glob(".lana-cell-*"))
    unload_ipython_extension(shell)
