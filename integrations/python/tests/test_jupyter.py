from __future__ import annotations

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
