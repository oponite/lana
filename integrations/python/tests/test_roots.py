from __future__ import annotations

from pathlib import Path

import pytest

from lana_integrations.roots import RootPolicy


def test_resolves_relative_program_inside_root(tmp_path: Path) -> None:
    program = tmp_path / "program.lana"
    program.write_text("", encoding="utf-8")
    assert RootPolicy([tmp_path]).resolve_program("program.lana") == program.resolve()


def test_rejects_parent_and_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.lana"
    outside.write_text("", encoding="utf-8")
    (root / "escape.lana").symlink_to(outside)
    policy = RootPolicy([root])
    with pytest.raises(ValueError):
        policy.resolve_program("../outside.lana")
    with pytest.raises(ValueError):
        policy.resolve_program("escape.lana")
