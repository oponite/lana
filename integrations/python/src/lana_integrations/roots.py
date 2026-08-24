"""Canonical root policy used by execution-capable integrations."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


class RootPolicy:
    """Resolve existing program files without permitting root or symlink escape."""

    def __init__(self, roots: Iterable[str | Path]) -> None:
        canonical = tuple(Path(root).expanduser().resolve(strict=True) for root in roots)
        if not canonical:
            raise ValueError("at least one root is required")
        for root in canonical:
            if not root.is_dir():
                raise ValueError(f"root is not a directory: {root}")
        self.roots = canonical

    def resolve_program(self, program: str | Path) -> Path:
        requested = Path(program).expanduser()
        candidates = (
            (requested,)
            if requested.is_absolute()
            else tuple(root / requested for root in self.roots)
        )
        for candidate in candidates:
            try:
                resolved = candidate.resolve(strict=True)
            except FileNotFoundError:
                continue
            if not resolved.is_file():
                continue
            if any(resolved.is_relative_to(root) for root in self.roots):
                return resolved
        raise ValueError(f"program is outside configured roots or missing: {program}")
