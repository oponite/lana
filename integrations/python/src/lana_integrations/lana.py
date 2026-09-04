"""Ergonomic Lana API: check and run programs with structured results.

The :class:`Lana` class prefers the native ctypes bridge when a compatible
``liblana_bridge`` is available and falls back to the subprocess
:class:`~lana_integrations.bridge.BridgeRunner` otherwise. Results are always
returned as :class:`LanaResult` so that unavailable, unresolved, and failed
outcomes stay distinct and are never coerced into ordinary values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

from .bridge import BridgeRunner, LanaCompatibilityError
from .native import NativeBridge

# A program signals an unresolved result (an Information or Sample value that
# was not reduced to an ordinary value) by writing a response object carrying
# this reserved key. The ergonomic API preserves the raw value and reports
# status "unresolved" rather than coercing it.
UNRESOLVED_MARKER = "__lana_unresolved__"


@dataclass
class LanaResult:
    """Structured outcome of a Lana check or run.

    ``status`` is one of ``"ok"``, ``"unavailable"``, ``"unresolved"``, or
    ``"failed"``. ``value`` is populated only for ``"ok"`` and ``"unresolved"``
    (for the latter it holds the raw, uncoerced value). ``error`` is populated
    only for ``"failed"`` and ``"unavailable"``.
    """

    status: str
    value: Any = None
    error: dict[str, Any] | None = None
    backend: str = "subprocess"
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _unavailable_result(reason: str) -> LanaResult:
    return LanaResult(
        "unavailable",
        error={"code": "LANA_UNAVAILABLE", "message": reason},
    )


class Lana:
    """Check and run Lana programs, preferring the native bridge when present."""

    def __init__(
        self,
        executable: str | os.PathLike[str] | None = None,
        library: str | os.PathLike[str] | None = None,
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._bridge: BridgeRunner | None = None
        self._native: NativeBridge | None = None
        self._unavailable_reason: str | None = None
        try:
            self._bridge = BridgeRunner(executable, timeout_seconds=timeout_seconds)
        except (FileNotFoundError, LanaCompatibilityError, ValueError) as error:
            self._unavailable_reason = str(error)
        try:
            self._native = NativeBridge(library)
        except (FileNotFoundError, RuntimeError, OSError):
            self._native = None

    @property
    def available(self) -> bool:
        return self._bridge is not None or self._native is not None

    @property
    def backend(self) -> str:
        if self._native is not None:
            return "native"
        if self._bridge is not None:
            return "subprocess"
        return "unavailable"

    def check(self, program: str | os.PathLike[str]) -> LanaResult:
        """Compile-check a source program. Always uses the subprocess compiler."""
        if self._bridge is None:
            return _unavailable_result(self._unavailable_reason or "no Lana runtime")
        envelope = self._bridge.check(program)
        return self._from_envelope(envelope, "subprocess")

    def run(self, program: str | os.PathLike[str], input_value: Any) -> LanaResult:
        """Run a source program with structured input.

        The native bridge runs precompiled bytecode, so a source program is
        compiled first (subprocess) and then executed in-process when the native
        bridge is available; otherwise the whole run is subprocess.
        """
        if self._bridge is None and self._native is None:
            return _unavailable_result(self._unavailable_reason or "no Lana runtime")
        if self._native is not None and self._bridge is not None:
            native = self._run_native(program, input_value)
            if native is not None:
                return native
        if self._bridge is None:
            return _unavailable_result(self._unavailable_reason or "no Lana runtime")
        envelope = self._bridge.run(program, input_value)
        return self._from_envelope(envelope, "subprocess")

    def run_labc(
        self, labc_path: str | os.PathLike[str], input_value: Any
    ) -> LanaResult:
        """Run a precompiled LABC program, preferring the native bridge."""
        if self._native is None:
            return _unavailable_result("native bridge is not available")
        return self._run_native_labc(labc_path, input_value)

    def _run_native(
        self, program: str | os.PathLike[str], input_value: Any
    ) -> LanaResult | None:
        """Compile source to LABC and run it in-process. Returns None on failure."""
        assert self._bridge is not None and self._native is not None
        import tempfile

        with tempfile.TemporaryDirectory(prefix="lana-native-") as temporary:
            labc_path = Path(temporary) / "program.labc"
            compiled = self._bridge._execute(
                [self._bridge.executable, "compile", str(Path(program).resolve()), "-o", str(labc_path)],
                cwd=Path(program).resolve().parent,
                timeout_seconds=None,
            )
            completed, _, timeout_error = compiled
            if timeout_error is not None or completed is None or completed.returncode != 0:
                return None
            return self._run_native_labc(labc_path, input_value)

    def _run_native_labc(
        self, labc_path: str | os.PathLike[str], input_value: Any
    ) -> LanaResult:
        assert self._native is not None
        import json
        import tempfile

        with tempfile.TemporaryDirectory(prefix="lana-native-") as temporary:
            temporary_path = Path(temporary)
            request_path = temporary_path / "request.json"
            response_path = temporary_path / "response.json"
            request_path.write_text(
                json.dumps(input_value, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            try:
                envelope = self._native.run_labc(labc_path, request_path, response_path)
            except (RuntimeError, OSError) as error:
                return LanaResult(
                    "failed",
                    error={"code": "LANA_NATIVE_FAILED", "message": str(error)},
                    backend="native",
                )
            return self._from_envelope(envelope, "native")

    @staticmethod
    def _from_envelope(envelope: dict[str, Any], backend: str) -> LanaResult:
        if envelope.get("ok"):
            result = envelope.get("result")
            if isinstance(result, dict) and result.get(UNRESOLVED_MARKER):
                return LanaResult(
                    "unresolved",
                    value=result,
                    backend=backend,
                    stdout=envelope.get("stdout", ""),
                    stderr=envelope.get("stderr", ""),
                )
            return LanaResult(
                "ok",
                value=result,
                backend=backend,
                stdout=envelope.get("stdout", ""),
                stderr=envelope.get("stderr", ""),
            )
        return LanaResult(
            "failed",
            error=envelope.get("error"),
            backend=backend,
            stdout=envelope.get("stdout", ""),
            stderr=envelope.get("stderr", ""),
        )
