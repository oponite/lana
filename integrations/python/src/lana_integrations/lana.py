"""Ergonomic Lana API with structured results and a persistent Rust worker."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from .bridge import BridgeRunner, LanaCompatibilityError
from .worker import WorkerRunner

# A program signals an unresolved result (an Information or Sample value that
# was not reduced to an ordinary value) by writing a response object carrying
# this reserved key. The ergonomic API preserves the raw value and reports
# status "unresolved" rather than coercing it.
UNRESOLVED_MARKER = "__lana_unresolved__"


@dataclass
class LanaResult:
    """Structured outcome of a Lana run.

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
    """Check and run Lana programs through the Rust executable."""

    def __init__(
        self,
        executable: str | os.PathLike[str] | None = None,
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._bridge: BridgeRunner | None = None
        self._worker: WorkerRunner | None = None
        self._timeout_seconds = timeout_seconds
        self._unavailable_reason: str | None = None
        try:
            self._bridge = BridgeRunner(executable, timeout_seconds=timeout_seconds)
        except (FileNotFoundError, LanaCompatibilityError, ValueError) as error:
            self._unavailable_reason = str(error)

    @property
    def available(self) -> bool:
        return self._bridge is not None

    @property
    def backend(self) -> str:
        if self._bridge is not None:
            return "worker" if self._bridge.version.startswith("4.") else "subprocess"
        return "unavailable"

    def run(self, program: str | os.PathLike[str], input_value: Any) -> LanaResult:
        """Run a source program with structured input."""
        if self._bridge is None:
            return _unavailable_result(self._unavailable_reason or "no Lana runtime")
        if self._bridge.version.startswith("4."):
            return self._run_worker("run", program, input_value)
        envelope = self._bridge.run(program, input_value)
        return self._from_envelope(envelope, "subprocess")

    def run_labc(
        self, labc_path: str | os.PathLike[str], input_value: Any
    ) -> LanaResult:
        """Run precompiled bytecode through the Rust worker."""
        if self._bridge is None or not self._bridge.version.startswith("4."):
            return _unavailable_result("Lana 4.0 worker is not available")
        return self._run_worker("run_labc", labc_path, input_value)

    def _run_worker(self, operation: str, path: str | os.PathLike[str], input_value: Any) -> LanaResult:
        assert self._bridge is not None
        try:
            if self._worker is None:
                self._worker = WorkerRunner(self._bridge.executable, self._timeout_seconds)
            envelope = self._worker.run(operation, Path(path), input_value)
        except (RuntimeError, OSError, TypeError, ValueError) as error:
            return LanaResult("failed", error={"code": "LANA_WORKER_FAILED", "message": str(error)}, backend="worker")
        return self._from_envelope(envelope, "worker")

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()
            self._worker = None

    def __enter__(self) -> "Lana":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

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
