"""Stable JSON-file subprocess boundary for Lana 1.0 programs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence

from .evidence import validate_evidence


SCHEMA_VERSION = 1
DEFAULT_TIMEOUT_SECONDS = 30.0
_VERSION_PATTERN = re.compile(r"^Lana (1\.0\.\d+) \(LABC v1,")


class LanaCompatibilityError(RuntimeError):
    """Raised when the selected executable is not Lana 1.0 with LABC v1."""


def _error_envelope(
    phase: str,
    message: str,
    *,
    code: str,
    exit_code: int | None = None,
    stdout: str = "",
    stderr: str = "",
    execution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    envelope: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "ok": False,
        "phase": phase,
        "error": {"code": code, "message": message},
        "stdout": stdout,
        "stderr": stderr,
        "execution": dict(execution or {}),
    }
    if exit_code is not None:
        envelope["exit_code"] = exit_code
    return envelope


class BridgeRunner:
    """Run Lana through a deterministic, machine-readable subprocess boundary."""

    def __init__(
        self,
        lana_executable: str | os.PathLike[str] | None = None,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        requested = (
            os.fspath(lana_executable)
            if lana_executable is not None
            else os.environ.get("LANA_EXECUTABLE", "lana")
        )
        resolved = shutil.which(requested)
        if resolved is None:
            candidate = Path(requested).expanduser()
            if candidate.is_file() and os.access(candidate, os.X_OK):
                resolved = str(candidate.resolve())
        if resolved is None:
            raise FileNotFoundError(f"Lana executable not found: {requested}")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.executable = str(Path(resolved).resolve())
        self.timeout_seconds = float(timeout_seconds)
        self.version = self._check_compatibility()

    def _check_compatibility(self) -> str:
        try:
            completed = subprocess.run(
                [self.executable, "version"],
                text=True,
                capture_output=True,
                timeout=5.0,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise LanaCompatibilityError("lana version timed out") from error
        output = completed.stdout.strip()
        match = _VERSION_PATTERN.match(output)
        if completed.returncode != 0 or match is None:
            detail = output or completed.stderr.strip() or "no version output"
            raise LanaCompatibilityError(
                f"expected Lana 1.0.x with LABC v1; got: {detail}"
            )
        return match.group(1)

    @staticmethod
    def _program_path(program: str | os.PathLike[str]) -> Path:
        path = Path(program).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Lana program not found: {program}")
        return path

    @staticmethod
    def _vm_arguments(
        *,
        seed: int | None,
        memory_limit_mib: int | None,
        instruction_limit: int | None,
        workers: int | None,
        max_tasks: int | None,
    ) -> list[str]:
        values = {
            "seed": seed,
            "memory-limit-mib": memory_limit_mib,
            "instruction-limit": instruction_limit,
            "workers": workers,
            "max-tasks": max_tasks,
        }
        arguments: list[str] = []
        for name, value in values.items():
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            arguments.extend([f"--{name}", str(value)])
        return arguments

    def _execute(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        timeout_seconds: float | None,
    ) -> tuple[subprocess.CompletedProcess[str] | None, float, dict[str, Any] | None]:
        timeout = self.timeout_seconds if timeout_seconds is None else timeout_seconds
        if timeout <= 0:
            raise ValueError("timeout_seconds must be positive")
        started = time.monotonic()
        try:
            completed = subprocess.run(
                list(command),
                cwd=cwd,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            elapsed = time.monotonic() - started
            stdout = error.stdout if isinstance(error.stdout, str) else ""
            stderr = error.stderr if isinstance(error.stderr, str) else ""
            return None, elapsed, _error_envelope(
                "timeout",
                f"Lana exceeded the {timeout:g}s timeout",
                code="LANA_BRIDGE_TIMEOUT",
                stdout=stdout,
                stderr=stderr,
                execution={"elapsed_seconds": elapsed, "lana_version": self.version},
            )
        return completed, time.monotonic() - started, None

    def check(
        self,
        program: str | os.PathLike[str],
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        path = self._program_path(program)
        completed, elapsed, timeout_error = self._execute(
            [self.executable, "check", str(path)],
            cwd=path.parent,
            timeout_seconds=timeout_seconds,
        )
        if timeout_error is not None:
            return timeout_error
        assert completed is not None
        execution = {"elapsed_seconds": elapsed, "lana_version": self.version}
        if completed.returncode != 0:
            return _error_envelope(
                "check",
                "Lana rejected the program",
                code="LANA_CHECK_FAILED",
                exit_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                execution=execution,
            )
        return {
            "schema": SCHEMA_VERSION,
            "ok": True,
            "result": {"checked": str(path)},
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "execution": execution,
        }

    def run_plain(
        self,
        program: str | os.PathLike[str],
        *,
        seed: int | None = None,
        memory_limit_mib: int | None = None,
        instruction_limit: int | None = None,
        workers: int | None = None,
        max_tasks: int | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        path = self._program_path(program)
        vm_arguments = self._vm_arguments(
            seed=seed,
            memory_limit_mib=memory_limit_mib,
            instruction_limit=instruction_limit,
            workers=workers,
            max_tasks=max_tasks,
        )
        completed, elapsed, timeout_error = self._execute(
            [self.executable, "run", str(path), *vm_arguments],
            cwd=path.parent,
            timeout_seconds=timeout_seconds,
        )
        if timeout_error is not None:
            return timeout_error
        assert completed is not None
        execution = {"elapsed_seconds": elapsed, "lana_version": self.version}
        if completed.returncode != 0:
            return _error_envelope(
                "run",
                "Lana execution failed",
                code="LANA_RUN_FAILED",
                exit_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                execution=execution,
            )
        return {
            "schema": SCHEMA_VERSION,
            "ok": True,
            "result": None,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "execution": execution,
        }

    def run(
        self,
        program: str | os.PathLike[str],
        input_value: Any,
        *,
        seed: int | None = None,
        memory_limit_mib: int | None = None,
        instruction_limit: int | None = None,
        workers: int | None = None,
        max_tasks: int | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        path = self._program_path(program)
        vm_arguments = self._vm_arguments(
            seed=seed,
            memory_limit_mib=memory_limit_mib,
            instruction_limit=instruction_limit,
            workers=workers,
            max_tasks=max_tasks,
        )
        with tempfile.TemporaryDirectory(prefix="lana-bridge-") as temporary:
            temporary_path = Path(temporary)
            request_path = temporary_path / "request.json"
            response_path = temporary_path / "response.json"
            request_path.write_text(
                json.dumps(input_value, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            command = [
                self.executable,
                "run",
                str(path),
                *vm_arguments,
                "--",
                str(request_path),
                str(response_path),
            ]
            completed, elapsed, timeout_error = self._execute(
                command, cwd=path.parent, timeout_seconds=timeout_seconds
            )
            if timeout_error is not None:
                return timeout_error
            assert completed is not None
            execution = {"elapsed_seconds": elapsed, "lana_version": self.version}
            if completed.returncode != 0:
                return _error_envelope(
                    "run",
                    "Lana execution failed",
                    code="LANA_RUN_FAILED",
                    exit_code=completed.returncode,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                    execution=execution,
                )
            if not response_path.is_file():
                return _error_envelope(
                    "protocol",
                    "program did not write the response JSON file",
                    code="LANA_RESPONSE_MISSING",
                    exit_code=completed.returncode,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                    execution=execution,
                )
            try:
                result = json.loads(response_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                return _error_envelope(
                    "protocol",
                    f"program wrote invalid response JSON: {error}",
                    code="LANA_RESPONSE_INVALID",
                    exit_code=completed.returncode,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                    execution=execution,
                )
            return {
                "schema": SCHEMA_VERSION,
                "ok": True,
                "result": result,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "execution": execution,
            }

    def run_evidence(
        self,
        program: str | os.PathLike[str],
        evidence: Mapping[str, Any],
        *,
        seed: int | None = None,
        memory_limit_mib: int | None = None,
        instruction_limit: int | None = None,
        workers: int | None = None,
        max_tasks: int | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Run a program with one validated evidence record."""

        return self.run(
            program,
            validate_evidence(evidence),
            seed=seed,
            memory_limit_mib=memory_limit_mib,
            instruction_limit=instruction_limit,
            workers=workers,
            max_tasks=max_tasks,
            timeout_seconds=timeout_seconds,
        )
