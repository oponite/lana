"""Persistent JSON connection to the Rust Lana executable."""

from __future__ import annotations

import json
from pathlib import Path
import queue
import subprocess
import threading
import weakref
from typing import Any

from .bridge import _validate_timeout, _validate_vm_options


def _read_replies(stream, replies: queue.Queue[str]) -> None:
    for line in stream:
        replies.put(line)
    replies.put("")


def _stop(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=1)


class WorkerRunner:
    def __init__(self, executable: str, timeout_seconds: float) -> None:
        self._timeout = _validate_timeout(timeout_seconds)
        self._process = subprocess.Popen(
            [executable, "bridge-worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        assert self._process.stdin is not None and self._process.stdout is not None
        self._lock = threading.Lock()
        self._replies: queue.Queue[str] = queue.Queue()
        threading.Thread(
            target=_read_replies,
            args=(self._process.stdout, self._replies),
            daemon=True,
        ).start()
        self._finalizer = weakref.finalize(self, _stop, self._process)

    def run(
        self, operation: str, path: str | Path, input_value: Any,
        *, timeout_seconds: float | None = None, **options: int | None,
    ) -> dict[str, Any]:
        timeout = self._timeout if timeout_seconds is None else _validate_timeout(timeout_seconds)
        controls = _validate_vm_options(**options)
        request = json.dumps(
            {"schema": 1, "op": operation, "path": str(Path(path).absolute()), "input": input_value, **controls},
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        with self._lock:
            if self._process.poll() is not None:
                raise RuntimeError("Lana worker exited")
            try:
                assert self._process.stdin is not None
                self._process.stdin.write(request + "\n")
                self._process.stdin.flush()
                line = self._replies.get(timeout=timeout)
            except (BrokenPipeError, OSError, queue.Empty) as error:
                self.close()
                raise RuntimeError("Lana worker failed or timed out") from error
            if not line:
                self.close()
                raise RuntimeError("Lana worker exited without a response")
            try:
                response = json.loads(line)
            except json.JSONDecodeError as error:
                self.close()
                raise RuntimeError("Lana worker sent invalid JSON") from error
            if not isinstance(response, dict) or response.get("schema") != 1:
                self.close()
                raise RuntimeError("Lana worker sent an invalid response")
            return response

    def close(self) -> None:
        self._finalizer()
