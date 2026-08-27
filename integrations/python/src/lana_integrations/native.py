"""ctypes wrapper for the optional liblana_bridge ABI v1."""

from __future__ import annotations

import ctypes
from ctypes.util import find_library
import json
import os
from pathlib import Path
import re
from typing import Any


LANA_BRIDGE_ABI_VERSION = 1
_SUPPORTED_VERSION = re.compile(r"1\.(?:0|1)\.\d+")


def _is_supported_version(version: str) -> bool:
    return _SUPPORTED_VERSION.fullmatch(version) is not None


class LanaBridgeOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("abi_version", ctypes.c_uint32),
        ("seed", ctypes.c_uint64),
        ("instruction_limit", ctypes.c_uint64),
        ("memory_limit_bytes", ctypes.c_size_t),
        ("workers", ctypes.c_size_t),
        ("max_tasks", ctypes.c_size_t),
    ]

    @classmethod
    def defaults(cls) -> "LanaBridgeOptions":
        options = cls()
        options.struct_size = ctypes.sizeof(cls)
        options.abi_version = LANA_BRIDGE_ABI_VERSION
        return options


class NativeBridge:
    """Run precompiled LABC through the stable native integration facade."""

    def __init__(self, library: str | os.PathLike[str] | None = None) -> None:
        requested = (
            os.fspath(library)
            if library is not None
            else os.environ.get("LANA_BRIDGE_LIBRARY") or find_library("lana_bridge")
        )
        if not requested:
            raise FileNotFoundError(
                "liblana_bridge not found; pass its path or set LANA_BRIDGE_LIBRARY"
            )
        self.library_path = os.fspath(Path(requested).expanduser())
        self._library = ctypes.CDLL(self.library_path)
        self._library.lana_bridge_version.argtypes = []
        self._library.lana_bridge_version.restype = ctypes.c_char_p
        self._library.lana_bridge_run_labc.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(LanaBridgeOptions),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._library.lana_bridge_run_labc.restype = ctypes.c_int
        self._library.lana_bridge_free.argtypes = [ctypes.c_void_p]
        self._library.lana_bridge_free.restype = None
        version = self._library.lana_bridge_version()
        self.version = version.decode("utf-8") if version else ""
        if not _is_supported_version(self.version):
            raise RuntimeError(f"unsupported lana_bridge version: {self.version}")

    def run_labc(
        self,
        labc_path: str | os.PathLike[str],
        request_path: str | os.PathLike[str],
        response_path: str | os.PathLike[str],
        options: LanaBridgeOptions | None = None,
    ) -> dict[str, Any]:
        configured = options or LanaBridgeOptions.defaults()
        envelope_pointer = ctypes.c_void_p()
        status = self._library.lana_bridge_run_labc(
            os.fsencode(labc_path),
            os.fsencode(request_path),
            os.fsencode(response_path),
            ctypes.byref(configured),
            ctypes.byref(envelope_pointer),
        )
        if not envelope_pointer.value:
            raise RuntimeError(f"native bridge returned {status} without an envelope")
        try:
            envelope = json.loads(ctypes.string_at(envelope_pointer.value).decode("utf-8"))
        finally:
            self._library.lana_bridge_free(envelope_pointer)
        if status == 0 and not envelope.get("ok"):
            raise RuntimeError("native bridge returned inconsistent success status")
        envelope["native_status"] = status
        return envelope
