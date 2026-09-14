"""ctypes wrapper for the Rust ``lana-ffi`` crate (handle-based ABI).

Unlike :mod:`lana_integrations.api`, which binds the C ``liblanaruntime``
public API and its by-value ``LanaValue`` union (``ctypes.c_char * 256``), the
Rust FFI crate represents a value as an opaque ``*mut Value`` handle.  This
module binds that handle-based ABI directly: every function that takes or
returns a value uses ``ctypes.c_void_p``, values are created with
``lana_json_parse`` (or the ``lana_value_*`` constructors) and released with
``lana_value_free``.

The Rust FFI also exposes the native bridge trio (``lana_bridge_version`` /
``lana_bridge_run_labc`` / ``lana_bridge_free``); that surface is already bound
by :mod:`lana_integrations.native`, which uses ``ctypes.c_void_p`` throughout
and works against this same ``liblana_ffi`` artifact unchanged.
"""

from __future__ import annotations

import ctypes
from ctypes.util import find_library
import json
import os
from pathlib import Path
from typing import Any, Optional

# Error codes, matching `LanaError` in `vm/rust/lana-bytecode/src/error.rs`
# (repr(i32), discriminants 0..=47).  The Rust FFI returns these as `i32`.
LANA_OK = 0
LANA_ERR_INVALID_STATE = 1
LANA_ERR_INVALID_PROBABILITY = 2
LANA_ERR_INVALID_DEPENDENCY = 3
LANA_ERR_TYPE = 4
LANA_ERR_REGISTER = 5
LANA_ERR_OPCODE = 6
LANA_ERR_CONSTANT = 7
LANA_ERR_JUMP = 8
LANA_ERR_TRANSFORM = 9
LANA_ERR_COMPOSE = 10
LANA_ERR_MEASURE = 11
LANA_ERR_HISTORY = 12
LANA_ERR_FORMAT = 13
LANA_ERR_INCOMPATIBLE_FORMAT = 14
LANA_ERR_IO = 15
LANA_ERR_OOM = 16
LANA_ERR_LIMIT = 17
LANA_ERR_TASK = 18
LANA_ERR_CANCELLED = 19
LANA_ERR_TIMEOUT = 20
LANA_ERR_INVALID_TRANSFORM_RESULT = 21
LANA_ERR_UNSUPPORTED_OPERATION = 22
LANA_ERR_UNSUPPORTED_EXACT_MEASUREMENT = 23
LANA_ERR_INVALID_DISTRIBUTION = 24
LANA_ERR_BUDGET_EXHAUSTED = 25
LANA_ERR_KEY = 26
LANA_ERR_PARSE = 27
LANA_ERR_ASSERTION = 28
LANA_ERR_INVALID_CONDITIONING = 29
LANA_ERR_UNRESOLVED_VALUE = 30
LANA_ERR_PATH_LIMIT = 31
LANA_ERR_CAPABILITY = 32
LANA_ERR_CONFLICT = 33
LANA_ERR_NOT_FOUND = 34
LANA_ERR_COMPACTED_HISTORY = 35
LANA_ERR_SCHEMA = 36
LANA_ERR_UNSUPPORTED_VALUE = 37
LANA_ERR_CORRUPTION = 38
LANA_ERR_INVALID_PARAMETERS = 39
LANA_ERR_CLAIM_MISMATCH = 40
LANA_ERR_CLAIM_REVOKED = 41
LANA_ERR_CLAIM_EXPIRED = 42
LANA_ERR_UNAUTHORIZED_ISSUER = 43
LANA_ERR_INTEGRITY = 44
LANA_ERR_NO_MATCHING_EVENT = 45
LANA_ERR_EXTERNAL = 46
LANA_ERR_NETWORK = 47

# Adapter kinds, matching `LanaAdapterKind` in `runtime/include/adapters.h`.
ADAPTER_JSON = 0
ADAPTER_CSV = 1
ADAPTER_SQLITE = 2
ADAPTER_HTTP_JSON = 3

# Opaque handles (the Rust FFI `Box<T> -> *mut T` representation).
LanaStore = ctypes.c_void_p
LanaVM = ctypes.c_void_p
LanaAdapter = ctypes.c_void_p
# A value handle is `*mut lana_vm::value::Value`, NOT the C by-value union.
LanaValue = ctypes.c_void_p


class LanaStoreOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("schema_version", ctypes.c_uint32),
        ("path", ctypes.c_char_p),
        ("timeout_ms", ctypes.c_uint32),
    ]


class LanaStoreRevisionInfo(ctypes.Structure):
    _fields_ = [
        ("revision_id", ctypes.c_uint64),
        ("schema_version", ctypes.c_uint32),
        ("timestamp", ctypes.c_uint64),
        ("digest", ctypes.c_ubyte * 32),
    ]


class LanaAdapterOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("schema_version", ctypes.c_uint32),
        ("kind", ctypes.c_int),
        ("config", ctypes.c_char_p),
    ]


class RustLanaAPI:
    """Drop-in handle-based binding for the Rust ``lana-ffi`` cdylib."""

    def __init__(self, library_path: Optional[str] = None) -> None:
        path = (
            library_path
            or os.environ.get("LANA_FFI_LIBRARY")
            or os.environ.get("LANA_RUNTIME_LIBRARY")
            or find_library("lana_ffi")
        )
        if not path:
            raise FileNotFoundError(
                "liblana_ffi not found; pass its path or set LANA_FFI_LIBRARY"
            )
        self._lib = ctypes.CDLL(os.fspath(Path(path).expanduser()))
        self._setup_functions()

    def _setup_functions(self) -> None:
        # Store API
        self._lib.lana_store_open.argtypes = [
            ctypes.POINTER(LanaStoreOptions),
            ctypes.POINTER(LanaStore),
        ]
        self._lib.lana_store_open.restype = ctypes.c_int

        self._lib.lana_store_close.argtypes = [LanaStore]
        self._lib.lana_store_close.restype = ctypes.c_int

        self._lib.lana_store_get.argtypes = [
            LanaStore,
            LanaVM,
            ctypes.c_char_p,
            ctypes.POINTER(LanaValue),
        ]
        self._lib.lana_store_get.restype = ctypes.c_int

        self._lib.lana_store_put.argtypes = [LanaStore, ctypes.c_char_p, LanaValue]
        self._lib.lana_store_put.restype = ctypes.c_int

        self._lib.lana_store_commit.argtypes = [
            LanaStore,
            ctypes.POINTER(LanaStoreRevisionInfo),
        ]
        self._lib.lana_store_commit.restype = ctypes.c_int

        self._lib.lana_store_current_revision.argtypes = [
            LanaStore,
            ctypes.POINTER(LanaStoreRevisionInfo),
        ]
        self._lib.lana_store_current_revision.restype = ctypes.c_int

        # Adapter API
        self._lib.lana_adapter_load.argtypes = [
            ctypes.POINTER(LanaAdapterOptions),
            ctypes.POINTER(LanaAdapter),
        ]
        self._lib.lana_adapter_load.restype = ctypes.c_int

        self._lib.lana_adapter_fetch.argtypes = [
            LanaAdapter,
            LanaVM,
            ctypes.c_char_p,
            ctypes.POINTER(LanaValue),
        ]
        self._lib.lana_adapter_fetch.restype = ctypes.c_int

        self._lib.lana_adapter_close.argtypes = [LanaAdapter]
        self._lib.lana_adapter_close.restype = None

        # VM API
        self._lib.lana_vm_create.argtypes = []
        self._lib.lana_vm_create.restype = LanaVM

        self._lib.lana_vm_destroy.argtypes = [LanaVM]
        self._lib.lana_vm_destroy.restype = None

        # Value handles
        self._lib.lana_value_free.argtypes = [LanaValue]
        self._lib.lana_value_free.restype = None

        self._lib.lana_value_null.argtypes = []
        self._lib.lana_value_null.restype = LanaValue

        self._lib.lana_value_number.argtypes = [ctypes.c_double]
        self._lib.lana_value_number.restype = LanaValue

        self._lib.lana_value_bool.argtypes = [ctypes.c_bool]
        self._lib.lana_value_bool.restype = LanaValue

        self._lib.lana_value_string.argtypes = [ctypes.c_char_p]
        self._lib.lana_value_string.restype = LanaValue

        # JSON round-trip helpers (create a value handle / read a value as JSON)
        self._lib.lana_json_parse.argtypes = [
            LanaVM,
            ctypes.c_char_p,
            ctypes.POINTER(LanaValue),
        ]
        self._lib.lana_json_parse.restype = ctypes.c_int

        self._lib.lana_buffer_new.argtypes = []
        self._lib.lana_buffer_new.restype = ctypes.c_void_p

        self._lib.lana_buffer_free.argtypes = [ctypes.c_void_p]
        self._lib.lana_buffer_free.restype = None

        self._lib.lana_codec_encode_value.argtypes = [
            ctypes.c_void_p,
            LanaValue,
        ]
        self._lib.lana_codec_encode_value.restype = ctypes.c_int

        self._lib.lana_buffer_data.argtypes = [ctypes.c_void_p]
        self._lib.lana_buffer_data.restype = ctypes.c_void_p

        self._lib.lana_buffer_length.argtypes = [ctypes.c_void_p]
        self._lib.lana_buffer_length.restype = ctypes.c_size_t

    # -- value helpers -----------------------------------------------------

    def _value_to_json(self, handle: Any) -> str:
        if not handle:
            return "null"
        buffer = self._lib.lana_buffer_new()
        try:
            code = self._lib.lana_codec_encode_value(buffer, handle)
            if code != LANA_OK:
                raise RuntimeError(f"lana_codec_encode_value failed with {code}")
            data = self._lib.lana_buffer_data(buffer)
            length = self._lib.lana_buffer_length(buffer)
            return ctypes.string_at(data, length).decode("utf-8")
        finally:
            self._lib.lana_buffer_free(buffer)

    def _value_from_json(self, text: str) -> Any:
        handle = LanaValue()
        code = self._lib.lana_json_parse(None, text.encode("utf-8"), ctypes.byref(handle))
        if code != LANA_OK:
            raise RuntimeError(f"lana_json_parse failed with {code}")
        return handle

    # -- store -------------------------------------------------------------

    def open_store(self, path: str) -> Any:
        opts = LanaStoreOptions()
        opts.struct_size = ctypes.sizeof(LanaStoreOptions)
        opts.schema_version = 1
        opts.path = os.fsencode(path)

        store = LanaStore()
        res = self._lib.lana_store_open(ctypes.byref(opts), ctypes.byref(store))
        if res != LANA_OK:
            raise RuntimeError(f"lana_store_open failed with {res}")
        return store

    def close_store(self, store: Any) -> None:
        res = self._lib.lana_store_close(store)
        if res != LANA_OK:
            raise RuntimeError(f"lana_store_close failed with {res}")

    def get_value(self, store: Any, vm: Any, key: str) -> Any:
        handle = LanaValue()
        res = self._lib.lana_store_get(store, vm, os.fsencode(key), ctypes.byref(handle))
        if res != LANA_OK:
            raise RuntimeError(f"lana_store_get failed with {res}")
        try:
            return json.loads(self._value_to_json(handle))
        finally:
            self._lib.lana_value_free(handle)

    def put_value(self, store: Any, key: str, value: Any) -> None:
        handle = self._value_from_json(
            json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        )
        try:
            res = self._lib.lana_store_put(store, os.fsencode(key), handle)
            if res != LANA_OK:
                raise RuntimeError(f"lana_store_put failed with {res}")
        finally:
            self._lib.lana_value_free(handle)

    def commit(self, store: Any) -> dict[str, Any]:
        info = LanaStoreRevisionInfo()
        res = self._lib.lana_store_commit(store, ctypes.byref(info))
        if res != LANA_OK:
            raise RuntimeError(f"lana_store_commit failed with {res}")
        return self._revision_dict(info)

    def current_revision(self, store: Any) -> dict[str, Any]:
        info = LanaStoreRevisionInfo()
        res = self._lib.lana_store_current_revision(store, ctypes.byref(info))
        if res != LANA_OK:
            raise RuntimeError(f"lana_store_current_revision failed with {res}")
        return self._revision_dict(info)

    @staticmethod
    def _revision_dict(info: LanaStoreRevisionInfo) -> dict[str, Any]:
        return {
            "revision_id": info.revision_id,
            "schema_version": info.schema_version,
            "timestamp": info.timestamp,
            "digest": bytes(info.digest),
        }

    # -- adapter -----------------------------------------------------------

    def load_adapter(self, kind: int, config: Optional[str] = None) -> Any:
        opts = LanaAdapterOptions()
        opts.struct_size = ctypes.sizeof(LanaAdapterOptions)
        opts.schema_version = 1
        opts.kind = kind
        opts.config = os.fsencode(config) if config else None

        adapter = LanaAdapter()
        res = self._lib.lana_adapter_load(ctypes.byref(opts), ctypes.byref(adapter))
        if res != LANA_OK:
            raise RuntimeError(f"lana_adapter_load failed with {res}")
        return adapter

    def fetch_evidence(self, adapter: Any, vm: Any, query: str) -> Any:
        handle = LanaValue()
        res = self._lib.lana_adapter_fetch(
            adapter, vm, os.fsencode(query), ctypes.byref(handle)
        )
        if res != LANA_OK:
            raise RuntimeError(f"lana_adapter_fetch failed with {res}")
        try:
            return json.loads(self._value_to_json(handle))
        finally:
            self._lib.lana_value_free(handle)

    def close_adapter(self, adapter: Any) -> None:
        self._lib.lana_adapter_close(adapter)

    # -- vm ----------------------------------------------------------------

    def init_vm(self) -> Any:
        vm = self._lib.lana_vm_create()
        if not vm:
            raise RuntimeError("lana_vm_create failed")
        return vm

    def free_vm(self, vm: Any) -> None:
        self._lib.lana_vm_destroy(vm)
