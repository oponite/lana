"""ctypes wrapper for the public Lana C API in liblanaruntime."""

from __future__ import annotations

import ctypes
from ctypes.util import find_library
import os
from pathlib import Path
from typing import Any, Optional, List, Dict

# Constants from lana/error.h and others
LANA_OK = 0
LANA_ERR_INVALID_STATE = 1
LANA_ERR_OOM = 2
LANA_ERR_IO = 3
LANA_ERR_SCHEMA = 4
LANA_ERR_CORRUPTION = 5
LANA_ERR_LIMIT = 6
LANA_ERR_NOT_FOUND = 7
LANA_ERR_CONFLICT = 8
LANA_ERR_TYPE = 9
LANA_ERR_KEY = 10
LANA_ERR_INCOMPATIBLE_FORMAT = 11
LANA_ERR_INVALID_PROBABILITY = 12
LANA_ERR_INTEGRITY = 13
LANA_ERR_CAPABILITY = 14

# Types
LanaStore = ctypes.c_void_p
LanaVM = ctypes.c_void_p
LanaAdapter = ctypes.c_void_p

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

class LanaValue(ctypes.Structure):
    # This is a complex union/struct. We'll treat it as an opaque blob 
    # and use helper functions for extraction.
    _fields_ = [("opaque", ctypes.c_char * 256)]

class LanaAPI:
    def __init__(self, library_path: Optional[str] = None):
        path = library_path or os.environ.get("LANA_RUNTIME_LIBRARY") or find_library("lanaruntime")
        if not path:
            raise FileNotFoundError("liblanaruntime not found")
        self._lib = ctypes.CDLL(os.fspath(Path(path).expanduser()))
        self._setup_functions()

    def _setup_functions(self):
        # Store API
        self._lib.lana_store_open.argtypes = [ctypes.POINTER(LanaStoreOptions), ctypes.POINTER(LanaStore)]
        self._lib.lana_store_open.restype = ctypes.c_int
        
        self._lib.lana_store_close.argtypes = [LanaStore]
        self._lib.lana_store_close.restype = ctypes.c_int
        
        self._lib.lana_store_get.argtypes = [LanaStore, LanaVM, ctypes.c_char_p, ctypes.POINTER(LanaValue)]
        self._lib.lana_store_get.restype = ctypes.c_int
        
        self._lib.lana_store_put.argtypes = [LanaStore, ctypes.c_char_p, ctypes.POINTER(LanaValue)]
        self._lib.lana_store_put.restype = ctypes.c_int
        
        self._lib.lana_store_commit.argtypes = [LanaStore, ctypes.POINTER(LanaStoreRevisionInfo)]
        self._lib.lana_store_commit.restype = ctypes.c_int
        
        self._lib.lana_store_current_revision.argtypes = [LanaStore, ctypes.POINTER(LanaStoreRevisionInfo)]
        self._lib.lana_store_current_revision.restype = ctypes.c_int

        # Adapter API
        self._lib.lana_adapter_load.argtypes = [ctypes.POINTER(LanaAdapterOptions), ctypes.POINTER(LanaAdapter)]
        self._lib.lana_adapter_load.restype = ctypes.c_int
        
        self._lib.lana_adapter_fetch.argtypes = [LanaAdapter, LanaVM, ctypes.c_char_p, ctypes.POINTER(LanaValue)]
        self._lib.lana_adapter_fetch.restype = ctypes.c_int

        self._lib.lana_adapter_close.argtypes = [LanaAdapter]
        self._lib.lana_adapter_close.restype = None
        
        # VM API (Minimal)
        self._lib.lana_vm_create.argtypes = []
        self._lib.lana_vm_create.restype = LanaVM

        self._lib.lana_vm_destroy.argtypes = [LanaVM]
        self._lib.lana_vm_destroy.restype = None

    def open_store(self, path: str) -> LanaStore:
        opts = LanaStoreOptions()
        opts.struct_size = ctypes.sizeof(LanaStoreOptions)
        opts.schema_version = 1
        opts.path = os.fsencode(path)
        
        store = LanaStore()
        res = self._lib.lana_store_open(ctypes.byref(opts), ctypes.byref(store))
        if res != LANA_OK:
            raise RuntimeError(f"lana_store_open failed with {res}")
        return store

    def get_value(self, store: LanaStore, vm: LanaVM, key: str) -> bytes:
        val = LanaValue()
        res = self._lib.lana_store_get(store, vm, os.fsencode(key), ctypes.byref(val))
        if res != LANA_OK:
            raise RuntimeError(f"lana_store_get failed with {res}")
        return bytes(val.opaque)

    def load_adapter(self, kind: int, config: Optional[str] = None) -> LanaAdapter:
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

    def fetch_evidence(self, adapter: LanaAdapter, vm: LanaVM, query: str) -> bytes:
        val = LanaValue()
        res = self._lib.lana_adapter_fetch(adapter, vm, os.fsencode(query), ctypes.byref(val))
        if res != LANA_OK:
            raise RuntimeError(f"lana_adapter_fetch failed with {res}")
        return bytes(val.opaque)

    def close_adapter(self, adapter: LanaAdapter) -> None:
        self._lib.lana_adapter_close(adapter)

    def init_vm(self) -> LanaVM:
        vm = self._lib.lana_vm_create()
        if not vm:
            raise RuntimeError("lana_vm_create failed")
        return vm

    def free_vm(self, vm: LanaVM) -> None:
        self._lib.lana_vm_destroy(vm)
