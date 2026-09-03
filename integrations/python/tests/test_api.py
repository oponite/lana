from __future__ import annotations

import os

import pytest

from lana_integrations.api import LanaAdapterOptions, LanaAPI


def _api() -> LanaAPI:
    path = os.environ.get("LANA_RUNTIME_LIBRARY")
    if not path:
        pytest.skip("LANA_RUNTIME_LIBRARY not set")
    return LanaAPI(path)


def test_adapter_options_layout() -> None:
    fields = [name for name, _ in LanaAdapterOptions._fields_]
    assert fields == ["struct_size", "schema_version", "kind", "config"]


def test_json_adapter_fetch() -> None:
    api = _api()
    vm = api.init_vm()
    adapter = api.load_adapter(0)  # ADAPTER_JSON
    raw = api.fetch_evidence(adapter, vm, '{"p":0.9}')
    assert isinstance(raw, bytes) and len(raw) > 0
    api.close_adapter(adapter)
    api.free_vm(vm)


def test_json_adapter_malformed_raises() -> None:
    api = _api()
    vm = api.init_vm()
    adapter = api.load_adapter(0)
    with pytest.raises(RuntimeError):
        api.fetch_evidence(adapter, vm, "{not-json")
    api.close_adapter(adapter)
    api.free_vm(vm)


def test_close_adapter_null_safe() -> None:
    api = _api()
    api.close_adapter(None)  # must not raise
