from __future__ import annotations

import os
from pathlib import Path

import pytest

from lana_integrations.api_rust import ADAPTER_JSON, LanaAdapterOptions, RustLanaAPI


def _api() -> RustLanaAPI:
    path = os.environ.get("LANA_FFI_LIBRARY") or os.environ.get("LANA_RUNTIME_LIBRARY")
    if not path:
        pytest.skip("LANA_FFI_LIBRARY not set")
    return RustLanaAPI(path)


def test_adapter_options_layout() -> None:
    fields = [name for name, _ in LanaAdapterOptions._fields_]
    assert fields == ["struct_size", "schema_version", "kind", "config"]


def test_json_adapter_fetch_round_trip() -> None:
    api = _api()
    vm = api.init_vm()
    adapter = api.load_adapter(ADAPTER_JSON)
    value = api.fetch_evidence(adapter, vm, '{"p":0.9,"n":[1,2,3]}')
    assert value == {"p": 0.9, "n": [1, 2, 3]}
    api.close_adapter(adapter)
    api.free_vm(vm)


def test_store_put_commit_get_round_trip(tmp_path: Path) -> None:
    api = _api()
    vm = api.init_vm()
    store = api.open_store(str(tmp_path / "store"))
    api.put_value(store, "belief", {"p": 0.75, "ok": True})
    revision = api.commit(store)
    assert revision["revision_id"] == 1
    current = api.current_revision(store)
    assert current["revision_id"] == 1
    assert api.get_value(store, vm, "belief") == {"p": 0.75, "ok": True}
    api.close_store(store)
    api.free_vm(vm)


def test_close_adapter_null_safe() -> None:
    api = _api()
    api.close_adapter(None)  # must not raise
