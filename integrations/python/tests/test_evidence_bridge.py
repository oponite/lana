from __future__ import annotations

from pathlib import Path

import pytest

from lana_integrations.bridge import BridgeRunner


ROOT = Path(__file__).resolve().parents[3]


def built_lana() -> Path:
    executable = ROOT / "build" / "lana"
    if not executable.is_file():
        pytest.skip("requires a built Lana executable")
    return executable


def test_evidence_bridge_round_trip() -> None:
    executable = built_lana()
    record = {
        "schema": 1,
        "status": "resolved",
        "source": "fixture",
        "observed_at": 100,
        "effective_at": 100,
        "exactness": "exact",
        "revision": 7,
        "confidence": 0.9,
        "provenance_id": "fixture:7",
        "dependency_ids": [],
        "value": False,
    }
    envelope = BridgeRunner(executable).run_evidence(
        ROOT / "integrations" / "lana" / "evidence_bridge.lana", record
    )
    assert envelope["ok"] is True
    assert envelope["result"] == {"schema": 1, "kind": "evidence", "record": record}


def test_replay_bridge_round_trip() -> None:
    executable = built_lana()
    evidence = {
        "schema": 1,
        "status": "resolved",
        "source": "fixture",
        "observed_at": 100,
        "effective_at": 100,
        "exactness": "exact",
        "revision": 7,
        "confidence": 0.9,
        "provenance_id": "fixture:7",
        "dependency_ids": [],
        "value": 0.8,
    }
    request = {
        "schema": 1,
        "kind": "threshold_authorization_request",
        "evidence": evidence,
        "policy": {"minimum": 0.7, "current_time": 120, "max_age": 30, "minimum_confidence": 0.8},
        "effect": {"kind": "io", "payload": {"job": "notify"}},
        "seed": 17,
    }
    runner = BridgeRunner(executable)
    recorded = runner.run(ROOT / "integrations" / "lana" / "replay_bridge.lana", request)
    assert recorded["ok"] is True
    assert recorded["result"]["authorization"]["authorized"] is True
    replayed = runner.run(
        ROOT / "integrations" / "lana" / "replay_bridge.lana",
        {"schema": 1, "kind": "threshold_authorization_replay", "record": recorded["result"]},
    )
    assert replayed["ok"] is True
    assert replayed["result"] == recorded["result"]


@pytest.mark.parametrize(
    ("kind", "metric", "value", "status", "expected_action"),
    [
        ("service_health_request", "availability_fraction", 0.9, "resolved", "serve"),
        ("document_routing_request", "classifier_score", 0.6, "resolved", "human_review"),
        ("sensor_fusion_request", "safety_margin", 0, "not_measured", "request_measurement"),
        ("advisory_forecast_request", "forecast_confidence", 0.9, "conflict", "review_model_evidence"),
    ],
)
def test_reference_applications(
    kind: str, metric: str, value: object, status: str, expected_action: str
) -> None:
    evidence = {
        "schema": 1,
        "status": status,
        "source": "fixture",
        "observed_at": 100,
        "effective_at": 100,
        "exactness": "exact",
        "revision": 7,
        "confidence": 0.9,
        "provenance_id": "fixture:7",
        "dependency_ids": [],
    }
    if status == "resolved":
        evidence["value"] = value
    request = {
        "schema": 1,
        "kind": kind,
        "metric": metric,
        "source_assumptions": "fixture supplies the named metric",
        "evidence": evidence,
        "policy": {"minimum": 0.7, "current_time": 120, "max_age": 30, "minimum_confidence": 0.8},
    }
    envelope = BridgeRunner(built_lana()).run(
        ROOT / "integrations" / "lana" / "reference_apps_bridge.lana", request
    )
    assert envelope["ok"] is True
    assert envelope["result"]["action"] == expected_action
