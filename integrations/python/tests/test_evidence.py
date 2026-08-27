from __future__ import annotations

import pytest

from lana_integrations.evidence import EvidenceValidationError, validate_evidence


def evidence(status: str = "resolved") -> dict[str, object]:
    record: dict[str, object] = {
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
        record["value"] = None
    return record


@pytest.mark.parametrize("value", [0, False, None])
def test_preserves_falsey_resolved_values(value: object) -> None:
    record = evidence()
    record["value"] = value
    assert validate_evidence(record)["value"] is value


@pytest.mark.parametrize("status", ["unknown", "not_measured", "insufficient_evidence", "conflict"])
def test_preserves_unresolved_statuses(status: str) -> None:
    assert validate_evidence(evidence(status))["status"] == status


def test_resolved_evidence_requires_value() -> None:
    record = evidence()
    del record["value"]
    with pytest.raises(EvidenceValidationError, match="value"):
        validate_evidence(record)


def test_rejects_implicit_status_and_duplicate_dependencies() -> None:
    record = evidence()
    record["status"] = "likely"
    with pytest.raises(EvidenceValidationError, match="status"):
        validate_evidence(record)
    record = evidence()
    record["dependency_ids"] = ["source:a", "source:a"]
    with pytest.raises(EvidenceValidationError, match="unique"):
        validate_evidence(record)


@pytest.mark.parametrize("dependencies", [["source:a", ""], ["source:a", 3]])
def test_rejects_invalid_dependency_identifiers(dependencies: list[object]) -> None:
    record = evidence()
    record["dependency_ids"] = dependencies
    with pytest.raises(EvidenceValidationError, match="dependency_ids"):
        validate_evidence(record)


def test_preserves_optional_lifecycle_metadata() -> None:
    record = evidence()
    record["reliability"] = {"score": 0.8, "method": "calibrated_fixture"}
    record["calibration"] = {"version": "fixture-1"}
    validated = validate_evidence(record)
    assert validated["reliability"] == record["reliability"]
    assert validated["calibration"] == record["calibration"]
