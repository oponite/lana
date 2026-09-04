"""Versioned evidence records for Lana 1.x integrations and callers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


EVIDENCE_SCHEMA_VERSION = 1
EVIDENCE_STATUSES = frozenset(
    {"resolved", "unknown", "not_measured", "insufficient_evidence", "conflict"}
)
EVIDENCE_EXACTNESS = frozenset({"exact", "sample", "approximate"})


class EvidenceValidationError(ValueError):
    """Raised when an evidence record violates the Lana bridge contract."""


def _require_string(record: Mapping[str, Any], name: str) -> None:
    value = record.get(name)
    if not isinstance(value, str) or not value:
        raise EvidenceValidationError(f"evidence {name} must be a nonempty string")


def _require_nonnegative_number(record: Mapping[str, Any], name: str) -> None:
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise EvidenceValidationError(f"evidence {name} must be a nonnegative number")


def validate_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a validated copy without coercing values or uncertainty states."""

    if not isinstance(value, Mapping):
        raise EvidenceValidationError("evidence must be an object")
    record = dict(value)
    if record.get("schema") != EVIDENCE_SCHEMA_VERSION or isinstance(
        record.get("schema"), bool
    ):
        raise EvidenceValidationError("unsupported evidence schema")
    status = record.get("status")
    if status not in EVIDENCE_STATUSES:
        raise EvidenceValidationError("unsupported evidence status")
    exactness = record.get("exactness")
    if exactness not in EVIDENCE_EXACTNESS:
        raise EvidenceValidationError("unsupported evidence exactness")
    for name in ("source", "provenance_id"):
        _require_string(record, name)
    for name in ("observed_at", "effective_at", "revision", "confidence"):
        _require_nonnegative_number(record, name)
    if record["confidence"] > 1:
        raise EvidenceValidationError("evidence confidence must not exceed one")
    dependencies = record.get("dependency_ids")
    if not isinstance(dependencies, list) or any(
        not isinstance(item, str) or not item for item in dependencies
    ):
        raise EvidenceValidationError("evidence dependency_ids must be an array of strings")
    if len(set(dependencies)) != len(dependencies):
        raise EvidenceValidationError("evidence dependency_ids must be unique")
    if status == "resolved" and "value" not in record:
        raise EvidenceValidationError("resolved evidence value is required")
    return record
