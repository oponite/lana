"""Typed client for the lana_http_service evidence endpoint."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict


class LanaHttpError(Exception):
    """Raised when the evidence service returns a structured error.

    Carries the Lana error code string (e.g. "LANA_ERR_NOT_FOUND") so callers
    can map it to the numeric LanaError value.
    """

    def __init__(self, code: str, message: str, status: int):
        super().__init__(f"{code}: {message} (HTTP {status})")
        self.code = code
        self.message = message
        self.status = status


def _raise_for_payload(payload: Dict[str, Any], status: int) -> None:
    if payload.get("ok", True):
        return
    error = payload.get("error", {})
    raise LanaHttpError(
        error.get("code", "LANA_ERR_IO"),
        error.get("message", "evidence service error"),
        status,
    )


def fetch_evidence(base_url: str, evidence_id: str) -> Dict[str, Any]:
    """Fetch evidence from lana_http_service.

    ``base_url`` is like ``http://127.0.0.1:8080``. Returns the evidence object
    (the ``evidence`` field of the response envelope).
    """
    url = f"{base_url.rstrip('/')}/evidence/{evidence_id}"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            raise LanaHttpError("LANA_ERR_IO", "non-JSON error response", error.code) from error
        _raise_for_payload(payload, error.code)
        raise LanaHttpError("LANA_ERR_IO", "unexpected HTTP error", error.code) from error
    except urllib.error.URLError as error:
        raise LanaHttpError("LANA_ERR_IO", str(error.reason), 0) from error
    _raise_for_payload(payload, 200)
    return payload.get("evidence", payload)
