"""Optional, explicit Ollama proposal adapter.

This module is intentionally outside the deterministic native VM. Callers must
select a model and validate the returned proposal before turning it into Lana
state or a model-fitting request.
"""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaError(RuntimeError):
    """Raised when a local Ollama proposal request is invalid or fails."""


def propose(
    model: str,
    prompt: str,
    *,
    base_url: str | None = None,
    timeout: float = 30.0,
    max_response_chars: int = 100_000,
) -> dict[str, object]:
    """Request one JSON proposal from an explicitly selected local model."""
    if not model.strip():
        raise OllamaError("model is required")
    if not prompt.strip():
        raise OllamaError("prompt is required")
    if timeout <= 0 or max_response_chars <= 0:
        raise OllamaError("timeout and max_response_chars must be positive")
    endpoint = (base_url or os.environ.get("LANA_OLLAMA_URL", "http://127.0.0.1:11434")).rstrip("/")
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
        }
    ).encode("utf-8")
    request = Request(
        f"{endpoint}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(max_response_chars + 1)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise OllamaError(f"Ollama request failed: {exc}") from exc
    if len(raw) > max_response_chars:
        raise OllamaError("Ollama response exceeded max_response_chars")
    try:
        envelope = json.loads(raw.decode("utf-8"))
        content = envelope["message"]["content"]
        proposal = json.loads(content)
    except (UnicodeDecodeError, KeyError, TypeError, ValueError) as exc:
        raise OllamaError("Ollama returned a non-JSON proposal") from exc
    if not isinstance(proposal, dict):
        raise OllamaError("Ollama proposal must be a JSON object")
    return proposal
