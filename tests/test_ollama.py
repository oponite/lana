import pytest

from lana_tooling.ollama import OllamaError, propose


def test_ollama_requires_explicit_model_and_prompt() -> None:
    with pytest.raises(OllamaError, match="model is required"):
        propose("", "prompt")
    with pytest.raises(OllamaError, match="prompt is required"):
        propose("qwen3:14b", "")
