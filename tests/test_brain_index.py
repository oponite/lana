"""Semantic Brain retrieval stays opt-in and tied to one frozen corpus."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

lana = Path(sys.argv[1]).resolve()
repo = Path(__file__).resolve().parents[1]
environment = dict(os.environ, LANA_HF=str(repo / "tools/lana-hf/lana_hf.py"))

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    brain = root / "brain.lbrn"
    tokenizer = root / "tokenizer.json"
    tokenizer.write_text(json.dumps({
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": [],
        "normalizer": None, "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": None, "decoder": None,
        "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "name": 1,
            "Lana": 2, "hello": 3}, "unk_token": "[UNK]"},
    }))

    def call(*args, ok=True):
        result = subprocess.run([lana, "brain", *map(str, args)], env=environment,
                                capture_output=True, text=True, timeout=60)
        assert (result.returncode == 0) == ok, result.stdout + result.stderr
        return json.loads(result.stdout.splitlines()[-1]) if ok else result.stderr

    call("new", brain, 4, 2, 2, 7)
    call("remember", brain, "name", "Lana")
    call("alias", brain, "name", "hello")
    saved_brain = brain.read_bytes()
    created = call("index", brain, tokenizer)
    index_path = Path(created["index"])
    assert created["records"] == 2 and index_path.exists()
    index = json.loads(index_path.read_bytes())
    assert index["calibration"] is None and index["schema_version"] == 1
    assert call("chat", brain, tokenizer, "name", "--semantic")["resolution"] == "unsupported"
    assert call("chat", brain, tokenizer, "unknown", "--semantic")["resolution"] == "unsupported"
    development = root / "development.jsonl"
    heldout = root / "heldout.jsonl"
    development.write_text("\n".join(json.dumps(case) for case in (
        {"id": "development-name", "question": "name", "relevant_ids": ["fact:name"], "forbidden_ids": []},
        {"id": "development-paraphrase", "question": "name Lana", "relevant_ids": ["fact:name"], "forbidden_ids": []},
        {"id": "development-negative", "question": "unknown", "relevant_ids": [], "forbidden_ids": []},
    )) + "\n")
    heldout.write_text(json.dumps({"id": "heldout-name", "question": "name Lana name",
        "relevant_ids": ["fact:name"], "forbidden_ids": []}) + "\n")
    inactive_development = root / "inactive-development.jsonl"
    inactive_development.write_text(json.dumps({"id": "hard-negative", "question": "name",
        "relevant_ids": [], "forbidden_ids": ["fact:name"]}) + "\n")
    inactive = call("index", brain, tokenizer, "--calibrate", inactive_development, heldout)
    assert inactive["calibration"]["status"] == "inactive"
    assert call("chat", brain, tokenizer, "name", "--semantic")["answer"] is None
    calibrated = call("index", brain, tokenizer, "--calibrate", development, heldout)
    assert calibrated["calibration"]["status"] == "active"
    assert calibrated["calibration"]["heldout_false_exact"] == 0
    assert calibrated["calibration"]["paired"] == [{"id": "heldout-name",
        "baseline_correct": False, "semantic_exact": True, "semantic_correct": True}]
    answer = call("chat", brain, tokenizer, "name", "--semantic")
    assert answer["resolution"] == "exact" and answer["answer"] == "Lana"
    assert answer["source_refs"] == ["fact:name"]
    assert brain.read_bytes() == saved_brain
    assert call("chat", brain, tokenizer, "unknown", "--semantic")["answer"] is None
    assert "LANA_ERR_LIMIT" in call("chat", brain, tokenizer, "name " * 4097, "--semantic", ok=False)

    calibrated_index = index_path.read_bytes()
    bad_development = root / "bad-development.jsonl"
    bad_development.write_text(json.dumps({"id": "bad", "question": "name",
        "relevant_ids": ["missing"], "forbidden_ids": []}) + "\n")
    assert "LANA_ERR_SCHEMA" in call("index", brain, tokenizer, "--calibrate",
        bad_development, heldout, ok=False)
    assert index_path.read_bytes() == calibrated_index
    bad_tokenizer = root / "bad-tokenizer.json"
    malformed = json.loads(tokenizer.read_text())
    malformed["model"]["type"] = "BPE"
    bad_tokenizer.write_text(json.dumps(malformed))
    assert "LANA_ERR_SCHEMA" in call("index", brain, bad_tokenizer, ok=False)
    assert index_path.read_bytes() == calibrated_index
    changed_tokenizer = root / "changed.json"
    changed_tokenizer.write_bytes(tokenizer.read_bytes() + b"\n")
    assert "LANA_ERR_INVALID_STATE" in call("chat", brain, changed_tokenizer, "name", "--semantic", ok=False)
    corrupted = json.loads(calibrated_index)
    corrupted["records"][0]["vector_bits"][0] = "zzzzzzzz"
    index_path.write_text(json.dumps(corrupted, sort_keys=True, separators=(",", ":")) + "\n")
    assert "LANA_ERR_SCHEMA" in call("chat", brain, tokenizer, "name", "--semantic", ok=False)
    index_path.write_bytes(calibrated_index)
    call("remember", brain, "other", "Lana")
    assert "LANA_ERR_INVALID_STATE" in call("chat", brain, tokenizer, "name", "--semantic", ok=False)
    assert index_path.read_bytes() == calibrated_index
    brain.write_bytes(saved_brain)
    assert call("chat", brain, tokenizer, "name", "--semantic")["resolution"] == "exact"
    call("train", brain, 2, 0.1, 0, 1)
    assert "LANA_ERR_INVALID_STATE" in call("chat", brain, tokenizer, "name", "--semantic", ok=False)

    tied_brain = root / "tied.lbrn"
    tied_tokenizer = root / "tied-tokenizer.json"
    tied = json.loads(tokenizer.read_text())
    tied["model"]["vocab"].update({"color": 4, "Red": 5})
    tied_tokenizer.write_text(json.dumps(tied))
    call("new", tied_brain, 6, 2, 2, 7)
    call("remember", tied_brain, "name", "Lana")
    call("remember", tied_brain, "color", "Red")
    call("alias", tied_brain, "name", "hello")
    call("alias", tied_brain, "color", "hello")
    call("index", tied_brain, tied_tokenizer)
    tied_answer = call("chat", tied_brain, tied_tokenizer, "hello", "--semantic")
    assert tied_answer["resolution"] == "ambiguous" and tied_answer["answer"] is None
    value = root / "text-value.json"
    value.write_text('{"tag":"definite","value":{"tag":"string","value":"hello"}}')
    call("memory", "add", tied_brain, "greeting", "note", value)
    typed_index = call("index", tied_brain, tied_tokenizer)
    typed_records = json.loads(Path(typed_index["index"]).read_bytes())["records"]
    assert any(record["id"] == "root:greeting" and record["typed_status"] == "definite"
               for record in typed_records)

print("BRAIN_INDEX_PASS")
