"""Learned context selection: frozen state, held-out gate, replay, failed publication."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

lana = Path(sys.argv[1]).resolve()
repo = Path(__file__).resolve().parents[1]
environment = dict(os.environ, LANA_HF=str(repo / "tools/lana-hf/lana_hf.py"))
faults = "--fault-injection" in sys.argv[2:]

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    brain, tokenizer, model = (root / name for name in ("brain.lbrn", "tokenizer.json", "selector.json"))
    train, valid = root / "train.jsonl", root / "valid.jsonl"
    tokenizer.write_text(json.dumps({"version": "1.0", "truncation": None, "padding": None,
        "added_tokens": [], "normalizer": None, "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": None, "decoder": None,
        "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "train": 1, "answer": 2,
            "value": 3, "question": 4}, "unk_token": "[UNK]"}}))

    def call(*args, ok=True, stage=None):
        env = dict(environment)
        if stage:
            env["LANA_TEST_ATOMIC_STAGE"] = stage
        result = subprocess.run([lana, "brain", *map(str, args)], env=env,
                                capture_output=True, text=True, timeout=120)
        assert (result.returncode == 0) == ok, (args, result.stdout, result.stderr)
        return json.loads(result.stdout.splitlines()[-1]) if ok else result.stderr

    def pair(question_id, question, record_id, relevant):
        return json.dumps(dict(question_id=question_id, question=question, record_id=record_id, relevant=relevant)) + "\n"

    call("new", brain, 5, 1, 2, 7)
    call("remember", brain, "train", "value")
    call("remember", brain, "answer", "value")
    call("alias", brain, "answer", "question")
    # Duplicated irrelevant corpus is kept durably, omitted from working context.
    for i in range(8):
        call("remember", brain, f"noise-{i}", "value")
    call("index", brain, tokenizer)
    frozen = brain.read_bytes()
    index_path = Path(str(brain) + ".index.json")
    frozen_index = index_path.read_bytes()
    train.write_text(pair("train-id", "train", "fact:train", False))
    valid.write_text(pair("valid-id", "question", "fact:answer", True))
    fit = ("compress", "fit", brain, tokenizer, train, valid, model)
    report = call(*fit)
    assert report["status"] == "active", report
    assert report["retained"] == report["relevant"] == 1
    assert report["baseline_average"] == report["selected_average"] == 1
    assert report["parameter_sha256_before"] == report["parameter_sha256_after"]
    assert report["paired"][0]["baseline_outcome"] == report["paired"][0]["selected_outcome"]
    assert report["work_units"] > 0 and int(report["elapsed_ns"]) > 0
    saved = model.read_bytes()
    call(*fit)
    assert model.read_bytes() == saved
    answer = call("chat", brain, tokenizer, "question", "--semantic", "--selector", model)
    assert answer["selected_record_ids"] == ["fact:answer"]
    assert len(answer["selected_context"]) == 1 and answer["selected_context"][0]["value"] == "value"
    assert answer["resolution"] == "unsupported"  # Uncalibrated semantics cannot become exact.
    assert brain.read_bytes() == frozen and index_path.read_bytes() == frozen_index

    # Held-out loss saves a separate inactive report and retains the prior model.
    valid.write_text(pair("valid-id", "value", "fact:answer", True))
    failed_gate = call(*fit)
    assert failed_gate["status"] == "inactive" and failed_gate["retained"] == 0
    inactive = json.loads(Path(str(model) + ".inactive.json").read_bytes())
    assert not inactive["selector"]["active"] and inactive["report"]["status"] == "inactive"
    assert model.read_bytes() == saved
    invalid_model = root / "inactive-model.json"
    invalid_model.write_text(json.dumps(inactive["selector"], sort_keys=True, separators=(",", ":")) + "\n")
    assert "LANA_ERR_INVALID_STATE" in call("chat", brain, tokenizer, "question", "--semantic", "--selector", invalid_model, ok=False)
    valid.write_text(pair("valid-id", "question", "fact:answer", True))

    # Reject leakage, duplicate/contradictory labels, and malformed artifacts.
    original_train = train.read_text()
    for bad in (pair("valid-id", "train", "fact:train", False),
                pair("train-id", "train", "fact:answer", False),
                original_train + pair("train-id", "train", "fact:train", True),
                pair("train-id", "train", "fact:train", "false")):
        train.write_text(bad)
        assert "LANA_ERR_SCHEMA" in call(*fit, ok=False)
        assert model.read_bytes() == saved
    train.write_text(original_train)
    for change in (dict(weight_bits=["7f800000"]), dict(extra=True), dict(feature_width="01")):
        corrupted = dict(json.loads(saved), **change)
        invalid_model.write_text(json.dumps(corrupted, sort_keys=True, separators=(",", ":")) + "\n")
        assert "LANA_ERR_SCHEMA" in call("chat", brain, tokenizer, "question", "--semantic", "--selector", invalid_model, ok=False)
    assert "LANA_ERR_LIMIT" in call("chat", brain, tokenizer, "question " * 4097,
                                    "--semantic", "--selector", model, ok=False)
    assert "overlaps" in call("compress", "fit", brain, tokenizer, train, valid, brain, ok=False)
    assert brain.read_bytes() == frozen
    call("remember", brain, "later", "value")
    call("index", brain, tokenizer)
    assert "LANA_ERR_INVALID_STATE" in call("chat", brain, tokenizer, "question", "--semantic", "--selector", model, ok=False)
    brain.write_bytes(frozen)
    index_path.write_bytes(frozen_index)
    tokenizer.write_bytes(tokenizer.read_bytes() + b"\n")
    call("index", brain, tokenizer)
    assert "LANA_ERR_INVALID_STATE" in call("chat", brain, tokenizer, "question", "--semantic", "--selector", model, ok=False)
    tokenizer.write_bytes(tokenizer.read_bytes()[:-1])
    index_path.write_bytes(frozen_index)
    call("train", brain, 1, 0.1, 1)
    call("index", brain, tokenizer)
    assert "LANA_ERR_INVALID_STATE" in call("chat", brain, tokenizer, "question", "--semantic", "--selector", model, ok=False)
    brain.write_bytes(frozen)
    index_path.write_bytes(frozen_index)

    if faults:
        previous = dict(json.loads(saved), bias_bits="bf800000")
        previous = (json.dumps(previous, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for stage in ("before_file_sync", "before_rename", "after_rename"):
            for present in (False, True):
                model.unlink(missing_ok=True)
                if present:
                    model.write_bytes(previous)
                error = call(*fit, ok=False, stage=stage)
                assert "LANA_ERR_IO" in error
                assert ("uncertain" in error) == (stage == "after_rename")
                assert (model.read_bytes() if model.exists() else None) == (
                    saved if stage == "after_rename" else previous if present else None)
                assert not list(root.glob(model.name + ".lana-*.tmp"))
                assert brain.read_bytes() == frozen and index_path.read_bytes() == frozen_index

print("BRAIN_SELECTOR_PASS")
