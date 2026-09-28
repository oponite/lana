"""Exercise the public brain CLI in isolated directories and separate processes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
lana = str(Path(sys.argv[1]).resolve())
env = dict(os.environ, LANA_HF=str(root / "tools/lana-hf/lana_hf.py"))
with tempfile.TemporaryDirectory() as directory:
    work = Path(directory)

    def run(*args, ok=True):
        result = subprocess.run([lana, *args], cwd=work, env=env, capture_output=True, text=True, timeout=60)
        assert (result.returncode == 0) == ok, result.stdout + result.stderr
        if ok:
            return json.loads(result.stdout.splitlines()[-1])
        return result

    run("brain", "new", "brain.lbrn", "3", "2", "2", "7")
    brain = work / "brain.lbrn"
    before = brain.read_bytes()
    run("brain", "train", "brain.lbrn", "2", "nan", "0", "1", ok=False)
    assert brain.read_bytes() == before
    run("brain", "train", "brain.lbrn", "2", "0.1", "0", "1")
    logits = run("brain", "evaluate", "brain.lbrn", "0", "1")["logits"]
    blocked = work / "blocked.lbrn"
    blocked.mkdir()
    failed = run("brain", "save", "brain.lbrn", "blocked.lbrn", ok=False)
    assert json.loads(failed.stderr.splitlines()[-1])["error"] == "LANA_ERR_IO", failed.stderr
    assert blocked.is_dir()
    tokenizer = work / "tokenizer.json"
    config = json.loads((root / "examples/brain/tokenizer.json").read_text())
    unusual = 'a"b\\c\n猫'
    config["model"]["vocab"] = {"[UNK]": 0, "hello": 1, unusual: 2}
    tokenizer.write_text(json.dumps(config))
    run("brain", "save", "brain.lbrn", "package", str(tokenizer))
    run("brain", "load", "package", "restored.lbrn")
    assert run("brain", "evaluate", "restored.lbrn", "0", "1")["logits"] == logits
    answer = run("brain", "chat", "restored.lbrn", str(tokenizer), "hello")
    assert answer["response"] == unusual, answer
    answer = run("brain", "chat", "restored.lbrn", str(tokenizer), "hello")
    assert answer["memory_revision"] == 2
    assert run("brain", "inspect", "restored.lbrn")["memory_revision"] == 2
    before = (work / "restored.lbrn").read_bytes()
    config["normalizer"] = {"type": "Lowercase"}
    tokenizer.write_text(json.dumps(config))
    run("brain", "chat", "restored.lbrn", str(tokenizer), "hello", ok=False)
    assert (work / "restored.lbrn").read_bytes() == before
    assert run("brain", "remember", "restored.lbrn", "dog_name", "Max")["fact_revision"] == 1
    assert run("brain", "recall", "restored.lbrn", "dog_name")["value"] == "Max"
    remembered = (work / "restored.lbrn").read_bytes()
    assert not run("brain", "remember", "restored.lbrn", "dog_name", "Max")["changed"]
    run("brain", "remember", "restored.lbrn", "dog_name", "Sam", ok=False)
    assert (work / "restored.lbrn").read_bytes() == remembered
    answer = run("brain", "chat", "restored.lbrn", str(tokenizer), "What is my dog's name?", "--fact", "dog_name")
    assert (answer["response"], answer["resolution"], answer["assumptions"], answer["unsupported"]) == ("Max", "exact", [], False)
    assert run("brain", "recall", "restored.lbrn", "dog_name")["value"] == "Max"
    assert run("brain", "inspect", "restored.lbrn")["fact_revision"] == 1
    config["normalizer"] = None
    tokenizer.write_text(json.dumps(config))
    assert run("brain", "chat", "restored.lbrn", str(tokenizer), "hello")["memory_revision"] == 4
    before = (work / "restored.lbrn").read_bytes()
    run("brain", "chat", "restored.lbrn", str(tokenizer), "Unknown?", "--fact", "unknown", ok=False)
    assert (work / "restored.lbrn").read_bytes() == before
    assert not list(work.glob("*.tmp"))
print("BRAIN_WORKFLOW_PASS")
