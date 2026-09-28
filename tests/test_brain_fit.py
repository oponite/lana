"""Multi-epoch Brain fit keeps only the best valid checkpoint."""
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import tempfile
import time

lana = str(Path(sys.argv[1]).resolve())

def call(*args, ok=True):
    result = subprocess.run([lana, "brain", *map(str, args)], capture_output=True, text=True, timeout=120)
    assert (result.returncode == 0) == ok, result.stdout + result.stderr
    return result

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    train, valid = root / "train.jsonl", root / "valid.jsonl"
    train.write_text('{"tokens":[0,1],"target":2}\n{"tokens":[1],"target":2}\n')
    valid.write_text('{"tokens":[0,1],"target":1}\n')
    models = [root / "a.lbrn", root / "b.lbrn"]
    reports = []
    for model in models:
        call("new", model, 3, 2, 2, 7)
        result = call("fit", model, train, valid, "--learning-rate", 0.1,
                      "--max-epochs", 4, "--patience", 2, "--warmup-steps", 2,
                      "--weight-decay", 0.01)
        report = json.loads(result.stdout)
        assert report["schema_version"] == 1
        assert set(report) == {"schema_version", "seed", "learning_rate", "weight_decay",
                               "warmup_steps", "max_epochs", "patience", "epochs",
                               "best_epoch", "stop_reason", "saved_brain_format", "saved_brain_version"}
        assert report["seed"] == "7"
        assert report["saved_brain_format"] == "LBRN1"
        assert abs(report["epochs"][0]["first_step_rate"] - 0.05) < 1e-6
        assert abs(report["epochs"][0]["last_step_rate"] - 0.1) < 1e-6
        assert report["best_epoch"] == min(report["epochs"], key=lambda row: row["validation_mean_loss"])["epoch"]
        inspected = json.loads(call("inspect", model).stdout.splitlines()[-1])
        assert inspected["training_steps"] == 2 * report["best_epoch"]
        assert inspected["version"] == int(report["saved_brain_version"])
        reports.append(report)
    assert models[0].read_bytes() == models[1].read_bytes()
    assert reports[0] == reports[1]
    before = models[0].read_bytes()
    assert not call("fit", models[0], train, train, "--learning-rate", 0.1,
                    "--max-epochs", 2, "--patience", 1, ok=False).stdout
    assert models[0].read_bytes() == before
    alias = root / "train-alias.jsonl"
    alias.symlink_to(train)
    assert not call("fit", models[0], train, alias, "--learning-rate", 0.1,
                    "--max-epochs", 2, "--patience", 1, ok=False).stdout
    assert models[0].read_bytes() == before
    hard_link = root / "train-hard-link.jsonl"
    os.link(train, hard_link)
    assert not call("fit", models[0], train, hard_link, "--learning-rate", 0.1,
                    "--max-epochs", 2, "--patience", 1, ok=False).stdout
    assert models[0].read_bytes() == before
    valid.write_text('{"tokens":[0],"tokens":[1],"target":2}\n')
    assert not call("fit", models[0], train, valid, "--learning-rate", 0.1,
                    "--max-epochs", 2, "--patience", 1, ok=False).stdout
    assert models[0].read_bytes() == before
    valid.write_text("")
    call("fit", models[0], train, valid, "--learning-rate", 0.1,
         "--max-epochs", 2, "--patience", 1, ok=False)
    assert models[0].read_bytes() == before
    valid.write_text('{"tokens":[0,1],"target":1}\n')
    for options in (
        ("--learning-rate", "nan", "--max-epochs", 2, "--patience", 1),
        ("--learning-rate", 0.1, "--max-epochs", 1001, "--patience", 1),
        ("--learning-rate", 0.1, "--max-epochs", 2, "--patience", 3),
        ("--learning-rate", 0.1, "--max-epochs", 2, "--patience", 1, "--warmup-steps", 5),
        ("--learning-rate", 0.1, "--max-epochs", 2, "--patience", 1, "--weight-decay", -0.1),
    ):
        assert not call("fit", models[0], train, valid, *options, ok=False).stdout
        assert models[0].read_bytes() == before
    for bad in ('{"tokens":[0],"target":3}\n',
                '{"tokens":[0],"target":2,"extra":1}\n',
                '{"tokens":[0],"tokens":[1],"target":2}\n',
                '\n'):
        train.write_text(bad)
        assert not call("fit", models[0], train, valid, "--learning-rate", 0.1,
                        "--max-epochs", 2, "--patience", 1, ok=False).stdout
        assert models[0].read_bytes() == before
    with train.open("wb") as oversized:
        oversized.truncate(64 * 1024 * 1024 + 1)
    assert not call("fit", models[0], train, valid, "--learning-rate", 0.1,
                    "--max-epochs", 2, "--patience", 1, ok=False).stdout
    assert models[0].read_bytes() == before
    train.write_text('{"tokens":[0,1],"target":2}\n{"tokens":[1],"target":2}\n')

    if os.name == "posix":
        long_train = root / "long-train.jsonl"
        long_train.write_text('{"tokens":[0,1],"target":2}\n' * 5000)
        for signum in (signal.SIGINT, signal.SIGTERM):
            process = subprocess.Popen(
                [lana, "brain", "fit", str(models[0]), str(long_train), str(valid),
                 "--learning-rate", "0.1", "--max-epochs", "1000", "--patience", "1000"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            time.sleep(0.25)
            assert process.poll() is None, "fit finished before interruption"
            process.send_signal(signum)
            stdout, stderr = process.communicate(timeout=30)
            assert process.returncode == 1 and not stdout, (process.returncode, stdout, stderr)
            assert json.loads(stderr.splitlines()[-1])["error"] == "LANA_ERR_CANCELLED", stderr
            assert models[0].read_bytes() == before
            assert json.loads(call("inspect", models[0]).stdout.splitlines()[-1])["training_steps"] == 2 * reports[0]["best_epoch"]

    architecture = root / "architecture.json"
    architecture.write_text('{"hidden":[{"width":4,"activation":"relu"},{"width":2,"activation":"gelu"}]}')
    layered = root / "layered.lbrn"
    architecture.write_text('{"hidden":[{"width":0,"activation":"relu"}]}')
    call("new", layered, 3, 2, "--architecture", architecture, ok=False)
    assert not layered.exists()
    architecture.write_text('{"hidden":[{"width":4,"activation":"relu"},{"width":2,"activation":"gelu"}]}')
    call("new", layered, 3, 2, "--architecture", architecture, "--seed", 7)
    assert layered.read_bytes().startswith(b"LBRN2")
    bridge = Path(__file__).resolve().parents[1] / "tools/lana-hf/lana_hf.py"
    tokenizer = bridge.parent / "tests/wordlevel.json"
    package = root / "layered-package"
    restored = root / "restored.lbrn"
    subprocess.run([sys.executable, str(bridge), "package", str(layered), str(package), str(tokenizer)],
                   check=True, capture_output=True, text=True)
    subprocess.run([sys.executable, str(bridge), "unpackage", str(package), str(restored)],
                   check=True, capture_output=True, text=True)
    assert restored.read_bytes() == layered.read_bytes()
    assert call("evaluate", restored, 0, 1).stdout == call("evaluate", layered, 0, 1).stdout
    logits = json.loads(call("evaluate", layered, 0, 1).stdout.splitlines()[-1])["logits"]
    copy = root / "layered-copy.lbrn"
    call("save", layered, copy)
    assert copy.read_bytes() == layered.read_bytes()
    assert json.loads(call("evaluate", copy, 0, 1).stdout.splitlines()[-1])["logits"] == logits
    call("train", layered, 2, 0.1, 0, 1)
    trained_logits = json.loads(call("evaluate", layered, 0, 1).stdout.splitlines()[-1])["logits"]
    assert trained_logits != logits
    valid.write_text('{"tokens":[0,1],"target":2}\n')
    report = json.loads(call("fit", layered, train, valid, "--learning-rate", 0.1,
                             "--max-epochs", 2, "--patience", 1).stdout)
    assert int(report["saved_brain_version"]) > 2
    assert report["saved_brain_format"] == "LBRN2"
    assert layered.read_bytes().startswith(b"LBRN2")
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        before = layered.read_bytes()
        root.chmod(0o555)
        try:
            failed = call("fit", layered, train, valid, "--learning-rate", 0.1,
                          "--max-epochs", 2, "--patience", 1, ok=False)
            assert not failed.stdout
            assert layered.read_bytes() == before
        finally:
            root.chmod(0o755)
print("BRAIN_FIT_PASS")
