"""Exercise the shared live runtime through the CLI, worker, and Python API."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
CLI = Path(sys.argv[1]).resolve()
SOURCE = ROOT / "examples/live.lana"
sys.path.insert(0, str(ROOT / "integrations/python/src"))
from lana_integrations import Lana  # noqa: E402


def check(result):
    assert result["ok"], result
    return result["result"]


with tempfile.TemporaryDirectory() as directory:
    labc = Path(directory) / "live.labc"
    subprocess.run([CLI, "compile", SOURCE, "-o", labc], check=True, capture_output=True, text=True)

    with subprocess.Popen([CLI, "bridge-worker"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          text=True, encoding="utf-8") as worker:
        def send(op, **fields):
            worker.stdin.write(json.dumps({"schema": 1, "op": op, **fields}) + "\n")
            worker.stdin.flush()
            return json.loads(worker.stdout.readline())

        started = send("start_live", path=str(SOURCE))
        first = check(started)["handle"]
        assert started["stdout"] == "LIVE_READY\n"
        second = check(send("start_live_labc", path=str(labc)))["handle"]
        assert first != second
        before = check(send("inspect_live", handle=first, name="doubled"))["inspection"]
        assert before["revision"] == 0 and before["derivation"]["operation"] == "binary"
        assert check(send("inspect_live", handle=first, name="doubled"))["inspection"]["revision"] == 0
        check(send("pause_live", handle=first))
        check(send("observe_live", handle=first, name="source", evidence=9))
        check(send("observe_live", handle=first, name="source", evidence={"possibility": [2, 3]}))
        events = check(send("resume_live", handle=first))["events"]
        assert events[0]["error"]["code"] == "LANA_ERR_INVALID_CONDITIONING", events
        assert events[1]["ok"], events
        doubled = check(send("inspect_live", handle=first, name="doubled"))["inspection"]
        assert doubled["revision"] == 1 and [row["value"] for row in doubled["support"]] == [4, 6]
        assert check(send("inspect_live", handle=first, name="other"))["inspection"]["revision"] == 0
        assert check(send("inspect_live", handle=second, name="source"))["inspection"]["revision"] == 0
        assert send("observe_live", handle=first, name="doubled", evidence=4)["error"]["code"] == "LANA_ERR_TYPE"
        assert check(send("delete_live", handle=first))["state"] == "DELETED"
        assert send("inspect_live", handle=first, name="source")["error"]["code"] == "LANA_ERR_NOT_FOUND"
        worker.stdin.close()

    commands = "\n".join([
        f"load {labc}", "inspect lanaprog_1 source", "observe lanaprog_1 source 2",
        "inspect lanaprog_1 doubled", "inspect lanaprog_2 source", "quit", "",
    ])
    session = subprocess.run([CLI, "live", SOURCE], input=commands, capture_output=True, text=True, check=True)
    replies = [json.loads(line) for line in session.stdout.splitlines()]
    assert check(replies[0])["handle"] == "lanaprog_1"
    assert check(replies[1])["handle"] == "lanaprog_2"
    assert check(replies[3])["revision"] == 1
    assert check(replies[4])["inspection"]["support"][0]["value"] == 4
    assert check(replies[5])["inspection"]["revision"] == 0

    with Lana(executable=CLI) as lana:
        started = lana.start_live(SOURCE)
        assert started.ok and started.stdout == "LIVE_READY\n", started
        handle = started.value["handle"]
        assert lana.observe_live(handle, "source", {"tag": "possibility", "dependency_id": "1",
            "support": [{"tag": "number", "bits": 2}, {"tag": "number", "bits": 3}]}).ok
        assert [row["value"] for row in lana.inspect_live(handle, "doubled").value["inspection"]["support"]] == [4, 6]
        assert lana.delete_live(handle).ok
        assert lana.inspect_live(handle, "source").error["code"] == "LANA_ERR_NOT_FOUND"

print("LIVE_HOST_PASS")
