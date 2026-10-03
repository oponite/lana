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

    with subprocess.Popen([CLI, "live", SOURCE], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          text=True, encoding="utf-8", bufsize=1) as session:
        first = check(json.loads(session.stdout.readline()))["handle"]

        def command(line):
            session.stdin.write(line + "\n")
            session.stdin.flush()
            return json.loads(session.stdout.readline())

        second = check(command(f"load {labc}"))["handle"]
        assert first != second
        assert check(command(f"inspect {first} source"))["inspection"]["revision"] == 0
        assert check(command(f"observe {first} source 2"))["revision"] == 1
        assert check(command(f"inspect {first} doubled"))["inspection"]["support"][0]["value"] == 4
        assert check(command(f"inspect {second} source"))["inspection"]["revision"] == 0
        session.stdin.write("quit\n")
        session.stdin.flush()

    with Lana(executable=CLI) as lana:
        started = lana.start_live(SOURCE)
        assert started.ok and started.stdout == "LIVE_READY\n", started
        handle = started.value["handle"]
        assert lana.observe_live(handle, "source", {"tag": "possibility", "dependency_id": "1",
            "support": [{"tag": "number", "bits": 2}, {"tag": "number", "bits": 3}]}).ok
        assert [row["value"] for row in lana.inspect_live(handle, "doubled").value["inspection"]["support"]] == [4, 6]
        assert lana.delete_live(handle).ok
        assert lana.inspect_live(handle, "source").error["code"] == "LANA_ERR_NOT_FOUND"

    with Lana(executable=CLI) as first_worker, Lana(executable=CLI) as second_worker:
        first = first_worker.start_live(SOURCE).value["handle"]
        second = second_worker.start_live(SOURCE).value["handle"]
        assert first != second
        assert second_worker.inspect_live(first, "source").error["code"] == "LANA_ERR_NOT_FOUND"

    array_source = Path(directory) / "array.lana"
    array_source.write_text('live_register("root", information([1, 2]));\n')
    map_source = Path(directory) / "map.lana"
    map_source.write_text('live_register("root", information({tag: "foo", possibility: 3}));\n')
    subset_key_source = Path(directory) / "subset_key.lana"
    subset_key_source.write_text('live_register("root", information({possibility: [2, 3]}));\n')
    tagged_shape_source = Path(directory) / "tagged_shape.lana"
    tagged_shape_source.write_text('live_register("root", information({tag: "number", bits: 2}));\n')
    with Lana(executable=CLI) as lana:
        array = lana.start_live(array_source)
        assert array.ok, array
        assert lana.observe_live(array.value["handle"], "root", [1, 2]).ok
        tagged_keys = lana.start_live(map_source)
        assert tagged_keys.ok, tagged_keys
        assert lana.observe_live(tagged_keys.value["handle"], "root", {"tag": "foo", "possibility": 3}).ok
        subset_key = lana.start_live(subset_key_source)
        assert subset_key.ok, subset_key
        assert lana.observe_live(subset_key.value["handle"], "root", {"possibility": [2, 3]}).ok
        tagged_shape = lana.start_live(tagged_shape_source)
        assert tagged_shape.ok, tagged_shape
        assert lana.observe_live(tagged_shape.value["handle"], "root", {"tag": "number", "bits": 2}).ok

print("LIVE_HOST_PASS")
