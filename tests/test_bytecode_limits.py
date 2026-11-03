"""Oversized bytecode is bounded at every file ingress; workers remain usable."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

cli = str(Path(sys.argv[1]).resolve())
with tempfile.TemporaryDirectory(prefix="lana-bytecode-limit-") as directory:
    work = Path(directory)
    oversized = work / "oversized.labc"
    with oversized.open("wb") as output:
        output.truncate(128 * 1024 * 1024)
    for command in ("run", "run-bytecode", "verify", "dis", "inspect"):
        result = subprocess.run([cli, command, str(oversized)], capture_output=True, text=True, timeout=30)
        assert result.returncode > 0 and "LANA_ERR_LIMIT" in result.stderr, (command, result)
    requests = [{"schema":1,"op":operation,"path":str(oversized),"input":{}}
                for operation in ("run_labc", "start_live_labc")]
    requests.append({"schema":1,"op":"not-an-operation"})
    result = subprocess.run([cli, "bridge-worker"], input="".join(json.dumps(r) + "\n" for r in requests),
                            capture_output=True, text=True, timeout=30)
    responses = [json.loads(line) for line in result.stdout.splitlines()]
    assert result.returncode == 0 and len(responses) == 3, result
    assert all(response["error"]["code"] == "LANA_ERR_LIMIT" for response in responses[:2]), responses
    assert responses[2]["error"]["code"] == "LANA_ERR_UNSUPPORTED_OPERATION", responses
print("BYTECODE_LIMITS_PASS")
