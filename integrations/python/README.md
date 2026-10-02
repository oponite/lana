# Use Lana from Python, MCP, and notebooks

`lana-integrations` connects Python callers to the Lana executable.
It requires Python 3.11 or later. The base package has no mandatory dependencies.

Run the installation and shell examples from the Lana repository root:

```bash
python3 tools/build.py build
python3 -m venv .venv
.venv/bin/python -m pip install -e integrations/python
export LANA_EXECUTABLE="$PWD/target/lana/bin/lana"
```

## Select the executable

The adapters select an executable in this order:

1. The explicit Python executable argument or the command-line `--lana` argument.
2. The `LANA_EXECUTABLE` environment variable.
3. The `lana` command on `PATH`.

The adapter runs `lana version` before use.
It accepts Lana 3.x or 4.x reporting LABC v2–v6.
`BridgeRunner.version` contains the Lana version string.
`BridgeRunner.labc_version` contains the reported LABC version as an integer.
That reported version does not describe the full loader compatibility range.

## Python API

Run this example with `.venv/bin/python` from the repository root:

```python
from lana_integrations import Lana

with Lana() as lana:
    result = lana.run(
        "integrations/lana/echo_bridge.lana",
        {"message": "hello", "value": False},
        seed=17,
        memory_limit_mib=256,
        instruction_limit=50_000_000,
        workers=2,
        max_tasks=8,
        timeout_seconds=30,
    )
    assert result.status == "ok", result.error
    assert result.value == {"message": "hello", "value": False}
```

To use precompiled bytecode, first compile the bridge program:

```bash
"$LANA_EXECUTABLE" compile integrations/lana/echo_bridge.lana -o /tmp/lana-echo.labc
```

Then call the 4.x worker:

```python
from lana_integrations import Lana

with Lana() as lana:
    result = lana.run_labc("/tmp/lana-echo.labc", {"message": "hello"}, seed=17)
    assert result.status == "ok", result.error
    assert result.value == {"message": "hello"}
```

`run_labc()` requires Lana 4.x. Source and bytecode programs both use the
request/response file convention described in the next section.

Lana 4.1 also retains an explicitly registered Information graph in the
worker. The handle is valid only while the `Lana` object keeps that worker
open:

```python
with Lana() as lana:
    started = lana.start_live("examples/live.lana")
    assert started.ok, started.error
    handle = started.value["handle"]
    result = lana.observe_live(handle, "source", {"possibility": [2, 3]})
    assert result.ok, result.error
    current = lana.inspect_live(handle, "doubled")
    assert current.ok, current.error
    lana.pause_live(handle)
    lana.observe_live(handle, "source", 2)
    events = lana.resume_live(handle)
    assert events.ok, events.error
    lana.delete_live(handle)
```

Use `start_live_labc()` for precompiled bytecode. A tagged finite value can be
passed as evidence when ordinary JSON cannot express it. `LanaResult` carries
the same status and error shape as one-shot methods.

| Result status | Meaning |
| --- | --- |
| `ok` | The result is available in `value`. |
| `unavailable` | The executable is missing, incompatible, or does not provide the required worker. |
| `failed` | Execution, transport, or response parsing failed. Details are in `error`. |
| `unresolved` | The program returned an object with a truthy `__lana_unresolved__` marker. `value` preserves that object. |

The unresolved marker is an explicit program convention.

## JSON command-line bridge

```bash
printf '{"message":"hello"}' |
  .venv/bin/lana-bridge run integrations/lana/echo_bridge.lana
```

`--input PATH` reads a JSON file instead of standard input.
The command returns exit code 0 for success and 1 for a bridge error.
It prints a schema-1 JSON envelope:

```json
{"schema":1,"ok":true,"result":{"message":"hello"},"stdout":"","stderr":"","execution":{"elapsed_seconds":0.1,"lana_version":"4.1.0"}}
```

Execution metadata varies by backend. Errors have `ok: false`, a `phase`, and
an `error` object with `code` and `message`.

`BridgeRunner.run()` writes temporary request and response paths, then starts:

```text
lana run PROGRAM [VM OPTIONS] -- REQUEST_PATH RESPONSE_PATH
```

A compatible Lana program reads the first path and writes JSON to the second.
For example, the bundled echo program contains:

```lana
import "./bridge.lana" as bridge;

let request = bridge.read_request();
bridge.write_response(request);
```

`bridge.read_request()` validates JSON parsing. `bridge.write_response()`
serializes the result.

## Execution controls and failures

`Lana.run()`, `Lana.run_labc()`, and the `BridgeRunner` execution methods accept
these optional keyword arguments:

| Argument | Meaning |
| --- | --- |
| `seed` | Positive integer seed, up to `2**64 - 1`. |
| `memory_limit_mib` | Positive VM memory limit in MiB. Its byte value must fit the native integer range. |
| `instruction_limit` | Positive VM instruction limit, up to `2**64 - 1`. |
| `workers` | Positive worker count within the native integer range. |
| `max_tasks` | Positive task limit within the native integer range. |
| `timeout_seconds` | Finite positive timeout for this call. The constructor default is 30 seconds. |

## Evidence validation

`validate_evidence()` returns a validated copy without changing supplied values.
`BridgeRunner.run_evidence()` validates the record before running a program.

```python
from lana_integrations import BridgeRunner

record = {
    "schema": 1,
    "status": "resolved",
    "source": "sensor-example",
    "observed_at": 100,
    "effective_at": 100,
    "exactness": "exact",
    "revision": 1,
    "confidence": 0.9,
    "provenance_id": "sensor-example:1",
    "dependency_ids": [],
    "value": False,
}
response = BridgeRunner().run_evidence("integrations/lana/evidence_bridge.lana", record)
assert response["ok"], response
assert response["result"]["record"]["value"] is False
```

Supported statuses are `resolved`, `unknown`, `not_measured`,
`insufficient_evidence`, and `conflict`. Resolved records require a `value`.
Exactness is `exact`, `sample`, or `approximate`.
Timestamps, revision, and confidence must be finite and nonnegative.
Confidence must not exceed 1. Dependency identifiers must be nonempty and unique.
Optional reliability and calibration fields remain unchanged.

## MCP

Install the existing MCP extra:

```bash
.venv/bin/python -m pip install -e 'integrations/python[mcp]'
.venv/bin/lana-mcp --root "$PWD"
```

An MCP client launches this command and communicates over standard input/output.

By default, the server exposes only `lana_version`.
Add `--allow-run` to expose `lana_run`.

The run tool accepts a program path, JSON `input`, and the execution controls.

For clients that use an `mcpServers` configuration, replace the absolute paths:

```json
{
  "mcpServers": {
    "lana": {
      "command": "/absolute/path/to/lana/.venv/bin/lana-mcp",
      "args": [
        "--root", "/absolute/path/to/lana",
        "--lana", "/absolute/path/to/lana/target/lana/bin/lana",
        "--allow-run"
      ]
    }
  }
}
```

Remove `--allow-run` for version reporting only.
Repeat `--root` to permit program selection from multiple directories.
Roots reject parent-directory and symlink escapes during program selection.
They do not sandbox imports or file operations performed by the selected program.
The execution tool is explicitly marked as capable of side effects.

## Jupyter and IPython

Install the existing notebook extra:

```bash
.venv/bin/python -m pip install -e 'integrations/python[jupyter]'
```

Use that environment in IPython or your notebook kernel.
Set `LANA_EXECUTABLE` to the built executable before loading the extension.
From the repository root, run this Python cell:

```python
%load_ext lana_integrations.jupyter
```

Run an ordinary Lana cell:

```lana
%%lana --seed 17
print("hello notebook");
```

For JSON input, first create a request file in a Python cell:

```python
import json
from pathlib import Path

Path("input.json").write_text(json.dumps({"message": "hello"}), encoding="utf-8")
```

Then run a bridge cell:

```lana
%%lana --input input.json --instruction-limit 50000000
import "./integrations/lana/bridge.lana" as bridge;
bridge.write_response(bridge.read_request());
```
