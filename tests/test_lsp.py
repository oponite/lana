#!/usr/bin/env python3
"""JSON-RPC round-trip test for the Lana LSP server.

Spawns `lana lsp`, drives it through initialize / didOpen / hover / definition /
references / completion / rename / shutdown / exit, and asserts each response.

Usage: test_lsp.py <path-to-lana-binary>
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def frame(message):
    body = json.dumps(message, separators=(",", ":"))
    return f"Content-Length: {len(body)}\r\n\r\n{body}".encode()


def read_message(stream):
    header = stream.readline()
    if not header:
        return None
    length = 0
    while header.strip():
        if header.lower().startswith(b"content-length:"):
            length = int(header.split(b":", 1)[1].strip())
        header = stream.readline()
    body = stream.read(length)
    return json.loads(body)


SOURCE = (
    "fn add(a, b) {\n"
    "    let total = a + b;\n"
    "    return total;\n"
    "}\n"
    "\n"
    "let x = 1;\n"
    "let y = add(x, 2);\n"
    "print(y);\n"
)
URI = "file:///tmp/lana-lsp-test.lana"

# A source with a parse error on line 4 (1-based), column 9 (1-based).
BAD_SOURCE = (
    "fn add(a, b) {\n"
    "    return a + b;\n"
    "}\n"
    "let x = ;\n"
)
BAD_URI = "file:///tmp/lana-lsp-bad.lana"


def workspace_checks(cli):
    with tempfile.TemporaryDirectory(prefix="lana-lsp-workspace-") as directory:
        root = Path(directory).resolve()
        dep = root / "math lib.lana"
        main = root / "main.lana"
        scopes = root / "scopes.lana"
        dep.write_text("fn twice(value) { return value * 2; }\n")
        main.write_text('import "./math lib.lana" as lib;\nlet icon = "🌊"; let answer = lib.twice(2); print(answer);\n')
        scopes.write_text("fn first(value) { value = value + 1; return value; }\nfn second(value) { return value; }\n")
        proc = subprocess.Popen([cli, "lsp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        counter = 0

        def request(method, params):
            nonlocal counter
            counter += 1
            proc.stdin.write(frame({"jsonrpc":"2.0", "id":counter, "method":method, "params":params}))
            proc.stdin.flush()
            while True:
                reply = read_message(proc.stdout)
                assert reply is not None, proc.stderr.read().decode()
                if reply.get("id") == counter:
                    return reply

        def opened(file, text):
            proc.stdin.write(frame({"jsonrpc":"2.0", "method":"textDocument/didOpen", "params":{"textDocument":{"uri":file.as_uri(), "text":text}}}))
            proc.stdin.flush()

        def at(file, line, character, **extra):
            return dict(textDocument={"uri":file.as_uri()}, position={"line":line,"character":character}, **extra)

        try:
            init = request("initialize", {"rootUri":root.as_uri()})
            assert init["result"]["serverInfo"]["version"] == (Path(__file__).resolve().parents[1]/"VERSION").read_text().strip(), init
            opened(main, main.read_text())
            line = main.read_text().splitlines()[1]
            byte_index = line.index("twice")
            column = len(line[:byte_index].encode("utf-16-le"))//2
            definition = request("textDocument/definition", at(main,1,column))["result"]
            assert definition[0]["uri"] == dep.as_uri(), definition
            assert definition[0]["range"]["start"] == {"line":0,"character":3}, definition
            prepared = request("textDocument/prepareRename", at(main,1,column))["result"]
            assert prepared["range"]["start"]["character"] == column, prepared
            renamed = request("textDocument/rename", at(main,1,column,newName="double_value"))["result"]["changes"]
            assert set(renamed) == {main.as_uri(),dep.as_uri()}, renamed
            assert renamed[main.as_uri()][0]["range"]["start"]["character"] == column, renamed
            assert "error" in request("textDocument/rename", at(main,1,column,newName="bad-name"))
            assert "error" in request("textDocument/rename", at(main,1,column,newName="answer"))
            local = request("textDocument/rename", at(scopes,0,10,newName="input_value"))["result"]["changes"]
            assert len(local[scopes.as_uri()]) == 4, local
            assert all(edit["range"]["start"]["line"] == 0 for edit in local[scopes.as_uri()]), local
            alias = request("textDocument/rename", at(main,0,28,newName="maths"))["result"]["changes"]
            assert len(alias[main.as_uri()]) == 2 and len(alias) == 1, alias
            bindings = root / "bindings.lana"
            text = "let values = [item for item in [1, 2]];\n"
            opened(bindings, text)
            renamed_binding = request("textDocument/rename", at(bindings,0,text.index("for item")+4,newName="element"))["result"]["changes"][bindings.as_uri()]
            assert len(renamed_binding) == 2, renamed_binding
            assert sorted(e["range"]["start"]["character"] for e in renamed_binding) == [text.index("item"), text.index("for item")+4], renamed_binding
            objects = root / "objects.lana"
            opened(objects, "class Counter { public count: number = 1; }\nlet counter: Counter = new Counter();\nprint(counter.count);\n")
            renamed_type = request("textDocument/rename", at(objects,1,14,newName="Tally"))["result"]["changes"][objects.as_uri()]
            assert len(renamed_type) == 3, renamed_type
            renamed_field = request("textDocument/rename", at(objects,2,16,newName="total"))["result"]["changes"][objects.as_uri()]
            assert len(renamed_field) == 2, renamed_field
            # Both the imported declaration and use exist only in editor overlays.
            opened(dep, "fn triple(value) { return value * 3; }\n")
            opened(main, main.read_text().replace("twice", "triple"))
            definition = request("textDocument/definition", at(main,1,column))["result"]
            assert definition[0]["uri"] == dep.as_uri(), definition
            rename = request("textDocument/rename", at(main,1,column,newName="triple_value"))["result"]["changes"]
            assert set(rename) == {main.as_uri(),dep.as_uri()}, rename
            # A missing/invalid source aborts the whole edit, even if other modules resolve.
            broken = root / "broken.lana"
            opened(broken, "let bad = ;")
            failure = request("textDocument/rename", at(main,1,column,newName="another"))
            assert "error" in failure and "result" not in failure, failure
            proc.stdin.write(frame({"jsonrpc":"2.0","method":"textDocument/didClose","params":{"textDocument":{"uri":broken.as_uri()}}})); proc.stdin.flush()
            missing = root / "unsaved.lana"
            opened(missing, "fn fresh(value) { return value; }\n")
            opened(main, 'import "./unsaved.lana" as lib;\nlet answer = lib.fresh(1);\n')
            definition = request("textDocument/definition", at(main,1,18))["result"]
            assert definition[0]["uri"] == missing.as_uri(), definition
            private = root / ".lana" / "deps"
            private.mkdir(parents=True)
            dependency = private / "locked.lana"
            dependency.write_text("fn locked(value) { return value; }\n")
            opened(main, 'import "./.lana/deps/locked.lana" as lib;\nlet answer = lib.locked(1);\n')
            definition = request("textDocument/definition", at(main,1,18))["result"]
            assert definition[0]["uri"] == dependency.as_uri(), definition
            denied = request("textDocument/rename", at(main,1,18,newName="changed"))
            assert "error" in denied and "read-only" in denied["error"]["message"], denied
            request("shutdown", None)
            proc.stdin.write(frame({"jsonrpc":"2.0","method":"exit"})); proc.stdin.flush()
            proc.wait(timeout=10)
            assert proc.returncode == 0
        finally:
            if proc.poll() is None:
                proc.kill(); proc.wait()


def main():
    if len(sys.argv) != 2:
        print("usage: test_lsp.py <lana-binary>", file=sys.stderr)
        return 1

    proc = subprocess.Popen(
        [sys.argv[1], "lsp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "method": "textDocument/didOpen",
         "params": {"textDocument": {"uri": URI, "text": SOURCE}}},
        {"jsonrpc": "2.0", "id": 2, "method": "textDocument/hover",
         "params": {"textDocument": {"uri": URI}, "position": {"line": 6, "character": 9}}},
        {"jsonrpc": "2.0", "id": 3, "method": "textDocument/definition",
         "params": {"textDocument": {"uri": URI}, "position": {"line": 6, "character": 9}}},
        {"jsonrpc": "2.0", "id": 4, "method": "textDocument/references",
         "params": {"textDocument": {"uri": URI}, "position": {"line": 6, "character": 9}}},
        {"jsonrpc": "2.0", "id": 5, "method": "textDocument/completion",
         "params": {"textDocument": {"uri": URI}, "position": {"line": 7, "character": 0}}},
        {"jsonrpc": "2.0", "id": 6, "method": "textDocument/rename",
         "params": {"textDocument": {"uri": URI}, "position": {"line": 6, "character": 9},
                    "newName": "z"}},
        {"jsonrpc": "2.0", "method": "textDocument/didOpen",
         "params": {"textDocument": {"uri": BAD_URI, "text": BAD_SOURCE}}},
        {"jsonrpc": "2.0", "id": 7, "method": "shutdown", "params": None},
        {"jsonrpc": "2.0", "method": "exit"},
    ]

    for message in messages:
        proc.stdin.write(frame(message))
        proc.stdin.flush()

    responses = {}
    diagnostics = []
    while True:
        response = read_message(proc.stdout)
        if response is None:
            break
        if "id" in response:
            responses[response["id"]] = response
        elif response.get("method") == "textDocument/publishDiagnostics":
            diagnostics.append(response)

    proc.stdin.close()
    proc.wait(timeout=30)

    failures = []

    init = responses.get(1)
    if init is None or "lana-lsp" not in json.dumps(init):
        failures.append("initialize response missing serverInfo")

    hover = responses.get(2)
    if hover is None or "add" not in json.dumps(hover):
        failures.append(f"hover did not resolve `add`: {hover}")

    definition = responses.get(3)
    if definition is None or not definition.get("result"):
        failures.append(f"definition empty: {definition}")

    references = responses.get(4)
    if references is None or not references.get("result"):
        failures.append(f"references empty: {references}")

    completion = responses.get(5)
    if completion is None:
        failures.append("completion missing")
    else:
        items = completion.get("result", {}).get("items", [])
        labels = [item.get("label") for item in items]
        if "add" not in labels:
            failures.append(f"completion missing `add`: {labels}")

    rename = responses.get(6)
    if rename is None:
        failures.append("rename missing")
    else:
        changes = rename.get("result", {}).get("changes", {})
        edits = changes.get(URI, [])
        if len(edits) != 2:
            failures.append(f"rename expected 2 edits, got {len(edits)}: {rename}")

    # Accurate-span check for an unsaved buffer: the parse error on line 4
    # (1-based) column 9 (1-based) must surface as 0-based line 3, character 8.
    bad_diag = None
    for notification in diagnostics:
        if notification.get("params", {}).get("uri") == BAD_URI:
            bad_diag = notification
            break
    if bad_diag is None:
        failures.append("no publishDiagnostics for bad source")
    else:
        items = bad_diag.get("params", {}).get("diagnostics", [])
        if not items:
            failures.append(f"bad source produced no diagnostics: {bad_diag}")
        else:
            start = items[0].get("range", {}).get("start", {})
            if start.get("line") != 3 or start.get("character") != 8:
                failures.append(
                    f"bad source span wrong: expected line 3 char 8, got {start}"
                )

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1

    workspace_checks(sys.argv[1])
    print("LSP_ROUNDTRIP_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
