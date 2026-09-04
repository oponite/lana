#!/usr/bin/env python3
"""JSON-RPC round-trip test for the Lana LSP server.

Spawns `lana lsp`, drives it through initialize / didOpen / hover / definition /
references / completion / rename / shutdown / exit, and asserts each response.

Usage: test_lsp.py <path-to-lana-binary>
"""

import json
import subprocess
import sys


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

    print("LSP_ROUNDTRIP_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
