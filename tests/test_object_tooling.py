"""Object member LSP ranges, debugger frames, formatter, and export boundaries."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from test_lsp import frame, read_message

cli = str(Path(sys.argv[1]).resolve())
source = '''class Counter {
    public mutable count: number = 2;
    public fn init(self, count: number) {
        self.count = count;
    }
    public fn read(self) -> number {
        return self.count;
    }
}
let counter = new Counter(7);
print(counter.read());
'''
with tempfile.TemporaryDirectory(prefix="lana-object-tooling-") as directory:
    root = Path(directory)
    path = root / "main.lana"
    path.write_text(source)
    for line, name in [(2, "Counter.<default:count>"), (4, "Counter.init"), (7, "Counter.read")]:
        result = subprocess.run([cli, "debug", str(path), "--break", str(line)], input="c\n", text=True, capture_output=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert f"function={name}" in result.stdout and "7\n" in result.stdout, result.stdout
    proc = subprocess.Popen([cli, "lsp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    uri = path.as_uri()
    messages = [
        {"id":1,"method":"initialize","params":{}},
        {"method":"textDocument/didOpen","params":{"textDocument":{"uri":uri,"text":source}}},
        {"id":2,"method":"textDocument/definition","params":{"textDocument":{"uri":uri},"position":{"line":10,"character":15}}},
        {"id":3,"method":"textDocument/rename","params":{"textDocument":{"uri":uri},"position":{"line":6,"character":22},"newName":"total"}},
        {"id":4,"method":"shutdown","params":None}, {"method":"exit"}]
    for message in messages:
        proc.stdin.write(frame(dict(jsonrpc="2.0", **message)))
    proc.stdin.flush()
    responses = {}
    while (message := read_message(proc.stdout)) is not None:
        if "id" in message: responses[message["id"]] = message
        if message.get("method") == "textDocument/publishDiagnostics":
            assert message["params"]["diagnostics"] == [], message
    assert proc.wait(timeout=30) == 0, proc.stderr.read()
    assert responses[2]["result"][0]["range"]["start"] == {"line":5,"character":14}, responses
    edits = responses[3]["result"]["changes"][uri]
    assert len(edits) == 3, edits
    for edit in edits:
        span = edit["range"]
        assert span["end"]["character"] - span["start"]["character"] == 5, edits
    # Formatter is syntax-neutral; check it preserves object behavior and is idempotent.
    (root / "src").mkdir()
    (root / "src" / "main.lana").write_text(source.replace(";\n", ";  \n"))
    for args in [("fmt",), ("fmt", "--check"), ("run", str(root / "src" / "main.lana"))]:
        result = subprocess.run([cli, *args], cwd=root, text=True, capture_output=True, timeout=30)
        assert result.returncode == 0, result.stderr
    for declaration, expression in [
        ("class C { public n: number = 1; }", "new C()"),
        ("value V { private n: number; public static fn make() -> Self { return Self(1); } }", "V.make()"),
        ("value Hidden { private n: number; public static fn make() -> Self { return Self(1); } } value V { public nested: Hidden; }", "V(Hidden.make())"),
    ]:
        path.write_text(declaration + " print(json_stringify(" + expression + "));\n")
        result = subprocess.run([cli, "run", str(path)], text=True, capture_output=True, timeout=30)
        assert result.returncode and "LANA_ERR_UNSUPPORTED_OPERATION" in result.stderr, result
        assert not result.stdout, result.stdout
print("OBJECT_TOOLING_PASS")
