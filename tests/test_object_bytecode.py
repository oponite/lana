"""CLI verification and execution of v6 values, classes, interfaces, and failure boundaries."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

cli = str(Path(sys.argv[1]).resolve())
descriptor = {"schema_version": 1, "kind": "value", "qualified_name": "file/main.lana/Reading",
              "fields": [{"name": "state", "type": "STATE", "visibility": "public",
                          "mutable": False, "default_function": None}],
              "methods": [], "implements": []}
with tempfile.TemporaryDirectory(prefix="lana-object-abi-") as directory:
    source, binary = Path(directory) / "object.lasm", Path(directory) / "object.labc"
    def write(version=6):
        text = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode().hex()
        source.write_text(f".version {version}\n.function main 0 3\nSTATE_NEW R0 0.5 0.3 0.4\nVALUE_NEW R1 R0 {text} 1\nOO_GET R2 R1 {text} 0\nCOMPARE R2 == R0 R2\nPRINT R2\nHALT\n")
    def run(*arguments):
        return subprocess.run([cli, *map(str, arguments)], capture_output=True, text=True, timeout=30)
    write()
    result = run("asm", source, "-o", binary)
    assert result.returncode == 0, result.stderr
    assert run("verify", binary).returncode == 0
    assert "VALUE_NEW" in run("dis", binary).stdout
    result = run("run", binary)
    assert result.returncode == 0 and "true" in result.stdout, result
    source.write_text(source.read_text().replace("STATE_NEW R0 0.5 0.3 0.4", "LOAD_CONST R0 42"))
    assert run("asm", source, "-o", binary).returncode == 0
    result = run("run", binary)
    assert result.returncode and "LANA_ERR_TYPE" in result.stderr, result
    assert not result.stdout, result.stdout
    blocked = dict(descriptor, kind="interface", fields=[])
    text = json.dumps(blocked, sort_keys=True, separators=(",", ":")).encode().hex()
    source.write_text(f".version 6\n.function main 0 1\nLOAD_CONST R0 42\nPRINT R0\nLOAD_STRING R0 {text}\nHALT\n")
    assert run("asm", source, "-o", binary).returncode == 0
    result = run("run", binary)
    assert result.returncode == 0, result
    assert "42" in result.stdout, result.stdout
    write()
    other = json.dumps(dict(descriptor, qualified_name="file/main.lana/Other"), sort_keys=True, separators=(",", ":")).encode().hex()
    original = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode().hex()
    source.write_text(source.read_text().replace(f"OO_GET R2 R1 {original}", f"OO_GET R2 R1 {other}"))
    assert run("asm", source, "-o", binary).returncode == 0
    result = run("run", binary)
    assert result.returncode and "LANA_ERR_TYPE" in result.stderr, result
    assert not result.stdout, result.stdout
    for version in range(1, 6):
        write(version)
        assert run("asm", source, "-o", binary).returncode
    source.write_text(".function main 0 1\nLOAD_STRING R0 ff\n.version 6\nHALT\n")
    assert run("asm", source, "-o", binary).returncode
    numeric = dict(descriptor, fields=[dict(descriptor["fields"][0], name="tensor", type="Tensor"),
                                      dict(descriptor["fields"][0], name="shape", type="Shape")])
    text = json.dumps(numeric, sort_keys=True, separators=(",", ":")).encode().hex()
    program = f".version 6\n.function main 0 6\nLOAD_CONST R2 2\nARRAY_NEW R1 R2 1\nHOST_CALL tensor_zeros R1 1 R0\nVALUE_NEW R3 R0 {text} 2\nOO_GET R4 R3 {text} 0\nHOST_CALL tensor_shape R4 1 R5\nPRINT R5\nOO_GET R4 R3 {text} 1\nPRINT R4\nHALT\n"
    source.write_text(program)
    assert run("asm", source, "-o", binary).returncode == 0
    result = run("run", binary)
    assert result.returncode == 0 and result.stdout.count("[2]") == 2, result
    source.write_text(program.replace("VALUE_NEW", "LOAD_CONST R2 -1\nARRAY_NEW R1 R2 1\nVALUE_NEW"))
    assert run("asm", source, "-o", binary).returncode == 0
    result = run("run", binary)
    assert result.returncode and "LANA_ERR_INVALID_PARAMETERS" in result.stderr, result
    assert not result.stdout, result.stdout
    def method(name, static, parameters, result, function):
        return {"name": name, "visibility": "public", "static": static, "parameter_types": parameters,
                "result_type": result, "effect_mask": 0, "function_index": function, "is_init": False}
    owned = dict(descriptor, fields=[dict(descriptor["fields"][0], visibility="private")],
                 methods=[method("create", True, ["STATE"], "Self", 1), method("read", False, [], "STATE", 2)])
    text = json.dumps(owned, sort_keys=True, separators=(",", ":")).encode().hex()
    program = f".version 6\n.function main 0 6\nSTATE_NEW R0 0.5 0.3 0.4\nARRAY_NEW R1 R0 1\nOO_STATIC_CALL R2 R1 {text} 0\nARRAY_NEW R3 R2 1\nOO_CALL R4 R3 {text} 1\nCOMPARE R0 == R4 R5\nPRINT R5\nHALT\n.function create 1 2\nVALUE_NEW R1 R0 {text} 1\nRETURN R1\n.function read 1 2\nOO_GET R1 R0 {text} 0\nRETURN R1\n"
    source.write_text(program)
    assert run("asm", source, "-o", binary).returncode == 0
    result = run("run", binary)
    assert result.returncode == 0 and "true" in result.stdout, result
    source.write_text(program.replace(".function create 1 2\n", ".function create 1 2\nPRINT R0\n")
                            .replace("STATE_NEW R0", "LOAD_CONST R5 42\nPRINT R5\nSTATE_NEW R0"))
    assert run("asm", source, "-o", binary).returncode == 0
    result = run("run", binary)
    assert result.returncode and "LANA_ERR_UNSUPPORTED_OPERATION" in result.stderr, result
    assert not result.stdout, result.stdout
    source.write_text(program.replace(f"OO_GET R1 R0 {text} 0", "LOAD_CONST R1 true"))
    assert run("asm", source, "-o", binary).returncode == 0
    result = run("run", binary)
    assert result.returncode and "LANA_ERR_TYPE" in result.stderr, result
    assert not result.stdout, result.stdout
    sensor = dict(descriptor, kind="class", fields=[dict(descriptor["fields"][0], mutable=True),
                  dict(descriptor["fields"][0], name="link", type="Dynamic", mutable=True)],
                  methods=[dict(method("init", False, ["STATE"], None, 1), is_init=True, effect_mask=8)])
    text = json.dumps(sensor, sort_keys=True, separators=(",", ":")).encode().hex()
    program = f".version 6\n.function main 0 8\nSTATE_NEW R0 0.5 0.3 0.4\nOBJECT_NEW R1 R0 {text} 1\nOO_GET R2 R1 {text} 1\nCOMPARE R1 == R2 R3\nPRINT R3\nFORK echo R1 1 R4\nJOIN R4 R5\nCOMPARE R1 != R5 R6\nPRINT R6\nHALT\n.function init 2 3\nOO_SET R0 R1 {text} 0\nOO_SET R0 R0 {text} 1\nRETURN R2\n.function echo 1 1\nRETURN R0\n"
    source.write_text(program)
    assert run("asm", source, "-o", binary).returncode == 0
    result = run("run", binary)
    assert result.returncode == 0 and result.stdout.count("true") == 2, result
    source.write_text(program.replace(f"OO_SET R0 R0 {text} 1", "NOP"))
    assert run("asm", source, "-o", binary).returncode == 0
    result = run("run", binary)
    assert result.returncode and "LANA_ERR_TYPE" in result.stderr, result
    assert not result.stdout, result.stdout
    descriptor["fields"][0]["visibility"] = "private"
    write()
    result = run("asm", source, "-o", binary)
    assert result.returncode and "private" in result.stderr, result
print("OBJECT_BYTECODE_PASS")
