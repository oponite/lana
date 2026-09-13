"""Exercise shipped CLI flags and debugger process boundaries."""
import pathlib
import subprocess
import sys
import tempfile


lana = str(pathlib.Path(sys.argv[1]).resolve())


def run(*args, input="", code=0):
    result = subprocess.run([lana, *map(str, args)], input=input, text=True,
                            capture_output=True, timeout=60)
    assert result.returncode == code, (args, result.stdout, result.stderr)
    return result.stdout + result.stderr


with tempfile.TemporaryDirectory(prefix="lana-cli-") as directory:
    root = pathlib.Path(directory)
    source = root / "debug.lana"
    source.write_text('let n = 1;\nprint(n);\n')
    bytecode = root / "debug.labc"
    run("compile", source, "-o", bytecode)
    for command, path in [("debug", source), ("run-bytecode", bytecode)]:
        flags = [] if command == "debug" else ["--debug"]
        output = run(command, path, *flags, input="s\nc\n")
        assert "BREAK line=1 instruction=0" in output, output
        assert "instruction=1" in output and "frames=1" in output, output
        for input in ["q\n", ""]:
            assert "LANA_ERR_CANCELLED" in run(command, path, *flags, input=input, code=1)
        for invalid in ["0", "-1", "abc", "4294967296"]:
            run(command, path, "--break", invalid, code=2)
        output = run(command, path, "--break", "2", input="c\n")
        assert "BREAK line=2" in output and "BREAK line=1" not in output, output
    trace = run("run-bytecode", bytecode, "--trace")
    assert "0000" in trace and "; line 1" in trace, trace
    run("run-bytecode", bytecode, "--break", code=2)
    tasks = pathlib.Path(__file__).resolve().parent / "conformance/differential/tasks/fork_join.lasm"
    run("asm", tasks, "-o", bytecode)
    assert "[task 1]" in run("run-bytecode", bytecode, "--trace")
    run("new", root / "project")
    (root / "project/src/main.lana").write_text('print(args()[0]);\n')
    assert "argument-value" in run("run", root / "project", "--", "argument-value")

print("RUST_CLI_PASS")
