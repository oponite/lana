#!/usr/bin/env python3
"""Run identical bytecode in isolated C/Rust processes with bounded execution."""
import argparse
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import re
import shlex
import socketserver
import subprocess
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).resolve().parent
GROUPS = ("core", "hostcalls", "tasks", "durable", "ffi", "net")
MEMORY = re.compile(rb"^  resource: memory limit ([0-9]+), observed ([0-9]+) bytes$", re.M)


def diagnostic(stderr):
    def memory(match):
        limit, observed = map(int, match.groups())
        if observed > limit:
            raise AssertionError(f"allocation exceeded its limit: {observed} > {limit}")
        # Different object layouts and reclamation strategies need not report
        # equal live bytes. Preserve the resource kind and exact configured limit.
        return b"  resource: memory limit " + match[1] + b", observed <implementation-specific> bytes"
    return MEMORY.sub(memory, stderr)


def statistics(stderr):
    lines = [line for line in stderr.splitlines() if line.startswith(b"LANAVM_STATS ")]
    if len(lines) != 1:
        raise AssertionError(f"expected exactly one stats record, found {len(lines)}")
    result = json.loads(lines[0].removeprefix(b"LANAVM_STATS "))
    assert set(result) == {"instructions", "state_transitions", "allocations",
                           "allocated_bytes", "elapsed_ns", "opcodes"}, result
    for key, value in result.items():
        if key == "opcodes":
            assert isinstance(value, dict) and value, result
            assert all(isinstance(k, str) and type(v) is int and v >= 0
                       for k, v in value.items()), result
        else:
            assert type(value) is int and value >= 0, (key, value)
    for key in ("allocations", "allocated_bytes", "elapsed_ns"):
        if key not in result or not isinstance(result[key], int) or result[key] < 0:
            raise AssertionError(f"missing or invalid stats field: {key}")
        del result[key]
    return result


def artifacts(directory):
    return {str(path.relative_to(directory)): path.read_bytes()
            for path in sorted(directory.rglob("*")) if path.is_file()}


def execute(command, directory):
    result = subprocess.run(command, cwd=directory, capture_output=True, timeout=30)
    if result.returncode < 0:
        raise AssertionError(f"process terminated by signal {-result.returncode}: {command}")
    return result


def assert_contract(result, expected):
    if result.returncode != expected["exit"]:
        raise AssertionError(f"expected exit {expected['exit']}, got {result.returncode}: {result.stderr!r}")
    if result.stdout != expected["stdout"].encode():
        raise AssertionError(f"unexpected stdout: {result.stdout!r}")
    if diagnostic(result.stderr) != expected["stderr"].encode():
        raise AssertionError(f"unexpected diagnostic: {result.stderr!r}")


@contextmanager
def timeout_server():
    class HoldConnection(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.settimeout(3)
            try:
                while self.request.recv(4096):
                    pass
            except OSError:
                pass
    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), HoldConnection) as server:
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
        thread.start()
        try:
            yield server.server_address[1]
        finally:
            server.shutdown()
            thread.join(timeout=3)
            assert not thread.is_alive(), "loopback server did not stop"


def run_fixture(fixture, c11, rust, directory, compiler=None, port=None):
    bytecode = directory / "fixture.labc"
    source = fixture
    if fixture.suffix == ".lana":
        assert compiler and Path(compiler).is_file(), "source fixture needs compiler bytecode"
        source = directory / "fixture.lasm"
        compiled = execute([c11, "run", compiler, "--memory-limit-mib", "256",
                            "--instruction-limit", "50000000", "--", str(fixture), str(source)], directory)
        assert compiled.returncode == 0 and compiled.stdout == b"" and compiled.stderr == b"", compiled
    elif "@TIMEOUT_URL@" in fixture.read_text():
        assert port is not None, "fixture needs a loopback server"
        source = directory / "fixture.lasm"
        source.write_text(fixture.read_text().replace("@TIMEOUT_URL@", f"http://127.0.0.1:{port}/".encode().hex()))
    assembly = execute([c11, "asm", str(source), "-o", str(bytecode)], directory)
    if assembly.returncode != 0 or not bytecode.is_file():
        raise AssertionError(f"assembly failed: {assembly.stderr!r}")
    arg_file = fixture.with_suffix(".args")
    arguments = shlex.split(arg_file.read_text()) if arg_file.exists() else []
    contract_file = fixture.with_suffix(".expect.json")
    expected = json.loads(contract_file.read_text()) if contract_file.exists() else None
    runs = []
    for implementation, binary in (("c11", c11), ("rust", rust)):
        run_dir = directory / implementation
        run_dir.mkdir()
        result = execute([binary, "run", str(bytecode), *arguments], run_dir)
        if expected is not None:
            assert_contract(result, expected)
        runs.append((result.returncode, result.stdout, diagnostic(result.stderr), artifacts(run_dir)))
        stats_dir = directory / (implementation + "-stats")
        stats_dir.mkdir()
        stats = execute([binary, "run", str(bytecode), *arguments, "--stats"], stats_dir)
        if stats.returncode != result.returncode or stats.stdout != result.stdout:
            raise AssertionError("--stats changed program exit or stdout")
        without_stats = b"".join(line for line in stats.stderr.splitlines(keepends=True)
                                 if not line.startswith(b"LANAVM_STATS "))
        if diagnostic(without_stats) != diagnostic(result.stderr):
            raise AssertionError("--stats changed the error diagnostic")
        if artifacts(stats_dir) != artifacts(run_dir):
            raise AssertionError("--stats changed filesystem effects")
        runs[-1] += (statistics(stats.stderr),)
    labels = ("exit", "stdout", "stderr", "filesystem effects", "stats")
    for label, left, right in zip(labels, *runs):
        if left != right:
            raise AssertionError(f"{label} differs: C={left!r}, Rust={right!r}; diagnostics={runs[0][2]!r}, {runs[1][2]!r}")
    if expected is not None and expected.get("no_files") and runs[0][3]:
        raise AssertionError(f"unexpected filesystem effects: {runs[0][3].keys()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("groups", nargs="+", choices=GROUPS)
    parser.add_argument("--c11", default=os.environ.get("C11", str(ROOT / "build/lanavm")))
    parser.add_argument("--rust", default=os.environ.get("RUST", str(ROOT / "target/debug/lana-cli")))
    parser.add_argument("--compiler", default=os.environ.get("COMPILER", str(ROOT / "build/lana-compiler.labc")))
    args = parser.parse_args()
    binaries = [str(Path(binary).resolve()) for binary in (args.c11, args.rust)]
    for binary in binaries:
        if not os.access(binary, os.X_OK):
            parser.error(f"executable not found: {binary}")
    count = failures = 0
    for group in args.groups:
        fixtures = sorted((FIXTURES / group).glob("*.lasm"))
        if group == "durable":
            fixtures.append(FIXTURES.parent / "durable/durable_pipeline.lana")
        if not fixtures:
            parser.error(f"no fixtures in {group}")
        for fixture in fixtures:
            count += 1
            try:
                with tempfile.TemporaryDirectory(prefix="lana-conformance-") as directory:
                    with timeout_server() if group == "net" else nullcontext(None) as port:
                        run_fixture(fixture, *binaries, Path(directory), str(Path(args.compiler).resolve()), port)
                print(f"ok   {group}/{fixture.stem}", flush=True)
            except (AssertionError, OSError, ValueError, subprocess.TimeoutExpired) as error:
                failures += 1
                print(f"FAIL {group}/{fixture.stem}: {error}", flush=True)
    print(f"{count - failures}/{count} fixtures passed")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
