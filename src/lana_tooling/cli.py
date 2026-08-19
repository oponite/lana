"""Public Lana CLI: Python source tooling driving the native C VM."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from .compiler import CompileError, compile_source


def native_vm() -> Path:
    configured = os.environ.get("LANA_SSVM")
    if configured:
        path = Path(configured)
    else:
        path = Path(__file__).resolve().parents[2] / "build" / "ssvm"
    if not path.is_file():
        raise RuntimeError(f"native VM not found at {path}; run: cmake -S . -B build && cmake --build build")
    return path


def assemble(source: Path, output: Path) -> int:
    assembly = compile_source(source.read_text())
    with tempfile.NamedTemporaryFile("w", suffix=".ssa", delete=False) as handle:
        handle.write(assembly)
        temporary = Path(handle.name)
    try:
        return subprocess.run([str(native_vm()), "asm", str(temporary), "-o", str(output)], check=False).returncode
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ss",
        description="Lana: a general-purpose language with first-class uncertain, evolving state",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("source", type=Path)
    run.add_argument("--trace", action="store_true")
    run.add_argument("--stats", action="store_true")
    run.add_argument("--seed", type=int)
    compile_command = commands.add_parser("compile")
    compile_command.add_argument("source", type=Path)
    compile_command.add_argument("-o", "--output", type=Path, required=True)
    for command in ("run-bytecode", "dis", "verify"):
        native = commands.add_parser(command)
        native.add_argument("program", type=Path)
        if command == "run-bytecode":
            native.add_argument("--trace", action="store_true")
            native.add_argument("--stats", action="store_true")
            native.add_argument("--seed", type=int)
    asm = commands.add_parser("asm")
    asm.add_argument("source", type=Path)
    asm.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "compile":
            return assemble(args.source, args.output)
        if args.command == "run":
            with tempfile.NamedTemporaryFile(suffix=".ssb", delete=False) as handle:
                bytecode = Path(handle.name)
            try:
                result = assemble(args.source, bytecode)
                if result != 0:
                    return result
                command = [str(native_vm()), "run", str(bytecode)]
                if args.trace:
                    command.append("--trace")
                if args.stats:
                    command.append("--stats")
                if args.seed is not None:
                    command.extend(["--seed", str(args.seed)])
                return subprocess.run(command, check=False).returncode
            finally:
                bytecode.unlink(missing_ok=True)
        if args.command == "asm":
            return subprocess.run([str(native_vm()), "asm", str(args.source), "-o", str(args.output)], check=False).returncode
        command = [str(native_vm()), args.command, str(args.program)]
        if getattr(args, "trace", False):
            command.append("--trace")
        if getattr(args, "stats", False):
            command.append("--stats")
        if getattr(args, "seed", None) is not None:
            command.extend(["--seed", str(args.seed)])
        return subprocess.run(command, check=False).returncode
    except (CompileError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
