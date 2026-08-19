from __future__ import annotations

import argparse
import json
from pathlib import Path

from structured_state.compiler.compiler import compile_source
from structured_state.compiler.disassembler import disassemble
from structured_state.vm.bytecode import Instruction, Op, Program
from structured_state.vm.machine import VM


def load_program(path: Path) -> Program:
    raw = json.loads(path.read_text())
    return Program(version=raw["version"], constants=raw["constants"], functions=raw.get("functions", {}),
                   instructions=[Instruction(Op(item["op"]), tuple(item["args"]), item.get("line", 0)) for item in raw["instructions"]])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ss", description="Lana structured-state language")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run"); run.add_argument("source", type=Path); run.add_argument("--seed", type=int); run.add_argument("--trace", action="store_true")
    compile_command = commands.add_parser("compile"); compile_command.add_argument("source", type=Path); compile_command.add_argument("-o", "--output", type=Path, required=True)
    bytecode = commands.add_parser("run-bytecode"); bytecode.add_argument("program", type=Path); bytecode.add_argument("--seed", type=int); bytecode.add_argument("--trace", action="store_true")
    dis = commands.add_parser("dis"); dis.add_argument("program", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run": VM(compile_source(args.source.read_text()), seed=args.seed, trace=args.trace).run()
    elif args.command == "compile": args.output.write_text(json.dumps(compile_source(args.source.read_text()).to_dict(), indent=2))
    elif args.command == "run-bytecode": VM(load_program(args.program), seed=args.seed, trace=args.trace).run()
    else: print(disassemble(load_program(args.program)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
