"""Command-line entrypoint for the Lana JSON bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from .bridge import BridgeRunner, LanaCompatibilityError, _error_envelope


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lana-bridge")
    parser.add_argument("--lana", help="path to the Lana 1.x executable")
    parser.add_argument("--timeout", type=float, default=30.0)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check", help="check a Lana source program")
    check.add_argument("program")
    run = subparsers.add_parser("run", help="run a JSON bridge program")
    run.add_argument("program")
    run.add_argument("--input", default="-", help="JSON file, or - for stdin")
    run.add_argument("--seed", type=int)
    run.add_argument("--memory-limit-mib", type=int)
    run.add_argument("--instruction-limit", type=int)
    run.add_argument("--workers", type=int)
    run.add_argument("--max-tasks", type=int)
    return parser


def _read_input(path: str) -> Any:
    text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    return json.loads(text)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        runner = BridgeRunner(arguments.lana, timeout_seconds=arguments.timeout)
        if arguments.command == "check":
            envelope = runner.check(arguments.program)
        else:
            input_value = _read_input(arguments.input)
            envelope = runner.run(
                arguments.program,
                input_value,
                seed=arguments.seed,
                memory_limit_mib=arguments.memory_limit_mib,
                instruction_limit=arguments.instruction_limit,
                workers=arguments.workers,
                max_tasks=arguments.max_tasks,
            )
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as error:
        envelope = _error_envelope(
            "input", str(error), code="LANA_BRIDGE_INPUT_ERROR"
        )
    except LanaCompatibilityError as error:
        envelope = _error_envelope(
            "compatibility", str(error), code="LANA_INCOMPATIBLE_VERSION"
        )
    print(json.dumps(envelope, ensure_ascii=False, separators=(",", ":")))
    return 0 if envelope.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
