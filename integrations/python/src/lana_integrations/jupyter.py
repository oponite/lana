"""IPython cell magic for source-installed Lana integrations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import tempfile
from typing import Any

from .bridge import BridgeRunner


def _line_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="%%lana", add_help=False, exit_on_error=False)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--memory-limit-mib", type=int)
    parser.add_argument("--instruction-limit", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--timeout", type=float)
    return parser


def _display_envelope(shell: Any, envelope: dict[str, Any]) -> None:
    shell.user_ns["_lana_result"] = envelope
    if envelope.get("stdout"):
        print(envelope["stdout"], end="" if envelope["stdout"].endswith("\n") else "\n")
    if envelope.get("stderr"):
        print(envelope["stderr"], end="" if envelope["stderr"].endswith("\n") else "\n")
    if envelope.get("ok") and envelope.get("result") is not None:
        from IPython.display import JSON, display

        display(JSON(envelope["result"], expanded=True))
    if not envelope.get("ok"):
        error = envelope.get("error", {})
        raise RuntimeError(
            f"Lana {envelope.get('phase', 'execution')} failed: "
            f"{error.get('message', 'unknown error')}"
        )


def load_ipython_extension(shell: Any) -> None:
    try:
        from IPython.core.magic import Magics, cell_magic, magics_class
    except ImportError as error:
        raise RuntimeError(
            "Jupyter support is not installed; install lana-integrations[jupyter]"
        ) from error

    runner = BridgeRunner()

    @magics_class
    class LanaMagics(Magics):
        @cell_magic
        def lana(self, line: str, cell: str) -> None:
            arguments = _line_parser().parse_args(shlex.split(line))
            working_directory = Path.cwd()
            descriptor, source_name = tempfile.mkstemp(
                prefix=".lana-cell-", suffix=".lana", dir=working_directory
            )
            source_path = Path(source_name)
            try:
                with open(descriptor, "w", encoding="utf-8", closefd=True) as stream:
                    stream.write(cell)
                if arguments.check:
                    envelope = runner.check(
                        source_path, timeout_seconds=arguments.timeout
                    )
                elif arguments.input is not None:
                    input_value = json.loads(
                        Path(arguments.input).read_text(encoding="utf-8")
                    )
                    envelope = runner.run(
                        source_path,
                        input_value,
                        seed=arguments.seed,
                        memory_limit_mib=arguments.memory_limit_mib,
                        instruction_limit=arguments.instruction_limit,
                        workers=arguments.workers,
                        max_tasks=arguments.max_tasks,
                        timeout_seconds=arguments.timeout,
                    )
                else:
                    envelope = runner.run_plain(
                        source_path,
                        seed=arguments.seed,
                        memory_limit_mib=arguments.memory_limit_mib,
                        instruction_limit=arguments.instruction_limit,
                        workers=arguments.workers,
                        max_tasks=arguments.max_tasks,
                        timeout_seconds=arguments.timeout,
                    )
                _display_envelope(self.shell, envelope)
            finally:
                source_path.unlink(missing_ok=True)

    shell.register_magics(LanaMagics)


def unload_ipython_extension(shell: Any) -> None:
    shell.user_ns.pop("_lana_result", None)
