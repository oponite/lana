"""Root-scoped stdio MCP server for Lana 1.0."""

from __future__ import annotations

import argparse
from typing import Any, Sequence

from .bridge import BridgeRunner
from .roots import RootPolicy


def build_server(
    *,
    roots: Sequence[str],
    lana_executable: str | None = None,
    allow_run: bool = False,
    timeout_seconds: float = 30.0,
) -> Any:
    try:
        from mcp.server import MCPServer
        from mcp_types import ToolAnnotations
    except ImportError as error:
        raise RuntimeError(
            "MCP support is not installed; install lana-integrations[mcp]"
        ) from error

    policy = RootPolicy(roots)
    runner = BridgeRunner(lana_executable, timeout_seconds=timeout_seconds)
    server = MCPServer(
        "lana",
        instructions=(
            "Check and run Lana 1.0 programs beneath configured roots. "
            "Execution is local and may perform file effects."
        ),
    )

    read_only = ToolAnnotations(readOnlyHint=True, destructiveHint=False)
    side_effectful = ToolAnnotations(readOnlyHint=False, destructiveHint=True)

    @server.tool(annotations=read_only)
    def lana_version() -> dict[str, Any]:
        """Return the selected compatible Lana runtime version."""
        return {
            "schema": 1,
            "ok": True,
            "result": {
                "lana_version": runner.version,
                "labc_version": 1,
                "executable": runner.executable,
            },
        }

    @server.tool(annotations=read_only)
    def lana_check(program: str) -> dict[str, Any]:
        """Check a Lana source program beneath an allowed root without running it."""
        resolved = policy.resolve_program(program)
        return runner.check(resolved)

    if allow_run:

        @server.tool(annotations=side_effectful)
        def lana_run(
            program: str,
            input: Any,
            seed: int | None = None,
            memory_limit_mib: int | None = None,
            instruction_limit: int | None = None,
            workers: int | None = None,
            max_tasks: int | None = None,
            timeout_seconds: float | None = None,
        ) -> dict[str, Any]:
            """Run an allowed Lana bridge program; it may perform local file effects."""
            resolved = policy.resolve_program(program)
            return runner.run(
                resolved,
                input,
                seed=seed,
                memory_limit_mib=memory_limit_mib,
                instruction_limit=instruction_limit,
                workers=workers,
                max_tasks=max_tasks,
                timeout_seconds=timeout_seconds,
            )

    return server


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lana-mcp")
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--lana")
    parser.add_argument("--allow-run", action="store_true")
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        server = build_server(
            roots=arguments.root,
            lana_executable=arguments.lana,
            allow_run=arguments.allow_run,
            timeout_seconds=arguments.timeout,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        _parser().error(str(error))
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
