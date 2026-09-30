from __future__ import annotations

import asyncio
from pathlib import Path
import sys

import pytest

from lana_integrations.mcp_server import build_server
from lana_integrations import BridgeRunner


mcp = pytest.importorskip("mcp")


def test_read_only_server_exposes_only_version(fake_lana: Path, tmp_path: Path) -> None:
    server = build_server(roots=[str(tmp_path)], lana_executable=str(fake_lana))

    async def exercise() -> None:
        async with mcp.Client(server, raise_exceptions=False) as client:
            tools = await client.list_tools()
            assert [tool.name for tool in tools.tools] == ["lana_version"]
            result = await client.call_tool("lana_version", {})
            assert result.is_error is False
            assert result.structured_content["ok"] is True
            assert result.structured_content["result"]["labc_version"] == 5

    asyncio.run(exercise())


@pytest.mark.parametrize("allow_run", [False, True])
def test_real_stdio_runtime(built_lana: Path, tmp_path: Path, allow_run: bool) -> None:
    from mcp.client.stdio import StdioServerParameters
    root = Path(__file__).resolve().parents[3]
    args = ["-m", "lana_integrations.mcp_server", "--root", str(root), "--lana", str(built_lana)]
    if allow_run:
        args.append("--allow-run")
    parameters = StdioServerParameters(command=sys.executable, args=args, cwd=root)

    async def exercise() -> None:
        async with mcp.Client(parameters, read_timeout_seconds=15) as client:
            tools = await client.list_tools()
            assert {tool.name for tool in tools.tools} == ({"lana_version", "lana_run"} if allow_run else {"lana_version"})
            version = await client.call_tool("lana_version", {})
            assert not version.is_error
            assert version.structured_content["result"]["labc_version"] == BridgeRunner(built_lana).labc_version
            if allow_run:
                result = await client.call_tool("lana_run", {
                    "program": "integrations/lana/echo_bridge.lana", "input": {"value": False},
                    "seed": 17, "instruction_limit": 50_000_000,
                })
                assert not result.is_error
                assert result.structured_content["result"] == {"value": False}
                outside = tmp_path / "outside.lana"
                outside.write_text('print("outside");\n')
                rejected = await client.call_tool("lana_run", {"program": str(outside), "input": {}})
                assert rejected.is_error

    asyncio.run(exercise())


def test_run_requires_opt_in(fake_lana: Path, program: Path, tmp_path: Path) -> None:
    server = build_server(
        roots=[str(tmp_path)], lana_executable=str(fake_lana), allow_run=True
    )

    async def exercise() -> None:
        async with mcp.Client(server, raise_exceptions=False) as client:
            tools = await client.list_tools()
            assert [tool.name for tool in tools.tools] == [
                "lana_version",
                "lana_run",
            ]

            result = await client.call_tool(
                "lana_run", {"program": str(program), "input": {"value": 7}}
            )
            assert result.structured_content["ok"] is True
            assert result.structured_content["result"] == {"value": 7}
            outside = tmp_path.parent / (tmp_path.name + "-outside.lana")
            outside.write_text("print(2);\n", encoding="utf-8")
            try:
                rejected = await client.call_tool(
                    "lana_run", {"program": str(outside), "input": {}}
                )
                assert rejected.is_error is True
            finally:
                outside.unlink()

    asyncio.run(exercise())
