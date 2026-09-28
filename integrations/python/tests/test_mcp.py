from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from lana_integrations.mcp_server import build_server


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
