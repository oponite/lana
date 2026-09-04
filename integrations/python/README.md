# Lana integrations

This optional Python package connects Lana 1.x programs to subprocess callers,
MCP hosts, and IPython. It does not add dependencies to Lana itself.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e integrations/python
.venv/bin/lana-bridge check integrations/lana/echo_bridge.lana
printf '{"message":"hello"}' |
  .venv/bin/lana-bridge run integrations/lana/echo_bridge.lana
```

The `lana` executable is resolved from `--lana`, `LANA_EXECUTABLE`, then
`PATH`. Only Lana 1.x reporting LABC v1 is accepted.

The ergonomic `Lana` class prefers the native ctypes bridge when a compatible
`liblana_bridge` is available and falls back to the subprocess bridge otherwise:

```python
from lana_integrations import Lana

lana = Lana()
result = lana.run("program.lana", {"message": "hello"})
assert result.status == "ok"
assert result.value == {"message": "hello"}
```

`LanaResult.status` is one of `"ok"`, `"unavailable"`, `"unresolved"`, or
`"failed"`; unresolved and failed outcomes are never coerced into ordinary
values.

Optional components:

```bash
.venv/bin/python -m pip install -e 'integrations/python[mcp,jupyter]'
.venv/bin/lana-mcp --root .
```

Execution-capable MCP tools are disabled unless `--allow-run` is passed.
Configured roots restrict which program can be selected; they do not sandbox
file operations performed by that program.
