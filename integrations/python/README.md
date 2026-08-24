# Lana integrations

This optional Python package connects Lana 1.0 programs to subprocess callers,
MCP hosts, and IPython. It does not add dependencies to Lana itself.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e integrations/python
.venv/bin/lana-bridge check integrations/lana/echo_bridge.lana
printf '{"message":"hello"}' |
  .venv/bin/lana-bridge run integrations/lana/echo_bridge.lana
```

The `lana` executable is resolved from `--lana`, `LANA_EXECUTABLE`, then
`PATH`. Only Lana 1.0.x reporting LABC v1 is accepted.

Optional components:

```bash
.venv/bin/python -m pip install -e 'integrations/python[mcp,jupyter]'
.venv/bin/lana-mcp --root .
```

Execution-capable MCP tools are disabled unless `--allow-run` is passed.
Configured roots restrict which program can be selected; they do not sandbox
file operations performed by that program.
