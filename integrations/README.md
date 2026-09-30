# Lana integrations

These optional adapters connect Lana to Python, JSON callers, MCP clients,
notebooks, and editors.

## Choose an interface

| Use Lana from | Start here |
| --- | --- |
| Python | [Python API](python/README.md#python-api) |
| A JSON command line | [JSON bridge](python/README.md#json-command-line-bridge) |
| An MCP client | [MCP setup](python/README.md#mcp) |
| Jupyter or IPython | [Notebook setup](python/README.md#jupyter-and-ipython) |
| VS Code | [VS Code setup](editors/vscode/README.md) |
| Neovim | [Neovim setup](editors/neovim/README.md) |

## Try the JSON bridge

From the repository root:

```bash
python3 tools/build.py build
python3 -m venv .venv
.venv/bin/python -m pip install -e integrations/python
printf '{"message":"hello"}' |
  .venv/bin/lana-bridge --lana "$PWD/target/lana/bin/lana" run integrations/lana/echo_bridge.lana
```

The JSON response has `"ok":true` and `"result":{"message":"hello"}`.

## Bundled programs

[`lana/`](lana/) contains the echo program, bridge helpers, and examples for
evidence, policy, replay, and forecasting. They use caller-supplied inputs.
See the linked adapter guides for usage and setup details.
