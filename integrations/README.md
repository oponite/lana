# Lana integrations

These optional adapters preserve the Lana 1.0 language, LABC v1, and default
dependency-free runtime.

## JSON, MCP, and Jupyter

```bash
python3 -m venv /tmp/lana-integrations-venv
/tmp/lana-integrations-venv/bin/python -m pip install -e integrations/python
printf '{"hello":"lana"}' |
  LANA_EXECUTABLE="$PWD/build/lana" \
  /tmp/lana-integrations-venv/bin/lana-bridge run \
  integrations/lana/echo_bridge.lana
```

Install `integrations/python[mcp]` for `lana-mcp` or
`integrations/python[jupyter]` for `%%lana`.

## Editors

- `editors/vscode`: source-build VS Code extension.
- `editors/neovim`: Lua configuration/plugin for `lana lsp`.

Both integrations currently diagnose files saved on disk.

## Native ABI

```bash
cmake -S . -B build-integrations \
  -DCMAKE_BUILD_TYPE=Release -DLANA_BUILD_INTEGRATIONS=ON
cmake --build build-integrations --parallel
ctest --test-dir build-integrations -R lana_native_bridge --output-on-failure
```

This produces `liblana_bridge` and its ABI-v1 header. The facade runs
precompiled LABC only. Python can load it through
`lana_integrations.native.NativeBridge`.

The integrations require Lana 1.0.x with LABC v1. Source installation is the
supported first release; registry and packaged release publication are deferred.
