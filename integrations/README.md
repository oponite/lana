# Lana integrations

These optional adapters preserve the Lana 1.1 language, LABC v1, and default
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

The `qqq_eow_probability.lana` program is a deterministic JSON-bridge inference
consumer for the Quant Research QQQ end-of-week probability artifact. Python
retrieves data and fits/calibrates the model; Lana validates the request,
calculates the calibrated probabilities, and writes an advisory-only response.
Its `confidence_status` is deliberately binary: `high` is emitted only when
the supplied walk-forward evidence passes every gate; all other cases emit
`low`.

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

The integrations require Lana 1.0.x or 1.1.x with LABC v1. Source installation
remains supported; packaged publication follows the corresponding Lana release.

## Evidence lifecycle contract

Schema-1 evidence keeps these fields distinct: `source`, `observed_at`,
`effective_at`, `exactness`, `revision`, `confidence`, `provenance_id`, and
`dependency_ids`. Optional `reliability` and `calibration` metadata is preserved
without coercion. `observed_at` and `max_age` determine staleness; evidence is
stale when `current_time - observed_at > max_age`. A policy result retains the
validated evidence record, and replay validates that record, the policy inputs,
the decision, and the requested effect without executing the effect. Low
confidence produces `abstain`; `policy.request_more_evidence` is the explicit
helper for the same non-executing outcome.
