# Changelog

## 2.0.0

- Introduce LABC v2, the Rust canonical VM, algebraic data types, evidence, and lazy datasets.
- Add the durable decision pipeline: store, policy, ledger, claims, and effects.
- Add dlopen adapter plugins (SQLite, HTTP/JSON) and a shared runtime for bindings.
- Add four reference applications (sensor fusion, service health, doc router, advisory forecast) with fixtures.
- Add the native bridge pipeline API (`lana_bridge_run_pipeline`) for durable decision execution.
- Add state codec, bytecode codec, and SHA-256 support.
- Add packaging and install verification (`package.sh`, `verify-install.sh`, local-install test).
- Fix release-build test assertions and strict-warning issues across the C runtime.

## 1.1.1

- Preserve the 1.1 evidence lifecycle contract across policy, replay, and
  optional integration boundaries.
- Add an explicit policy helper for requesting more evidence without
  authorizing an effect.
- Prepare the release metadata and validation surface for package distribution.

## 1.0.0

- Establish the single LABC v1 format for Lana 1.0.
- Add insertion-ordered maps, strict JSON, and RFC 4180 CSV host APIs.
- Add relative namespaced modules and source-aware diagnostics.
- Bound task execution with a shared worker scheduler and deterministic child RNG derivation.
- Add project scaffolding, formatting, linting, testing, LSP, and debugger tools.
- Add a precise tracing collector with generational and incremental paths.
- Publish a clean LABC boundary with no pre-release bytecode compatibility.
