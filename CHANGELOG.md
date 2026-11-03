# Changelog

## 4.2.0

- Accept decimal exponent literals such as `1e-5` and `2.5E+3`; an incomplete
  exponent is a lexical error.
- Keep unweighted guard paths unweighted: `sample` on them returns
  `LANA_ERR_UNSUPPORTED_OPERATION`. Weighted guards retain branch weights
  through pure operations.
- Use the checked tensor path for unaligned state views.
- Add the `lana_execution` VM fuzz target and expand CI and release checks.

Robustness and hardening. Security: store journals now reject symlinks,
non-regular files, and shared hard links (defense in depth; see `SECURITY.md`
for the trust model).

- Size legacy VM function frames safely when control flow crosses function
  boundaries or arguments occupy registers unused by the function body,
  including `ASYNC` argument ranges in the caller's frame.
- Document the trust model and private vulnerability reporting in `SECURITY.md`.
- Bound bootstrap temporary memory, sample counts, and native work; return
  structured errors instead of aborting the worker on oversized requests.
- Bound bytecode file reads before loading and preserve loader limit errors
  across CLI, worker, and live entry points.
- Publish store manifests and snapshots through exclusive temporary files;
  reject unsafe journal inputs and report uncertain commit outcomes with
  reopen guidance through CLI and Python errors.
- Update Rustls to 0.23.46, fixing five GitHub advisories against 0.23.44
  (two moderate, three low), and certificate validation and trust roots.
- Remove unused internal compiler helpers and storage, regenerate the checked
  bootstrap, and retain public APIs and legacy bytecode support.
- Validate benchmark candidates against VERSION and measure precompiled CLI
  execution separately from source compilation in report schema 2.

## 4.1.0

- Add process-local persistent live programs with explicit named Information
  registration, transactional host observation, inspection, pause/resume, and
  deletion through the Rust, CLI, and Python worker surfaces.
- Publish each later release's verified source formula to the latest-version
  `oponite/oponite` Homebrew tap after the GitHub Release is public.

## 4.0.0

- Distribute the Rust CLI and VM as the sole Lana engine; retire the C VM,
  C-facing APIs, and standalone C evidence server.
- Add LABC v6 object declarations, checked methods, and task-local classes;
  preserve v1-v5 loading and frozen v1-v2 golden bytecode checks.
- Add finite information, durable dataset history, bounded rules and trees,
  walk-forward evaluation, and immutable information snapshots.
- Route repeated Python calls through a persistent Rust worker with a fresh VM
  per request; port read-only SQLite and localhost HTTP_JSON adapters to Rust.

## 3.0.2

- Reject unresolved dataset predicates and keys; fix self-join locking.
- Validate webhook authorization and receipt transitions before transport;
  disable curl configuration, redirects, retries, and URL expansion.
- Correct CLI JSON escaping and Rust FFI chunk ownership and record headers.
- Add Core matrix, compiler failure, bridge, and C-consumer regressions;
  document the actual supported operation and integration boundaries.

## 3.0.1

- Make the Rust CLI/VM release gates, bytecode fuzzing, execution configuration,
  support matrix, and migration path explicit.
- Harden execution configuration and temporary credential-file handling.
- Keep the C11 v1-v2 reference and native ABI-v1 bridge frozen and supported.

## 3.0.0

- Establish the Rust CLI/VM as the canonical runtime for LABC v1-v5.
- Add LABC v5 Core information distribution and map-refinement support.

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
