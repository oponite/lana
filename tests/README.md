# Tests

Run the daily checks from the repository root:

```sh
python3 tests/run.py quick
```

The runner builds the candidate before it runs tests. Each run creates a new
directory under `build/test-evidence/`. The directory contains command logs,
CTest JUnit results, and `report.json`.

## Profiles

| Profile | Checks |
| --- | --- |
| `quick` | CTest without optional/hardware tests, the Rust workspace, differential fixtures, numerical oracles, and seeded bytecode mutations. |
| `full` | Quick checks, two byte-stable Rust compiler bootstrap passes, and every native CTest, including Java, Node, WASI, and platform-specific tests. |
| `asan` / `tsan` | The complete CTest suite in an Address/UndefinedBehaviorSanitizer or ThreadSanitizer build. |
| `integrations` | The complete integration-enabled CTest suite and Python bridge tests in an isolated environment. |
| `nightly` | Full checks, Address/UndefinedBehaviorSanitizer, ThreadSanitizer, and ten minutes of libFuzzer. |
| `hardware` | macOS Metal tests and a universal clean install with both architecture slices. |
| `release` | Nightly checks, a universal clean install, and native/Python integration tests. Requires macOS. |

These profiles check source behavior. They do not create a tag, publish a
release, sign binaries, or submit a package-manager formula.

Required tools for `quick` are Python 3.9+, CMake with JUnit output support,
a C11 compiler, Cargo, libffi, and OpenSSL. Assertions must remain enabled.
`full` also requires a JDK, Node, wasm-bindgen 0.2.100, wasmtime, and both Rust
targets: `wasm32-unknown-unknown` and `wasm32-wasip1`.
The sanitizer and integration profiles need these tools too.
Use CMake's `-DBUILD_TESTING=OFF` for a source-only build without Python test
dependencies. The profile runner always enables testing.

On macOS, fuzzing requires a Clang installation with libFuzzer. The runner
uses Homebrew LLVM by default. `--fuzz-cc` selects another installation.
Universal builds require dependencies for both target architectures and an
x86_64 execution environment.

Select a compiler or build directory when necessary:

```sh
python3 tests/run.py full --cc clang --build-dir build-clang
python3 tests/run.py nightly --fuzz-cc /opt/homebrew/opt/llvm/bin/clang
```

## What the evidence means

The report identifies the Git commit and a SHA-256 fingerprint of the candidate
files. The fingerprint includes tracked changes and untracked source files.
If those files change during a run, the runner rejects the evidence.

Every selected test must run. Missing prerequisites, skipped tests, signals,
timeouts, and missing JUnit records cannot count as passes. A direct CTest run
still permits exit-77 skips. The profile runner rejects those skips.

`claims.json` maps bounded claims to authority documents and executable checks.
The registry rejects missing files, unknown tests, and unknown gates. The
report distinguishes checked claims from claims outside the selected profile.

Passing tests do not guarantee correctness for all programs. They establish
specific behavior for the inputs, platforms, and fault cases that ran. Coverage
counts are not a formal proof. The registry states known limits explicitly.

## Independent checks

`test_oracles.py` computes expected results without either VM implementation.
It checks 128 state cases against matrix arithmetic, rectangular tensor
products and reductions, and every Lana code block in the root README.

The differential runner compares exit status, stdout, diagnostics, filesystem
contents, and instruction statistics. It excludes only timing and
implementation-specific allocation counts. Each VM receives a fresh directory.
An `.expect.json` file adds an independent expected outcome.

`test_bytecode.py` checks known malformed files and deterministic mutations.
It preserves mismatches under the report directory. The libFuzzer target also
runs verified, allowlisted instructions with 1 MiB and 1,000-instruction limits.
It does not permit host calls, output, worker threads, or inner-loop estimators.

The scalar-versus-density comparison checks model choices, not language power.
Python and Java can represent density matrices. Benchmark checks validate
accounting and missing-data labels, not a machine-specific speed claim.

`test_store_process.py` checks C/Rust lock contention, explicit close, killed
writers, every byte cut in a two-revision journal, and manifest publication
failure. Recovery tests distinguish acknowledged data from an unacknowledged
commit. They also verify a new commit after recovery and compaction.

`test_net_live.py` uses real HTTP/TLS servers on ephemeral loopback ports.
Both VMs must return the expected body and provenance. A default TLS rejection
must also produce a certificate alert at the server, without an HTTP request.
An OpenSSL trust-file test accepts the matching IP certificate and rejects the
same trusted certificate for a different hostname. Framing tests reject truncated
bodies and invalid or conflicting lengths. Transfer-coded responses are rejected
until a decoder is implemented; they cannot appear as successful raw bodies.

## Add a regression

1. Add the smallest input that reproduces the fault.
2. Register source fixtures in `cmake/CTestTargets.cmake` or an existing source-test script.
3. Add an independent expected result, including the error code for a negative case.
4. For rejected effects, check empty output and unchanged external state.
5. Run the focused test before the profile runner.
6. Update `claims.json` when a new behavior requires a new claim or gate.

The registry checks for unregistered Python/source tests and orphan expectation
files. Rust unit tests remain beside the code. C unit tests remain in `unit/`.
The shared conformance runner discovers every fixture in its selected groups.
