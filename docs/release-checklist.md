# Lana release qualification checklist

Use the five gates in [AGENTS.md](../AGENTS.md) on the exact candidate. Record
commands, exit status, host, and log paths. Earlier logs do not qualify changed
content. A failed or skipped required gate remains open.

## Candidate identity

- Version and commit: ________
- Working-source content manifest, including required untracked files: ________
- Rust toolchain, operating system, architecture, and build profile: ________
- Local qualification status: ________
- Tag/publication status (separate from qualification): ________

## Required local gates

1. Build with `python3 tools/build.py build`; run `python3 tests/run.py --no-build`,
   `cargo test --locked --workspace --no-fail-fast`, `lana version`, and
   `git diff --check`. Require twice-repeated byte-stable bootstrap, frozen
   compatibility, generated projects/imports, LSP, debugger, installation,
   packaging and isolated publication-failure checks.
2. Run the Rust malformed-bytecode fuzz target for 600 seconds with the matching
   nightly compiler and a five-second input timeout. Preserve any failing input.
3. Build and install the universal macOS prefix. Verify arm64/x86_64 headers and
   actually execute both slices. Extract the checksum-verified archive outside
   the checkout and run source using its adjacent compiler and installed stdlib.
4. Run Python, MCP, Jupyter, HF tokenizer/package checks, actual wasm32 Node
   conformance, and real Neovim/VS Code client tests. Build the source archive in
   a clean directory, verify checksums, and run its installed example.
5. Run the paired Release method in [rust-only-performance.md](../plans/rust-only-performance.md):
   30 calls per path, discard five warmups, require every candidate warm median
   within 5% of the paired 3.0.2 median. Report Python first calls separately.
   Do not run fuzzing or builds concurrently with measurements.

Evidence for each gate: ________
Remaining failures, required skips, or uncertain results: ________

## Distribution boundary

After qualification, separately verify the clean tracked candidate, protected
tag, release-workflow checksums, and downloaded artifacts before publication.
Local qualification does not publish a release. Signing/notarization and Homebrew
Core submission remain explicitly deferred distribution steps. Never interpret a
formula artifact as Homebrew acceptance.
