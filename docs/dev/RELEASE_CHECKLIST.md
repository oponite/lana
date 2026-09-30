# Lana release qualification checklist

Copy this template to `docs/releases/vX.Y.Z.md` for each release. Keep this
template blank.

Use the five gates in [AGENTS.md](../../AGENTS.md) on the exact candidate. Record
commands, exit status, host, and log paths. Earlier logs do not qualify changed
content. A failed or skipped required gate remains open.

## Candidate identity

- Version: `X.Y.Z`
- Candidate commit SHA: ________
- Tag: `vX.Y.Z`
- Release date: ________
- Maintainer: ________
- Working-source content manifest, including required untracked files: ________
- Rust toolchain, operating system, architecture, and build profile: ________
- Local qualification status (pending/pass/fail): ________
- Tagged release gates status (pending/pass/fail): ________
- Publication status (pending/published/failed): ________

## Prepare the candidate

- [ ] Confirm the release type follows [VERSIONING.md](../../VERSIONING.md).
- [ ] Match `VERSION` with Cargo, Python, editor, CI, and release workflow version fields.
- [ ] Update the changelog, migration notes, and support matrix for actual behavior.
- [ ] Review the authority files when language, bytecode, or VM behavior changes.
- [ ] Record the candidate SHA and confirm the release tree has no uncommitted changes.
- [ ] Confirm the required branch checks pass for the candidate SHA.
- [ ] Protect the exact release tag before pushing it. Follow
  [BRANCH_PROTECTION.md](../../.github/BRANCH_PROTECTION.md).

Evidence (commit, checks, and tag protection): ________

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
4. Run Python, MCP, Jupyter checks, actual wasm32 Node
   conformance, and real Neovim/VS Code client tests. Build the source archive in
   a clean directory, verify checksums, and run its installed example.
5. Run `python3 tools/benchmark.py --help` and use the paired Release harness
   with explicit baseline root, CLI, and native library paths. Use the same
   compiler artifact for both CLIs. The harness alternates 30 calls per path,
   discards five warmups, records samples and SHA-256 identities, and fails if
   any candidate warm median exceeds 1.05 times its paired 3.0.2 median.
   Check that the old C11 and current Rust echo wrappers return the same input.
   Report Python first calls separately. Do not run fuzzing or builds during
   measurements.

Evidence for each gate: ________
Remaining failures, required skips, or uncertain results: ________

## Verify the tagged release

Use the exact tag and commit recorded above. Check the jobs in the
[release workflow](../../.github/workflows/release.yml). A branch build does not
substitute for the tag workflow.

- [ ] The protected tag points to the candidate SHA. The workflow reports the same version as `VERSION`.
- [ ] The Linux release job passes its build, regression, Rust workspace, fuzz, and integration checks.
- [ ] The macOS release job builds and installs both architecture slices.
- [ ] The downloaded macOS archive passes SHA-256 verification; both slices report their version, and the copied example runs.
- [ ] The source archive passes SHA-256 verification, builds and installs in a clean directory, and runs the example.
- [ ] The release includes `SHA256SUMS` and a Homebrew formula with the source archive digest.

Tag SHA, workflow run, and artifact evidence: ________

## Confirm publication

- [ ] The GitHub Release exists for the recorded tag and candidate SHA.
- [ ] The release is public, non-draft, non-prerelease, and has the expected archives, `SHA256SUMS`, and formula.
- [ ] Download the published archives and verify them against the published checksums.

Release URL and verification log: ________

## Distribution boundary

After qualification, separately verify the clean tracked candidate, protected
tag, release-workflow checksums, and downloaded artifacts before publication.
Local qualification does not publish a release. Signing/notarization and Homebrew
Core submission remain explicitly deferred distribution steps. Never interpret a
formula artifact as Homebrew acceptance.

Record each distribution step as `deferred`, `pending`, or `complete`, with
evidence for completed steps:

- Code signing: ________
- macOS notarization: ________
- Homebrew Core submission: ________
