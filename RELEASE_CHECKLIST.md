# Lana release checklist template

Copy this file to `docs/releases/vX.Y.Z.md` for each release. Keep this template
blank. Use the commands and pass criteria in `AGENTS.md`.
Record output or a run link for each check. A skipped or failed required check
does not pass. Run the checks on the exact candidate commit; earlier results do
not qualify a changed tree.

## Release record

- Version: `X.Y.Z`
- Candidate commit SHA: `________`
- Tag: `vX.Y.Z`
- Release date: `________`
- Maintainer: `________`
- Local candidate status (pending/pass/fail): `________`
- Tagged release gates status (pending/pass/fail): `________`
- Publication status (pending/published/failed): `________`

## Prepare the candidate

- [ ] Confirm the release type follows `VERSIONING.md`.
- [ ] Match `VERSION` with Cargo, Python, editor, CI, and release workflow version fields.
- [ ] Update the changelog, migration notes, and support matrix for actual behavior.
- [ ] Review the authority files when language, bytecode, or VM behavior changes.
- [ ] Record the candidate SHA and confirm the release tree has no uncommitted changes.
- [ ] Confirm the required branch checks pass for the candidate SHA.
- [ ] Protect the exact release tag before pushing it. Follow `.github/BRANCH_PROTECTION.md`.

Evidence (commit, checks, and tag protection): `________`

## Run the local gates

Use the five gates in `AGENTS.md`. Record the host,
result, and log path for each gate. The Rust and C fuzz tests each need a full
ten-minute run. Preserve any crash input as a regression.

- [ ] Gate 1: Debug build, CTest, Rust workspace tests, native compiler bootstrap, version, and `git diff --check` pass.
  Host/result/log: `________`
- [ ] Gate 2: Address, undefined-behavior, and thread sanitizer suites pass without a report.
  Host/result/log: `________`
- [ ] Gate 3: C reference and Rust bytecode fuzz targets each complete ten minutes without failure.
  Host/result/logs: `________`
- [ ] Gate 4: The universal macOS install runs both architecture slices and a source example outside the build tree.
  Host/result/log: `________`
- [ ] Gate 5: Optional integration builds and tests pass, including the Python bridge and reference tokenizer tests.
  Host/result/log: `________`

## Verify the tagged release

Use the exact tag and commit recorded above. Check the jobs in the
`.github/workflows/release.yml`; a branch build is not a
substitute for the tag workflow.

- [ ] The protected tag points to the candidate SHA. The workflow reports the same version as `VERSION`.
- [ ] The Linux release job passes its build, Rust, sanitizer, fuzz, and integration steps.
- [ ] The macOS release job builds and installs both architecture slices.
- [ ] The downloaded macOS archive passes SHA-256 verification; both slices report their version, and the copied example runs.
- [ ] The source archive passes SHA-256 verification, builds and installs in a clean directory, and runs the example.
- [ ] The release includes `SHA256SUMS` and a Homebrew formula with the source archive digest.

Tag SHA, workflow run, and artifact evidence: `________`

## Confirm publication

- [ ] The GitHub Release exists for the recorded tag and candidate SHA.
- [ ] The release is public, non-draft, non-prerelease, and has the expected archives, `SHA256SUMS`, and formula.
- [ ] Download the published archives and verify them against the published checksums.

Release URL and verification log: `________`

Distribution steps are separate from source release qualification. Record each
as `deferred`, `pending`, or `complete`; include evidence for completed steps.

- Code signing: `________`
- macOS notarization: `________`
- Homebrew Core submission: `________`
