# Required release branch checks

Protect `main` and `dev` in the GitHub repository. Require pull requests, a
branch that is up to date before merge, and these status checks:

- `Rust CLI and regression tests`
- `Optional integrations`
- `Rust tests`
- `Actual WASM execution`
- `Real editor clients`
- `Rust fuzz smoke test`

Do not require `Rust full fuzz test` for ordinary pull requests. It runs on the
weekly schedule and on version tags.

Create a repository ruleset for each exact release tag, including `v4.1.0`,
before pushing it.
Restrict tag creation and updates to release maintainers, and disallow tag
deletion. The release workflow checks `github.ref_protected`, so it cannot
publish unless GitHub reports that this tag is protected.

Keep workflow permissions read-only by default. Only the `Publish GitHub
Release` job may request `contents: write`. The later tap job uses the
`HOMEBREW_TAP_DEPLOY_KEY` Actions secret, whose write deploy key belongs only
to `oponite/homebrew-oponite`.
