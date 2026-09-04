# Required 1.1 branch checks

Protect `main` and `dev` in the GitHub repository. Require pull requests, a
branch that is up to date before merge, and these status checks:

- `Native (gcc)`
- `Native (clang)`
- `Address and undefined behavior sanitizers`
- `Thread sanitizer`
- `Optional integrations`
- `Fuzz smoke test`

Do not require `Full fuzz test` for ordinary pull requests. It runs on the
weekly schedule and on version tags.

Create a repository ruleset for the exact tag `v2.0.0` before pushing it.
Restrict tag creation and updates to release maintainers, and disallow tag
deletion. The release workflow checks `github.ref_protected`, so it cannot
publish unless GitHub reports that this tag is protected.

Keep workflow permissions read-only by default. Only the `Publish GitHub
Release` job may request `contents: write`.
