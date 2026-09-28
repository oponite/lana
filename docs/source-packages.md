# Hosted source packages

Use exact public GitHub dependencies from a project directory:

```sh
lana package add owner/repo@1.2.3
lana build
```

Import a locked source module with the existing quoted-import syntax:

```lana
import "pkg/owner/repo/src/module.lana" as module;
```

`package add` needs `curl`. It downloads the protected release convention
`lana-v1.2.3`, the `repo-1.2.3-lana.tar.gz` asset, and its `SHA256SUMS` file.
It verifies the archive and complete exact-version dependency closure before
atomically replacing `lana.lock`. The lock is canonical schema-1 JSON. The
first explicit add replaces the legacy local-build lock; local dependencies
stay in the root manifest and continue to affect the build cache.

Build and source compilation use only the locked local cache. Each build
rechecks the compressed digest and extracted source bytes. Editing an extracted
package fails validation, including when compiled bytecode is already cached.
An identical add rechecks the release without rewriting the lock or cache.
A replaced asset, conflicting version, cycle, unsafe path, truncated download,
or failed pre-replacement write leaves the old lock usable. If a directory
sync fails after replacement, reload the reported lock path before retrying.

The cache is `.lana/packages/<sha256>/archive.tar.gz` plus its extracted
`repo-X.Y.Z/` directory. An unused cache directory grants no import authority.
A digest verifies bytes; it does not establish that third-party source is safe.

## Prepare a package

A package's `lana.toml` contains:

```toml
schema = 1
name = "repo"
version = "1.2.3"
entry = "src/main.lana"

[hosted_dependencies]
math = "owner/math@2.0.0"
```

Only `lana.toml`, `src/`, and optional `tests/` enter the archive. Local
path dependencies and manifest build hooks are rejected. Packing does not run
source code or publish anything:

```sh
lana package pack . -o repo-1.2.3-lana.tar.gz
```

The JSON report contains the archive SHA-256. `SHA256SUMS` must contain exactly
that digest, two ASCII spaces, the asset filename, and one LF.

## Publication workflow

Copy [the workflow template](templates/source-package-release.yml) to the
package repository's `.github/workflows/lana-package.yml`, and copy
[`tools/package_release.py`](../tools/package_release.py) to the same relative
path in that repository. Set `LANA_TOOLCHAIN_REPOSITORY` to the Lana toolchain
repository and `LANA_TOOLCHAIN_REVISION` to a full, reviewed 40-character commit
that includes package support. Protect the package repository's `lana-v*` tags.

On an authorized tag push, the read-only qualification job builds that pinned
CLI, packs the source, checks repository/tag/manifest identity and checksum,
then builds and tests a clean extraction. The final job alone gets
`contents: write`. It rechecks the tag commit, returns success for identical
existing assets, and refuses changed or incomplete existing releases. It never
uses overwrite flags. A failed partial GitHub publication requires inspection;
the workflow does not repair it by overwriting assets.

This follows GitHub's [release model](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)
and [job permission rules](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#jobsjob_idpermissions).
The template and publication guard are locally tested; installing it and
publishing in a package repository are separate actions.

## Verification

The regression suite covers packing and the publication guard. The complete local HTTP
fixture and publication fault checks use an explicitly instrumented binary:

```sh
cargo build --locked -p lana-cli --features package-test-origin,publication-fault-injection
python3 tests/test_packages.py target/debug/lana --network-fixture --fault-injection
cargo test --locked -p lana-cli packages::tests::public_release_smoke -- --ignored --nocapture
cargo build --locked -p lana-cli
```

The HTTP-origin override and publication failpoints are absent from normal
builds. The public smoke test downloads a real GitHub CLI release asset and
checks its published SHA-256; it does not publish a Lana package.
