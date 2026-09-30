# Using, Publishing, and Testing Lana Packages

## 1. Install a library.

```sh
lana package add owner/repo@1.2.3
lana build
```

- Retrieve version `1.2.3` of the library in Github repo `owner/repo`.
- Lana then downloads it, checks its checksum, and downloads any other required libraries (external dependency: `curl`).

Additionally, Lana records the exact versions and checksums in `lana.lock`, so future builds know which files to use.

## 2. Use an installed library in your program.

Import a source file from the installed library:

```lana
import "pkg/owner/repo/src/module.lana" as module;
```

## 3. Prepare your library for others.

A package's `lana.toml` contains:

```toml
schema = 1
name = "repo"
version = "1.2.3"
entry = "src/main.lana"

[hosted_dependencies]
math = "owner/math@2.0.0"
```

Bundle `lana.toml`, `src/`, and `optional tests/`:

```sh
lana package pack . -o repo-1.2.3-lana.tar.gz
```

It rejects dependencies on local paths and build hooks, because those would depend on the author’s machine or extra execution.

## 4. Publish the library on GitHub.

Set up these files in your library's GitHub repository:

1. Copy [the workflow template](../templates/source-package-release.yml)
   to `.github/workflows/lana-package.yml`.
2. Copy [`tools/package_release.py`](../../tools/package_release.py) (here)
   to `tools/package_release.py` (your library).
3. Set the GitHub Actions variable `LANA_TOOLCHAIN_REPOSITORY`
   to the Lana compiler repository.
4. Set `LANA_TOOLCHAIN_REVISION` to a reviewed, full 40-character
   commit SHA from that repository. The commit must include package support.
5. Protect your library repository's `lana-v*` tags.

To publish version `1.2.3`, push the protected tag `lana-v1.2.3`.
The tag version must match the version in `lana.toml`.

## 5. Test Lana's package support.

These commands are for contributors testing Lana itself. Library authors use the publication workflow in section 4.

```sh
cargo build --locked -p lana-cli --features package-test-origin,publication-fault-injection
python3 tests/test_packages.py target/debug/lana --network-fixture --fault-injection
cargo test --locked -p lana-cli packages::tests::public_release_smoke -- --ignored --nocapture
cargo build --locked -p lana-cli
```

These test packaging, simulated downloads, and deliberately triggered failures. Special test features enable those simulations; normal builds exclude them.