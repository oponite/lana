# Release installers

1. Generate `install.sh` and `install.ps1` with
   `python3 tools/build.py installers --output target/installers`. The renderer
   embeds `VERSION`; both scripts fetch that exact release's archive and
   `SHA256SUMS`. The latest-release URLs select the script once, so a concurrent
   release cannot mix an archive with another version's checksum. Do not publish
   the templates from `tools/installers` directly.
2. Install into `~/.local/share/lana` on macOS/Linux or
   `%LOCALAPPDATA%\Programs\Lana` on Windows. The layout keeps the executable and
   compiler in `bin`, stdlib in `share/lana/stdlib`, and license in
   `share/doc/lana`. A `.lana-install` marker identifies installer-owned bundles.
   Existing unmarked directories and linked install roots are refused.
3. Validate the selected checksum, archive paths/types, required files, version,
   and a source program importing `std/core` before replacing an installation.
   Download, extraction, or smoke-test failures leave the previous installation
   intact. Replacement uses a sibling staging directory and backup; startup-file
   or user PATH failures restore the old bundle and configuration. Concurrent
   installs fail on an exclusive lock. Windows refuses upgrades while files in
   the installed bundle are locked; close those programs and retry.
4. Configure macOS/Linux Zsh through `${ZDOTDIR:-$HOME}/.zshrc`, or Bash through
   the first existing `.bash_profile`/`.bash_login`, or `.profile` when neither
   exists, plus `.bashrc`. Existing contents and permissions are retained;
   profile symlinks are followed to their target. One marked PATH block is added
   to each file. Modified or malformed existing Lana blocks are refused. Other
   shells receive manual instructions. A child `sh` cannot update the calling
   terminal, so open a new terminal or run the printed export command.
5. Configure the Windows user PATH and current PowerShell process. Existing
   PATH entries, variable references, and the registry value type are preserved;
   reinstalling does not duplicate the Lana entry.
   Restart other terminals to inherit the change. The installer does not modify
   machine PATH or execution policy. Test it with Windows PowerShell 5.1 and
   PowerShell 7 on each native architecture.

## Verification and release

`tests/test_installers.py` exercises generated scripts with real bundles and a
fixture transport. It covers repeated installation, simulated older bundles,
checksum/download/archive failures, missing assets, unsafe paths, rollback,
unrelated directories, lock contention, and shell configuration. It runs in a
temporary home; Windows tests save and restore the account's user PATH.

CI and the release workflow share native Linux and Windows x64/ARM64 jobs. These execute
Rust workspace tests, the compiler bootstrap, generated-project workflow,
packaging, and installation checks. Release jobs verify the macOS universal
bundle and both slices. All five platform archives and both rendered installers
are included in `SHA256SUMS`; publication retries compare existing assets and
never overwrite a mismatched release. After publication, native jobs execute
the published installers and validate their installed bundles.

Preserve the normal release gates. Installer tests are not proof that all
optional integrations work on Windows. Publishing a release, signing,
notarization, and changing repository protection are separate operations.

## Interrupted installation

Normal errors roll back automatically. A killed process or failed rollback can
leave an installer lock, a sibling `lana.previous`/`Lana.previous`, and staging
files. The installer refuses to guess which copy should be deleted.

1. Confirm that no installer is still running.
2. Inspect the installed bundle, its `.previous` sibling, and any recovery path
   printed by the installer. If the installation is missing, restore the previous
   bundle by renaming the `.previous` directory to its original name.
3. Restore any shell-profile originals retained in the reported staging folder
   if rollback reported a failure. On macOS, `profile.N.path` identifies the
   target and `profile.N.original` contains its previous contents.
4. Verify the restored `bin/lana` or `bin/lana.exe` with `version` and a source
   program before removing an obsolete `.previous` bundle, leftover staging
   files, and `.lana-install.lock`.
5. Retry the installer only after recovery is complete.

The scripts execute release-provided code. Checksums detect corrupted or mixed
downloads; they do not protect against a compromised release account. Releases
remain unsigned and may trigger operating-system security warnings.

## Linux compatibility

Linux x64 and ARM64 archives are built and tested on Ubuntu 22.04 with glibc
2.35. They require glibc 2.35 or newer; musl-based distributions such as Alpine
are not supported by these binaries. SQLite is bundled. The installer uses
`sha256sum` on Linux and `shasum` on macOS. Bash and Zsh receive PATH setup;
other shells receive a command to add the installed `bin` directory manually.

## Switching installation methods

The installer does not delete Homebrew or source-built installations. After
opening a new terminal, use `command -v lana` and `lana version` to confirm
which copy runs. If switching from Homebrew, first install and verify the new
copy, then optionally run `brew uninstall lana` to remove the Homebrew copy.
For a source install, inspect the old prefix before removing any files.
