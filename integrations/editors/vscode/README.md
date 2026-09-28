# Lana Language Support for VS Code

This source-install extension launches `lana lsp` for `.lana` files.

```bash
cd integrations/editors/vscode
npm install
npm test
npm run package
```

Install `lana-language-support-4.0.0.vsix` using VS Code's **Extensions: Install
from VSIX...** command, then open a `.lana` file. Syntax highlighting is included.
Set `lana.server.path` to the absolute path of your built executable (for example,
`/path/to/lana/target/lana/bin/lana`) when `lana` is not on PATH. Build that executable
from the repository root with `python3 tools/build.py build`.

The Lana 4.0 server diagnoses both saved files and unsaved buffers, and
provides hover, completion, go-to-definition, find-references, and rename via
the in-process compiler service.

Workspace definition, references, and rename follow compiler declaration identity
across imports and scopes. Unsaved imported buffers participate in analysis;
positions use UTF-16. Rename returns all edits together or an error. Dependency
sources are navigable and read-only. Invalid workspace sources must be repaired
before a workspace rename can be validated.
