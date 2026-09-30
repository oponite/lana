# Use Lana in an editor

The [VS Code extension](vscode/README.md) and [Neovim plugin](neovim/README.md)
connect to the language server in the Lana executable.
Both adapters accept Lana 3.x or 4.x reporting LABC v2–v6.
Features depend on the selected executable.

The current Lana 4.0 server provides these features:

- Diagnostics for saved files and unsaved buffers
- Hover information and completion
- Go-to-definition and find-references
- Rename across workspace imports and scopes.

The adapters start `lana lsp` and exchange Language Server Protocol messages
over standard input/output. The server analyzes code through the compiler.
It does not run the program.

Other LSP-capable editors can start the same command for `.lana` files:

```bash
lana lsp
```

Use `lana.toml` or the repository root as the workspace root.
Unsaved imported buffers participate in analysis. Positions use UTF-16.
Dependency sources permit navigation but remain read-only.
Rename returns all edits together or an error.
Invalid workspace sources must be repaired before a workspace rename can succeed.
