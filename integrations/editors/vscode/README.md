# Lana Language Support for VS Code

This source-install extension launches `lana lsp` for `.lana` files.

```bash
cd integrations/editors/vscode
npm install
npm run compile
```

Run the extension from VS Code's Extension Development Host or build a local
VSIX with `npm run package`. Set `lana.server.path` when `lana` is not on PATH.

The Lana 1.0 server currently diagnoses files saved on disk. Unsaved-buffer
diagnostics are not promised by this integration.
