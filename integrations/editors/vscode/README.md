# Lana Language Support for VS Code

This source-install extension launches `lana lsp` for `.lana` files.

```bash
cd integrations/editors/vscode
npm install
npm run compile
```

Run the extension from VS Code's Extension Development Host or build a local
VSIX with `npm run package`. Set `lana.server.path` when `lana` is not on PATH.

The Lana 1.x server diagnoses both saved files and unsaved buffers, and
provides hover, completion, go-to-definition, find-references, and rename via
the in-process compiler service.
