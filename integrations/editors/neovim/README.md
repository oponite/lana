# Lana for Neovim

Add this directory to Neovim's runtime path with your plugin manager. It
registers `.lana` and starts `lana lsp`, using `lana.toml`, then `.git`, as the
workspace root.

To use a nonstandard executable path, load only the Lua module and configure it:

```lua
require("lana").setup({ cmd = { "/absolute/path/to/lana", "lsp" } })
```

The current Lana 4.1 server diagnoses saved files and unsaved buffers.
It also provides hover, completion, definition, references, and rename.
The plugin checks that the configured executable reports Lana 3.x or 4.x with
LABC v2-v6 before starting the server.

Run the version-gate regression from the repository root:
`lua integrations/editors/neovim/test.lua`.

Workspace definition, references, and rename follow compiler declaration identity
across imports and scopes. Unsaved imported buffers participate in analysis;
positions use UTF-16. Rename returns all edits together or an error. Dependency
sources are navigable and read-only. Invalid workspace sources must be repaired
before a workspace rename can be validated.
