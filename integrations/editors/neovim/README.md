# Lana for Neovim

Add this directory to Neovim's runtime path with your plugin manager. It
registers `.lana` and starts `lana lsp`, using `lana.toml`, then `.git`, as the
workspace root.

To use a nonstandard executable path, load only the Lua module and configure it:

```lua
require("lana").setup({ cmd = { "/absolute/path/to/lana", "lsp" } })
```

Diagnostics reflect files saved on disk.
The plugin checks that the configured executable reports Lana 2.0 with LABC
v2 before starting the server.
