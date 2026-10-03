-- Run from the repository root: lua integrations/editors/neovim/test.lua
package.path = "integrations/editors/neovim/lua/?.lua;integrations/editors/neovim/lua/?/init.lua;" .. package.path
local plugin = require("lana")
local callback, output, status, starts, errors, probes
vim = {
    api = {
        nvim_buf_get_name = function() return "/project/main.lana" end,
        nvim_create_autocmd = function(_, options) callback = options.callback end,
    },
    fs = { find = function() return {} end, dirname = function() return "/project" end },
    env = {},
    filetype = { add = function() end },
    fn = { system = function(command)
        assert(command[1] == "/custom/lana" and command[2] == "version")
        probes = probes + 1
        vim.v.shell_error = status
        return output
    end },
    v = {},
    log = { levels = { ERROR = 1 } },
    notify = function() errors = errors + 1 end,
    lsp = { start = function(config, options)
        assert(config.cmd[1] == "/custom/lana" and config.cmd[2] == "lsp")
        assert(config.root_dir == "/project" and options.bufnr == 7)
        starts = starts + 1
    end },
}
local function check(version, exit_code, accepted)
    output, status, starts, errors, probes = version, exit_code, 0, 0, 0
    plugin.setup({ cmd = { "/custom/lana", "lsp" } })
    callback({ buf = 7 })
    callback({ buf = 7 })
    assert(probes == 1)
    assert(starts == (accepted and 2 or 0), version)
    assert(errors == (accepted and 0 or 2), version)
end
for _, major in ipairs({ 3, 4 }) do
    for labc = 2, 6 do
        check("Lana " .. major .. ".0.12 (LABC v" .. labc .. ", Rust VM, native compiler)\n", 0, true)
    end
end
for _, version in ipairs({
    "Lana 2.0.0 (LABC v2,", "Lana 5.0.0 (LABC v2,",
    "Lana 4.1.0 (LABC v1,", "Lana 4.1.0 (LABC v7,",
    "Lana 4.1.0 (LABC v20,", "Lana 4x0x0 (LABC v2,", "", "unrelated output",
}) do check(version, 0, false) end
check("Lana 4.1.0 (LABC v2, Rust VM, native compiler)", 1, false)
print("Neovim compatibility checks passed")
