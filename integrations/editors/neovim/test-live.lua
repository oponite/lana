-- nvim --headless -u NONE -l integrations/editors/neovim/test-live.lua
local root = vim.fn.getcwd()
package.path = root .. "/integrations/editors/neovim/lua/?.lua;" .. root .. "/integrations/editors/neovim/lua/?/init.lua;" .. package.path
local work = vim.fn.tempname()
vim.fn.mkdir(work, "p")
work = vim.uv.fs_realpath(work)
local main = work .. "/main.lana"
local dependency = work .. "/math.lana"
vim.fn.writefile({ "fn twice(value) { return value * 2; }" }, dependency)
vim.fn.writefile({ 'import "./math.lana" as math;', 'let answer = math.twice(2);' }, main)
local ok, error = pcall(function()
    require("lana").setup({ cmd = { assert(vim.env.LANA_CLI), "lsp" } })
    vim.cmd("filetype on")
    vim.cmd.edit(main)
    local buffer = vim.api.nvim_get_current_buf()
    assert(vim.wait(15000, function()
        local clients = vim.lsp.get_clients({ bufnr = buffer, name = "lana-lsp" })
        return #clients == 1 and clients[1].initialized
    end, 50), "Lana client did not initialize")
    local client = vim.lsp.get_clients({ bufnr = buffer, name = "lana-lsp" })[1]
    local params = { textDocument = { uri = vim.uri_from_bufnr(buffer) }, position = { line = 1, character = 19 } }
    local result = client:request_sync("textDocument/definition", params, 15000, buffer)
    assert(result and not result.err and result.result[1].uri == vim.uri_from_fname(dependency), vim.inspect(result))
    params.newName = "double_value"
    result = client:request_sync("textDocument/rename", params, 15000, buffer)
    assert(result and not result.err and result.result.changes[vim.uri_from_fname(dependency)], vim.inspect(result))
    client:stop()
    assert(vim.wait(5000, function() return client:is_stopped() end, 20), "Lana server did not stop")
    print("NEOVIM_LIVE_PASS")
end)
vim.fn.delete(work, "rf")
if not ok then io.stderr:write(tostring(error) .. "\n"); vim.cmd("cquit 1") end
vim.cmd("qa!")
