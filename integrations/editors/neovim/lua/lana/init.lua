local M = {}

local function root_dir(buffer)
    local name = vim.api.nvim_buf_get_name(buffer)
    local markers = vim.fs.find({ "lana.toml", ".git" }, {
        path = name,
        upward = true,
        stop = vim.env.HOME,
    })
    if #markers > 0 then
        return vim.fs.dirname(markers[1])
    end
    return vim.fs.dirname(name)
end

function M.setup(options)
    options = options or {}
    local command = options.cmd or { "lana", "lsp" }
    local compatibility_checked = false
    local compatible = false
    vim.filetype.add({ extension = { lana = "lana" } })
    vim.api.nvim_create_autocmd("FileType", {
        pattern = "lana",
        callback = function(event)
            if not compatibility_checked then
                local output = vim.fn.system({ command[1], "version" })
                compatible = vim.v.shell_error == 0
                    and output:match("^Lana 1%.0%.%d+ %(LABC v1,") ~= nil
                compatibility_checked = true
            end
            if not compatible then
                vim.notify(
                    "Lana integration requires Lana 1.0.x with LABC v1: "
                        .. command[1],
                    vim.log.levels.ERROR
                )
                return
            end
            vim.lsp.start({
                name = "lana-lsp",
                cmd = command,
                root_dir = root_dir(event.buf),
            }, { bufnr = event.buf })
        end,
    })
end

return M
