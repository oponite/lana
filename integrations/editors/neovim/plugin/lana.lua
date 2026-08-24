if vim.g.loaded_lana_integration == 1 then
    return
end
vim.g.loaded_lana_integration = 1

require("lana").setup()
