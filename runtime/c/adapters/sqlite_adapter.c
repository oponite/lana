#include "adapters.h"
#include "data.h"
#include "vm.h"
#include "adapter_plugin.h"

#include <sqlite3.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    sqlite3 *db;
} SqliteAdapter;

LanaError lana_adapter_plugin_load(const LanaAdapterOptions *options, void **out_ctx) {
    SqliteAdapter *adapter;
    if (options == NULL || out_ctx == NULL || options->config == NULL) return LANA_ERR_SCHEMA;
    adapter = calloc(1u, sizeof(*adapter));
    if (adapter == NULL) return LANA_ERR_OOM;
    if (sqlite3_open_v2(options->config, &adapter->db, SQLITE_OPEN_READONLY, NULL) != SQLITE_OK) {
        sqlite3_close(adapter->db);
        free(adapter);
        return LANA_ERR_IO;
    }
    *out_ctx = adapter;
    return LANA_OK;
}

LanaError lana_adapter_plugin_fetch(void *ctx, LanaVM *vm, const char *query,
                                    Value *out_value) {
    SqliteAdapter *adapter = (SqliteAdapter *)ctx;
    sqlite3_stmt *stmt;
    LanaMap *map;
    int columns, index;
    LanaError error;
    if (adapter == NULL || vm == NULL || query == NULL || out_value == NULL)
        return LANA_ERR_INVALID_STATE;
    if (sqlite3_prepare_v2(adapter->db, query, -1, &stmt, NULL) != SQLITE_OK)
        return LANA_ERR_PARSE;
    if (sqlite3_step(stmt) != SQLITE_ROW) {
        sqlite3_finalize(stmt);
        *out_value = lana_value_null();
        return LANA_OK;
    }
    columns = sqlite3_column_count(stmt);
    error = lana_map_new(vm, (size_t)columns, &map);
    if (error != LANA_OK) { sqlite3_finalize(stmt); return error; }
    for (index = 0; index < columns; ++index) {
        const char *name = sqlite3_column_name(stmt, index);
        Value value;
        switch (sqlite3_column_type(stmt, index)) {
            case SQLITE_NULL:
                value = lana_value_null();
                break;
            case SQLITE_INTEGER:
            case SQLITE_FLOAT:
                value = lana_value_number(sqlite3_column_double(stmt, index));
                break;
            default: {
                const unsigned char *text = sqlite3_column_text(stmt, index);
                size_t length = text == NULL ? 0u : strlen((const char *)text);
                char *stored = lana_vm_alloc(vm, length + 1u);
                if (stored == NULL) { sqlite3_finalize(stmt); return LANA_ERR_OOM; }
                if (length != 0u) memcpy(stored, text, length);
                stored[length] = '\0';
                value = lana_value_string(stored);
                break;
            }
        }
        error = lana_map_set(vm, map, name, &value, true);
        if (error != LANA_OK) { sqlite3_finalize(stmt); return error; }
    }
    sqlite3_finalize(stmt);
    *out_value = lana_value_map(map);
    return LANA_OK;
}

void lana_adapter_plugin_close(void *ctx) {
    SqliteAdapter *adapter = (SqliteAdapter *)ctx;
    if (adapter == NULL) return;
    if (adapter->db != NULL) sqlite3_close(adapter->db);
    free(adapter);
}
