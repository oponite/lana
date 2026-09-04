#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "adapters.h"
#include "data.h"
#include "vm.h"

#ifdef LANA_ADAPTER_SQLITE_AVAILABLE
#include <sqlite3.h>
#endif

static void test_json_adapter(void) {
    LanaVM vm;
    LanaAdapterOptions options = {sizeof(options), 1u, ADAPTER_JSON, NULL};
    void *adapter = NULL;
    Value out;
    lana_vm_init(&vm, NULL);
    assert(lana_adapter_load(&options, &adapter) == LANA_OK);
    assert(lana_adapter_fetch(adapter, &vm, "{\"p\":0.9,\"ok\":true}", &out) == LANA_OK);
    assert(out.type == VAL_MAP);
    {
        Value p;
        assert(lana_map_get(out.as.map, "p", &p) == LANA_OK);
        assert(p.type == VAL_NUMBER && p.as.number == 0.9);
    }
    /* Malformed JSON surfaces the parse error. */
    assert(lana_adapter_fetch(adapter, &vm, "{not-json", &out) == LANA_ERR_PARSE);
    lana_adapter_close(adapter);
    lana_vm_free(&vm);
    printf("Pass.\n");
}

static void test_csv_adapter(void) {
    char path[] = "/tmp/lana-adapter-csv-XXXXXX";
    char file[256];
    LanaVM vm;
    LanaAdapterOptions options = {sizeof(options), 1u, ADAPTER_CSV, NULL};
    void *adapter = NULL;
    Value out;
    assert(mkdtemp(path) != NULL);
    (void)snprintf(file, sizeof(file), "%s/data.csv", path);
    {
        FILE *f = fopen(file, "w");
        assert(f != NULL);
        fprintf(f, "sensor,reading\ns1,0.8\ns2,0.6\n");
        fclose(f);
    }
    lana_vm_init(&vm, NULL);
    assert(lana_adapter_load(&options, &adapter) == LANA_OK);
    assert(lana_adapter_fetch(adapter, &vm, file, &out) == LANA_OK);
    assert(out.type == VAL_ARRAY && out.as.array->count == 2u);
    {
        Value row = out.as.array->items[0];
        Value reading;
        assert(row.type == VAL_MAP);
        assert(lana_map_get(row.as.map, "reading", &reading) == LANA_OK);
        assert(reading.type == VAL_STRING && strcmp(reading.as.string, "0.8") == 0);
    }
    lana_adapter_close(adapter);
    lana_vm_free(&vm);
    (void)unlink(file);
    (void)rmdir(path);
    printf("Pass.\n");
}

#ifdef LANA_ADAPTER_SQLITE_AVAILABLE
static void test_sqlite_adapter(void) {
    char path[] = "/tmp/lana-adapter-sqlite-XXXXXX";
    char db_path[256];
    LanaVM vm;
    LanaAdapterOptions options = {sizeof(options), 1u, ADAPTER_SQLITE, NULL};
    void *adapter = NULL;
    Value out;
    sqlite3 *db = NULL;
    assert(mkdtemp(path) != NULL);
    (void)snprintf(db_path, sizeof(db_path), "%s/evidence.db", path);
    assert(sqlite3_open(db_path, &db) == SQLITE_OK);
    assert(sqlite3_exec(db, "CREATE TABLE readings (sensor TEXT, reading REAL);"
                            "INSERT INTO readings VALUES ('s1', 0.8);"
                            "INSERT INTO readings VALUES ('s2', NULL);", NULL, NULL, NULL) == SQLITE_OK);
    sqlite3_close(db);

    lana_vm_init(&vm, NULL);
    options.config = db_path;
    assert(lana_adapter_load(&options, &adapter) == LANA_OK);
    assert(lana_adapter_fetch(adapter, &vm, "SELECT sensor, reading FROM readings WHERE sensor='s1'", &out) == LANA_OK);
    assert(out.type == VAL_MAP);
    {
        Value reading;
        assert(lana_map_get(out.as.map, "reading", &reading) == LANA_OK);
        assert(reading.type == VAL_NUMBER && reading.as.number == 0.8);
    }
    /* NULL cell decodes to VAL_NULL. */
    assert(lana_adapter_fetch(adapter, &vm, "SELECT sensor, reading FROM readings WHERE sensor='s2'", &out) == LANA_OK);
    {
        Value reading;
        assert(lana_map_get(out.as.map, "reading", &reading) == LANA_OK);
        assert(reading.type == VAL_NULL);
    }
    /* No row -> VAL_NULL. */
    assert(lana_adapter_fetch(adapter, &vm, "SELECT sensor FROM readings WHERE sensor='nope'", &out) == LANA_OK);
    assert(out.type == VAL_NULL);
    lana_adapter_close(adapter);
    lana_vm_free(&vm);
    (void)unlink(db_path);
    (void)rmdir(path);
    printf("Pass.\n");
}
#endif

static void test_malformed(void) {
    LanaVM vm;
    LanaAdapterOptions options = {sizeof(options), 1u, ADAPTER_JSON, NULL};
    void *adapter = NULL;
    Value out;
    lana_vm_init(&vm, NULL);
    assert(lana_adapter_load(NULL, &adapter) == LANA_ERR_INVALID_STATE);
    assert(lana_adapter_load(&options, NULL) == LANA_ERR_INVALID_STATE);
    options.struct_size = sizeof(options) - 1u;
    assert(lana_adapter_load(&options, &adapter) == LANA_ERR_INVALID_STATE);
    options.struct_size = sizeof(options);
    options.schema_version = 2u;
    assert(lana_adapter_load(&options, &adapter) == LANA_ERR_INVALID_STATE);
    options.schema_version = 1u;
    options.kind = (LanaAdapterKind)99;
    assert(lana_adapter_load(&options, &adapter) == LANA_ERR_UNSUPPORTED_OPERATION);
    options.kind = ADAPTER_JSON;
    assert(lana_adapter_load(&options, &adapter) == LANA_OK);
    assert(lana_adapter_fetch(adapter, NULL, "{}", &out) == LANA_ERR_INVALID_STATE);
    assert(lana_adapter_fetch(adapter, &vm, NULL, &out) == LANA_ERR_INVALID_STATE);
    assert(lana_adapter_fetch(adapter, &vm, "{}", NULL) == LANA_ERR_INVALID_STATE);
    lana_adapter_close(adapter);
    lana_adapter_close(NULL); /* NULL-safe */
    lana_vm_free(&vm);
    printf("Pass.\n");
}

int main(void) {
    test_json_adapter();
    test_csv_adapter();
#ifdef LANA_ADAPTER_SQLITE_AVAILABLE
    test_sqlite_adapter();
#else
    printf("SQLite adapter not built; skipping.\n");
#endif
    test_malformed();
    return 0;
}
