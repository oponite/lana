#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "store.h"
#include "state_codec.h"

static void cleanup_store(const char *path) {
    char file[512];
    (void)snprintf(file, sizeof(file), "%s/journal", path); (void)unlink(file);
    (void)snprintf(file, sizeof(file), "%s/manifest", path); (void)unlink(file);
    (void)snprintf(file, sizeof(file), "%s/snapshot-2", path); (void)unlink(file);
    (void)snprintf(file, sizeof(file), "%s/snapshot-1", path); (void)unlink(file);
    (void)rmdir(path);
}

static LanaStore *open_store(const char *path) {
    LanaStoreOptions options = {
        .struct_size = sizeof(LanaStoreOptions), .schema_version = 1u,
        .path = path, .timeout_ms = 1000u
    };
    LanaStore *store = NULL;
    assert(lana_store_open(&options, &store) == LANA_OK);
    return store;
}

static void test_recovery_and_history(void) {
    char path[] = "/tmp/lana-store-XXXXXX";
    LanaStoreRevisionInfo revision;
    LanaStoreHistory history;
    LanaStore *store;
    LanaVM vm;
    Value value;

    assert(mkdtemp(path) != NULL);
    lana_vm_init(&vm, NULL);
    store = open_store(path);
    assert(lana_store_put(store, "answer", lana_value_number(41.0)) == LANA_OK);
    assert(lana_store_commit(store, &revision) == LANA_OK);
    assert(revision.revision_id == 1u);
    assert(lana_store_put(store, "answer", lana_value_number(42.0)) == LANA_OK);
    assert(lana_store_commit(store, &revision) == LANA_OK);
    assert(lana_store_close(store) == LANA_OK);

    store = open_store(path);
    assert(lana_store_get(store, &vm, "answer", &value) == LANA_OK);
    assert(value.type == VAL_NUMBER && value.as.number == 42.0);
    assert(lana_store_get_at(store, &vm, 1u, "answer", &value) == LANA_OK);
    assert(value.type == VAL_NUMBER && value.as.number == 41.0);
    assert(lana_store_history(store, "answer", &history) == LANA_OK);
    assert(history.count == 2u);
    assert(history.records[0].revision == 1u);
    assert(history.records[1].revision == 2u);
    lana_store_history_free(&history);
    assert(lana_store_snapshot(store, &vm, &value, &revision) == LANA_OK);
    assert(value.type == VAL_MAP && revision.revision_id == 2u);
    assert(lana_store_compact(store, 0u, &revision) == LANA_OK);
    {
        char journal[512];
        struct stat status;
        (void)snprintf(journal, sizeof(journal), "%s/journal", path);
        assert(stat(journal, &status) == 0 && status.st_size == 0);
    }
    assert(lana_store_get_at(store, &vm, 1u, "answer", &value) ==
           LANA_ERR_COMPACTED_HISTORY);
    assert(lana_store_get_at(store, &vm, 2u, "answer", &value) == LANA_OK);
    assert(lana_store_close(store) == LANA_OK);
    store = open_store(path);
    assert(lana_store_get(store, &vm, "answer", &value) == LANA_OK);
    assert(value.type == VAL_NUMBER && value.as.number == 42.0);
    assert(lana_store_get_at(store, &vm, 1u, "answer", &value) ==
           LANA_ERR_COMPACTED_HISTORY);
    assert(lana_store_close(store) == LANA_OK);
    lana_vm_free(&vm);
    cleanup_store(path);
}

static void test_trailing_partial_revision_is_ignored(void) {
    char path[] = "/tmp/lana-store-partial-XXXXXX";
    char journal[512];
    LanaStore *store;
    LanaVM vm;
    Value value;
    int file;

    assert(mkdtemp(path) != NULL);
    lana_vm_init(&vm, NULL);
    store = open_store(path);
    assert(lana_store_put(store, "key", lana_value_bool(true)) == LANA_OK);
    assert(lana_store_commit(store, NULL) == LANA_OK);
    assert(lana_store_close(store) == LANA_OK);
    (void)snprintf(journal, sizeof(journal), "%s/journal", path);
    file = open(journal, O_WRONLY | O_APPEND);
    assert(file >= 0);
    assert(write(file, "LREV", 4u) == 4);
    assert(close(file) == 0);

    store = open_store(path);
    assert(lana_store_get(store, &vm, "key", &value) == LANA_OK);
    assert(value.type == VAL_BOOL && value.as.boolean);
    assert(lana_store_close(store) == LANA_OK);
    lana_vm_free(&vm);
    cleanup_store(path);
}

static void test_corrupt_snapshot_fails_open(void) {
    char path[] = "/tmp/lana-store-corrupt-XXXXXX";
    char snapshot[512];
    LanaStoreOptions options = {
        .struct_size = sizeof(LanaStoreOptions), .schema_version = 1u,
        .path = path, .timeout_ms = 1000u
    };
    LanaStore *store;
    LanaVM vm;
    Value value;
    int file;
    unsigned char byte;

    assert(mkdtemp(path) != NULL);
    lana_vm_init(&vm, NULL);
    store = open_store(path);
    assert(lana_store_put(store, "key", lana_value_number(1.0)) == LANA_OK);
    assert(lana_store_commit(store, NULL) == LANA_OK);
    assert(lana_store_snapshot(store, &vm, &value, NULL) == LANA_OK);
    assert(lana_store_close(store) == LANA_OK);
    (void)snprintf(snapshot, sizeof(snapshot), "%s/snapshot-1", path);
    file = open(snapshot, O_RDWR);
    assert(file >= 0);
    assert(pread(file, &byte, 1u, 52) == 1);
    byte ^= 0x01u;
    assert(pwrite(file, &byte, 1u, 52) == 1);
    assert(close(file) == 0);
    store = NULL;
    assert(lana_store_open(&options, &store) == LANA_ERR_CORRUPTION);
    assert(store == NULL);
    lana_vm_free(&vm);
    cleanup_store(path);
}

static void test_corrupt_committed_journal_fails_open(void) {
    char path[] = "/tmp/lana-store-journal-corrupt-XXXXXX";
    char journal[512];
    LanaStoreOptions options = {
        .struct_size = sizeof(LanaStoreOptions), .schema_version = 1u,
        .path = path, .timeout_ms = 1000u
    };
    LanaStore *store;
    int file;
    unsigned char byte;

    assert(mkdtemp(path) != NULL);
    store = open_store(path);
    assert(lana_store_put(store, "key", lana_value_number(1.0)) == LANA_OK);
    assert(lana_store_commit(store, NULL) == LANA_OK);
    assert(lana_store_close(store) == LANA_OK);
    (void)snprintf(journal, sizeof(journal), "%s/journal", path);
    file = open(journal, O_RDWR);
    assert(file >= 0);
    assert(pread(file, &byte, 1u, 4) == 1);
    byte ^= 0x01u;
    assert(pwrite(file, &byte, 1u, 4) == 1);
    assert(close(file) == 0);
    store = NULL;
    assert(lana_store_open(&options, &store) == LANA_ERR_CORRUPTION);
    assert(store == NULL);
    cleanup_store(path);
}

static void test_persistent_state_current_and_history(void) {
    char path[] = "/tmp/lana-store-state-XXXXXX";
    LanaPersistentState state = {
        .struct_size = sizeof(LanaPersistentState), .schema_version = 1u,
        .state = {.p = 0.4, .d_re = 0.1, .d_im = 0.2},
        .metadata = {.has_source = true, .source = "sensor"}
    };
    LanaPersistentState decoded = {0};
    LanaStore *store;
    LanaVM vm;

    assert(mkdtemp(path) != NULL);
    lana_vm_init(&vm, NULL);
    store = open_store(path);
    assert(lana_store_put_persistent_state(store, "state", &state) == LANA_OK);
    assert(lana_store_commit(store, NULL) == LANA_OK);
    state.state.p = 0.6;
    assert(lana_store_put_persistent_state(store, "state", &state) == LANA_OK);
    assert(lana_store_commit(store, NULL) == LANA_OK);
    assert(lana_store_close(store) == LANA_OK);
    store = open_store(path);
    assert(lana_store_get_persistent_state(store, &vm, "state", &decoded) == LANA_OK);
    assert(decoded.state.p == 0.6 && strcmp(decoded.metadata.source, "sensor") == 0);
    lana_persistent_state_free(&decoded);
    assert(lana_store_get_persistent_state_at(store, &vm, 1u, "state", &decoded) == LANA_OK);
    assert(decoded.state.p == 0.4);
    lana_persistent_state_free(&decoded);
    assert(lana_store_close(store) == LANA_OK);
    lana_vm_free(&vm);
    cleanup_store(path);
}

static void test_scan_prefix(void) {
    char path[] = "/tmp/lana-store-scan-XXXXXX";
    LanaStore *store;
    LanaVM vm;
    LanaStoreScanRecord *records = NULL;
    size_t count = 0u;

    assert(mkdtemp(path) != NULL);
    lana_vm_init(&vm, NULL);
    store = open_store(path);
    assert(lana_store_put(store, "event/1", lana_value_string("{\"a\":1}")) == LANA_OK);
    assert(lana_store_put(store, "event/2", lana_value_string("{\"a\":2}")) == LANA_OK);
    assert(lana_store_put(store, "decision/7", lana_value_string("{\"d\":7}")) == LANA_OK);
    assert(lana_store_put(store, "policy/health/x", lana_value_string("{\"p\":1}")) == LANA_OK);
    assert(lana_store_commit(store, NULL) == LANA_OK);

    /* Prefix scan returns only matching keys, sorted deterministically. */
    assert(lana_store_scan(store, &vm, "event/", &records, &count) == LANA_OK);
    assert(count == 2u);
    assert(strcmp(records[0].key, "event/1") == 0 && records[0].value.type == VAL_STRING);
    assert(strcmp(records[1].key, "event/2") == 0);
    lana_store_scan_free(records, count); records = NULL; count = 0u;

    assert(lana_store_scan(store, &vm, "decision/", &records, &count) == LANA_OK);
    assert(count == 1u && strcmp(records[0].key, "decision/7") == 0);
    lana_store_scan_free(records, count); records = NULL; count = 0u;

    /* No matches: OK with zero count. */
    assert(lana_store_scan(store, &vm, "nope/", &records, &count) == LANA_OK);
    assert(count == 0u && records == NULL);

    /* Malformed arguments. */
    assert(lana_store_scan(NULL, &vm, "event/", &records, &count) == LANA_ERR_INVALID_STATE);
    assert(lana_store_scan(store, &vm, NULL, &records, &count) == LANA_ERR_INVALID_STATE);
    assert(lana_store_scan(store, &vm, "event/", NULL, &count) == LANA_ERR_INVALID_STATE);

    assert(lana_store_close(store) == LANA_OK);
    lana_vm_free(&vm);
    cleanup_store(path);
    printf("Pass.\n");
}

int main(void) {
    test_recovery_and_history();
    test_trailing_partial_revision_is_ignored();
    test_corrupt_snapshot_fails_open();
    test_corrupt_committed_journal_fails_open();
    test_persistent_state_current_and_history();
    test_scan_prefix();
    return 0;
}
