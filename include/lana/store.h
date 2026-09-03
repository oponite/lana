#ifndef LANA_STORE_H
#define LANA_STORE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "lana/error.h"
#include "lana/state_codec.h"
#include "lana/value.h"
#include "lana/vm.h"

/* Opaque handles */
typedef struct LanaStore LanaStore;
typedef struct LanaStoreRevision LanaStoreRevision;

typedef struct {
    size_t struct_size;
    uint32_t schema_version;
    const char *path;
    uint32_t timeout_ms;
} LanaStoreOptions;

typedef struct {
    uint64_t revision_id;
    uint32_t schema_version;
    uint64_t timestamp;
    unsigned char digest[32];
} LanaStoreRevisionInfo;

/* Public API */
LanaError lana_store_open(const LanaStoreOptions *options, LanaStore **out_store);
LanaError lana_store_close(LanaStore *store);

LanaError lana_store_get_path(LanaStore *store, char **out_path);
LanaError lana_store_get(LanaStore *store, LanaVM *vm, const char *key, Value *out_value);
LanaError lana_store_put(LanaStore *store, const char *key, Value value);
LanaError lana_store_delete(LanaStore *store, const char *key);

LanaError lana_store_commit(LanaStore *store, LanaStoreRevisionInfo *out_revision);
LanaError lana_store_current_revision(LanaStore *store, LanaStoreRevisionInfo *out_revision);

LanaError lana_store_get_at(LanaStore *store, LanaVM *vm, uint64_t revision, const char *key, Value *out_value);
LanaError lana_store_put_persistent_state(LanaStore *store, const char *key,
                                          const LanaPersistentState *state);
LanaError lana_store_get_persistent_state(LanaStore *store, LanaVM *vm,
                                          const char *key, LanaPersistentState *out_state);
LanaError lana_store_get_persistent_state_at(LanaStore *store, LanaVM *vm,
                                             uint64_t revision, const char *key,
                                             LanaPersistentState *out_state);

typedef struct {
    uint64_t revision;
    char *key;
    Value value;
} LanaStoreHistoryRecord;

typedef struct {
    LanaStoreHistoryRecord *records;
    size_t count;
} LanaStoreHistory;

LanaError lana_store_history(LanaStore *store, const char *key, LanaStoreHistory *out_history);

typedef struct {
    char *key;
    Value value;
} LanaStoreScanRecord;

/* Scan committed keys with the given prefix, sorted by key (deterministic). */
LanaError lana_store_scan(LanaStore *store, LanaVM *vm, const char *prefix,
                          LanaStoreScanRecord **out_records, size_t *out_count);
void lana_store_scan_free(LanaStoreScanRecord *records, size_t count);

LanaError lana_store_snapshot(LanaStore *store, LanaVM *vm, Value *out_value, LanaStoreRevisionInfo *out_revision);
LanaError lana_store_compact(LanaStore *store, uint64_t retention, LanaStoreRevisionInfo *out_revision);

#endif
