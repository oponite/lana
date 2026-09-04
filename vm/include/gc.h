#ifndef LANA_GC_H
#define LANA_GC_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct LanaGC LanaGC;
typedef struct LanaGCAllocation LanaGCAllocation;

typedef enum {
    LANA_GC_OPAQUE = 0,
    LANA_GC_STRING,
    LANA_GC_VALUE,
    LANA_GC_VALUE_ARRAY,
    LANA_GC_ARRAY,
    LANA_GC_MAP,
    LANA_GC_JOINT,
    LANA_GC_POSSIBILITY,
    LANA_GC_PATH_SET,
    LANA_GC_STATE_DIST,
    LANA_GC_ADT,
    LANA_GC_DERIVATION,
    LANA_GC_REACTIVE,
    LANA_GC_REACTIVE_HISTORY,
    LANA_GC_CLAIM,
    LANA_GC_PLANNED_EFFECT,
    LANA_GC_EFFECT_RECEIPT,
    LANA_GC_RUNTIME_INTERNAL
} LanaGCObjectKind;

typedef enum {
    LANA_GC_OWNER_VM = 0,
    LANA_GC_OWNER_NATIVE,
    LANA_GC_OWNER_TASK,
    LANA_GC_OWNER_SHARED
} LanaGCOwnership;

typedef enum {
    LANA_GC_GENERATION_YOUNG = 0,
    LANA_GC_GENERATION_OLD,
    LANA_GC_GENERATION_SHARED
} LanaGCGeneration;

typedef void (*LanaGCTraceFn)(LanaGC *gc, void *payload);
typedef void (*LanaGCRootsFn)(LanaGC *gc, void *context);

typedef struct {
    void *root;
    LanaGCTraceFn trace;
} LanaGCRoot;

struct LanaGC {
    LanaGCAllocation *allocations;
    LanaGCAllocation *native_allocations;
    LanaGCAllocation **allocation_index;
    size_t allocation_index_count;
    size_t allocation_index_capacity;
    size_t allocation_index_tombstones;
    LanaGCRoot *native_roots;
    size_t native_root_count;
    size_t native_root_capacity;
    LanaGCAllocation **worklist;
    size_t worklist_count;
    size_t worklist_capacity;
    size_t worklist_cursor;
    size_t allocated_bytes;
    size_t memory_limit;
    size_t collection_threshold;
    size_t last_reclaimed_bytes;
    size_t last_reclaimed_objects;
    uint64_t allocation_count;
    uint64_t collection_count;
    uint64_t reclaimed_objects;
    uint64_t reclaimed_bytes;
    uint64_t minor_collection_count;
    uint64_t incremental_step_count;
    size_t promoted_objects;
    size_t remembered_count;
    LanaGCRootsFn trace_roots;
    void *roots_context;
    bool collecting;
    bool mark_failed;
    bool deferred;
};

void lana_gc_init(LanaGC *gc, size_t memory_limit, size_t collection_threshold,
                  LanaGCRootsFn trace_roots, void *roots_context);
void lana_gc_free(LanaGC *gc);
void *lana_gc_alloc(LanaGC *gc, size_t size, LanaGCObjectKind kind,
                    LanaGCOwnership ownership, LanaGCTraceFn trace);
bool lana_gc_publish(LanaGC *gc, void *payload);
bool lana_gc_configure(LanaGC *gc, void *payload, LanaGCObjectKind kind,
                       LanaGCOwnership ownership, LanaGCTraceFn trace);
bool lana_gc_mark(LanaGC *gc, void *payload);
size_t lana_gc_root_push(LanaGC *gc, void *root, LanaGCTraceFn trace);
void lana_gc_root_pop(LanaGC *gc, size_t previous_count);
bool lana_gc_collect(LanaGC *gc);
bool lana_gc_collect_young(LanaGC *gc);
bool lana_gc_incremental_begin(LanaGC *gc);
bool lana_gc_incremental_step(LanaGC *gc, size_t budget, bool *complete);
bool lana_gc_promote(LanaGC *gc, void *payload, bool shared);
bool lana_gc_write_barrier(LanaGC *gc, void *owner, void *referenced);
void lana_gc_set_deferred(LanaGC *gc, bool deferred);
void lana_gc_release_native(LanaGC *gc);
size_t lana_gc_payload_size(const LanaGC *gc, const void *payload);
LanaGCObjectKind lana_gc_object_kind(const LanaGC *gc, const void *payload);
LanaGCGeneration lana_gc_generation(const LanaGC *gc, const void *payload);

#endif
