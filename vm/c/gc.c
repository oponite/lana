#include "gc.h"

#include <stdlib.h>
#include <string.h>

#define LANA_GC_MAGIC UINT32_C(0x4c47434f)

struct LanaGCAllocation {
    uint32_t magic;
    LanaGCObjectKind kind;
    LanaGCOwnership ownership;
    LanaGCTraceFn trace;
    size_t size;
    bool marked;
    bool initialized;
    bool native_pending;
    bool remembered;
    uint8_t age;
    LanaGCGeneration generation;
    LanaGC *gc;
    LanaGCAllocation *next;
    LanaGCAllocation *native_next;
};

static LanaGCAllocation allocation_index_tombstone;

static size_t payload_hash(const void *payload) {
    uintptr_t value = (uintptr_t)payload;
    value ^= value >> 4u;
    value *= (uintptr_t)UINT32_C(2654435761);
    value ^= value >> 16u;
    return (size_t)value;
}

static void *allocation_payload(LanaGCAllocation *allocation) {
    return (void *)(allocation + 1);
}

static const void *allocation_const_payload(const LanaGCAllocation *allocation) {
    return (const void *)(allocation + 1);
}

static LanaGCAllocation *index_lookup(const LanaGC *gc, const void *payload) {
    size_t slot;
    size_t mask;
    if (gc == NULL || payload == NULL || gc->allocation_index_capacity == 0u)
        return NULL;
    mask = gc->allocation_index_capacity - 1u;
    slot = payload_hash(payload) & mask;
    while (gc->allocation_index[slot] != NULL) {
        LanaGCAllocation *allocation = gc->allocation_index[slot];
        if (allocation != &allocation_index_tombstone &&
            allocation_const_payload(allocation) == payload)
            return allocation;
        slot = (slot + 1u) & mask;
    }
    return NULL;
}

static void index_place(LanaGC *gc, LanaGCAllocation *allocation) {
    size_t slot = payload_hash(allocation_const_payload(allocation)) &
                  (gc->allocation_index_capacity - 1u);
    size_t tombstone = SIZE_MAX;
    while (gc->allocation_index[slot] != NULL) {
        if (gc->allocation_index[slot] == &allocation_index_tombstone &&
            tombstone == SIZE_MAX)
            tombstone = slot;
        slot = (slot + 1u) & (gc->allocation_index_capacity - 1u);
    }
    if (tombstone != SIZE_MAX) {
        slot = tombstone;
        --gc->allocation_index_tombstones;
    }
    gc->allocation_index[slot] = allocation;
    ++gc->allocation_index_count;
}

static bool index_reserve(LanaGC *gc, size_t needed) {
    LanaGCAllocation **index;
    LanaGCAllocation *allocation;
    size_t capacity = gc->allocation_index_capacity;
    if (capacity > 0u &&
        gc->allocation_index_tombstones <= capacity - capacity / 4u &&
        needed <= capacity - capacity / 4u - gc->allocation_index_tombstones)
        return true;
    if (capacity == 0u) capacity = 32u;
    while (needed > capacity - capacity / 4u) {
        if (capacity > SIZE_MAX / 2u) return false;
        capacity *= 2u;
    }
    index = calloc(capacity, sizeof(*index));
    if (index == NULL) return false;
    free(gc->allocation_index);
    gc->allocation_index = index;
    gc->allocation_index_capacity = capacity;
    gc->allocation_index_count = 0u;
    gc->allocation_index_tombstones = 0u;
    for (allocation = gc->allocations; allocation != NULL; allocation = allocation->next)
        index_place(gc, allocation);
    return true;
}

static void index_remove(LanaGC *gc, LanaGCAllocation *allocation) {
    size_t slot;
    size_t mask = gc->allocation_index_capacity - 1u;
    const void *payload = allocation_const_payload(allocation);
    slot = payload_hash(payload) & mask;
    while (gc->allocation_index[slot] != NULL) {
        if (gc->allocation_index[slot] == allocation) {
            gc->allocation_index[slot] = &allocation_index_tombstone;
            --gc->allocation_index_count;
            ++gc->allocation_index_tombstones;
            return;
        }
        slot = (slot + 1u) & mask;
    }
}

static void native_add(LanaGC *gc, LanaGCAllocation *allocation) {
    if (allocation->native_pending) return;
    allocation->native_pending = true;
    allocation->native_next = gc->native_allocations;
    gc->native_allocations = allocation;
}

static void native_remove(LanaGC *gc, LanaGCAllocation *allocation) {
    LanaGCAllocation **cursor;
    if (!allocation->native_pending) return;
    cursor = &gc->native_allocations;
    while (*cursor != NULL && *cursor != allocation)
        cursor = &(*cursor)->native_next;
    if (*cursor == allocation) *cursor = allocation->native_next;
    allocation->native_pending = false;
    allocation->native_next = NULL;
}

static LanaGCAllocation *payload_allocation(const LanaGC *gc, const void *payload) {
    LanaGCAllocation *allocation = index_lookup(gc, payload);
    if (allocation == NULL || allocation->magic != LANA_GC_MAGIC ||
        allocation->gc != gc)
        return NULL;
    return allocation;
}

static bool reserve_roots(LanaGC *gc, size_t needed) {
    LanaGCRoot *roots;
    size_t capacity;
    if (needed <= gc->native_root_capacity) return true;
    capacity = gc->native_root_capacity == 0u ? 16u : gc->native_root_capacity;
    while (capacity < needed) {
        if (capacity > SIZE_MAX / 2u) return false;
        capacity *= 2u;
    }
    roots = realloc(gc->native_roots, capacity * sizeof(*roots));
    if (roots == NULL) return false;
    gc->native_roots = roots;
    gc->native_root_capacity = capacity;
    return true;
}

static bool reserve_worklist(LanaGC *gc, size_t needed) {
    LanaGCAllocation **worklist;
    size_t capacity;
    if (needed <= gc->worklist_capacity) return true;
    capacity = gc->worklist_capacity == 0u ? 32u : gc->worklist_capacity;
    while (capacity < needed) {
        if (capacity > SIZE_MAX / 2u) return false;
        capacity *= 2u;
    }
    worklist = realloc(gc->worklist, capacity * sizeof(*worklist));
    if (worklist == NULL) return false;
    gc->worklist = worklist;
    gc->worklist_capacity = capacity;
    return true;
}

static void clear_marks(LanaGC *gc) {
    LanaGCAllocation *allocation;
    for (allocation = gc->allocations; allocation != NULL; allocation = allocation->next)
        allocation->marked = false;
}

static bool mark_roots(LanaGC *gc, bool include_remembered) {
    size_t root_index;
    gc->mark_failed = false;
    gc->worklist_count = 0u;
    gc->worklist_cursor = 0u;
    if (gc->trace_roots != NULL) gc->trace_roots(gc, gc->roots_context);
    for (root_index = 0u; root_index < gc->native_root_count; ++root_index)
        gc->native_roots[root_index].trace(gc, gc->native_roots[root_index].root);
    {
        LanaGCAllocation *allocation;
        for (allocation = gc->allocations; allocation != NULL; allocation = allocation->next)
            if (!allocation->initialized || allocation->ownership == LANA_GC_OWNER_NATIVE ||
                (include_remembered && allocation->remembered))
                (void)lana_gc_mark(gc, allocation_payload(allocation));
    }
    return !gc->mark_failed;
}

static bool drain_marks(LanaGC *gc, size_t budget) {
    size_t traced = 0u;
    while (!gc->mark_failed && gc->worklist_cursor < gc->worklist_count &&
           (budget == 0u || traced < budget)) {
        LanaGCAllocation *allocation = gc->worklist[gc->worklist_cursor++];
        if (allocation->initialized && allocation->trace != NULL)
            allocation->trace(gc, allocation_payload(allocation));
        ++traced;
    }
    return !gc->mark_failed;
}

void lana_gc_init(LanaGC *gc, size_t memory_limit, size_t collection_threshold,
                  LanaGCRootsFn trace_roots, void *roots_context) {
    if (gc == NULL) return;
    memset(gc, 0, sizeof(*gc));
    gc->memory_limit = memory_limit;
    gc->collection_threshold = collection_threshold == 0u
        ? memory_limit : collection_threshold;
    gc->trace_roots = trace_roots;
    gc->roots_context = roots_context;
}

void lana_gc_free(LanaGC *gc) {
    LanaGCAllocation *allocation;
    if (gc == NULL) return;
    allocation = gc->allocations;
    while (allocation != NULL) {
        LanaGCAllocation *next = allocation->next;
        allocation->magic = 0u;
        free(allocation);
        allocation = next;
    }
    free(gc->native_roots);
    free(gc->worklist);
    free(gc->allocation_index);
    memset(gc, 0, sizeof(*gc));
}

bool lana_gc_mark(LanaGC *gc, void *payload) {
    LanaGCAllocation *allocation = payload_allocation(gc, payload);
    if (allocation == NULL || allocation->marked) return allocation != NULL;
    allocation->marked = true;
    if (!reserve_worklist(gc, gc->worklist_count + 1u)) {
        gc->mark_failed = true;
        return false;
    }
    gc->worklist[gc->worklist_count++] = allocation;
    return true;
}

size_t lana_gc_root_push(LanaGC *gc, void *root, LanaGCTraceFn trace) {
    size_t previous;
    if (gc == NULL || root == NULL || trace == NULL) return SIZE_MAX;
    previous = gc->native_root_count;
    if (!reserve_roots(gc, previous + 1u)) return SIZE_MAX;
    gc->native_roots[previous].root = root;
    gc->native_roots[previous].trace = trace;
    ++gc->native_root_count;
    return previous;
}

void lana_gc_root_pop(LanaGC *gc, size_t previous_count) {
    if (gc == NULL || previous_count > gc->native_root_count) return;
    gc->native_root_count = previous_count;
}

bool lana_gc_collect(LanaGC *gc) {
    LanaGCAllocation **cursor;
    size_t root_index;
    if (gc == NULL || gc->collecting) return gc != NULL;
    gc->collecting = true;
    gc->mark_failed = false;
    gc->worklist_count = 0u;
    gc->worklist_cursor = 0u;
    gc->last_reclaimed_bytes = 0u;
    gc->last_reclaimed_objects = 0u;

    if (gc->trace_roots != NULL) gc->trace_roots(gc, gc->roots_context);
    for (root_index = 0u; root_index < gc->native_root_count; ++root_index)
        gc->native_roots[root_index].trace(gc, gc->native_roots[root_index].root);
    {
        LanaGCAllocation *allocation;
        for (allocation = gc->allocations; allocation != NULL; allocation = allocation->next)
            if (!allocation->initialized || allocation->ownership == LANA_GC_OWNER_NATIVE)
                (void)lana_gc_mark(gc, allocation_payload(allocation));
    }

    while (!gc->mark_failed && gc->worklist_cursor < gc->worklist_count) {
        LanaGCAllocation *allocation = gc->worklist[gc->worklist_cursor++];
        if (allocation->initialized && allocation->trace != NULL)
            allocation->trace(gc, allocation_payload(allocation));
    }
    if (gc->mark_failed) {
        clear_marks(gc);
        gc->collecting = false;
        return false;
    }

    cursor = &gc->allocations;
    while (*cursor != NULL) {
        LanaGCAllocation *allocation = *cursor;
        if (allocation->marked) {
            allocation->marked = false;
            cursor = &allocation->next;
            continue;
        }
        *cursor = allocation->next;
        native_remove(gc, allocation);
        index_remove(gc, allocation);
        gc->allocated_bytes -= allocation->size;
        gc->last_reclaimed_bytes += allocation->size;
        ++gc->last_reclaimed_objects;
        allocation->magic = 0u;
        free(allocation);
    }
    ++gc->collection_count;
    gc->reclaimed_bytes += gc->last_reclaimed_bytes;
    gc->reclaimed_objects += gc->last_reclaimed_objects;
    gc->collecting = false;
    return true;
}

bool lana_gc_collect_young(LanaGC *gc) {
    LanaGCAllocation **cursor;
    if (gc == NULL || gc->collecting) return gc != NULL;
    gc->collecting = true;
    gc->last_reclaimed_bytes = 0u;
    gc->last_reclaimed_objects = 0u;
    if (!mark_roots(gc, true) || !drain_marks(gc, 0u)) {
        clear_marks(gc); gc->collecting = false; return false;
    }
    cursor = &gc->allocations;
    while (*cursor != NULL) {
        LanaGCAllocation *allocation = *cursor;
        if (allocation->generation != LANA_GC_GENERATION_YOUNG) {
            allocation->marked = false; cursor = &allocation->next; continue;
        }
        if (allocation->marked) {
            allocation->marked = false;
            if (++allocation->age >= 2u) {
                allocation->generation = allocation->ownership == LANA_GC_OWNER_SHARED
                    ? LANA_GC_GENERATION_SHARED : LANA_GC_GENERATION_OLD;
                ++gc->promoted_objects;
            }
            cursor = &allocation->next; continue;
        }
        *cursor = allocation->next;
        native_remove(gc, allocation); index_remove(gc, allocation);
        gc->allocated_bytes -= allocation->size;
        gc->last_reclaimed_bytes += allocation->size;
        ++gc->last_reclaimed_objects;
        allocation->magic = 0u; free(allocation);
    }
    ++gc->minor_collection_count;
    gc->reclaimed_bytes += gc->last_reclaimed_bytes;
    gc->reclaimed_objects += gc->last_reclaimed_objects;
    gc->collecting = false;
    return true;
}

bool lana_gc_incremental_begin(LanaGC *gc) {
    if (gc == NULL || gc->collecting) return false;
    gc->collecting = true;
    clear_marks(gc);
    if (!mark_roots(gc, false)) { clear_marks(gc); gc->collecting = false; return false; }
    return true;
}

bool lana_gc_incremental_step(LanaGC *gc, size_t budget, bool *complete) {
    LanaGCAllocation **cursor;
    if (gc == NULL || !gc->collecting || budget == 0u) return false;
    ++gc->incremental_step_count;
    if (!drain_marks(gc, budget)) {
        clear_marks(gc); gc->collecting = false; return false;
    }
    if (gc->worklist_cursor < gc->worklist_count) {
        if (complete != NULL) *complete = false;
        return true;
    }
    cursor = &gc->allocations;
    while (*cursor != NULL) {
        LanaGCAllocation *allocation = *cursor;
        if (allocation->marked) {
            allocation->marked = false; cursor = &allocation->next; continue;
        }
        *cursor = allocation->next;
        native_remove(gc, allocation); index_remove(gc, allocation);
        gc->allocated_bytes -= allocation->size;
        ++gc->reclaimed_objects; gc->reclaimed_bytes += allocation->size;
        allocation->magic = 0u; free(allocation);
    }
    ++gc->collection_count;
    gc->collecting = false;
    if (complete != NULL) *complete = true;
    return true;
}

bool lana_gc_promote(LanaGC *gc, void *payload, bool shared) {
    LanaGCAllocation *allocation = payload_allocation(gc, payload);
    if (allocation == NULL) return false;
    allocation->generation = shared ? LANA_GC_GENERATION_SHARED
                                    : LANA_GC_GENERATION_OLD;
    allocation->age = UINT8_MAX;
    if (shared) allocation->ownership = LANA_GC_OWNER_SHARED;
    ++gc->promoted_objects;
    return true;
}

bool lana_gc_write_barrier(LanaGC *gc, void *owner, void *referenced) {
    LanaGCAllocation *source = payload_allocation(gc, owner);
    LanaGCAllocation *target = payload_allocation(gc, referenced);
    if (source == NULL || target == NULL) return false;
    if (source->generation != LANA_GC_GENERATION_YOUNG &&
        target->generation == LANA_GC_GENERATION_YOUNG && !source->remembered) {
        source->remembered = true;
        ++gc->remembered_count;
    }
    if (source->generation == LANA_GC_GENERATION_SHARED &&
        target->generation == LANA_GC_GENERATION_YOUNG)
        return lana_gc_promote(gc, referenced, true);
    return true;
}

void *lana_gc_alloc(LanaGC *gc, size_t size, LanaGCObjectKind kind,
                    LanaGCOwnership ownership, LanaGCTraceFn trace) {
    LanaGCAllocation *allocation;
    size_t actual_size = size == 0u ? 1u : size;
    if (gc == NULL || actual_size > SIZE_MAX - sizeof(*allocation)) return NULL;
    if (!gc->collecting && !gc->deferred && gc->allocated_bytes > 0u &&
        actual_size <= SIZE_MAX - gc->allocated_bytes &&
        gc->allocated_bytes + actual_size > gc->collection_threshold)
        (void)lana_gc_collect(gc);
    if (gc->allocated_bytes > gc->memory_limit ||
        actual_size > gc->memory_limit ||
        actual_size > gc->memory_limit - gc->allocated_bytes) {
        if (!gc->collecting && !gc->deferred && !lana_gc_collect(gc)) return NULL;
        if (gc->allocated_bytes > gc->memory_limit ||
            actual_size > gc->memory_limit ||
            actual_size > gc->memory_limit - gc->allocated_bytes)
            return NULL;
    }
    allocation = calloc(1u, sizeof(*allocation) + actual_size);
    if (allocation == NULL) {
        if (!gc->collecting && !gc->deferred && lana_gc_collect(gc))
            allocation = calloc(1u, sizeof(*allocation) + actual_size);
        if (allocation == NULL) return NULL;
    }
    if (!index_reserve(gc, gc->allocation_index_count + 1u)) {
        free(allocation);
        return NULL;
    }
    allocation->magic = LANA_GC_MAGIC;
    allocation->kind = kind;
    allocation->ownership = ownership;
    allocation->generation = ownership == LANA_GC_OWNER_SHARED
        ? LANA_GC_GENERATION_SHARED : LANA_GC_GENERATION_YOUNG;
    allocation->trace = trace;
    allocation->size = actual_size;
    allocation->gc = gc;
    allocation->next = gc->allocations;
    gc->allocations = allocation;
    if (ownership == LANA_GC_OWNER_NATIVE) native_add(gc, allocation);
    index_place(gc, allocation);
    gc->allocated_bytes += actual_size;
    ++gc->allocation_count;
    return allocation_payload(allocation);
}

bool lana_gc_publish(LanaGC *gc, void *payload) {
    LanaGCAllocation *allocation = payload_allocation(gc, payload);
    if (allocation == NULL) return false;
    allocation->initialized = true;
    return true;
}

bool lana_gc_configure(LanaGC *gc, void *payload, LanaGCObjectKind kind,
                       LanaGCOwnership ownership, LanaGCTraceFn trace) {
    LanaGCAllocation *allocation = payload_allocation(gc, payload);
    if (allocation == NULL) return false;
    if (ownership == LANA_GC_OWNER_NATIVE) native_add(gc, allocation);
    else native_remove(gc, allocation);
    allocation->kind = kind;
    allocation->ownership = ownership;
    allocation->trace = trace;
    return true;
}

void lana_gc_set_deferred(LanaGC *gc, bool deferred) {
    if (gc != NULL) gc->deferred = deferred;
}

void lana_gc_release_native(LanaGC *gc) {
    LanaGCAllocation *allocation;
    if (gc == NULL) return;
    allocation = gc->native_allocations;
    gc->native_allocations = NULL;
    while (allocation != NULL) {
        LanaGCAllocation *next = allocation->native_next;
        if (allocation->ownership == LANA_GC_OWNER_NATIVE)
            allocation->ownership = LANA_GC_OWNER_VM;
        allocation->native_pending = false;
        allocation->native_next = NULL;
        allocation = next;
    }
}

size_t lana_gc_payload_size(const LanaGC *gc, const void *payload) {
    LanaGCAllocation *allocation = payload_allocation(gc, payload);
    return allocation == NULL ? 0u : allocation->size;
}

LanaGCObjectKind lana_gc_object_kind(const LanaGC *gc, const void *payload) {
    LanaGCAllocation *allocation = payload_allocation(gc, payload);
    return allocation == NULL ? LANA_GC_OPAQUE : allocation->kind;
}

LanaGCGeneration lana_gc_generation(const LanaGC *gc, const void *payload) {
    LanaGCAllocation *allocation = payload_allocation(gc, payload);
    return allocation == NULL ? LANA_GC_GENERATION_OLD : allocation->generation;
}
