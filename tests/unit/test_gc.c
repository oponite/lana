#include "gc.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CHECK(condition) do { if (!(condition)) { \
    (void)fprintf(stderr, "check failed: %s:%d: %s\n", __FILE__, __LINE__, #condition); \
    return 1; \
} } while (0)

typedef struct Node {
    struct Node *left;
    struct Node *right;
    size_t value;
} Node;

typedef struct {
    Node *root;
} Roots;

static void trace_node(LanaGC *gc, void *payload) {
    Node *node = payload;
    if (node->left != NULL) (void)lana_gc_mark(gc, node->left);
    if (node->right != NULL) (void)lana_gc_mark(gc, node->right);
}

static void trace_roots(LanaGC *gc, void *context) {
    Roots *roots = context;
    if (roots->root != NULL) (void)lana_gc_mark(gc, roots->root);
}

static void trace_node_root(LanaGC *gc, void *payload) {
    (void)lana_gc_mark(gc, payload);
}

static Node *node_new(LanaGC *gc, size_t value) {
    Node *node = lana_gc_alloc(gc, sizeof(*node), LANA_GC_RUNTIME_INTERNAL,
                              LANA_GC_OWNER_NATIVE, trace_node);
    if (node == NULL) return NULL;
    node->value = value;
    return lana_gc_publish(gc, node) ? node : NULL;
}

static int test_cycle_and_reachable_preservation(void) {
    LanaGC gc;
    Roots roots = {0};
    Node *reachable, *left, *right;
    lana_gc_init(&gc, 1024u * 1024u, 1024u * 1024u, trace_roots, &roots);
    reachable = node_new(&gc, 1u);
    left = node_new(&gc, 2u);
    right = node_new(&gc, 3u);
    CHECK(reachable != NULL && left != NULL && right != NULL);
    left->right = right;
    right->left = left;
    roots.root = reachable;
    lana_gc_release_native(&gc);
    CHECK(lana_gc_collect(&gc));
    CHECK(gc.last_reclaimed_objects == 2u);
    CHECK(roots.root->value == 1u);
    CHECK(lana_gc_object_kind(&gc, roots.root) == LANA_GC_RUNTIME_INTERNAL);
    roots.root = NULL;
    CHECK(lana_gc_collect(&gc));
    CHECK(gc.last_reclaimed_objects == 1u);
    lana_gc_free(&gc);
    return 0;
}

static int test_deep_graph_and_native_root(void) {
    enum { DEPTH = 20000 };
    LanaGC gc;
    Roots roots = {0};
    Node *head = NULL;
    size_t index, root_mark;
    lana_gc_init(&gc, 8u * 1024u * 1024u, 8u * 1024u * 1024u, trace_roots, &roots);
    for (index = 0u; index < DEPTH; ++index) {
        Node *node = node_new(&gc, index);
        CHECK(node != NULL);
        node->left = head;
        head = node;
    }
    lana_gc_release_native(&gc);
    root_mark = lana_gc_root_push(&gc, head, trace_node_root);
    CHECK(root_mark != SIZE_MAX);
    CHECK(lana_gc_collect(&gc));
    CHECK(gc.last_reclaimed_objects == 0u);
    CHECK(head->value == DEPTH - 1u);
    lana_gc_root_pop(&gc, root_mark);
    CHECK(lana_gc_collect(&gc));
    CHECK(gc.last_reclaimed_objects == DEPTH);
    lana_gc_free(&gc);
    return 0;
}

static int test_unpublished_and_limit_retry(void) {
    LanaGC gc;
    Roots roots = {0};
    Node *unpublished, *garbage, *replacement;
    lana_gc_init(&gc, sizeof(Node) * 2u, sizeof(Node) * 2u, trace_roots, &roots);
    unpublished = lana_gc_alloc(&gc, sizeof(*unpublished), LANA_GC_RUNTIME_INTERNAL,
                                LANA_GC_OWNER_VM, trace_node);
    garbage = node_new(&gc, 7u);
    CHECK(unpublished != NULL && garbage != NULL);
    lana_gc_release_native(&gc);
    replacement = lana_gc_alloc(&gc, sizeof(*replacement), LANA_GC_RUNTIME_INTERNAL,
                                LANA_GC_OWNER_NATIVE, trace_node);
    CHECK(replacement != NULL);
    CHECK(gc.reclaimed_objects == 1u);
    unpublished->value = 9u;
    CHECK(lana_gc_publish(&gc, unpublished));
    roots.root = unpublished;
    CHECK(lana_gc_publish(&gc, replacement));
    lana_gc_release_native(&gc);
    CHECK(lana_gc_collect(&gc));
    CHECK(roots.root->value == 9u);
    CHECK(gc.last_reclaimed_objects == 1u);
    lana_gc_free(&gc);
    return 0;
}

static int test_metadata_and_invalid_pointers(void) {
    LanaGC gc;
    Roots roots = {0};
    int unmanaged = 0;
    void *payload;
    lana_gc_init(&gc, 1024u, 1024u, trace_roots, &roots);
    payload = lana_gc_alloc(&gc, 0u, LANA_GC_OPAQUE,
                            LANA_GC_OWNER_NATIVE, NULL);
    CHECK(payload != NULL);
    CHECK(lana_gc_payload_size(&gc, payload) == 1u);
    CHECK(lana_gc_object_kind(&gc, payload) == LANA_GC_OPAQUE);
    CHECK(lana_gc_configure(&gc, payload, LANA_GC_STRING,
                            LANA_GC_OWNER_VM, NULL));
    CHECK(lana_gc_publish(&gc, payload));
    CHECK(lana_gc_object_kind(&gc, payload) == LANA_GC_STRING);
    CHECK(!lana_gc_mark(&gc, &unmanaged));
    CHECK(!lana_gc_publish(&gc, &unmanaged));
    CHECK(!lana_gc_configure(&gc, &unmanaged, LANA_GC_OPAQUE,
                             LANA_GC_OWNER_VM, NULL));
    CHECK(lana_gc_payload_size(&gc, &unmanaged) == 0u);
    lana_gc_set_deferred(&gc, true);
    gc.memory_limit = 0u;
    CHECK(lana_gc_alloc(&gc, 1u, LANA_GC_OPAQUE,
                        LANA_GC_OWNER_NATIVE, NULL) == NULL);
    CHECK(lana_gc_payload_size(&gc, payload) == 1u);
    lana_gc_free(&gc);
    return 0;
}

static int test_generations_barrier_and_incremental_steps(void) {
    LanaGC gc;
    Roots roots = {0};
    Node *old;
    Node *young;
    Node *garbage;
    bool complete = false;
    size_t steps = 0u;
    lana_gc_init(&gc, 1024u * 1024u, 1024u * 1024u, trace_roots, &roots);
    old = node_new(&gc, 1u);
    young = node_new(&gc, 2u);
    garbage = node_new(&gc, 3u);
    CHECK(old != NULL && young != NULL && garbage != NULL);
    CHECK(lana_gc_generation(&gc, old) == LANA_GC_GENERATION_YOUNG);
    CHECK(lana_gc_promote(&gc, old, false));
    CHECK(lana_gc_generation(&gc, old) == LANA_GC_GENERATION_OLD);
    old->left = young;
    CHECK(lana_gc_write_barrier(&gc, old, young));
    CHECK(gc.remembered_count == 1u);
    roots.root = old;
    lana_gc_release_native(&gc);
    CHECK(lana_gc_collect_young(&gc));
    CHECK(old->left == young && young->value == 2u);
    CHECK(lana_gc_payload_size(&gc, garbage) == 0u);
    CHECK(gc.minor_collection_count == 1u);
    CHECK(lana_gc_promote(&gc, young, true));
    CHECK(lana_gc_generation(&gc, young) == LANA_GC_GENERATION_SHARED);
    roots.root = NULL;
    CHECK(lana_gc_incremental_begin(&gc));
    while (!complete) {
        CHECK(lana_gc_incremental_step(&gc, 1u, &complete));
        CHECK(++steps < 16u);
    }
    CHECK(gc.incremental_step_count > 0u);
    CHECK(gc.allocated_bytes == 0u);
    lana_gc_free(&gc);
    return 0;
}

static int test_incremental_pause_target(void) {
    enum { DEPTH = 20000 };
    LanaGC gc; Roots roots = {0}; Node *head = NULL; size_t index;
    bool complete = false; uint64_t maximum_ns = 0u;
    lana_gc_init(&gc, 8u * 1024u * 1024u, 8u * 1024u * 1024u,
                 trace_roots, &roots);
    for (index = 0u; index < DEPTH; ++index) {
        Node *node = node_new(&gc, index); CHECK(node != NULL);
        node->left = head; head = node;
    }
    roots.root = head; lana_gc_release_native(&gc);
    CHECK(lana_gc_incremental_begin(&gc));
    while (!complete) {
        struct timespec before, after; uint64_t elapsed;
        (void)timespec_get(&before, TIME_UTC);
        CHECK(lana_gc_incremental_step(&gc, 128u, &complete));
        (void)timespec_get(&after, TIME_UTC);
        elapsed = (uint64_t)(after.tv_sec - before.tv_sec) * UINT64_C(1000000000) +
                  (uint64_t)(after.tv_nsec - before.tv_nsec);
        if (elapsed > maximum_ns) maximum_ns = elapsed;
    }
    CHECK(maximum_ns < UINT64_C(10000000));
    (void)printf("incremental pmax=%llu ns budget=128\n",
                 (unsigned long long)maximum_ns);
    lana_gc_free(&gc); return 0;
}

int main(void) {
    CHECK(test_cycle_and_reachable_preservation() == 0);
    CHECK(test_deep_graph_and_native_root() == 0);
    CHECK(test_unpublished_and_limit_retry() == 0);
    CHECK(test_metadata_and_invalid_pointers() == 0);
    CHECK(test_generations_barrier_and_incremental_steps() == 0);
    CHECK(test_incremental_pause_target() == 0);
    (void)puts("gc tests passed");
    return 0;
}
