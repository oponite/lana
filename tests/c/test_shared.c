#include "lana/shared.h"
#include "lana/vm.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        (void)fprintf(stderr, "CHECK failed at %s:%d: %s\n", \
                      __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

typedef struct {
    const LanaChunk *chunk;
    LanaSharedInformation *shared;
    LanaCapabilityToken *capability;
    double effective_time;
    LanaError error;
    uint64_t revision;
} ObserveContext;

typedef struct {
    const LanaChunk *chunk;
    LanaSharedInformation *shared;
    LanaCapabilityToken *capability;
    uint64_t after_revision;
    LanaError error;
    uint64_t revision;
    Value snapshot;
} WaitContext;

static void *observe_worker(void *argument) {
    ObserveContext *context = argument;
    LanaVM *vm = calloc(1u, sizeof(*vm));
    Value evidence = lana_value_number(2.0);
    if (vm == NULL) {
        context->error = LANA_ERR_OOM;
        return NULL;
    }
    lana_vm_init(vm, context->chunk);
    context->error = lana_shared_information_observe(
        vm, context->shared, context->capability, &evidence,
        context->effective_time, &context->revision);
    lana_vm_free(vm);
    free(vm);
    return NULL;
}

static void *wait_worker(void *argument) {
    WaitContext *context = argument;
    LanaVM *vm = calloc(1u, sizeof(*vm));
    if (vm == NULL) {
        context->error = LANA_ERR_OOM;
        return NULL;
    }
    lana_vm_init(vm, context->chunk);
    context->error = lana_shared_information_wait(
        vm, context->shared, context->capability, context->after_revision,
        5000u, &context->snapshot, &context->revision);
    if (context->error == LANA_OK && context->snapshot.reactive != NULL)
        context->snapshot = *context->snapshot.reactive->current;
    lana_vm_free(vm);
    free(vm);
    return NULL;
}

static int test_shared_information_transactions(void) {
    enum { WORKER_COUNT = 8 };
    LanaChunk chunk;
    LanaVM source_vm;
    LanaVM reader_vm;
    LanaPossibility *possibility;
    Value alternatives[] = {lana_value_number(1.0), lana_value_number(2.0)};
    Value source;
    Value snapshot;
    Value two = lana_value_number(2.0);
    Value one = lana_value_number(1.0);
    LanaSharedInformation *shared;
    LanaCapabilityToken *admin;
    LanaCapabilityToken *read;
    LanaCapabilityToken *observe;
    LanaCapabilityToken *revoked;
    uint64_t revision;
    uint64_t before;
    pthread_t waiter_thread;
    WaitContext waiter;
    pthread_t workers[WORKER_COUNT];
    ObserveContext contexts[WORKER_COUNT];
    size_t index;

    lana_chunk_init(&chunk);
    chunk.version = LABC_VERSION;
    lana_vm_init(&source_vm, &chunk);
    lana_vm_init(&reader_vm, &chunk);
    CHECK(lana_vm_possibility_build(&source_vm, alternatives, 2u,
                                    &possibility) == LANA_OK);
    source = lana_value_possibility(possibility);
    CHECK(lana_shared_information_create(&source_vm, &source, &shared,
                                         &admin) == LANA_OK);
    CHECK(lana_shared_information_identity(shared) != 0u);
    CHECK(lana_shared_information_revision(shared) == 0u);
    CHECK(lana_shared_information_snapshot(&reader_vm, shared, admin,
                                           &snapshot, NULL) ==
          LANA_ERR_CAPABILITY);
    CHECK(lana_shared_capability_grant(admin, LANA_CAPABILITY_READ,
                                      &read) == LANA_OK);
    CHECK(lana_shared_capability_grant(admin, LANA_CAPABILITY_OBSERVE,
                                      &observe) == LANA_OK);
    CHECK(lana_shared_capability_grant(admin, LANA_CAPABILITY_OBSERVE,
                                      &revoked) == LANA_OK);
    CHECK(lana_shared_capability_revoke(admin, revoked) == LANA_OK);
    CHECK(!lana_shared_capability_allows(revoked, LANA_CAPABILITY_OBSERVE));
    CHECK(lana_shared_information_observe(&source_vm, shared, revoked, &two,
                                          1.0, NULL) == LANA_ERR_CAPABILITY);

    CHECK(lana_shared_information_snapshot(&reader_vm, shared, read,
                                           &snapshot, &revision) == LANA_OK);
    CHECK(revision == 0u);
    CHECK(snapshot.reactive != NULL);
    CHECK(snapshot.reactive->current->type == VAL_POSSIBILITY);

    CHECK(lana_shared_information_observe(&source_vm, shared, observe, &two,
                                          20.0, &revision) == LANA_OK);
    CHECK(revision != 0u);
    before = revision;
    CHECK(lana_shared_information_observe(&source_vm, shared, observe, &two,
                                          20.0, &revision) == LANA_OK);
    CHECK(revision == before);
    CHECK(lana_shared_information_observe(&source_vm, shared, observe, &one,
                                          20.0, NULL) == LANA_ERR_CONFLICT);
    CHECK(lana_shared_information_revision(shared) == before);

    CHECK(lana_shared_information_observe(&source_vm, shared, observe, &two,
                                          10.0, &revision) == LANA_OK);
    CHECK(revision > before);
    CHECK(lana_shared_information_at(&reader_vm, shared, read, 5.0,
                                     &snapshot, NULL) == LANA_OK);
    CHECK(snapshot.reactive->current->type == VAL_POSSIBILITY);
    CHECK(lana_shared_information_at(&reader_vm, shared, read, 10.0,
                                     &snapshot, NULL) == LANA_OK);
    CHECK(snapshot.reactive->current->type == VAL_NUMBER);
    CHECK(snapshot.reactive->current->as.number == 2.0);

    waiter = (WaitContext){&chunk, shared, read, revision, LANA_OK, 0u,
                           lana_value_null()};
    CHECK(pthread_create(&waiter_thread, NULL, wait_worker, &waiter) == 0);
    CHECK(lana_shared_information_observe(&source_vm, shared, observe, &two,
                                          30.0, &before) == LANA_OK);
    CHECK(pthread_join(waiter_thread, NULL) == 0);
    CHECK(waiter.error == LANA_OK);
    CHECK(waiter.revision == before);
    CHECK(waiter.snapshot.type == VAL_NUMBER);
    CHECK(waiter.snapshot.as.number == 2.0);

    for (index = 0u; index < WORKER_COUNT; ++index) {
        contexts[index] = (ObserveContext){&chunk, shared, observe, 40.0,
                                           LANA_OK, 0u};
        CHECK(pthread_create(&workers[index], NULL, observe_worker,
                             &contexts[index]) == 0);
    }
    for (index = 0u; index < WORKER_COUNT; ++index) {
        CHECK(pthread_join(workers[index], NULL) == 0);
        CHECK(contexts[index].error == LANA_OK);
    }
    revision = lana_shared_information_revision(shared);
    CHECK(revision > before);
    CHECK(lana_shared_information_wait(&reader_vm, shared, read, revision,
                                       1u, &snapshot, NULL) ==
          LANA_ERR_TIMEOUT);

    before = revision;
    atomic_store(&source_vm.cancelled, true);
    CHECK(lana_shared_information_observe(&source_vm, shared, observe, &two,
                                          50.0, NULL) ==
          LANA_ERR_CANCELLED);
    CHECK(lana_shared_information_revision(shared) == before);
    atomic_store(&source_vm.cancelled, false);

    lana_shared_information_release(shared);
    lana_vm_free(&reader_vm);
    lana_vm_free(&source_vm);
    lana_chunk_free(&chunk);
    return 0;
}

int main(void) {
    CHECK(test_shared_information_transactions() == 0);
    (void)printf("Shared Information tests passed\n");
    return 0;
}
