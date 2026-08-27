#define _POSIX_C_SOURCE 200809L

#include "lana/shared.h"
#include "lana/vm.h"

#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    double effective_time;
    uint64_t sequence;
    LanaVM *storage;
    Value evidence;
} LanaSharedObservation;

typedef struct {
    double effective_time;
    uint64_t observation_sequence;
    LanaVM *storage;
    Value snapshot;
} LanaSharedVersion;

typedef struct LanaSharedCommit {
    uint64_t revision;
    LanaSharedVersion *versions;
    size_t version_count;
} LanaSharedCommit;

struct LanaCapabilityToken {
    LanaSharedInformation *shared;
    uint64_t id;
    uint32_t permissions;
    bool revoked;
    struct LanaCapabilityToken *next;
};

struct LanaSharedInformation {
    atomic_uint_fast64_t references;
    uint64_t identity;
    const LanaChunk *chunk;
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    uint64_t capability_epoch;
    uint64_t next_capability_id;
    uint64_t next_observation_sequence;
    LanaCapabilityToken *capabilities;
    LanaVM *base_storage;
    Value base_snapshot;
    LanaSharedObservation *observations;
    size_t observation_count;
    size_t observation_capacity;
    LanaSharedCommit *current;
};

static atomic_uint_fast64_t next_shared_identity = ATOMIC_VAR_INIT(1u);
static atomic_uint_fast64_t next_commit_revision = ATOMIC_VAR_INIT(1u);

static LanaVM *storage_vm_new(const LanaSharedInformation *shared) {
    LanaVM *storage = calloc(1u, sizeof(*storage));
    if (storage == NULL) return NULL;
    lana_vm_init(storage, shared->chunk);
    return storage;
}

static void storage_vm_free(LanaVM *storage) {
    if (storage == NULL) return;
    lana_vm_free(storage);
    free(storage);
}

static void commit_free(LanaSharedCommit *commit) {
    size_t index;
    if (commit == NULL) return;
    for (index = 0u; index < commit->version_count; ++index)
        storage_vm_free(commit->versions[index].storage);
    free(commit->versions);
    free(commit);
}

static bool capability_allows_locked(const LanaSharedInformation *shared,
                                     const LanaCapabilityToken *capability,
                                     uint32_t permissions) {
    return capability != NULL && capability->shared == shared &&
           !capability->revoked &&
           (capability->permissions & permissions) == permissions;
}

static LanaCapabilityToken *capability_new_locked(
    LanaSharedInformation *shared, uint32_t permissions) {
    LanaCapabilityToken *capability = calloc(1u, sizeof(*capability));
    if (capability == NULL) return NULL;
    capability->shared = shared;
    capability->id = shared->next_capability_id++;
    capability->permissions = permissions;
    capability->next = shared->capabilities;
    shared->capabilities = capability;
    return capability;
}

static bool effective_time_valid(double effective_time) {
    return isfinite(effective_time) && floor(effective_time) == effective_time &&
           fabs(effective_time) <= 9007199254740991.0;
}

static int observation_compare(const void *left, const void *right) {
    const LanaSharedObservation *a = *(const LanaSharedObservation *const *)left;
    const LanaSharedObservation *b = *(const LanaSharedObservation *const *)right;
    if (a->effective_time < b->effective_time) return -1;
    if (a->effective_time > b->effective_time) return 1;
    if (a->sequence < b->sequence) return -1;
    if (a->sequence > b->sequence) return 1;
    return 0;
}

static LanaError build_commit_candidate(
    LanaSharedInformation *shared,
    LanaSharedObservation *observations,
    size_t observation_count,
    LanaSharedObservation *pending,
    LanaSharedCommit **out) {
    LanaSharedObservation **ordered;
    LanaSharedCommit *commit;
    size_t index;
    LanaError error = LANA_OK;
    ordered = calloc(observation_count + 1u, sizeof(*ordered));
    commit = calloc(1u, sizeof(*commit));
    if (ordered == NULL || commit == NULL) {
        free(ordered);
        free(commit);
        return LANA_ERR_OOM;
    }
    for (index = 0u; index < observation_count; ++index)
        ordered[index] = &observations[index];
    ordered[observation_count] = pending;
    qsort(ordered, observation_count + 1u, sizeof(*ordered),
          observation_compare);
    commit->version_count = observation_count + 1u;
    commit->versions = calloc(commit->version_count, sizeof(*commit->versions));
    if (commit->versions == NULL) {
        free(ordered);
        free(commit);
        return LANA_ERR_OOM;
    }
    for (index = 0u; index < commit->version_count; ++index) {
        LanaSharedVersion *version = &commit->versions[index];
        const Value *source = index == 0u
            ? &shared->base_snapshot
            : &commit->versions[index - 1u].snapshot;
        Value local_source;
        Value local_evidence;
        version->storage = storage_vm_new(shared);
        if (version->storage == NULL) {
            error = LANA_ERR_OOM;
            break;
        }
        error = lana_vm_clone_live_value(version->storage, source,
                                         &local_source);
        if (error == LANA_OK)
            error = lana_vm_clone_value(version->storage,
                &ordered[index]->evidence, &local_evidence);
        if (error == LANA_OK)
            error = lana_vm_reactive_observe(version->storage, &local_source,
                                              &local_evidence,
                                              &version->snapshot);
        if (error != LANA_OK) break;
        version->effective_time = ordered[index]->effective_time;
        version->observation_sequence = ordered[index]->sequence;
    }
    free(ordered);
    if (error != LANA_OK) {
        commit_free(commit);
        return error;
    }
    *out = commit;
    return LANA_OK;
}

LanaError lana_shared_information_create(LanaVM *source_vm,
                                         const Value *source,
                                         LanaSharedInformation **out,
                                         LanaCapabilityToken **admin) {
    LanaSharedInformation *shared;
    LanaCapabilityToken *admin_token;
    LanaError error;
    if (source_vm == NULL || source == NULL || out == NULL || admin == NULL)
        return LANA_ERR_FORMAT;
    shared = calloc(1u, sizeof(*shared));
    if (shared == NULL) return LANA_ERR_OOM;
    atomic_init(&shared->references, 1u);
    shared->identity = atomic_fetch_add(&next_shared_identity, 1u);
    shared->chunk = source_vm->chunk;
    shared->next_capability_id = 1u;
    shared->next_observation_sequence = 1u;
    if (pthread_mutex_init(&shared->mutex, NULL) != 0 ||
        pthread_cond_init(&shared->condition, NULL) != 0) {
        free(shared);
        return LANA_ERR_TASK;
    }
    shared->base_storage = storage_vm_new(shared);
    if (shared->base_storage == NULL) {
        lana_shared_information_release(shared);
        return LANA_ERR_OOM;
    }
    error = lana_vm_clone_live_value(shared->base_storage, source,
                                     &shared->base_snapshot);
    if (error == LANA_OK && shared->base_snapshot.reactive == NULL) {
        Value root;
        error = lana_vm_reactive_root(shared->base_storage,
            &shared->base_snapshot, LANA_EXACTNESS_EXACT, &root);
        if (error == LANA_OK) shared->base_snapshot = root;
    }
    if (error != LANA_OK) {
        lana_shared_information_release(shared);
        return error;
    }
    shared->current = calloc(1u, sizeof(*shared->current));
    if (shared->current == NULL) {
        lana_shared_information_release(shared);
        return LANA_ERR_OOM;
    }
    admin_token = capability_new_locked(shared, LANA_CAPABILITY_ADMIN);
    if (admin_token == NULL) {
        lana_shared_information_release(shared);
        return LANA_ERR_OOM;
    }
    *out = shared;
    *admin = admin_token;
    return LANA_OK;
}

void lana_shared_information_retain(LanaSharedInformation *shared) {
    if (shared != NULL)
        (void)atomic_fetch_add(&shared->references, 1u);
}

void lana_shared_information_release(LanaSharedInformation *shared) {
    LanaCapabilityToken *capability;
    size_t index;
    if (shared == NULL || atomic_fetch_sub(&shared->references, 1u) != 1u)
        return;
    storage_vm_free(shared->base_storage);
    for (index = 0u; index < shared->observation_count; ++index)
        storage_vm_free(shared->observations[index].storage);
    free(shared->observations);
    commit_free(shared->current);
    capability = shared->capabilities;
    while (capability != NULL) {
        LanaCapabilityToken *next = capability->next;
        free(capability);
        capability = next;
    }
    (void)pthread_cond_destroy(&shared->condition);
    (void)pthread_mutex_destroy(&shared->mutex);
    free(shared);
}

uint64_t lana_shared_information_identity(
    const LanaSharedInformation *shared) {
    return shared == NULL ? 0u : shared->identity;
}

uint64_t lana_shared_information_revision(
    const LanaSharedInformation *shared) {
    uint64_t revision;
    if (shared == NULL) return 0u;
    (void)pthread_mutex_lock((pthread_mutex_t *)&shared->mutex);
    revision = shared->current == NULL ? 0u : shared->current->revision;
    (void)pthread_mutex_unlock((pthread_mutex_t *)&shared->mutex);
    return revision;
}

LanaError lana_shared_capability_grant(LanaCapabilityToken *admin,
                                       uint32_t permissions,
                                       LanaCapabilityToken **out) {
    LanaSharedInformation *shared;
    LanaCapabilityToken *capability;
    const uint32_t valid = LANA_CAPABILITY_READ | LANA_CAPABILITY_OBSERVE |
                           LANA_CAPABILITY_ADMIN;
    if (admin == NULL || out == NULL || permissions == 0u ||
        (permissions & ~valid) != 0u) return LANA_ERR_FORMAT;
    shared = admin->shared;
    (void)pthread_mutex_lock(&shared->mutex);
    if (!capability_allows_locked(shared, admin, LANA_CAPABILITY_ADMIN)) {
        (void)pthread_mutex_unlock(&shared->mutex);
        return LANA_ERR_CAPABILITY;
    }
    capability = capability_new_locked(shared, permissions);
    if (capability == NULL) {
        (void)pthread_mutex_unlock(&shared->mutex);
        return LANA_ERR_OOM;
    }
    ++shared->capability_epoch;
    (void)pthread_mutex_unlock(&shared->mutex);
    *out = capability;
    return LANA_OK;
}

LanaError lana_shared_capability_revoke(LanaCapabilityToken *admin,
                                        LanaCapabilityToken *target) {
    LanaSharedInformation *shared;
    if (admin == NULL || target == NULL || admin->shared != target->shared)
        return LANA_ERR_CAPABILITY;
    shared = admin->shared;
    (void)pthread_mutex_lock(&shared->mutex);
    if (!capability_allows_locked(shared, admin, LANA_CAPABILITY_ADMIN)) {
        (void)pthread_mutex_unlock(&shared->mutex);
        return LANA_ERR_CAPABILITY;
    }
    target->revoked = true;
    ++shared->capability_epoch;
    (void)pthread_cond_broadcast(&shared->condition);
    (void)pthread_mutex_unlock(&shared->mutex);
    return LANA_OK;
}

bool lana_shared_capability_allows(const LanaCapabilityToken *capability,
                                   uint32_t permissions) {
    LanaSharedInformation *shared;
    bool allowed;
    if (capability == NULL) return false;
    shared = capability->shared;
    (void)pthread_mutex_lock(&shared->mutex);
    allowed = capability_allows_locked(shared, capability, permissions);
    (void)pthread_mutex_unlock(&shared->mutex);
    return allowed;
}

LanaSharedInformation *lana_shared_capability_information(
    const LanaCapabilityToken *capability) {
    return capability == NULL ? NULL : capability->shared;
}

static const Value *current_snapshot_locked(LanaSharedInformation *shared) {
    if (shared->current == NULL || shared->current->version_count == 0u)
        return &shared->base_snapshot;
    return &shared->current->versions[
        shared->current->version_count - 1u].snapshot;
}

LanaError lana_shared_information_snapshot(LanaVM *destination,
                                           LanaSharedInformation *shared,
                                           LanaCapabilityToken *read,
                                           Value *out,
                                           uint64_t *revision) {
    const Value *snapshot;
    LanaError error;
    if (destination == NULL || shared == NULL || out == NULL)
        return LANA_ERR_FORMAT;
    (void)pthread_mutex_lock(&shared->mutex);
    if (!capability_allows_locked(shared, read, LANA_CAPABILITY_READ)) {
        (void)pthread_mutex_unlock(&shared->mutex);
        return LANA_ERR_CAPABILITY;
    }
    snapshot = current_snapshot_locked(shared);
    error = lana_vm_clone_live_value(destination, snapshot, out);
    if (error == LANA_OK && revision != NULL)
        *revision = shared->current->revision;
    (void)pthread_mutex_unlock(&shared->mutex);
    return error;
}

LanaError lana_shared_information_at(LanaVM *destination,
                                     LanaSharedInformation *shared,
                                     LanaCapabilityToken *read,
                                     double effective_time,
                                     Value *out,
                                     uint64_t *revision) {
    const Value *snapshot;
    size_t index;
    LanaError error;
    if (destination == NULL || shared == NULL || out == NULL ||
        !effective_time_valid(effective_time)) return LANA_ERR_FORMAT;
    (void)pthread_mutex_lock(&shared->mutex);
    if (!capability_allows_locked(shared, read, LANA_CAPABILITY_READ)) {
        (void)pthread_mutex_unlock(&shared->mutex);
        return LANA_ERR_CAPABILITY;
    }
    snapshot = &shared->base_snapshot;
    for (index = 0u; index < shared->current->version_count; ++index) {
        if (shared->current->versions[index].effective_time > effective_time)
            break;
        snapshot = &shared->current->versions[index].snapshot;
    }
    error = lana_vm_clone_live_value(destination, snapshot, out);
    if (error == LANA_OK && revision != NULL)
        *revision = shared->current->revision;
    (void)pthread_mutex_unlock(&shared->mutex);
    return error;
}

LanaError lana_shared_information_observe(LanaVM *caller,
                                          LanaSharedInformation *shared,
                                          LanaCapabilityToken *observe,
                                          const Value *evidence,
                                          double effective_time,
                                          uint64_t *revision) {
    LanaSharedObservation pending = {0};
    size_t attempt;
    LanaError error;
    if (caller == NULL || shared == NULL || evidence == NULL ||
        !effective_time_valid(effective_time)) return LANA_ERR_FORMAT;
    pending.effective_time = effective_time;
    pending.storage = storage_vm_new(shared);
    if (pending.storage == NULL) return LANA_ERR_OOM;
    error = lana_vm_clone_value(pending.storage, evidence, &pending.evidence);
    if (error != LANA_OK) {
        storage_vm_free(pending.storage);
        return error;
    }
    for (attempt = 0u; attempt < 32u; ++attempt) {
        LanaSharedCommit *old_commit;
        LanaSharedCommit *candidate = NULL;
        LanaSharedObservation *observations = NULL;
        uint64_t capability_epoch;
        size_t observation_count;
        size_t index;
        (void)pthread_mutex_lock(&shared->mutex);
        if (!capability_allows_locked(shared, observe,
                                      LANA_CAPABILITY_OBSERVE)) {
            (void)pthread_mutex_unlock(&shared->mutex);
            storage_vm_free(pending.storage);
            return LANA_ERR_CAPABILITY;
        }
        for (index = 0u; index < shared->observation_count; ++index) {
            if (shared->observations[index].effective_time != effective_time)
                continue;
            if (lana_vm_value_equal(&shared->observations[index].evidence,
                                    &pending.evidence)) {
                if (revision != NULL) *revision = shared->current->revision;
                (void)pthread_mutex_unlock(&shared->mutex);
                storage_vm_free(pending.storage);
                return LANA_OK;
            }
            (void)pthread_mutex_unlock(&shared->mutex);
            storage_vm_free(pending.storage);
            return LANA_ERR_CONFLICT;
        }
        old_commit = shared->current;
        capability_epoch = shared->capability_epoch;
        observation_count = shared->observation_count;
        if (observation_count > 0u) {
            observations = malloc(observation_count * sizeof(*observations));
            if (observations == NULL) {
                (void)pthread_mutex_unlock(&shared->mutex);
                storage_vm_free(pending.storage);
                return LANA_ERR_OOM;
            }
            memcpy(observations, shared->observations,
                   observation_count * sizeof(*observations));
        }
        pending.sequence = shared->next_observation_sequence;
        (void)pthread_mutex_unlock(&shared->mutex);
        if (atomic_load(&caller->cancelled)) {
            free(observations);
            storage_vm_free(pending.storage);
            return LANA_ERR_CANCELLED;
        }
        error = build_commit_candidate(shared, observations,
                                       observation_count, &pending,
                                       &candidate);
        free(observations);
        if (error != LANA_OK) {
            storage_vm_free(pending.storage);
            return error;
        }
        (void)pthread_mutex_lock(&shared->mutex);
        if (shared->current != old_commit ||
            shared->observation_count != observation_count ||
            shared->capability_epoch != capability_epoch) {
            (void)pthread_mutex_unlock(&shared->mutex);
            commit_free(candidate);
            continue;
        }
        if (!capability_allows_locked(shared, observe,
                                      LANA_CAPABILITY_OBSERVE)) {
            (void)pthread_mutex_unlock(&shared->mutex);
            commit_free(candidate);
            storage_vm_free(pending.storage);
            return LANA_ERR_CAPABILITY;
        }
        if (shared->observation_count == shared->observation_capacity) {
            size_t capacity = shared->observation_capacity == 0u
                ? 4u : shared->observation_capacity * 2u;
            LanaSharedObservation *observations = realloc(shared->observations,
                capacity * sizeof(*observations));
            if (observations == NULL) {
                (void)pthread_mutex_unlock(&shared->mutex);
                commit_free(candidate);
                storage_vm_free(pending.storage);
                return LANA_ERR_OOM;
            }
            shared->observations = observations;
            shared->observation_capacity = capacity;
        }
        candidate->revision = atomic_fetch_add(&next_commit_revision, 1u);
        shared->observations[shared->observation_count++] = pending;
        ++shared->next_observation_sequence;
        shared->current = candidate;
        if (revision != NULL) *revision = candidate->revision;
        (void)pthread_cond_broadcast(&shared->condition);
        (void)pthread_mutex_unlock(&shared->mutex);
        commit_free(old_commit);
        return LANA_OK;
    }
    storage_vm_free(pending.storage);
    return LANA_ERR_CONFLICT;
}

LanaError lana_shared_information_wait(LanaVM *destination,
                                       LanaSharedInformation *shared,
                                       LanaCapabilityToken *read,
                                       uint64_t after_revision,
                                       uint64_t timeout_milliseconds,
                                       Value *out,
                                       uint64_t *revision) {
    struct timespec deadline = {0};
    int wait_result = 0;
    if (destination == NULL || shared == NULL || out == NULL)
        return LANA_ERR_FORMAT;
    if (timeout_milliseconds > 0u) {
        if (clock_gettime(CLOCK_REALTIME, &deadline) != 0) return LANA_ERR_TASK;
        deadline.tv_sec += (time_t)(timeout_milliseconds / 1000u);
        deadline.tv_nsec += (long)((timeout_milliseconds % 1000u) * 1000000u);
        if (deadline.tv_nsec >= 1000000000L) {
            ++deadline.tv_sec;
            deadline.tv_nsec -= 1000000000L;
        }
    }
    (void)pthread_mutex_lock(&shared->mutex);
    while (shared->current->revision <= after_revision) {
        if (!capability_allows_locked(shared, read, LANA_CAPABILITY_READ)) {
            (void)pthread_mutex_unlock(&shared->mutex);
            return LANA_ERR_CAPABILITY;
        }
        if (atomic_load(&destination->cancelled)) {
            (void)pthread_mutex_unlock(&shared->mutex);
            return LANA_ERR_CANCELLED;
        }
        if (timeout_milliseconds == 0u)
            wait_result = pthread_cond_wait(&shared->condition, &shared->mutex);
        else
            wait_result = pthread_cond_timedwait(&shared->condition,
                                                 &shared->mutex, &deadline);
        if (wait_result == ETIMEDOUT) {
            (void)pthread_mutex_unlock(&shared->mutex);
            return LANA_ERR_TIMEOUT;
        }
        if (wait_result != 0) {
            (void)pthread_mutex_unlock(&shared->mutex);
            return LANA_ERR_TASK;
        }
    }
    if (!capability_allows_locked(shared, read, LANA_CAPABILITY_READ)) {
        (void)pthread_mutex_unlock(&shared->mutex);
        return LANA_ERR_CAPABILITY;
    }
    {
        LanaError error = lana_vm_clone_live_value(destination,
            current_snapshot_locked(shared), out);
        if (error == LANA_OK && revision != NULL)
            *revision = shared->current->revision;
        (void)pthread_mutex_unlock(&shared->mutex);
        return error;
    }
}
