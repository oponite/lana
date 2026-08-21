#include "ss/vm.h"
#include "ss/data.h"

#include <math.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>

extern char *realpath(const char *path, char *resolved_path);

struct SSScheduler {
    pthread_mutex_t mutex;
    pthread_cond_t available;
    pthread_t *workers;
    size_t worker_count;
    size_t task_limit;
    size_t live_tasks;
    uint64_t next_task_id;
    bool stopping;
    SSTask *queue_head;
    SSTask *queue_tail;
    SSTask *all_tasks;
};

struct SSPathExecution {
    SSFrame *false_frames;
    SSFrame *true_frames;
    size_t frame_count;
    size_t false_ip;
    uint64_t dependency_id;
    double true_weight;
    double false_weight;
    size_t previous_path_count;
    bool running_false;
    struct SSPathExecution *next;
};

static uint64_t mix64(uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30u)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27u)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31u);
}

static SSFrame *current_frame(VM *vm) { return &vm->frames[vm->frame_count - 1u]; }
static SSError consume_sampling_budget(VM *vm);
static bool joint_value_is_definite(const Value *value);

static SSError clone_value(VM *destination, const Value *source, Value *out);
static SSError wait_task(VM *vm, SSTask *task, double timeout_seconds, Value *out);
static void scheduler_shutdown(SSScheduler *scheduler);
static void scheduler_destroy(SSScheduler *scheduler);

static SSError clone_model(VM *destination, const SSMLModel *source, SSMLModel **out) {
    SSMLModel *model;
    size_t coefficient_count;
    model = ss_vm_alloc(destination, sizeof(*model));
    if (model == NULL) return SS_ERR_OOM;
    *model = *source;
    coefficient_count = source->feature_count + 1u;
    model->coefficients = ss_vm_alloc(destination, coefficient_count * sizeof(double));
    model->means = ss_vm_alloc(destination, source->feature_count * sizeof(double));
    model->scales = ss_vm_alloc(destination, source->feature_count * sizeof(double));
    if ((model->coefficients == NULL && coefficient_count > 0u) ||
        (model->means == NULL && source->feature_count > 0u) ||
        (model->scales == NULL && source->feature_count > 0u)) return SS_ERR_OOM;
    memcpy(model->coefficients, source->coefficients, coefficient_count * sizeof(double));
    memcpy(model->means, source->means, source->feature_count * sizeof(double));
    memcpy(model->scales, source->scales, source->feature_count * sizeof(double));
    *out = model;
    return SS_OK;
}

static SSError vm_fail(VM *vm, SSError code, size_t ip, const SSInstruction *ins,
                       const char *message) {
    if (vm->error.code != SS_OK && vm->error.message[0] != '\0') {
        vm->running = false;
        return code;
    }
    ss_error_set(&vm->error, code, ip, ins == NULL ? OP_NOP : ins->opcode,
                 ins == NULL ? 0u : ins->line, "%s", message);
    if (vm->frame_count > 0u && current_frame(vm)->function < vm->chunk->function_count) {
        const char *name = vm->chunk->functions[current_frame(vm)->function].name;
        (void)snprintf(vm->error.function, sizeof(vm->error.function), "%s", name);
    }
    vm->running = false;
    return code;
}

void *ss_vm_alloc(VM *vm, size_t size) {
    SSAllocation *allocation;
    if (size > vm->memory_limit - vm->allocated_bytes) return NULL;
    void *pointer = calloc(1u, size == 0 ? 1u : size);
    if (pointer == NULL) return NULL;
    allocation = malloc(sizeof(*allocation));
    if (allocation == NULL) { free(pointer); return NULL; }
    allocation->pointer = pointer;
    allocation->size = size;
    allocation->next = vm->allocations;
    vm->allocations = allocation;
    vm->allocated_bytes += size;
    vm->allocation_count += 1u;
    return pointer;
}

uint32_t ss_vm_random(VM *vm) {
    uint64_t old_state = vm->rng_state;
    uint32_t xor_shifted;
    uint32_t rotation;
    vm->rng_state = old_state * UINT64_C(6364136223846793005) + vm->rng_increment;
    xor_shifted = (uint32_t)(((old_state >> 18u) ^ old_state) >> 27u);
    rotation = (uint32_t)(old_state >> 59u);
    return (xor_shifted >> rotation) | (xor_shifted << ((0u - rotation) & 31u));
}

void ss_vm_seed(VM *vm, uint64_t seed) {
    vm->root_seed = seed;
    vm->rng_state = 0u;
    vm->rng_increment = (UINT64_C(1442695040888963407) << 1u) | 1u;
    (void)ss_vm_random(vm);
    vm->rng_state += seed;
    (void)ss_vm_random(vm);
}

void ss_vm_init(VM *vm, const SSChunk *chunk) {
    size_t frame_index, register_index;
    memset(vm, 0, sizeof(*vm));
    vm->chunk = chunk;
    vm->ip = chunk == NULL ? 0u : chunk->entry;
    vm->running = true;
    vm->instruction_limit = UINT64_C(10000000);
    vm->memory_limit = 64u * 1024u * 1024u;
    vm->frame_count = 1u;
    vm->frames[0].function = UINT32_MAX;
    vm->result = ss_value_null();
    vm->next_task_id = 1u;
    vm->next_group_id = 1u;
    {
        long processors = sysconf(_SC_NPROCESSORS_ONLN);
        vm->configured_worker_count = processors > 0 && processors < 8 ? (size_t)processors : 8u;
        if (vm->configured_worker_count == 0u) vm->configured_worker_count = 1u;
    }
    vm->configured_task_limit = 64u;
    vm->path_limit = 64u;
    vm->active_path_count = 1u;
    vm->next_dependency_id = 1u;
    atomic_init(&vm->cancelled, false);
    for (frame_index = 0; frame_index < SS_MAX_CALL_FRAMES; ++frame_index)
        for (register_index = 0; register_index < SS_MAX_REGISTERS; ++register_index)
            vm->frames[frame_index].registers[register_index] = ss_value_null();
    ss_vm_seed(vm, UINT64_C(0x4c414e41));
}

void ss_vm_set_program_args(VM *vm, int argc, const char **argv) {
    if (vm == NULL) return;
    vm->program_argc = argc;
    vm->program_argv = argv;
}

SSError ss_vm_set_worker_count(VM *vm, size_t workers) {
    if (vm == NULL || workers == 0u) return SS_ERR_TASK;
    if (vm->scheduler != NULL) return SS_ERR_TASK;
    vm->configured_worker_count = workers;
    return SS_OK;
}

SSError ss_vm_set_task_limit(VM *vm, size_t tasks) {
    if (vm == NULL || tasks == 0u) return SS_ERR_TASK;
    if (vm->scheduler != NULL) return SS_ERR_TASK;
    vm->configured_task_limit = tasks;
    return SS_OK;
}

static void cancel_task(SSTask *task) {
    if (task != NULL && task->child != NULL) atomic_store(&task->child->cancelled, true);
}

static void destroy_task(SSTask *task) {
    SSScheduler *scheduler;
    SSTask **cursor;
    if (task == NULL) return;
    cancel_task(task);
    if (task->child != NULL) { ss_vm_free(task->child); free(task->child); }
    scheduler = task->scheduler;
    if (scheduler != NULL) {
        (void)pthread_mutex_lock(&scheduler->mutex);
        cursor = &scheduler->all_tasks;
        while (*cursor != NULL && *cursor != task) cursor = &(*cursor)->all_next;
        if (*cursor == task) *cursor = task->all_next;
        if (!task->joined && scheduler->live_tasks > 0u) --scheduler->live_tasks;
        (void)pthread_mutex_unlock(&scheduler->mutex);
    }
    (void)pthread_cond_destroy(&task->completed_condition);
    (void)pthread_mutex_destroy(&task->mutex);
    free(task);
}

void ss_vm_free(VM *vm) {
    size_t frame_index, register_index;
    SSAllocation *allocation;
    SSTask *task;
    if (vm == NULL) return;
    SSScheduler *owned_scheduler = vm->scheduler_owner ? vm->scheduler : NULL;
    if (owned_scheduler != NULL) scheduler_shutdown(owned_scheduler);
    task = vm->tasks;
    while (task != NULL) {
        SSTask *next = task->next;
        destroy_task(task);
        task = next;
    }
    vm->tasks = NULL;
    allocation = vm->allocations;
    while (allocation != NULL) {
        SSAllocation *next = allocation->next;
        free(allocation->pointer);
        free(allocation);
        allocation = next;
    }
    for (frame_index = 0; frame_index < SS_MAX_CALL_FRAMES; ++frame_index)
        for (register_index = 0; register_index < SS_MAX_REGISTERS; ++register_index)
            free(vm->frames[frame_index].histories[register_index].versions);
    vm->allocations = NULL;
    vm->allocated_bytes = 0;
    if (owned_scheduler != NULL) {
        scheduler_destroy(owned_scheduler);
        vm->scheduler = NULL;
    }
}

static SSError clone_state_value(VM *destination, const SSStateValue *source,
                                 SSStateValue *out) {
    size_t length;
    char *source_copy;
    *out = *source;
    if (!source->indexes.has_source || source->indexes.source == NULL) return SS_OK;
    length = strlen(source->indexes.source);
    source_copy = ss_vm_alloc(destination, length + 1u);
    if (source_copy == NULL) return SS_ERR_OOM;
    memcpy(source_copy, source->indexes.source, length + 1u);
    out->indexes.source = source_copy;
    return SS_OK;
}

static SSError clone_history(VM *destination, const SSHistory *source,
                             SSHistory *out) {
    size_t index;
    if (source == out) return SS_OK;
    free(out->versions);
    memset(out, 0, sizeof(*out));
    out->policy = source->policy; out->amount = source->amount;
    if (source->count == 0u) return SS_OK;
    out->versions = calloc(source->count, sizeof(*out->versions));
    if (out->versions == NULL) return SS_ERR_OOM;
    out->capacity = source->count;
    for (index = 0; index < source->count; ++index) {
        SSError error = clone_state_value(destination, &source->versions[index],
                                          &out->versions[index]);
        if (error != SS_OK) return error;
        ++out->count;
    }
    return SS_OK;
}

typedef struct SSDistCloneMemo {
    const SSStateDist *source;
    SSStateDist *copy;
    struct SSDistCloneMemo *next;
} SSDistCloneMemo;

typedef struct SSContainerCloneMemo {
    const void *source;
    void *copy;
    ValueType type;
    struct SSContainerCloneMemo *next;
} SSContainerCloneMemo;

static SSError clone_state_dist_node(VM *destination, const SSStateDist *source,
                                     SSStateDist **out, SSDistCloneMemo **memo) {
    SSDistCloneMemo *entry;
    SSStateDist *copy;
    SSError error = SS_OK;
    if (source == NULL || out == NULL) return SS_ERR_INVALID_DISTRIBUTION;
    for (entry = *memo; entry != NULL; entry = entry->next) {
        if (entry->source == source) {
            *out = entry->copy;
            return SS_OK;
        }
    }
    copy = ss_vm_alloc(destination, sizeof(*copy));
    if (copy == NULL) return SS_ERR_OOM;
    entry = malloc(sizeof(*entry));
    if (entry == NULL) return SS_ERR_OOM;
    entry->source = source;
    entry->copy = copy;
    entry->next = *memo;
    *memo = entry;
    copy->kind = source->kind;
    switch (source->kind) {
        case SS_DIST_DIRAC:
            error = clone_state_value(destination, &source->as.dirac, &copy->as.dirac);
            break;
        case SS_DIST_APPEND:
            copy->as.append = source->as.append;
            error = clone_state_dist_node(destination, source->as.append.left,
                                          &copy->as.append.left, memo);
            if (error == SS_OK)
                error = clone_state_dist_node(destination, source->as.append.right,
                                              &copy->as.append.right, memo);
            break;
        case SS_DIST_TRANSFORM:
            copy->as.transform.transform_id = source->as.transform.transform_id;
            error = clone_state_dist_node(destination, source->as.transform.child,
                                          &copy->as.transform.child, memo);
            break;
        default:
            error = SS_ERR_INVALID_DISTRIBUTION;
            break;
    }
    return error;
}

static SSError clone_value_memo(VM *destination, const Value *source, Value *out,
                                SSDistCloneMemo **memo,
                                SSContainerCloneMemo **containers) {
    size_t index, length;
    SSError error;
    *out = *source;
    if (source->type == VAL_STRING) {
        char *copy;
        length = strlen(source->as.string);
        copy = ss_vm_alloc(destination, length + 1u);
        if (copy == NULL) return SS_ERR_OOM;
        memcpy(copy, source->as.string, length + 1u);
        out->as.string = copy;
    } else if (source->type == VAL_STATE) {
        error = clone_state_value(destination, &source->as.state, &out->as.state);
        if (error != SS_OK) return error;
    } else if (source->type == VAL_ARRAY) {
        SSArray *array;
        SSContainerCloneMemo *entry;
        for (entry = *containers; entry != NULL; entry = entry->next)
            if (entry->type == VAL_ARRAY && entry->source == source->as.array) {
                out->as.array = entry->copy; return SS_OK;
            }
        array = ss_vm_alloc(destination, sizeof(*array));
        if (array == NULL) return SS_ERR_OOM;
        entry = malloc(sizeof(*entry));
        if (entry == NULL) return SS_ERR_OOM;
        entry->source = source->as.array; entry->copy = array; entry->type = VAL_ARRAY;
        entry->next = *containers; *containers = entry;
        array->count = source->as.array->count;
        array->capacity = array->count;
        array->items = ss_vm_alloc(destination, array->count * sizeof(*array->items));
        if (array->items == NULL && array->count > 0u) return SS_ERR_OOM;
        for (index = 0; index < array->count; ++index) {
            error = clone_value_memo(destination, &source->as.array->items[index],
                                     &array->items[index], memo, containers);
            if (error != SS_OK) return error;
        }
        out->as.array = array;
    } else if (source->type == VAL_JOINT_STATE) {
        SSJointState *joint;
        SSContainerCloneMemo *entry;
        for (entry = *containers; entry != NULL; entry = entry->next)
            if (entry->type == VAL_JOINT_STATE && entry->source == source->as.joint) {
                out->as.joint = entry->copy; return SS_OK;
            }
        joint = ss_vm_alloc(destination, sizeof(*joint));
        if (joint == NULL) return SS_ERR_OOM;
        entry = malloc(sizeof(*entry));
        if (entry == NULL) return SS_ERR_OOM;
        entry->source = source->as.joint; entry->copy = joint;
        entry->type = VAL_JOINT_STATE; entry->next = *containers; *containers = entry;
        joint->count = source->as.joint->count;
        joint->kind = source->as.joint->kind;
        joint->capabilities = source->as.joint->capabilities;
        joint->row_count = source->as.joint->row_count;
        joint->names = ss_vm_alloc(destination, joint->count * sizeof(*joint->names));
        joint->domains = ss_vm_alloc(destination, joint->count * sizeof(*joint->domains));
        joint->values = source->as.joint->values == NULL ? NULL :
            ss_vm_alloc(destination, joint->count * sizeof(*joint->values));
        joint->rows = joint->row_count == 0u ? NULL :
            ss_vm_alloc(destination, joint->row_count * sizeof(*joint->rows));
        if ((joint->names == NULL || joint->domains == NULL ||
             (source->as.joint->values != NULL && joint->values == NULL) ||
             (joint->row_count > 0u && joint->rows == NULL)) && joint->count > 0u)
            return SS_ERR_OOM;
        for (index = 0; index < joint->count; ++index) {
            size_t length = strlen(source->as.joint->names[index]);
            joint->names[index] = ss_vm_alloc(destination, length + 1u);
            if (joint->names[index] == NULL) return SS_ERR_OOM;
            memcpy(joint->names[index], source->as.joint->names[index], length + 1u);
            joint->domains[index] = source->as.joint->domains[index];
            if (joint->values != NULL) {
                error = clone_value_memo(destination, &source->as.joint->values[index],
                                         &joint->values[index], memo, containers);
                if (error != SS_OK) return error;
            }
        }
        for (index = 0; index < joint->row_count; ++index) {
            size_t column;
            joint->rows[index].weight = source->as.joint->rows[index].weight;
            joint->rows[index].values = ss_vm_alloc(
                destination, joint->count * sizeof(*joint->rows[index].values));
            if (joint->rows[index].values == NULL && joint->count > 0u) return SS_ERR_OOM;
            for (column = 0; column < joint->count; ++column) {
                error = clone_value_memo(destination,
                    &source->as.joint->rows[index].values[column],
                    &joint->rows[index].values[column], memo, containers);
                if (error != SS_OK) return error;
            }
        }
        out->as.joint = joint;
    } else if (source->type == VAL_MODEL) {
        error = clone_model(destination, source->as.model, &out->as.model);
        if (error != SS_OK) return error;
    } else if (source->type == VAL_STATE_DIST) {
        error = clone_state_dist_node(destination, source->as.state_dist,
                                      &out->as.state_dist, memo);
        if (error != SS_OK) return error;
    } else if (source->type == VAL_MAP) {
        SSMap *map;
        SSContainerCloneMemo *entry;
        for (entry = *containers; entry != NULL; entry = entry->next)
            if (entry->type == VAL_MAP && entry->source == source->as.map) {
                out->as.map = entry->copy; return SS_OK;
            }
        error = ss_map_new(destination, source->as.map->count, &map);
        if (error != SS_OK) return error;
        entry = malloc(sizeof(*entry));
        if (entry == NULL) return SS_ERR_OOM;
        entry->source = source->as.map; entry->copy = map; entry->type = VAL_MAP;
        entry->next = *containers; *containers = entry;
        out->as.map = map;
        for (index = 0; index < source->as.map->count; ++index) {
            Value cloned;
            error = clone_value_memo(destination, source->as.map->entries[index].value,
                                     &cloned, memo, containers);
            if (error != SS_OK) return error;
            error = ss_map_set(destination, map, source->as.map->entries[index].key,
                               &cloned, true);
            if (error != SS_OK) return error;
        }
    } else if (source->type == VAL_POSSIBILITY) {
        SSPossibility *possibility;
        SSContainerCloneMemo *entry;
        for (entry = *containers; entry != NULL; entry = entry->next)
            if (entry->type == VAL_POSSIBILITY &&
                entry->source == source->as.possibility) {
                out->as.possibility = entry->copy; return SS_OK;
            }
        possibility = ss_vm_alloc(destination, sizeof(*possibility));
        if (possibility == NULL) return SS_ERR_OOM;
        entry = malloc(sizeof(*entry));
        if (entry == NULL) return SS_ERR_OOM;
        entry->source = source->as.possibility; entry->copy = possibility;
        entry->type = VAL_POSSIBILITY; entry->next = *containers; *containers = entry;
        possibility->count = source->as.possibility->count;
        possibility->dependency_id = source->as.possibility->dependency_id;
        possibility->values = ss_vm_alloc(destination,
            possibility->count * sizeof(*possibility->values));
        possibility->weights = source->as.possibility->weights == NULL ? NULL :
            ss_vm_alloc(destination, possibility->count * sizeof(*possibility->weights));
        if (possibility->values == NULL ||
            (source->as.possibility->weights != NULL && possibility->weights == NULL))
            return SS_ERR_OOM;
        for (index = 0; index < possibility->count; ++index) {
            error = clone_value_memo(destination, &source->as.possibility->values[index],
                                     &possibility->values[index], memo, containers);
            if (error != SS_OK) return error;
            if (possibility->weights != NULL)
                possibility->weights[index] = source->as.possibility->weights[index];
        }
        out->as.possibility = possibility;
    } else if (source->type == VAL_PATH_SET) {
        SSPathSet *paths;
        SSContainerCloneMemo *entry;
        for (entry = *containers; entry != NULL; entry = entry->next)
            if (entry->type == VAL_PATH_SET && entry->source == source->as.paths) {
                out->as.paths = entry->copy; return SS_OK;
            }
        paths = ss_vm_alloc(destination, sizeof(*paths));
        if (paths == NULL) return SS_ERR_OOM;
        entry = malloc(sizeof(*entry));
        if (entry == NULL) return SS_ERR_OOM;
        entry->source = source->as.paths; entry->copy = paths;
        entry->type = VAL_PATH_SET; entry->next = *containers; *containers = entry;
        paths->count = source->as.paths->count;
        paths->dependency_id = source->as.paths->dependency_id;
        paths->alternatives = ss_vm_alloc(destination,
            paths->count * sizeof(*paths->alternatives));
        if (paths->alternatives == NULL) return SS_ERR_OOM;
        for (index = 0; index < paths->count; ++index) {
            paths->alternatives[index].guard =
                source->as.paths->alternatives[index].guard;
            paths->alternatives[index].weight =
                source->as.paths->alternatives[index].weight;
            paths->alternatives[index].result = ss_vm_alloc(destination, sizeof(Value));
            if (paths->alternatives[index].result == NULL) return SS_ERR_OOM;
            error = clone_value_memo(destination,
                source->as.paths->alternatives[index].result,
                paths->alternatives[index].result, memo, containers);
            if (error != SS_OK) return error;
        }
        out->as.paths = paths;
    } else if (source->type == VAL_TASK) {
        return SS_ERR_TYPE;
    }
    return SS_OK;
}

static SSError clone_value(VM *destination, const Value *source, Value *out) {
    SSDistCloneMemo *memo = NULL;
    SSContainerCloneMemo *containers = NULL;
    SSDistCloneMemo *entry;
    SSError error = clone_value_memo(destination, source, out, &memo, &containers);
    while (memo != NULL) {
        entry = memo;
        memo = memo->next;
        free(entry);
    }
    while (containers != NULL) {
        SSContainerCloneMemo *entry = containers;
        containers = containers->next;
        free(entry);
    }
    return error;
}

typedef struct {
    char *name;
    size_t source_index;
} SSJointName;

static int joint_name_compare(const void *left, const void *right) {
    const SSJointName *a = left;
    const SSJointName *b = right;
    return strcmp(a->name, b->name);
}

static bool joint_value_equal(const Value *left, const Value *right) {
    if (left->type != right->type) return false;
    switch (left->type) {
        case VAL_NULL: return true;
        case VAL_NUMBER: return left->as.number == right->as.number;
        case VAL_BOOL: return left->as.boolean == right->as.boolean;
        case VAL_STRING: return strcmp(left->as.string, right->as.string) == 0;
        case VAL_SAMPLE: return left->as.sample == right->as.sample;
        case VAL_STATE:
            return left->as.state.state.p == right->as.state.state.p &&
                   left->as.state.state.d_re == right->as.state.state.d_re &&
                   left->as.state.state.d_im == right->as.state.state.d_im;
        case VAL_ARRAY: return left->as.array == right->as.array;
        case VAL_MAP: return left->as.map == right->as.map;
        case VAL_JOINT_STATE: return left->as.joint == right->as.joint;
        case VAL_STATE_DIST: return left->as.state_dist == right->as.state_dist;
        case VAL_POSSIBILITY: return left->as.possibility == right->as.possibility;
        case VAL_PATH_SET: return left->as.paths == right->as.paths;
        default: return false;
    }
}

SSError ss_vm_possibility_build(VM *vm, const Value *values, size_t count,
                                SSPossibility **out) {
    SSPossibility *possibility;
    size_t index, unique_count = 0u;
    Value *unique;
    SSError error;
    if (vm == NULL || values == NULL || out == NULL || count == 0u)
        return SS_ERR_FORMAT;
    unique = calloc(count, sizeof(*unique));
    if (unique == NULL) return SS_ERR_OOM;
    for (index = 0; index < count; ++index) {
        size_t existing;
        if (!joint_value_is_definite(&values[index])) {
            free(unique); return SS_ERR_TYPE;
        }
        for (existing = 0; existing < unique_count; ++existing)
            if (joint_value_equal(&values[index], &unique[existing])) break;
        if (existing == unique_count) unique[unique_count++] = values[index];
    }
    possibility = ss_vm_alloc(vm, sizeof(*possibility));
    if (possibility == NULL) { free(unique); return SS_ERR_OOM; }
    possibility->count = unique_count;
    possibility->weights = NULL;
    possibility->dependency_id = vm->next_dependency_id++;
    possibility->values = ss_vm_alloc(vm, unique_count * sizeof(*possibility->values));
    if (possibility->values == NULL) { free(unique); return SS_ERR_OOM; }
    for (index = 0; index < unique_count; ++index) {
        error = clone_value(vm, &unique[index], &possibility->values[index]);
        if (error != SS_OK) { free(unique); return error; }
    }
    free(unique); *out = possibility; return SS_OK;
}

static SSError snapshot_frames(VM *vm, SSFrame **out) {
    SSFrame *frames;
    size_t frame_index, register_index, history_index;
    SSError error;
    frames = ss_vm_alloc(vm, vm->frame_count * sizeof(*frames));
    if (frames == NULL) return SS_ERR_OOM;
    memcpy(frames, vm->frames, vm->frame_count * sizeof(*frames));
    for (frame_index = 0; frame_index < vm->frame_count; ++frame_index) {
        for (register_index = 0; register_index < SS_MAX_REGISTERS; ++register_index) {
            error = clone_value(vm, &vm->frames[frame_index].registers[register_index],
                                &frames[frame_index].registers[register_index]);
            if (error != SS_OK) return error;
        }
        for (history_index = 0; history_index < SS_MAX_REGISTERS; ++history_index) {
            SSHistory *history = &frames[frame_index].histories[history_index];
            const SSHistory *source = &vm->frames[frame_index].histories[history_index];
            size_t version;
            if (source->count == 0u) { history->versions = NULL; continue; }
            history->versions = ss_vm_alloc(vm, source->count * sizeof(*history->versions));
            if (history->versions == NULL) return SS_ERR_OOM;
            history->capacity = source->count;
            for (version = 0; version < source->count; ++version) {
                error = clone_state_value(vm, &source->versions[version],
                                          &history->versions[version]);
                if (error != SS_OK) return error;
            }
        }
    }
    *out = frames; return SS_OK;
}

static SSError path_split(VM *vm, const Value *condition, size_t false_ip) {
    const SSPossibility *possibility;
    bool has_true = false, has_false = false;
    double true_weight = 0.0, false_weight = 0.0;
    size_t index;
    SSPathExecution *execution;
    SSError error;
    if (condition->type == VAL_BOOL) {
        if (!condition->as.boolean) vm->ip = false_ip;
        return SS_OK;
    }
    if (condition->type != VAL_POSSIBILITY) return SS_ERR_TYPE;
    possibility = condition->as.possibility;
    for (index = 0; index < possibility->count; ++index) {
        double weight = possibility->weights == NULL
            ? 1.0 / (double)possibility->count : possibility->weights[index];
        if (possibility->values[index].type != VAL_BOOL) return SS_ERR_TYPE;
        if (possibility->values[index].as.boolean) {
            has_true = true; true_weight += weight;
        } else {
            has_false = true; false_weight += weight;
        }
    }
    if (!has_true) { vm->ip = false_ip; return SS_OK; }
    if (!has_false) return SS_OK;
    if (vm->active_path_count > vm->path_limit / 2u) return SS_ERR_PATH_LIMIT;
    execution = ss_vm_alloc(vm, sizeof(*execution));
    if (execution == NULL) return SS_ERR_OOM;
    error = snapshot_frames(vm, &execution->false_frames);
    if (error != SS_OK) return error;
    execution->frame_count = vm->frame_count;
    execution->false_ip = false_ip;
    execution->dependency_id = possibility->dependency_id;
    execution->true_weight = true_weight;
    execution->false_weight = false_weight;
    execution->previous_path_count = vm->active_path_count;
    execution->running_false = false;
    execution->next = vm->path_execution;
    vm->path_execution = execution;
    vm->active_path_count *= 2u;
    return SS_OK;
}

static SSError path_join(VM *vm) {
    SSPathExecution *execution = vm->path_execution;
    size_t frame_index, register_index, index;
    SSError error;
    if (execution == NULL) return SS_OK;
    if (!execution->running_false) {
        error = snapshot_frames(vm, &execution->true_frames);
        if (error != SS_OK) return error;
        vm->frame_count = execution->frame_count;
        memcpy(vm->frames, execution->false_frames,
               execution->frame_count * sizeof(*vm->frames));
        execution->running_false = true;
        vm->ip = execution->false_ip;
        return SS_OK;
    }
    if (vm->frame_count != execution->frame_count) return SS_ERR_UNSUPPORTED_OPERATION;
    for (frame_index = 0; frame_index < vm->frame_count; ++frame_index) {
        for (register_index = 0; register_index < SS_MAX_REGISTERS; ++register_index) {
            const Value *true_value =
                &execution->true_frames[frame_index].registers[register_index];
            Value *false_value = &vm->frames[frame_index].registers[register_index];
            SSPathSet *paths;
            if (joint_value_equal(true_value, false_value)) continue;
            if (execution->true_frames[frame_index].histories[register_index].policy !=
                    SS_HISTORY_NONE ||
                vm->frames[frame_index].histories[register_index].policy != SS_HISTORY_NONE)
                return SS_ERR_UNSUPPORTED_OPERATION;
            paths = ss_vm_alloc(vm, sizeof(*paths));
            if (paths == NULL) return SS_ERR_OOM;
            paths->count = 2u;
            paths->dependency_id = execution->dependency_id;
            paths->alternatives = ss_vm_alloc(vm, 2u * sizeof(*paths->alternatives));
            if (paths->alternatives == NULL) return SS_ERR_OOM;
            for (index = 0; index < 2u; ++index) {
                paths->alternatives[index].result = ss_vm_alloc(vm, sizeof(Value));
                if (paths->alternatives[index].result == NULL) return SS_ERR_OOM;
            }
            paths->alternatives[0].guard = true;
            paths->alternatives[0].weight = execution->true_weight;
            error = clone_value(vm, true_value, paths->alternatives[0].result);
            if (error != SS_OK) return error;
            paths->alternatives[1].guard = false;
            paths->alternatives[1].weight = execution->false_weight;
            error = clone_value(vm, false_value, paths->alternatives[1].result);
            if (error != SS_OK) return error;
            *false_value = ss_value_paths(paths);
        }
    }
    vm->active_path_count = execution->previous_path_count;
    vm->path_execution = execution->next;
    return SS_OK;
}

static SSError parse_joint_names(const char *text, size_t expected,
                                 SSJointKind *kind, char ***names_out,
                                 size_t *count_out) {
    char *copy, *cursor, *token;
    char **names;
    size_t count = 0u;
    if (text == NULL || *text == '\0') return SS_ERR_FORMAT;
    copy = malloc(strlen(text) + 1u);
    if (copy == NULL) return SS_ERR_OOM;
    strcpy(copy, text);
    cursor = strchr(copy, ':');
    if (cursor == NULL) { free(copy); return SS_ERR_FORMAT; }
    *cursor++ = '\0';
    if (strcmp(copy, "independent") == 0) *kind = SS_JOINT_INDEPENDENT;
    else if (strcmp(copy, "correlated") == 0) *kind = SS_JOINT_FINITE_LAW;
    else if (strcmp(copy, "conditional") == 0) *kind = SS_JOINT_CONDITIONAL;
    else { free(copy); return SS_ERR_FORMAT; }
    names = calloc(expected == 0u ? 1u : expected, sizeof(*names));
    if (names == NULL) { free(copy); return SS_ERR_OOM; }
    token = strtok(cursor, ",;");
    while (token != NULL) {
        size_t index;
        while (isspace((unsigned char)*token)) ++token;
        if (*token == '\0') { free(names); free(copy); return SS_ERR_FORMAT; }
        for (index = 0; index < count; ++index)
            if (strcmp(names[index], token) == 0) { free(names); free(copy); return SS_ERR_INVALID_DEPENDENCY; }
        if (count >= expected) { free(names); free(copy); return SS_ERR_FORMAT; }
        names[count] = malloc(strlen(token) + 1u);
        if (names[count] == NULL) { while (count > 0u) free(names[--count]); free(names); free(copy); return SS_ERR_OOM; }
        strcpy(names[count++], token);
        token = strtok(NULL, ",;");
    }
    free(copy);
    if (count != expected) { while (count > 0u) free(names[--count]); free(names); return SS_ERR_FORMAT; }
    *names_out = names;
    *count_out = count;
    return SS_OK;
}

static void free_joint_names(char **names, size_t count) {
    size_t index;
    for (index = 0; index < count; ++index) free(names[index]);
    free(names);
}

SSError ss_vm_joint_build(VM *vm, const Value *values, size_t count,
                          const char *descriptor, SSJointState **out) {
    SSJointKind kind;
    char **names = NULL;
    size_t name_count, index;
    SSJointName *ordered;
    SSJointState *joint;
    SSError error;
    if (vm == NULL || values == NULL || out == NULL || count == 0u) return SS_ERR_FORMAT;
    error = parse_joint_names(descriptor, count, &kind, &names, &name_count);
    if (error != SS_OK) return error;
    /* A correlation label plus unrelated marginals is not a joint law. */
    if (kind == SS_JOINT_FINITE_LAW) {
        free_joint_names(names, name_count);
        return SS_ERR_UNSUPPORTED_OPERATION;
    }
    ordered = calloc(count, sizeof(*ordered));
    if (ordered == NULL) { free_joint_names(names, name_count); return SS_ERR_OOM; }
    for (index = 0; index < count; ++index) { ordered[index].name = names[index]; ordered[index].source_index = index; }
    qsort(ordered, count, sizeof(*ordered), joint_name_compare);
    joint = ss_vm_alloc(vm, sizeof(*joint));
    if (joint == NULL) { free(ordered); free_joint_names(names, name_count); return SS_ERR_OOM; }
    joint->count = count; joint->kind = kind;
    joint->capabilities = kind == SS_JOINT_INDEPENDENT
        ? (SS_JOINT_CAN_PROJECT | SS_JOINT_CAN_CONDITION |
           SS_JOINT_CAN_SAMPLE | SS_JOINT_CAN_RESOLVE)
        : 0u;
    joint->row_count = 0u; joint->rows = NULL;
    joint->names = ss_vm_alloc(vm, count * sizeof(*joint->names));
    joint->domains = ss_vm_alloc(vm, count * sizeof(*joint->domains));
    joint->values = ss_vm_alloc(vm, count * sizeof(*joint->values));
    if (joint->names == NULL || joint->domains == NULL || joint->values == NULL) { free(ordered); free_joint_names(names, name_count); return SS_ERR_OOM; }
    for (index = 0; index < count; ++index) {
        size_t length = strlen(ordered[index].name);
        joint->names[index] = ss_vm_alloc(vm, length + 1u);
        if (joint->names[index] == NULL) { free(ordered); free_joint_names(names, name_count); return SS_ERR_OOM; }
        memcpy(joint->names[index], ordered[index].name, length + 1u);
        error = clone_value(vm, &values[ordered[index].source_index], &joint->values[index]);
        if (error != SS_OK) { free(ordered); free_joint_names(names, name_count); return error; }
        joint->domains[index].type = joint->values[index].type;
    }
    free(ordered); free_joint_names(names, name_count); *out = joint; return SS_OK;
}

static bool joint_value_is_definite(const Value *value) {
    return value != NULL && value->type != VAL_STATE_DIST &&
           value->type != VAL_JOINT_STATE && value->type != VAL_TASK &&
           value->type != VAL_FUNCTION && value->type != VAL_MODEL;
}

SSError ss_vm_joint_build_finite(VM *vm, const char *names_text,
                                 const Value *rows, const double *weights,
                                 size_t row_count, size_t variable_count,
                                 SSJointState **out) {
    char *descriptor;
    char **names = NULL;
    size_t name_count = 0u, row, column, unique_count = 0u;
    SSJointKind parsed_kind;
    SSJointName *ordered = NULL;
    Value *unique_values = NULL;
    double *unique_weights = NULL;
    double total = 0.0;
    SSJointState *joint;
    SSError error = SS_OK;
    if (vm == NULL || names_text == NULL || rows == NULL || weights == NULL ||
        out == NULL || row_count == 0u || variable_count == 0u)
        return SS_ERR_FORMAT;
    descriptor = malloc(strlen(names_text) + sizeof("correlated:"));
    if (descriptor == NULL) return SS_ERR_OOM;
    (void)snprintf(descriptor, strlen(names_text) + sizeof("correlated:"),
                   "correlated:%s", names_text);
    error = parse_joint_names(descriptor, variable_count, &parsed_kind,
                              &names, &name_count);
    free(descriptor);
    if (error != SS_OK) return error;
    ordered = calloc(variable_count, sizeof(*ordered));
    unique_values = calloc(row_count * variable_count, sizeof(*unique_values));
    unique_weights = calloc(row_count, sizeof(*unique_weights));
    if (ordered == NULL || unique_values == NULL || unique_weights == NULL) {
        error = SS_ERR_OOM; goto cleanup;
    }
    for (column = 0; column < variable_count; ++column) {
        ordered[column].name = names[column];
        ordered[column].source_index = column;
    }
    qsort(ordered, variable_count, sizeof(*ordered), joint_name_compare);
    for (row = 0; row < row_count; ++row) {
        size_t existing;
        bool found = false;
        if (!isfinite(weights[row]) || weights[row] <= 0.0) {
            error = SS_ERR_INVALID_DISTRIBUTION; goto cleanup;
        }
        total += weights[row];
        for (column = 0; column < variable_count; ++column) {
            const Value *value = &rows[row * variable_count + ordered[column].source_index];
            if (!joint_value_is_definite(value)) { error = SS_ERR_TYPE; goto cleanup; }
            if (row > 0u && value->type !=
                rows[ordered[column].source_index].type) {
                error = SS_ERR_TYPE; goto cleanup;
            }
        }
        for (existing = 0; existing < unique_count && !found; ++existing) {
            found = true;
            for (column = 0; column < variable_count; ++column) {
                const Value *value = &rows[row * variable_count + ordered[column].source_index];
                if (!joint_value_equal(value,
                        &unique_values[existing * variable_count + column])) {
                    found = false; break;
                }
            }
            if (found) unique_weights[existing] += weights[row];
        }
        if (!found) {
            for (column = 0; column < variable_count; ++column)
                unique_values[unique_count * variable_count + column] =
                    rows[row * variable_count + ordered[column].source_index];
            unique_weights[unique_count++] = weights[row];
        }
    }
    if (!isfinite(total) || fabs(total - 1.0) > 1e-12) {
        error = SS_ERR_INVALID_DISTRIBUTION; goto cleanup;
    }
    joint = ss_vm_alloc(vm, sizeof(*joint));
    if (joint == NULL) { error = SS_ERR_OOM; goto cleanup; }
    joint->count = variable_count;
    joint->kind = SS_JOINT_FINITE_LAW;
    joint->capabilities = SS_JOINT_CAN_PROJECT | SS_JOINT_CAN_CONDITION |
        SS_JOINT_CAN_SAMPLE | SS_JOINT_CAN_RESOLVE;
    joint->values = NULL;
    joint->row_count = unique_count;
    joint->names = ss_vm_alloc(vm, variable_count * sizeof(*joint->names));
    joint->domains = ss_vm_alloc(vm, variable_count * sizeof(*joint->domains));
    joint->rows = ss_vm_alloc(vm, unique_count * sizeof(*joint->rows));
    if (joint->names == NULL || joint->domains == NULL || joint->rows == NULL) {
        error = SS_ERR_OOM; goto cleanup;
    }
    for (column = 0; column < variable_count; ++column) {
        size_t length = strlen(ordered[column].name);
        joint->names[column] = ss_vm_alloc(vm, length + 1u);
        if (joint->names[column] == NULL) { error = SS_ERR_OOM; goto cleanup; }
        memcpy(joint->names[column], ordered[column].name, length + 1u);
        joint->domains[column].type = unique_values[column].type;
    }
    for (row = 0; row < unique_count; ++row) {
        joint->rows[row].weight = unique_weights[row] / total;
        joint->rows[row].values = ss_vm_alloc(
            vm, variable_count * sizeof(*joint->rows[row].values));
        if (joint->rows[row].values == NULL) { error = SS_ERR_OOM; goto cleanup; }
        for (column = 0; column < variable_count; ++column) {
            error = clone_value(vm, &unique_values[row * variable_count + column],
                                &joint->rows[row].values[column]);
            if (error != SS_OK) goto cleanup;
        }
    }
    *out = joint;
cleanup:
    free(ordered); free(unique_values); free(unique_weights);
    free_joint_names(names, name_count);
    return error;
}

static SSError joint_build_finite_array(VM *vm, const Value *rows_value,
                                        const char *names_text,
                                        SSJointState **out) {
    const SSArray *outer;
    size_t row, column, variable_count;
    Value *values;
    double *weights;
    SSError error;
    if (rows_value == NULL || rows_value->type != VAL_ARRAY ||
        rows_value->as.array == NULL || rows_value->as.array->count == 0u)
        return SS_ERR_TYPE;
    outer = rows_value->as.array;
    if (outer->items[0].type != VAL_ARRAY || outer->items[0].as.array == NULL ||
        outer->items[0].as.array->count < 2u) return SS_ERR_FORMAT;
    variable_count = outer->items[0].as.array->count - 1u;
    values = calloc(outer->count * variable_count, sizeof(*values));
    weights = calloc(outer->count, sizeof(*weights));
    if (values == NULL || weights == NULL) {
        free(values); free(weights); return SS_ERR_OOM;
    }
    for (row = 0; row < outer->count; ++row) {
        const SSArray *inner;
        if (outer->items[row].type != VAL_ARRAY ||
            outer->items[row].as.array == NULL) { error = SS_ERR_TYPE; goto done; }
        inner = outer->items[row].as.array;
        if (inner->count != variable_count + 1u) { error = SS_ERR_FORMAT; goto done; }
        if (inner->items[variable_count].type != VAL_NUMBER) {
            error = SS_ERR_TYPE; goto done;
        }
        weights[row] = inner->items[variable_count].as.number;
        for (column = 0; column < variable_count; ++column)
            values[row * variable_count + column] = inner->items[column];
    }
    error = ss_vm_joint_build_finite(vm, names_text, values, weights,
                                     outer->count, variable_count, out);
done:
    free(values); free(weights);
    return error;
}

static ssize_t joint_find(const SSJointState *joint, const char *name) {
    size_t index;
    if (joint == NULL || name == NULL) return -1;
    for (index = 0; index < joint->count; ++index)
        if (strcmp(joint->names[index], name) == 0) return (ssize_t)index;
    return -1;
}

SSError ss_vm_joint_project(VM *vm, const SSJointState *source,
                            const char *names_text, SSJointState **out) {
    char *copy, *token;
    size_t count = 0u, index;
    ssize_t *positions = NULL;
    Value *values = NULL;
    char descriptor[1024];
    SSError error;
    if (vm == NULL || source == NULL || out == NULL || names_text == NULL) return SS_ERR_FORMAT;
    if ((source->capabilities & SS_JOINT_CAN_PROJECT) == 0u)
        return SS_ERR_UNSUPPORTED_OPERATION;
    copy = malloc(strlen(names_text) + 1u); if (copy == NULL) return SS_ERR_OOM;
    strcpy(copy, names_text); token = strtok(copy, ",;");
    while (token != NULL) {
        if (joint_find(source, token) < 0) { free(copy); return SS_ERR_KEY; }
        ++count; token = strtok(NULL, ",;");
    }
    free(copy); if (count == 0u) return SS_ERR_FORMAT;
    positions = calloc(count, sizeof(*positions));
    if (positions == NULL) return SS_ERR_OOM;
    copy = malloc(strlen(names_text) + 1u); if (copy == NULL) { free(positions); return SS_ERR_OOM; }
    strcpy(copy, names_text); token = strtok(copy, ",;");
    for (index = 0; token != NULL; ++index, token = strtok(NULL, ",;")) {
        size_t previous;
        positions[index] = joint_find(source, token);
        for (previous = 0; previous < index; ++previous)
            if (positions[previous] == positions[index]) {
                free(copy); free(positions); return SS_ERR_INVALID_DEPENDENCY;
            }
    }
    free(copy);
    if (strlen(names_text) + sizeof("independent:") >= sizeof(descriptor)) {
        free(positions); return SS_ERR_LIMIT;
    }
    if (source->rows != NULL) {
        size_t row, column;
        double *weights = calloc(source->row_count, sizeof(*weights));
        values = calloc(source->row_count * count, sizeof(*values));
        if (weights == NULL || values == NULL) {
            free(weights); free(values); free(positions); return SS_ERR_OOM;
        }
        for (row = 0; row < source->row_count; ++row) {
            weights[row] = source->rows[row].weight;
            for (column = 0; column < count; ++column)
                values[row * count + column] =
                    source->rows[row].values[positions[column]];
        }
        error = ss_vm_joint_build_finite(vm, names_text, values, weights,
                                         source->row_count, count, out);
        free(weights); free(values);
    } else {
        values = calloc(count, sizeof(*values));
        if (values == NULL) { free(positions); return SS_ERR_OOM; }
        for (index = 0; index < count; ++index)
            values[index] = source->values[positions[index]];
        (void)snprintf(descriptor, sizeof(descriptor), "independent:%s", names_text);
        error = ss_vm_joint_build(vm, values, count, descriptor, out);
        free(values);
    }
    free(positions);
    if (error == SS_OK) (*out)->kind = SS_JOINT_PROJECTED;
    return error;
}

SSError ss_vm_joint_condition(VM *vm, const SSJointState *source,
                              const char *name, const Value *evidence,
                              SSJointState **out) {
    ssize_t position = joint_find(source, name);
    size_t index;
    if (vm == NULL || source == NULL || evidence == NULL || out == NULL) return SS_ERR_FORMAT;
    if ((source->capabilities & SS_JOINT_CAN_CONDITION) == 0u)
        return SS_ERR_UNSUPPORTED_OPERATION;
    if (position < 0) return SS_ERR_KEY;
    if (source->rows != NULL) {
        size_t row, kept = 0u, names_length = 1u;
        Value *rows;
        double *weights;
        char *names_text;
        SSError error;
        for (row = 0; row < source->row_count; ++row)
            if (joint_value_equal(&source->rows[row].values[position], evidence)) ++kept;
        if (kept == 0u) return SS_ERR_INVALID_CONDITIONING;
        rows = calloc(kept * source->count, sizeof(*rows));
        weights = calloc(kept, sizeof(*weights));
        for (index = 0; index < source->count; ++index)
            names_length += strlen(source->names[index]) + 1u;
        names_text = malloc(names_length);
        if (rows == NULL || weights == NULL || names_text == NULL) {
            free(rows); free(weights); free(names_text); return SS_ERR_OOM;
        }
        names_text[0] = '\0';
        for (index = 0; index < source->count; ++index) {
            if (index > 0u) strcat(names_text, ",");
            strcat(names_text, source->names[index]);
        }
        kept = 0u;
        for (row = 0; row < source->row_count; ++row) {
            if (!joint_value_equal(&source->rows[row].values[position], evidence)) continue;
            memcpy(&rows[kept * source->count], source->rows[row].values,
                   source->count * sizeof(*rows));
            weights[kept++] = source->rows[row].weight;
        }
        {
            double mass = 0.0;
            for (row = 0; row < kept; ++row) mass += weights[row];
            for (row = 0; row < kept; ++row) weights[row] /= mass;
        }
        error = ss_vm_joint_build_finite(vm, names_text, rows, weights,
                                         kept, source->count, out);
        free(rows); free(weights); free(names_text);
        if (error == SS_OK) (*out)->kind = SS_JOINT_CONDITIONAL;
        return error;
    }
    if (!joint_value_is_definite(&source->values[position]))
        return SS_ERR_UNSUPPORTED_OPERATION;
    if (!joint_value_equal(&source->values[position], evidence))
        return SS_ERR_INVALID_CONDITIONING;
    {
        Value wrapped = {.type = VAL_JOINT_STATE, .as.joint = (SSJointState *)source};
        Value cloned;
        SSError error = clone_value(vm, &wrapped, &cloned);
        if (error != SS_OK) return error;
        cloned.as.joint->kind = SS_JOINT_CONDITIONAL;
        *out = cloned.as.joint;
    }
    return SS_OK;
}

SSError ss_vm_joint_observe(VM *vm, const SSJointState *source,
                            const char *name, const Value *evidence,
                            SSJointState **out) {
    SSError error;
    if (vm == NULL) return SS_ERR_FORMAT;
    if (vm->active_path_count > 1u) return SS_ERR_UNSUPPORTED_OPERATION;
    error = ss_vm_joint_condition(vm, source, name, evidence, out);
    if (error == SS_OK) ++vm->observation_count;
    return error;
}

SSError ss_vm_joint_sample(VM *vm, const SSJointState *source, Value *out) {
    SSArray *array; size_t index; SSError error;
    if (vm == NULL || source == NULL || out == NULL) return SS_ERR_FORMAT;
    if ((source->capabilities & SS_JOINT_CAN_SAMPLE) == 0u)
        return SS_ERR_UNSUPPORTED_OPERATION;
    array = ss_vm_alloc(vm, sizeof(*array)); if (array == NULL) return SS_ERR_OOM;
    array->count = source->count; array->capacity = array->count; array->items = ss_vm_alloc(vm, array->count * sizeof(*array->items));
    if (array->items == NULL && array->count > 0u) return SS_ERR_OOM;
    if (source->rows != NULL) {
        double draw, cumulative = 0.0;
        size_t selected = source->row_count - 1u;
        error = consume_sampling_budget(vm);
        if (error != SS_OK) return error;
        draw = (double)ss_vm_random(vm) / 4294967296.0;
        for (index = 0; index < source->row_count; ++index) {
            cumulative += source->rows[index].weight;
            if (draw < cumulative) { selected = index; break; }
        }
        for (index = 0; index < source->count; ++index) {
            error = clone_value(vm, &source->rows[selected].values[index],
                                &array->items[index]);
            if (error != SS_OK) return error;
        }
        *out = ss_value_array(array); return SS_OK;
    }
    for (index = 0; index < source->count; ++index) {
        if (source->values[index].type == VAL_STATE_DIST) {
            SSStateValue state;
            error = ss_vm_state_dist_sample(vm, source->values[index].as.state_dist, &state);
            if (error != SS_OK) return error;
            array->items[index] = (Value){.type = VAL_STATE, .as.state = state};
        } else { error = clone_value(vm, &source->values[index], &array->items[index]); if (error != SS_OK) return error; }
    }
    *out = ss_value_array(array); return SS_OK;
}

SSError ss_vm_joint_resolve(VM *vm, const SSJointState *source, Value *out) {
    const Value *values;
    size_t index;
    SSArray *array;
    SSError error;
    if (vm == NULL || source == NULL || out == NULL) return SS_ERR_FORMAT;
    if ((source->capabilities & SS_JOINT_CAN_RESOLVE) == 0u)
        return SS_ERR_UNSUPPORTED_OPERATION;
    if (source->rows != NULL) {
        if (source->row_count != 1u) return SS_ERR_UNRESOLVED_VALUE;
        values = source->rows[0].values;
    } else {
        values = source->values;
        for (index = 0; index < source->count; ++index)
            if (!joint_value_is_definite(&values[index]))
                return SS_ERR_UNRESOLVED_VALUE;
    }
    if (source->count == 1u) return clone_value(vm, &values[0], out);
    array = ss_vm_alloc(vm, sizeof(*array));
    if (array == NULL) return SS_ERR_OOM;
    array->count = source->count;
    array->capacity = array->count;
    array->items = ss_vm_alloc(vm, array->count * sizeof(*array->items));
    if (array->items == NULL) return SS_ERR_OOM;
    for (index = 0; index < array->count; ++index) {
        error = clone_value(vm, &values[index], &array->items[index]);
        if (error != SS_OK) return error;
    }
    *out = ss_value_array(array);
    return SS_OK;
}

SSError ss_vm_information_resolve(VM *vm, const Value *source, Value *out) {
    size_t index;
    if (vm == NULL || source == NULL || out == NULL) return SS_ERR_FORMAT;
    if (source->type == VAL_JOINT_STATE)
        return ss_vm_joint_resolve(vm, source->as.joint, out);
    if (source->type == VAL_POSSIBILITY) {
        if (source->as.possibility->count != 1u) return SS_ERR_UNRESOLVED_VALUE;
        return clone_value(vm, &source->as.possibility->values[0], out);
    }
    if (source->type == VAL_PATH_SET) {
        const SSPathSet *paths = source->as.paths;
        if (paths->count == 0u) return SS_ERR_UNRESOLVED_VALUE;
        for (index = 1u; index < paths->count; ++index)
            if (!joint_value_equal(paths->alternatives[0].result,
                                   paths->alternatives[index].result))
                return SS_ERR_UNRESOLVED_VALUE;
        return clone_value(vm, paths->alternatives[0].result, out);
    }
    if (!joint_value_is_definite(source)) return SS_ERR_UNRESOLVED_VALUE;
    return clone_value(vm, source, out);
}

SSError ss_vm_information_sample(VM *vm, const Value *source, Value *out) {
    size_t selected;
    SSError error;
    if (vm == NULL || source == NULL || out == NULL) return SS_ERR_FORMAT;
    if (source->type == VAL_JOINT_STATE)
        return ss_vm_joint_sample(vm, source->as.joint, out);
    if (source->type == VAL_STATE_DIST) {
        SSStateValue state;
        error = ss_vm_state_dist_sample(vm, source->as.state_dist, &state);
        if (error == SS_OK) *out = (Value){.type = VAL_STATE, .as.state = state};
        return error;
    }
    error = consume_sampling_budget(vm);
    if (error != SS_OK) return error;
    if (source->type == VAL_POSSIBILITY) {
        selected = (size_t)(ss_vm_random(vm) % source->as.possibility->count);
        return clone_value(vm, &source->as.possibility->values[selected], out);
    }
    if (source->type == VAL_PATH_SET) {
        double draw = (double)ss_vm_random(vm) / 4294967296.0;
        double cumulative = 0.0;
        const SSPathSet *paths = source->as.paths;
        selected = paths->count - 1u;
        for (size_t index = 0; index < paths->count; ++index) {
            cumulative += paths->alternatives[index].weight;
            if (draw < cumulative) { selected = index; break; }
        }
        return clone_value(vm, paths->alternatives[selected].result, out);
    }
    return SS_ERR_TYPE;
}

static bool value_is_unresolved(const Value *value) {
    size_t index;
    if (value == NULL) return false;
    if (value->type == VAL_POSSIBILITY || value->type == VAL_PATH_SET)
        return true;
    if (value->type == VAL_ARRAY && value->as.array != NULL) {
        for (index = 0; index < value->as.array->count; ++index)
            if (value_is_unresolved(&value->as.array->items[index])) return true;
    }
    if (value->type == VAL_MAP && value->as.map != NULL) {
        for (index = 0; index < value->as.map->count; ++index)
            if (value_is_unresolved(value->as.map->entries[index].value)) return true;
    }
    return false;
}

SSError ss_vm_joint_rename(VM *vm, const SSJointState *source,
                           const char *old_name, const char *new_name,
                           SSJointState **out) {
    ssize_t position;
    size_t index, length = 1u;
    char *names_text;
    SSError error;
    if (vm == NULL || source == NULL || old_name == NULL || new_name == NULL ||
        out == NULL || *new_name == '\0') return SS_ERR_FORMAT;
    position = joint_find(source, old_name);
    if (position < 0) return SS_ERR_KEY;
    if (joint_find(source, new_name) >= 0) return SS_ERR_INVALID_DEPENDENCY;
    for (index = 0; index < source->count; ++index)
        length += strlen(index == (size_t)position ? new_name : source->names[index]) + 1u;
    names_text = malloc(length);
    if (names_text == NULL) return SS_ERR_OOM;
    names_text[0] = '\0';
    for (index = 0; index < source->count; ++index) {
        if (index > 0u) strcat(names_text, ",");
        strcat(names_text, index == (size_t)position ? new_name : source->names[index]);
    }
    if (source->rows != NULL) {
        Value *rows = calloc(source->row_count * source->count, sizeof(*rows));
        double *weights = calloc(source->row_count, sizeof(*weights));
        size_t row;
        if (rows == NULL || weights == NULL) {
            free(rows); free(weights); free(names_text); return SS_ERR_OOM;
        }
        for (row = 0; row < source->row_count; ++row) {
            memcpy(&rows[row * source->count], source->rows[row].values,
                   source->count * sizeof(*rows));
            weights[row] = source->rows[row].weight;
        }
        error = ss_vm_joint_build_finite(vm, names_text, rows, weights,
                                         source->row_count, source->count, out);
        free(rows); free(weights);
    } else {
        char *descriptor = malloc(strlen(names_text) + sizeof("independent:"));
        if (descriptor == NULL) { free(names_text); return SS_ERR_OOM; }
        (void)snprintf(descriptor, strlen(names_text) + sizeof("independent:"),
                       "independent:%s", names_text);
        error = ss_vm_joint_build(vm, source->values, source->count,
                                  descriptor, out);
        free(descriptor);
    }
    free(names_text);
    return error;
}

static void *run_task(void *context) {
    SSTask *task = context;
    task->status = ss_vm_run(task->child);
    if (task->status == SS_OK) task->result = task->child->result;
    else task->error = task->child->error;
    (void)pthread_mutex_lock(&task->mutex);
    task->completed = true;
    (void)pthread_cond_broadcast(&task->completed_condition);
    (void)pthread_mutex_unlock(&task->mutex);
    return NULL;
}

static SSTask *scheduler_take_locked(SSScheduler *scheduler) {
    SSTask *task = scheduler->queue_head;
    if (task != NULL) {
        scheduler->queue_head = task->queue_next;
        if (scheduler->queue_head == NULL) scheduler->queue_tail = NULL;
        task->queue_next = NULL; task->queued = false;
    }
    return task;
}

static void *scheduler_worker(void *context) {
    SSScheduler *scheduler = context;
    for (;;) {
        SSTask *task;
        (void)pthread_mutex_lock(&scheduler->mutex);
        while (scheduler->queue_head == NULL && !scheduler->stopping)
            (void)pthread_cond_wait(&scheduler->available, &scheduler->mutex);
        if (scheduler->queue_head == NULL && scheduler->stopping) {
            (void)pthread_mutex_unlock(&scheduler->mutex);
            return NULL;
        }
        task = scheduler_take_locked(scheduler);
        (void)pthread_mutex_unlock(&scheduler->mutex);
        (void)run_task(task);
    }
}

static SSError scheduler_initialize(VM *owner) {
    SSScheduler *scheduler;
    size_t index;
    if (owner->scheduler != NULL) return SS_OK;
    scheduler = calloc(1u, sizeof(*scheduler));
    if (scheduler == NULL) return SS_ERR_OOM;
    scheduler->worker_count = owner->configured_worker_count;
    scheduler->task_limit = owner->configured_task_limit;
    scheduler->next_task_id = 1u;
    scheduler->workers = calloc(scheduler->worker_count, sizeof(*scheduler->workers));
    if (scheduler->workers == NULL || pthread_mutex_init(&scheduler->mutex, NULL) != 0 ||
        pthread_cond_init(&scheduler->available, NULL) != 0) {
        free(scheduler->workers); free(scheduler); return SS_ERR_TASK;
    }
    owner->scheduler = scheduler; owner->scheduler_owner = true;
    for (index = 0; index < scheduler->worker_count; ++index) {
        if (pthread_create(&scheduler->workers[index], NULL, scheduler_worker, scheduler) != 0) {
            scheduler->worker_count = index; scheduler_shutdown(scheduler);
            scheduler_destroy(scheduler);
            owner->scheduler = NULL; owner->scheduler_owner = false;
            return SS_ERR_TASK;
        }
    }
    return SS_OK;
}

static void scheduler_shutdown(SSScheduler *scheduler) {
    SSTask *task;
    size_t index;
    if (scheduler == NULL) return;
    (void)pthread_mutex_lock(&scheduler->mutex);
    scheduler->stopping = true;
    for (task = scheduler->all_tasks; task != NULL; task = task->all_next) cancel_task(task);
    (void)pthread_cond_broadcast(&scheduler->available);
    (void)pthread_mutex_unlock(&scheduler->mutex);
    for (index = 0; index < scheduler->worker_count; ++index)
        (void)pthread_join(scheduler->workers[index], NULL);
}

static void scheduler_destroy(SSScheduler *scheduler) {
    (void)pthread_cond_destroy(&scheduler->available);
    (void)pthread_mutex_destroy(&scheduler->mutex);
    free(scheduler->workers);
    free(scheduler);
}

static SSError start_task(VM *parent, uint32_t function_index,
                          const Value *arguments, const SSHistory *argument_histories,
                          size_t argc, SSTask **out) {
    const SSFunction *function = &parent->chunk->functions[function_index];
    SSTask *task;
    size_t index;
    SSError error = SS_OK;
    SSDistCloneMemo *memo = NULL;
    SSContainerCloneMemo *containers = NULL;
    if (argc != function->arity) return SS_ERR_TYPE;
    error = scheduler_initialize(parent);
    if (error != SS_OK) return error;
    (void)pthread_mutex_lock(&parent->scheduler->mutex);
    if (parent->scheduler->stopping) error = SS_ERR_TASK;
    else if (parent->scheduler->live_tasks >= parent->scheduler->task_limit) error = SS_ERR_LIMIT;
    else ++parent->scheduler->live_tasks;
    (void)pthread_mutex_unlock(&parent->scheduler->mutex);
    if (error != SS_OK) return error;
    task = calloc(1u, sizeof(*task));
    if (task == NULL) { error = SS_ERR_OOM; goto release_slot; }
    task->child = malloc(sizeof(*task->child));
    if (task->child == NULL) { free(task); error = SS_ERR_OOM; goto release_slot; }
    if (pthread_mutex_init(&task->mutex, NULL) != 0) {
        free(task->child); free(task); error = SS_ERR_TASK; goto release_slot;
    }
    if (pthread_cond_init(&task->completed_condition, NULL) != 0) {
        (void)pthread_mutex_destroy(&task->mutex);
        free(task->child); free(task); error = SS_ERR_TASK; goto release_slot;
    }
    (void)pthread_mutex_lock(&parent->scheduler->mutex);
    task->id = parent->scheduler->next_task_id++;
    (void)pthread_mutex_unlock(&parent->scheduler->mutex);
    task->group_id = parent->current_group_id;
    task->scheduler = parent->scheduler;
    task->result = ss_value_null();
    ss_vm_init(task->child, parent->chunk);
    task->child->scheduler = parent->scheduler;
    task->child->scheduler_owner = false;
    task->child->ip = function->entry;
    task->child->frames[0].function = function_index;
    task->child->task_id = task->id;
    task->child->instruction_limit = parent->instruction_limit;
    task->child->memory_limit = parent->memory_limit;
    task->child->trace = parent->trace;
    task->child->lineage = mix64(parent->lineage ^ ++parent->spawn_counter);
    ss_vm_seed(task->child, mix64(parent->root_seed ^ task->child->lineage));
    task->child->root_seed = parent->root_seed;
    ss_vm_set_program_args(task->child, parent->program_argc, parent->program_argv);
    for (index = 0; index < argc && error == SS_OK; ++index)
        error = clone_value_memo(task->child, &arguments[index],
                                 &task->child->frames[0].registers[index], &memo,
                                 &containers);
    if (parent->chunk->version >= 3u)
        for (index = 0; index < argc && error == SS_OK; ++index)
            error = clone_history(task->child, &argument_histories[index],
                                  &task->child->frames[0].histories[index]);
    while (memo != NULL) {
        SSDistCloneMemo *entry = memo;
        memo = memo->next;
        free(entry);
    }
    while (containers != NULL) {
        SSContainerCloneMemo *entry = containers;
        containers = containers->next;
        free(entry);
    }
    if (error != SS_OK) { destroy_task(task); goto release_slot; }
    task->next = parent->tasks;
    parent->tasks = task;
    (void)pthread_mutex_lock(&parent->scheduler->mutex);
    task->queued = true;
    task->all_next = parent->scheduler->all_tasks;
    parent->scheduler->all_tasks = task;
    if (parent->scheduler->queue_tail == NULL) parent->scheduler->queue_head = task;
    else parent->scheduler->queue_tail->queue_next = task;
    parent->scheduler->queue_tail = task;
    (void)pthread_cond_signal(&parent->scheduler->available);
    (void)pthread_mutex_unlock(&parent->scheduler->mutex);
    *out = task;
    return SS_OK;
release_slot:
    (void)pthread_mutex_lock(&parent->scheduler->mutex);
    --parent->scheduler->live_tasks;
    (void)pthread_mutex_unlock(&parent->scheduler->mutex);
    return error;
}

static SSError wait_task(VM *vm, SSTask *task, double timeout_seconds, Value *out) {
    int wait_result = 0;
    struct timespec monotonic_deadline = {0};
    if (task == NULL) return SS_ERR_TASK;
    if (timeout_seconds >= 0.0) {
        (void)clock_gettime(CLOCK_MONOTONIC, &monotonic_deadline);
        monotonic_deadline.tv_sec += (time_t)timeout_seconds;
        monotonic_deadline.tv_nsec += (long)((timeout_seconds - floor(timeout_seconds)) * 1000000000.0);
        if (monotonic_deadline.tv_nsec >= 1000000000L) { ++monotonic_deadline.tv_sec; monotonic_deadline.tv_nsec -= 1000000000L; }
    }
    for (;;) {
        bool completed;
        (void)pthread_mutex_lock(&task->mutex);
        completed = task->completed;
        (void)pthread_mutex_unlock(&task->mutex);
        if (completed) break;
        if (timeout_seconds < 0.0) {
            SSTask *helper;
            (void)pthread_mutex_lock(&vm->scheduler->mutex);
            helper = scheduler_take_locked(vm->scheduler);
            (void)pthread_mutex_unlock(&vm->scheduler->mutex);
            if (helper != NULL) { (void)run_task(helper); continue; }
            (void)pthread_mutex_lock(&task->mutex);
            if (!task->completed) wait_result = pthread_cond_wait(&task->completed_condition, &task->mutex);
            (void)pthread_mutex_unlock(&task->mutex);
        } else {
            struct timespec now, realtime_deadline;
            double remaining;
            (void)clock_gettime(CLOCK_MONOTONIC, &now);
            remaining = (double)(monotonic_deadline.tv_sec - now.tv_sec) +
                        (double)(monotonic_deadline.tv_nsec - now.tv_nsec) / 1000000000.0;
            if (remaining <= 0.0) { wait_result = ETIMEDOUT; break; }
            if (remaining > 0.01) remaining = 0.01;
            (void)timespec_get(&realtime_deadline, TIME_UTC);
            realtime_deadline.tv_sec += (time_t)remaining;
            realtime_deadline.tv_nsec += (long)((remaining - floor(remaining)) * 1000000000.0);
            if (realtime_deadline.tv_nsec >= 1000000000L) { ++realtime_deadline.tv_sec; realtime_deadline.tv_nsec -= 1000000000L; }
            (void)pthread_mutex_lock(&task->mutex);
            if (!task->completed) wait_result = pthread_cond_timedwait(&task->completed_condition, &task->mutex, &realtime_deadline);
            (void)pthread_mutex_unlock(&task->mutex);
            if (wait_result == ETIMEDOUT) wait_result = 0;
        }
        if (wait_result != 0) break;
    }
    if (timeout_seconds >= 0.0) {
        bool completed;
        (void)pthread_mutex_lock(&task->mutex);
        completed = task->completed;
        (void)pthread_mutex_unlock(&task->mutex);
        if (!completed) wait_result = ETIMEDOUT;
    }
    if (wait_result == ETIMEDOUT) return SS_ERR_TIMEOUT;
    if (wait_result != 0) return SS_ERR_TASK;
    if (task->status != SS_OK) {
        vm->error = task->error;
        return task->status;
    }
    if (!task->joined) {
        Value cloned;
        SSError error = clone_value(vm, &task->result, &cloned);
        if (error != SS_OK) return error;
        task->result = cloned;
        ss_vm_free(task->child); free(task->child); task->child = NULL; task->joined = true;
        (void)pthread_mutex_lock(&vm->scheduler->mutex);
        if (vm->scheduler->live_tasks > 0u) --vm->scheduler->live_tasks;
        (void)pthread_mutex_unlock(&vm->scheduler->mutex);
    }
    *out = task->result;
    return SS_OK;
}

static SSError host_read_text(VM *vm, const Value *argument, Value *out) {
    FILE *file;
    long length;
    char *contents;
    if (argument->type != VAL_STRING) return SS_ERR_TYPE;
    file = fopen(argument->as.string, "rb");
    if (file == NULL) return SS_ERR_IO;
    if (fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0) { (void)fclose(file); return SS_ERR_IO; }
    if ((size_t)length > vm->memory_limit - vm->allocated_bytes) {
        (void)fclose(file); return SS_ERR_LIMIT;
    }
    contents = ss_vm_alloc(vm, (size_t)length + 1u);
    if (contents == NULL) { (void)fclose(file); return SS_ERR_OOM; }
    if (fread(contents, 1, (size_t)length, file) != (size_t)length) {
        (void)fclose(file); return SS_ERR_IO;
    }
    contents[length] = '\0';
    if (fclose(file) != 0) return SS_ERR_IO;
    *out = ss_value_string(contents);
    return SS_OK;
}

static SSError array_number(const Value *value, double *out) {
    if (value->type != VAL_NUMBER || !isfinite(value->as.number)) return SS_ERR_TYPE;
    *out = value->as.number;
    return SS_OK;
}

static SSError matrix_shape(const Value *value, size_t *rows, size_t *columns) {
    size_t row, column;
    if (value->type != VAL_ARRAY) return SS_ERR_TYPE;
    *rows = value->as.array->count;
    *columns = 0u;
    if (*rows == 0u) return SS_ERR_TYPE;
    for (row = 0; row < *rows; ++row) {
        const Value *line = &value->as.array->items[row];
        if (line->type != VAL_ARRAY) return SS_ERR_TYPE;
        if (row == 0u) *columns = line->as.array->count;
        if (line->as.array->count != *columns || *columns == 0u) return SS_ERR_TYPE;
        for (column = 0; column < *columns; ++column)
            if (array_number(&line->as.array->items[column], &(double){0}) != SS_OK) return SS_ERR_TYPE;
    }
    return SS_OK;
}

static SSError matrix_copy(const Value *value, double **out, size_t *rows, size_t *columns) {
    size_t row, column;
    SSError error = matrix_shape(value, rows, columns);
    double *data;
    if (error != SS_OK) return error;
    data = malloc(*rows * *columns * sizeof(double));
    if (data == NULL) return SS_ERR_OOM;
    for (row = 0; row < *rows; ++row)
        for (column = 0; column < *columns; ++column)
            data[row * *columns + column] = value->as.array->items[row].as.array->items[column].as.number;
    *out = data;
    return SS_OK;
}

static SSError vector_copy(const Value *value, double **out, size_t *count) {
    size_t index;
    double *data;
    if (value->type != VAL_ARRAY || value->as.array->count == 0u) return SS_ERR_TYPE;
    data = malloc(value->as.array->count * sizeof(double));
    if (data == NULL) return SS_ERR_OOM;
    for (index = 0; index < value->as.array->count; ++index) {
        if (array_number(&value->as.array->items[index], &data[index]) != SS_OK) { free(data); return SS_ERR_TYPE; }
    }
    *out = data; *count = value->as.array->count;
    return SS_OK;
}

static SSError make_number_array(VM *vm, const double *values, size_t count, Value *out) {
    SSArray *array = ss_vm_alloc(vm, sizeof(*array));
    if (array == NULL) return SS_ERR_OOM;
    array->count = count;
    array->capacity = count;
    array->items = ss_vm_alloc(vm, count * sizeof(*array->items));
    if (array->items == NULL && count > 0u) return SS_ERR_OOM;
    for (size_t index = 0; index < count; ++index) array->items[index] = ss_value_number(values[index]);
    out->type = VAL_ARRAY; out->as.array = array;
    return SS_OK;
}

static SSError make_matrix(VM *vm, const double *values, size_t rows, size_t columns, Value *out) {
    SSArray *outer = ss_vm_alloc(vm, sizeof(*outer));
    if (outer == NULL) return SS_ERR_OOM;
    outer->count = rows;
    outer->capacity = rows;
    outer->items = ss_vm_alloc(vm, rows * sizeof(*outer->items));
    if (outer->items == NULL && rows > 0u) return SS_ERR_OOM;
    for (size_t row = 0; row < rows; ++row) {
        SSArray *line = ss_vm_alloc(vm, sizeof(*line));
        if (line == NULL) return SS_ERR_OOM;
        line->count = columns;
        line->capacity = columns;
        line->items = ss_vm_alloc(vm, columns * sizeof(*line->items));
        if (line->items == NULL && columns > 0u) return SS_ERR_OOM;
        for (size_t column = 0; column < columns; ++column)
            line->items[column] = ss_value_number(values[row * columns + column]);
        outer->items[row].type = VAL_ARRAY; outer->items[row].as.array = line;
    }
    out->type = VAL_ARRAY; out->as.array = outer;
    return SS_OK;
}

static SSError fit_model(VM *vm, const Value *x_value, const Value *y_value,
                         SSMLModelKind kind, double regularization, size_t iterations, Value *out) {
    double *x = NULL, *y = NULL, *aug = NULL, *beta = NULL;
    size_t rows, columns, y_count, size, row, column, iteration;
    SSMLModel *model;
    SSError error = matrix_copy(x_value, &x, &rows, &columns);
    if (error != SS_OK) return error;
    error = vector_copy(y_value, &y, &y_count);
    if (error != SS_OK || y_count != rows || regularization < 0.0 || !isfinite(regularization)) { free(x); free(y); return SS_ERR_TYPE; }
    if (iterations == 0u || iterations > 100000u) { free(x); free(y); return SS_ERR_LIMIT; }
    model = ss_vm_alloc(vm, sizeof(*model));
    if (model == NULL) { free(x); free(y); return SS_ERR_OOM; }
    model->kind = kind; model->feature_count = columns; model->regularization = regularization;
    model->coefficients = ss_vm_alloc(vm, (columns + 1u) * sizeof(double));
    model->means = ss_vm_alloc(vm, columns * sizeof(double));
    model->scales = ss_vm_alloc(vm, columns * sizeof(double));
    if (model->coefficients == NULL || model->means == NULL || model->scales == NULL) { free(x); free(y); return SS_ERR_OOM; }
    for (column = 0; column < columns; ++column) {
        model->means[column] = 0.0;
        for (row = 0; row < rows; ++row) model->means[column] += x[row * columns + column];
        model->means[column] /= (double)rows; model->scales[column] = 0.0;
        for (row = 0; row < rows; ++row) { double delta = x[row * columns + column] - model->means[column]; model->scales[column] += delta * delta; }
        model->scales[column] = sqrt(model->scales[column] / (double)(rows > 1u ? rows - 1u : 1u));
        if (!isfinite(model->scales[column]) || model->scales[column] == 0.0) model->scales[column] = 1.0;
    }
    size = columns + 1u;
    beta = calloc(size, sizeof(double));
    if (beta == NULL) { free(x); free(y); return SS_ERR_OOM; }
    if (kind == SS_ML_RIDGE) {
        aug = calloc(size * (size + 1u), sizeof(double));
        if (aug == NULL) { free(x); free(y); free(beta); return SS_ERR_OOM; }
        for (row = 0; row < rows; ++row) {
            double *design = calloc(size, sizeof(double));
            if (design == NULL) { free(x); free(y); free(beta); free(aug); return SS_ERR_OOM; }
            design[0] = 1.0; for (column = 0; column < columns; ++column) design[column + 1u] = (x[row * columns + column] - model->means[column]) / model->scales[column];
            for (column = 0; column < size; ++column) { for (size_t other = 0; other < size; ++other) aug[column * (size + 1u) + other] += design[column] * design[other]; aug[column * (size + 1u) + size] += design[column] * y[row]; }
            free(design);
        }
        for (column = 1; column < size; ++column) aug[column * (size + 1u) + column] += regularization;
        for (column = 0; column < size; ++column) {
            size_t pivot = column;
            for (row = column + 1u; row < size; ++row) if (fabs(aug[row * (size + 1u) + column]) > fabs(aug[pivot * (size + 1u) + column])) pivot = row;
            if (fabs(aug[pivot * (size + 1u) + column]) < 1e-12) { free(x); free(y); free(beta); free(aug); return SS_ERR_TYPE; }
            for (size_t other = column; other <= size; ++other) { double swap = aug[column * (size + 1u) + other]; aug[column * (size + 1u) + other] = aug[pivot * (size + 1u) + other]; aug[pivot * (size + 1u) + other] = swap; }
            { double divisor = aug[column * (size + 1u) + column]; for (size_t other = column; other <= size; ++other) aug[column * (size + 1u) + other] /= divisor; }
            for (row = 0; row < size; ++row) if (row != column) { double factor = aug[row * (size + 1u) + column]; for (size_t other = column; other <= size; ++other) aug[row * (size + 1u) + other] -= factor * aug[column * (size + 1u) + other]; }
        }
        for (column = 0; column < size; ++column) beta[column] = aug[column * (size + 1u) + size];
    } else {
        for (iteration = 0; iteration < iterations; ++iteration) {
            double max_step = 0.0;
            for (row = 0; row < rows; ++row) {
                double logit = beta[0];
                for (column = 0; column < columns; ++column) logit += beta[column + 1u] * ((x[row * columns + column] - model->means[column]) / model->scales[column]);
                double probability = 1.0 / (1.0 + exp(-fmax(-30.0, fmin(30.0, logit))));
                double error_term = y[row] - probability;
                beta[0] += 0.25 * error_term / (double)rows;
                for (column = 0; column < columns; ++column) { double step = 0.25 * (error_term * ((x[row * columns + column] - model->means[column]) / model->scales[column]) / (double)rows - regularization * beta[column + 1u] / (double)rows); beta[column + 1u] += step; if (fabs(step) > max_step) max_step = fabs(step); }
            }
            if (max_step < 1e-3) break;
        }
        /* A bounded final iterate is still useful for inspection; callers can
           validate its metrics rather than receiving an unbounded computation. */
    }
    memcpy(model->coefficients, beta, size * sizeof(double));
    free(x); free(y); free(beta); free(aug); *out = ss_value_model(model); return SS_OK;
}

static SSError predict_model(VM *vm, const SSMLModel *model, const Value *features, Value *out) {
    double *x = NULL, *predictions = NULL; size_t rows, columns, row, column; SSError error = matrix_copy(features, &x, &rows, &columns);
    if (error != SS_OK || columns != model->feature_count) { free(x); return SS_ERR_TYPE; }
    predictions = malloc(rows * sizeof(double)); if (predictions == NULL) { free(x); return SS_ERR_OOM; }
    for (row = 0; row < rows; ++row) { double value = model->coefficients[0]; for (column = 0; column < columns; ++column) value += model->coefficients[column + 1u] * ((x[row * columns + column] - model->means[column]) / model->scales[column]); predictions[row] = model->kind == SS_ML_LOGISTIC ? 1.0 / (1.0 + exp(-fmax(-30.0, fmin(30.0, value)))) : value; }
    error = make_number_array(vm, predictions, rows, out); free(x); free(predictions); return error;
}

static SSError execute_host_call(VM *vm, uint32_t host_id, const Value *arguments,
                                 size_t argc, Value *out) {
    size_t index;
    *out = ss_value_null();
    switch ((SSHostCallId)host_id) {
        case SS_HOST_ARGS: {
            SSArray *array;
            if (argc != 0u) return SS_ERR_TYPE;
            array = ss_vm_alloc(vm, sizeof(*array));
            if (array == NULL) return SS_ERR_OOM;
            array->count = (size_t)vm->program_argc;
            array->capacity = array->count;
            array->items = ss_vm_alloc(vm, array->count * sizeof(*array->items));
            if (array->items == NULL && array->count > 0u) return SS_ERR_OOM;
            for (index = 0; index < array->count; ++index) {
                Value source = ss_value_string(vm->program_argv[index]);
                SSError error = clone_value(vm, &source, &array->items[index]);
                if (error != SS_OK) return error;
            }
            out->type = VAL_ARRAY; out->as.array = array; return SS_OK;
        }
        case SS_HOST_READ_TEXT:
            return argc == 1u ? host_read_text(vm, &arguments[0], out) : SS_ERR_TYPE;
        case SS_HOST_WRITE_TEXT: {
            FILE *file;
            size_t length;
            if (argc != 2u || arguments[0].type != VAL_STRING || arguments[1].type != VAL_STRING)
                return SS_ERR_TYPE;
            file = fopen(arguments[0].as.string, "wb");
            if (file == NULL) return SS_ERR_IO;
            length = strlen(arguments[1].as.string);
            if (fwrite(arguments[1].as.string, 1, length, file) != length) {
                (void)fclose(file); return SS_ERR_IO;
            }
            if (fclose(file) != 0) return SS_ERR_IO;
            return SS_OK;
        }
        case SS_HOST_NOW: {
            struct timespec now;
            if (argc != 0u || timespec_get(&now, TIME_UTC) != TIME_UTC) return SS_ERR_TYPE;
            *out = ss_value_number((double)now.tv_sec + (double)now.tv_nsec / 1000000000.0);
            return SS_OK;
        }
        case SS_HOST_RANDOM:
            if (argc != 0u) return SS_ERR_TYPE;
            *out = ss_value_number((double)ss_vm_random(vm) / 4294967296.0); return SS_OK;
        case SS_HOST_ASSERT:
            if ((argc != 1u && (vm->chunk->version < 3u || argc != 2u)) ||
                arguments[0].type != VAL_BOOL ||
                (argc == 2u && arguments[1].type != VAL_STRING)) return SS_ERR_TYPE;
            return arguments[0].as.boolean ? SS_OK :
                   (vm->chunk->version >= 3u ? SS_ERR_ASSERTION : SS_ERR_TASK);
        case SS_HOST_ML_FIT_RIDGE:
            if (argc != 3u || arguments[2].type != VAL_NUMBER) return SS_ERR_TYPE;
            return fit_model(vm, &arguments[0], &arguments[1], SS_ML_RIDGE,
                             arguments[2].as.number, 1u, out);
        case SS_HOST_ML_FIT_LOGISTIC: {
            double regularization, iterations;
            if (argc != 4u || array_number(&arguments[2], &regularization) != SS_OK ||
                array_number(&arguments[3], &iterations) != SS_OK || iterations < 1.0 ||
                floor(iterations) != iterations) return SS_ERR_TYPE;
            return fit_model(vm, &arguments[0], &arguments[1], SS_ML_LOGISTIC,
                             regularization, (size_t)iterations, out);
        }
        case SS_HOST_ML_PREDICT:
            if (argc != 2u || arguments[0].type != VAL_MODEL || arguments[0].as.model == NULL)
                return SS_ERR_TYPE;
            return predict_model(vm, arguments[0].as.model, &arguments[1], out);
        case SS_HOST_ML_STANDARDIZE: {
            double *data = NULL; size_t rows, columns, row, column;
            SSError error = matrix_copy(&arguments[0], &data, &rows, &columns);
            if (argc != 1u || error != SS_OK) { free(data); return SS_ERR_TYPE; }
            for (column = 0; column < columns; ++column) {
                double mean = 0.0, scale = 0.0;
                for (row = 0; row < rows; ++row) mean += data[row * columns + column];
                mean /= (double)rows;
                for (row = 0; row < rows; ++row) { double delta = data[row * columns + column] - mean; scale += delta * delta; }
                scale = sqrt(scale / (double)(rows > 1u ? rows - 1u : 1u)); if (scale == 0.0) scale = 1.0;
                for (row = 0; row < rows; ++row) data[row * columns + column] = (data[row * columns + column] - mean) / scale;
            }
            error = make_matrix(vm, data, rows, columns, out); free(data); return error;
        }
        case SS_HOST_ML_POLYNOMIAL: {
            double *data = NULL, *expanded = NULL, degree; size_t rows, columns, row, column, power, output_columns;
            SSError error;
            if (argc != 2u || array_number(&arguments[1], &degree) != SS_OK || degree < 1.0 || degree > 5.0 || floor(degree) != degree) return SS_ERR_TYPE;
            error = matrix_copy(&arguments[0], &data, &rows, &columns); if (error != SS_OK) return error;
            output_columns = columns * (size_t)degree; expanded = malloc(rows * output_columns * sizeof(double)); if (expanded == NULL) { free(data); return SS_ERR_OOM; }
            for (row = 0; row < rows; ++row) for (column = 0; column < columns; ++column) for (power = 1; power <= (size_t)degree; ++power) expanded[row * output_columns + column * (size_t)degree + power - 1u] = pow(data[row * columns + column], (double)power);
            error = make_matrix(vm, expanded, rows, output_columns, out); free(data); free(expanded); return error;
        }
        case SS_HOST_ML_RBF: {
            double *data = NULL, *centers = NULL, *features = NULL, gamma; size_t rows, columns, center_rows, center_columns, row, center, column;
            SSError error;
            if (argc != 3u || array_number(&arguments[2], &gamma) != SS_OK || gamma <= 0.0) return SS_ERR_TYPE;
            error = matrix_copy(&arguments[0], &data, &rows, &columns); if (error != SS_OK) return error;
            error = matrix_copy(&arguments[1], &centers, &center_rows, &center_columns); if (error != SS_OK || center_columns != columns) { free(data); free(centers); return SS_ERR_TYPE; }
            features = malloc(rows * center_rows * sizeof(double)); if (features == NULL) { free(data); free(centers); return SS_ERR_OOM; }
            for (row = 0; row < rows; ++row) for (center = 0; center < center_rows; ++center) { double distance = 0.0; for (column = 0; column < columns; ++column) { double delta = data[row * columns + column] - centers[center * columns + column]; distance += delta * delta; } features[row * center_rows + center] = exp(-gamma * distance); }
            error = make_matrix(vm, features, rows, center_rows, out); free(data); free(centers); free(features); return error;
        }
        case SS_HOST_ML_REGRESSION_METRICS: {
            double *actual = NULL, *prediction = NULL, values[5] = {0}, mean = 0.0, sse = 0.0, total = 0.0, corr = 0.0; size_t count, other_count, index; SSError error = vector_copy(&arguments[0], &actual, &count);
            if (argc != 2u || error != SS_OK || vector_copy(&arguments[1], &prediction, &other_count) != SS_OK || other_count != count) { free(actual); free(prediction); return SS_ERR_TYPE; }
            for (index = 0; index < count; ++index) mean += actual[index];
            mean /= (double)count;
            for (index = 0; index < count; ++index) { double delta = actual[index] - prediction[index]; values[0] += fabs(delta); sse += delta * delta; total += (actual[index] - mean) * (actual[index] - mean); }
            values[0] /= (double)count; values[1] = sqrt(sse / (double)count); values[2] = 0.0; for (index = 0; index < count; ++index) values[2] += actual[index] - prediction[index]; values[2] /= (double)count; values[3] = total == 0.0 ? 0.0 : 1.0 - sse / total;
            { double actual_mean = mean, prediction_mean = 0.0, denominator_a = 0.0, denominator_b = 0.0; for (index = 0; index < count; ++index) prediction_mean += prediction[index]; prediction_mean /= (double)count; for (index = 0; index < count; ++index) { double a = actual[index] - actual_mean, b = prediction[index] - prediction_mean; corr += a * b; denominator_a += a * a; denominator_b += b * b; } values[4] = denominator_a == 0.0 || denominator_b == 0.0 ? 0.0 : corr / sqrt(denominator_a * denominator_b); }
            free(actual); free(prediction); return make_number_array(vm, values, 5u, out);
        }
        case SS_HOST_ML_CLASSIFICATION_METRICS: {
            double *actual = NULL, *probability = NULL, values[8] = {0}; size_t count, other_count, index, positives = 0u, negatives = 0u; SSError error = vector_copy(&arguments[0], &actual, &count);
            if (argc != 2u || error != SS_OK || vector_copy(&arguments[1], &probability, &other_count) != SS_OK || other_count != count) { free(actual); free(probability); return SS_ERR_TYPE; }
            for (index = 0; index < count; ++index) { int actual_label = actual[index] >= 0.5, predicted_label = probability[index] >= 0.5; double p = fmax(1e-9, fmin(1.0 - 1e-9, probability[index])); if (actual_label) positives++; else negatives++; if (actual_label && predicted_label) values[4]++; else if (!actual_label && !predicted_label) values[5]++; else if (predicted_label) values[6]++; else values[7]++; values[2] += -(actual_label * log(p) + (1 - actual_label) * log(1.0 - p)); values[3] += (actual_label - p) * (actual_label - p); }
            values[0] = (values[4] + values[5]) / (double)count; values[1] = ((positives ? values[4] / (double)positives : 0.0) + (negatives ? values[5] / (double)negatives : 0.0)) / 2.0; values[2] /= (double)count; values[3] /= (double)count; free(actual); free(probability); return make_number_array(vm, values, 8u, out);
        }
        case SS_HOST_ML_MODEL_WRITE: {
            FILE *file; SSMLModel *model; const char *path; size_t index, count;
            if (argc != 2u || arguments[0].type != VAL_MODEL || arguments[0].as.model == NULL || arguments[1].type != VAL_STRING) return SS_ERR_TYPE;
            model = arguments[0].as.model; path = arguments[1].as.string; file = fopen(path, "wb"); if (file == NULL) return SS_ERR_IO;
            count = model->feature_count + 1u; (void)fprintf(file, "SSML1 %d %zu %.17g\n", (int)model->kind, model->feature_count, model->regularization); for (index = 0; index < count; ++index) (void)fprintf(file, "%.17g ", model->coefficients[index]); (void)fputc('\n', file); for (index = 0; index < model->feature_count; ++index) (void)fprintf(file, "%.17g ", model->means[index]); (void)fputc('\n', file); for (index = 0; index < model->feature_count; ++index) (void)fprintf(file, "%.17g ", model->scales[index]); (void)fputc('\n', file); if (fclose(file) != 0) return SS_ERR_IO; *out = ss_value_bool(true); return SS_OK;
        }
        case SS_HOST_ML_MODEL_READ: {
            FILE *file; SSMLModel *model; int kind; size_t feature_count, index; double regularization; char magic[8];
            if (argc != 1u || arguments[0].type != VAL_STRING) return SS_ERR_TYPE;
            file = fopen(arguments[0].as.string, "rb");
            if (file == NULL) return SS_ERR_IO;
            if (fscanf(file, "%7s %d %zu %lf", magic, &kind, &feature_count, &regularization) != 4 || strcmp(magic, "SSML1") != 0 || feature_count == 0u || kind < 0 || kind > 1) { (void)fclose(file); return SS_ERR_FORMAT; }
            model = ss_vm_alloc(vm, sizeof(*model)); if (model == NULL) { (void)fclose(file); return SS_ERR_OOM; } model->kind = (SSMLModelKind)kind; model->feature_count = feature_count; model->regularization = regularization; model->coefficients = ss_vm_alloc(vm, (feature_count + 1u) * sizeof(double)); model->means = ss_vm_alloc(vm, feature_count * sizeof(double)); model->scales = ss_vm_alloc(vm, feature_count * sizeof(double)); if (model->coefficients == NULL || model->means == NULL || model->scales == NULL) { (void)fclose(file); return SS_ERR_OOM; }
            for (index = 0; index < feature_count + 1u; ++index) {
                if (fscanf(file, "%lf", &model->coefficients[index]) != 1) {
                    (void)fclose(file);
                    return SS_ERR_FORMAT;
                }
            }
            for (index = 0; index < feature_count; ++index) {
                if (fscanf(file, "%lf", &model->means[index]) != 1) {
                    (void)fclose(file);
                    return SS_ERR_FORMAT;
                }
            }
            for (index = 0; index < feature_count; ++index) {
                if (fscanf(file, "%lf", &model->scales[index]) != 1 || model->scales[index] == 0.0) {
                    (void)fclose(file);
                    return SS_ERR_FORMAT;
                }
            }
            (void)fclose(file);
            *out = ss_value_model(model);
            return SS_OK;
        }
        case SS_HOST_MAP_NEW: {
            SSMap *map; SSError error;
            if (argc % 2u != 0u) return SS_ERR_TYPE;
            error = ss_map_new(vm, argc / 2u, &map);
            for (index = 0u; error == SS_OK && index < argc; index += 2u) {
                if (arguments[index].type != VAL_STRING) return SS_ERR_TYPE;
                error = ss_map_set(vm, map, arguments[index].as.string,
                                   &arguments[index + 1u], true);
            }
            if (error == SS_OK) *out = ss_value_map(map);
            return error;
        }
        case SS_HOST_INDEX_GET:
            if (argc != 2u) return SS_ERR_TYPE;
            if (arguments[0].type == VAL_MAP && arguments[1].type == VAL_STRING)
                return ss_map_get(arguments[0].as.map, arguments[1].as.string, out);
            if (arguments[0].type == VAL_ARRAY && arguments[1].type == VAL_NUMBER &&
                arguments[1].as.number >= 0.0 && floor(arguments[1].as.number) == arguments[1].as.number &&
                (size_t)arguments[1].as.number < arguments[0].as.array->count) {
                *out = arguments[0].as.array->items[(size_t)arguments[1].as.number]; return SS_OK;
            }
            return arguments[0].type == VAL_ARRAY ? SS_ERR_LIMIT : SS_ERR_TYPE;
        case SS_HOST_INDEX_SET:
            if (argc != 3u) return SS_ERR_TYPE;
            if (arguments[0].type == VAL_MAP && arguments[1].type == VAL_STRING) {
                SSError error = ss_map_set(vm, arguments[0].as.map, arguments[1].as.string,
                                           &arguments[2], false);
                if (error == SS_OK) *out = arguments[2]; return error;
            }
            if (arguments[0].type == VAL_ARRAY && arguments[1].type == VAL_NUMBER &&
                arguments[1].as.number >= 0.0 && floor(arguments[1].as.number) == arguments[1].as.number &&
                (size_t)arguments[1].as.number < arguments[0].as.array->count) {
                arguments[0].as.array->items[(size_t)arguments[1].as.number] = arguments[2];
                *out = arguments[2]; return SS_OK;
            }
            return arguments[0].type == VAL_ARRAY ? SS_ERR_LIMIT : SS_ERR_TYPE;
        case SS_HOST_JSON_PARSE:
            return argc == 1u && arguments[0].type == VAL_STRING ? ss_json_parse(vm, arguments[0].as.string, out) : SS_ERR_TYPE;
        case SS_HOST_JSON_STRINGIFY:
            return argc == 1u ? ss_json_stringify(vm, &arguments[0], out) : SS_ERR_TYPE;
        case SS_HOST_CSV_READ:
            return argc == 1u && arguments[0].type == VAL_STRING ? ss_csv_read(vm, arguments[0].as.string, out) : SS_ERR_TYPE;
        case SS_HOST_CSV_WRITE:
            return argc == 2u && arguments[0].type == VAL_STRING ? ss_csv_write(vm, arguments[0].as.string, &arguments[1], out) : SS_ERR_TYPE;
        case SS_HOST_STRING_LENGTH:
            if (argc != 1u || arguments[0].type != VAL_STRING) return SS_ERR_TYPE;
            *out = ss_value_number((double)strlen(arguments[0].as.string)); return SS_OK;
        case SS_HOST_STRING_BYTE_AT: {
            size_t position, length;
            if (argc != 2u || arguments[0].type != VAL_STRING ||
                arguments[1].type != VAL_NUMBER || arguments[1].as.number < 0.0 ||
                floor(arguments[1].as.number) != arguments[1].as.number)
                return SS_ERR_TYPE;
            position = (size_t)arguments[1].as.number;
            length = strlen(arguments[0].as.string);
            if (position >= length) return SS_ERR_LIMIT;
            *out = ss_value_number((unsigned char)arguments[0].as.string[position]);
            return SS_OK;
        }
        case SS_HOST_STRING_SLICE: {
            size_t start, end, length;
            char *copy;
            if (argc != 3u || arguments[0].type != VAL_STRING ||
                arguments[1].type != VAL_NUMBER || arguments[2].type != VAL_NUMBER ||
                arguments[1].as.number < 0.0 || arguments[2].as.number < 0.0 ||
                floor(arguments[1].as.number) != arguments[1].as.number ||
                floor(arguments[2].as.number) != arguments[2].as.number)
                return SS_ERR_TYPE;
            start = (size_t)arguments[1].as.number;
            end = (size_t)arguments[2].as.number;
            length = strlen(arguments[0].as.string);
            if (start > end || end > length) return SS_ERR_LIMIT;
            copy = ss_vm_alloc(vm, end - start + 1u);
            if (copy == NULL) return SS_ERR_OOM;
            memcpy(copy, arguments[0].as.string + start, end - start);
            copy[end - start] = '\0'; *out = ss_value_string(copy); return SS_OK;
        }
        case SS_HOST_STRING_CONCAT: {
            size_t total = 0u, offset = 0u;
            char *copy;
            for (index = 0; index < argc; ++index) {
                size_t length;
                if (arguments[index].type != VAL_STRING) return SS_ERR_TYPE;
                length = strlen(arguments[index].as.string);
                if (length > SIZE_MAX - total - 1u) return SS_ERR_LIMIT;
                total += length;
            }
            copy = ss_vm_alloc(vm, total + 1u);
            if (copy == NULL) return SS_ERR_OOM;
            for (index = 0; index < argc; ++index) {
                size_t length = strlen(arguments[index].as.string);
                memcpy(copy + offset, arguments[index].as.string, length); offset += length;
            }
            copy[offset] = '\0'; *out = ss_value_string(copy); return SS_OK;
        }
        case SS_HOST_NUMBER_TO_STRING: {
            char buffer[64]; char *copy; int written;
            if (argc != 1u || arguments[0].type != VAL_NUMBER) return SS_ERR_TYPE;
            written = snprintf(buffer, sizeof(buffer), "%.17g", arguments[0].as.number);
            if (written < 0 || (size_t)written >= sizeof(buffer)) return SS_ERR_FORMAT;
            copy = ss_vm_alloc(vm, (size_t)written + 1u);
            if (copy == NULL) return SS_ERR_OOM;
            memcpy(copy, buffer, (size_t)written + 1u);
            *out = ss_value_string(copy); return SS_OK;
        }
        case SS_HOST_ARRAY_NEW: {
            SSArray *array; size_t count;
            if (argc != 1u || arguments[0].type != VAL_NUMBER ||
                arguments[0].as.number < 0.0 ||
                floor(arguments[0].as.number) != arguments[0].as.number ||
                arguments[0].as.number > (double)SS_MAX_REGISTERS * 4096.0)
                return SS_ERR_TYPE;
            count = (size_t)arguments[0].as.number;
            array = ss_vm_alloc(vm, sizeof(*array));
            if (array == NULL) return SS_ERR_OOM;
            array->count = count;
            array->capacity = count;
            array->items = ss_vm_alloc(vm, count * sizeof(*array->items));
            if (array->items == NULL && count > 0u) return SS_ERR_OOM;
            *out = ss_value_array(array); return SS_OK;
        }
        case SS_HOST_ARRAY_PUSH: {
            SSArray *array; Value *items;
            if (argc != 2u || arguments[0].type != VAL_ARRAY ||
                arguments[0].as.array == NULL) return SS_ERR_TYPE;
            array = arguments[0].as.array;
            if (array->count == SIZE_MAX / sizeof(*items)) return SS_ERR_LIMIT;
            if (array->count == array->capacity) {
                size_t capacity = array->capacity == 0u ? 8u : array->capacity * 2u;
                if (capacity <= array->capacity) return SS_ERR_LIMIT;
                items = ss_vm_alloc(vm, capacity * sizeof(*items));
                if (items == NULL) return SS_ERR_OOM;
                if (array->count > 0u)
                    memcpy(items, array->items, array->count * sizeof(*items));
                array->items = items; array->capacity = capacity;
            }
            array->items[array->count++] = arguments[1];
            *out = arguments[0]; return SS_OK;
        }
        case SS_HOST_STRING_HEX: {
            static const char digits[] = "0123456789abcdef";
            const unsigned char *source; size_t length; char *hex;
            if (argc != 1u || arguments[0].type != VAL_STRING) return SS_ERR_TYPE;
            source = (const unsigned char *)arguments[0].as.string;
            length = strlen(arguments[0].as.string);
            if (length > (SIZE_MAX - 1u) / 2u) return SS_ERR_LIMIT;
            hex = ss_vm_alloc(vm, length * 2u + 1u);
            if (hex == NULL) return SS_ERR_OOM;
            for (index = 0; index < length; ++index) {
                hex[index * 2u] = digits[source[index] >> 4u];
                hex[index * 2u + 1u] = digits[source[index] & 15u];
            }
            hex[length * 2u] = '\0'; *out = ss_value_string(hex); return SS_OK;
        }
        case SS_HOST_STRING_JOIN: {
            const SSArray *array; const char *separator; size_t separator_length;
            size_t total = 0u, offset = 0u; char *joined;
            if (argc != 2u || arguments[0].type != VAL_ARRAY ||
                arguments[1].type != VAL_STRING) return SS_ERR_TYPE;
            array = arguments[0].as.array; separator = arguments[1].as.string;
            separator_length = strlen(separator);
            for (index = 0; index < array->count; ++index) {
                size_t length;
                if (array->items[index].type != VAL_STRING) return SS_ERR_TYPE;
                length = strlen(array->items[index].as.string);
                if (length > SIZE_MAX - total - 1u) return SS_ERR_LIMIT;
                total += length;
                if (index + 1u < array->count) {
                    if (separator_length > SIZE_MAX - total - 1u) return SS_ERR_LIMIT;
                    total += separator_length;
                }
            }
            joined = ss_vm_alloc(vm, total + 1u);
            if (joined == NULL) return SS_ERR_OOM;
            for (index = 0; index < array->count; ++index) {
                size_t length = strlen(array->items[index].as.string);
                memcpy(joined + offset, array->items[index].as.string, length); offset += length;
                if (index + 1u < array->count) {
                    memcpy(joined + offset, separator, separator_length);
                    offset += separator_length;
                }
            }
            joined[offset] = '\0'; *out = ss_value_string(joined); return SS_OK;
        }
        case SS_HOST_ARRAY_LENGTH:
            if (argc != 1u || arguments[0].type != VAL_ARRAY) return SS_ERR_TYPE;
            *out = ss_value_number((double)arguments[0].as.array->count); return SS_OK;
        case SS_HOST_STRING_UNESCAPE: {
            const char *source; size_t length, read = 0u, written = 0u; char *decoded;
            if (argc != 1u || arguments[0].type != VAL_STRING) return SS_ERR_TYPE;
            source = arguments[0].as.string; length = strlen(source);
            decoded = ss_vm_alloc(vm, length + 1u);
            if (decoded == NULL) return SS_ERR_OOM;
            while (read < length) {
                char value = source[read++];
                if (value != '\\') { decoded[written++] = value; continue; }
                if (read >= length) return SS_ERR_FORMAT;
                value = source[read++];
                if (value == 'n') decoded[written++] = '\n';
                else if (value == 'r') decoded[written++] = '\r';
                else if (value == 't') decoded[written++] = '\t';
                else if (value == '\\' || value == '"') decoded[written++] = value;
                else return SS_ERR_FORMAT;
            }
            decoded[written] = '\0'; *out = ss_value_string(decoded); return SS_OK;
        }
        case SS_HOST_PATH_RESOLVE: {
            const char *base, *relative, *separator; size_t directory_length, needed; char *candidate, resolved[PATH_MAX], *copy;
            if (argc != 2u || arguments[0].type != VAL_STRING || arguments[1].type != VAL_STRING) return SS_ERR_TYPE;
            base = arguments[0].as.string; relative = arguments[1].as.string;
            if (relative[0] == '\0') {
                if (realpath(base, resolved) == NULL) return SS_ERR_IO;
                copy = ss_vm_alloc(vm, strlen(resolved) + 1u); if (copy == NULL) return SS_ERR_OOM;
                (void)strcpy(copy, resolved); *out = ss_value_string(copy); return SS_OK;
            }
            separator = strrchr(base, '/');
            directory_length = separator == NULL ? 1u : (size_t)(separator - base);
            if (strlen(relative) > SIZE_MAX - directory_length - 2u) return SS_ERR_LIMIT;
            needed = directory_length + strlen(relative) + 2u; candidate = malloc(needed);
            if (candidate == NULL) return SS_ERR_OOM;
            if (separator == NULL) (void)snprintf(candidate, needed, "./%s", relative);
            else (void)snprintf(candidate, needed, "%.*s/%s", (int)directory_length, base, relative);
            if (realpath(candidate, resolved) == NULL) { free(candidate); return SS_ERR_IO; }
            free(candidate); copy = ss_vm_alloc(vm, strlen(resolved) + 1u); if (copy == NULL) return SS_ERR_OOM;
            (void)strcpy(copy, resolved); *out = ss_value_string(copy); return SS_OK;
        }
        default: return SS_ERR_FORMAT;
    }
}

static SSError close_task_group(VM *vm, uint64_t group_id) {
    SSTask *task;
    Value ignored;
    SSError first_error = SS_OK;
    for (task = vm->tasks; task != NULL; task = task->next)
        if (task->group_id == group_id && !task->completed) cancel_task(task);
    for (task = vm->tasks; task != NULL; task = task->next) {
        if (task->group_id == group_id && !task->joined) {
            SSError error = wait_task(vm, task, -1.0, &ignored);
            if (error == SS_ERR_CANCELLED) memset(&vm->error, 0, sizeof(vm->error));
            else if (error != SS_OK && first_error == SS_OK) first_error = error;
        }
    }
    return first_error;
}

static SSError history_append(VM *vm, SSHistory *history, SSStateValue state) {
    SSStateValue *versions;
    size_t keep_from = 0;
    if (history->policy == SS_HISTORY_NONE) return SS_OK;
    if (history->count == history->capacity) {
        size_t capacity = history->capacity == 0 ? 8u : history->capacity * 2u;
        size_t additional = (capacity - history->capacity) * sizeof(*versions);
        if (additional > vm->memory_limit - vm->allocated_bytes) return SS_ERR_OOM;
        versions = realloc(history->versions, capacity * sizeof(*versions));
        if (versions == NULL) return SS_ERR_OOM;
        history->versions = versions;
        history->capacity = capacity;
        vm->allocated_bytes += additional;
        vm->allocation_count += 1u;
    }
    history->versions[history->count++] = state;
    if (history->policy == SS_HISTORY_LATEST && history->count > (size_t)history->amount)
        keep_from = history->count - (size_t)history->amount;
    else if (history->policy == SS_HISTORY_DURATION && state.indexes.has_timestamp) {
        double cutoff = state.indexes.timestamp - history->amount;
        while (keep_from < history->count && history->versions[keep_from].indexes.has_timestamp &&
               history->versions[keep_from].indexes.timestamp < cutoff) ++keep_from;
    }
    if (keep_from > 0) {
        memmove(history->versions, history->versions + keep_from,
                (history->count - keep_from) * sizeof(*history->versions));
        history->count -= keep_from;
    }
    return SS_OK;
}

static SSError store_state(VM *vm, uint32_t reg, SSStateValue state) {
    SSFrame *frame = current_frame(vm);
    SSError error;
    if (!ss_state_valid(&state.state) && !ss_state_legacy_valid(&state.state))
        return SS_ERR_INVALID_STATE;
    frame->registers[reg].type = VAL_STATE;
    frame->registers[reg].as.state = state;
    vm->state_transition_count += 1u;
    error = history_append(vm, &frame->histories[reg], state);
    return error;
}

static SSError require_type(const Value *value, ValueType type) {
    return value->type == type ? SS_OK : SS_ERR_TYPE;
}

static SSError require_number_args(const Value *args, size_t argc) {
    size_t index;
    for (index = 0; index < argc; ++index)
        if (args[index].type != VAL_NUMBER) return SS_ERR_TYPE;
    return SS_OK;
}

SSError ss_vm_state_dist_dirac(VM *vm, const SSStateValue *state, SSStateDist **out) {
    SSStateDist *distribution;
    if (vm == NULL || state == NULL || out == NULL || !ss_state_valid(&state->state))
        return SS_ERR_INVALID_STATE;
    distribution = ss_vm_alloc(vm, sizeof(*distribution));
    if (distribution == NULL) return SS_ERR_OOM;
    distribution->kind = SS_DIST_DIRAC;
    distribution->as.dirac = *state;
    *out = distribution;
    return SS_OK;
}

static SSError distribution_from_value(VM *vm, const Value *value, SSStateDist **out) {
    if (value == NULL || out == NULL) return SS_ERR_TYPE;
    if (value->type == VAL_STATE)
        return ss_vm_state_dist_dirac(vm, &value->as.state, out);
    if (value->type == VAL_STATE_DIST && value->as.state_dist != NULL) {
        *out = value->as.state_dist;
        return SS_OK;
    }
    return SS_ERR_TYPE;
}

SSError ss_vm_state_dist_append(VM *vm, const Value *left, const Value *right,
                                SSStateDist **out) {
    SSStateDist *distribution;
    SSError error;
    if (vm == NULL || out == NULL) return SS_ERR_INVALID_DISTRIBUTION;
    distribution = ss_vm_alloc(vm, sizeof(*distribution));
    if (distribution == NULL) return SS_ERR_OOM;
    distribution->kind = SS_DIST_APPEND;
    error = distribution_from_value(vm, left, &distribution->as.append.left);
    if (error == SS_OK)
        error = distribution_from_value(vm, right, &distribution->as.append.right);
    if (error != SS_OK) return error;
    distribution->as.append.has_cached_parameters = false;
    if (left->type == VAL_STATE && right->type == VAL_STATE) {
        error = ss_state_append_parameters(&left->as.state.state, &right->as.state.state,
                                           &distribution->as.append.p,
                                           &distribution->as.append.m_re,
                                           &distribution->as.append.m_im,
                                           &distribution->as.append.sigma);
        if (error != SS_OK) return error;
        distribution->as.append.has_cached_parameters = true;
    }
    *out = distribution;
    return SS_OK;
}

SSError ss_vm_state_dist_transform(VM *vm, uint32_t transform_id,
                                   SSStateDist *child, SSStateDist **out) {
    const SSTransformSpecV3 *specification = ss_transform_v3_spec(transform_id);
    SSStateDist *distribution;
    if (vm == NULL || child == NULL || out == NULL) return SS_ERR_INVALID_DISTRIBUTION;
    if (specification == NULL || !specification->distribution_liftable ||
        specification->exact_expected_probability == NULL)
        return SS_ERR_UNSUPPORTED_OPERATION;
    distribution = ss_vm_alloc(vm, sizeof(*distribution));
    if (distribution == NULL) return SS_ERR_OOM;
    distribution->kind = SS_DIST_TRANSFORM;
    distribution->as.transform.child = child;
    distribution->as.transform.transform_id = transform_id;
    *out = distribution;
    return SS_OK;
}

static SSError expected_probability_recursive(const SSStateDist *distribution,
                                              double *out, size_t depth) {
    double left, right;
    SSError error;
    if (distribution == NULL || out == NULL || depth > 1024u)
        return SS_ERR_INVALID_DISTRIBUTION;
    switch (distribution->kind) {
        case SS_DIST_DIRAC:
            if (!ss_state_valid(&distribution->as.dirac.state))
                return SS_ERR_INVALID_DISTRIBUTION;
            *out = distribution->as.dirac.state.p;
            return SS_OK;
        case SS_DIST_APPEND:
            error = expected_probability_recursive(distribution->as.append.left,
                                                   &left, depth + 1u);
            if (error == SS_OK)
                error = expected_probability_recursive(distribution->as.append.right,
                                                       &right, depth + 1u);
            if (error != SS_OK) return error;
            *out = 1.0 - (1.0 - left) * (1.0 - right);
            return isfinite(*out) && *out >= 0.0 && *out <= 1.0
                       ? SS_OK : SS_ERR_INVALID_DISTRIBUTION;
        case SS_DIST_TRANSFORM:
            error = expected_probability_recursive(distribution->as.transform.child,
                                                   &left, depth + 1u);
            if (error != SS_OK) return error;
            return ss_transform_v3_expected_probability(
                distribution->as.transform.transform_id, left, out);
        default:
            return SS_ERR_INVALID_DISTRIBUTION;
    }
}

SSError ss_vm_state_dist_expected_probability(const SSStateDist *distribution,
                                              double *out) {
    return expected_probability_recursive(distribution, out, 0u);
}

static SSError consume_sampling_budget(VM *vm) {
    if (atomic_load(&vm->cancelled)) return SS_ERR_CANCELLED;
    if (vm->instruction_count >= vm->instruction_limit) return SS_ERR_BUDGET_EXHAUSTED;
    ++vm->instruction_count;
    return SS_OK;
}

static double uniform_signed(VM *vm) {
    return 2.0 * ((double)ss_vm_random(vm) / 4294967296.0) - 1.0;
}

static SSError sample_append_kernel(VM *vm, const SSStateValue *left,
                                    const SSStateValue *right, SSStateValue *out) {
    double p, m_re, m_im, sigma;
    SSError error = ss_state_append_parameters(&left->state, &right->state,
                                               &p, &m_re, &m_im, &sigma);
    if (error != SS_OK) return SS_ERR_INVALID_DISTRIBUTION;
    memset(&out->indexes, 0, sizeof(out->indexes));
    if (p == 0.0 || p == 1.0)
        return ss_state_make_complex(p, 0.0, 0.0, &out->state);
    if (sigma == 0.0)
        return ss_state_make_complex(p, m_re, m_im, &out->state);
    for (;;) {
        double x, y, radius_squared, factor, d_re, d_im;
        error = consume_sampling_budget(vm);
        if (error != SS_OK) return error;
        x = uniform_signed(vm);
        y = uniform_signed(vm);
        radius_squared = x * x + y * y;
        if (radius_squared <= 0.0 || radius_squared >= 1.0) continue;
        factor = sqrt(-2.0 * log(radius_squared) / radius_squared);
        d_re = m_re + sigma * x * factor;
        d_im = m_im + sigma * y * factor;
        if (d_re * d_re + d_im * d_im > 1.0) continue;
        return ss_state_make_complex(p, d_re, d_im, &out->state);
    }
}

static SSError sample_distribution_recursive(VM *vm, const SSStateDist *distribution,
                                             SSStateValue *out, size_t depth) {
    SSStateValue left, right;
    SSError error;
    if (distribution == NULL || out == NULL || depth > 1024u)
        return SS_ERR_INVALID_DISTRIBUTION;
    switch (distribution->kind) {
        case SS_DIST_DIRAC:
            if (!ss_state_valid(&distribution->as.dirac.state))
                return SS_ERR_INVALID_DISTRIBUTION;
            *out = distribution->as.dirac;
            return SS_OK;
        case SS_DIST_APPEND:
            error = sample_distribution_recursive(vm, distribution->as.append.left,
                                                  &left, depth + 1u);
            if (error == SS_OK)
                error = sample_distribution_recursive(vm, distribution->as.append.right,
                                                      &right, depth + 1u);
            return error == SS_OK ? sample_append_kernel(vm, &left, &right, out) : error;
        case SS_DIST_TRANSFORM:
            error = sample_distribution_recursive(vm, distribution->as.transform.child,
                                                  out, depth + 1u);
            if (error != SS_OK) return error;
            return ss_transform_v3_apply(distribution->as.transform.transform_id,
                                         &out->state, &out->state);
        default:
            return SS_ERR_INVALID_DISTRIBUTION;
    }
}

SSError ss_vm_state_dist_sample(VM *vm, const SSStateDist *distribution,
                                SSStateValue *out) {
    if (vm == NULL) return SS_ERR_INVALID_DISTRIBUTION;
    return sample_distribution_recursive(vm, distribution, out, 0u);
}

static SSError transform_decay(VM *vm, SSStateValue *value, const Value *args, size_t argc) {
    double rate;
    (void)vm;
    if (argc != 1 || require_number_args(args, argc) != SS_OK) return SS_ERR_TYPE;
    rate = args[0].as.number;
    if (rate < 0.0 || rate > 1.0) return SS_ERR_TRANSFORM;
    value->state.p += rate * (0.5 - value->state.p);
    value->state.d *= 1.0 - rate;
    return ss_state_legacy_valid(&value->state) ? SS_OK : SS_ERR_INVALID_STATE;
}

static SSError transform_reinforce(VM *vm, SSStateValue *value, const Value *args, size_t argc) {
    double p, strength;
    (void)vm;
    if (argc != 1 || require_number_args(args, argc) != SS_OK) return SS_ERR_TYPE;
    strength = args[0].as.number;
    if (strength < 0.0 || strength > 1.0) return SS_ERR_TRANSFORM;
    p = 0.5 + (1.0 + strength) * (value->state.p - 0.5);
    if (p < 0.0) p = 0.0;
    if (p > 1.0) p = 1.0;
    value->state.p = p;
    return SS_OK;
}

static SSError transform_invert(VM *vm, SSStateValue *value, const Value *args, size_t argc) {
    (void)vm; (void)args;
    if (argc != 0) return SS_ERR_TYPE;
    value->state.p = 1.0 - value->state.p;
    value->state.d = -value->state.d;
    return SS_OK;
}

static SSError transform_neutralize(VM *vm, SSStateValue *value, const Value *args, size_t argc) {
    (void)vm; (void)args;
    if (argc != 0) return SS_ERR_TYPE;
    return ss_state_make(0.5, 0.0, &value->state);
}

static SSError transform_shift(VM *vm, SSStateValue *value, const Value *args, size_t argc) {
    double p;
    (void)vm;
    if (argc != 1 || require_number_args(args, argc) != SS_OK) return SS_ERR_TYPE;
    p = value->state.p + args[0].as.number;
    if (p < 0.0) p = 0.0;
    if (p > 1.0) p = 1.0;
    value->state.p = p;
    return SS_OK;
}

static SSError transform_clamp(VM *vm, SSStateValue *value, const Value *args, size_t argc) {
    double low, high;
    (void)vm;
    if (argc != 2 || require_number_args(args, argc) != SS_OK) return SS_ERR_TYPE;
    low = args[0].as.number; high = args[1].as.number;
    if (low < 0.0 || high > 1.0 || low > high) return SS_ERR_TRANSFORM;
    if (value->state.p < low) value->state.p = low;
    if (value->state.p > high) value->state.p = high;
    return SS_OK;
}

static SSError transform_reset(VM *vm, SSStateValue *value, const Value *args, size_t argc) {
    (void)vm; (void)args;
    if (argc != 0) return SS_ERR_TYPE;
    value->state.d = 0.0;
    return SS_OK;
}

static SSTransformFn transform_registry[] = {
    transform_decay, transform_reinforce, transform_invert, transform_neutralize,
    transform_shift, transform_clamp, transform_reset
};

static SSError measure_distribution(VM *vm, SSStateValue *state, Value *out, const Value *args, size_t argc) {
    (void)vm; (void)args;
    if (argc != 0) return SS_ERR_TYPE;
    *out = ss_value_distribution(1.0 - state->state.p, state->state.p); return SS_OK;
}
static SSError measure_probability(VM *vm, SSStateValue *state, Value *out, const Value *args, size_t argc) {
    (void)vm; (void)args;
    if (argc != 0) return SS_ERR_TYPE;
    *out = ss_value_number(state->state.p); return SS_OK;
}
static int draw_sample(VM *vm, double p) {
    double draw = (double)ss_vm_random(vm) / 4294967296.0;
    return draw < p ? 1 : 0;
}

static SSError measure_basis_state(const SSStateValue *state, uint32_t basis,
                                   double *out) {
    if (state == NULL || out == NULL) return SS_ERR_INVALID_STATE;
    return ss_state_basis_probability(basis, &state->state, out);
}

static SSError estimate_basis_probability(VM *vm, const SSStateDist *distribution,
                                           uint32_t basis, uint32_t samples,
                                           double *out) {
    double total = 0.0;
    uint32_t index;
    if (vm == NULL || distribution == NULL || out == NULL || samples == 0u)
        return SS_ERR_FORMAT;
    for (index = 0u; index < samples; ++index) {
        SSStateValue state;
        double probability;
        SSError error = consume_sampling_budget(vm);
        if (error != SS_OK) return error;
        error = ss_vm_state_dist_sample(vm, distribution, &state);
        if (error != SS_OK) return error;
        if (atomic_load(&vm->cancelled)) return SS_ERR_CANCELLED;
        error = measure_basis_state(&state, basis, &probability);
        if (error != SS_OK) return error;
        total += probability;
    }
    *out = total / (double)samples;
    return isfinite(*out) && *out >= 0.0 && *out <= 1.0
               ? SS_OK : SS_ERR_INVALID_DISTRIBUTION;
}
static SSError measure_sample(VM *vm, SSStateValue *state, Value *out, const Value *args, size_t argc) {
    (void)args;
    if (argc != 0) return SS_ERR_TYPE;
    *out = ss_value_sample(draw_sample(vm, state->state.p)); return SS_OK;
}
static SSError measure_collapse(VM *vm, SSStateValue *state, Value *out, const Value *args, size_t argc) {
    int sample;
    (void)args;
    if (argc != 0) return SS_ERR_TYPE;
    sample = draw_sample(vm, state->state.p);
    *out = ss_value_sample(sample);
    return ss_state_make((double)sample, 0.0, &state->state);
}
static SSMeasureFn measure_registry[] = {
    measure_distribution, measure_probability, measure_sample, measure_collapse
};

static SSError values_equal(const Value *left, const Value *right, bool *out) {
    if (left->type == VAL_STATE_DIST || right->type == VAL_STATE_DIST ||
        left->type == VAL_MAP || right->type == VAL_MAP ||
        left->type == VAL_JOINT_STATE || right->type == VAL_JOINT_STATE ||
        left->type == VAL_POSSIBILITY || right->type == VAL_POSSIBILITY ||
        left->type == VAL_PATH_SET || right->type == VAL_PATH_SET)
        return SS_ERR_UNSUPPORTED_OPERATION;
    if (left->type != right->type) {
        *out = false;
        return SS_OK;
    }
    switch (left->type) {
        case VAL_NULL: *out = true; break;
        case VAL_NUMBER: *out = left->as.number == right->as.number; break;
        case VAL_BOOL: *out = left->as.boolean == right->as.boolean; break;
        case VAL_STRING: *out = strcmp(left->as.string, right->as.string) == 0; break;
        case VAL_SAMPLE: *out = left->as.sample == right->as.sample; break;
        case VAL_STATE:
            *out = left->as.state.state.p == right->as.state.state.p &&
                   left->as.state.state.d_re == right->as.state.state.d_re &&
                   left->as.state.state.d_im == right->as.state.state.d_im;
            break;
        default: *out = left == right; break;
    }
    return SS_OK;
}

typedef enum { SS_PURE_BINARY, SS_PURE_COMPARE } SSPureKind;

static SSError pure_scalar_binary(const Value *left, const Value *right,
                                  SSPureKind kind, uint32_t operation,
                                  Value *out) {
    bool result = false;
    SSError error;
    if (kind == SS_PURE_BINARY) {
        if (left->type != VAL_NUMBER || right->type != VAL_NUMBER) return SS_ERR_TYPE;
        if (operation == SS_BINARY_ADD) *out = ss_value_number(left->as.number + right->as.number);
        else if (operation == SS_BINARY_SUBTRACT) *out = ss_value_number(left->as.number - right->as.number);
        else if (operation == SS_BINARY_MULTIPLY) *out = ss_value_number(left->as.number * right->as.number);
        else if (operation == SS_BINARY_DIVIDE && right->as.number != 0.0)
            *out = ss_value_number(left->as.number / right->as.number);
        else return SS_ERR_TYPE;
        return SS_OK;
    }
    if (operation == SS_COMPARE_EQUAL || operation == SS_COMPARE_NOT_EQUAL) {
        error = values_equal(left, right, &result);
        if (error != SS_OK) return error;
        if (operation == SS_COMPARE_NOT_EQUAL) result = !result;
    } else if (left->type == VAL_NUMBER && right->type == VAL_NUMBER) {
        if (operation == SS_COMPARE_LESS) result = left->as.number < right->as.number;
        else if (operation == SS_COMPARE_LESS_EQUAL) result = left->as.number <= right->as.number;
        else if (operation == SS_COMPARE_GREATER) result = left->as.number > right->as.number;
        else if (operation == SS_COMPARE_GREATER_EQUAL) result = left->as.number >= right->as.number;
        else return SS_ERR_TYPE;
    } else return SS_ERR_TYPE;
    *out = ss_value_bool(result); return SS_OK;
}

static SSError lift_binary(VM *vm, const Value *left, const Value *right,
                           SSPureKind kind, uint32_t operation, Value *out) {
    const SSPathSet *left_paths = left->type == VAL_PATH_SET ? left->as.paths : NULL;
    const SSPathSet *right_paths = right->type == VAL_PATH_SET ? right->as.paths : NULL;
    const SSPossibility *left_possibility = left->type == VAL_POSSIBILITY ? left->as.possibility : NULL;
    const SSPossibility *right_possibility = right->type == VAL_POSSIBILITY ? right->as.possibility : NULL;
    size_t count, index, left_index, right_index;
    SSError error;
    if (left_paths != NULL || right_paths != NULL) {
        SSPathSet *paths;
        if (left_paths != NULL && right_paths != NULL &&
            (left_paths->dependency_id != right_paths->dependency_id ||
             left_paths->count != right_paths->count)) return SS_ERR_UNSUPPORTED_OPERATION;
        count = left_paths != NULL ? left_paths->count : right_paths->count;
        paths = ss_vm_alloc(vm, sizeof(*paths));
        if (paths == NULL) return SS_ERR_OOM;
        paths->count = count;
        paths->dependency_id = left_paths != NULL
            ? left_paths->dependency_id : right_paths->dependency_id;
        paths->alternatives = ss_vm_alloc(vm, count * sizeof(*paths->alternatives));
        if (paths->alternatives == NULL) return SS_ERR_OOM;
        for (index = 0; index < count; ++index) {
            const Value *left_value = left_paths == NULL ? left : left_paths->alternatives[index].result;
            const Value *right_value = right_paths == NULL ? right : right_paths->alternatives[index].result;
            paths->alternatives[index].guard = (left_paths != NULL
                ? left_paths : right_paths)->alternatives[index].guard;
            paths->alternatives[index].weight = (left_paths != NULL
                ? left_paths : right_paths)->alternatives[index].weight;
            paths->alternatives[index].result = ss_vm_alloc(vm, sizeof(Value));
            if (paths->alternatives[index].result == NULL) return SS_ERR_OOM;
            error = lift_binary(vm, left_value, right_value, kind, operation,
                                paths->alternatives[index].result);
            if (error != SS_OK) return error;
        }
        *out = ss_value_paths(paths); return SS_OK;
    }
    if (left_possibility != NULL || right_possibility != NULL) {
        Value *results;
        SSPossibility *possibility;
        size_t left_count = left_possibility == NULL ? 1u : left_possibility->count;
        size_t right_count = right_possibility == NULL ? 1u : right_possibility->count;
        bool zipped = left_possibility != NULL && right_possibility != NULL &&
            left_possibility->dependency_id == right_possibility->dependency_id &&
            left_count == right_count;
        count = zipped ? left_count : left_count * right_count;
        results = calloc(count, sizeof(*results));
        if (results == NULL) return SS_ERR_OOM;
        index = 0u;
        for (left_index = 0; left_index < left_count; ++left_index) {
            size_t right_start = zipped ? left_index : 0u;
            size_t right_end = zipped ? left_index + 1u : right_count;
            for (right_index = right_start; right_index < right_end; ++right_index) {
                const Value *left_value = left_possibility == NULL ? left :
                    &left_possibility->values[left_index];
                const Value *right_value = right_possibility == NULL ? right :
                    &right_possibility->values[right_index];
                error = pure_scalar_binary(left_value, right_value, kind, operation,
                                           &results[index++]);
                if (error != SS_OK) { free(results); return error; }
            }
        }
        error = ss_vm_possibility_build(vm, results, count, &possibility);
        free(results);
        if (error != SS_OK) return error;
        if (zipped || left_possibility == NULL || right_possibility == NULL)
            possibility->dependency_id = left_possibility != NULL
                ? left_possibility->dependency_id : right_possibility->dependency_id;
        *out = ss_value_possibility(possibility); return SS_OK;
    }
    return pure_scalar_binary(left, right, kind, operation, out);
}

static SSError lift_unary(VM *vm, const Value *source, uint32_t operation,
                          Value *out) {
    size_t index;
    SSError error;
    if (source->type == VAL_PATH_SET) {
        SSPathSet *paths = ss_vm_alloc(vm, sizeof(*paths));
        if (paths == NULL) return SS_ERR_OOM;
        paths->count = source->as.paths->count;
        paths->dependency_id = source->as.paths->dependency_id;
        paths->alternatives = ss_vm_alloc(vm, paths->count * sizeof(*paths->alternatives));
        if (paths->alternatives == NULL) return SS_ERR_OOM;
        for (index = 0; index < paths->count; ++index) {
            paths->alternatives[index] = source->as.paths->alternatives[index];
            paths->alternatives[index].result = ss_vm_alloc(vm, sizeof(Value));
            if (paths->alternatives[index].result == NULL) return SS_ERR_OOM;
            error = lift_unary(vm, source->as.paths->alternatives[index].result,
                               operation, paths->alternatives[index].result);
            if (error != SS_OK) return error;
        }
        *out = ss_value_paths(paths); return SS_OK;
    }
    if (source->type == VAL_POSSIBILITY) {
        Value *results = calloc(source->as.possibility->count, sizeof(*results));
        SSPossibility *possibility;
        if (results == NULL) return SS_ERR_OOM;
        for (index = 0; index < source->as.possibility->count; ++index) {
            error = lift_unary(vm, &source->as.possibility->values[index],
                               operation, &results[index]);
            if (error != SS_OK) { free(results); return error; }
        }
        error = ss_vm_possibility_build(vm, results, source->as.possibility->count,
                                        &possibility);
        free(results);
        if (error != SS_OK) return error;
        possibility->dependency_id = source->as.possibility->dependency_id;
        *out = ss_value_possibility(possibility); return SS_OK;
    }
    if (source->type == VAL_NUMBER && operation == 0u) {
        *out = ss_value_number(-source->as.number); return SS_OK;
    }
    if (source->type == VAL_BOOL && operation == 1u) {
        *out = ss_value_bool(!source->as.boolean); return SS_OK;
    }
    return SS_ERR_TYPE;
}

static void trace_instruction(VM *vm, size_t ip, const SSInstruction *ins) {
    SSFrame *frame = current_frame(vm);
    if (vm->task_id != 0u) (void)printf("[task %llu] ", (unsigned long long)vm->task_id);
    ss_disassemble_instruction(vm->chunk, ip, stdout);
    if (ins->opcode == OP_APPLY && frame->registers[ins->a].type == VAL_STATE &&
        frame->registers[ins->b].type == VAL_STATE) {
        const SSState *source = &frame->registers[ins->a].as.state.state;
        const SSState *target = &frame->registers[ins->b].as.state.state;
        (void)printf("  R%u source: p=%.12g d=%.12g\n", ins->a, source->p, source->d);
        (void)printf("  R%u before: p=%.12g d=%.12g\n", ins->b, target->p, target->d);
    }
}

SSError ss_vm_run(VM *vm) {
    SSError verify;
    if (vm == NULL || vm->chunk == NULL) return SS_ERR_FORMAT;
    verify = ss_chunk_verify(vm->chunk, &vm->error);
    if (verify != SS_OK) return verify;
    while (vm->running) {
        SSFrame *frame;
        const SSInstruction *ins;
        size_t instruction_ip;
        SSError error = SS_OK;
        if (atomic_load(&vm->cancelled))
            return vm_fail(vm, SS_ERR_CANCELLED, vm->ip, NULL, "task cancelled");
        if (vm->instruction_count++ >= vm->instruction_limit)
            return vm_fail(vm, SS_ERR_LIMIT, vm->ip, NULL, "instruction limit exceeded");
        if (vm->ip >= vm->chunk->code_count)
            return vm_fail(vm, SS_ERR_JUMP, vm->ip, NULL, "instruction pointer is out of range");
        instruction_ip = vm->ip;
        ins = &vm->chunk->code[vm->ip++];
        ++vm->opcode_counts[ins->opcode];
        frame = current_frame(vm);
        if (vm->trace) trace_instruction(vm, instruction_ip, ins);
        switch ((OpCode)ins->opcode) {
            case OP_NOP: break;
            case OP_LOAD_CONST: frame->registers[ins->a] = vm->chunk->constants[ins->imm]; break;
            case OP_MOVE: case OP_STATE_COPY:
                frame->registers[ins->a] = frame->registers[ins->b];
                if (ins->opcode == OP_MOVE && vm->chunk->version >= 3u)
                    error = clone_history(vm, &frame->histories[ins->b],
                                          &frame->histories[ins->a]);
                break;
            case OP_STATE_NEW: {
                const Value *p = &vm->chunk->constants[ins->b];
                const Value *d = &vm->chunk->constants[ins->c];
                SSState state;
                if (p->type != VAL_NUMBER || d->type != VAL_NUMBER) error = SS_ERR_TYPE;
                else error = ss_state_make(p->as.number, d->as.number, &state);
                if (error == SS_OK) error = store_state(vm, ins->a, ss_value_state(state).as.state);
                break;
            }
            case OP_STATE_BUILD: {
                SSState state;
                if (frame->registers[ins->a].type != VAL_NUMBER ||
                    frame->registers[ins->b].type != VAL_NUMBER) error = SS_ERR_TYPE;
                else error = ss_state_make(frame->registers[ins->a].as.number,
                                           frame->registers[ins->b].as.number, &state);
                if (error == SS_OK) error = store_state(vm, ins->c, ss_value_state(state).as.state);
                break;
            }
            case OP_STATE_NEW_V3: {
                const Value *p = &vm->chunk->constants[ins->b];
                const Value *d_re = &vm->chunk->constants[ins->c];
                const Value *d_im = &vm->chunk->constants[ins->imm];
                SSState state;
                if (p->type != VAL_NUMBER || d_re->type != VAL_NUMBER ||
                    d_im->type != VAL_NUMBER) error = SS_ERR_TYPE;
                else error = ss_state_make_complex(p->as.number, d_re->as.number,
                                                   d_im->as.number, &state);
                if (error == SS_OK)
                    error = store_state(vm, ins->a, ss_value_state(state).as.state);
                break;
            }
            case OP_STATE_BUILD_V3: {
                SSState state;
                if (frame->registers[ins->a].type != VAL_NUMBER ||
                    frame->registers[ins->b].type != VAL_NUMBER ||
                    frame->registers[ins->c].type != VAL_NUMBER) error = SS_ERR_TYPE;
                else error = ss_state_make_complex(frame->registers[ins->a].as.number,
                                                   frame->registers[ins->b].as.number,
                                                   frame->registers[ins->c].as.number,
                                                   &state);
                if (error == SS_OK)
                    error = store_state(vm, ins->imm, ss_value_state(state).as.state);
                break;
            }
            case OP_STATE_SET:
                if (require_type(&frame->registers[ins->b], VAL_STATE) != SS_OK) error = SS_ERR_TYPE;
                else error = store_state(vm, ins->a, frame->registers[ins->b].as.state);
                break;
            case OP_APPLY:
                if (require_type(&frame->registers[ins->a], VAL_STATE) != SS_OK || require_type(&frame->registers[ins->b], VAL_STATE) != SS_OK) error = SS_ERR_TYPE;
                else {
                    SSStateValue target = frame->registers[ins->b].as.state;
                    error = ss_apply(&frame->registers[ins->a].as.state.state, &target.state);
                    if (error == SS_OK) error = store_state(vm, ins->b, target);
                    if (vm->trace && error == SS_OK) {
                        const SSState *after = &frame->registers[ins->b].as.state.state;
                        (void)printf("  R%u after: p=%.12g d=%.12g\n", ins->b, after->p, after->d);
                    }
                }
                break;
            case OP_APPLY_MANY: {
                SSStateValue target;
                size_t index, strongest = 0;
                double total_weight = 0.0, aggregate_p = 0.0, aggregate_d = 0.0;
                if (frame->registers[ins->c].type != VAL_STATE) { error = SS_ERR_TYPE; break; }
                target = frame->registers[ins->c].as.state;
                if (ins->imm == SS_AGG_SEQUENTIAL) {
                    for (index = 0; index < ins->b && error == SS_OK; ++index) {
                        Value *source = &frame->registers[ins->a + index];
                        if (source->type != VAL_STATE) error = SS_ERR_TYPE;
                        else error = ss_apply(&source->as.state.state, &target.state);
                    }
                } else {
                    double strongest_score = -1.0;
                    for (index = 0; index < ins->b; ++index) {
                        Value *source = &frame->registers[ins->a + index];
                        double weight, score;
                        if (source->type != VAL_STATE) { error = SS_ERR_TYPE; break; }
                        weight = ins->imm == SS_AGG_MEAN ? 1.0 :
                                 (source->as.state.indexes.has_weight ? source->as.state.indexes.weight : 1.0) *
                                 (source->as.state.indexes.has_confidence ? source->as.state.indexes.confidence : 1.0);
                        score = fabs(source->as.state.state.d) * weight;
                        if (score > strongest_score) { strongest_score = score; strongest = index; }
                        aggregate_p += source->as.state.state.p * weight;
                        aggregate_d += source->as.state.state.d * weight;
                        total_weight += weight;
                    }
                    if (error == SS_OK && ins->imm == SS_AGG_STRONGEST)
                        error = ss_apply(&frame->registers[ins->a + strongest].as.state.state, &target.state);
                    else if (error == SS_OK && total_weight <= 0.0) error = SS_ERR_COMPOSE;
                    else if (error == SS_OK) {
                        SSState aggregate;
                        error = ss_state_make(aggregate_p / total_weight, aggregate_d / total_weight, &aggregate);
                        if (error == SS_OK) error = ss_apply(&aggregate, &target.state);
                    }
                }
                if (error == SS_OK) error = store_state(vm, ins->c, target);
                break;
            }
            case OP_TRANSFORM:
                if (frame->registers[ins->a].type != VAL_STATE) error = SS_ERR_TYPE;
                else if (ins->b >= sizeof(transform_registry) / sizeof(transform_registry[0])) error = SS_ERR_TRANSFORM;
                else {
                    SSStateValue state = frame->registers[ins->a].as.state;
                    error = transform_registry[ins->b](vm, &state, &frame->registers[ins->c], ins->imm);
                    if (error == SS_OK) error = store_state(vm, ins->a, state);
                }
                break;
            case OP_TRANSFORM_V3: {
                const Value *source = &frame->registers[ins->b];
                if (source->type == VAL_STATE) {
                    SSStateValue transformed = source->as.state;
                    error = ss_transform_v3_apply(ins->c, &source->as.state.state,
                                                  &transformed.state);
                    if (error == SS_OK) error = store_state(vm, ins->a, transformed);
                } else if (source->type == VAL_STATE_DIST) {
                    SSStateDist *distribution;
                    error = ss_vm_state_dist_transform(vm, ins->c,
                                                       source->as.state_dist,
                                                       &distribution);
                    if (error == SS_OK)
                        frame->registers[ins->a] = ss_value_state_dist(distribution);
                } else error = SS_ERR_TYPE;
                break;
            }
            case OP_APPEND: {
                SSStateDist *distribution;
                error = ss_vm_state_dist_append(vm, &frame->registers[ins->a],
                                                &frame->registers[ins->b],
                                                &distribution);
                if (error == SS_OK)
                    frame->registers[ins->c] = ss_value_state_dist(distribution);
                break;
            }
            case OP_COMPOSE_MERGE: {
                SSState state;
                if (frame->registers[ins->a].type != VAL_STATE || frame->registers[ins->b].type != VAL_STATE) error = SS_ERR_TYPE;
                else error = ss_compose_merge(&frame->registers[ins->a].as.state.state,
                                              &frame->registers[ins->b].as.state.state, &state);
                if (error == SS_OK) error = store_state(vm, ins->c, ss_value_state(state).as.state);
                break;
            }
            case OP_COMPOSE_UPDATE: {
                SSState left, right;
                if (frame->registers[ins->a].type != VAL_STATE || frame->registers[ins->b].type != VAL_STATE) error = SS_ERR_TYPE;
                else error = ss_compose_update(&frame->registers[ins->a].as.state.state,
                                               &frame->registers[ins->b].as.state.state, &left, &right);
                if (error == SS_OK) error = store_state(vm, ins->a, (SSStateValue){.state = left});
                if (error == SS_OK) error = store_state(vm, ins->b, (SSStateValue){.state = right});
                break;
            }
            case OP_COMPOSE_JOINT: {
                SSJointState *joint;
                Value pair[2];
                if (frame->registers[ins->a].type != VAL_STATE || frame->registers[ins->b].type != VAL_STATE) { error = SS_ERR_TYPE; break; }
                pair[0] = frame->registers[ins->a]; pair[1] = frame->registers[ins->b];
                error = ss_vm_joint_build(vm, pair, 2u, "independent:left,right", &joint);
                if (error == SS_OK) frame->registers[ins->c] = (Value){.type = VAL_JOINT_STATE, .as.joint = joint};
                break;
            }
            case OP_JOINT_BUILD_V5: {
                const Value *descriptor;
                SSJointState *joint;
                /* The descriptor operand is a constant index, not a register. */
                if (ins->imm >= vm->chunk->constant_count ||
                    vm->chunk->constants[ins->imm].type != VAL_STRING) error = SS_ERR_TYPE;
                else {
                    descriptor = &vm->chunk->constants[ins->imm];
                    error = ss_vm_joint_build(vm, &frame->registers[ins->b], ins->c,
                                              descriptor->as.string, &joint);
                }
                if (error == SS_OK) frame->registers[ins->a] = (Value){.type = VAL_JOINT_STATE, .as.joint = joint};
                break;
            }
            case OP_JOINT_PROJECT_V5: {
                SSJointState *joint;
                const Value *source = &frame->registers[ins->a];
                if (source->type != VAL_JOINT_STATE || ins->c >= vm->chunk->constant_count ||
                    vm->chunk->constants[ins->c].type != VAL_STRING) error = SS_ERR_TYPE;
                else error = ss_vm_joint_project(vm, source->as.joint,
                                                 vm->chunk->constants[ins->c].as.string, &joint);
                if (error == SS_OK) frame->registers[ins->b] = (Value){.type = VAL_JOINT_STATE, .as.joint = joint};
                break;
            }
            case OP_JOINT_CONDITION_V5: {
                SSJointState *joint;
                const Value *source = &frame->registers[ins->a];
                if (source->type != VAL_JOINT_STATE || ins->c >= vm->chunk->constant_count ||
                    vm->chunk->constants[ins->c].type != VAL_STRING || ins->imm >= SS_MAX_REGISTERS)
                    error = SS_ERR_TYPE;
                else error = ss_vm_joint_condition(vm, source->as.joint,
                                                    vm->chunk->constants[ins->c].as.string,
                                                    &frame->registers[ins->imm], &joint);
                if (error == SS_OK) frame->registers[ins->b] = (Value){.type = VAL_JOINT_STATE, .as.joint = joint};
                break;
            }
            case OP_JOINT_SAMPLE_V5: {
                if (frame->registers[ins->a].type != VAL_JOINT_STATE) error = SS_ERR_TYPE;
                else error = ss_vm_joint_sample(vm, frame->registers[ins->a].as.joint,
                                                &frame->registers[ins->b]);
                break;
            }
            case OP_RESOLVE_V5:
                error = ss_vm_information_resolve(vm, &frame->registers[ins->a],
                                                  &frame->registers[ins->b]);
                break;
            case OP_JOINT_BUILD_FINITE_V5: {
                SSJointState *joint;
                if (ins->c >= vm->chunk->constant_count ||
                    vm->chunk->constants[ins->c].type != VAL_STRING)
                    error = SS_ERR_TYPE;
                else error = joint_build_finite_array(
                    vm, &frame->registers[ins->a],
                    vm->chunk->constants[ins->c].as.string, &joint);
                if (error == SS_OK)
                    frame->registers[ins->b] = (Value){.type = VAL_JOINT_STATE,
                                                       .as.joint = joint};
                break;
            }
            case OP_JOINT_RENAME_V5: {
                SSJointState *joint;
                if (frame->registers[ins->a].type != VAL_JOINT_STATE ||
                    ins->c >= vm->chunk->constant_count ||
                    ins->imm >= vm->chunk->constant_count ||
                    vm->chunk->constants[ins->c].type != VAL_STRING ||
                    vm->chunk->constants[ins->imm].type != VAL_STRING)
                    error = SS_ERR_TYPE;
                else error = ss_vm_joint_rename(
                    vm, frame->registers[ins->a].as.joint,
                    vm->chunk->constants[ins->c].as.string,
                    vm->chunk->constants[ins->imm].as.string, &joint);
                if (error == SS_OK)
                    frame->registers[ins->b] = (Value){.type = VAL_JOINT_STATE,
                                                       .as.joint = joint};
                break;
            }
            case OP_POSSIBILITY_BUILD_V5: {
                SSPossibility *possibility;
                const Value *source = &frame->registers[ins->a];
                if (source->type != VAL_ARRAY || source->as.array == NULL)
                    error = SS_ERR_TYPE;
                else error = ss_vm_possibility_build(vm, source->as.array->items,
                                                      source->as.array->count,
                                                      &possibility);
                if (error == SS_OK)
                    frame->registers[ins->b] = ss_value_possibility(possibility);
                break;
            }
            case OP_PATH_SPLIT_V5:
                error = path_split(vm, &frame->registers[ins->a], ins->imm);
                break;
            case OP_PATH_JOIN_V5:
                error = path_join(vm);
                break;
            case OP_OBSERVE_V5: {
                SSJointState *joint;
                const Value *source = &frame->registers[ins->a];
                if (vm->active_path_count > 1u) { error = SS_ERR_UNSUPPORTED_OPERATION; break; }
                if (source->type != VAL_JOINT_STATE ||
                    ins->c >= vm->chunk->constant_count ||
                    vm->chunk->constants[ins->c].type != VAL_STRING)
                    error = SS_ERR_TYPE;
                else error = ss_vm_joint_observe(
                    vm, source->as.joint, vm->chunk->constants[ins->c].as.string,
                    &frame->registers[ins->imm], &joint);
                if (error == SS_OK) {
                    frame->registers[ins->b] = (Value){.type = VAL_JOINT_STATE,
                                                       .as.joint = joint};
                }
                break;
            }
            case OP_INFO_SAMPLE_V5:
                if (vm->active_path_count > 1u) error = SS_ERR_UNSUPPORTED_OPERATION;
                else error = ss_vm_information_sample(vm, &frame->registers[ins->a],
                                                       &frame->registers[ins->b]);
                break;
            case OP_MEASURE:
                if (frame->registers[ins->a].type != VAL_STATE) error = SS_ERR_TYPE;
                else if (ins->c >= sizeof(measure_registry) / sizeof(measure_registry[0])) error = SS_ERR_MEASURE;
                else {
                    SSStateValue state = frame->registers[ins->a].as.state;
                    error = measure_registry[ins->c](vm, &state, &frame->registers[ins->b], NULL, 0);
                    if (error == SS_OK && ins->c == SS_MEASURE_COLLAPSE) error = store_state(vm, ins->a, state);
                    if (vm->trace && error == SS_OK) { (void)printf("  output: "); ss_value_print(&frame->registers[ins->b]); (void)printf("\n"); }
                }
                break;
            case OP_MEASURE_V3: {
                const Value *source = &frame->registers[ins->a];
                double probability;
                if (source->type == VAL_STATE) probability = source->as.state.state.p;
                else if (source->type == VAL_STATE_DIST)
                    error = ss_vm_state_dist_expected_probability(source->as.state_dist,
                                                                 &probability);
                else error = SS_ERR_TYPE;
                if (error == SS_OK && ins->c == SS_MEASURE_V3_PROBABILITY)
                    frame->registers[ins->b] = ss_value_number(probability);
                else if (error == SS_OK && ins->c == SS_MEASURE_V3_DISTRIBUTION)
                    frame->registers[ins->b] = ss_value_distribution(1.0 - probability,
                                                                     probability);
                else if (error == SS_OK && ins->c == SS_MEASURE_V3_SAMPLE)
                    frame->registers[ins->b] = ss_value_sample(draw_sample(vm, probability));
                else if (error == SS_OK) error = SS_ERR_MEASURE;
                break;
            }
            case OP_MEASURE_BASIS_V4: {
                const Value *source = &frame->registers[ins->a];
                double probability;
                if (source->type == VAL_STATE) {
                    error = measure_basis_state(&source->as.state, ins->c, &probability);
                    if (error == SS_OK && ins->imm == SS_MEASURE_V3_PROBABILITY)
                        frame->registers[ins->b] = ss_value_number(probability);
                    else if (error == SS_OK && ins->imm == SS_MEASURE_V3_DISTRIBUTION)
                        frame->registers[ins->b] = ss_value_distribution(1.0 - probability,
                                                                           probability);
                    else if (error == SS_OK && ins->imm == SS_MEASURE_V3_SAMPLE)
                        frame->registers[ins->b] = ss_value_sample(draw_sample(vm, probability));
                    else if (error == SS_OK) error = SS_ERR_MEASURE;
                } else if (source->type == VAL_STATE_DIST) {
                    SSStateValue state;
                    if (ins->imm != SS_MEASURE_V3_SAMPLE) {
                        error = SS_ERR_UNSUPPORTED_EXACT_MEASUREMENT;
                    } else {
                        error = ss_vm_state_dist_sample(vm, source->as.state_dist, &state);
                        if (error == SS_OK)
                            error = measure_basis_state(&state, ins->c, &probability);
                        if (error == SS_OK)
                            frame->registers[ins->b] = ss_value_sample(draw_sample(vm, probability));
                    }
                } else {
                    error = SS_ERR_TYPE;
                }
                break;
            }
            case OP_ESTIMATE_MEASURE_PROBABILITY_V4:
            case OP_ESTIMATE_MEASURE_DISTRIBUTION_V4: {
                const Value *source = &frame->registers[ins->a];
                double probability;
                if (source->type != VAL_STATE_DIST) {
                    error = SS_ERR_TYPE;
                } else {
                    error = estimate_basis_probability(vm, source->as.state_dist,
                                                       ins->c, ins->imm, &probability);
                    if (error == SS_OK && ins->opcode == OP_ESTIMATE_MEASURE_PROBABILITY_V4)
                        frame->registers[ins->b] = ss_value_number(probability);
                    else if (error == SS_OK)
                        frame->registers[ins->b] = ss_value_distribution(1.0 - probability,
                                                                           probability);
                }
                break;
            }
            case OP_SAMPLE_STATE_DIST: {
                SSStateValue state;
                if (frame->registers[ins->a].type != VAL_STATE_DIST)
                    error = SS_ERR_TYPE;
                else error = ss_vm_state_dist_sample(vm,
                                                     frame->registers[ins->a].as.state_dist,
                                                     &state);
                if (error == SS_OK) error = store_state(vm, ins->b, state);
                break;
            }
            case OP_GET_FIELD: {
                const Value *source = &frame->registers[ins->a];
                if (source->type == VAL_STATE && ins->c <= 2u) {
                    double field = ins->c == 0u ? source->as.state.state.p :
                                   ins->c == 1u ? source->as.state.state.d_re :
                                                 source->as.state.state.d_im;
                    frame->registers[ins->b] = ss_value_number(field);
                }
                else if (source->type == VAL_DISTRIBUTION && ins->c <= 1u)
                    frame->registers[ins->b] = ss_value_number(ins->c == 0u ? source->as.distribution.p0 : source->as.distribution.p1);
                else error = SS_ERR_TYPE;
                break;
            }
            case OP_GET_INDEX: {
                const SSIndexes *indexes;
                if (frame->registers[ins->a].type != VAL_STATE) { error = SS_ERR_TYPE; break; }
                indexes = &frame->registers[ins->a].as.state.indexes;
                if (ins->c == 0u && indexes->has_timestamp) frame->registers[ins->b] = ss_value_number(indexes->timestamp);
                else if (ins->c == 1u && indexes->has_source) frame->registers[ins->b] = ss_value_string(indexes->source);
                else if (ins->c == 2u && indexes->has_weight) frame->registers[ins->b] = ss_value_number(indexes->weight);
                else if (ins->c == 3u && indexes->has_confidence) frame->registers[ins->b] = ss_value_number(indexes->confidence);
                else frame->registers[ins->b] = ss_value_null();
                break;
            }
            case OP_SET_INDEX: {
                SSStateValue state;
                const Value *source = &frame->registers[ins->c];
                if (frame->registers[ins->a].type != VAL_STATE) { error = SS_ERR_TYPE; break; }
                state = frame->registers[ins->a].as.state;
                if (ins->b == 0u && source->type == VAL_NUMBER) { state.indexes.has_timestamp = true; state.indexes.timestamp = source->as.number; }
                else if (ins->b == 1u && source->type == VAL_STRING) { state.indexes.has_source = true; state.indexes.source = source->as.string; }
                else if (ins->b == 2u && source->type == VAL_NUMBER && source->as.number >= 0.0) { state.indexes.has_weight = true; state.indexes.weight = source->as.number; }
                else if (ins->b == 3u && source->type == VAL_NUMBER && source->as.number >= 0.0 && source->as.number <= 1.0) { state.indexes.has_confidence = true; state.indexes.confidence = source->as.number; }
                else { error = SS_ERR_TYPE; break; }
                error = store_state(vm, ins->a, state); break;
            }
            case OP_HISTORY_CONFIG: {
                SSHistory *history = &frame->histories[ins->a];
                if (frame->registers[ins->a].type != VAL_STATE || frame->registers[ins->b].type != VAL_NUMBER ||
                    ins->c > SS_HISTORY_DURATION || frame->registers[ins->b].as.number <= 0.0) { error = SS_ERR_HISTORY; break; }
                history->policy = (SSHistoryPolicy)ins->c; history->amount = frame->registers[ins->b].as.number;
                error = history_append(vm, history, frame->registers[ins->a].as.state); break;
            }
            case OP_PREVIOUS: case OP_CHANGE: case OP_VELOCITY: {
                SSHistory *history = &frame->histories[ins->a];
                SSStateValue *current, *previous;
                if (history->count < 2u) { error = SS_ERR_HISTORY; break; }
                current = &history->versions[history->count - 1u]; previous = &history->versions[history->count - 2u];
                if (ins->opcode == OP_PREVIOUS) frame->registers[ins->b] = (Value){.type = VAL_STATE, .as.state = *previous};
                else if (ins->opcode == OP_CHANGE) frame->registers[ins->b] = ss_value_number(current->state.p - previous->state.p);
                else if (!current->indexes.has_timestamp || !previous->indexes.has_timestamp || current->indexes.timestamp <= previous->indexes.timestamp) error = SS_ERR_HISTORY;
                else frame->registers[ins->b] = ss_value_number((current->state.p - previous->state.p) / (current->indexes.timestamp - previous->indexes.timestamp));
                break;
            }
            case OP_BINARY: {
                const Value *left = &frame->registers[ins->a], *right = &frame->registers[ins->b];
                error = lift_binary(vm, left, right, SS_PURE_BINARY, ins->imm,
                                    &frame->registers[ins->c]);
                break;
            }
            case OP_UNARY:
                error = lift_unary(vm, &frame->registers[ins->a], ins->imm,
                                   &frame->registers[ins->b]);
                break;
            case OP_COMPARE: {
                const Value *left = &frame->registers[ins->a], *right = &frame->registers[ins->b];
                error = lift_binary(vm, left, right, SS_PURE_COMPARE, ins->imm,
                                    &frame->registers[ins->c]);
                break;
            }
            case OP_JUMP: vm->ip = ins->imm; break;
            case OP_JUMP_IF_TRUE: case OP_JUMP_IF_FALSE:
                if (frame->registers[ins->a].type != VAL_BOOL) error = SS_ERR_TYPE;
                else if ((ins->opcode == OP_JUMP_IF_TRUE && frame->registers[ins->a].as.boolean) ||
                         (ins->opcode == OP_JUMP_IF_FALSE && !frame->registers[ins->a].as.boolean)) vm->ip = ins->imm;
                break;
            case OP_ARRAY_NEW: {
                SSArray *array = ss_vm_alloc(vm, sizeof(*array));
                if (array == NULL) { error = SS_ERR_OOM; break; }
                array->count = ins->c; array->capacity = ins->c; array->items = ss_vm_alloc(vm, ins->c * sizeof(*array->items));
                if (array->items == NULL && ins->c > 0u) { error = SS_ERR_OOM; break; }
                if (ins->c > 0u) memcpy(array->items, &frame->registers[ins->b], ins->c * sizeof(*array->items));
                frame->registers[ins->a].type = VAL_ARRAY; frame->registers[ins->a].as.array = array; break;
            }
            case OP_ARRAY_GET: case OP_ARRAY_SET: {
                Value *array_value = &frame->registers[ins->a];
                const Value *index_value = &frame->registers[ins->b];
                size_t index;
                if (array_value->type != VAL_ARRAY || index_value->type != VAL_NUMBER || index_value->as.number < 0.0 || floor(index_value->as.number) != index_value->as.number) { error = SS_ERR_TYPE; break; }
                index = (size_t)index_value->as.number;
                if (index >= array_value->as.array->count) { error = SS_ERR_LIMIT; break; }
                if (ins->opcode == OP_ARRAY_GET) frame->registers[ins->c] = array_value->as.array->items[index];
                else array_value->as.array->items[index] = frame->registers[ins->c];
                break;
            }
            case OP_CALL: {
                SSFrame *callee;
                const SSFunction *function = &vm->chunk->functions[ins->b];
                size_t index;
                if (ins->imm != function->arity) { error = SS_ERR_TYPE; break; }
                if (vm->frame_count >= SS_MAX_CALL_FRAMES) { error = SS_ERR_LIMIT; break; }
                callee = &vm->frames[vm->frame_count++];
                for (index = 0; index < function->register_count; ++index) {
                    callee->registers[index] = ss_value_null();
                    free(callee->histories[index].versions);
                    memset(&callee->histories[index], 0, sizeof(callee->histories[index]));
                }
                callee->return_ip = vm->ip; callee->return_register = ins->a; callee->function = ins->b;
                for (index = 0; index < ins->imm; ++index) {
                    callee->registers[index] = frame->registers[ins->c + index];
                    if (vm->chunk->version >= 3u) {
                        error = clone_history(vm, &frame->histories[ins->c + index],
                                              &callee->histories[index]);
                        if (error != SS_OK) break;
                    }
                }
                if (error != SS_OK) break;
                vm->ip = function->entry; break;
            }
            case OP_FORK: {
                SSTask *task;
                size_t argument;
                if (vm->active_path_count > 1u) { error = SS_ERR_UNSUPPORTED_OPERATION; break; }
                for (argument = 0; argument < ins->imm; ++argument)
                    if (value_is_unresolved(&frame->registers[ins->c + argument])) {
                        error = SS_ERR_UNRESOLVED_VALUE; break;
                    }
                if (error != SS_OK) break;
                error = start_task(vm, ins->b, &frame->registers[ins->c],
                                   &frame->histories[ins->c], ins->imm, &task);
                if (error == SS_OK) {
                    frame->registers[ins->a].type = VAL_TASK;
                    frame->registers[ins->a].as.task = task;
                    if (vm->trace) (void)printf("  forked task %llu\n", (unsigned long long)task->id);
                }
                break;
            }
            case OP_JOIN: {
                Value *task_value = &frame->registers[ins->a];
                if (task_value->type != VAL_TASK) error = SS_ERR_TYPE;
                else error = wait_task(vm, task_value->as.task, -1.0, &frame->registers[ins->b]);
                break;
            }
            case OP_JOIN_TIMEOUT: {
                Value *task_value = &frame->registers[ins->a];
                Value *timeout = &frame->registers[ins->b];
                if (task_value->type != VAL_TASK || timeout->type != VAL_NUMBER || timeout->as.number < 0.0)
                    error = SS_ERR_TYPE;
                else error = wait_task(vm, task_value->as.task, timeout->as.number,
                                       &frame->registers[ins->c]);
                break;
            }
            case OP_JOIN_ALL: {
                Value *tasks = &frame->registers[ins->a];
                SSArray *results;
                size_t index;
                if (tasks->type != VAL_ARRAY) { error = SS_ERR_TYPE; break; }
                results = ss_vm_alloc(vm, sizeof(*results));
                if (results == NULL) { error = SS_ERR_OOM; break; }
                results->count = tasks->as.array->count;
                results->capacity = results->count;
                results->items = ss_vm_alloc(vm, results->count * sizeof(*results->items));
                if (results->items == NULL && results->count > 0u) { error = SS_ERR_OOM; break; }
                for (index = 0; index < results->count && error == SS_OK; ++index) {
                    Value *task_value = &tasks->as.array->items[index];
                    if (task_value->type != VAL_TASK) error = SS_ERR_TYPE;
                    else error = wait_task(vm, task_value->as.task, -1.0, &results->items[index]);
                }
                if (error == SS_OK) {
                    frame->registers[ins->b].type = VAL_ARRAY;
                    frame->registers[ins->b].as.array = results;
                }
                break;
            }
            case OP_CANCEL:
                if (frame->registers[ins->a].type != VAL_TASK) error = SS_ERR_TYPE;
                else cancel_task(frame->registers[ins->a].as.task);
                break;
            case OP_TASKGROUP_ENTER:
                if (vm->group_depth >= SS_MAX_CALL_FRAMES) error = SS_ERR_LIMIT;
                else {
                    vm->group_stack[vm->group_depth++] = vm->current_group_id;
                    vm->current_group_id = vm->next_group_id++;
                }
                break;
            case OP_TASKGROUP_EXIT: {
                uint64_t group_id = vm->current_group_id;
                if (vm->group_depth == 0u) error = SS_ERR_TASK;
                else {
                    vm->current_group_id = vm->group_stack[--vm->group_depth];
                    error = close_task_group(vm, group_id);
                }
                break;
            }
            case OP_HOST_CALL: {
                size_t argument;
                if (vm->active_path_count > 1u) { error = SS_ERR_UNSUPPORTED_OPERATION; break; }
                for (argument = 0; argument < ins->imm; ++argument)
                    if (value_is_unresolved(&frame->registers[ins->c + argument])) {
                        error = SS_ERR_UNRESOLVED_VALUE; break;
                    }
                if (error == SS_OK)
                    error = execute_host_call(vm, ins->b, &frame->registers[ins->c], ins->imm,
                                              &frame->registers[ins->a]);
                break;
            }
            case OP_RETURN: {
                Value returned = frame->registers[ins->a];
                if (vm->frame_count == 1u) { vm->result = returned; vm->running = false; }
                else { size_t return_ip = frame->return_ip; uint32_t destination = frame->return_register; --vm->frame_count; current_frame(vm)->registers[destination] = returned; vm->ip = return_ip; }
                break;
            }
            case OP_PRINT:
                if (vm->active_path_count > 1u) error = SS_ERR_UNSUPPORTED_OPERATION;
                else if (value_is_unresolved(&frame->registers[ins->a]))
                    error = SS_ERR_UNRESOLVED_VALUE;
                else { ss_value_print(&frame->registers[ins->a]); (void)printf("\n"); }
                break;
            case OP_HALT:
                if (vm->path_execution != NULL) error = SS_ERR_UNSUPPORTED_OPERATION;
                else vm->running = false;
                break;
            case OP_COUNT: error = SS_ERR_OPCODE; break;
        }
        if (error != SS_OK) return vm_fail(vm, error, instruction_ip, ins, ss_error_name(error));
    }
    return SS_OK;
}
