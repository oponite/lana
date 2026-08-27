#define _POSIX_C_SOURCE 200809L

#include "lana/vm.h"
#include "lana/data.h"
#include "lana/shared.h"

#include <math.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/types.h>

extern char *realpath(const char *path, char *resolved_path);

struct LanaScheduler {
    pthread_mutex_t mutex;
    pthread_cond_t available;
    pthread_t *workers;
    size_t worker_count;
    size_t task_limit;
    size_t live_tasks;
    uint64_t next_task_id;
    bool stopping;
    LanaTask *queue_head;
    LanaTask *queue_tail;
    LanaTask *all_tasks;
};

struct LanaSharedReference {
    LanaSharedInformation *shared;
    struct LanaSharedReference *next;
};

static LanaError vm_track_shared(LanaVM *vm, LanaSharedInformation *shared,
                                 bool retain) {
    LanaSharedReference *reference = malloc(sizeof(*reference));
    if (reference == NULL) return LANA_ERR_OOM;
    if (retain) lana_shared_information_retain(shared);
    reference->shared = shared;
    reference->next = vm->shared_references;
    vm->shared_references = reference;
    return LANA_OK;
}

struct LanaPathExecution {
    LanaFrame *false_frames;
    LanaFrame *true_frames;
    size_t frame_count;
    size_t false_ip;
    uint64_t dependency_id;
    double true_weight;
    double false_weight;
    size_t previous_path_count;
    bool running_false;
    struct LanaPathExecution *next;
};

static uint64_t mix64(uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30u)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27u)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31u);
}

static LanaFrame *current_frame(LanaVM *vm) { return &vm->frames[vm->frame_count - 1u]; }
static LanaError consume_sampling_budget(LanaVM *vm);
static bool joint_value_is_definite(const Value *value);
static bool value_is_unresolved(const Value *value);
static LanaError reactive_recompute_transaction(LanaVM *vm, LanaReactive *root,
                                                const Value *replacement);

static const Value *reactive_value(const Value *value) {
    if (value != NULL && value->reactive != NULL &&
        value->reactive->current != NULL)
        return value->reactive->current;
    return value;
}

static LanaError clone_value(LanaVM *destination, const Value *source, Value *out);
static LanaError wait_task(LanaVM *vm, LanaTask *task, double timeout_seconds, Value *out);
static void scheduler_shutdown(LanaScheduler *scheduler);
static void scheduler_destroy(LanaScheduler *scheduler);

static void gc_trace_value_contents(LanaGC *gc, const Value *value);
static void gc_trace_value_block(LanaGC *gc, void *payload);
static void gc_trace_array(LanaGC *gc, void *payload);
static void gc_trace_map(LanaGC *gc, void *payload);
static void gc_trace_joint(LanaGC *gc, void *payload);
static void gc_trace_possibility(LanaGC *gc, void *payload);
static void gc_trace_paths(LanaGC *gc, void *payload);
static void gc_trace_state_dist(LanaGC *gc, void *payload);
static void gc_trace_derivation(LanaGC *gc, void *payload);
static void gc_trace_reactive(LanaGC *gc, void *payload);
static void gc_trace_claim(LanaGC *gc, void *payload);
static void gc_trace_planned_effect(LanaGC *gc, void *payload);
static void gc_trace_effect_receipt(LanaGC *gc, void *payload);
static void gc_trace_frame_block(LanaGC *gc, void *payload);
static void gc_trace_path_execution(LanaGC *gc, void *payload);

static bool gc_is_managed(LanaGC *gc, const void *payload) {
    return payload != NULL && lana_gc_payload_size(gc, payload) > 0u;
}

static void gc_mark_leaf(LanaGC *gc, const void *payload, LanaGCObjectKind kind) {
    if (!gc_is_managed(gc, payload)) return;
    (void)lana_gc_configure(gc, (void *)payload, kind, LANA_GC_OWNER_VM, NULL);
    (void)lana_gc_mark(gc, (void *)payload);
}

static void gc_mark_object(LanaGC *gc, const void *payload, LanaGCObjectKind kind,
                           LanaGCTraceFn trace) {
    if (!gc_is_managed(gc, payload)) return;
    (void)lana_gc_configure(gc, (void *)payload, kind, LANA_GC_OWNER_VM, trace);
    (void)lana_gc_mark(gc, (void *)payload);
}

static void gc_mark_value_block(LanaGC *gc, const Value *values) {
    gc_mark_object(gc, values, LANA_GC_VALUE_ARRAY, gc_trace_value_block);
}

static void gc_mark_state_source(LanaGC *gc, const LanaStateValue *state) {
    if (state != NULL && state->indexes.has_source)
        gc_mark_leaf(gc, state->indexes.source, LANA_GC_STRING);
}

static void gc_trace_value_pointer(LanaGC *gc, void *payload) {
    gc_trace_value_contents(gc, payload);
}

static void gc_mark_value_pointer(LanaGC *gc, const Value *value) {
    gc_mark_object(gc, value, LANA_GC_VALUE, gc_trace_value_pointer);
}

static void gc_trace_array(LanaGC *gc, void *payload) {
    LanaArray *array = payload;
    if (array->items != NULL) gc_mark_value_block(gc, array->items);
}

static void gc_trace_map(LanaGC *gc, void *payload) {
    LanaMap *map = payload;
    size_t index;
    gc_mark_leaf(gc, map->entries, LANA_GC_OPAQUE);
    for (index = 0u; index < map->count; ++index) {
        gc_mark_leaf(gc, map->entries[index].key, LANA_GC_STRING);
        gc_mark_value_pointer(gc, map->entries[index].value);
    }
}

static void gc_trace_joint(LanaGC *gc, void *payload) {
    LanaJointState *joint = payload;
    size_t index;
    gc_mark_leaf(gc, joint->names, LANA_GC_OPAQUE);
    gc_mark_leaf(gc, joint->domains, LANA_GC_OPAQUE);
    for (index = 0u; index < joint->count; ++index)
        gc_mark_leaf(gc, joint->names[index], LANA_GC_STRING);
    if (joint->values != NULL) gc_mark_value_block(gc, joint->values);
    gc_mark_leaf(gc, joint->rows, LANA_GC_OPAQUE);
    for (index = 0u; index < joint->row_count; ++index)
        if (joint->rows[index].values != NULL)
            gc_mark_value_block(gc, joint->rows[index].values);
}

static void gc_trace_possibility(LanaGC *gc, void *payload) {
    LanaPossibility *possibility = payload;
    if (possibility->values != NULL) gc_mark_value_block(gc, possibility->values);
    gc_mark_leaf(gc, possibility->weights, LANA_GC_OPAQUE);
}

static void gc_trace_paths(LanaGC *gc, void *payload) {
    LanaPathSet *paths = payload;
    size_t index;
    gc_mark_leaf(gc, paths->alternatives, LANA_GC_OPAQUE);
    for (index = 0u; index < paths->count; ++index)
        gc_mark_value_pointer(gc, paths->alternatives[index].result);
}

static void gc_trace_state_dist(LanaGC *gc, void *payload) {
    LanaStateDist *distribution = payload;
    switch (distribution->kind) {
        case LANA_DIST_DIRAC:
            gc_mark_state_source(gc, &distribution->as.dirac);
            break;
        case LANA_DIST_APPEND:
            gc_mark_object(gc, distribution->as.append.left, LANA_GC_STATE_DIST,
                           gc_trace_state_dist);
            gc_mark_object(gc, distribution->as.append.right, LANA_GC_STATE_DIST,
                           gc_trace_state_dist);
            break;
        case LANA_DIST_TRANSFORM:
            gc_mark_object(gc, distribution->as.transform.child, LANA_GC_STATE_DIST,
                           gc_trace_state_dist);
            break;
    }
}

static void gc_trace_derivation(LanaGC *gc, void *payload) {
    LanaDerivation *derivation = payload;
    size_t index;
    gc_mark_leaf(gc, derivation->operation, LANA_GC_STRING);
    gc_mark_leaf(gc, derivation->label, LANA_GC_STRING);
    gc_mark_leaf(gc, derivation->function, LANA_GC_STRING);
    gc_mark_leaf(gc, derivation->details, LANA_GC_STRING);
    gc_mark_leaf(gc, derivation->reason, LANA_GC_STRING);
    gc_mark_leaf(gc, derivation->inputs, LANA_GC_OPAQUE);
    for (index = 0u; index < derivation->input_count; ++index)
        gc_mark_object(gc, derivation->inputs[index], LANA_GC_DERIVATION,
                       gc_trace_derivation);
}

static void gc_trace_reactive(LanaGC *gc, void *payload) {
    LanaReactive *node = payload;
    size_t index;
    for (index = 0u; index < 2u; ++index) {
        gc_mark_object(gc, node->inputs[index], LANA_GC_REACTIVE,
                       gc_trace_reactive);
        gc_mark_value_pointer(gc, node->constants[index]);
    }
    gc_mark_value_pointer(gc, node->current);
    gc_mark_object(gc, node->history, LANA_GC_REACTIVE_HISTORY, NULL);
    for (index = 0u; index < node->history_count; ++index)
        gc_mark_value_pointer(gc, node->history[index].value);
}

static void gc_trace_claim(LanaGC *gc, void *payload) {
    LanaClaim *claim = payload;
    gc_mark_value_pointer(gc, claim->value);
    gc_mark_leaf(gc, claim->proposition, LANA_GC_STRING);
}

static void gc_trace_effect_receipt(LanaGC *gc, void *payload) {
    LanaEffectReceipt *receipt = payload;
    gc_mark_value_pointer(gc, receipt->result);
    gc_mark_object(gc, receipt->next, LANA_GC_EFFECT_RECEIPT,
                   gc_trace_effect_receipt);
}

static void gc_trace_planned_effect(LanaGC *gc, void *payload) {
    LanaPlannedEffect *plan = payload;
    gc_mark_leaf(gc, plan->kind, LANA_GC_STRING);
    gc_mark_value_pointer(gc, plan->payload);
    gc_mark_object(gc, plan->receipts, LANA_GC_EFFECT_RECEIPT,
                   gc_trace_effect_receipt);
}

static void gc_trace_value_contents(LanaGC *gc, const Value *value) {
    if (value == NULL) return;
    gc_mark_object(gc, value->derivation, LANA_GC_DERIVATION, gc_trace_derivation);
    gc_mark_object(gc, value->reactive, LANA_GC_REACTIVE, gc_trace_reactive);
    gc_mark_object(gc, value->claim, LANA_GC_CLAIM, gc_trace_claim);
    gc_mark_object(gc, value->planned_effect, LANA_GC_PLANNED_EFFECT,
                   gc_trace_planned_effect);
    switch (value->type) {
        case VAL_STRING:
            gc_mark_leaf(gc, value->as.string, LANA_GC_STRING);
            break;
        case VAL_STATE:
            gc_mark_state_source(gc, &value->as.state);
            break;
        case VAL_JOINT_STATE:
            gc_mark_object(gc, value->as.joint, LANA_GC_JOINT, gc_trace_joint);
            break;
        case VAL_ARRAY:
            gc_mark_object(gc, value->as.array, LANA_GC_ARRAY, gc_trace_array);
            break;
        case VAL_STATE_DIST:
            gc_mark_object(gc, value->as.state_dist, LANA_GC_STATE_DIST,
                           gc_trace_state_dist);
            break;
        case VAL_MAP:
            gc_mark_object(gc, value->as.map, LANA_GC_MAP, gc_trace_map);
            break;
        case VAL_POSSIBILITY:
            gc_mark_object(gc, value->as.possibility, LANA_GC_POSSIBILITY,
                           gc_trace_possibility);
            break;
        case VAL_PATH_SET:
            gc_mark_object(gc, value->as.paths, LANA_GC_PATH_SET, gc_trace_paths);
            break;
        case VAL_NULL:
        case VAL_NUMBER:
        case VAL_BOOL:
        case VAL_DISTRIBUTION:
        case VAL_SAMPLE:
        case VAL_FUNCTION:
        case VAL_TASK:
        case VAL_SHARED_CAPABILITY:
            break;
    }
}

static void gc_trace_value_block(LanaGC *gc, void *payload) {
    Value *values = payload;
    size_t count = lana_gc_payload_size(gc, payload) / sizeof(*values);
    size_t index;
    for (index = 0u; index < count; ++index)
        gc_trace_value_contents(gc, &values[index]);
}

static void gc_trace_history(LanaGC *gc, const LanaHistory *history) {
    size_t index;
    gc_mark_leaf(gc, history->versions, LANA_GC_OPAQUE);
    for (index = 0u; index < history->count; ++index)
        gc_mark_state_source(gc, &history->versions[index]);
}

static void gc_trace_frame(LanaGC *gc, const LanaFrame *frame) {
    size_t index;
    for (index = 0u; index < LANA_MAX_REGISTERS; ++index) {
        gc_trace_value_contents(gc, &frame->registers[index]);
        gc_trace_history(gc, &frame->histories[index]);
    }
}

static void gc_trace_frame_block(LanaGC *gc, void *payload) {
    LanaFrame *frames = payload;
    size_t count = lana_gc_payload_size(gc, payload) / sizeof(*frames);
    size_t index;
    for (index = 0u; index < count; ++index) gc_trace_frame(gc, &frames[index]);
}

static void gc_trace_path_execution(LanaGC *gc, void *payload) {
    LanaPathExecution *execution = payload;
    gc_mark_object(gc, execution->false_frames, LANA_GC_RUNTIME_INTERNAL,
                   gc_trace_frame_block);
    gc_mark_object(gc, execution->true_frames, LANA_GC_RUNTIME_INTERNAL,
                   gc_trace_frame_block);
    gc_mark_object(gc, execution->next, LANA_GC_RUNTIME_INTERNAL,
                   gc_trace_path_execution);
}

static void gc_trace_vm_roots(LanaGC *gc, void *context) {
    LanaVM *vm = context;
    LanaTask *task;
    size_t frame_index;
    for (frame_index = 0u; frame_index < vm->frame_count; ++frame_index)
        gc_trace_frame(gc, &vm->frames[frame_index]);
    gc_trace_value_contents(gc, &vm->result);
    gc_mark_object(gc, vm->path_execution, LANA_GC_RUNTIME_INTERNAL,
                   gc_trace_path_execution);
    for (task = vm->tasks; task != NULL; task = task->next)
        if (task->joined) gc_trace_value_contents(gc, &task->result);
}

static LanaError vm_fail(LanaVM *vm, LanaError code, size_t ip, const LanaInstruction *ins,
                       const char *message) {
    bool has_derivation = vm->error.has_derivation;
    uint64_t derivation_task_lineage = vm->error.derivation_task_lineage;
    uint64_t derivation_local_sequence = vm->error.derivation_local_sequence;
    if (vm->error.code != LANA_OK && vm->error.message[0] != '\0') {
        vm->result = lana_value_null();
        vm->running = false;
        return code;
    }
    lana_error_set(&vm->error, code, ip, ins == NULL ? OP_NOP : ins->opcode,
                 ins == NULL ? 0u : ins->line, "%s", message);
    if (vm->frame_count > 0u && current_frame(vm)->function < vm->chunk->function_count) {
        const char *name = vm->chunk->functions[current_frame(vm)->function].name;
        (void)snprintf(vm->error.function, sizeof(vm->error.function), "%s", name);
    }
    lana_error_set_source_span(&vm->error,
                               vm->error.function[0] == '\0' ? "<bytecode>" : vm->error.function,
                               ins == NULL ? 0u : ins->line, 1u,
                               ins == NULL ? 0u : ins->line, 1u);
    lana_error_set_operation(&vm->error, ins == NULL ? "execute" : lana_opcode_name(ins->opcode));
    if (code == LANA_ERR_INVALID_CONDITIONING) {
        lana_error_set_resolution(&vm->error, LANA_RESOLUTION_REASON_INVALID_CONDITIONING, 0u);
    } else if (code == LANA_ERR_UNRESOLVED_VALUE) {
        size_t alternatives = 0u;
        if (ins != NULL && vm->frame_count > 0u && ins->a < LANA_MAX_REGISTERS) {
            const Value *source = &current_frame(vm)->registers[ins->a];
            if (source->type == VAL_JOINT_STATE && source->as.joint->rows != NULL)
                alternatives = source->as.joint->row_count;
            else if (source->type == VAL_POSSIBILITY)
                alternatives = source->as.possibility->count;
            else if (source->type == VAL_PATH_SET)
                alternatives = source->as.paths->count;
        }
        lana_error_set_resolution(&vm->error,
                                  alternatives == 0u ? LANA_RESOLUTION_REASON_NO_ALTERNATIVES
                                                     : LANA_RESOLUTION_REASON_MULTIPLE_ALTERNATIVES,
                                  alternatives);
    } else if (code == LANA_ERR_UNSUPPORTED_EXACT_MEASUREMENT) {
        lana_error_set_resolution(&vm->error, LANA_RESOLUTION_REASON_UNSUPPORTED_EXACT, 0u);
        lana_error_set_exact_support(&vm->error, LANA_EXACT_SUPPORT_UNAVAILABLE,
                                     "operation requires explicit sampling or approximation");
    } else if (code == LANA_ERR_CANCELLED) {
        lana_error_set_resolution(&vm->error, LANA_RESOLUTION_REASON_CANCELLED, 0u);
        lana_error_set_cancellation(&vm->error, vm->lineage, message);
    } else if (code == LANA_ERR_OOM) {
        lana_error_set_resource_limit(&vm->error, LANA_RESOURCE_MEMORY,
                                      (uint64_t)vm->memory_limit,
                                      (uint64_t)vm->allocated_bytes, "bytes");
    } else if (code == LANA_ERR_PATH_LIMIT) {
        lana_error_set_resolution(&vm->error, LANA_RESOLUTION_REASON_RESOURCE_LIMIT, 0u);
        lana_error_set_resource_limit(&vm->error, LANA_RESOURCE_PATHS,
                                      (uint64_t)vm->path_limit,
                                      (uint64_t)vm->active_path_count, "paths");
    } else if (code == LANA_ERR_BUDGET_EXHAUSTED ||
               (code == LANA_ERR_LIMIT && message != NULL &&
                strstr(message, "instruction") != NULL)) {
        lana_error_set_resolution(&vm->error, LANA_RESOLUTION_REASON_RESOURCE_LIMIT, 0u);
        lana_error_set_resource_limit(&vm->error, LANA_RESOURCE_INSTRUCTIONS,
                                      vm->instruction_limit, vm->instruction_count,
                                      "instructions");
    } else if (code == LANA_ERR_LIMIT && ins != NULL && ins->opcode == OP_FORK) {
        uint64_t observed = vm->scheduler == NULL ? 0u : (uint64_t)vm->scheduler->live_tasks;
        lana_error_set_resource_limit(&vm->error, LANA_RESOURCE_TASKS,
                                      (uint64_t)vm->configured_task_limit,
                                      observed, "tasks");
    }
    if (has_derivation) {
        vm->error.has_derivation = true;
        vm->error.derivation_task_lineage = derivation_task_lineage;
        vm->error.derivation_local_sequence = derivation_local_sequence;
    }
    vm->result = lana_value_null();
    vm->running = false;
    return code;
}

void *lana_vm_alloc(LanaVM *vm, size_t size) {
    void *pointer;
    if (vm == NULL) return NULL;
    vm->gc.memory_limit = vm->memory_limit;
    pointer = lana_gc_alloc(&vm->gc, size, LANA_GC_OPAQUE,
                            LANA_GC_OWNER_NATIVE, NULL);
    if (pointer != NULL && !lana_gc_publish(&vm->gc, pointer)) return NULL;
    vm->allocated_bytes = vm->gc.allocated_bytes;
    vm->allocation_count = vm->gc.allocation_count;
    return pointer;
}

bool lana_vm_collect(LanaVM *vm) {
    bool collected;
    if (vm == NULL) return false;
    vm->gc.memory_limit = vm->memory_limit;
    lana_gc_set_deferred(&vm->gc, false);
    lana_gc_release_native(&vm->gc);
    collected = lana_gc_collect(&vm->gc);
    lana_gc_set_deferred(&vm->gc, true);
    vm->allocated_bytes = vm->gc.allocated_bytes;
    vm->allocation_count = vm->gc.allocation_count;
    return collected && vm->allocated_bytes <= vm->memory_limit;
}

size_t lana_vm_root_push(LanaVM *vm, Value *value) {
    if (vm == NULL) return SIZE_MAX;
    return lana_gc_root_push(&vm->gc, value, gc_trace_value_pointer);
}

void lana_vm_root_pop(LanaVM *vm, size_t previous_count) {
    if (vm != NULL) lana_gc_root_pop(&vm->gc, previous_count);
}

void lana_vm_write_barrier_value(LanaVM *vm, void *owner,
                                 const Value *value) {
    void *target = NULL;
    if (vm == NULL || owner == NULL || value == NULL) return;
    if (value->reactive != NULL) target = value->reactive;
    else if (value->claim != NULL) target = value->claim;
    else if (value->planned_effect != NULL) target = value->planned_effect;
    else if (value->type == VAL_STRING) target = (void *)value->as.string;
    else if (value->type == VAL_ARRAY) target = value->as.array;
    else if (value->type == VAL_MAP) target = value->as.map;
    else if (value->type == VAL_JOINT_STATE) target = value->as.joint;
    else if (value->type == VAL_STATE_DIST) target = value->as.state_dist;
    else if (value->type == VAL_POSSIBILITY) target = value->as.possibility;
    else if (value->type == VAL_PATH_SET) target = value->as.paths;
    if (target != NULL) (void)lana_gc_write_barrier(&vm->gc, owner, target);
}

static bool vm_gc_safepoint(LanaVM *vm) {
    size_t threshold;
    if (vm == NULL) return false;
    vm->gc.memory_limit = vm->memory_limit;
    threshold = vm->memory_limit / 2u;
    if (threshold == 0u && vm->memory_limit > 0u) threshold = 1u;
    vm->gc.collection_threshold = threshold;
    lana_gc_release_native(&vm->gc);
    if ((vm->gc.allocated_bytes >= threshold && vm->gc.allocated_bytes > 0u) ||
        vm->gc.allocated_bytes > vm->memory_limit) {
        lana_gc_set_deferred(&vm->gc, false);
        if (!lana_gc_collect_young(&vm->gc)) {
            lana_gc_set_deferred(&vm->gc, true);
            return false;
        }
        if (vm->gc.allocated_bytes > vm->memory_limit &&
            !lana_gc_collect(&vm->gc)) {
            lana_gc_set_deferred(&vm->gc, true);
            return false;
        }
    }
    lana_gc_set_deferred(&vm->gc, true);
    vm->allocated_bytes = vm->gc.allocated_bytes;
    vm->allocation_count = vm->gc.allocation_count;
    return vm->allocated_bytes <= vm->memory_limit;
}

static const char *derivation_kind_name(LanaDerivationKind kind) {
    static const char *names[] = {
        "evidence", "assumption", "operation", "observation",
        "path", "sample", "approximation", "resolution"
    };
    return (size_t)kind < sizeof(names) / sizeof(names[0]) ? names[kind] : "operation";
}

static const char *derivation_exactness_name(LanaDerivationExactness exactness) {
    static const char *names[] = {"exact", "sample", "approximate"};
    return (size_t)exactness < sizeof(names) / sizeof(names[0]) ? names[exactness] : "exact";
}

static const char *derivation_outcome_name(LanaDerivationOutcome outcome) {
    static const char *names[] = {"success", "unresolved", "unsupported", "error"};
    return (size_t)outcome < sizeof(names) / sizeof(names[0]) ? names[outcome] : "error";
}

static char *derivation_string(LanaVM *vm, const char *text) {
    size_t length;
    char *copy;
    if (text == NULL) text = "";
    length = strlen(text);
    copy = lana_vm_alloc(vm, length + 1u);
    if (copy != NULL) memcpy(copy, text, length + 1u);
    return copy;
}

static const char *derivation_function_name(const LanaVM *vm) {
    const LanaFrame *frame;
    if (vm == NULL || vm->chunk == NULL || vm->frame_count == 0u) return "<main>";
    frame = &vm->frames[vm->frame_count - 1u];
    if (frame->function >= vm->chunk->function_count) return "<main>";
    return vm->chunk->functions[frame->function].name;
}

static LanaDerivation *record_derivation(LanaVM *vm, LanaDerivationKind kind,
                                         const char *operation,
                                         const Value *const *inputs, size_t input_count,
                                         const char *label, uint32_t line,
                                         LanaDerivationExactness exactness,
                                         const char *details,
                                         LanaDerivationOutcome outcome,
                                         const char *reason) {
    LanaDerivation *node;
    size_t index, retained = 0u;
    if (vm == NULL) return NULL;
    node = lana_vm_alloc(vm, sizeof(*node));
    if (node == NULL) return NULL;
    for (index = 0; index < input_count; ++index)
        if (inputs[index] != NULL && inputs[index]->derivation != NULL) ++retained;
    node->inputs = retained == 0u ? NULL :
        lana_vm_alloc(vm, retained * sizeof(*node->inputs));
    if (retained > 0u && node->inputs == NULL) return NULL;
    retained = 0u;
    for (index = 0; index < input_count; ++index)
        if (inputs[index] != NULL && inputs[index]->derivation != NULL)
            node->inputs[retained++] = inputs[index]->derivation;
    node->input_count = retained;
    node->task_lineage = vm->lineage;
    node->local_sequence = ++vm->derivation_sequence;
    node->revision = vm->revision;
    node->kind = kind;
    node->operation = derivation_string(vm, operation);
    node->label = derivation_string(vm, label);
    node->function = derivation_string(vm, derivation_function_name(vm));
    node->line = line;
    node->exactness = exactness;
    node->details = derivation_string(vm, details);
    node->outcome = outcome;
    node->reason = derivation_string(vm, reason == NULL ? "none" : reason);
    if (node->operation == NULL || node->label == NULL || node->function == NULL ||
        node->details == NULL || node->reason == NULL) return NULL;
    return node;
}

static LanaError attach_derivation(LanaVM *vm, Value *out, LanaDerivationKind kind,
                                   const char *operation,
                                   const Value *const *inputs, size_t input_count,
                                   const char *label, uint32_t line,
                                   LanaDerivationExactness exactness,
                                   const char *details) {
    out->derivation = record_derivation(vm, kind, operation, inputs, input_count,
                                        label, line, exactness, details,
                                        LANA_DERIVATION_SUCCESS, "none");
    return out->derivation == NULL ? LANA_ERR_OOM : LANA_OK;
}

LanaError lana_vm_provenance_root(LanaVM *vm, const Value *source, const char *label,
                              uint32_t line, bool assumption, Value *out) {
    if (vm == NULL || source == NULL || label == NULL || out == NULL)
        return LANA_ERR_FORMAT;
    *out = *source;
    out->derivation = record_derivation(
        vm, assumption ? LANA_DERIVATION_ASSUMPTION : LANA_DERIVATION_EVIDENCE,
        assumption ? "assume" : "evidence", NULL, 0u, label, line,
        LANA_EXACTNESS_EXACT, "root", LANA_DERIVATION_SUCCESS, "none");
    return out->derivation == NULL ? LANA_ERR_OOM : LANA_OK;
}

static LanaError map_put(LanaVM *vm, LanaMap *map, const char *key, Value value) {
    return lana_map_set(vm, map, key, &value, true);
}

static LanaError derivation_id_to_value(LanaVM *vm, const LanaDerivation *node,
                                        Value *out) {
    LanaArray *id;
    if (node == NULL || out == NULL) return LANA_ERR_FORMAT;
    id = lana_vm_alloc(vm, sizeof(*id));
    if (id == NULL) return LANA_ERR_OOM;
    id->count = id->capacity = 2u;
    id->items = lana_vm_alloc(vm, 2u * sizeof(*id->items));
    if (id->items == NULL) return LANA_ERR_OOM;
    id->items[0] = lana_value_number((double)node->task_lineage);
    id->items[1] = lana_value_number((double)node->local_sequence);
    *out = lana_value_array(id);
    return LANA_OK;
}

static LanaError derivation_to_value(LanaVM *vm, const LanaDerivation *node,
                                     Value *out) {
    LanaMap *map, *source_map, *details_map;
    LanaArray *inputs;
    Value id;
    size_t index;
    LanaError error;
    if (node == NULL) return LANA_ERR_FORMAT;
    if ((error = lana_map_new(vm, 12u, &map)) != LANA_OK) return error;
    inputs = lana_vm_alloc(vm, sizeof(*inputs));
    if (inputs == NULL) return LANA_ERR_OOM;
    inputs->count = inputs->capacity = node->input_count;
    inputs->items = node->input_count == 0u ? NULL :
        lana_vm_alloc(vm, node->input_count * sizeof(*inputs->items));
    if (node->input_count > 0u && inputs->items == NULL)
        return LANA_ERR_OOM;
    if ((error = derivation_id_to_value(vm, node, &id)) != LANA_OK) return error;
    for (index = 0; index < node->input_count; ++index) {
        error = derivation_id_to_value(vm, node->inputs[index], &inputs->items[index]);
        if (error != LANA_OK) return error;
    }
    if ((error = lana_map_new(vm, 3u, &source_map)) != LANA_OK ||
        (error = lana_map_new(vm, 1u, &details_map)) != LANA_OK) return error;
    if ((error = map_put(vm, source_map, "label", lana_value_string(node->label))) != LANA_OK ||
        (error = map_put(vm, source_map, "function", lana_value_string(node->function))) != LANA_OK ||
        (error = map_put(vm, source_map, "line", lana_value_number((double)node->line))) != LANA_OK ||
        (error = map_put(vm, details_map, "summary", lana_value_string(node->details))) != LANA_OK)
        return error;
    if ((error = map_put(vm, map, "id", id)) != LANA_OK ||
        (error = map_put(vm, map, "revision", lana_value_number((double)node->revision))) != LANA_OK ||
        (error = map_put(vm, map, "kind", lana_value_string(derivation_kind_name(node->kind)))) != LANA_OK ||
        (error = map_put(vm, map, "operation", lana_value_string(node->operation))) != LANA_OK ||
        (error = map_put(vm, map, "inputs", lana_value_array(inputs))) != LANA_OK ||
        (error = map_put(vm, map, "source", lana_value_map(source_map))) != LANA_OK ||
        (error = map_put(vm, map, "exactness", lana_value_string(derivation_exactness_name(node->exactness)))) != LANA_OK ||
        (error = map_put(vm, map, "details", lana_value_map(details_map))) != LANA_OK ||
        (error = map_put(vm, map, "outcome", lana_value_string(derivation_outcome_name(node->outcome)))) != LANA_OK ||
        (error = map_put(vm, map, "reason", lana_value_string(node->reason))) != LANA_OK)
        return error;
    *out = lana_value_map(map);
    return LANA_OK;
}

LanaError lana_vm_derivation(LanaVM *vm, const Value *source, Value *out) {
    if (vm == NULL || source == NULL || out == NULL) return LANA_ERR_FORMAT;
    if (source->derivation == NULL) return LANA_ERR_UNSUPPORTED_OPERATION;
    return derivation_to_value(vm, source->derivation, out);
}

LanaError lana_vm_explain(LanaVM *vm, const Value *source, Value *out) {
    const LanaDerivation *node;
    char rendered[1024];
    int length;
    char *copy;
    if (vm == NULL || source == NULL || out == NULL) return LANA_ERR_FORMAT;
    node = source->derivation;
    if (node == NULL) return LANA_ERR_UNSUPPORTED_OPERATION;
    length = snprintf(rendered, sizeof(rendered),
        "%s %s id=[%llu,%llu] revision=%llu exactness=%s outcome=%s reason=%s label=%s inputs=%zu",
        derivation_kind_name(node->kind), node->operation,
        (unsigned long long)node->task_lineage,
        (unsigned long long)node->local_sequence,
        (unsigned long long)node->revision,
        derivation_exactness_name(node->exactness),
        derivation_outcome_name(node->outcome), node->reason, node->label,
        node->input_count);
    if (length < 0 || (size_t)length >= sizeof(rendered)) return LANA_ERR_LIMIT;
    copy = derivation_string(vm, rendered);
    if (copy == NULL) return LANA_ERR_OOM;
    *out = lana_value_string(copy);
    return LANA_OK;
}

uint32_t lana_vm_random(LanaVM *vm) {
    uint64_t old_state = vm->rng_state;
    uint32_t xor_shifted;
    uint32_t rotation;
    vm->rng_state = old_state * UINT64_C(6364136223846793005) + vm->rng_increment;
    xor_shifted = (uint32_t)(((old_state >> 18u) ^ old_state) >> 27u);
    rotation = (uint32_t)(old_state >> 59u);
    return (xor_shifted >> rotation) | (xor_shifted << ((0u - rotation) & 31u));
}

void lana_vm_seed(LanaVM *vm, uint64_t seed) {
    vm->root_seed = seed;
    vm->rng_state = 0u;
    vm->rng_increment = (UINT64_C(1442695040888963407) << 1u) | 1u;
    (void)lana_vm_random(vm);
    vm->rng_state += seed;
    (void)lana_vm_random(vm);
}

void lana_vm_init(LanaVM *vm, const LanaChunk *chunk) {
    size_t frame_index, register_index;
    memset(vm, 0, sizeof(*vm));
    vm->chunk = chunk;
    vm->ip = chunk == NULL ? 0u : chunk->entry;
    vm->running = true;
    vm->instruction_limit = UINT64_C(10000000);
    vm->memory_limit = 64u * 1024u * 1024u;
    lana_gc_init(&vm->gc, vm->memory_limit, vm->memory_limit / 2u,
                 gc_trace_vm_roots, vm);
    lana_gc_set_deferred(&vm->gc, true);
    vm->frame_count = 1u;
    vm->frames[0].function = UINT32_MAX;
    vm->result = lana_value_null();
    vm->next_task_id = 1u;
    vm->next_group_id = 1u;
    vm->configured_worker_count = 1u;
#if defined(_SC_NPROCESSORS_ONLN)
    {
        long processors = sysconf(_SC_NPROCESSORS_ONLN);
        vm->configured_worker_count = processors > 0 && processors < 8 ? (size_t)processors : 8u;
    }
#endif
    vm->configured_task_limit = 64u;
    vm->path_limit = 64u;
    vm->active_path_count = 1u;
    vm->next_dependency_id = 1u;
    vm->next_reactive_id = 1u;
    vm->next_effect_id = 1u;
    atomic_init(&vm->cancelled, false);
    for (frame_index = 0; frame_index < LANA_MAX_CALL_FRAMES; ++frame_index)
        for (register_index = 0; register_index < LANA_MAX_REGISTERS; ++register_index)
            vm->frames[frame_index].registers[register_index] = lana_value_null();
    lana_vm_seed(vm, UINT64_C(0x4c414e41));
}

void lana_vm_set_program_args(LanaVM *vm, int argc, const char **argv) {
    if (vm == NULL) return;
    vm->program_argc = argc;
    vm->program_argv = argv;
}

LanaError lana_vm_set_worker_count(LanaVM *vm, size_t workers) {
    if (vm == NULL || workers == 0u) return LANA_ERR_TASK;
    if (vm->scheduler != NULL) return LANA_ERR_TASK;
    vm->configured_worker_count = workers;
    return LANA_OK;
}

LanaError lana_vm_set_task_limit(LanaVM *vm, size_t tasks) {
    if (vm == NULL || tasks == 0u) return LANA_ERR_TASK;
    if (vm->scheduler != NULL) return LANA_ERR_TASK;
    vm->configured_task_limit = tasks;
    return LANA_OK;
}

static void cancel_task(LanaTask *task) {
    if (task != NULL && task->child != NULL) atomic_store(&task->child->cancelled, true);
}

static void destroy_task(LanaTask *task) {
    LanaScheduler *scheduler;
    LanaTask **cursor;
    if (task == NULL) return;
    cancel_task(task);
    if (task->child != NULL) { lana_vm_free(task->child); free(task->child); }
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

void lana_vm_free(LanaVM *vm) {
    LanaTask *task;
    LanaSharedReference *reference;
    if (vm == NULL) return;
    LanaScheduler *owned_scheduler = vm->scheduler_owner ? vm->scheduler : NULL;
    if (owned_scheduler != NULL) scheduler_shutdown(owned_scheduler);
    task = vm->tasks;
    while (task != NULL) {
        LanaTask *next = task->next;
        destroy_task(task);
        task = next;
    }
    vm->tasks = NULL;
    reference = vm->shared_references;
    while (reference != NULL) {
        LanaSharedReference *next = reference->next;
        lana_shared_information_release(reference->shared);
        free(reference);
        reference = next;
    }
    vm->shared_references = NULL;
    lana_gc_free(&vm->gc);
    vm->allocated_bytes = 0;
    if (owned_scheduler != NULL) {
        scheduler_destroy(owned_scheduler);
        vm->scheduler = NULL;
    }
}

static LanaError clone_state_value(LanaVM *destination, const LanaStateValue *source,
                                 LanaStateValue *out) {
    size_t length;
    char *source_copy;
    *out = *source;
    if (!source->indexes.has_source || source->indexes.source == NULL) return LANA_OK;
    length = strlen(source->indexes.source);
    source_copy = lana_vm_alloc(destination, length + 1u);
    if (source_copy == NULL) return LANA_ERR_OOM;
    memcpy(source_copy, source->indexes.source, length + 1u);
    out->indexes.source = source_copy;
    return LANA_OK;
}

static LanaError clone_history(LanaVM *destination, const LanaHistory *source,
                             LanaHistory *out) {
    size_t index;
    if (source == out) return LANA_OK;
    memset(out, 0, sizeof(*out));
    out->policy = source->policy; out->amount = source->amount;
    if (source->count == 0u) return LANA_OK;
    out->versions = lana_vm_alloc(destination,
                                  source->count * sizeof(*out->versions));
    if (out->versions == NULL) return LANA_ERR_OOM;
    out->capacity = source->count;
    for (index = 0; index < source->count; ++index) {
        LanaError error = clone_state_value(destination, &source->versions[index],
                                          &out->versions[index]);
        if (error != LANA_OK) return error;
        ++out->count;
    }
    return LANA_OK;
}

typedef struct LanaDistCloneMemo {
    const LanaStateDist *source;
    LanaStateDist *copy;
    struct LanaDistCloneMemo *next;
} LanaDistCloneMemo;

typedef struct LanaContainerCloneMemo {
    const void *source;
    void *copy;
    ValueType type;
    struct LanaContainerCloneMemo *next;
} LanaContainerCloneMemo;

typedef struct LanaDerivationCloneMemo {
    const LanaDerivation *source;
    LanaDerivation *copy;
    struct LanaDerivationCloneMemo *next;
} LanaDerivationCloneMemo;

static LanaError clone_derivation_node(LanaVM *destination,
                                       const LanaDerivation *source,
                                       LanaDerivation **out,
                                       LanaDerivationCloneMemo **memo) {
    LanaDerivationCloneMemo *entry;
    LanaDerivation *copy;
    size_t index;
    LanaError error;
    if (source == NULL) { *out = NULL; return LANA_OK; }
    for (entry = *memo; entry != NULL; entry = entry->next) {
        if (entry->source == source) { *out = entry->copy; return LANA_OK; }
    }
    copy = lana_vm_alloc(destination, sizeof(*copy));
    entry = malloc(sizeof(*entry));
    if (copy == NULL || entry == NULL) { free(entry); return LANA_ERR_OOM; }
    entry->source = source; entry->copy = copy; entry->next = *memo; *memo = entry;
    *copy = *source;
    copy->operation = derivation_string(destination, source->operation);
    copy->label = derivation_string(destination, source->label);
    copy->function = derivation_string(destination, source->function);
    copy->details = derivation_string(destination, source->details);
    copy->reason = derivation_string(destination, source->reason);
    copy->inputs = source->input_count == 0u ? NULL :
        lana_vm_alloc(destination, source->input_count * sizeof(*copy->inputs));
    if (copy->operation == NULL || copy->label == NULL || copy->function == NULL ||
        copy->details == NULL || copy->reason == NULL ||
        (source->input_count > 0u && copy->inputs == NULL)) return LANA_ERR_OOM;
    for (index = 0; index < source->input_count; ++index) {
        error = clone_derivation_node(destination, source->inputs[index],
                                      &copy->inputs[index], memo);
        if (error != LANA_OK) return error;
    }
    *out = copy;
    return LANA_OK;
}

static LanaError clone_state_dist_node(LanaVM *destination, const LanaStateDist *source,
                                     LanaStateDist **out, LanaDistCloneMemo **memo) {
    LanaDistCloneMemo *entry;
    LanaStateDist *copy;
    LanaError error = LANA_OK;
    if (source == NULL || out == NULL) return LANA_ERR_INVALID_DISTRIBUTION;
    for (entry = *memo; entry != NULL; entry = entry->next) {
        if (entry->source == source) {
            *out = entry->copy;
            return LANA_OK;
        }
    }
    copy = lana_vm_alloc(destination, sizeof(*copy));
    if (copy == NULL) return LANA_ERR_OOM;
    entry = malloc(sizeof(*entry));
    if (entry == NULL) return LANA_ERR_OOM;
    entry->source = source;
    entry->copy = copy;
    entry->next = *memo;
    *memo = entry;
    copy->kind = source->kind;
    switch (source->kind) {
        case LANA_DIST_DIRAC:
            error = clone_state_value(destination, &source->as.dirac, &copy->as.dirac);
            break;
        case LANA_DIST_APPEND:
            copy->as.append = source->as.append;
            error = clone_state_dist_node(destination, source->as.append.left,
                                          &copy->as.append.left, memo);
            if (error == LANA_OK)
                error = clone_state_dist_node(destination, source->as.append.right,
                                              &copy->as.append.right, memo);
            break;
        case LANA_DIST_TRANSFORM:
            copy->as.transform.transform_id = source->as.transform.transform_id;
            error = clone_state_dist_node(destination, source->as.transform.child,
                                          &copy->as.transform.child, memo);
            break;
        default:
            error = LANA_ERR_INVALID_DISTRIBUTION;
            break;
    }
    return error;
}

static LanaError clone_value_memo(LanaVM *destination, const Value *source, Value *out,
                                LanaDistCloneMemo **memo,
                                LanaContainerCloneMemo **containers,
                                LanaDerivationCloneMemo **derivations) {
    const Value *effective = source;
    bool local_reactive = source->reactive == NULL ||
        lana_gc_payload_size(&destination->gc, source->reactive) > 0u;
    bool local_claim = source->claim == NULL ||
        lana_gc_payload_size(&destination->gc, source->claim) > 0u;
    bool local_plan = source->planned_effect == NULL ||
        lana_gc_payload_size(&destination->gc, source->planned_effect) > 0u;
    size_t index, length;
    LanaError error;
    if (!local_reactive) effective = reactive_value(source);
    *out = *effective;
    if (!local_reactive) out->reactive = NULL;
    if (!local_claim) out->claim = NULL;
    if (!local_plan) out->planned_effect = NULL;
    error = clone_derivation_node(destination, effective->derivation,
                                  &out->derivation, derivations);
    if (error != LANA_OK) return error;
    source = effective;
    if (source->type == VAL_STRING) {
        char *copy;
        length = strlen(source->as.string);
        copy = lana_vm_alloc(destination, length + 1u);
        if (copy == NULL) return LANA_ERR_OOM;
        memcpy(copy, source->as.string, length + 1u);
        out->as.string = copy;
    } else if (source->type == VAL_STATE) {
        error = clone_state_value(destination, &source->as.state, &out->as.state);
        if (error != LANA_OK) return error;
    } else if (source->type == VAL_ARRAY) {
        LanaArray *array;
        LanaContainerCloneMemo *entry;
        for (entry = *containers; entry != NULL; entry = entry->next)
            if (entry->type == VAL_ARRAY && entry->source == source->as.array) {
                out->as.array = entry->copy; return LANA_OK;
            }
        array = lana_vm_alloc(destination, sizeof(*array));
        if (array == NULL) return LANA_ERR_OOM;
        entry = malloc(sizeof(*entry));
        if (entry == NULL) return LANA_ERR_OOM;
        entry->source = source->as.array; entry->copy = array; entry->type = VAL_ARRAY;
        entry->next = *containers; *containers = entry;
        array->count = source->as.array->count;
        array->capacity = array->count;
        array->items = lana_vm_alloc(destination, array->count * sizeof(*array->items));
        if (array->items == NULL && array->count > 0u) return LANA_ERR_OOM;
        for (index = 0; index < array->count; ++index) {
            error = clone_value_memo(destination, &source->as.array->items[index],
                                     &array->items[index], memo, containers, derivations);
            if (error != LANA_OK) return error;
        }
        out->as.array = array;
    } else if (source->type == VAL_JOINT_STATE) {
        LanaJointState *joint;
        LanaContainerCloneMemo *entry;
        for (entry = *containers; entry != NULL; entry = entry->next)
            if (entry->type == VAL_JOINT_STATE && entry->source == source->as.joint) {
                out->as.joint = entry->copy; return LANA_OK;
            }
        joint = lana_vm_alloc(destination, sizeof(*joint));
        if (joint == NULL) return LANA_ERR_OOM;
        entry = malloc(sizeof(*entry));
        if (entry == NULL) return LANA_ERR_OOM;
        entry->source = source->as.joint; entry->copy = joint;
        entry->type = VAL_JOINT_STATE; entry->next = *containers; *containers = entry;
        joint->count = source->as.joint->count;
        joint->kind = source->as.joint->kind;
        joint->capabilities = source->as.joint->capabilities;
        joint->row_count = source->as.joint->row_count;
        joint->names = lana_vm_alloc(destination, joint->count * sizeof(*joint->names));
        joint->domains = lana_vm_alloc(destination, joint->count * sizeof(*joint->domains));
        joint->values = source->as.joint->values == NULL ? NULL :
            lana_vm_alloc(destination, joint->count * sizeof(*joint->values));
        joint->rows = joint->row_count == 0u ? NULL :
            lana_vm_alloc(destination, joint->row_count * sizeof(*joint->rows));
        if ((joint->names == NULL || joint->domains == NULL ||
             (source->as.joint->values != NULL && joint->values == NULL) ||
             (joint->row_count > 0u && joint->rows == NULL)) && joint->count > 0u)
            return LANA_ERR_OOM;
        for (index = 0; index < joint->count; ++index) {
            size_t length = strlen(source->as.joint->names[index]);
            joint->names[index] = lana_vm_alloc(destination, length + 1u);
            if (joint->names[index] == NULL) return LANA_ERR_OOM;
            memcpy(joint->names[index], source->as.joint->names[index], length + 1u);
            joint->domains[index] = source->as.joint->domains[index];
            if (joint->values != NULL) {
                error = clone_value_memo(destination, &source->as.joint->values[index],
                                         &joint->values[index], memo, containers, derivations);
                if (error != LANA_OK) return error;
            }
        }
        for (index = 0; index < joint->row_count; ++index) {
            size_t column;
            joint->rows[index].weight = source->as.joint->rows[index].weight;
            joint->rows[index].values = lana_vm_alloc(
                destination, joint->count * sizeof(*joint->rows[index].values));
            if (joint->rows[index].values == NULL && joint->count > 0u) return LANA_ERR_OOM;
            for (column = 0; column < joint->count; ++column) {
                error = clone_value_memo(destination,
                    &source->as.joint->rows[index].values[column],
                    &joint->rows[index].values[column], memo, containers, derivations);
                if (error != LANA_OK) return error;
            }
        }
        out->as.joint = joint;
    } else if (source->type == VAL_STATE_DIST) {
        error = clone_state_dist_node(destination, source->as.state_dist,
                                      &out->as.state_dist, memo);
        if (error != LANA_OK) return error;
    } else if (source->type == VAL_MAP) {
        LanaMap *map;
        LanaContainerCloneMemo *entry;
        for (entry = *containers; entry != NULL; entry = entry->next)
            if (entry->type == VAL_MAP && entry->source == source->as.map) {
                out->as.map = entry->copy; return LANA_OK;
            }
        error = lana_map_new(destination, source->as.map->count, &map);
        if (error != LANA_OK) return error;
        entry = malloc(sizeof(*entry));
        if (entry == NULL) return LANA_ERR_OOM;
        entry->source = source->as.map; entry->copy = map; entry->type = VAL_MAP;
        entry->next = *containers; *containers = entry;
        out->as.map = map;
        for (index = 0; index < source->as.map->count; ++index) {
            Value cloned;
            error = clone_value_memo(destination, source->as.map->entries[index].value,
                                     &cloned, memo, containers, derivations);
            if (error != LANA_OK) return error;
            error = lana_map_set(destination, map, source->as.map->entries[index].key,
                               &cloned, true);
            if (error != LANA_OK) return error;
        }
    } else if (source->type == VAL_POSSIBILITY) {
        LanaPossibility *possibility;
        LanaContainerCloneMemo *entry;
        for (entry = *containers; entry != NULL; entry = entry->next)
            if (entry->type == VAL_POSSIBILITY &&
                entry->source == source->as.possibility) {
                out->as.possibility = entry->copy; return LANA_OK;
            }
        possibility = lana_vm_alloc(destination, sizeof(*possibility));
        if (possibility == NULL) return LANA_ERR_OOM;
        entry = malloc(sizeof(*entry));
        if (entry == NULL) return LANA_ERR_OOM;
        entry->source = source->as.possibility; entry->copy = possibility;
        entry->type = VAL_POSSIBILITY; entry->next = *containers; *containers = entry;
        possibility->count = source->as.possibility->count;
        possibility->dependency_id = source->as.possibility->dependency_id;
        possibility->values = lana_vm_alloc(destination,
            possibility->count * sizeof(*possibility->values));
        possibility->weights = source->as.possibility->weights == NULL ? NULL :
            lana_vm_alloc(destination, possibility->count * sizeof(*possibility->weights));
        if (possibility->values == NULL ||
            (source->as.possibility->weights != NULL && possibility->weights == NULL))
            return LANA_ERR_OOM;
        for (index = 0; index < possibility->count; ++index) {
            error = clone_value_memo(destination, &source->as.possibility->values[index],
                                     &possibility->values[index], memo, containers, derivations);
            if (error != LANA_OK) return error;
            if (possibility->weights != NULL)
                possibility->weights[index] = source->as.possibility->weights[index];
        }
        out->as.possibility = possibility;
    } else if (source->type == VAL_PATH_SET) {
        LanaPathSet *paths;
        LanaContainerCloneMemo *entry;
        for (entry = *containers; entry != NULL; entry = entry->next)
            if (entry->type == VAL_PATH_SET && entry->source == source->as.paths) {
                out->as.paths = entry->copy; return LANA_OK;
            }
        paths = lana_vm_alloc(destination, sizeof(*paths));
        if (paths == NULL) return LANA_ERR_OOM;
        entry = malloc(sizeof(*entry));
        if (entry == NULL) return LANA_ERR_OOM;
        entry->source = source->as.paths; entry->copy = paths;
        entry->type = VAL_PATH_SET; entry->next = *containers; *containers = entry;
        paths->count = source->as.paths->count;
        paths->dependency_id = source->as.paths->dependency_id;
        paths->alternatives = lana_vm_alloc(destination,
            paths->count * sizeof(*paths->alternatives));
        if (paths->alternatives == NULL) return LANA_ERR_OOM;
        for (index = 0; index < paths->count; ++index) {
            paths->alternatives[index].guard =
                source->as.paths->alternatives[index].guard;
            paths->alternatives[index].weight =
                source->as.paths->alternatives[index].weight;
            paths->alternatives[index].result = lana_vm_alloc(destination, sizeof(Value));
            if (paths->alternatives[index].result == NULL) return LANA_ERR_OOM;
            error = clone_value_memo(destination,
                source->as.paths->alternatives[index].result,
                paths->alternatives[index].result, memo, containers, derivations);
            if (error != LANA_OK) return error;
        }
        out->as.paths = paths;
    } else if (source->type == VAL_TASK) {
        return LANA_ERR_TYPE;
    } else if (source->type == VAL_SHARED_CAPABILITY) {
        LanaSharedInformation *shared = lana_shared_capability_information(
            source->as.capability);
        if (shared == NULL) return LANA_ERR_CAPABILITY;
        error = vm_track_shared(destination, shared, true);
        if (error != LANA_OK) return error;
    }
    return LANA_OK;
}

static LanaError clone_value(LanaVM *destination, const Value *source, Value *out) {
    LanaDistCloneMemo *memo = NULL;
    LanaContainerCloneMemo *containers = NULL;
    LanaDerivationCloneMemo *derivations = NULL;
    LanaDistCloneMemo *entry;
    LanaError error = clone_value_memo(destination, source, out, &memo, &containers,
                                       &derivations);
    while (memo != NULL) {
        entry = memo;
        memo = memo->next;
        free(entry);
    }
    while (containers != NULL) {
        LanaContainerCloneMemo *entry = containers;
        containers = containers->next;
        free(entry);
    }
    while (derivations != NULL) {
        LanaDerivationCloneMemo *entry = derivations;
        derivations = derivations->next;
        free(entry);
    }
    return error;
}

typedef struct {
    char *name;
    size_t source_index;
} LanaJointName;

static int joint_name_compare(const void *left, const void *right) {
    const LanaJointName *a = left;
    const LanaJointName *b = right;
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
        case VAL_SHARED_CAPABILITY:
            return left->as.capability == right->as.capability;
        default: return false;
    }
}

typedef struct LanaReactiveCloneMemo {
    const LanaReactive *source;
    LanaReactive *copy;
    struct LanaReactiveCloneMemo *next;
} LanaReactiveCloneMemo;

static LanaError clone_live_reactive_node(LanaVM *destination,
                                          const LanaReactive *source,
                                          LanaReactive **out,
                                          LanaReactiveCloneMemo **memo) {
    LanaReactiveCloneMemo *entry;
    LanaReactive *copy;
    size_t index;
    LanaError error;
    if (source == NULL) {
        *out = NULL;
        return LANA_OK;
    }
    for (entry = *memo; entry != NULL; entry = entry->next) {
        if (entry->source == source) {
            *out = entry->copy;
            return LANA_OK;
        }
    }
    copy = lana_vm_alloc(destination, sizeof(*copy));
    entry = malloc(sizeof(*entry));
    if (copy == NULL || entry == NULL) {
        free(entry);
        return LANA_ERR_OOM;
    }
    entry->source = source;
    entry->copy = copy;
    entry->next = *memo;
    *memo = entry;
    *copy = *source;
    copy->inputs[0] = NULL;
    copy->inputs[1] = NULL;
    copy->constants[0] = NULL;
    copy->constants[1] = NULL;
    copy->current = NULL;
    copy->history = NULL;
    error = clone_live_reactive_node(destination, source->inputs[0],
                                     &copy->inputs[0], memo);
    if (error == LANA_OK)
        error = clone_live_reactive_node(destination, source->inputs[1],
                                         &copy->inputs[1], memo);
    if (error != LANA_OK) return error;
    for (index = 0u; index < 2u; ++index) {
        if (source->constants[index] == NULL) continue;
        copy->constants[index] = lana_vm_alloc(
            destination, sizeof(*copy->constants[index]));
        if (copy->constants[index] == NULL) return LANA_ERR_OOM;
        error = clone_value(destination, source->constants[index],
                            copy->constants[index]);
        if (error != LANA_OK) return error;
    }
    if (source->current != NULL) {
        copy->current = lana_vm_alloc(destination, sizeof(*copy->current));
        if (copy->current == NULL) return LANA_ERR_OOM;
        error = clone_value(destination, source->current, copy->current);
        if (error != LANA_OK) return error;
    }
    if (source->history_count > 0u) {
        copy->history = lana_vm_alloc(destination,
            source->history_count * sizeof(*copy->history));
        if (copy->history == NULL) return LANA_ERR_OOM;
        for (index = 0u; index < source->history_count; ++index) {
            copy->history[index].revision = source->history[index].revision;
            copy->history[index].value = NULL;
            if (source->history[index].value == NULL) continue;
            copy->history[index].value = lana_vm_alloc(
                destination, sizeof(*copy->history[index].value));
            if (copy->history[index].value == NULL) return LANA_ERR_OOM;
            error = clone_value(destination, source->history[index].value,
                                copy->history[index].value);
            if (error != LANA_OK) return error;
        }
    }
    *out = copy;
    return LANA_OK;
}

static LanaError attach_live_reactive_values(LanaVM *destination,
                                             const Value *source, Value *copy,
                                             LanaReactiveCloneMemo **memo) {
    const Value *contents = reactive_value(source);
    size_t index;
    LanaError error;
    if (source->reactive != NULL) {
        error = clone_live_reactive_node(destination, source->reactive,
                                         &copy->reactive, memo);
        if (error != LANA_OK) return error;
    }
    if (contents->type == VAL_ARRAY && contents->as.array != NULL &&
        copy->type == VAL_ARRAY && copy->as.array != NULL) {
        for (index = 0u; index < contents->as.array->count; ++index) {
            error = attach_live_reactive_values(destination,
                &contents->as.array->items[index],
                &copy->as.array->items[index], memo);
            if (error != LANA_OK) return error;
        }
    } else if (contents->type == VAL_MAP && contents->as.map != NULL &&
               copy->type == VAL_MAP && copy->as.map != NULL) {
        for (index = 0u; index < contents->as.map->count; ++index) {
            error = attach_live_reactive_values(destination,
                contents->as.map->entries[index].value,
                copy->as.map->entries[index].value, memo);
            if (error != LANA_OK) return error;
        }
    } else if (contents->type == VAL_POSSIBILITY &&
               contents->as.possibility != NULL &&
               copy->type == VAL_POSSIBILITY && copy->as.possibility != NULL) {
        for (index = 0u; index < contents->as.possibility->count; ++index) {
            error = attach_live_reactive_values(destination,
                &contents->as.possibility->values[index],
                &copy->as.possibility->values[index], memo);
            if (error != LANA_OK) return error;
        }
    } else if (contents->type == VAL_PATH_SET && contents->as.paths != NULL &&
               copy->type == VAL_PATH_SET && copy->as.paths != NULL) {
        for (index = 0u; index < contents->as.paths->count; ++index) {
            error = attach_live_reactive_values(destination,
                contents->as.paths->alternatives[index].result,
                copy->as.paths->alternatives[index].result, memo);
            if (error != LANA_OK) return error;
        }
    }
    return LANA_OK;
}

LanaError lana_vm_clone_value(LanaVM *destination, const Value *source,
                              Value *out) {
    if (destination == NULL || source == NULL || out == NULL)
        return LANA_ERR_FORMAT;
    return clone_value(destination, source, out);
}

LanaError lana_vm_clone_live_value(LanaVM *destination, const Value *source,
                                   Value *out) {
    LanaReactiveCloneMemo *memo = NULL;
    LanaError error;
    if (destination == NULL || source == NULL || out == NULL)
        return LANA_ERR_FORMAT;
    error = clone_value(destination, source, out);
    if (error == LANA_OK)
        error = attach_live_reactive_values(destination, source, out, &memo);
    while (memo != NULL) {
        LanaReactiveCloneMemo *entry = memo;
        memo = memo->next;
        free(entry);
    }
    return error;
}

static bool value_equal_deep(const Value *left, const Value *right,
                             size_t depth) {
    size_t index;
    if (left == right) return true;
    if (left == NULL || right == NULL || depth > 256u ||
        left->type != right->type) return false;
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
        case VAL_DISTRIBUTION:
            return left->as.distribution.p0 == right->as.distribution.p0 &&
                   left->as.distribution.p1 == right->as.distribution.p1;
        case VAL_ARRAY:
            if (left->as.array->count != right->as.array->count) return false;
            for (index = 0u; index < left->as.array->count; ++index)
                if (!value_equal_deep(&left->as.array->items[index],
                                      &right->as.array->items[index], depth + 1u))
                    return false;
            return true;
        case VAL_MAP:
            if (left->as.map->count != right->as.map->count) return false;
            for (index = 0u; index < left->as.map->count; ++index) {
                ssize_t found = lana_map_has(right->as.map,
                    left->as.map->entries[index].key);
                if (found < 0 || !value_equal_deep(
                        left->as.map->entries[index].value,
                        right->as.map->entries[(size_t)found].value,
                        depth + 1u)) return false;
            }
            return true;
        case VAL_POSSIBILITY:
            if (left->as.possibility->count != right->as.possibility->count)
                return false;
            for (index = 0u; index < left->as.possibility->count; ++index)
                if (!value_equal_deep(&left->as.possibility->values[index],
                        &right->as.possibility->values[index], depth + 1u))
                    return false;
            return true;
        case VAL_PATH_SET:
            if (left->as.paths->count != right->as.paths->count) return false;
            for (index = 0u; index < left->as.paths->count; ++index)
                if (left->as.paths->alternatives[index].guard !=
                        right->as.paths->alternatives[index].guard ||
                    !value_equal_deep(left->as.paths->alternatives[index].result,
                        right->as.paths->alternatives[index].result, depth + 1u))
                    return false;
            return true;
        default:
            return joint_value_equal(left, right);
    }
}

bool lana_vm_value_equal(const Value *left, const Value *right) {
    return value_equal_deep(reactive_value(left), reactive_value(right), 0u);
}

LanaError lana_vm_possibility_build(LanaVM *vm, const Value *values, size_t count,
                                LanaPossibility **out) {
    LanaPossibility *possibility;
    size_t index, unique_count = 0u;
    Value *unique;
    LanaError error;
    if (vm == NULL || values == NULL || out == NULL || count == 0u)
        return LANA_ERR_FORMAT;
    unique = calloc(count, sizeof(*unique));
    if (unique == NULL) return LANA_ERR_OOM;
    for (index = 0; index < count; ++index) {
        size_t existing;
        if (!joint_value_is_definite(&values[index])) {
            free(unique); return LANA_ERR_TYPE;
        }
        for (existing = 0; existing < unique_count; ++existing)
            if (joint_value_equal(&values[index], &unique[existing])) break;
        if (existing == unique_count) unique[unique_count++] = values[index];
    }
    possibility = lana_vm_alloc(vm, sizeof(*possibility));
    if (possibility == NULL) { free(unique); return LANA_ERR_OOM; }
    possibility->count = unique_count;
    possibility->weights = NULL;
    possibility->dependency_id = vm->next_dependency_id++;
    possibility->values = lana_vm_alloc(vm, unique_count * sizeof(*possibility->values));
    if (possibility->values == NULL) { free(unique); return LANA_ERR_OOM; }
    for (index = 0; index < unique_count; ++index) {
        error = clone_value(vm, &unique[index], &possibility->values[index]);
        if (error != LANA_OK) { free(unique); return error; }
    }
    free(unique); *out = possibility; return LANA_OK;
}

static LanaError snapshot_frames(LanaVM *vm, LanaFrame **out) {
    LanaFrame *frames;
    size_t frame_index, register_index, history_index;
    LanaError error;
    frames = lana_vm_alloc(vm, vm->frame_count * sizeof(*frames));
    if (frames == NULL) return LANA_ERR_OOM;
    memcpy(frames, vm->frames, vm->frame_count * sizeof(*frames));
    for (frame_index = 0; frame_index < vm->frame_count; ++frame_index) {
        for (register_index = 0; register_index < LANA_MAX_REGISTERS; ++register_index) {
            error = clone_value(vm, &vm->frames[frame_index].registers[register_index],
                                &frames[frame_index].registers[register_index]);
            if (error != LANA_OK) return error;
        }
        for (history_index = 0; history_index < LANA_MAX_REGISTERS; ++history_index) {
            LanaHistory *history = &frames[frame_index].histories[history_index];
            const LanaHistory *source = &vm->frames[frame_index].histories[history_index];
            size_t version;
            if (source->count == 0u) { history->versions = NULL; continue; }
            history->versions = lana_vm_alloc(vm, source->count * sizeof(*history->versions));
            if (history->versions == NULL) return LANA_ERR_OOM;
            history->capacity = source->count;
            for (version = 0; version < source->count; ++version) {
                error = clone_state_value(vm, &source->versions[version],
                                          &history->versions[version]);
                if (error != LANA_OK) return error;
            }
        }
    }
    *out = frames; return LANA_OK;
}

static LanaError path_split(LanaVM *vm, const Value *condition, size_t false_ip) {
    const LanaPossibility *possibility;
    bool has_true = false, has_false = false;
    double true_weight = 0.0, false_weight = 0.0;
    size_t index;
    LanaPathExecution *execution;
    LanaError error;
    if (condition->type == VAL_BOOL) {
        if (!condition->as.boolean) vm->ip = false_ip;
        return LANA_OK;
    }
    if (condition->type != VAL_POSSIBILITY) return LANA_ERR_TYPE;
    possibility = condition->as.possibility;
    for (index = 0; index < possibility->count; ++index) {
        double weight = possibility->weights == NULL
            ? 1.0 / (double)possibility->count : possibility->weights[index];
        if (possibility->values[index].type != VAL_BOOL) return LANA_ERR_TYPE;
        if (possibility->values[index].as.boolean) {
            has_true = true; true_weight += weight;
        } else {
            has_false = true; false_weight += weight;
        }
    }
    if (!has_true) { vm->ip = false_ip; return LANA_OK; }
    if (!has_false) return LANA_OK;
    if (vm->active_path_count > vm->path_limit / 2u) return LANA_ERR_PATH_LIMIT;
    execution = lana_vm_alloc(vm, sizeof(*execution));
    if (execution == NULL) return LANA_ERR_OOM;
    error = snapshot_frames(vm, &execution->false_frames);
    if (error != LANA_OK) return error;
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
    return LANA_OK;
}

static LanaError path_join(LanaVM *vm, uint32_t line) {
    LanaPathExecution *execution = vm->path_execution;
    size_t frame_index, register_index, index;
    LanaError error;
    if (execution == NULL) return LANA_OK;
    if (!execution->running_false) {
        error = snapshot_frames(vm, &execution->true_frames);
        if (error != LANA_OK) return error;
        vm->frame_count = execution->frame_count;
        memcpy(vm->frames, execution->false_frames,
               execution->frame_count * sizeof(*vm->frames));
        execution->running_false = true;
        vm->ip = execution->false_ip;
        return LANA_OK;
    }
    if (vm->frame_count != execution->frame_count) return LANA_ERR_UNSUPPORTED_OPERATION;
    for (frame_index = 0; frame_index < vm->frame_count; ++frame_index) {
        for (register_index = 0; register_index < LANA_MAX_REGISTERS; ++register_index) {
            const Value *true_value =
                &execution->true_frames[frame_index].registers[register_index];
            Value *false_value = &vm->frames[frame_index].registers[register_index];
            LanaPathSet *paths;
            if (joint_value_equal(true_value, false_value)) continue;
            if (execution->true_frames[frame_index].histories[register_index].policy !=
                    LANA_HISTORY_NONE ||
                vm->frames[frame_index].histories[register_index].policy != LANA_HISTORY_NONE)
                return LANA_ERR_UNSUPPORTED_OPERATION;
            paths = lana_vm_alloc(vm, sizeof(*paths));
            if (paths == NULL) return LANA_ERR_OOM;
            paths->count = 2u;
            paths->dependency_id = execution->dependency_id;
            paths->alternatives = lana_vm_alloc(vm, 2u * sizeof(*paths->alternatives));
            if (paths->alternatives == NULL) return LANA_ERR_OOM;
            for (index = 0; index < 2u; ++index) {
                paths->alternatives[index].result = lana_vm_alloc(vm, sizeof(Value));
                if (paths->alternatives[index].result == NULL) return LANA_ERR_OOM;
            }
            paths->alternatives[0].guard = true;
            paths->alternatives[0].weight = execution->true_weight;
            error = clone_value(vm, true_value, paths->alternatives[0].result);
            if (error != LANA_OK) return error;
            paths->alternatives[1].guard = false;
            paths->alternatives[1].weight = execution->false_weight;
            error = clone_value(vm, false_value, paths->alternatives[1].result);
            if (error != LANA_OK) return error;
            {
                Value false_input = *false_value;
                const Value *inputs[] = {true_value, &false_input};
                *false_value = lana_value_paths(paths);
                error = attach_derivation(vm, false_value, LANA_DERIVATION_PATH,
                    "guarded_path", inputs, 2u, "", line,
                    LANA_EXACTNESS_EXACT, "true_false_alternatives");
                if (error != LANA_OK) return error;
            }
        }
    }
    vm->active_path_count = execution->previous_path_count;
    vm->path_execution = execution->next;
    return LANA_OK;
}

static LanaError parse_joint_names(const char *text, size_t expected,
                                 LanaJointKind *kind, char ***names_out,
                                 size_t *count_out) {
    char *copy, *cursor, *token;
    char **names;
    size_t count = 0u;
    if (text == NULL || *text == '\0') return LANA_ERR_FORMAT;
    copy = malloc(strlen(text) + 1u);
    if (copy == NULL) return LANA_ERR_OOM;
    strcpy(copy, text);
    cursor = strchr(copy, ':');
    if (cursor == NULL) { free(copy); return LANA_ERR_FORMAT; }
    *cursor++ = '\0';
    if (strcmp(copy, "independent") == 0) *kind = LANA_JOINT_INDEPENDENT;
    else if (strcmp(copy, "correlated") == 0) *kind = LANA_JOINT_FINITE_LAW;
    else if (strcmp(copy, "conditional") == 0) *kind = LANA_JOINT_CONDITIONAL;
    else { free(copy); return LANA_ERR_FORMAT; }
    names = calloc(expected == 0u ? 1u : expected, sizeof(*names));
    if (names == NULL) { free(copy); return LANA_ERR_OOM; }
    token = strtok(cursor, ",;");
    while (token != NULL) {
        size_t index;
        while (isspace((unsigned char)*token)) ++token;
        if (*token == '\0') { free(names); free(copy); return LANA_ERR_FORMAT; }
        for (index = 0; index < count; ++index)
            if (strcmp(names[index], token) == 0) { free(names); free(copy); return LANA_ERR_INVALID_DEPENDENCY; }
        if (count >= expected) { free(names); free(copy); return LANA_ERR_FORMAT; }
        names[count] = malloc(strlen(token) + 1u);
        if (names[count] == NULL) { while (count > 0u) free(names[--count]); free(names); free(copy); return LANA_ERR_OOM; }
        strcpy(names[count++], token);
        token = strtok(NULL, ",;");
    }
    free(copy);
    if (count != expected) { while (count > 0u) free(names[--count]); free(names); return LANA_ERR_FORMAT; }
    *names_out = names;
    *count_out = count;
    return LANA_OK;
}

static void free_joint_names(char **names, size_t count) {
    size_t index;
    for (index = 0; index < count; ++index) free(names[index]);
    free(names);
}

LanaError lana_vm_joint_build(LanaVM *vm, const Value *values, size_t count,
                          const char *descriptor, LanaJointState **out) {
    LanaJointKind kind;
    char **names = NULL;
    size_t name_count, index;
    LanaJointName *ordered;
    LanaJointState *joint;
    LanaError error;
    if (vm == NULL || values == NULL || out == NULL || count == 0u) return LANA_ERR_FORMAT;
    error = parse_joint_names(descriptor, count, &kind, &names, &name_count);
    if (error != LANA_OK) return error;
    /* A correlation label plus unrelated marginals is not a joint law. */
    if (kind == LANA_JOINT_FINITE_LAW) {
        free_joint_names(names, name_count);
        return LANA_ERR_UNSUPPORTED_OPERATION;
    }
    ordered = calloc(count, sizeof(*ordered));
    if (ordered == NULL) { free_joint_names(names, name_count); return LANA_ERR_OOM; }
    for (index = 0; index < count; ++index) { ordered[index].name = names[index]; ordered[index].source_index = index; }
    qsort(ordered, count, sizeof(*ordered), joint_name_compare);
    joint = lana_vm_alloc(vm, sizeof(*joint));
    if (joint == NULL) { free(ordered); free_joint_names(names, name_count); return LANA_ERR_OOM; }
    joint->count = count; joint->kind = kind;
    joint->capabilities = kind == LANA_JOINT_INDEPENDENT
        ? (LANA_JOINT_CAN_PROJECT | LANA_JOINT_CAN_CONDITION |
           LANA_JOINT_CAN_SAMPLE | LANA_JOINT_CAN_RESOLVE)
        : 0u;
    joint->row_count = 0u; joint->rows = NULL;
    joint->names = lana_vm_alloc(vm, count * sizeof(*joint->names));
    joint->domains = lana_vm_alloc(vm, count * sizeof(*joint->domains));
    joint->values = lana_vm_alloc(vm, count * sizeof(*joint->values));
    if (joint->names == NULL || joint->domains == NULL || joint->values == NULL) { free(ordered); free_joint_names(names, name_count); return LANA_ERR_OOM; }
    for (index = 0; index < count; ++index) {
        size_t length = strlen(ordered[index].name);
        joint->names[index] = lana_vm_alloc(vm, length + 1u);
        if (joint->names[index] == NULL) { free(ordered); free_joint_names(names, name_count); return LANA_ERR_OOM; }
        memcpy(joint->names[index], ordered[index].name, length + 1u);
        error = clone_value(vm, &values[ordered[index].source_index], &joint->values[index]);
        if (error != LANA_OK) { free(ordered); free_joint_names(names, name_count); return error; }
        joint->domains[index].type = joint->values[index].type;
    }
    free(ordered); free_joint_names(names, name_count); *out = joint; return LANA_OK;
}

static bool joint_value_is_definite(const Value *value) {
    return value != NULL && value->type != VAL_STATE_DIST &&
           value->type != VAL_JOINT_STATE && value->type != VAL_TASK &&
           value->type != VAL_FUNCTION;
}

LanaError lana_vm_joint_build_finite(LanaVM *vm, const char *names_text,
                                 const Value *rows, const double *weights,
                                 size_t row_count, size_t variable_count,
                                 LanaJointState **out) {
    char *descriptor;
    char **names = NULL;
    size_t name_count = 0u, row, column, unique_count = 0u;
    LanaJointKind parsed_kind;
    LanaJointName *ordered = NULL;
    Value *unique_values = NULL;
    double *unique_weights = NULL;
    double total = 0.0;
    LanaJointState *joint;
    LanaError error = LANA_OK;
    if (vm == NULL || names_text == NULL || rows == NULL || weights == NULL ||
        out == NULL || row_count == 0u || variable_count == 0u)
        return LANA_ERR_FORMAT;
    descriptor = malloc(strlen(names_text) + sizeof("correlated:"));
    if (descriptor == NULL) return LANA_ERR_OOM;
    (void)snprintf(descriptor, strlen(names_text) + sizeof("correlated:"),
                   "correlated:%s", names_text);
    error = parse_joint_names(descriptor, variable_count, &parsed_kind,
                              &names, &name_count);
    free(descriptor);
    if (error != LANA_OK) return error;
    ordered = calloc(variable_count, sizeof(*ordered));
    unique_values = calloc(row_count * variable_count, sizeof(*unique_values));
    unique_weights = calloc(row_count, sizeof(*unique_weights));
    if (ordered == NULL || unique_values == NULL || unique_weights == NULL) {
        error = LANA_ERR_OOM; goto cleanup;
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
            error = LANA_ERR_INVALID_DISTRIBUTION; goto cleanup;
        }
        total += weights[row];
        for (column = 0; column < variable_count; ++column) {
            const Value *value = &rows[row * variable_count + ordered[column].source_index];
            if (!joint_value_is_definite(value)) { error = LANA_ERR_TYPE; goto cleanup; }
            if (row > 0u && value->type !=
                rows[ordered[column].source_index].type) {
                error = LANA_ERR_TYPE; goto cleanup;
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
        error = LANA_ERR_INVALID_DISTRIBUTION; goto cleanup;
    }
    joint = lana_vm_alloc(vm, sizeof(*joint));
    if (joint == NULL) { error = LANA_ERR_OOM; goto cleanup; }
    joint->count = variable_count;
    joint->kind = LANA_JOINT_FINITE_LAW;
    joint->capabilities = LANA_JOINT_CAN_PROJECT | LANA_JOINT_CAN_CONDITION |
        LANA_JOINT_CAN_SAMPLE | LANA_JOINT_CAN_RESOLVE;
    joint->values = NULL;
    joint->row_count = unique_count;
    joint->names = lana_vm_alloc(vm, variable_count * sizeof(*joint->names));
    joint->domains = lana_vm_alloc(vm, variable_count * sizeof(*joint->domains));
    joint->rows = lana_vm_alloc(vm, unique_count * sizeof(*joint->rows));
    if (joint->names == NULL || joint->domains == NULL || joint->rows == NULL) {
        error = LANA_ERR_OOM; goto cleanup;
    }
    for (column = 0; column < variable_count; ++column) {
        size_t length = strlen(ordered[column].name);
        joint->names[column] = lana_vm_alloc(vm, length + 1u);
        if (joint->names[column] == NULL) { error = LANA_ERR_OOM; goto cleanup; }
        memcpy(joint->names[column], ordered[column].name, length + 1u);
        joint->domains[column].type = unique_values[column].type;
    }
    for (row = 0; row < unique_count; ++row) {
        joint->rows[row].weight = unique_weights[row] / total;
        joint->rows[row].values = lana_vm_alloc(
            vm, variable_count * sizeof(*joint->rows[row].values));
        if (joint->rows[row].values == NULL) { error = LANA_ERR_OOM; goto cleanup; }
        for (column = 0; column < variable_count; ++column) {
            error = clone_value(vm, &unique_values[row * variable_count + column],
                                &joint->rows[row].values[column]);
            if (error != LANA_OK) goto cleanup;
        }
    }
    *out = joint;
cleanup:
    free(ordered); free(unique_values); free(unique_weights);
    free_joint_names(names, name_count);
    return error;
}

static LanaError joint_build_finite_array(LanaVM *vm, const Value *rows_value,
                                        const char *names_text,
                                        LanaJointState **out) {
    const LanaArray *outer;
    size_t row, column, variable_count;
    Value *values;
    double *weights;
    LanaError error;
    if (rows_value == NULL || rows_value->type != VAL_ARRAY ||
        rows_value->as.array == NULL || rows_value->as.array->count == 0u)
        return LANA_ERR_TYPE;
    outer = rows_value->as.array;
    if (outer->items[0].type != VAL_ARRAY || outer->items[0].as.array == NULL ||
        outer->items[0].as.array->count < 2u) return LANA_ERR_FORMAT;
    variable_count = outer->items[0].as.array->count - 1u;
    values = calloc(outer->count * variable_count, sizeof(*values));
    weights = calloc(outer->count, sizeof(*weights));
    if (values == NULL || weights == NULL) {
        free(values); free(weights); return LANA_ERR_OOM;
    }
    for (row = 0; row < outer->count; ++row) {
        const LanaArray *inner;
        if (outer->items[row].type != VAL_ARRAY ||
            outer->items[row].as.array == NULL) { error = LANA_ERR_TYPE; goto done; }
        inner = outer->items[row].as.array;
        if (inner->count != variable_count + 1u) { error = LANA_ERR_FORMAT; goto done; }
        if (inner->items[variable_count].type != VAL_NUMBER) {
            error = LANA_ERR_TYPE; goto done;
        }
        weights[row] = inner->items[variable_count].as.number;
        for (column = 0; column < variable_count; ++column)
            values[row * variable_count + column] = inner->items[column];
    }
    error = lana_vm_joint_build_finite(vm, names_text, values, weights,
                                     outer->count, variable_count, out);
done:
    free(values); free(weights);
    return error;
}

static ssize_t joint_find(const LanaJointState *joint, const char *name) {
    size_t index;
    if (joint == NULL || name == NULL) return -1;
    for (index = 0; index < joint->count; ++index)
        if (strcmp(joint->names[index], name) == 0) return (ssize_t)index;
    return -1;
}

LanaError lana_vm_joint_project(LanaVM *vm, const LanaJointState *source,
                            const char *names_text, LanaJointState **out) {
    char *copy, *token;
    size_t count = 0u, index;
    ssize_t *positions = NULL;
    Value *values = NULL;
    char descriptor[1024];
    LanaError error;
    if (vm == NULL || source == NULL || out == NULL || names_text == NULL) return LANA_ERR_FORMAT;
    if ((source->capabilities & LANA_JOINT_CAN_PROJECT) == 0u)
        return LANA_ERR_UNSUPPORTED_OPERATION;
    copy = malloc(strlen(names_text) + 1u); if (copy == NULL) return LANA_ERR_OOM;
    strcpy(copy, names_text); token = strtok(copy, ",;");
    while (token != NULL) {
        if (joint_find(source, token) < 0) { free(copy); return LANA_ERR_KEY; }
        ++count; token = strtok(NULL, ",;");
    }
    free(copy); if (count == 0u) return LANA_ERR_FORMAT;
    positions = calloc(count, sizeof(*positions));
    if (positions == NULL) return LANA_ERR_OOM;
    copy = malloc(strlen(names_text) + 1u); if (copy == NULL) { free(positions); return LANA_ERR_OOM; }
    strcpy(copy, names_text); token = strtok(copy, ",;");
    for (index = 0; token != NULL; ++index, token = strtok(NULL, ",;")) {
        size_t previous;
        positions[index] = joint_find(source, token);
        for (previous = 0; previous < index; ++previous)
            if (positions[previous] == positions[index]) {
                free(copy); free(positions); return LANA_ERR_INVALID_DEPENDENCY;
            }
    }
    free(copy);
    if (strlen(names_text) + sizeof("independent:") >= sizeof(descriptor)) {
        free(positions); return LANA_ERR_LIMIT;
    }
    if (source->rows != NULL) {
        size_t row, column;
        double *weights = calloc(source->row_count, sizeof(*weights));
        values = calloc(source->row_count * count, sizeof(*values));
        if (weights == NULL || values == NULL) {
            free(weights); free(values); free(positions); return LANA_ERR_OOM;
        }
        for (row = 0; row < source->row_count; ++row) {
            weights[row] = source->rows[row].weight;
            for (column = 0; column < count; ++column)
                values[row * count + column] =
                    source->rows[row].values[positions[column]];
        }
        error = lana_vm_joint_build_finite(vm, names_text, values, weights,
                                         source->row_count, count, out);
        free(weights); free(values);
    } else {
        values = calloc(count, sizeof(*values));
        if (values == NULL) { free(positions); return LANA_ERR_OOM; }
        for (index = 0; index < count; ++index)
            values[index] = source->values[positions[index]];
        (void)snprintf(descriptor, sizeof(descriptor), "independent:%s", names_text);
        error = lana_vm_joint_build(vm, values, count, descriptor, out);
        free(values);
    }
    free(positions);
    if (error == LANA_OK) (*out)->kind = LANA_JOINT_PROJECTED;
    return error;
}

LanaError lana_vm_joint_condition(LanaVM *vm, const LanaJointState *source,
                              const char *name, const Value *evidence,
                              LanaJointState **out) {
    ssize_t position = joint_find(source, name);
    size_t index;
    if (vm == NULL || source == NULL || evidence == NULL || out == NULL) return LANA_ERR_FORMAT;
    if ((source->capabilities & LANA_JOINT_CAN_CONDITION) == 0u)
        return LANA_ERR_UNSUPPORTED_OPERATION;
    if (position < 0) return LANA_ERR_KEY;
    if (source->rows != NULL) {
        size_t row, kept = 0u, names_length = 1u;
        Value *rows;
        double *weights;
        char *names_text;
        LanaError error;
        for (row = 0; row < source->row_count; ++row)
            if (joint_value_equal(&source->rows[row].values[position], evidence)) ++kept;
        if (kept == 0u) return LANA_ERR_INVALID_CONDITIONING;
        rows = calloc(kept * source->count, sizeof(*rows));
        weights = calloc(kept, sizeof(*weights));
        for (index = 0; index < source->count; ++index)
            names_length += strlen(source->names[index]) + 1u;
        names_text = malloc(names_length);
        if (rows == NULL || weights == NULL || names_text == NULL) {
            free(rows); free(weights); free(names_text); return LANA_ERR_OOM;
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
        error = lana_vm_joint_build_finite(vm, names_text, rows, weights,
                                         kept, source->count, out);
        free(rows); free(weights); free(names_text);
        if (error == LANA_OK) (*out)->kind = LANA_JOINT_CONDITIONAL;
        return error;
    }
    if (!joint_value_is_definite(&source->values[position]))
        return LANA_ERR_UNSUPPORTED_OPERATION;
    if (!joint_value_equal(&source->values[position], evidence))
        return LANA_ERR_INVALID_CONDITIONING;
    {
        Value wrapped = {.type = VAL_JOINT_STATE, .as.joint = (LanaJointState *)source};
        Value cloned;
        LanaError error = clone_value(vm, &wrapped, &cloned);
        if (error != LANA_OK) return error;
        cloned.as.joint->kind = LANA_JOINT_CONDITIONAL;
        *out = cloned.as.joint;
    }
    return LANA_OK;
}

LanaError lana_vm_joint_observe(LanaVM *vm, const LanaJointState *source,
                            const char *name, const Value *evidence,
                            LanaJointState **out) {
    LanaError error;
    if (vm == NULL) return LANA_ERR_FORMAT;
    if (vm->active_path_count > 1u) return LANA_ERR_UNSUPPORTED_OPERATION;
    error = lana_vm_joint_condition(vm, source, name, evidence, out);
    if (error == LANA_OK) {
        ++vm->observation_count;
        ++vm->revision;
    }
    return error;
}

LanaError lana_vm_joint_sample(LanaVM *vm, const LanaJointState *source, Value *out) {
    LanaArray *array; size_t index; LanaError error;
    if (vm == NULL || source == NULL || out == NULL) return LANA_ERR_FORMAT;
    if ((source->capabilities & LANA_JOINT_CAN_SAMPLE) == 0u)
        return LANA_ERR_UNSUPPORTED_OPERATION;
    array = lana_vm_alloc(vm, sizeof(*array)); if (array == NULL) return LANA_ERR_OOM;
    array->count = source->count; array->capacity = array->count; array->items = lana_vm_alloc(vm, array->count * sizeof(*array->items));
    if (array->items == NULL && array->count > 0u) return LANA_ERR_OOM;
    if (source->rows != NULL) {
        double draw, cumulative = 0.0;
        size_t selected = source->row_count - 1u;
        error = consume_sampling_budget(vm);
        if (error != LANA_OK) return error;
        draw = (double)lana_vm_random(vm) / 4294967296.0;
        for (index = 0; index < source->row_count; ++index) {
            cumulative += source->rows[index].weight;
            if (draw < cumulative) { selected = index; break; }
        }
        for (index = 0; index < source->count; ++index) {
            error = clone_value(vm, &source->rows[selected].values[index],
                                &array->items[index]);
            if (error != LANA_OK) return error;
        }
        *out = lana_value_array(array); return LANA_OK;
    }
    for (index = 0; index < source->count; ++index) {
        if (source->values[index].type == VAL_STATE_DIST) {
            LanaStateValue state;
            error = lana_vm_state_dist_sample(vm, source->values[index].as.state_dist, &state);
            if (error != LANA_OK) return error;
            array->items[index] = (Value){.type = VAL_STATE, .as.state = state};
        } else { error = clone_value(vm, &source->values[index], &array->items[index]); if (error != LANA_OK) return error; }
    }
    *out = lana_value_array(array); return LANA_OK;
}

LanaError lana_vm_joint_resolve(LanaVM *vm, const LanaJointState *source, Value *out) {
    const Value *values;
    size_t index;
    LanaArray *array;
    LanaError error;
    if (vm == NULL || source == NULL || out == NULL) return LANA_ERR_FORMAT;
    if ((source->capabilities & LANA_JOINT_CAN_RESOLVE) == 0u)
        return LANA_ERR_UNSUPPORTED_OPERATION;
    if (source->rows != NULL) {
        if (source->row_count != 1u) return LANA_ERR_UNRESOLVED_VALUE;
        values = source->rows[0].values;
    } else {
        values = source->values;
        for (index = 0; index < source->count; ++index)
            if (!joint_value_is_definite(&values[index]))
                return LANA_ERR_UNRESOLVED_VALUE;
    }
    if (source->count == 1u) return clone_value(vm, &values[0], out);
    array = lana_vm_alloc(vm, sizeof(*array));
    if (array == NULL) return LANA_ERR_OOM;
    array->count = source->count;
    array->capacity = array->count;
    array->items = lana_vm_alloc(vm, array->count * sizeof(*array->items));
    if (array->items == NULL) return LANA_ERR_OOM;
    for (index = 0; index < array->count; ++index) {
        error = clone_value(vm, &values[index], &array->items[index]);
        if (error != LANA_OK) return error;
    }
    *out = lana_value_array(array);
    return LANA_OK;
}

LanaError lana_vm_information_resolve(LanaVM *vm, const Value *source, Value *out) {
    size_t index;
    if (vm == NULL || source == NULL || out == NULL) return LANA_ERR_FORMAT;
    source = reactive_value(source);
    if (source->type == VAL_JOINT_STATE)
        return lana_vm_joint_resolve(vm, source->as.joint, out);
    if (source->type == VAL_POSSIBILITY) {
        if (source->as.possibility->count != 1u) return LANA_ERR_UNRESOLVED_VALUE;
        return clone_value(vm, &source->as.possibility->values[0], out);
    }
    if (source->type == VAL_PATH_SET) {
        const LanaPathSet *paths = source->as.paths;
        if (paths->count == 0u) return LANA_ERR_UNRESOLVED_VALUE;
        for (index = 1u; index < paths->count; ++index)
            if (!joint_value_equal(paths->alternatives[0].result,
                                   paths->alternatives[index].result))
                return LANA_ERR_UNRESOLVED_VALUE;
        return clone_value(vm, paths->alternatives[0].result, out);
    }
    if (!joint_value_is_definite(source)) return LANA_ERR_UNRESOLVED_VALUE;
    return clone_value(vm, source, out);
}

LanaError lana_vm_information_sample(LanaVM *vm, const Value *source, Value *out) {
    size_t selected;
    LanaError error;
    if (vm == NULL || source == NULL || out == NULL) return LANA_ERR_FORMAT;
    source = reactive_value(source);
    if (source->type == VAL_JOINT_STATE)
        return lana_vm_joint_sample(vm, source->as.joint, out);
    if (source->type == VAL_STATE_DIST) {
        LanaStateValue state;
        error = lana_vm_state_dist_sample(vm, source->as.state_dist, &state);
        if (error == LANA_OK) *out = (Value){.type = VAL_STATE, .as.state = state};
        return error;
    }
    error = consume_sampling_budget(vm);
    if (error != LANA_OK) return error;
    if (source->type == VAL_POSSIBILITY) {
        selected = (size_t)(lana_vm_random(vm) % source->as.possibility->count);
        return clone_value(vm, &source->as.possibility->values[selected], out);
    }
    if (source->type == VAL_PATH_SET) {
        double draw = (double)lana_vm_random(vm) / 4294967296.0;
        double cumulative = 0.0;
        const LanaPathSet *paths = source->as.paths;
        selected = paths->count - 1u;
        for (size_t index = 0; index < paths->count; ++index) {
            cumulative += paths->alternatives[index].weight;
            if (draw < cumulative) { selected = index; break; }
        }
        return clone_value(vm, paths->alternatives[selected].result, out);
    }
    return LANA_ERR_TYPE;
}

static LanaError clone_without_runtime_metadata(LanaVM *vm, const Value *source,
                                                Value *out) {
    Value plain = *reactive_value(source);
    plain.reactive = NULL;
    plain.claim = NULL;
    plain.planned_effect = NULL;
    return clone_value(vm, &plain, out);
}

static LanaError allocate_plain_value(LanaVM *vm, const Value *source,
                                      Value **out) {
    Value *copy = lana_vm_alloc(vm, sizeof(*copy));
    LanaError error;
    if (copy == NULL) return LANA_ERR_OOM;
    error = clone_without_runtime_metadata(vm, source, copy);
    if (error != LANA_OK) return error;
    *out = copy;
    return LANA_OK;
}

static LanaError materialize_value(LanaVM *vm, const Value *source, Value *out) {
    const Value *current = reactive_value(source);
    size_t index;
    LanaError error;
    if (current->type == VAL_ARRAY && current->as.array != NULL) {
        LanaArray *array = lana_vm_alloc(vm, sizeof(*array));
        if (array == NULL) return LANA_ERR_OOM;
        array->count = current->as.array->count;
        array->capacity = array->count;
        array->items = lana_vm_alloc(vm, array->count * sizeof(*array->items));
        if (array->items == NULL && array->count > 0u) return LANA_ERR_OOM;
        for (index = 0u; index < array->count; ++index) {
            error = materialize_value(vm, &current->as.array->items[index],
                                      &array->items[index]);
            if (error != LANA_OK) return error;
        }
        *out = lana_value_array(array);
        return LANA_OK;
    }
    if (current->type == VAL_MAP && current->as.map != NULL) {
        LanaMap *map;
        error = lana_map_new(vm, current->as.map->count, &map);
        if (error != LANA_OK) return error;
        for (index = 0u; index < current->as.map->count; ++index) {
            Value item;
            error = materialize_value(vm, current->as.map->entries[index].value,
                                      &item);
            if (error != LANA_OK) return error;
            error = lana_map_set(vm, map, current->as.map->entries[index].key,
                                 &item, true);
            if (error != LANA_OK) return error;
        }
        *out = lana_value_map(map);
        return LANA_OK;
    }
    return clone_without_runtime_metadata(vm, current, out);
}

LanaError lana_vm_reactive_root(LanaVM *vm, const Value *source,
                                LanaDerivationExactness exactness, Value *out) {
    LanaReactive *node;
    LanaError error;
    if (vm == NULL || source == NULL || out == NULL || source->reactive != NULL)
        return LANA_ERR_FORMAT;
    node = lana_vm_alloc(vm, sizeof(*node));
    if (node == NULL) return LANA_ERR_OOM;
    memset(node, 0, sizeof(*node));
    node->id = vm->next_reactive_id++;
    node->kind = LANA_REACTIVE_ROOT;
    node->revision = vm->revision;
    node->exactness = exactness;
    node->relationship = LANA_RELATION_EXACT;
    if (source->type == VAL_POSSIBILITY && source->as.possibility != NULL)
        node->dependency_id = source->as.possibility->dependency_id;
    else if (source->type == VAL_PATH_SET && source->as.paths != NULL)
        node->dependency_id = source->as.paths->dependency_id;
    else
        node->dependency_id = vm->next_dependency_id++;
    error = allocate_plain_value(vm, source, &node->current);
    if (error != LANA_OK) return error;
    *out = *source;
    out->reactive = node;
    return LANA_OK;
}

LanaError lana_vm_reactive_observe(LanaVM *vm, const Value *source,
                                   const Value *evidence, Value *out) {
    LanaReactive *root;
    const Value *current;
    const Value *replacement = reactive_value(evidence);
    size_t index;
    bool supported = false;
    LanaError error;
    if (vm == NULL || source == NULL || evidence == NULL || out == NULL ||
        source->reactive == NULL || source->reactive->kind != LANA_REACTIVE_ROOT)
        return LANA_ERR_FORMAT;
    if (vm->active_path_count > 1u || value_is_unresolved(replacement))
        return LANA_ERR_UNRESOLVED_VALUE;
    root = source->reactive;
    current = root->current;
    if (current->type == VAL_POSSIBILITY && current->as.possibility != NULL) {
        for (index = 0u; index < current->as.possibility->count; ++index)
            if (joint_value_equal(&current->as.possibility->values[index],
                                  replacement)) {
                supported = true;
                break;
            }
        if (!supported) return LANA_ERR_INVALID_CONDITIONING;
    } else if (current->type == VAL_PATH_SET && current->as.paths != NULL) {
        for (index = 0u; index < current->as.paths->count; ++index)
            if (joint_value_equal(current->as.paths->alternatives[index].result,
                                  replacement)) {
                supported = true;
                break;
            }
        if (!supported) return LANA_ERR_INVALID_CONDITIONING;
    } else if (!joint_value_equal(current, replacement)) {
        return LANA_ERR_INVALID_CONDITIONING;
    }
    error = reactive_recompute_transaction(vm, root, replacement);
    if (error != LANA_OK) return error;
    ++vm->observation_count;
    *out = *source;
    return LANA_OK;
}

LanaError lana_vm_claim(LanaVM *vm, const Value *source, const char *proposition,
                        LanaDerivationExactness exactness, double tolerance,
                        bool source_valid, Value *out) {
    LanaClaim *claim;
    size_t length;
    char *label;
    LanaError error;
    if (vm == NULL || source == NULL || proposition == NULL || out == NULL ||
        tolerance < 0.0) return LANA_ERR_FORMAT;
    claim = lana_vm_alloc(vm, sizeof(*claim));
    if (claim == NULL) return LANA_ERR_OOM;
    memset(claim, 0, sizeof(*claim));
    claim->value = lana_vm_alloc(vm, sizeof(*claim->value));
    if (claim->value == NULL) return LANA_ERR_OOM;
    error = clone_value(vm, source, claim->value);
    if (error != LANA_OK) return error;
    length = strlen(proposition);
    label = lana_vm_alloc(vm, length + 1u);
    if (label == NULL) return LANA_ERR_OOM;
    memcpy(label, proposition, length + 1u);
    claim->proposition = label;
    claim->exactness = exactness;
    claim->tolerance = tolerance;
    claim->source_valid = source_valid;
    *out = *source;
    out->claim = claim;
    return LANA_OK;
}

LanaError lana_vm_planned_effect(LanaVM *vm, const char *kind,
                                 const Value *payload, Value *out) {
    LanaPlannedEffect *plan;
    size_t length;
    char *kind_copy;
    LanaError error;
    if (vm == NULL || kind == NULL || *kind == '\0' || payload == NULL || out == NULL)
        return LANA_ERR_FORMAT;
    plan = lana_vm_alloc(vm, sizeof(*plan));
    if (plan == NULL) return LANA_ERR_OOM;
    memset(plan, 0, sizeof(*plan));
    plan->id = vm->next_effect_id++;
    length = strlen(kind);
    kind_copy = lana_vm_alloc(vm, length + 1u);
    if (kind_copy == NULL) return LANA_ERR_OOM;
    memcpy(kind_copy, kind, length + 1u);
    plan->kind = kind_copy;
    error = allocate_plain_value(vm, payload, &plan->payload);
    if (error != LANA_OK) return error;
    *out = *payload;
    out->planned_effect = plan;
    return LANA_OK;
}

LanaError lana_vm_execute_planned_effect(LanaVM *vm, const Value *plan_value,
                                         LanaEffectExecutor executor,
                                         void *context, Value *out) {
    LanaPlannedEffect *plan;
    LanaEffectReceipt *receipt;
    Value result;
    LanaError error;
    if (vm == NULL || plan_value == NULL || out == NULL ||
        plan_value->planned_effect == NULL || executor == NULL)
        return LANA_ERR_FORMAT;
    plan = plan_value->planned_effect;
    for (receipt = plan->receipts; receipt != NULL; receipt = receipt->next)
        if (receipt->revision == vm->revision)
            return clone_value(vm, receipt->result, out);
    if (value_is_unresolved(plan->payload)) return LANA_ERR_UNRESOLVED_VALUE;
    error = executor(vm, plan->kind, reactive_value(plan->payload), context, &result);
    if (error != LANA_OK) return error;
    receipt = lana_vm_alloc(vm, sizeof(*receipt));
    if (receipt == NULL) return LANA_ERR_OOM;
    memset(receipt, 0, sizeof(*receipt));
    receipt->revision = vm->revision;
    error = allocate_plain_value(vm, &result, &receipt->result);
    if (error != LANA_OK) return error;
    receipt->next = plan->receipts;
    plan->receipts = receipt;
    ++plan->execution_count;
    return clone_value(vm, receipt->result, out);
}

static bool value_is_unresolved(const Value *value) {
    size_t index;
    if (value == NULL) return false;
    value = reactive_value(value);
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

LanaError lana_vm_joint_rename(LanaVM *vm, const LanaJointState *source,
                           const char *old_name, const char *new_name,
                           LanaJointState **out) {
    ssize_t position;
    size_t index, length = 1u;
    char *names_text;
    LanaError error;
    if (vm == NULL || source == NULL || old_name == NULL || new_name == NULL ||
        out == NULL || *new_name == '\0') return LANA_ERR_FORMAT;
    position = joint_find(source, old_name);
    if (position < 0) return LANA_ERR_KEY;
    if (joint_find(source, new_name) >= 0) return LANA_ERR_INVALID_DEPENDENCY;
    for (index = 0; index < source->count; ++index)
        length += strlen(index == (size_t)position ? new_name : source->names[index]) + 1u;
    names_text = malloc(length);
    if (names_text == NULL) return LANA_ERR_OOM;
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
            free(rows); free(weights); free(names_text); return LANA_ERR_OOM;
        }
        for (row = 0; row < source->row_count; ++row) {
            memcpy(&rows[row * source->count], source->rows[row].values,
                   source->count * sizeof(*rows));
            weights[row] = source->rows[row].weight;
        }
        error = lana_vm_joint_build_finite(vm, names_text, rows, weights,
                                         source->row_count, source->count, out);
        free(rows); free(weights);
    } else {
        char *descriptor = malloc(strlen(names_text) + sizeof("independent:"));
        if (descriptor == NULL) { free(names_text); return LANA_ERR_OOM; }
        (void)snprintf(descriptor, strlen(names_text) + sizeof("independent:"),
                       "independent:%s", names_text);
        error = lana_vm_joint_build(vm, source->values, source->count,
                                  descriptor, out);
        free(descriptor);
    }
    free(names_text);
    return error;
}

static void *run_task(void *context) {
    LanaTask *task = context;
    task->status = lana_vm_run(task->child);
    if (task->status == LANA_OK) task->result = task->child->result;
    else task->error = task->child->error;
    (void)pthread_mutex_lock(&task->mutex);
    task->completed = true;
    (void)pthread_cond_broadcast(&task->completed_condition);
    (void)pthread_mutex_unlock(&task->mutex);
    return NULL;
}

static LanaTask *scheduler_take_locked(LanaScheduler *scheduler) {
    LanaTask *task = scheduler->queue_head;
    if (task != NULL) {
        scheduler->queue_head = task->queue_next;
        if (scheduler->queue_head == NULL) scheduler->queue_tail = NULL;
        task->queue_next = NULL; task->queued = false;
    }
    return task;
}

static void *scheduler_worker(void *context) {
    LanaScheduler *scheduler = context;
    for (;;) {
        LanaTask *task;
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

static LanaError scheduler_initialize(LanaVM *owner) {
    LanaScheduler *scheduler;
    size_t index;
    if (owner->scheduler != NULL) return LANA_OK;
    scheduler = calloc(1u, sizeof(*scheduler));
    if (scheduler == NULL) return LANA_ERR_OOM;
    scheduler->worker_count = owner->configured_worker_count;
    scheduler->task_limit = owner->configured_task_limit;
    scheduler->next_task_id = 1u;
    scheduler->workers = calloc(scheduler->worker_count, sizeof(*scheduler->workers));
    if (scheduler->workers == NULL || pthread_mutex_init(&scheduler->mutex, NULL) != 0 ||
        pthread_cond_init(&scheduler->available, NULL) != 0) {
        free(scheduler->workers); free(scheduler); return LANA_ERR_TASK;
    }
    owner->scheduler = scheduler; owner->scheduler_owner = true;
    for (index = 0; index < scheduler->worker_count; ++index) {
        if (pthread_create(&scheduler->workers[index], NULL, scheduler_worker, scheduler) != 0) {
            scheduler->worker_count = index; scheduler_shutdown(scheduler);
            scheduler_destroy(scheduler);
            owner->scheduler = NULL; owner->scheduler_owner = false;
            return LANA_ERR_TASK;
        }
    }
    return LANA_OK;
}

static void scheduler_shutdown(LanaScheduler *scheduler) {
    LanaTask *task;
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

static void scheduler_destroy(LanaScheduler *scheduler) {
    (void)pthread_cond_destroy(&scheduler->available);
    (void)pthread_mutex_destroy(&scheduler->mutex);
    free(scheduler->workers);
    free(scheduler);
}

static LanaError start_task(LanaVM *parent, uint32_t function_index,
                          const Value *arguments, const LanaHistory *argument_histories,
                          size_t argc, LanaTask **out) {
    const LanaFunction *function = &parent->chunk->functions[function_index];
    LanaTask *task;
    size_t index;
    LanaError error = LANA_OK;
    LanaDistCloneMemo *memo = NULL;
    LanaContainerCloneMemo *containers = NULL;
    LanaDerivationCloneMemo *derivations = NULL;
    if (argc != function->arity) return LANA_ERR_TYPE;
    error = scheduler_initialize(parent);
    if (error != LANA_OK) return error;
    (void)pthread_mutex_lock(&parent->scheduler->mutex);
    if (parent->scheduler->stopping) error = LANA_ERR_TASK;
    else if (parent->scheduler->live_tasks >= parent->scheduler->task_limit) error = LANA_ERR_LIMIT;
    else ++parent->scheduler->live_tasks;
    (void)pthread_mutex_unlock(&parent->scheduler->mutex);
    if (error != LANA_OK) return error;
    task = calloc(1u, sizeof(*task));
    if (task == NULL) { error = LANA_ERR_OOM; goto release_slot; }
    task->child = malloc(sizeof(*task->child));
    if (task->child == NULL) { free(task); error = LANA_ERR_OOM; goto release_slot; }
    if (pthread_mutex_init(&task->mutex, NULL) != 0) {
        free(task->child); free(task); error = LANA_ERR_TASK; goto release_slot;
    }
    if (pthread_cond_init(&task->completed_condition, NULL) != 0) {
        (void)pthread_mutex_destroy(&task->mutex);
        free(task->child); free(task); error = LANA_ERR_TASK; goto release_slot;
    }
    (void)pthread_mutex_lock(&parent->scheduler->mutex);
    task->id = parent->scheduler->next_task_id++;
    (void)pthread_mutex_unlock(&parent->scheduler->mutex);
    task->group_id = parent->current_group_id;
    task->scheduler = parent->scheduler;
    task->result = lana_value_null();
    lana_vm_init(task->child, parent->chunk);
    task->child->scheduler = parent->scheduler;
    task->child->scheduler_owner = false;
    task->child->ip = function->entry;
    task->child->frames[0].function = function_index;
    task->child->task_id = task->id;
    task->child->instruction_limit = parent->instruction_limit;
    task->child->memory_limit = parent->memory_limit;
    task->child->trace = parent->trace;
    task->child->lineage = mix64(parent->lineage ^ ++parent->spawn_counter);
    lana_vm_seed(task->child, mix64(parent->root_seed ^ task->child->lineage));
    task->child->root_seed = parent->root_seed;
    lana_vm_set_program_args(task->child, parent->program_argc, parent->program_argv);
    for (index = 0; index < argc && error == LANA_OK; ++index)
        error = clone_value_memo(task->child, &arguments[index],
                                 &task->child->frames[0].registers[index], &memo,
                                 &containers, &derivations);
    for (index = 0; index < argc && error == LANA_OK; ++index)
        error = clone_history(task->child, &argument_histories[index],
                              &task->child->frames[0].histories[index]);
    while (memo != NULL) {
        LanaDistCloneMemo *entry = memo;
        memo = memo->next;
        free(entry);
    }
    while (containers != NULL) {
        LanaContainerCloneMemo *entry = containers;
        containers = containers->next;
        free(entry);
    }
    while (derivations != NULL) {
        LanaDerivationCloneMemo *entry = derivations;
        derivations = derivations->next;
        free(entry);
    }
    if (error != LANA_OK) { destroy_task(task); goto release_slot; }
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
    return LANA_OK;
release_slot:
    (void)pthread_mutex_lock(&parent->scheduler->mutex);
    --parent->scheduler->live_tasks;
    (void)pthread_mutex_unlock(&parent->scheduler->mutex);
    return error;
}

static LanaError wait_task(LanaVM *vm, LanaTask *task, double timeout_seconds, Value *out) {
    int wait_result = 0;
    struct timespec monotonic_deadline = {0};
    if (task == NULL) return LANA_ERR_TASK;
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
            LanaTask *helper;
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
    if (wait_result == ETIMEDOUT) return LANA_ERR_TIMEOUT;
    if (wait_result != 0) return LANA_ERR_TASK;
    if (task->status != LANA_OK) {
        vm->error = task->error;
        return task->status;
    }
    if (!task->joined) {
        Value cloned;
        LanaError error = clone_value(vm, &task->result, &cloned);
        if (error != LANA_OK) return error;
        task->result = cloned;
        lana_vm_free(task->child); free(task->child); task->child = NULL; task->joined = true;
        (void)pthread_mutex_lock(&vm->scheduler->mutex);
        if (vm->scheduler->live_tasks > 0u) --vm->scheduler->live_tasks;
        (void)pthread_mutex_unlock(&vm->scheduler->mutex);
    }
    *out = task->result;
    return LANA_OK;
}

static LanaError host_read_text(LanaVM *vm, const Value *argument, Value *out) {
    FILE *file;
    long length;
    char *contents;
    if (argument->type != VAL_STRING) return LANA_ERR_TYPE;
    file = fopen(argument->as.string, "rb");
    if (file == NULL) return LANA_ERR_IO;
    if (fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0) { (void)fclose(file); return LANA_ERR_IO; }
    if (vm->allocated_bytes > vm->memory_limit ||
        (size_t)length > vm->memory_limit - vm->allocated_bytes) {
        (void)fclose(file); return LANA_ERR_LIMIT;
    }
    contents = lana_vm_alloc(vm, (size_t)length + 1u);
    if (contents == NULL) { (void)fclose(file); return LANA_ERR_OOM; }
    if (fread(contents, 1, (size_t)length, file) != (size_t)length) {
        (void)fclose(file); return LANA_ERR_IO;
    }
    contents[length] = '\0';
    if (fclose(file) != 0) return LANA_ERR_IO;
    *out = lana_value_string(contents);
    return LANA_OK;
}

static LanaError execute_captured_payload(LanaVM *vm, const char *kind,
                                          const Value *payload, void *context,
                                          Value *out) {
    (void)kind;
    (void)context;
    return clone_value(vm, payload, out);
}

static uint32_t shared_permission(const Value *value) {
    if (value->type != VAL_STRING) return 0u;
    if (strcmp(value->as.string, "read") == 0) return LANA_CAPABILITY_READ;
    if (strcmp(value->as.string, "observe") == 0)
        return LANA_CAPABILITY_OBSERVE;
    if (strcmp(value->as.string, "admin") == 0) return LANA_CAPABILITY_ADMIN;
    return 0u;
}

static bool nonnegative_integer(const Value *value) {
    return value->type == VAL_NUMBER && isfinite(value->as.number) &&
           value->as.number >= 0.0 && floor(value->as.number) == value->as.number &&
           value->as.number <= 9007199254740991.0;
}

typedef struct {
    char **names;
    size_t count;
    size_t capacity;
} HostDirectoryEntries;

static void host_directory_entries_free(HostDirectoryEntries *entries) {
    size_t index;
    for (index = 0u; index < entries->count; ++index) free(entries->names[index]);
    free(entries->names);
    entries->names = NULL; entries->count = 0u; entries->capacity = 0u;
}

static int host_directory_entry_compare(const void *left, const void *right) {
    return strcmp(*(const char *const *)left, *(const char *const *)right);
}

static char *host_string_copy(LanaVM *vm, const char *text) {
    size_t length = strlen(text);
    char *copy = lana_vm_alloc(vm, length + 1u);
    if (copy != NULL) memcpy(copy, text, length + 1u);
    return copy;
}

static LanaError host_hash_update(LanaVM *vm, const Value *seed,
                                  const Value *text, Value *out) {
    static const char digits[] = "0123456789abcdef";
    uint64_t hash = 0u;
    char *result;
    size_t index;
    if (seed->type != VAL_STRING || text->type != VAL_STRING ||
        strlen(seed->as.string) != 16u) return LANA_ERR_TYPE;
    for (index = 0u; index < 16u; ++index) {
        unsigned char byte = (unsigned char)seed->as.string[index];
        unsigned char value;
        if (byte >= '0' && byte <= '9') value = (unsigned char)(byte - '0');
        else if (byte >= 'a' && byte <= 'f') value = (unsigned char)(byte - 'a' + 10u);
        else if (byte >= 'A' && byte <= 'F') value = (unsigned char)(byte - 'A' + 10u);
        else return LANA_ERR_FORMAT;
        hash = (hash << 4u) | value;
    }
    for (index = 0u; index < strlen(text->as.string); ++index) {
        hash ^= (unsigned char)text->as.string[index];
        hash *= UINT64_C(1099511628211);
    }
    result = lana_vm_alloc(vm, 17u);
    if (result == NULL) return LANA_ERR_OOM;
    for (index = 0u; index < 16u; ++index)
        result[index] = digits[(hash >> ((15u - index) * 4u)) & 15u];
    result[16] = '\0';
    *out = lana_value_string(result);
    return LANA_OK;
}

static LanaError host_hash_xor(LanaVM *vm, const Value *left,
                               const Value *right, Value *out) {
    char *result;
    size_t index;
    if (left->type != VAL_STRING || right->type != VAL_STRING ||
        strlen(left->as.string) != 16u || strlen(right->as.string) != 16u)
        return LANA_ERR_TYPE;
    result = lana_vm_alloc(vm, 17u);
    if (result == NULL) return LANA_ERR_OOM;
    for (index = 0u; index < 16u; ++index) {
        unsigned char l = (unsigned char)left->as.string[index];
        unsigned char r = (unsigned char)right->as.string[index];
        int high = (l >= '0' && l <= '9') ? l - '0' :
                   (l >= 'a' && l <= 'f') ? l - 'a' + 10 :
                   (l >= 'A' && l <= 'F') ? l - 'A' + 10 : -1;
        int low = (r >= '0' && r <= '9') ? r - '0' :
                  (r >= 'a' && r <= 'f') ? r - 'a' + 10 :
                  (r >= 'A' && r <= 'F') ? r - 'A' + 10 : -1;
        if (high < 0 || low < 0) return LANA_ERR_FORMAT;
        result[index] = "0123456789abcdef"[high ^ low];
    }
    result[16] = '\0'; *out = lana_value_string(result); return LANA_OK;
}

static LanaError host_directory_list(LanaVM *vm, const Value *argument,
                                     Value *out) {
    DIR *directory;
    struct dirent *entry;
    HostDirectoryEntries entries = {0};
    LanaArray *array;
    size_t index;
    if (argument->type != VAL_STRING) return LANA_ERR_TYPE;
    directory = opendir(argument->as.string);
    if (directory == NULL) return LANA_ERR_IO;
    while ((entry = readdir(directory)) != NULL) {
        char **grown;
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
            continue;
        if (entries.count == entries.capacity) {
            size_t capacity = entries.capacity == 0u ? 8u : entries.capacity * 2u;
            if (capacity <= entries.capacity || capacity > SIZE_MAX / sizeof(*grown)) {
                (void)closedir(directory); host_directory_entries_free(&entries);
                return LANA_ERR_LIMIT;
            }
            grown = realloc(entries.names, capacity * sizeof(*grown));
            if (grown == NULL) {
                (void)closedir(directory); host_directory_entries_free(&entries);
                return LANA_ERR_OOM;
            }
            entries.names = grown; entries.capacity = capacity;
        }
        entries.names[entries.count] = strdup(entry->d_name);
        if (entries.names[entries.count] == NULL) {
            (void)closedir(directory); host_directory_entries_free(&entries);
            return LANA_ERR_OOM;
        }
        ++entries.count;
    }
    if (closedir(directory) != 0) {
        host_directory_entries_free(&entries); return LANA_ERR_IO;
    }
    qsort(entries.names, entries.count, sizeof(*entries.names),
          host_directory_entry_compare);
    array = lana_vm_alloc(vm, sizeof(*array));
    if (array == NULL) { host_directory_entries_free(&entries); return LANA_ERR_OOM; }
    array->count = entries.count; array->capacity = entries.count;
    array->items = entries.count == 0u ? NULL :
        lana_vm_alloc(vm, entries.count * sizeof(*array->items));
    if (entries.count > 0u && array->items == NULL) {
        host_directory_entries_free(&entries); return LANA_ERR_OOM;
    }
    for (index = 0u; index < entries.count; ++index) {
        char path[PATH_MAX];
        struct stat metadata;
        LanaMap *map;
        Value name;
        Value kind;
        LanaError error;
        int written = snprintf(path, sizeof(path), "%s/%s",
                               argument->as.string, entries.names[index]);
        if (written < 0 || (size_t)written >= sizeof(path) ||
            stat(path, &metadata) != 0) {
            host_directory_entries_free(&entries); return LANA_ERR_IO;
        }
        error = lana_map_new(vm, 2u, &map);
        if (error != LANA_OK) { host_directory_entries_free(&entries); return error; }
        name = lana_value_string(host_string_copy(vm, entries.names[index]));
        kind = lana_value_string(S_ISDIR(metadata.st_mode) ? "directory" : "file");
        if (name.as.string == NULL ||
            (error = lana_map_set(vm, map, "name", &name, true)) != LANA_OK ||
            (error = lana_map_set(vm, map, "kind", &kind, true)) != LANA_OK) {
            host_directory_entries_free(&entries);
            return name.as.string == NULL ? LANA_ERR_OOM : error;
        }
        array->items[index] = lana_value_map(map);
    }
    host_directory_entries_free(&entries);
    *out = lana_value_array(array);
    return LANA_OK;
}

static LanaError host_directory_create(const Value *argument) {
    struct stat metadata;
    if (argument->type != VAL_STRING) return LANA_ERR_TYPE;
    if (mkdir(argument->as.string, 0755) == 0) return LANA_OK;
    if (errno != EEXIST || stat(argument->as.string, &metadata) != 0 ||
        !S_ISDIR(metadata.st_mode)) return LANA_ERR_IO;
    return LANA_OK;
}

static LanaError host_path_exists(const Value *argument, Value *out) {
    struct stat metadata;
    if (argument->type != VAL_STRING) return LANA_ERR_TYPE;
    if (stat(argument->as.string, &metadata) == 0) {
        *out = lana_value_bool(true); return LANA_OK;
    }
    if (errno == ENOENT) { *out = lana_value_bool(false); return LANA_OK; }
    return LANA_ERR_IO;
}

static LanaError host_write_text_atomic(const Value *path,
                                        const Value *contents) {
    char *temporary;
    size_t length, content_length;
    int descriptor;
    FILE *file = NULL;
    bool success = false;
    if (path->type != VAL_STRING || contents->type != VAL_STRING)
        return LANA_ERR_TYPE;
    length = strlen(path->as.string);
    if (length > SIZE_MAX - sizeof(".lana-tmp-XXXXXX")) return LANA_ERR_LIMIT;
    temporary = malloc(length + sizeof(".lana-tmp-XXXXXX"));
    if (temporary == NULL) return LANA_ERR_OOM;
    (void)snprintf(temporary, length + sizeof(".lana-tmp-XXXXXX"),
                   "%s.lana-tmp-XXXXXX", path->as.string);
    descriptor = mkstemp(temporary);
    if (descriptor < 0) { free(temporary); return LANA_ERR_IO; }
    file = fdopen(descriptor, "wb");
    if (file == NULL) {
        (void)close(descriptor); (void)unlink(temporary); free(temporary);
        return LANA_ERR_IO;
    }
    content_length = strlen(contents->as.string);
    {
        bool write_ok = fwrite(contents->as.string, 1u, content_length, file) ==
                        content_length;
        int close_result = fclose(file);
        file = NULL;
        if (write_ok && close_result == 0 && rename(temporary, path->as.string) == 0)
            success = true;
    }
    if (!success) (void)unlink(temporary);
    free(temporary);
    return success ? LANA_OK : LANA_ERR_IO;
}

static LanaError execute_host_call(LanaVM *vm, uint32_t host_id, const Value *arguments,
                                 size_t argc, Value *out) {
    size_t index;
    *out = lana_value_null();
    switch ((LanaHostCallId)host_id) {
        case LANA_HOST_ARGS: {
            LanaArray *array;
            if (argc != 0u) return LANA_ERR_TYPE;
            array = lana_vm_alloc(vm, sizeof(*array));
            if (array == NULL) return LANA_ERR_OOM;
            array->count = (size_t)vm->program_argc;
            array->capacity = array->count;
            array->items = lana_vm_alloc(vm, array->count * sizeof(*array->items));
            if (array->items == NULL && array->count > 0u) return LANA_ERR_OOM;
            for (index = 0; index < array->count; ++index) {
                Value source = lana_value_string(vm->program_argv[index]);
                LanaError error = clone_value(vm, &source, &array->items[index]);
                if (error != LANA_OK) return error;
            }
            out->type = VAL_ARRAY; out->as.array = array; return LANA_OK;
        }
        case LANA_HOST_READ_TEXT:
            return argc == 1u ? host_read_text(vm, &arguments[0], out) : LANA_ERR_TYPE;
        case LANA_HOST_WRITE_TEXT: {
            FILE *file;
            size_t length;
            if (argc != 2u || arguments[0].type != VAL_STRING || arguments[1].type != VAL_STRING)
                return LANA_ERR_TYPE;
            file = fopen(arguments[0].as.string, "wb");
            if (file == NULL) return LANA_ERR_IO;
            length = strlen(arguments[1].as.string);
            if (fwrite(arguments[1].as.string, 1, length, file) != length) {
                (void)fclose(file); return LANA_ERR_IO;
            }
            if (fclose(file) != 0) return LANA_ERR_IO;
            return LANA_OK;
        }
        case LANA_HOST_DIRECTORY_LIST:
            if (argc != 1u) return LANA_ERR_TYPE;
            return host_directory_list(vm, &arguments[0], out);
        case LANA_HOST_DIRECTORY_CREATE:
            if (argc != 1u) return LANA_ERR_TYPE;
            return host_directory_create(&arguments[0]);
        case LANA_HOST_PATH_EXISTS:
            if (argc != 1u) return LANA_ERR_TYPE;
            return host_path_exists(&arguments[0], out);
        case LANA_HOST_WRITE_TEXT_ATOMIC:
            if (argc != 2u) return LANA_ERR_TYPE;
            return host_write_text_atomic(&arguments[0], &arguments[1]);
        case LANA_HOST_HASH_UPDATE:
            if (argc == 3u && arguments[2].type == VAL_STRING &&
                strcmp(arguments[2].as.string, "xor") == 0)
                return host_hash_xor(vm, &arguments[0], &arguments[1], out);
            if (argc != 2u) return LANA_ERR_TYPE;
            return host_hash_update(vm, &arguments[0], &arguments[1], out);
        case LANA_HOST_NOW: {
            struct timespec now;
            if (argc != 0u || timespec_get(&now, TIME_UTC) != TIME_UTC) return LANA_ERR_TYPE;
            *out = lana_value_number((double)now.tv_sec + (double)now.tv_nsec / 1000000000.0);
            return LANA_OK;
        }
        case LANA_HOST_RANDOM:
            if (argc != 0u) return LANA_ERR_TYPE;
            *out = lana_value_number((double)lana_vm_random(vm) / 4294967296.0); return LANA_OK;
        case LANA_HOST_ASSERT:
            if (argc != 2u ||
                arguments[0].type != VAL_BOOL ||
                (argc == 2u && arguments[1].type != VAL_STRING)) return LANA_ERR_TYPE;
            return arguments[0].as.boolean ? LANA_OK : LANA_ERR_ASSERTION;
        case LANA_HOST_MAP_NEW: {
            LanaMap *map; LanaError error;
            if (argc % 2u != 0u) return LANA_ERR_TYPE;
            error = lana_map_new(vm, argc / 2u, &map);
            for (index = 0u; error == LANA_OK && index < argc; index += 2u) {
                if (arguments[index].type != VAL_STRING) return LANA_ERR_TYPE;
                error = lana_map_set(vm, map, arguments[index].as.string,
                                   &arguments[index + 1u], true);
            }
            if (error == LANA_OK) *out = lana_value_map(map);
            return error;
        }
        case LANA_HOST_MAP_HAS: {
            if (argc != 2u || arguments[0].type != VAL_MAP || arguments[1].type != VAL_STRING) return LANA_ERR_TYPE;
            bool found = lana_map_has(arguments[0].as.map, arguments[1].as.string) >= 0;
            *out = lana_value_bool(found);
            return LANA_OK;
        }
        case LANA_HOST_MAP_GET: {
            if (argc != 2u || arguments[0].type != VAL_MAP || arguments[1].type != VAL_STRING)
                return LANA_ERR_TYPE;
            return lana_map_get(arguments[0].as.map, arguments[1].as.string, out);
        }
        case LANA_HOST_MAP_SET: {
            if (argc != 3u || arguments[0].type != VAL_MAP || arguments[1].type != VAL_STRING)
                return LANA_ERR_TYPE;
            LanaError error = lana_map_set(vm, arguments[0].as.map, arguments[1].as.string,
                                           &arguments[2], false);
            if (error == LANA_OK) *out = arguments[2];
            return error;
        }
        case LANA_HOST_MAP_KEYS: {
            if (argc != 1u || arguments[0].type != VAL_MAP) return LANA_ERR_TYPE;
            LanaMap *map = arguments[0].as.map;
            LanaArray *keys = lana_vm_alloc(vm, sizeof(*keys));
            if (keys == NULL) return LANA_ERR_OOM;
            keys->count = map->count;
            keys->capacity = map->count;
            keys->items = lana_vm_alloc(vm, map->count * sizeof(*keys->items));
            if (keys->items == NULL && map->count > 0u) return LANA_ERR_OOM;
            for (index = 0u; index < map->count; ++index) {
                keys->items[index] = lana_value_string(map->entries[index].key);
            }
            *out = lana_value_array(keys);
            return LANA_OK;
        }
        case LANA_HOST_INDEX_GET:
            if (argc != 2u) return LANA_ERR_TYPE;
            if (arguments[0].type == VAL_MAP && arguments[1].type == VAL_STRING)
                return lana_map_get(arguments[0].as.map, arguments[1].as.string, out);
            if (arguments[0].type == VAL_ARRAY && arguments[1].type == VAL_NUMBER &&
                arguments[1].as.number >= 0.0 && floor(arguments[1].as.number) == arguments[1].as.number &&
                (size_t)arguments[1].as.number < arguments[0].as.array->count) {
                *out = arguments[0].as.array->items[(size_t)arguments[1].as.number]; return LANA_OK;
            }
            return arguments[0].type == VAL_ARRAY ? LANA_ERR_LIMIT : LANA_ERR_TYPE;
        case LANA_HOST_INDEX_SET:
            if (argc != 3u) return LANA_ERR_TYPE;
            if (arguments[0].type == VAL_MAP && arguments[1].type == VAL_STRING) {
                LanaError error = lana_map_set(vm, arguments[0].as.map, arguments[1].as.string,
                                           &arguments[2], false);
                if (error == LANA_OK) *out = arguments[2];
                return error;
            }
            if (arguments[0].type == VAL_ARRAY && arguments[1].type == VAL_NUMBER &&
                arguments[1].as.number >= 0.0 && floor(arguments[1].as.number) == arguments[1].as.number &&
                (size_t)arguments[1].as.number < arguments[0].as.array->count) {
                lana_vm_write_barrier_value(vm, arguments[0].as.array,
                                            &arguments[2]);
                arguments[0].as.array->items[(size_t)arguments[1].as.number] = arguments[2];
                *out = arguments[2]; return LANA_OK;
            }
            return arguments[0].type == VAL_ARRAY ? LANA_ERR_LIMIT : LANA_ERR_TYPE;
        case LANA_HOST_JSON_PARSE:
            return argc == 1u && arguments[0].type == VAL_STRING ? lana_json_parse(vm, arguments[0].as.string, out) : LANA_ERR_TYPE;
        case LANA_HOST_JSON_STRINGIFY:
            return argc == 1u ? lana_json_stringify(vm, &arguments[0], out) : LANA_ERR_TYPE;
        case LANA_HOST_CSV_READ:
            return argc == 1u && arguments[0].type == VAL_STRING ? lana_csv_read(vm, arguments[0].as.string, out) : LANA_ERR_TYPE;
        case LANA_HOST_CSV_WRITE:
            return argc == 2u && arguments[0].type == VAL_STRING ? lana_csv_write(vm, arguments[0].as.string, &arguments[1], out) : LANA_ERR_TYPE;
        case LANA_HOST_STRING_LENGTH:
            if (argc != 1u || arguments[0].type != VAL_STRING) return LANA_ERR_TYPE;
            *out = lana_value_number((double)strlen(arguments[0].as.string)); return LANA_OK;
        case LANA_HOST_STRING_BYTE_AT: {
            size_t position, length;
            if (argc != 2u || arguments[0].type != VAL_STRING ||
                arguments[1].type != VAL_NUMBER || arguments[1].as.number < 0.0 ||
                floor(arguments[1].as.number) != arguments[1].as.number)
                return LANA_ERR_TYPE;
            position = (size_t)arguments[1].as.number;
            length = strlen(arguments[0].as.string);
            if (position >= length) return LANA_ERR_LIMIT;
            *out = lana_value_number((unsigned char)arguments[0].as.string[position]);
            return LANA_OK;
        }
        case LANA_HOST_STRING_SLICE: {
            size_t start, end, length;
            char *copy;
            if (argc != 3u || arguments[0].type != VAL_STRING ||
                arguments[1].type != VAL_NUMBER || arguments[2].type != VAL_NUMBER ||
                arguments[1].as.number < 0.0 || arguments[2].as.number < 0.0 ||
                floor(arguments[1].as.number) != arguments[1].as.number ||
                floor(arguments[2].as.number) != arguments[2].as.number)
                return LANA_ERR_TYPE;
            start = (size_t)arguments[1].as.number;
            end = (size_t)arguments[2].as.number;
            length = strlen(arguments[0].as.string);
            if (start > end || end > length) return LANA_ERR_LIMIT;
            copy = lana_vm_alloc(vm, end - start + 1u);
            if (copy == NULL) return LANA_ERR_OOM;
            memcpy(copy, arguments[0].as.string + start, end - start);
            copy[end - start] = '\0'; *out = lana_value_string(copy); return LANA_OK;
        }
        case LANA_HOST_STRING_CONCAT: {
            size_t total = 0u, offset = 0u;
            char *copy;
            for (index = 0; index < argc; ++index) {
                size_t length;
                if (arguments[index].type != VAL_STRING) return LANA_ERR_TYPE;
                length = strlen(arguments[index].as.string);
                if (length > SIZE_MAX - total - 1u) return LANA_ERR_LIMIT;
                total += length;
            }
            copy = lana_vm_alloc(vm, total + 1u);
            if (copy == NULL) return LANA_ERR_OOM;
            for (index = 0; index < argc; ++index) {
                size_t length = strlen(arguments[index].as.string);
                memcpy(copy + offset, arguments[index].as.string, length); offset += length;
            }
            copy[offset] = '\0'; *out = lana_value_string(copy); return LANA_OK;
        }
        case LANA_HOST_NUMBER_TO_STRING: {
            char buffer[64]; char *copy; int written;
            if (argc != 1u || arguments[0].type != VAL_NUMBER) return LANA_ERR_TYPE;
            written = snprintf(buffer, sizeof(buffer), "%.17g", arguments[0].as.number);
            if (written < 0 || (size_t)written >= sizeof(buffer)) return LANA_ERR_FORMAT;
            copy = lana_vm_alloc(vm, (size_t)written + 1u);
            if (copy == NULL) return LANA_ERR_OOM;
            memcpy(copy, buffer, (size_t)written + 1u);
            *out = lana_value_string(copy); return LANA_OK;
        }
        case LANA_HOST_ARRAY_NEW: {
            LanaArray *array; size_t count;
            if (argc != 1u || arguments[0].type != VAL_NUMBER ||
                arguments[0].as.number < 0.0 ||
                floor(arguments[0].as.number) != arguments[0].as.number ||
                arguments[0].as.number > (double)LANA_MAX_REGISTERS * 4096.0)
                return LANA_ERR_TYPE;
            count = (size_t)arguments[0].as.number;
            array = lana_vm_alloc(vm, sizeof(*array));
            if (array == NULL) return LANA_ERR_OOM;
            array->count = count;
            array->capacity = count;
            array->items = lana_vm_alloc(vm, count * sizeof(*array->items));
            if (array->items == NULL && count > 0u) return LANA_ERR_OOM;
            *out = lana_value_array(array); return LANA_OK;
        }
        case LANA_HOST_ARRAY_PUSH: {
            LanaArray *array; Value *items;
            if (argc != 2u || arguments[0].type != VAL_ARRAY ||
                arguments[0].as.array == NULL) return LANA_ERR_TYPE;
            array = arguments[0].as.array;
            if (array->count == SIZE_MAX / sizeof(*items)) return LANA_ERR_LIMIT;
            if (array->count == array->capacity) {
                size_t capacity = array->capacity == 0u ? 8u : array->capacity * 2u;
                if (capacity <= array->capacity) return LANA_ERR_LIMIT;
                items = lana_vm_alloc(vm, capacity * sizeof(*items));
                if (items == NULL) return LANA_ERR_OOM;
                if (array->count > 0u)
                    memcpy(items, array->items, array->count * sizeof(*items));
                array->items = items; array->capacity = capacity;
            }
            lana_vm_write_barrier_value(vm, array, &arguments[1]);
            array->items[array->count++] = arguments[1];
            *out = arguments[0]; return LANA_OK;
        }
        case LANA_HOST_STRING_HEX: {
            static const char digits[] = "0123456789abcdef";
            const unsigned char *source; size_t length; char *hex;
            if (argc != 1u || arguments[0].type != VAL_STRING) return LANA_ERR_TYPE;
            source = (const unsigned char *)arguments[0].as.string;
            length = strlen(arguments[0].as.string);
            if (length > (SIZE_MAX - 1u) / 2u) return LANA_ERR_LIMIT;
            hex = lana_vm_alloc(vm, length * 2u + 1u);
            if (hex == NULL) return LANA_ERR_OOM;
            for (index = 0; index < length; ++index) {
                hex[index * 2u] = digits[source[index] >> 4u];
                hex[index * 2u + 1u] = digits[source[index] & 15u];
            }
            hex[length * 2u] = '\0'; *out = lana_value_string(hex); return LANA_OK;
        }
        case LANA_HOST_STRING_JOIN: {
            const LanaArray *array; const char *separator; size_t separator_length;
            size_t total = 0u, offset = 0u; char *joined;
            if (argc != 2u || arguments[0].type != VAL_ARRAY ||
                arguments[1].type != VAL_STRING) return LANA_ERR_TYPE;
            array = arguments[0].as.array; separator = arguments[1].as.string;
            separator_length = strlen(separator);
            for (index = 0; index < array->count; ++index) {
                size_t length;
                if (array->items[index].type != VAL_STRING) return LANA_ERR_TYPE;
                length = strlen(array->items[index].as.string);
                if (length > SIZE_MAX - total - 1u) return LANA_ERR_LIMIT;
                total += length;
                if (index + 1u < array->count) {
                    if (separator_length > SIZE_MAX - total - 1u) return LANA_ERR_LIMIT;
                    total += separator_length;
                }
            }
            joined = lana_vm_alloc(vm, total + 1u);
            if (joined == NULL) return LANA_ERR_OOM;
            for (index = 0; index < array->count; ++index) {
                size_t length = strlen(array->items[index].as.string);
                memcpy(joined + offset, array->items[index].as.string, length); offset += length;
                if (index + 1u < array->count) {
                    memcpy(joined + offset, separator, separator_length);
                    offset += separator_length;
                }
            }
            joined[offset] = '\0'; *out = lana_value_string(joined); return LANA_OK;
        }
        case LANA_HOST_ARRAY_LENGTH:
            if (argc != 1u || arguments[0].type != VAL_ARRAY) return LANA_ERR_TYPE;
            *out = lana_value_number((double)arguments[0].as.array->count); return LANA_OK;
        case LANA_HOST_STRING_UNESCAPE: {
            const char *source; size_t length, read = 0u, written = 0u; char *decoded;
            if (argc != 1u || arguments[0].type != VAL_STRING) return LANA_ERR_TYPE;
            source = arguments[0].as.string; length = strlen(source);
            decoded = lana_vm_alloc(vm, length + 1u);
            if (decoded == NULL) return LANA_ERR_OOM;
            while (read < length) {
                char value = source[read++];
                if (value != '\\') { decoded[written++] = value; continue; }
                if (read >= length) return LANA_ERR_FORMAT;
                value = source[read++];
                if (value == 'n') decoded[written++] = '\n';
                else if (value == 'r') decoded[written++] = '\r';
                else if (value == 't') decoded[written++] = '\t';
                else if (value == '\\' || value == '"') decoded[written++] = value;
                else return LANA_ERR_FORMAT;
            }
            decoded[written] = '\0'; *out = lana_value_string(decoded); return LANA_OK;
        }
        case LANA_HOST_PATH_RESOLVE: {
            const char *base, *relative, *separator; size_t directory_length, needed; char *candidate, resolved[PATH_MAX], *copy;
            if (argc != 2u || arguments[0].type != VAL_STRING || arguments[1].type != VAL_STRING) return LANA_ERR_TYPE;
            base = arguments[0].as.string; relative = arguments[1].as.string;
            if (relative[0] == '\0') {
                if (realpath(base, resolved) == NULL) return LANA_ERR_IO;
                copy = lana_vm_alloc(vm, strlen(resolved) + 1u); if (copy == NULL) return LANA_ERR_OOM;
                (void)strcpy(copy, resolved); *out = lana_value_string(copy); return LANA_OK;
            }
            separator = strrchr(base, '/');
            directory_length = separator == NULL ? 1u : (size_t)(separator - base);
            if (strlen(relative) > SIZE_MAX - directory_length - 2u) return LANA_ERR_LIMIT;
            needed = directory_length + strlen(relative) + 2u; candidate = malloc(needed);
            if (candidate == NULL) return LANA_ERR_OOM;
            if (separator == NULL) (void)snprintf(candidate, needed, "./%s", relative);
            else (void)snprintf(candidate, needed, "%.*s/%s", (int)directory_length, base, relative);
            if (realpath(candidate, resolved) == NULL) { free(candidate); return LANA_ERR_IO; }
            free(candidate); copy = lana_vm_alloc(vm, strlen(resolved) + 1u); if (copy == NULL) return LANA_ERR_OOM;
            (void)strcpy(copy, resolved); *out = lana_value_string(copy); return LANA_OK;
        }
        case LANA_HOST_SAMPLE_RECORD: {
            LanaArray *sample;
            LanaMap *metadata;
            LanaError error;
            if (argc != 3u || arguments[1].type != VAL_STRING ||
                arguments[2].type != VAL_STRING) return LANA_ERR_TYPE;
            sample = lana_vm_alloc(vm, sizeof(*sample));
            if (sample == NULL) return LANA_ERR_OOM;
            sample->count = 2u;
            sample->capacity = 2u;
            sample->items = lana_vm_alloc(vm, 2u * sizeof(*sample->items));
            if (sample->items == NULL) return LANA_ERR_OOM;
            error = lana_map_new(vm, 5u, &metadata);
            if (error != LANA_OK) return error;
            sample->items[0] = arguments[0];
            sample->items[1] = lana_value_map(metadata);
            if ((error = map_put(vm, metadata, "source_dependency", arguments[1])) != LANA_OK ||
                (error = map_put(vm, metadata, "rng_seed", lana_value_number((double)vm->root_seed))) != LANA_OK ||
                (error = map_put(vm, metadata, "task_lineage", lana_value_number((double)vm->lineage))) != LANA_OK ||
                (error = map_put(vm, metadata, "operation", arguments[2])) != LANA_OK ||
                (error = map_put(vm, metadata, "revision", lana_value_number((double)vm->revision))) != LANA_OK)
                return error;
            *out = lana_value_array(sample);
            return LANA_OK;
        }
        case LANA_HOST_INFORMATION_NEW:
            if (argc != 1u) return LANA_ERR_TYPE;
            return lana_vm_reactive_root(vm, &arguments[0],
                                         LANA_EXACTNESS_EXACT, out);
        case LANA_HOST_CLAIM_NEW:
            if (argc != 2u || arguments[1].type != VAL_STRING)
                return LANA_ERR_TYPE;
            return lana_vm_claim(vm, &arguments[0], arguments[1].as.string,
                                 LANA_EXACTNESS_EXACT, 0.0, true, out);
        case LANA_HOST_CLAIM_VALUE:
            if (argc != 1u || arguments[0].claim == NULL)
                return LANA_ERR_TYPE;
            return clone_value(vm, arguments[0].claim->value, out);
        case LANA_HOST_CLAIM_PROPOSITION:
            if (argc != 1u || arguments[0].claim == NULL)
                return LANA_ERR_TYPE;
            *out = lana_value_string(arguments[0].claim->proposition);
            return LANA_OK;
        case LANA_HOST_CLAIM_STATUS: {
            LanaMap *status;
            LanaClaim *claim;
            LanaError error;
            if (argc != 1u || arguments[0].claim == NULL)
                return LANA_ERR_TYPE;
            claim = arguments[0].claim;
            error = lana_map_new(vm, 3u, &status);
            if (error != LANA_OK) return error;
            if ((error = map_put(vm, status, "exactness",
                    lana_value_string(derivation_exactness_name(claim->exactness)))) != LANA_OK ||
                (error = map_put(vm, status, "tolerance",
                    lana_value_number(claim->tolerance))) != LANA_OK ||
                (error = map_put(vm, status, "source_valid",
                    lana_value_bool(claim->source_valid))) != LANA_OK)
                return error;
            *out = lana_value_map(status);
            return LANA_OK;
        }
        case LANA_HOST_PLANNED_EFFECT_NEW:
            if (argc != 2u || arguments[0].type != VAL_STRING)
                return LANA_ERR_TYPE;
            return lana_vm_planned_effect(vm, arguments[0].as.string,
                                          &arguments[1], out);
        case LANA_HOST_PLANNED_EFFECT_EXECUTE:
            if (argc != 1u) return LANA_ERR_TYPE;
            return lana_vm_execute_planned_effect(vm, &arguments[0],
                                                  execute_captured_payload,
                                                  NULL, out);
        case LANA_HOST_PLANNED_EFFECT_STATUS: {
            LanaMap *status;
            LanaPlannedEffect *plan;
            LanaError error;
            if (argc != 1u || arguments[0].planned_effect == NULL)
                return LANA_ERR_TYPE;
            plan = arguments[0].planned_effect;
            error = lana_map_new(vm, 3u, &status);
            if (error != LANA_OK) return error;
            if ((error = map_put(vm, status, "identity",
                    lana_value_number((double)plan->id))) != LANA_OK ||
                (error = map_put(vm, status, "execution_count",
                    lana_value_number((double)plan->execution_count))) != LANA_OK ||
                (error = map_put(vm, status, "kind",
                    lana_value_string(plan->kind))) != LANA_OK)
                return error;
            *out = lana_value_map(status);
            return LANA_OK;
        }
        case LANA_HOST_SHARED_INFORMATION: {
            LanaSharedInformation *shared;
            LanaCapabilityToken *admin;
            LanaError error;
            if (argc != 1u) return LANA_ERR_TYPE;
            error = lana_shared_information_create(vm, &arguments[0], &shared,
                                                   &admin);
            if (error != LANA_OK) return error;
            error = vm_track_shared(vm, shared, false);
            if (error != LANA_OK) {
                lana_shared_information_release(shared);
                return error;
            }
            *out = lana_value_shared_capability(admin);
            return LANA_OK;
        }
        case LANA_HOST_SHARED_GRANT: {
            LanaCapabilityToken *capability;
            uint32_t permission;
            if (argc != 2u || arguments[0].type != VAL_SHARED_CAPABILITY)
                return LANA_ERR_TYPE;
            permission = shared_permission(&arguments[1]);
            if (permission == 0u) return LANA_ERR_TYPE;
            {
                LanaError error = lana_shared_capability_grant(
                    arguments[0].as.capability, permission, &capability);
                if (error != LANA_OK) return error;
            }
            *out = lana_value_shared_capability(capability);
            return LANA_OK;
        }
        case LANA_HOST_SHARED_REVOKE:
            if (argc != 2u || arguments[0].type != VAL_SHARED_CAPABILITY ||
                arguments[1].type != VAL_SHARED_CAPABILITY)
                return LANA_ERR_TYPE;
            return lana_shared_capability_revoke(arguments[0].as.capability,
                                                 arguments[1].as.capability);
        case LANA_HOST_SHARED_SNAPSHOT: {
            LanaSharedInformation *shared;
            if (argc != 1u || arguments[0].type != VAL_SHARED_CAPABILITY)
                return LANA_ERR_TYPE;
            shared = lana_shared_capability_information(
                arguments[0].as.capability);
            return lana_shared_information_snapshot(vm, shared,
                arguments[0].as.capability, out, NULL);
        }
        case LANA_HOST_SHARED_AT: {
            LanaSharedInformation *shared;
            if (argc != 2u || arguments[0].type != VAL_SHARED_CAPABILITY ||
                !nonnegative_integer(&arguments[1])) return LANA_ERR_TYPE;
            shared = lana_shared_capability_information(
                arguments[0].as.capability);
            return lana_shared_information_at(vm, shared,
                arguments[0].as.capability, arguments[1].as.number, out, NULL);
        }
        case LANA_HOST_SHARED_OBSERVE: {
            LanaSharedInformation *shared;
            uint64_t revision;
            LanaError error;
            if (argc != 3u || arguments[0].type != VAL_SHARED_CAPABILITY ||
                !nonnegative_integer(&arguments[2])) return LANA_ERR_TYPE;
            shared = lana_shared_capability_information(
                arguments[0].as.capability);
            error = lana_shared_information_observe(vm, shared,
                arguments[0].as.capability, &arguments[1],
                arguments[2].as.number, &revision);
            if (error == LANA_OK) *out = lana_value_number((double)revision);
            return error;
        }
        case LANA_HOST_SHARED_REVISION: {
            LanaSharedInformation *shared;
            if (argc != 1u || arguments[0].type != VAL_SHARED_CAPABILITY)
                return LANA_ERR_TYPE;
            shared = lana_shared_capability_information(
                arguments[0].as.capability);
            *out = lana_value_number(
                (double)lana_shared_information_revision(shared));
            return LANA_OK;
        }
        case LANA_HOST_SHARED_IDENTITY: {
            LanaSharedInformation *shared;
            if (argc != 1u || arguments[0].type != VAL_SHARED_CAPABILITY)
                return LANA_ERR_TYPE;
            shared = lana_shared_capability_information(
                arguments[0].as.capability);
            *out = lana_value_number(
                (double)lana_shared_information_identity(shared));
            return LANA_OK;
        }
        case LANA_HOST_SHARED_WAIT: {
            LanaSharedInformation *shared;
            if (argc != 3u || arguments[0].type != VAL_SHARED_CAPABILITY ||
                !nonnegative_integer(&arguments[1]) ||
                !nonnegative_integer(&arguments[2])) return LANA_ERR_TYPE;
            shared = lana_shared_capability_information(
                arguments[0].as.capability);
            return lana_shared_information_wait(vm, shared,
                arguments[0].as.capability, (uint64_t)arguments[1].as.number,
                (uint64_t)arguments[2].as.number, out, NULL);
        }
        case LANA_HOST_INFORMATION_INSPECT: {
            LanaMap *inspection;
            LanaError error;
            const Value *value;
            size_t alternatives = 0u;
            if (argc != 1u) return LANA_ERR_TYPE;
            error = lana_map_new(vm, 10u, &inspection);
            if (error != LANA_OK) return error;
            if (arguments[0].type == VAL_SHARED_CAPABILITY) {
                LanaSharedInformation *shared = lana_shared_capability_information(
                    arguments[0].as.capability);
                if ((error = map_put(vm, inspection, "kind",
                        lana_value_string("shared_information"))) != LANA_OK ||
                    (error = map_put(vm, inspection, "identity",
                        lana_value_number((double)lana_shared_information_identity(shared)))) != LANA_OK ||
                    (error = map_put(vm, inspection, "revision",
                        lana_value_number((double)lana_shared_information_revision(shared)))) != LANA_OK ||
                    (error = map_put(vm, inspection, "can_read",
                        lana_value_bool(lana_shared_capability_allows(arguments[0].as.capability, LANA_CAPABILITY_READ)))) != LANA_OK ||
                    (error = map_put(vm, inspection, "can_observe",
                        lana_value_bool(lana_shared_capability_allows(arguments[0].as.capability, LANA_CAPABILITY_OBSERVE)))) != LANA_OK ||
                    (error = map_put(vm, inspection, "can_admin",
                        lana_value_bool(lana_shared_capability_allows(arguments[0].as.capability, LANA_CAPABILITY_ADMIN)))) != LANA_OK)
                    return error;
                *out = lana_value_map(inspection);
                return LANA_OK;
            }
            value = reactive_value(&arguments[0]);
            if (value->type == VAL_POSSIBILITY)
                alternatives = value->as.possibility->count;
            else if (value->type == VAL_PATH_SET)
                alternatives = value->as.paths->count;
            if ((error = map_put(vm, inspection, "kind",
                    lana_value_string("information_snapshot"))) != LANA_OK ||
                (error = map_put(vm, inspection, "type",
                    lana_value_string(lana_value_type_name(value->type)))) != LANA_OK ||
                (error = map_put(vm, inspection, "revision",
                    lana_value_number(arguments[0].reactive == NULL ? 0.0 :
                        (double)arguments[0].reactive->revision))) != LANA_OK ||
                (error = map_put(vm, inspection, "remaining_alternatives",
                    lana_value_number((double)alternatives))) != LANA_OK ||
                (error = map_put(vm, inspection, "reactive",
                    lana_value_bool(arguments[0].reactive != NULL))) != LANA_OK ||
                (error = map_put(vm, inspection, "sample",
                    lana_value_bool(arguments[0].type == VAL_SAMPLE))) != LANA_OK ||
                (error = map_put(vm, inspection, "approximate",
                    lana_value_bool(arguments[0].derivation != NULL &&
                        arguments[0].derivation->exactness == LANA_EXACTNESS_APPROXIMATE))) != LANA_OK)
                return error;
            if (arguments[0].reactive != NULL) {
                const char *relationship = "exact";
                if (arguments[0].reactive->relationship == LANA_RELATION_SAME_DEPENDENCY)
                    relationship = "same_dependency";
                else if (arguments[0].reactive->relationship == LANA_RELATION_EXPLICIT_JOINT)
                    relationship = "explicit_joint";
                if ((error = map_put(vm, inspection, "dependency_identity",
                        lana_value_number((double)arguments[0].reactive->dependency_id))) != LANA_OK ||
                    (error = map_put(vm, inspection, "relationship",
                        lana_value_string(relationship))) != LANA_OK ||
                    (error = map_put(vm, inspection, "history_count",
                        lana_value_number((double)arguments[0].reactive->history_count))) != LANA_OK ||
                    (error = map_put(vm, inspection, "exactness",
                        lana_value_string(derivation_exactness_name(
                            arguments[0].reactive->exactness)))) != LANA_OK)
                    return error;
            }
            if ((error = map_put(vm, inspection, "planned_effect",
                    lana_value_bool(arguments[0].planned_effect != NULL))) != LANA_OK)
                return error;
            if (arguments[0].derivation != NULL) {
                Value derivation;
                error = lana_vm_derivation(vm, &arguments[0], &derivation);
                if (error != LANA_OK ||
                    (error = map_put(vm, inspection, "derivation", derivation)) != LANA_OK)
                    return error;
            }
            *out = lana_value_map(inspection);
            return LANA_OK;
        }
        default: return LANA_ERR_FORMAT;
    }
}

static LanaError close_task_group(LanaVM *vm, uint64_t group_id) {
    LanaTask *task;
    Value ignored;
    LanaError first_error = LANA_OK;
    for (task = vm->tasks; task != NULL; task = task->next)
        if (task->group_id == group_id && !task->completed) cancel_task(task);
    for (task = vm->tasks; task != NULL; task = task->next) {
        if (task->group_id == group_id && !task->joined) {
            LanaError error = wait_task(vm, task, -1.0, &ignored);
            if (error == LANA_ERR_CANCELLED) memset(&vm->error, 0, sizeof(vm->error));
            else if (error != LANA_OK && first_error == LANA_OK) first_error = error;
        }
    }
    return first_error;
}

static LanaError history_append(LanaVM *vm, LanaHistory *history, LanaStateValue state) {
    LanaStateValue *versions;
    size_t keep_from = 0;
    if (history->policy == LANA_HISTORY_NONE) return LANA_OK;
    if (history->count == history->capacity) {
        size_t capacity = history->capacity == 0 ? 8u : history->capacity * 2u;
        if (capacity < history->capacity || capacity > SIZE_MAX / sizeof(*versions))
            return LANA_ERR_OOM;
        versions = lana_vm_alloc(vm, capacity * sizeof(*versions));
        if (versions == NULL) return LANA_ERR_OOM;
        if (history->count > 0u)
            memcpy(versions, history->versions,
                   history->count * sizeof(*versions));
        history->versions = versions;
        history->capacity = capacity;
    }
    history->versions[history->count++] = state;
    if (history->policy == LANA_HISTORY_LATEST && history->count > (size_t)history->amount)
        keep_from = history->count - (size_t)history->amount;
    else if (history->policy == LANA_HISTORY_DURATION && state.indexes.has_timestamp) {
        double cutoff = state.indexes.timestamp - history->amount;
        while (keep_from < history->count && history->versions[keep_from].indexes.has_timestamp &&
               history->versions[keep_from].indexes.timestamp < cutoff) ++keep_from;
    }
    if (keep_from > 0) {
        memmove(history->versions, history->versions + keep_from,
                (history->count - keep_from) * sizeof(*history->versions));
        history->count -= keep_from;
    }
    return LANA_OK;
}

static LanaError store_state(LanaVM *vm, uint32_t reg, LanaStateValue state) {
    LanaFrame *frame = current_frame(vm);
    LanaError error;
    if (!lana_state_valid(&state.state)) return LANA_ERR_INVALID_STATE;
    frame->registers[reg].type = VAL_STATE;
    frame->registers[reg].as.state = state;
    vm->state_transition_count += 1u;
    error = history_append(vm, &frame->histories[reg], state);
    return error;
}

LanaError lana_vm_state_dist_dirac(LanaVM *vm, const LanaStateValue *state, LanaStateDist **out) {
    LanaStateDist *distribution;
    if (vm == NULL || state == NULL || out == NULL || !lana_state_valid(&state->state))
        return LANA_ERR_INVALID_STATE;
    distribution = lana_vm_alloc(vm, sizeof(*distribution));
    if (distribution == NULL) return LANA_ERR_OOM;
    distribution->kind = LANA_DIST_DIRAC;
    distribution->as.dirac = *state;
    *out = distribution;
    return LANA_OK;
}

static LanaError distribution_from_value(LanaVM *vm, const Value *value, LanaStateDist **out) {
    if (value == NULL || out == NULL) return LANA_ERR_TYPE;
    if (value->type == VAL_STATE)
        return lana_vm_state_dist_dirac(vm, &value->as.state, out);
    if (value->type == VAL_STATE_DIST && value->as.state_dist != NULL) {
        *out = value->as.state_dist;
        return LANA_OK;
    }
    return LANA_ERR_TYPE;
}

LanaError lana_vm_state_dist_append(LanaVM *vm, const Value *left, const Value *right,
                                LanaStateDist **out) {
    LanaStateDist *distribution;
    LanaError error;
    if (vm == NULL || out == NULL) return LANA_ERR_INVALID_DISTRIBUTION;
    distribution = lana_vm_alloc(vm, sizeof(*distribution));
    if (distribution == NULL) return LANA_ERR_OOM;
    distribution->kind = LANA_DIST_APPEND;
    error = distribution_from_value(vm, left, &distribution->as.append.left);
    if (error == LANA_OK)
        error = distribution_from_value(vm, right, &distribution->as.append.right);
    if (error != LANA_OK) return error;
    distribution->as.append.has_cached_parameters = false;
    if (left->type == VAL_STATE && right->type == VAL_STATE) {
        error = lana_state_append_parameters(&left->as.state.state, &right->as.state.state,
                                           &distribution->as.append.p,
                                           &distribution->as.append.m_re,
                                           &distribution->as.append.m_im,
                                           &distribution->as.append.sigma);
        if (error != LANA_OK) return error;
        distribution->as.append.has_cached_parameters = true;
    }
    *out = distribution;
    return LANA_OK;
}

LanaError lana_vm_state_dist_transform(LanaVM *vm, uint32_t transform_id,
                                   LanaStateDist *child, LanaStateDist **out) {
    const LanaTransformSpec *specification = lana_transform_spec(transform_id);
    LanaStateDist *distribution;
    if (vm == NULL || child == NULL || out == NULL) return LANA_ERR_INVALID_DISTRIBUTION;
    if (specification == NULL || !specification->distribution_liftable ||
        specification->exact_expected_probability == NULL)
        return LANA_ERR_UNSUPPORTED_OPERATION;
    distribution = lana_vm_alloc(vm, sizeof(*distribution));
    if (distribution == NULL) return LANA_ERR_OOM;
    distribution->kind = LANA_DIST_TRANSFORM;
    distribution->as.transform.child = child;
    distribution->as.transform.transform_id = transform_id;
    *out = distribution;
    return LANA_OK;
}

static LanaError expected_probability_recursive(const LanaStateDist *distribution,
                                              double *out, size_t depth) {
    double left, right;
    LanaError error;
    if (distribution == NULL || out == NULL || depth > 1024u)
        return LANA_ERR_INVALID_DISTRIBUTION;
    switch (distribution->kind) {
        case LANA_DIST_DIRAC:
            if (!lana_state_valid(&distribution->as.dirac.state))
                return LANA_ERR_INVALID_DISTRIBUTION;
            *out = distribution->as.dirac.state.p;
            return LANA_OK;
        case LANA_DIST_APPEND:
            error = expected_probability_recursive(distribution->as.append.left,
                                                   &left, depth + 1u);
            if (error == LANA_OK)
                error = expected_probability_recursive(distribution->as.append.right,
                                                       &right, depth + 1u);
            if (error != LANA_OK) return error;
            *out = 1.0 - (1.0 - left) * (1.0 - right);
            return isfinite(*out) && *out >= 0.0 && *out <= 1.0
                       ? LANA_OK : LANA_ERR_INVALID_DISTRIBUTION;
        case LANA_DIST_TRANSFORM:
            error = expected_probability_recursive(distribution->as.transform.child,
                                                   &left, depth + 1u);
            if (error != LANA_OK) return error;
            return lana_transform_expected_probability(
                distribution->as.transform.transform_id, left, out);
        default:
            return LANA_ERR_INVALID_DISTRIBUTION;
    }
}

LanaError lana_vm_state_dist_expected_probability(const LanaStateDist *distribution,
                                              double *out) {
    return expected_probability_recursive(distribution, out, 0u);
}

static LanaError consume_sampling_budget(LanaVM *vm) {
    if (atomic_load(&vm->cancelled)) return LANA_ERR_CANCELLED;
    if (vm->instruction_count >= vm->instruction_limit) return LANA_ERR_BUDGET_EXHAUSTED;
    ++vm->instruction_count;
    return LANA_OK;
}

static double uniform_signed(LanaVM *vm) {
    return 2.0 * ((double)lana_vm_random(vm) / 4294967296.0) - 1.0;
}

static LanaError sample_append_kernel(LanaVM *vm, const LanaStateValue *left,
                                    const LanaStateValue *right, LanaStateValue *out) {
    double p, m_re, m_im, sigma;
    LanaError error = lana_state_append_parameters(&left->state, &right->state,
                                               &p, &m_re, &m_im, &sigma);
    if (error != LANA_OK) return LANA_ERR_INVALID_DISTRIBUTION;
    memset(&out->indexes, 0, sizeof(out->indexes));
    if (p == 0.0 || p == 1.0)
        return lana_state_make_complex(p, 0.0, 0.0, &out->state);
    if (sigma == 0.0)
        return lana_state_make_complex(p, m_re, m_im, &out->state);
    for (;;) {
        double x, y, radius_squared, factor, d_re, d_im;
        error = consume_sampling_budget(vm);
        if (error != LANA_OK) return error;
        x = uniform_signed(vm);
        y = uniform_signed(vm);
        radius_squared = x * x + y * y;
        if (radius_squared <= 0.0 || radius_squared >= 1.0) continue;
        factor = sqrt(-2.0 * log(radius_squared) / radius_squared);
        d_re = m_re + sigma * x * factor;
        d_im = m_im + sigma * y * factor;
        if (d_re * d_re + d_im * d_im > 1.0) continue;
        return lana_state_make_complex(p, d_re, d_im, &out->state);
    }
}

static LanaError sample_distribution_recursive(LanaVM *vm, const LanaStateDist *distribution,
                                             LanaStateValue *out, size_t depth) {
    LanaStateValue left, right;
    LanaError error;
    if (distribution == NULL || out == NULL || depth > 1024u)
        return LANA_ERR_INVALID_DISTRIBUTION;
    switch (distribution->kind) {
        case LANA_DIST_DIRAC:
            if (!lana_state_valid(&distribution->as.dirac.state))
                return LANA_ERR_INVALID_DISTRIBUTION;
            *out = distribution->as.dirac;
            return LANA_OK;
        case LANA_DIST_APPEND:
            error = sample_distribution_recursive(vm, distribution->as.append.left,
                                                  &left, depth + 1u);
            if (error == LANA_OK)
                error = sample_distribution_recursive(vm, distribution->as.append.right,
                                                      &right, depth + 1u);
            return error == LANA_OK ? sample_append_kernel(vm, &left, &right, out) : error;
        case LANA_DIST_TRANSFORM:
            error = sample_distribution_recursive(vm, distribution->as.transform.child,
                                                  out, depth + 1u);
            if (error != LANA_OK) return error;
            return lana_transform_apply(distribution->as.transform.transform_id,
                                         &out->state, &out->state);
        default:
            return LANA_ERR_INVALID_DISTRIBUTION;
    }
}

LanaError lana_vm_state_dist_sample(LanaVM *vm, const LanaStateDist *distribution,
                                LanaStateValue *out) {
    if (vm == NULL) return LANA_ERR_INVALID_DISTRIBUTION;
    return sample_distribution_recursive(vm, distribution, out, 0u);
}

static int draw_sample(LanaVM *vm, double p) {
    double draw = (double)lana_vm_random(vm) / 4294967296.0;
    return draw < p ? 1 : 0;
}

static LanaError measure_basis_state(const LanaStateValue *state, uint32_t basis,
                                   double *out) {
    if (state == NULL || out == NULL) return LANA_ERR_INVALID_STATE;
    return lana_state_basis_probability(basis, &state->state, out);
}

static LanaError estimate_basis_probability(LanaVM *vm, const LanaStateDist *distribution,
                                           uint32_t basis, uint32_t samples,
                                           double *out) {
    double total = 0.0;
    uint32_t index;
    if (vm == NULL || distribution == NULL || out == NULL || samples == 0u)
        return LANA_ERR_FORMAT;
    for (index = 0u; index < samples; ++index) {
        LanaStateValue state;
        double probability;
        LanaError error = consume_sampling_budget(vm);
        if (error != LANA_OK) return error;
        error = lana_vm_state_dist_sample(vm, distribution, &state);
        if (error != LANA_OK) return error;
        if (atomic_load(&vm->cancelled)) return LANA_ERR_CANCELLED;
        error = measure_basis_state(&state, basis, &probability);
        if (error != LANA_OK) return error;
        total += probability;
    }
    *out = total / (double)samples;
    return isfinite(*out) && *out >= 0.0 && *out <= 1.0
               ? LANA_OK : LANA_ERR_INVALID_DISTRIBUTION;
}
static LanaError values_equal(const Value *left, const Value *right, bool *out) {
    if (left->type == VAL_STATE_DIST || right->type == VAL_STATE_DIST ||
        left->type == VAL_MAP || right->type == VAL_MAP ||
        left->type == VAL_JOINT_STATE || right->type == VAL_JOINT_STATE ||
        left->type == VAL_POSSIBILITY || right->type == VAL_POSSIBILITY ||
        left->type == VAL_PATH_SET || right->type == VAL_PATH_SET ||
        left->type == VAL_SHARED_CAPABILITY ||
        right->type == VAL_SHARED_CAPABILITY)
        return LANA_ERR_UNSUPPORTED_OPERATION;
    if (left->type != right->type) {
        *out = false;
        return LANA_OK;
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
    return LANA_OK;
}

typedef enum { LANA_PURE_BINARY, LANA_PURE_COMPARE } LanaPureKind;

static LanaError pure_scalar_binary(const Value *left, const Value *right,
                                  LanaPureKind kind, uint32_t operation,
                                  Value *out) {
    bool result = false;
    LanaError error;
    if (kind == LANA_PURE_BINARY) {
        if (left->type != VAL_NUMBER || right->type != VAL_NUMBER) return LANA_ERR_TYPE;
        if (operation == LANA_BINARY_ADD) *out = lana_value_number(left->as.number + right->as.number);
        else if (operation == LANA_BINARY_SUBTRACT) *out = lana_value_number(left->as.number - right->as.number);
        else if (operation == LANA_BINARY_MULTIPLY) *out = lana_value_number(left->as.number * right->as.number);
        else if (operation == LANA_BINARY_DIVIDE && right->as.number != 0.0)
            *out = lana_value_number(left->as.number / right->as.number);
        else return LANA_ERR_TYPE;
        return LANA_OK;
    }
    if (operation == LANA_COMPARE_EQUAL || operation == LANA_COMPARE_NOT_EQUAL) {
        error = values_equal(left, right, &result);
        if (error != LANA_OK) return error;
        if (operation == LANA_COMPARE_NOT_EQUAL) result = !result;
    } else if (left->type == VAL_NUMBER && right->type == VAL_NUMBER) {
        if (operation == LANA_COMPARE_LESS) result = left->as.number < right->as.number;
        else if (operation == LANA_COMPARE_LESS_EQUAL) result = left->as.number <= right->as.number;
        else if (operation == LANA_COMPARE_GREATER) result = left->as.number > right->as.number;
        else if (operation == LANA_COMPARE_GREATER_EQUAL) result = left->as.number >= right->as.number;
        else return LANA_ERR_TYPE;
    } else return LANA_ERR_TYPE;
    *out = lana_value_bool(result); return LANA_OK;
}

static LanaError lift_binary_raw(LanaVM *vm, const Value *left, const Value *right,
                               LanaPureKind kind, uint32_t operation, Value *out) {
    const LanaPathSet *left_paths = left->type == VAL_PATH_SET ? left->as.paths : NULL;
    const LanaPathSet *right_paths = right->type == VAL_PATH_SET ? right->as.paths : NULL;
    const LanaPossibility *left_possibility = left->type == VAL_POSSIBILITY ? left->as.possibility : NULL;
    const LanaPossibility *right_possibility = right->type == VAL_POSSIBILITY ? right->as.possibility : NULL;
    size_t count, index, left_index, right_index;
    LanaError error;
    if (left_paths != NULL || right_paths != NULL) {
        LanaPathSet *paths;
        if (left_paths != NULL && right_paths != NULL &&
            (left_paths->dependency_id != right_paths->dependency_id ||
             left_paths->count != right_paths->count)) return LANA_ERR_UNSUPPORTED_OPERATION;
        count = left_paths != NULL ? left_paths->count : right_paths->count;
        paths = lana_vm_alloc(vm, sizeof(*paths));
        if (paths == NULL) return LANA_ERR_OOM;
        paths->count = count;
        paths->dependency_id = left_paths != NULL
            ? left_paths->dependency_id : right_paths->dependency_id;
        paths->alternatives = lana_vm_alloc(vm, count * sizeof(*paths->alternatives));
        if (paths->alternatives == NULL) return LANA_ERR_OOM;
        for (index = 0; index < count; ++index) {
            const Value *left_value = left_paths == NULL ? left : left_paths->alternatives[index].result;
            const Value *right_value = right_paths == NULL ? right : right_paths->alternatives[index].result;
            paths->alternatives[index].guard = (left_paths != NULL
                ? left_paths : right_paths)->alternatives[index].guard;
            paths->alternatives[index].weight = (left_paths != NULL
                ? left_paths : right_paths)->alternatives[index].weight;
            paths->alternatives[index].result = lana_vm_alloc(vm, sizeof(Value));
            if (paths->alternatives[index].result == NULL) return LANA_ERR_OOM;
            error = lift_binary_raw(vm, left_value, right_value, kind, operation,
                                    paths->alternatives[index].result);
            if (error != LANA_OK) return error;
        }
        *out = lana_value_paths(paths); return LANA_OK;
    }
    if (left_possibility != NULL || right_possibility != NULL) {
        Value *results;
        LanaPossibility *possibility;
        size_t left_count = left_possibility == NULL ? 1u : left_possibility->count;
        size_t right_count = right_possibility == NULL ? 1u : right_possibility->count;
        bool zipped = left_possibility != NULL && right_possibility != NULL &&
            left_possibility->dependency_id == right_possibility->dependency_id &&
            left_count == right_count;
        if (left_possibility != NULL && right_possibility != NULL && !zipped)
            return LANA_ERR_UNSUPPORTED_OPERATION;
        count = zipped ? left_count : left_count * right_count;
        results = calloc(count, sizeof(*results));
        if (results == NULL) return LANA_ERR_OOM;
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
                if (error != LANA_OK) { free(results); return error; }
            }
        }
        error = lana_vm_possibility_build(vm, results, count, &possibility);
        free(results);
        if (error != LANA_OK) return error;
        if (zipped || left_possibility == NULL || right_possibility == NULL)
            possibility->dependency_id = left_possibility != NULL
                ? left_possibility->dependency_id : right_possibility->dependency_id;
        *out = lana_value_possibility(possibility); return LANA_OK;
    }
    return pure_scalar_binary(left, right, kind, operation, out);
}

static LanaError reactive_derived_value(LanaVM *vm, const Value *left,
                                        const Value *right,
                                        LanaReactiveKind kind,
                                        uint32_t operation, Value *out) {
    LanaReactive *node;
    LanaDerivationExactness exactness = LANA_EXACTNESS_EXACT;
    LanaError error;
    if (left->reactive != NULL && right != NULL && right->reactive != NULL &&
        left->reactive->dependency_id != right->reactive->dependency_id)
        return LANA_ERR_UNSUPPORTED_OPERATION;
    node = lana_vm_alloc(vm, sizeof(*node));
    if (node == NULL) return LANA_ERR_OOM;
    memset(node, 0, sizeof(*node));
    node->id = vm->next_reactive_id++;
    node->kind = kind;
    node->operation = operation;
    node->revision = vm->revision;
    node->inputs[0] = left->reactive;
    node->inputs[1] = right == NULL ? NULL : right->reactive;
    if (left->reactive != NULL) {
        node->dependency_id = left->reactive->dependency_id;
        exactness = left->reactive->exactness;
    }
    if (right != NULL && right->reactive != NULL) {
        node->dependency_id = right->reactive->dependency_id;
        if (right->reactive->exactness > exactness)
            exactness = right->reactive->exactness;
    }
    node->relationship = node->inputs[0] != NULL && node->inputs[1] != NULL
        ? LANA_RELATION_SAME_DEPENDENCY : LANA_RELATION_EXACT;
    node->exactness = exactness;
    if (node->inputs[0] == NULL) {
        error = allocate_plain_value(vm, left, &node->constants[0]);
        if (error != LANA_OK) return error;
    }
    if (right != NULL && node->inputs[1] == NULL) {
        error = allocate_plain_value(vm, right, &node->constants[1]);
        if (error != LANA_OK) return error;
    }
    error = allocate_plain_value(vm, out, &node->current);
    if (error != LANA_OK) return error;
    out->reactive = node;
    return LANA_OK;
}

static LanaError lift_binary(LanaVM *vm, const Value *left, const Value *right,
                            LanaPureKind kind, uint32_t operation, Value *out) {
    const Value *left_current = reactive_value(left);
    const Value *right_current = reactive_value(right);
    LanaError error;
    error = lift_binary_raw(vm, left_current, right_current, kind, operation, out);
    if (error != LANA_OK || (left->reactive == NULL && right->reactive == NULL))
        return error;
    return reactive_derived_value(vm, left, right,
        kind == LANA_PURE_COMPARE ? LANA_REACTIVE_COMPARE : LANA_REACTIVE_BINARY,
        operation, out);
}

static LanaError lift_unary_raw(LanaVM *vm, const Value *source,
                               uint32_t operation, Value *out) {
    size_t index;
    LanaError error;
    if (source->type == VAL_PATH_SET) {
        LanaPathSet *paths = lana_vm_alloc(vm, sizeof(*paths));
        if (paths == NULL) return LANA_ERR_OOM;
        paths->count = source->as.paths->count;
        paths->dependency_id = source->as.paths->dependency_id;
        paths->alternatives = lana_vm_alloc(vm, paths->count * sizeof(*paths->alternatives));
        if (paths->alternatives == NULL) return LANA_ERR_OOM;
        for (index = 0; index < paths->count; ++index) {
            paths->alternatives[index] = source->as.paths->alternatives[index];
            paths->alternatives[index].result = lana_vm_alloc(vm, sizeof(Value));
            if (paths->alternatives[index].result == NULL) return LANA_ERR_OOM;
            error = lift_unary_raw(vm, source->as.paths->alternatives[index].result,
                                   operation, paths->alternatives[index].result);
            if (error != LANA_OK) return error;
        }
        *out = lana_value_paths(paths); return LANA_OK;
    }
    if (source->type == VAL_POSSIBILITY) {
        Value *results = calloc(source->as.possibility->count, sizeof(*results));
        LanaPossibility *possibility;
        if (results == NULL) return LANA_ERR_OOM;
        for (index = 0; index < source->as.possibility->count; ++index) {
            error = lift_unary_raw(vm, &source->as.possibility->values[index],
                                   operation, &results[index]);
            if (error != LANA_OK) { free(results); return error; }
        }
        error = lana_vm_possibility_build(vm, results, source->as.possibility->count,
                                        &possibility);
        free(results);
        if (error != LANA_OK) return error;
        possibility->dependency_id = source->as.possibility->dependency_id;
        *out = lana_value_possibility(possibility); return LANA_OK;
    }
    if (source->type == VAL_NUMBER && operation == 0u) {
        *out = lana_value_number(-source->as.number); return LANA_OK;
    }
    if (source->type == VAL_BOOL && operation == 1u) {
        *out = lana_value_bool(!source->as.boolean); return LANA_OK;
    }
    return LANA_ERR_TYPE;
}

static LanaError lift_unary(LanaVM *vm, const Value *source, uint32_t operation,
                           Value *out) {
    LanaError error = lift_unary_raw(vm, reactive_value(source), operation, out);
    if (error != LANA_OK || source->reactive == NULL) return error;
    return reactive_derived_value(vm, source, NULL, LANA_REACTIVE_UNARY,
                                  operation, out);
}

typedef struct {
    LanaReactive **items;
    size_t count;
    size_t capacity;
} LanaReactiveList;

static bool reactive_list_has(const LanaReactiveList *list,
                              const LanaReactive *node) {
    size_t index;
    for (index = 0u; index < list->count; ++index)
        if (list->items[index] == node) return true;
    return false;
}

static bool reactive_list_add(LanaReactiveList *list, LanaReactive *node) {
    LanaReactive **items;
    size_t capacity;
    if (node == NULL || reactive_list_has(list, node)) return true;
    if (!reactive_list_add(list, node->inputs[0]) ||
        !reactive_list_add(list, node->inputs[1])) return false;
    if (list->count == list->capacity) {
        capacity = list->capacity == 0u ? 16u : list->capacity * 2u;
        if (capacity < list->capacity) return false;
        items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) return false;
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = node;
    return true;
}

static bool reactive_collect_value(LanaReactiveList *list, const Value *value) {
    size_t index;
    if (value == NULL) return true;
    if (!reactive_list_add(list, value->reactive)) return false;
    if (value->type == VAL_ARRAY && value->as.array != NULL)
        for (index = 0u; index < value->as.array->count; ++index)
            if (!reactive_collect_value(list, &value->as.array->items[index]))
                return false;
    if (value->type == VAL_MAP && value->as.map != NULL)
        for (index = 0u; index < value->as.map->count; ++index)
            if (!reactive_collect_value(list, value->as.map->entries[index].value))
                return false;
    return true;
}

static ssize_t reactive_list_index(const LanaReactiveList *list,
                                   const LanaReactive *node) {
    size_t index;
    for (index = 0u; index < list->count; ++index)
        if (list->items[index] == node) return (ssize_t)index;
    return -1;
}

static const Value *reactive_staged_input(const LanaReactiveList *list,
                                          Value **staged,
                                          LanaReactive *input,
                                          const Value *constant) {
    ssize_t index;
    if (input == NULL) return constant;
    index = reactive_list_index(list, input);
    if (index >= 0 && staged[index] != NULL) return staged[index];
    return input->current;
}

static LanaError reactive_recompute_transaction(LanaVM *vm, LanaReactive *root,
                                                const Value *replacement) {
    LanaReactiveList list = {0};
    Value **staged = NULL;
    LanaReactiveVersion **histories = NULL;
    bool *affected = NULL;
    size_t frame_index, register_index, index;
    uint64_t revision;
    LanaError error = LANA_OK;
    for (frame_index = 0u; frame_index < vm->frame_count; ++frame_index)
        for (register_index = 0u; register_index < LANA_MAX_REGISTERS;
             ++register_index)
            if (!reactive_collect_value(&list,
                    &vm->frames[frame_index].registers[register_index])) {
                error = LANA_ERR_OOM;
                goto done;
            }
    if (!reactive_collect_value(&list, &vm->result)) {
        error = LANA_ERR_OOM;
        goto done;
    }
    if (!reactive_list_has(&list, root) && !reactive_list_add(&list, root)) {
        error = LANA_ERR_OOM;
        goto done;
    }
    staged = calloc(list.count, sizeof(*staged));
    histories = calloc(list.count, sizeof(*histories));
    affected = calloc(list.count, sizeof(*affected));
    if (staged == NULL || histories == NULL || affected == NULL) {
        error = LANA_ERR_OOM;
        goto done;
    }
    for (index = 0u; index < list.count; ++index) {
        LanaReactive *node = list.items[index];
        const Value *left;
        const Value *right;
        if (node == root) affected[index] = true;
        else {
            ssize_t left_index = reactive_list_index(&list, node->inputs[0]);
            ssize_t right_index = reactive_list_index(&list, node->inputs[1]);
            affected[index] = (left_index >= 0 && affected[left_index]) ||
                              (right_index >= 0 && affected[right_index]);
        }
        if (!affected[index]) continue;
        staged[index] = lana_vm_alloc(vm, sizeof(*staged[index]));
        if (staged[index] == NULL) { error = LANA_ERR_OOM; goto done; }
        if (node == root) {
            error = clone_without_runtime_metadata(vm, replacement, staged[index]);
        } else {
            left = reactive_staged_input(&list, staged, node->inputs[0],
                                         node->constants[0]);
            right = reactive_staged_input(&list, staged, node->inputs[1],
                                          node->constants[1]);
            if (node->kind == LANA_REACTIVE_BINARY)
                error = lift_binary_raw(vm, left, right, LANA_PURE_BINARY,
                                        node->operation, staged[index]);
            else if (node->kind == LANA_REACTIVE_COMPARE)
                error = lift_binary_raw(vm, left, right, LANA_PURE_COMPARE,
                                        node->operation, staged[index]);
            else if (node->kind == LANA_REACTIVE_UNARY)
                error = lift_unary_raw(vm, left, node->operation, staged[index]);
            else
                error = LANA_ERR_UNSUPPORTED_OPERATION;
        }
        if (error != LANA_OK) goto done;
    }
    for (index = 0u; index < list.count; ++index) {
        LanaReactive *node = list.items[index];
        if (!affected[index]) continue;
        histories[index] = lana_vm_alloc(vm,
            (node->history_count + 1u) * sizeof(*histories[index]));
        if (histories[index] == NULL) { error = LANA_ERR_OOM; goto done; }
        if (node->history_count > 0u)
            memcpy(histories[index], node->history,
                   node->history_count * sizeof(*histories[index]));
        histories[index][node->history_count].revision = node->revision;
        histories[index][node->history_count].value = node->current;
    }
    revision = vm->revision + 1u;
    for (index = 0u; index < list.count; ++index) {
        LanaReactive *node = list.items[index];
        if (!affected[index]) continue;
        node->history = histories[index];
        ++node->history_count;
        node->current = staged[index];
        node->revision = revision;
    }
    vm->revision = revision;
done:
    free(affected);
    free(histories);
    free(staged);
    free(list.items);
    return error;
}

static void trace_instruction(LanaVM *vm, size_t ip) {
    if (vm->task_id != 0u) (void)printf("[task %llu] ", (unsigned long long)vm->task_id);
    lana_disassemble_instruction(vm->chunk, ip, stdout);
}

LanaError lana_vm_run(LanaVM *vm) {
    LanaError verify;
    if (vm == NULL || vm->chunk == NULL) return LANA_ERR_FORMAT;
    verify = lana_chunk_verify(vm->chunk, &vm->error);
    if (verify != LANA_OK) return verify;
    while (vm->running) {
        LanaFrame *frame;
        const LanaInstruction *ins;
        size_t instruction_ip;
        LanaError error = LANA_OK;
        const char *error_message = NULL;
        if (!vm_gc_safepoint(vm))
            return vm_fail(vm, LANA_ERR_OOM, vm->ip, NULL,
                           "garbage collection failed");
        if (atomic_load(&vm->cancelled))
            return vm_fail(vm, LANA_ERR_CANCELLED, vm->ip, NULL, "task cancelled");
        if (vm->instruction_count++ >= vm->instruction_limit)
            return vm_fail(vm, LANA_ERR_LIMIT, vm->ip, NULL, "instruction limit exceeded");
        if (vm->ip >= vm->chunk->code_count)
            return vm_fail(vm, LANA_ERR_JUMP, vm->ip, NULL, "instruction pointer is out of range");
        instruction_ip = vm->ip;
        ins = &vm->chunk->code[vm->ip++];
        ++vm->opcode_counts[ins->opcode];
        frame = current_frame(vm);
        if (vm->debug_hook != NULL &&
            (vm->debug_step || (vm->debug_break_line != 0u &&
                                ins->line == vm->debug_break_line))) {
            vm->debug_step = false;
            if (!vm->debug_hook(vm, instruction_ip, ins->line,
                                vm->debug_context))
                return vm_fail(vm, LANA_ERR_CANCELLED, instruction_ip, ins,
                               "debugger stopped execution");
        }
        if (vm->trace) trace_instruction(vm, instruction_ip);
        switch ((OpCode)ins->opcode) {
            case OP_NOP: break;
            case OP_LOAD_CONST: frame->registers[ins->a] = vm->chunk->constants[ins->imm]; break;
            case OP_MOVE:
                frame->registers[ins->a] = frame->registers[ins->b];
                error = clone_history(vm, &frame->histories[ins->b],
                                      &frame->histories[ins->a]);
                break;
            case OP_STATE_NEW: {
                const Value *p = &vm->chunk->constants[ins->b];
                const Value *d_re = &vm->chunk->constants[ins->c];
                const Value *d_im = &vm->chunk->constants[ins->imm];
                LanaState state;
                if (p->type != VAL_NUMBER || d_re->type != VAL_NUMBER ||
                    d_im->type != VAL_NUMBER) error = LANA_ERR_TYPE;
                else error = lana_state_make_complex(p->as.number, d_re->as.number,
                                                   d_im->as.number, &state);
                if (error == LANA_OK)
                    error = store_state(vm, ins->a, lana_value_state(state).as.state);
                break;
            }
            case OP_STATE_BUILD: {
                LanaState state;
                if (frame->registers[ins->a].type != VAL_NUMBER ||
                    frame->registers[ins->b].type != VAL_NUMBER ||
                    frame->registers[ins->c].type != VAL_NUMBER) error = LANA_ERR_TYPE;
                else error = lana_state_make_complex(frame->registers[ins->a].as.number,
                                                   frame->registers[ins->b].as.number,
                                                   frame->registers[ins->c].as.number,
                                                   &state);
                if (error == LANA_OK)
                    error = store_state(vm, ins->imm, lana_value_state(state).as.state);
                break;
            }
            case OP_TRANSFORM: {
                const Value *source = &frame->registers[ins->b];
                if (source->type == VAL_STATE) {
                    LanaStateValue transformed = source->as.state;
                    error = lana_transform_apply(ins->c, &source->as.state.state,
                                                  &transformed.state);
                    if (error == LANA_OK) error = store_state(vm, ins->a, transformed);
                } else if (source->type == VAL_STATE_DIST) {
                    LanaStateDist *distribution;
                    error = lana_vm_state_dist_transform(vm, ins->c,
                                                       source->as.state_dist,
                                                       &distribution);
                    if (error == LANA_OK)
                        frame->registers[ins->a] = lana_value_state_dist(distribution);
                } else error = LANA_ERR_TYPE;
                break;
            }
            case OP_APPEND: {
                LanaStateDist *distribution;
                error = lana_vm_state_dist_append(vm, &frame->registers[ins->a],
                                                &frame->registers[ins->b],
                                                &distribution);
                if (error == LANA_OK)
                    frame->registers[ins->c] = lana_value_state_dist(distribution);
                break;
            }
            case OP_JOINT_BUILD: {
                const Value *descriptor;
                LanaJointState *joint;
                /* The descriptor operand is a constant index, not a register. */
                if (ins->imm >= vm->chunk->constant_count ||
                    vm->chunk->constants[ins->imm].type != VAL_STRING) error = LANA_ERR_TYPE;
                else {
                    descriptor = &vm->chunk->constants[ins->imm];
                    error = lana_vm_joint_build(vm, &frame->registers[ins->b], ins->c,
                                              descriptor->as.string, &joint);
                }
                if (error == LANA_OK) {
                    const Value *inputs[] = {&frame->registers[ins->b]};
                    frame->registers[ins->a] = (Value){.type = VAL_JOINT_STATE, .as.joint = joint};
                    error = attach_derivation(vm, &frame->registers[ins->a],
                        LANA_DERIVATION_OPERATION, "joint_build", inputs, 1u, "",
                        ins->line, LANA_EXACTNESS_EXACT, descriptor->as.string);
                }
                break;
            }
            case OP_JOINT_PROJECT: {
                LanaJointState *joint;
                const Value *source = &frame->registers[ins->a];
                if (source->type != VAL_JOINT_STATE || ins->c >= vm->chunk->constant_count ||
                    vm->chunk->constants[ins->c].type != VAL_STRING) error = LANA_ERR_TYPE;
                else error = lana_vm_joint_project(vm, source->as.joint,
                                                 vm->chunk->constants[ins->c].as.string, &joint);
                if (error == LANA_OK) {
                    const Value *inputs[] = {source};
                    frame->registers[ins->b] = (Value){.type = VAL_JOINT_STATE, .as.joint = joint};
                    error = attach_derivation(vm, &frame->registers[ins->b],
                        LANA_DERIVATION_OPERATION, "project", inputs, 1u, "",
                        ins->line, LANA_EXACTNESS_EXACT,
                        vm->chunk->constants[ins->c].as.string);
                }
                break;
            }
            case OP_JOINT_CONDITION: {
                LanaJointState *joint;
                const Value *source = &frame->registers[ins->a];
                if (source->type != VAL_JOINT_STATE || ins->c >= vm->chunk->constant_count ||
                    vm->chunk->constants[ins->c].type != VAL_STRING || ins->imm >= LANA_MAX_REGISTERS)
                    error = LANA_ERR_TYPE;
                else error = lana_vm_joint_condition(vm, source->as.joint,
                                                    vm->chunk->constants[ins->c].as.string,
                                                    &frame->registers[ins->imm], &joint);
                if (error == LANA_OK) {
                    const Value *inputs[] = {source, &frame->registers[ins->imm]};
                    frame->registers[ins->b] = (Value){.type = VAL_JOINT_STATE, .as.joint = joint};
                    error = attach_derivation(vm, &frame->registers[ins->b],
                        LANA_DERIVATION_OPERATION, "condition", inputs, 2u, "",
                        ins->line, LANA_EXACTNESS_EXACT,
                        vm->chunk->constants[ins->c].as.string);
                }
                break;
            }
            case OP_JOINT_SAMPLE: {
                if (frame->registers[ins->a].type != VAL_JOINT_STATE) error = LANA_ERR_TYPE;
                else error = lana_vm_joint_sample(vm, frame->registers[ins->a].as.joint,
                                                &frame->registers[ins->b]);
                if (error == LANA_OK) {
                    const Value *inputs[] = {&frame->registers[ins->a]};
                    error = attach_derivation(vm, &frame->registers[ins->b],
                        LANA_DERIVATION_SAMPLE, "joint_sample", inputs, 1u, "",
                        ins->line, LANA_EXACTNESS_SAMPLE, "seeded_rng");
                }
                break;
            }
            case OP_RESOLVE:
                error = lana_vm_information_resolve(vm, &frame->registers[ins->a],
                                                  &frame->registers[ins->b]);
                if (error == LANA_OK) {
                    const Value *inputs[] = {&frame->registers[ins->a]};
                    error = attach_derivation(vm, &frame->registers[ins->b],
                        LANA_DERIVATION_RESOLUTION, "resolve", inputs, 1u, "",
                        ins->line, LANA_EXACTNESS_EXACT, "singleton");
                }
                break;
            case OP_JOINT_BUILD_FINITE: {
                LanaJointState *joint;
                if (ins->c >= vm->chunk->constant_count ||
                    vm->chunk->constants[ins->c].type != VAL_STRING)
                    error = LANA_ERR_TYPE;
                else error = joint_build_finite_array(
                    vm, &frame->registers[ins->a],
                    vm->chunk->constants[ins->c].as.string, &joint);
                if (error == LANA_OK)
                    frame->registers[ins->b] = (Value){.type = VAL_JOINT_STATE,
                                                       .as.joint = joint};
                if (error == LANA_OK) {
                    const Value *inputs[] = {&frame->registers[ins->a]};
                    error = attach_derivation(vm, &frame->registers[ins->b],
                        LANA_DERIVATION_OPERATION, "joint_build_finite", inputs,
                        1u, "", ins->line, LANA_EXACTNESS_EXACT,
                        vm->chunk->constants[ins->c].as.string);
                }
                break;
            }
            case OP_JOINT_RENAME: {
                LanaJointState *joint;
                if (frame->registers[ins->a].type != VAL_JOINT_STATE ||
                    ins->c >= vm->chunk->constant_count ||
                    ins->imm >= vm->chunk->constant_count ||
                    vm->chunk->constants[ins->c].type != VAL_STRING ||
                    vm->chunk->constants[ins->imm].type != VAL_STRING)
                    error = LANA_ERR_TYPE;
                else error = lana_vm_joint_rename(
                    vm, frame->registers[ins->a].as.joint,
                    vm->chunk->constants[ins->c].as.string,
                    vm->chunk->constants[ins->imm].as.string, &joint);
                if (error == LANA_OK)
                    frame->registers[ins->b] = (Value){.type = VAL_JOINT_STATE,
                                                       .as.joint = joint};
                if (error == LANA_OK) {
                    const Value *inputs[] = {&frame->registers[ins->a]};
                    error = attach_derivation(vm, &frame->registers[ins->b],
                        LANA_DERIVATION_OPERATION, "rename", inputs, 1u, "",
                        ins->line, LANA_EXACTNESS_EXACT,
                        vm->chunk->constants[ins->imm].as.string);
                }
                break;
            }
            case OP_POSSIBILITY_BUILD: {
                LanaPossibility *possibility;
                const Value *source = &frame->registers[ins->a];
                if (source->type != VAL_ARRAY || source->as.array == NULL)
                    error = LANA_ERR_TYPE;
                else error = lana_vm_possibility_build(vm, source->as.array->items,
                                                      source->as.array->count,
                                                      &possibility);
                if (error == LANA_OK)
                    frame->registers[ins->b] = lana_value_possibility(possibility);
                if (error == LANA_OK) {
                    const Value *inputs[] = {source};
                    error = attach_derivation(vm, &frame->registers[ins->b],
                        LANA_DERIVATION_OPERATION, "possibility", inputs, 1u, "",
                        ins->line, LANA_EXACTNESS_EXACT, "equipossible_support");
                }
                break;
            }
            case OP_PATH_SPLIT:
                error = path_split(vm, &frame->registers[ins->a], ins->imm);
                break;
            case OP_PATH_JOIN:
                error = path_join(vm, ins->line);
                break;
            case OP_OBSERVE: {
                LanaJointState *joint;
                const Value *source = &frame->registers[ins->a];
                if (vm->active_path_count > 1u) { error = LANA_ERR_UNSUPPORTED_OPERATION; break; }
                if (ins->c >= vm->chunk->constant_count ||
                    vm->chunk->constants[ins->c].type != VAL_STRING)
                    error = LANA_ERR_TYPE;
                else if (source->reactive != NULL)
                    error = lana_vm_reactive_observe(
                        vm, source, &frame->registers[ins->imm],
                        &frame->registers[ins->b]);
                else if (source->type != VAL_JOINT_STATE)
                    error = LANA_ERR_TYPE;
                else
                    error = lana_vm_joint_observe(
                        vm, source->as.joint,
                        vm->chunk->constants[ins->c].as.string,
                        &frame->registers[ins->imm], &joint);
                if (error == LANA_OK) {
                    const Value *inputs[] = {source, &frame->registers[ins->imm]};
                    if (source->reactive == NULL)
                        frame->registers[ins->b] = (Value){.type = VAL_JOINT_STATE,
                                                           .as.joint = joint};
                    error = attach_derivation(vm, &frame->registers[ins->b],
                        LANA_DERIVATION_OBSERVATION, "observe", inputs, 2u, "",
                        ins->line, LANA_EXACTNESS_EXACT,
                        vm->chunk->constants[ins->c].as.string);
                }
                break;
            }
            case OP_INFO_SAMPLE:
                if (vm->active_path_count > 1u) error = LANA_ERR_UNSUPPORTED_OPERATION;
                else error = lana_vm_information_sample(vm, &frame->registers[ins->a],
                                                       &frame->registers[ins->b]);
                if (error == LANA_OK) {
                    const Value *inputs[] = {&frame->registers[ins->a]};
                    error = attach_derivation(vm, &frame->registers[ins->b],
                        LANA_DERIVATION_SAMPLE, "sample", inputs, 1u, "",
                        ins->line, LANA_EXACTNESS_SAMPLE, "seeded_rng");
                }
                break;
            case OP_EVIDENCE:
            case OP_ASSUME:
                error = lana_vm_provenance_root(
                    vm, &frame->registers[ins->a],
                    vm->chunk->constants[ins->c].as.string, ins->line,
                    ins->opcode == OP_ASSUME, &frame->registers[ins->b]);
                break;
            case OP_DERIVATION:
                error = lana_vm_derivation(vm, &frame->registers[ins->a],
                                         &frame->registers[ins->b]);
                break;
            case OP_EXPLAIN:
                error = lana_vm_explain(vm, &frame->registers[ins->a],
                                      &frame->registers[ins->b]);
                break;
            case OP_MEASURE: {
                const Value *source = &frame->registers[ins->a];
                double probability;
                if (source->type == VAL_STATE) probability = source->as.state.state.p;
                else if (source->type == VAL_STATE_DIST)
                    error = lana_vm_state_dist_expected_probability(source->as.state_dist,
                                                                 &probability);
                else error = LANA_ERR_TYPE;
                if (error == LANA_OK && ins->c == LANA_MEASURE_PROBABILITY)
                    frame->registers[ins->b] = lana_value_number(probability);
                else if (error == LANA_OK && ins->c == LANA_MEASURE_DISTRIBUTION)
                    frame->registers[ins->b] = lana_value_distribution(1.0 - probability,
                                                                     probability);
                else if (error == LANA_OK && ins->c == LANA_MEASURE_SAMPLE)
                    frame->registers[ins->b] = lana_value_sample(draw_sample(vm, probability));
                else if (error == LANA_OK) error = LANA_ERR_MEASURE;
                break;
            }
            case OP_MEASURE_BASIS: {
                const Value *source = &frame->registers[ins->a];
                double probability;
                if (source->type == VAL_STATE) {
                    error = measure_basis_state(&source->as.state, ins->c, &probability);
                    if (error == LANA_OK && ins->imm == LANA_MEASURE_PROBABILITY)
                        frame->registers[ins->b] = lana_value_number(probability);
                    else if (error == LANA_OK && ins->imm == LANA_MEASURE_DISTRIBUTION)
                        frame->registers[ins->b] = lana_value_distribution(1.0 - probability,
                                                                           probability);
                    else if (error == LANA_OK && ins->imm == LANA_MEASURE_SAMPLE)
                        frame->registers[ins->b] = lana_value_sample(draw_sample(vm, probability));
                    else if (error == LANA_OK) error = LANA_ERR_MEASURE;
                } else if (source->type == VAL_STATE_DIST) {
                    LanaStateValue state;
                    if (ins->imm != LANA_MEASURE_SAMPLE) {
                        error = LANA_ERR_UNSUPPORTED_EXACT_MEASUREMENT;
                    } else {
                        error = lana_vm_state_dist_sample(vm, source->as.state_dist, &state);
                        if (error == LANA_OK)
                            error = measure_basis_state(&state, ins->c, &probability);
                        if (error == LANA_OK)
                            frame->registers[ins->b] = lana_value_sample(draw_sample(vm, probability));
                    }
                } else {
                    error = LANA_ERR_TYPE;
                }
                break;
            }
            case OP_ESTIMATE_MEASURE_PROBABILITY:
            case OP_ESTIMATE_MEASURE_DISTRIBUTION: {
                const Value *source = &frame->registers[ins->a];
                double probability;
                if (source->type != VAL_STATE_DIST) {
                    error = LANA_ERR_TYPE;
                } else {
                    error = estimate_basis_probability(vm, source->as.state_dist,
                                                       ins->c, ins->imm, &probability);
                    if (error == LANA_OK && ins->opcode == OP_ESTIMATE_MEASURE_PROBABILITY)
                        frame->registers[ins->b] = lana_value_number(probability);
                    else if (error == LANA_OK)
                        frame->registers[ins->b] = lana_value_distribution(1.0 - probability,
                                                                           probability);
                }
                if (error == LANA_OK) {
                    const Value *inputs[] = {source};
                    error = attach_derivation(vm, &frame->registers[ins->b],
                        LANA_DERIVATION_APPROXIMATION, "estimate_measure", inputs,
                        1u, "", ins->line, LANA_EXACTNESS_APPROXIMATE,
                        "explicit_sample_count");
                }
                break;
            }
            case OP_SAMPLE_STATE_DIST: {
                LanaStateValue state;
                if (frame->registers[ins->a].type != VAL_STATE_DIST)
                    error = LANA_ERR_TYPE;
                else error = lana_vm_state_dist_sample(vm,
                                                     frame->registers[ins->a].as.state_dist,
                                                     &state);
                if (error == LANA_OK) error = store_state(vm, ins->b, state);
                break;
            }
            case OP_GET_FIELD: {
                const Value *source = &frame->registers[ins->a];
                if (source->type == VAL_STATE && ins->c <= 2u) {
                    double field = ins->c == 0u ? source->as.state.state.p :
                                   ins->c == 1u ? source->as.state.state.d_re :
                                                 source->as.state.state.d_im;
                    frame->registers[ins->b] = lana_value_number(field);
                }
                else if (source->type == VAL_DISTRIBUTION && ins->c <= 1u)
                    frame->registers[ins->b] = lana_value_number(ins->c == 0u ? source->as.distribution.p0 : source->as.distribution.p1);
                else error = LANA_ERR_TYPE;
                break;
            }
            case OP_GET_INDEX: {
                const LanaIndexes *indexes;
                if (frame->registers[ins->a].type != VAL_STATE) { error = LANA_ERR_TYPE; break; }
                indexes = &frame->registers[ins->a].as.state.indexes;
                if (ins->c == 0u && indexes->has_timestamp) frame->registers[ins->b] = lana_value_number(indexes->timestamp);
                else if (ins->c == 1u && indexes->has_source) frame->registers[ins->b] = lana_value_string(indexes->source);
                else if (ins->c == 2u && indexes->has_weight) frame->registers[ins->b] = lana_value_number(indexes->weight);
                else if (ins->c == 3u && indexes->has_confidence) frame->registers[ins->b] = lana_value_number(indexes->confidence);
                else frame->registers[ins->b] = lana_value_null();
                break;
            }
            case OP_SET_INDEX: {
                LanaStateValue state;
                const Value *source = &frame->registers[ins->c];
                if (frame->registers[ins->a].type != VAL_STATE) { error = LANA_ERR_TYPE; break; }
                state = frame->registers[ins->a].as.state;
                if (ins->b == 0u && source->type == VAL_NUMBER) { state.indexes.has_timestamp = true; state.indexes.timestamp = source->as.number; }
                else if (ins->b == 1u && source->type == VAL_STRING) { state.indexes.has_source = true; state.indexes.source = source->as.string; }
                else if (ins->b == 2u && source->type == VAL_NUMBER && source->as.number >= 0.0) { state.indexes.has_weight = true; state.indexes.weight = source->as.number; }
                else if (ins->b == 3u && source->type == VAL_NUMBER && source->as.number >= 0.0 && source->as.number <= 1.0) { state.indexes.has_confidence = true; state.indexes.confidence = source->as.number; }
                else { error = LANA_ERR_TYPE; break; }
                error = store_state(vm, ins->a, state); break;
            }
            case OP_HISTORY_CONFIG: {
                LanaHistory *history = &frame->histories[ins->a];
                if (frame->registers[ins->a].type != VAL_STATE || frame->registers[ins->b].type != VAL_NUMBER ||
                    ins->c > LANA_HISTORY_DURATION || frame->registers[ins->b].as.number <= 0.0) { error = LANA_ERR_HISTORY; break; }
                history->policy = (LanaHistoryPolicy)ins->c; history->amount = frame->registers[ins->b].as.number;
                error = history_append(vm, history, frame->registers[ins->a].as.state); break;
            }
            case OP_PREVIOUS: case OP_CHANGE: case OP_VELOCITY: {
                LanaHistory *history = &frame->histories[ins->a];
                LanaStateValue *current, *previous;
                if (history->count < 2u) { error = LANA_ERR_HISTORY; break; }
                current = &history->versions[history->count - 1u]; previous = &history->versions[history->count - 2u];
                if (ins->opcode == OP_PREVIOUS) frame->registers[ins->b] = (Value){.type = VAL_STATE, .as.state = *previous};
                else if (ins->opcode == OP_CHANGE) frame->registers[ins->b] = lana_value_number(current->state.p - previous->state.p);
                else if (!current->indexes.has_timestamp || !previous->indexes.has_timestamp || current->indexes.timestamp <= previous->indexes.timestamp) error = LANA_ERR_HISTORY;
                else frame->registers[ins->b] = lana_value_number((current->state.p - previous->state.p) / (current->indexes.timestamp - previous->indexes.timestamp));
                break;
            }
            case OP_BINARY: {
                const Value *left = &frame->registers[ins->a], *right = &frame->registers[ins->b];
                error = lift_binary(vm, left, right, LANA_PURE_BINARY, ins->imm,
                                    &frame->registers[ins->c]);
                if (error == LANA_OK && (left->derivation != NULL || right->derivation != NULL)) {
                    const Value *inputs[] = {left, right};
                    error = attach_derivation(vm, &frame->registers[ins->c],
                        LANA_DERIVATION_OPERATION, "binary", inputs, 2u, "",
                        ins->line, LANA_EXACTNESS_EXACT, "pure");
                }
                break;
            }
            case OP_UNARY: {
                const Value *source = &frame->registers[ins->a];
                error = lift_unary(vm, &frame->registers[ins->a], ins->imm,
                                   &frame->registers[ins->b]);
                if (error == LANA_OK && source->derivation != NULL) {
                    const Value *inputs[] = {source};
                    error = attach_derivation(vm, &frame->registers[ins->b],
                        LANA_DERIVATION_OPERATION, "unary", inputs, 1u, "",
                        ins->line, LANA_EXACTNESS_EXACT, "pure");
                }
                break;
            }
            case OP_COMPARE: {
                const Value *left = &frame->registers[ins->a], *right = &frame->registers[ins->b];
                error = lift_binary(vm, left, right, LANA_PURE_COMPARE, ins->imm,
                                    &frame->registers[ins->c]);
                if (error == LANA_OK && (left->derivation != NULL || right->derivation != NULL)) {
                    const Value *inputs[] = {left, right};
                    error = attach_derivation(vm, &frame->registers[ins->c],
                        LANA_DERIVATION_OPERATION, "compare", inputs, 2u, "",
                        ins->line, LANA_EXACTNESS_EXACT, "pure");
                }
                break;
            }
            case OP_JUMP: vm->ip = ins->imm; break;
            case OP_JUMP_IF_TRUE: case OP_JUMP_IF_FALSE:
                if (frame->registers[ins->a].type != VAL_BOOL) error = LANA_ERR_TYPE;
                else if ((ins->opcode == OP_JUMP_IF_TRUE && frame->registers[ins->a].as.boolean) ||
                         (ins->opcode == OP_JUMP_IF_FALSE && !frame->registers[ins->a].as.boolean)) vm->ip = ins->imm;
                break;
            case OP_ARRAY_NEW: {
                LanaArray *array = lana_vm_alloc(vm, sizeof(*array));
                if (array == NULL) { error = LANA_ERR_OOM; break; }
                array->count = ins->c; array->capacity = ins->c; array->items = lana_vm_alloc(vm, ins->c * sizeof(*array->items));
                if (array->items == NULL && ins->c > 0u) { error = LANA_ERR_OOM; break; }
                if (ins->c > 0u) memcpy(array->items, &frame->registers[ins->b], ins->c * sizeof(*array->items));
                frame->registers[ins->a].type = VAL_ARRAY; frame->registers[ins->a].as.array = array; break;
            }
            case OP_ARRAY_GET: case OP_ARRAY_SET: {
                Value *array_value = &frame->registers[ins->a];
                const Value *index_value = &frame->registers[ins->b];
                size_t index;
                if (array_value->type != VAL_ARRAY || index_value->type != VAL_NUMBER || index_value->as.number < 0.0 || floor(index_value->as.number) != index_value->as.number) { error = LANA_ERR_TYPE; break; }
                index = (size_t)index_value->as.number;
                if (index >= array_value->as.array->count) { error = LANA_ERR_LIMIT; break; }
                if (ins->opcode == OP_ARRAY_GET) frame->registers[ins->c] = array_value->as.array->items[index];
                else array_value->as.array->items[index] = frame->registers[ins->c];
                break;
            }
            case OP_CALL: {
                LanaFrame *callee;
                const LanaFunction *function = &vm->chunk->functions[ins->b];
                size_t index;
                if (ins->imm != function->arity) { error = LANA_ERR_TYPE; break; }
                if (vm->frame_count >= LANA_MAX_CALL_FRAMES) { error = LANA_ERR_LIMIT; break; }
                callee = &vm->frames[vm->frame_count++];
                for (index = 0; index < function->register_count; ++index) {
                    callee->registers[index] = lana_value_null();
                    memset(&callee->histories[index], 0, sizeof(callee->histories[index]));
                }
                callee->return_ip = vm->ip; callee->return_register = ins->a; callee->function = ins->b;
                for (index = 0; index < ins->imm; ++index) {
                    callee->registers[index] = frame->registers[ins->c + index];
                    error = clone_history(vm, &frame->histories[ins->c + index],
                                          &callee->histories[index]);
                    if (error != LANA_OK) break;
                }
                if (error != LANA_OK) break;
                vm->ip = function->entry; break;
            }
            case OP_FORK: {
                LanaTask *task;
                size_t argument;
                if (vm->active_path_count > 1u) { error = LANA_ERR_UNSUPPORTED_OPERATION; break; }
                for (argument = 0; argument < ins->imm; ++argument)
                    if (value_is_unresolved(&frame->registers[ins->c + argument])) {
                        error = LANA_ERR_UNRESOLVED_VALUE; break;
                    }
                if (error != LANA_OK) break;
                error = start_task(vm, ins->b, &frame->registers[ins->c],
                                   &frame->histories[ins->c], ins->imm, &task);
                if (error == LANA_OK) {
                    frame->registers[ins->a].type = VAL_TASK;
                    frame->registers[ins->a].as.task = task;
                    if (vm->trace) (void)printf("  forked task %llu\n", (unsigned long long)task->id);
                }
                break;
            }
            case OP_JOIN: {
                Value *task_value = &frame->registers[ins->a];
                if (task_value->type != VAL_TASK) error = LANA_ERR_TYPE;
                else error = wait_task(vm, task_value->as.task, -1.0, &frame->registers[ins->b]);
                if (error == LANA_OK) {
                    Value joined = frame->registers[ins->b];
                    const Value *inputs[] = {&joined};
                    error = attach_derivation(vm, &frame->registers[ins->b],
                        LANA_DERIVATION_OPERATION, "task_join", inputs, 1u, "",
                        ins->line, LANA_EXACTNESS_EXACT, "joined_task_result");
                }
                break;
            }
            case OP_JOIN_TIMEOUT: {
                Value *task_value = &frame->registers[ins->a];
                Value *timeout = &frame->registers[ins->b];
                if (task_value->type != VAL_TASK || timeout->type != VAL_NUMBER || timeout->as.number < 0.0)
                    error = LANA_ERR_TYPE;
                else error = wait_task(vm, task_value->as.task, timeout->as.number,
                                       &frame->registers[ins->c]);
                if (error == LANA_OK) {
                    Value joined = frame->registers[ins->c];
                    const Value *inputs[] = {&joined};
                    error = attach_derivation(vm, &frame->registers[ins->c],
                        LANA_DERIVATION_OPERATION, "task_join_timeout", inputs, 1u,
                        "", ins->line, LANA_EXACTNESS_EXACT, "joined_task_result");
                }
                break;
            }
            case OP_JOIN_ALL: {
                Value *tasks = &frame->registers[ins->a];
                LanaArray *results;
                size_t index;
                if (tasks->type != VAL_ARRAY) { error = LANA_ERR_TYPE; break; }
                results = lana_vm_alloc(vm, sizeof(*results));
                if (results == NULL) { error = LANA_ERR_OOM; break; }
                results->count = tasks->as.array->count;
                results->capacity = results->count;
                results->items = lana_vm_alloc(vm, results->count * sizeof(*results->items));
                if (results->items == NULL && results->count > 0u) { error = LANA_ERR_OOM; break; }
                for (index = 0; index < results->count && error == LANA_OK; ++index) {
                    Value *task_value = &tasks->as.array->items[index];
                    if (task_value->type != VAL_TASK) error = LANA_ERR_TYPE;
                    else error = wait_task(vm, task_value->as.task, -1.0, &results->items[index]);
                }
                if (error == LANA_OK) {
                    const Value **inputs = results->count == 0u ? NULL :
                        malloc(results->count * sizeof(*inputs));
                    if (results->count > 0u && inputs == NULL) {
                        error = LANA_ERR_OOM;
                        break;
                    }
                    for (index = 0; index < results->count; ++index)
                        inputs[index] = &results->items[index];
                    frame->registers[ins->b].type = VAL_ARRAY;
                    frame->registers[ins->b].as.array = results;
                    error = attach_derivation(vm, &frame->registers[ins->b],
                        LANA_DERIVATION_OPERATION, "task_join_all", inputs,
                        results->count, "", ins->line, LANA_EXACTNESS_EXACT,
                        "joined_task_results");
                    free(inputs);
                }
                break;
            }
            case OP_CANCEL:
                if (frame->registers[ins->a].type != VAL_TASK) error = LANA_ERR_TYPE;
                else cancel_task(frame->registers[ins->a].as.task);
                break;
            case OP_TASKGROUP_ENTER:
                if (vm->group_depth >= LANA_MAX_CALL_FRAMES) error = LANA_ERR_LIMIT;
                else {
                    vm->group_stack[vm->group_depth++] = vm->current_group_id;
                    vm->current_group_id = vm->next_group_id++;
                }
                break;
            case OP_TASKGROUP_EXIT: {
                uint64_t group_id = vm->current_group_id;
                if (vm->group_depth == 0u) error = LANA_ERR_TASK;
                else {
                    vm->current_group_id = vm->group_stack[--vm->group_depth];
                    error = close_task_group(vm, group_id);
                }
                break;
            }
            case OP_HOST_CALL: {
                size_t argument;
                bool accepts_unresolved =
                    ins->b == LANA_HOST_MAP_NEW ||
                    ins->b == LANA_HOST_MAP_HAS ||
                    ins->b == LANA_HOST_MAP_GET ||
                    ins->b == LANA_HOST_MAP_SET ||
                    ins->b == LANA_HOST_MAP_KEYS ||
                    ins->b == LANA_HOST_INDEX_GET ||
                    ins->b == LANA_HOST_INDEX_SET ||
                    ins->b == LANA_HOST_ARRAY_PUSH ||
                    ins->b == LANA_HOST_ARRAY_LENGTH ||
                    ins->b == LANA_HOST_INFORMATION_NEW ||
                    ins->b == LANA_HOST_CLAIM_NEW ||
                    ins->b == LANA_HOST_CLAIM_VALUE ||
                    ins->b == LANA_HOST_CLAIM_PROPOSITION ||
                    ins->b == LANA_HOST_CLAIM_STATUS ||
                    ins->b == LANA_HOST_PLANNED_EFFECT_NEW ||
                    ins->b == LANA_HOST_PLANNED_EFFECT_EXECUTE ||
                    ins->b == LANA_HOST_PLANNED_EFFECT_STATUS ||
                    ins->b == LANA_HOST_SHARED_INFORMATION ||
                    ins->b == LANA_HOST_SHARED_OBSERVE ||
                    ins->b == LANA_HOST_INFORMATION_INSPECT;
                bool materialize =
                    ins->b == LANA_HOST_WRITE_TEXT ||
                    ins->b == LANA_HOST_JSON_STRINGIFY ||
                    ins->b == LANA_HOST_CSV_WRITE ||
                    ins->b == LANA_HOST_ASSERT;
                Value *arguments = &frame->registers[ins->c];
                if (vm->active_path_count > 1u) { error = LANA_ERR_UNSUPPORTED_OPERATION; break; }
                for (argument = 0; argument < ins->imm; ++argument)
                    if (!accepts_unresolved &&
                        value_is_unresolved(&frame->registers[ins->c + argument])) {
                        error = LANA_ERR_UNRESOLVED_VALUE; break;
                    }
                if (error == LANA_OK && materialize && ins->imm > 0u) {
                    arguments = lana_vm_alloc(vm, ins->imm * sizeof(*arguments));
                    if (arguments == NULL) error = LANA_ERR_OOM;
                    for (argument = 0u; error == LANA_OK && argument < ins->imm;
                         ++argument)
                        error = materialize_value(
                            vm, &frame->registers[ins->c + argument],
                            &arguments[argument]);
                }
                if (error == LANA_OK)
                    error = execute_host_call(vm, ins->b, arguments, ins->imm,
                                              &frame->registers[ins->a]);
                if (error == LANA_ERR_ASSERTION && ins->b == LANA_HOST_ASSERT &&
                    ins->imm == 2u && frame->registers[ins->c + 1u].type == VAL_STRING)
                    error_message = frame->registers[ins->c + 1u].as.string;
                break;
            }
            case OP_RETURN: {
                Value returned = frame->registers[ins->a];
                if (vm->frame_count == 1u) { vm->result = returned; vm->running = false; }
                else { size_t return_ip = frame->return_ip; uint32_t destination = frame->return_register; --vm->frame_count; current_frame(vm)->registers[destination] = returned; vm->ip = return_ip; }
                break;
            }
            case OP_PRINT:
                if (vm->active_path_count > 1u) error = LANA_ERR_UNSUPPORTED_OPERATION;
                else if (value_is_unresolved(&frame->registers[ins->a]))
                    error = LANA_ERR_UNRESOLVED_VALUE;
                else { lana_value_print(reactive_value(&frame->registers[ins->a])); (void)printf("\n"); }
                break;
            case OP_HALT:
                if (vm->path_execution != NULL) error = LANA_ERR_UNSUPPORTED_OPERATION;
                else vm->running = false;
                break;
            case OP_COUNT: error = LANA_ERR_OPCODE; break;
        }
        if (error != LANA_OK) {
            if (ins->opcode == OP_OBSERVE || ins->opcode == OP_RESOLVE) {
                const Value *inputs[] = {&frame->registers[ins->a]};
                LanaDerivation *failure = record_derivation(
                    vm,
                    ins->opcode == OP_OBSERVE ? LANA_DERIVATION_OBSERVATION
                                                 : LANA_DERIVATION_RESOLUTION,
                    ins->opcode == OP_OBSERVE ? "observe" : "resolve",
                    inputs, 1u, "", ins->line, LANA_EXACTNESS_EXACT, "failure",
                    LANA_DERIVATION_ERROR, lana_error_name(error));
                if (failure != NULL) {
                    vm->error.has_derivation = true;
                    vm->error.derivation_task_lineage = failure->task_lineage;
                    vm->error.derivation_local_sequence = failure->local_sequence;
                }
            }
            return vm_fail(vm, error, instruction_ip, ins,
                           error_message == NULL ? lana_error_name(error) : error_message);
        }
    }
    return LANA_OK;
}
