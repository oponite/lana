#define _POSIX_C_SOURCE 200809L

#include "vm.h"
#include "backend.h"
#include "metal.h"
#include "data.h"
#include "shared.h"
#include "sha256.h"
#include "tensor.h"
#include "unicode_case.h"
#include "store.h"
#include "ledger.h"
#include "policy.h"
#include "adapters.h"

static LanaError dataset_array_new(LanaVM *vm, LanaArray **out);
static LanaError dataset_array_push(LanaVM *vm, LanaArray *array, const Value *value);
static void net_socket_close(LanaSocket *sock);

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
#include <dlfcn.h>
#include <signal.h>
#include <ffi.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <poll.h>
#include <openssl/ssl.h>
#include <openssl/err.h>

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

/* LIP-006 authorized execution: `train` requires a non-revoked READ capability
 * over a shared information whose base snapshot is the string `name`. The
 * capability is granted by `capability("train")` (LIP-012) and tracked in the
 * VM's shared-reference list. */
static bool vm_has_named_capability(LanaVM *vm, const char *name) {
    LanaSharedReference *reference;
    if (vm == NULL || name == NULL) return false;
    for (reference = vm->shared_references; reference != NULL;
         reference = reference->next) {
        if (lana_shared_information_allows_named_read(reference->shared, name))
            return true;
    }
    return false;
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
static bool value_has_revoked_capability(const Value *value);
static LanaError reactive_recompute_transaction(LanaVM *vm, LanaReactive *root,
                                                const Value *replacement,
                                                uint32_t scratch_register);

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
static LanaError ad_grad(LanaVM *vm, const Value *function_value, const Value *x,
                         uint32_t scratch_register, Value *out);
static LanaError ad_vjp(LanaVM *vm, const Value *function_value, const Value *x,
                        const Value *v, uint32_t scratch_register, Value *out);
static LanaError run_function2(LanaVM *vm, uint32_t function_index,
                               const Value *arg0, const Value *arg1,
                               uint32_t scratch_register, Value *result);
static LanaError run_function(LanaVM *vm, uint32_t function_index,
                              const Value *arg, uint32_t scratch_register,
                              Value *result);
static LanaError dataset_new(LanaVM *vm, LanaDatasetOp op, Value source,
                             uint32_t function, Value columns, Value key,
                             Value limit, Value other, Value aggregate,
                             LanaDataset **out);
static LanaError dataset_materialize(LanaVM *vm, const LanaDataset *dataset,
                                    uint32_t scratch, LanaArray **out);
static LanaError dataset_explain(LanaVM *vm, const LanaDataset *dataset, Value *out);
static LanaError host_sgd(LanaVM *vm, const Value *arguments, size_t argc, Value *out);
static LanaError host_adam(LanaVM *vm, const Value *arguments, size_t argc, Value *out);
static LanaError host_train(LanaVM *vm, const Value *arguments, size_t argc,
                            uint32_t scratch_register, Value *out);
static LanaError host_update(LanaVM *vm, const Value *arguments, size_t argc,
                             uint32_t scratch_register, Value *out);
static LanaError host_resume(LanaVM *vm, const Value *arguments, size_t argc,
                             uint32_t scratch_register, Value *out);
static LanaError reactive_train_recompute(LanaVM *vm, LanaReactive *node,
                                          const Value *observation,
                                          uint32_t scratch_register, Value *out);
static LanaError host_mcmc(LanaVM *vm, const Value *arguments, size_t argc, Value *out);
static LanaError host_vi(LanaVM *vm, const Value *arguments, size_t argc, Value *out);
static LanaError host_smc(LanaVM *vm, const Value *arguments, size_t argc, Value *out);
static LanaError host_infer(LanaVM *vm, const Value *arguments, size_t argc,
                            uint32_t scratch_register, Value *out);
static bool vm_has_named_capability(LanaVM *vm, const char *name);

static void gc_trace_value_contents(LanaGC *gc, const Value *value);
static void gc_trace_value_block(LanaGC *gc, void *payload);
static void gc_trace_array(LanaGC *gc, void *payload);
static void gc_trace_map(LanaGC *gc, void *payload);
static void gc_trace_dataset(LanaGC *gc, void *payload);
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

static void gc_trace_adt(LanaGC *gc, void *payload) {
    LanaAdt *adt = payload;
    if (adt->fields != NULL) gc_mark_value_block(gc, adt->fields);
}

static void gc_trace_generator(LanaGC *gc, void *payload) {
    LanaGenerator *generator = payload;
    if (generator->registers != NULL) gc_mark_value_block(gc, generator->registers);
}

static void gc_trace_future(LanaGC *gc, void *payload) {
    LanaFuture *future = payload;
    size_t index;
    if (future->registers != NULL) gc_mark_value_block(gc, future->registers);
    if (future->inputs != NULL)
        for (index = 0u; index < future->input_count; ++index)
            gc_mark_object(gc, future->inputs[index], LANA_GC_FUTURE,
                           gc_trace_future);
}

static void gc_trace_set(LanaGC *gc, void *payload) {
    LanaSet *set = payload;
    if (set->items != NULL) gc_mark_value_block(gc, set->items);
}

static void gc_trace_dataset(LanaGC *gc, void *payload) {
    LanaDataset *dataset = payload;
    gc_trace_value_contents(gc, &dataset->source);
    gc_trace_value_contents(gc, &dataset->columns);
    gc_trace_value_contents(gc, &dataset->key);
    gc_trace_value_contents(gc, &dataset->limit);
    gc_trace_value_contents(gc, &dataset->other);
    gc_trace_value_contents(gc, &dataset->aggregate);
}

static void gc_trace_state_dist(LanaGC *gc, void *payload) {
    LanaStateDist *distribution = payload;
    LanaDistOperand *operands;
    switch (distribution->kind) {
        case LANA_DIST_DIRAC:
            gc_mark_state_source(gc, &distribution->as.dirac);
            break;
        case LANA_DIST_APPEND:
            operands = &distribution->as.append.left;
            for (size_t index = 0u; index < 2u; ++index) {
                if (operands[index].is_inline)
                    gc_mark_state_source(gc, &operands[index].as.state);
                else
                    gc_mark_object(gc, operands[index].as.node, LANA_GC_STATE_DIST,
                                   gc_trace_state_dist);
            }
            break;
        case LANA_DIST_TRANSFORM:
            gc_mark_object(gc, distribution->as.transform.child, LANA_GC_STATE_DIST,
                           gc_trace_state_dist);
            break;
        case LANA_DIST_ATTENUATE:
            gc_mark_object(gc, distribution->as.attenuate.child, LANA_GC_STATE_DIST,
                           gc_trace_state_dist);
            break;
    }
}

static void gc_trace_tensor_chain(LanaGC *gc, const LanaTensor *tensor) {
    while (tensor != NULL) {
        gc_mark_leaf(gc, tensor, LANA_GC_OPAQUE);
        if (tensor->shape != NULL)
            gc_mark_leaf(gc, tensor->shape, LANA_GC_OPAQUE);
        if (tensor->strides != NULL)
            gc_mark_leaf(gc, tensor->strides, LANA_GC_OPAQUE);
        if (tensor->data != NULL)
            gc_mark_leaf(gc, tensor->data, LANA_GC_OPAQUE);
        tensor = tensor->base;
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
    /* LIP-011 autodiff fields: saved operands, input derivations, and the
     * accumulated gradient buffer. */
    gc_trace_tensor_chain(gc, derivation->ad_a);
    gc_trace_tensor_chain(gc, derivation->ad_b);
    gc_mark_object(gc, derivation->ad_a_deriv, LANA_GC_DERIVATION,
                   gc_trace_derivation);
    gc_mark_object(gc, derivation->ad_b_deriv, LANA_GC_DERIVATION,
                   gc_trace_derivation);
    gc_trace_tensor_chain(gc, derivation->ad_grad);
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
        case VAL_ADT:
            gc_mark_object(gc, value->as.adt, LANA_GC_ADT, gc_trace_adt);
            break;
        case VAL_GENERATOR:
            gc_mark_object(gc, value->as.generator, LANA_GC_GENERATOR, gc_trace_generator);
            break;
        case VAL_FUTURE:
            gc_mark_object(gc, value->as.future, LANA_GC_FUTURE, gc_trace_future);
            break;
        case VAL_SET:
            gc_mark_object(gc, value->as.set, LANA_GC_SET, gc_trace_set);
            break;
        case VAL_DATASET:
            gc_mark_object(gc, value->as.dataset, LANA_GC_OPAQUE, gc_trace_dataset);
            break;
        case VAL_REGEX:
            gc_mark_leaf(gc, value->as.regex, LANA_GC_OPAQUE);
            if (value->as.regex != NULL) {
                gc_mark_leaf(gc, value->as.regex->insts, LANA_GC_OPAQUE);
                gc_mark_leaf(gc, value->as.regex->classes, LANA_GC_OPAQUE);
            }
            break;
        case VAL_TENSOR:
        case VAL_NQUBIT_STATE:
        case VAL_POVM:
        case VAL_CHANNEL:
        case VAL_OBSERVABLE: {
            /* Walk the base chain so a view keeps its shared buffer alive; a
             * chain is a loop, not recursion, so arbitrarily deep view chains
             * cannot overflow the mark stack. */
            const LanaTensor *tensor = value->as.tensor;
            while (tensor != NULL) {
                gc_mark_leaf(gc, tensor, LANA_GC_OPAQUE);
                if (tensor->shape != NULL)
                    gc_mark_leaf(gc, tensor->shape, LANA_GC_OPAQUE);
                if (tensor->strides != NULL)
                    gc_mark_leaf(gc, tensor->strides, LANA_GC_OPAQUE);
                if (tensor->data != NULL)
                    gc_mark_leaf(gc, tensor->data, LANA_GC_OPAQUE);
                tensor = tensor->base;
            }
            break;
        }
        case VAL_OPTIMIZER:
            if (value->as.optimizer != NULL) {
                gc_mark_leaf(gc, value->as.optimizer, LANA_GC_OPAQUE);
                gc_mark_leaf(gc, value->as.optimizer->name, LANA_GC_STRING);
            }
            break;
        case VAL_TRAINING_RESULT: {
            if (value->as.training_result != NULL) {
                gc_mark_leaf(gc, value->as.training_result, LANA_GC_OPAQUE);
                const LanaTensor *tensor = value->as.training_result->params;
                while (tensor != NULL) {
                    gc_mark_leaf(gc, tensor, LANA_GC_OPAQUE);
                    if (tensor->shape != NULL)
                        gc_mark_leaf(gc, tensor->shape, LANA_GC_OPAQUE);
                    if (tensor->strides != NULL)
                        gc_mark_leaf(gc, tensor->strides, LANA_GC_OPAQUE);
                    if (tensor->data != NULL)
                        gc_mark_leaf(gc, tensor->data, LANA_GC_OPAQUE);
                    tensor = tensor->base;
                }
                gc_mark_object(gc, value->as.training_result->steps,
                               LANA_GC_ARRAY, gc_trace_array);
                if (value->as.training_result->optimizer != NULL) {
                    gc_mark_leaf(gc, value->as.training_result->optimizer,
                                 LANA_GC_OPAQUE);
                    gc_mark_leaf(gc, value->as.training_result->optimizer->name,
                                 LANA_GC_STRING);
                }
                if (value->as.training_result->data != NULL)
                    gc_mark_value_pointer(gc, value->as.training_result->data);
            }
            break;
        }
        case VAL_INFERENCE_ALGORITHM:
            if (value->as.inference_algorithm != NULL) {
                gc_mark_leaf(gc, value->as.inference_algorithm, LANA_GC_OPAQUE);
                gc_mark_leaf(gc, value->as.inference_algorithm->name, LANA_GC_STRING);
                gc_mark_leaf(gc, value->as.inference_algorithm->family, LANA_GC_STRING);
            }
            break;
        case VAL_POSTERIOR: {
            if (value->as.posterior != NULL) {
                gc_mark_leaf(gc, value->as.posterior, LANA_GC_OPAQUE);
                const LanaTensor *tensors[3] = {
                    value->as.posterior->mean,
                    value->as.posterior->variance,
                    value->as.posterior->samples
                };
                for (size_t t = 0; t < 3u; ++t) {
                    const LanaTensor *tensor = tensors[t];
                    while (tensor != NULL) {
                        gc_mark_leaf(gc, tensor, LANA_GC_OPAQUE);
                        if (tensor->shape != NULL)
                            gc_mark_leaf(gc, tensor->shape, LANA_GC_OPAQUE);
                        if (tensor->strides != NULL)
                            gc_mark_leaf(gc, tensor->strides, LANA_GC_OPAQUE);
                        if (tensor->data != NULL)
                            gc_mark_leaf(gc, tensor->data, LANA_GC_OPAQUE);
                        tensor = tensor->base;
                    }
                }
                gc_mark_object(gc, value->as.posterior->steps,
                               LANA_GC_ARRAY, gc_trace_array);
            }
            break;
        }
        case VAL_NULL:
        case VAL_NUMBER:
        case VAL_BOOL:
        case VAL_DISTRIBUTION:
        case VAL_SAMPLE:
        case VAL_FUNCTION:
        case VAL_TASK:
        case VAL_SHARED_CAPABILITY:
        case VAL_LAZY:
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
    for (frame_index = 0u; frame_index < vm->ready_count; ++frame_index)
        gc_mark_object(gc, vm->ready_queue[frame_index], LANA_GC_FUTURE,
                       gc_trace_future);
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

void lana_vm_set_memory_limit(LanaVM *vm, size_t memory_limit) {
    size_t threshold;
    if (vm == NULL) return;
    vm->memory_limit = memory_limit;
    vm->gc.memory_limit = memory_limit;
    threshold = memory_limit / 2u;
    if (threshold == 0u && memory_limit > 0u) threshold = 1u;
    vm->gc.collection_threshold = threshold;
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
    else if (value->type == VAL_ADT) target = value->as.adt;
    if (target != NULL) (void)lana_gc_write_barrier(&vm->gc, owner, target);
}

static bool vm_gc_safepoint(LanaVM *vm) {
    if (vm == NULL) return false;
    if (vm->gc.native_allocations != NULL)
        lana_gc_release_native(&vm->gc);
    if (vm->gc.allocated_bytes > vm->gc.memory_limit) {
        /* Hard limit: full collection, never delayed. */
        lana_gc_set_deferred(&vm->gc, false);
        if (!lana_gc_collect_young(&vm->gc)) {
            lana_gc_set_deferred(&vm->gc, true);
            return false;
        }
        if (vm->gc.allocated_bytes > vm->gc.memory_limit &&
            !lana_gc_collect(&vm->gc)) {
            lana_gc_set_deferred(&vm->gc, true);
            return false;
        }
        lana_gc_set_deferred(&vm->gc, true);
    } else if ((vm->instruction_count & 255u) == 0u &&
               vm->gc.allocated_bytes >= vm->gc.collection_threshold &&
               vm->gc.allocated_bytes > 0u) {
        /* Proactive young collection, throttled to every 256 instructions. */
        lana_gc_set_deferred(&vm->gc, false);
        if (!lana_gc_collect_young(&vm->gc)) {
            lana_gc_set_deferred(&vm->gc, true);
            return false;
        }
        lana_gc_set_deferred(&vm->gc, true);
    }
    vm->allocated_bytes = vm->gc.allocated_bytes;
    vm->allocation_count = vm->gc.allocation_count;
    return vm->allocated_bytes <= vm->gc.memory_limit;
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

const char *lana_evidence_status_name(LanaEvidenceStatus status) {
    static const char *names[] = {"unknown", "sampled", "modeled", "exact", "observed"};
    return (size_t)status < sizeof(names) / sizeof(names[0]) ? names[status] : "unknown";
}

LanaEvidenceStatus lana_derivation_status(const LanaDerivation *d) {
    if (d == NULL) return LANA_EVIDENCE_EXACT;
    if (d->outcome == LANA_DERIVATION_UNRESOLVED) return LANA_EVIDENCE_UNKNOWN;
    if (d->kind == LANA_DERIVATION_OBSERVATION && d->exactness == LANA_EXACTNESS_EXACT)
        return LANA_EVIDENCE_OBSERVED;
    if (d->kind == LANA_DERIVATION_ASSUMPTION && d->exactness == LANA_EXACTNESS_APPROXIMATE)
        return LANA_EVIDENCE_MODELED;
    if (d->kind == LANA_DERIVATION_SAMPLE && d->exactness == LANA_EXACTNESS_SAMPLE)
        return LANA_EVIDENCE_SAMPLED;
    if (d->exactness == LANA_EXACTNESS_EXACT) return LANA_EVIDENCE_EXACT;
    if (d->exactness == LANA_EXACTNESS_APPROXIMATE) return LANA_EVIDENCE_MODELED;
    if (d->exactness == LANA_EXACTNESS_SAMPLE) return LANA_EVIDENCE_SAMPLED;
    return LANA_EVIDENCE_UNKNOWN;
}

/* Least-certain evidence status over a set of input values. A value with no
 * derivation is a bare literal and is treated as exact. */
static LanaEvidenceStatus least_certain_evidence_status(const Value *const *inputs,
                                                        size_t count) {
    LanaEvidenceStatus status = LANA_EVIDENCE_OBSERVED;
    size_t index;
    for (index = 0; index < count; ++index) {
        LanaEvidenceStatus current = inputs[index] == NULL
            ? LANA_EVIDENCE_EXACT
            : lana_derivation_status(inputs[index]->derivation);
        if (current < status) status = current;
    }
    return status;
}

/* Map an evidence status to the (kind, exactness, outcome) a combining
 * operation's result derivation must carry. */
static void evidence_status_derivation_fields(LanaEvidenceStatus status,
                                              LanaDerivationKind *kind,
                                              LanaDerivationExactness *exactness,
                                              LanaDerivationOutcome *outcome) {
    switch (status) {
        case LANA_EVIDENCE_OBSERVED:
            *kind = LANA_DERIVATION_OBSERVATION;
            *exactness = LANA_EXACTNESS_EXACT;
            *outcome = LANA_DERIVATION_SUCCESS;
            break;
        case LANA_EVIDENCE_MODELED:
            *kind = LANA_DERIVATION_ASSUMPTION;
            *exactness = LANA_EXACTNESS_APPROXIMATE;
            *outcome = LANA_DERIVATION_SUCCESS;
            break;
        case LANA_EVIDENCE_SAMPLED:
            *kind = LANA_DERIVATION_SAMPLE;
            *exactness = LANA_EXACTNESS_SAMPLE;
            *outcome = LANA_DERIVATION_SUCCESS;
            break;
        case LANA_EVIDENCE_UNKNOWN:
            *kind = LANA_DERIVATION_OPERATION;
            *exactness = LANA_EXACTNESS_EXACT;
            *outcome = LANA_DERIVATION_UNRESOLVED;
            break;
        case LANA_EVIDENCE_EXACT:
        default:
            *kind = LANA_DERIVATION_OPERATION;
            *exactness = LANA_EXACTNESS_EXACT;
            *outcome = LANA_DERIVATION_SUCCESS;
            break;
    }
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
    node->ad_op = -1;
    node->ad_a = NULL;
    node->ad_b = NULL;
    node->ad_a_deriv = NULL;
    node->ad_b_deriv = NULL;
    node->ad_grad = NULL;
    node->ad_axis = -1;
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

/* Attach a derivation to a combining operation's result, applying the
 * least-certain-wins evidence rule: the result's kind/exactness/outcome are
 * set from the least-certain input status. */
static LanaError attach_combine_derivation(LanaVM *vm, Value *out,
                                           const char *operation,
                                           const Value *const *inputs, size_t input_count,
                                           uint32_t line, const char *details) {
    LanaDerivationKind kind;
    LanaDerivationExactness exactness;
    LanaDerivationOutcome outcome;
    evidence_status_derivation_fields(least_certain_evidence_status(inputs, input_count),
                                      &kind, &exactness, &outcome);
    out->derivation = record_derivation(vm, kind, operation, inputs, input_count,
                                        "", line, exactness, details, outcome, "none");
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
        assumption ? LANA_EXACTNESS_APPROXIMATE : LANA_EXACTNESS_EXACT,
        "root", LANA_DERIVATION_SUCCESS, "none");
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
    if ((error = lana_map_new(vm, 13u, &map)) != LANA_OK) return error;
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
        (error = map_put(vm, map, "status", lana_value_string(lana_evidence_status_name(lana_derivation_status(node))))) != LANA_OK ||
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

LanaVM *lana_vm_create(void) {
    LanaVM *vm = calloc(1u, sizeof(*vm));
    if (vm == NULL) return NULL;
    lana_vm_init(vm, NULL);
    return vm;
}

void lana_vm_destroy(LanaVM *vm) {
    if (vm == NULL) return;
    lana_vm_free(vm);
    free(vm);
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
    if (vm->ledger != NULL) { lana_ledger_close(vm->ledger); vm->ledger = NULL; }
    if (vm->store != NULL) { lana_store_close(vm->store); vm->store = NULL; }
    if (vm->adapter != NULL) { lana_adapter_close(vm->adapter); vm->adapter = NULL; }
    if (vm->ffi_lib != NULL) { dlclose(vm->ffi_lib); vm->ffi_lib = NULL; }
    if (vm->ffi_sigs != NULL) {
        size_t index;
        for (index = 0; index < vm->ffi_sig_count; ++index) free(vm->ffi_sigs[index]);
        free(vm->ffi_sigs);
        vm->ffi_sigs = NULL;
        vm->ffi_sig_count = 0u;
    }
    if (vm->sockets != NULL) {
        size_t index;
        for (index = 0; index < vm->socket_count; ++index) net_socket_close(&vm->sockets[index]);
        free(vm->sockets);
        vm->sockets = NULL;
        vm->socket_count = 0u;
        vm->socket_capacity = 0u;
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
            for (size_t index = 0u; index < 2u && error == LANA_OK; ++index) {
                LanaDistOperand *operand = index == 0u
                    ? &copy->as.append.left : &copy->as.append.right;
                const LanaDistOperand *source_operand = index == 0u
                    ? &source->as.append.left : &source->as.append.right;
                if (source_operand->is_inline) {
                    error = clone_state_value(destination, &source_operand->as.state,
                                              &operand->as.state);
                } else {
                    error = clone_state_dist_node(destination, source_operand->as.node,
                                                  &operand->as.node, memo);
                }
            }
            break;
        case LANA_DIST_TRANSFORM:
            copy->as.transform.transform_id = source->as.transform.transform_id;
            error = clone_state_dist_node(destination, source->as.transform.child,
                                          &copy->as.transform.child, memo);
            break;
        case LANA_DIST_ATTENUATE:
            copy->as.attenuate.factor = source->as.attenuate.factor;
            error = clone_state_dist_node(destination, source->as.attenuate.child,
                                          &copy->as.attenuate.child, memo);
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
    } else if (source->type == VAL_ADT) {
        LanaAdt *adt;
        LanaContainerCloneMemo *entry;
        for (entry = *containers; entry != NULL; entry = entry->next)
            if (entry->type == VAL_ADT && entry->source == source->as.adt) {
                out->as.adt = entry->copy; return LANA_OK;
            }
        adt = lana_vm_alloc(destination, sizeof(*adt));
        if (adt == NULL) return LANA_ERR_OOM;
        entry = malloc(sizeof(*entry));
        if (entry == NULL) return LANA_ERR_OOM;
        entry->source = source->as.adt; entry->copy = adt;
        entry->type = VAL_ADT; entry->next = *containers; *containers = entry;
        adt->variant = source->as.adt->variant;
        adt->field_count = source->as.adt->field_count;
        adt->fields = lana_vm_alloc(destination, adt->field_count * sizeof(*adt->fields));
        if (adt->fields == NULL && adt->field_count > 0u) return LANA_ERR_OOM;
        for (index = 0; index < adt->field_count; ++index) {
            error = clone_value_memo(destination, &source->as.adt->fields[index],
                                     &adt->fields[index], memo, containers, derivations);
            if (error != LANA_OK) return error;
        }
        out->as.adt = adt;
    } else if (source->type == VAL_SET) {
        LanaSet *set;
        LanaContainerCloneMemo *entry;
        for (entry = *containers; entry != NULL; entry = entry->next)
            if (entry->type == VAL_SET && entry->source == source->as.set) {
                out->as.set = entry->copy; return LANA_OK;
            }
        set = lana_vm_alloc(destination, sizeof(*set));
        if (set == NULL) return LANA_ERR_OOM;
        entry = malloc(sizeof(*entry));
        if (entry == NULL) return LANA_ERR_OOM;
        entry->source = source->as.set; entry->copy = set; entry->type = VAL_SET;
        entry->next = *containers; *containers = entry;
        set->count = source->as.set->count;
        set->capacity = set->count;
        set->items = lana_vm_alloc(destination, set->count * sizeof(*set->items));
        if (set->items == NULL && set->count > 0u) return LANA_ERR_OOM;
        for (index = 0; index < set->count; ++index) {
            error = clone_value_memo(destination, &source->as.set->items[index],
                                     &set->items[index], memo, containers, derivations);
            if (error != LANA_OK) return error;
        }
        out->as.set = set;
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
        case VAL_SET:
            if (left->as.set->count != right->as.set->count) return false;
            for (index = 0u; index < left->as.set->count; ++index) {
                size_t other;
                bool found = false;
                for (other = 0u; other < right->as.set->count; ++other)
                    if (value_equal_deep(&left->as.set->items[index],
                                         &right->as.set->items[other], depth + 1u)) {
                        found = true;
                        break;
                    }
                if (!found) return false;
            }
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

static void free_joint_names(char **names, size_t count) {
    size_t index;
    for (index = 0; index < count; ++index) free(names[index]);
    free(names);
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
        if (*token == '\0') { free_joint_names(names, count); free(copy); return LANA_ERR_FORMAT; }
        for (index = 0; index < count; ++index)
            if (strcmp(names[index], token) == 0) { free_joint_names(names, count); free(copy); return LANA_ERR_INVALID_DEPENDENCY; }
        if (count >= expected) { free_joint_names(names, count); free(copy); return LANA_ERR_FORMAT; }
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
    if (current->type == VAL_SET && current->as.set != NULL) {
        LanaSet *set = lana_vm_alloc(vm, sizeof(*set));
        if (set == NULL) return LANA_ERR_OOM;
        set->count = current->as.set->count;
        set->capacity = set->count;
        set->items = lana_vm_alloc(vm, set->count * sizeof(*set->items));
        if (set->items == NULL && set->count > 0u) return LANA_ERR_OOM;
        for (index = 0u; index < set->count; ++index) {
            error = materialize_value(vm, &current->as.set->items[index],
                                      &set->items[index]);
            if (error != LANA_OK) return error;
        }
        *out = lana_value_set(set);
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

static LanaError reactive_observe_scratch(LanaVM *vm, const Value *source,
                                          const Value *evidence,
                                          uint32_t scratch_register, Value *out) {
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
    } else if (root->is_training_data) {
        /* LIP-010: a training data root's support is structural — a single
         * [x, target] observation. */
        if (replacement->type != VAL_ARRAY || replacement->as.array == NULL ||
            replacement->as.array->count != 2u)
            return LANA_ERR_INVALID_PARAMETERS;
        supported = true;
    } else if (!joint_value_equal(current, replacement)) {
        return LANA_ERR_INVALID_CONDITIONING;
    }
    error = reactive_recompute_transaction(vm, root, replacement, scratch_register);
    if (error != LANA_OK) return error;
    ++vm->observation_count;
    *out = *source;
    return LANA_OK;
}

LanaError lana_vm_reactive_observe(LanaVM *vm, const Value *source,
                                   const Value *evidence, Value *out) {
    /* The public entry point is only used for non-training observes (shared
     * Information and tests), which never run a model/loss function, so the
     * scratch register is unused. */
    return reactive_observe_scratch(vm, source, evidence, 0u, out);
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
    if (value_has_revoked_capability(plan->payload)) return LANA_ERR_CLAIM_REVOKED;
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

typedef struct LanaValuePath {
    const void *container;
    const struct LanaValuePath *parent;
} LanaValuePath;

static bool value_is_unresolved_at(const Value *value, const LanaValuePath *parent) {
    size_t index;
    if (value == NULL) return false;
    value = reactive_value(value);
    if (value->type == VAL_POSSIBILITY || value->type == VAL_PATH_SET)
        return true;
    const void *container = value->type == VAL_ARRAY ? (const void *)value->as.array :
                            value->type == VAL_MAP ? (const void *)value->as.map :
                            value->type == VAL_SET ? (const void *)value->as.set : NULL;
    if (container == NULL) return false;
    /* A back edge adds no new unresolved leaves. Keep checking its siblings.
     * ponytail: O(depth^2) ancestor checks; use an iterative visited set if
     * deeply nested general value traversal becomes a supported workload. */
    for (const LanaValuePath *p = parent; p != NULL; p = p->parent)
        if (p->container == container) return false;
    LanaValuePath path = {container, parent};
    if (value->type == VAL_ARRAY && value->as.array != NULL) {
        for (index = 0; index < value->as.array->count; ++index)
            if (value_is_unresolved_at(&value->as.array->items[index], &path)) return true;
    }
    if (value->type == VAL_MAP && value->as.map != NULL) {
        for (index = 0; index < value->as.map->count; ++index)
            if (value_is_unresolved_at(value->as.map->entries[index].value, &path)) return true;
    }
    if (value->type == VAL_SET && value->as.set != NULL) {
        for (index = 0; index < value->as.set->count; ++index)
            if (value_is_unresolved_at(&value->as.set->items[index], &path)) return true;
    }
    return false;
}

static bool value_is_unresolved(const Value *value) {
    return value_is_unresolved_at(value, NULL);
}

static bool value_has_revoked_capability_at(const Value *value,
                                            const LanaValuePath *parent) {
    size_t index;
    if (value == NULL) return false;
    value = reactive_value(value);
    if (value->type == VAL_SHARED_CAPABILITY &&
        !lana_shared_capability_allows(value->as.capability, 0u))
        return true;
    const void *container = value->type == VAL_ARRAY ? (const void *)value->as.array :
                            value->type == VAL_MAP ? (const void *)value->as.map :
                            value->type == VAL_SET ? (const void *)value->as.set : NULL;
    if (container == NULL) return false;
    for (const LanaValuePath *p = parent; p != NULL; p = p->parent)
        if (p->container == container) return false;
    LanaValuePath path = {container, parent};
    if (value->type == VAL_ARRAY && value->as.array != NULL) {
        for (index = 0; index < value->as.array->count; ++index)
            if (value_has_revoked_capability_at(&value->as.array->items[index], &path))
                return true;
    }
    if (value->type == VAL_MAP && value->as.map != NULL) {
        for (index = 0; index < value->as.map->count; ++index)
            if (value_has_revoked_capability_at(value->as.map->entries[index].value, &path))
                return true;
    }
    if (value->type == VAL_SET && value->as.set != NULL) {
        for (index = 0; index < value->as.set->count; ++index)
            if (value_has_revoked_capability_at(&value->as.set->items[index], &path))
                return true;
    }
    return false;
}

static bool value_has_revoked_capability(const Value *value) {
    return value_has_revoked_capability_at(value, NULL);
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
    lana_vm_set_memory_limit(task->child, parent->memory_limit);
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

static uint32_t grant_permission(const Value *value) {
    if (value->type != VAL_STRING) return 0u;
    if (strcmp(value->as.string, "use") == 0) return LANA_CAPABILITY_READ;
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

/* Build the bivariate Bernoulli joint law for `correlated(x, y, d)`, reusing
 * the normalized off-diagonal of STATE (semantics-2.md §9.8.4). The joint
 * distribution over {0,1}×{0,1} has
 *   p11 = p_x p_y + d sqrt(p_x(1-p_x) p_y(1-p_y)),
 *   p10 = p_x - p11, p01 = p_y - p11, p00 = 1 - p_x - p_y + p11. */
static LanaError host_correlated(LanaVM *vm, const Value *x, const Value *y,
                                 const Value *d, Value *out) {
    double p_x, p_y, coefficient, cross, p11, p10, p01, p00;
    double cell[4];
    Value rows[8];
    double weights[4];
    size_t row_count = 0u, index;
    LanaJointState *joint;
    LanaError error;
    if (x->type != VAL_STATE || y->type != VAL_STATE || d->type != VAL_NUMBER)
        return LANA_ERR_TYPE;
    p_x = x->as.state.state.p;
    p_y = y->as.state.state.p;
    coefficient = d->as.number;
    if (!isfinite(coefficient) || coefficient < -1.0 || coefficient > 1.0)
        return LANA_ERR_INVALID_PARAMETERS;
    if (p_x < 0.0 || p_x > 1.0 || p_y < 0.0 || p_y > 1.0)
        return LANA_ERR_TYPE;
    cross = coefficient * sqrt(p_x * (1.0 - p_x) * p_y * (1.0 - p_y));
    p11 = p_x * p_y + cross;
    p10 = p_x - p11;
    p01 = p_y - p11;
    p00 = 1.0 - p_x - p_y + p11;
    cell[0] = p00; cell[1] = p01; cell[2] = p10; cell[3] = p11;
    for (index = 0; index < 4u; ++index) {
        if (cell[index] < -LANA_STATE_EPSILON ||
            cell[index] > 1.0 + LANA_STATE_EPSILON)
            return LANA_ERR_INVALID_PARAMETERS;
        if (cell[index] < 0.0) cell[index] = 0.0;
        if (cell[index] > 1.0) cell[index] = 1.0;
    }
    /* |d| = 1 collapses the 2x2 law to its diagonal; the finite joint law
     * rejects zero-weight rows, so emit only the cells with positive mass. */
    if (cell[0] > LANA_STATE_EPSILON) {
        rows[row_count * 2 + 0] = lana_value_number(0.0);
        rows[row_count * 2 + 1] = lana_value_number(0.0);
        weights[row_count++] = cell[0];
    }
    if (cell[1] > LANA_STATE_EPSILON) {
        rows[row_count * 2 + 0] = lana_value_number(0.0);
        rows[row_count * 2 + 1] = lana_value_number(1.0);
        weights[row_count++] = cell[1];
    }
    if (cell[2] > LANA_STATE_EPSILON) {
        rows[row_count * 2 + 0] = lana_value_number(1.0);
        rows[row_count * 2 + 1] = lana_value_number(0.0);
        weights[row_count++] = cell[2];
    }
    if (cell[3] > LANA_STATE_EPSILON) {
        rows[row_count * 2 + 0] = lana_value_number(1.0);
        rows[row_count * 2 + 1] = lana_value_number(1.0);
        weights[row_count++] = cell[3];
    }
    error = lana_vm_joint_build_finite(vm, "x;y", rows, weights, row_count, 2u, &joint);
    if (error != LANA_OK) return error;
    *out = (Value){.type = VAL_JOINT_STATE, .as.joint = joint};
    return LANA_OK;
}

/* ===== LIP-004 tensor helpers ===== */

/* LIP-027: dtype string conversion. `dtype_from_string` returns -1 for an
 * unknown dtype string (the caller maps that to LANA_ERR_INVALID_PARAMETERS). */
static const char *dtype_to_string(LanaTensorDtype dtype) {
    switch (dtype) {
        case LANA_TENSOR_F64: return "f64";
        case LANA_TENSOR_F32: return "f32";
        case LANA_TENSOR_F16: return "f16";
        case LANA_TENSOR_BF16: return "bf16";
        case LANA_TENSOR_COMPLEX: return "complex";
    }
    return "f64";
}

static int dtype_from_string(const char *s) {
    if (s == NULL) return -1;
    if (strcmp(s, "f64") == 0) return LANA_TENSOR_F64;
    if (strcmp(s, "f32") == 0) return LANA_TENSOR_F32;
    if (strcmp(s, "f16") == 0) return LANA_TENSOR_F16;
    if (strcmp(s, "bf16") == 0) return LANA_TENSOR_BF16;
    if (strcmp(s, "complex") == 0) return LANA_TENSOR_COMPLEX;
    return -1;
}

static double f16_to_double(uint16_t h);

/* LIP-027: relative precision rank used to pick the default matmul output
 * dtype for a mixed-dtype pair. Higher rank wins on a tie; f16 and bf16 are
 * equal (both 16-bit). Complex is handled separately by the caller. */
static int dtype_rank(LanaTensorDtype d) {
    switch (d) {
        case LANA_TENSOR_F64: return 3;
        case LANA_TENSOR_F32: return 2;
        case LANA_TENSOR_F16: return 1;
        case LANA_TENSOR_BF16: return 1;
        default: return 0;
    }
}

/* LIP-027: default matmul output dtype. Same dtype -> that dtype; otherwise
 * the higher-precision operand (f64/f32 mix -> f64, f32/f16 -> f32). */
static LanaTensorDtype matmul_default_dtype(const LanaTensor *a, const LanaTensor *b) {
    if (a->dtype == b->dtype) return a->dtype;
    return (dtype_rank(a->dtype) >= dtype_rank(b->dtype)) ? a->dtype : b->dtype;
}

/* LIP-027: whether matmul accumulates in binary32. f16/bf16 inputs always
 * accumulate in fp32 regardless of the output dtype; f32 x f32 also uses fp32.
 * A binary64 (or complex) input forces fp64 accumulation so it is never
 * silently downcast. */
static bool matmul_accumulation_fp32(const LanaTensor *a, const LanaTensor *b) {
    if (a->dtype == LANA_TENSOR_F16 || a->dtype == LANA_TENSOR_BF16 ||
        b->dtype == LANA_TENSOR_F16 || b->dtype == LANA_TENSOR_BF16) return true;
    return a->dtype == LANA_TENSOR_F32 && b->dtype == LANA_TENSOR_F32;
}

/* LIP-027: convert a double to a binary16 (f16) bit pattern,
 * round-to-nearest-even. Handles normal, subnormal, and overflow-to-inf. */
static uint16_t f64_to_f16_bits(double x) {
    if (isnan(x)) return 0x7E00u; /* canonical f16 NaN */
    if (isinf(x)) return x < 0 ? 0xFC00u : 0x7C00u;
    if (x == 0.0) {
        union { double d; uint64_t u; } z;
        z.d = x;
        return (z.u >> 63) ? 0x8000u : 0x0000u; /* preserve signed zero */
    }
    union { double d; uint64_t u; } v;
    v.d = x;
    uint32_t sign = (uint32_t)(v.u >> 63);
    int exp = (int)((v.u >> 52) & 0x7FF) - 1023;
    uint64_t sig = (1ull << 52) | (v.u & ((1ull << 52) - 1)); /* 53-bit significand */

    /* Round the 53-bit significand to 11 bits (1 implicit + 10 explicit). */
    uint64_t drop = 42;
    uint64_t round_bit = 1ull << (drop - 1);
    uint64_t mask = (1ull << drop) - 1;
    uint64_t lsb = 1ull << drop;
    uint64_t rounded = sig + round_bit;
    if ((sig & mask) == round_bit && (sig & lsb) == 0) rounded = sig;
    uint64_t sig11 = rounded >> drop;
    if (sig11 == (1ull << 11)) { sig11 = 0; exp += 1; }
    uint64_t mant10 = sig11 & 0x3FF;

    if (exp > 15) return (uint16_t)((sign << 15) | 0x7C00u); /* overflow to inf */
    if (exp >= -14) { /* normal f16 */
        return (uint16_t)((sign << 15) | ((uint16_t)(exp + 15) << 10) | (uint16_t)mant10);
    }
    /* subnormal: value = sig11 * 2^exp, exp < -14 */
    int shift = -14 - exp;
    if (shift >= 11) {
        /* rounds to zero; exactly 2^-25 (halfway to min subnormal) rounds to
         * zero (even) */
        return (uint16_t)(sign << 15);
    }
    uint64_t sub_mant = sig11 >> shift;
    uint64_t dropped = sig11 & ((1ull << shift) - 1);
    uint64_t half = 1ull << (shift - 1);
    if (dropped > half || (dropped == half && (sub_mant & 1))) sub_mant += 1;
    if (sub_mant == (1ull << 10)) { /* rounded up to min normal 2^-14 */
        return (uint16_t)((sign << 15) | (1u << 10));
    }
    return (uint16_t)((sign << 15) | (uint16_t)sub_mant);
}

static double f16_to_double(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    double value;
    if (exp == 0) {
        value = ldexp((double)mant, -24);
    } else if (exp == 31) {
        value = mant ? NAN : INFINITY;
    } else {
        value = ldexp((double)((1 << 10) | mant), (int)exp - 15 - 10);
    }
    return sign ? -value : value;
}

/* LIP-027: convert a double to a bfloat16 (bf16) bit pattern,
 * round-to-nearest-even. bf16 shares fp32's exponent range; overflow to inf. */
static uint16_t f64_to_bf16_bits(double x) {
    if (isnan(x)) return 0x7FC0u; /* canonical bf16 NaN */
    if (isinf(x)) return x < 0 ? 0xFF80u : 0x7F80u;
    if (x == 0.0) {
        union { double d; uint64_t u; } z;
        z.d = x;
        return (z.u >> 63) ? 0x8000u : 0x0000u; /* preserve signed zero */
    }
    union { double d; uint64_t u; } v;
    v.d = x;
    int exp = (int)((v.u >> 52) & 0x7FF) - 1023;
    if (exp > 127) return x < 0 ? 0xFF80u : 0x7F80u; /* overflow beyond fp32 range */
    /* Round the 52-bit mantissa to 7 bits (drop 45), round-to-nearest-even. */
    uint64_t low = v.u & 0x1FFFFFFFFFFF;
    uint64_t half = 1ull << 44;
    uint64_t lsb = 1ull << 45;
    if (low > half || (low == half && (v.u & lsb))) v.u += 1ull << 45;
    v.u &= ~0x1FFFFFFFFFFFull;
    /* The rounded f64 has 8 significant mantissa bits, so it converts to f32
     * exactly; bf16 is the top 16 bits of that f32 representation. */
    float f = (float)v.d;
    uint32_t f32_bits;
    memcpy(&f32_bits, &f, sizeof(f32_bits));
    return (uint16_t)(f32_bits >> 16);
}

/* LIP-027: convert a bfloat16 (bf16) bit pattern to a double. bf16 is the top
 * 16 bits of an f32, so widen the f32 to f64. */
static double bf16_to_double(uint16_t h) {
    uint32_t f32_bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &f32_bits, sizeof(f));
    return (double)f;
}

/* LIP-027: parse an optional trailing dtype string argument (the last of
 * `argc` arguments). Returns the dtype, or -1 if the argument is not a valid
 * dtype string. Callers must already have checked argc is 1 or 2. */
static int tensor_optional_dtype(const Value *arguments, unsigned argc) {
    if (argc == 1u) return LANA_TENSOR_F64;
    if (arguments[1].type != VAL_STRING) return -1;
    return dtype_from_string(arguments[1].as.string);
}

/* LIP-027: element width in bytes for a tensor's dtype. */
static size_t tensor_elem_bytes(const LanaTensor *t) {
    switch (t->dtype) {
    case LANA_TENSOR_F32: return 4u;
    case LANA_TENSOR_F16: return 2u;
    case LANA_TENSOR_BF16: return 2u;
    case LANA_TENSOR_COMPLEX: return 16u;
    default: return 8u; /* F64 */
    }
}

/* LIP-027: read the real part of element `i` (absolute element index, already
 * including `offset`), converting from the storage dtype to double. */
double tensor_get_real(const LanaTensor *t, size_t i) {
    switch (t->dtype) {
    case LANA_TENSOR_F32: { float f; memcpy(&f, t->data + i * 4u, 4u); return (double)f; }
    case LANA_TENSOR_F16: { uint16_t h; memcpy(&h, t->data + i * 2u, 2u); return f16_to_double(h); }
    case LANA_TENSOR_BF16: { uint16_t h; memcpy(&h, t->data + i * 2u, 2u); return bf16_to_double(h); }
    case LANA_TENSOR_COMPLEX: { double d; memcpy(&d, t->data + i * 16u, 8u); return d; }
    default: { double d; memcpy(&d, t->data + i * 8u, 8u); return d; } /* F64 */
    }
}

/* LIP-027: read the imaginary part of element `i` (0.0 for a real dtype). */
double tensor_get_imag(const LanaTensor *t, size_t i) {
    if (t->dtype != LANA_TENSOR_COMPLEX) return 0.0;
    double d;
    memcpy(&d, t->data + i * 16u + 8u, 8u);
    return d;
}

/* LIP-027: write the real part of element `i`, converting from double to the
 * storage dtype (round-to-nearest-even for f16/bf16). */
static void tensor_set_real(LanaTensor *t, size_t i, double v) {
    switch (t->dtype) {
    case LANA_TENSOR_F32: { float f = (float)v; memcpy(t->data + i * 4u, &f, 4u); break; }
    case LANA_TENSOR_F16: { uint16_t h = f64_to_f16_bits(v); memcpy(t->data + i * 2u, &h, 2u); break; }
    case LANA_TENSOR_BF16: { uint16_t h = f64_to_bf16_bits(v); memcpy(t->data + i * 2u, &h, 2u); break; }
    case LANA_TENSOR_COMPLEX: { memcpy(t->data + i * 16u, &v, 8u); break; }
    default: { memcpy(t->data + i * 8u, &v, 8u); break; } /* F64 */
    }
}

/* LIP-027: write the imaginary part of element `i` (no-op for a real dtype). */
static void tensor_set_imag(LanaTensor *t, size_t i, double v) {
    if (t->dtype != LANA_TENSOR_COMPLEX) return;
    memcpy(t->data + i * 16u + 8u, &v, 8u);
}

static LanaTensor *tensor_new_dtype(LanaVM *vm, size_t ndim, const size_t *shape,
                                    LanaTensorDtype dtype);

/* LIP-027: cast a tensor to another real dtype. Same-dtype is a no-op (returns
 * the same tensor). Cast to/from complex is out of scope (LANA_ERR_TYPE). The
 * result is a fresh base tensor; a view is never mutated. */
static LanaError tensor_cast(LanaVM *vm, const LanaTensor *t, LanaTensorDtype dtype,
                             Value *out) {
    if (t->dtype == LANA_TENSOR_COMPLEX || dtype == LANA_TENSOR_COMPLEX)
        return LANA_ERR_TYPE;
    if (t->dtype == dtype) {
        *out = lana_value_tensor((LanaTensor *)t);
        return LANA_OK;
    }
    LanaTensor *r = tensor_new_dtype(vm, t->ndim, t->shape, dtype);
    if (r == NULL) return LANA_ERR_OOM;
    size_t total = 1;
    for (size_t i = 0; i < t->ndim; ++i) total *= t->shape[i];
    size_t *idx = lana_vm_alloc(vm, t->ndim * sizeof(*idx));
    if (idx == NULL && t->ndim > 0) return LANA_ERR_OOM;
    for (size_t lin = 0; lin < total; ++lin) {
        size_t rem = lin;
        for (ssize_t i = (ssize_t)t->ndim - 1; i >= 0; --i) {
            idx[i] = (t->shape[i] == 0) ? 0 : rem % t->shape[i];
            rem /= t->shape[i];
        }
        size_t src = 0;
        for (size_t i = 0; i < t->ndim; ++i) src += idx[i] * t->strides[i];
        tensor_set_real(r, lin, tensor_get_real(t, t->offset + src));
    }
    *out = lana_value_tensor(r);
    return LANA_OK;
}

/* Allocate a zero-initialized tensor with the given shape and dtype. `shape`
 * is copied; the caller owns the input array. Returns NULL on allocation
 * failure or shape overflow. All memory is GC-managed so the tensor's buffer
 * counts against the VM memory limit. */
static LanaTensor *tensor_new_dtype(LanaVM *vm, size_t ndim, const size_t *shape,
                                    LanaTensorDtype dtype) {
    if (ndim > LANA_TENSOR_MAX_RANK) return NULL;
    bool is_complex = (dtype == LANA_TENSOR_COMPLEX);
    LanaTensor *tensor = lana_vm_alloc(vm, sizeof(*tensor));
    if (tensor == NULL) return NULL;
    tensor->ndim = ndim;
    tensor->is_complex = is_complex;
    tensor->dtype = dtype;
    tensor->is_state = false;
    tensor->shape = NULL;
    tensor->strides = NULL;
    tensor->data = NULL;
    tensor->offset = 0;
    tensor->base = NULL;
    if (ndim > 0) {
        tensor->shape = lana_vm_alloc(vm, ndim * sizeof(*tensor->shape));
        tensor->strides = lana_vm_alloc(vm, ndim * sizeof(*tensor->strides));
        if (tensor->shape == NULL || tensor->strides == NULL) return NULL;
        size_t stride = 1;
        for (ssize_t i = (ssize_t)ndim - 1; i >= 0; --i) {
            tensor->shape[i] = shape[i];
            tensor->strides[i] = stride;
            if (shape[i] != 0 && stride > SIZE_MAX / shape[i]) return NULL;
            stride *= shape[i];
        }
    }
    size_t total = 1;
    for (size_t i = 0; i < ndim; ++i) {
        if (shape[i] == 0) { total = 0; break; }
        if (total > SIZE_MAX / shape[i]) return NULL;
        total *= shape[i];
    }
    size_t components = is_complex ? 2u : 1u;
    size_t width = tensor_elem_bytes(tensor);
    if (total > SIZE_MAX / components / width) return NULL;
    size_t elem_count = total * components;
    if (elem_count > 0) {
        tensor->data = lana_vm_alloc(vm, elem_count * width);
        if (tensor->data == NULL) return NULL;
        memset(tensor->data, 0, elem_count * width);
    }
    return tensor;
}

/* Allocate a zero-initialized tensor with the given shape, defaulting to F64
 * (or COMPLEX when `is_complex`). Kept for the common case; dtype-specific
 * callers use `tensor_new_dtype`. */
static LanaTensor *tensor_new(LanaVM *vm, size_t ndim, const size_t *shape, bool is_complex) {
    return tensor_new_dtype(vm, ndim, shape, is_complex ? LANA_TENSOR_COMPLEX : LANA_TENSOR_F64);
}

static LanaError tensor_dimension(double n, size_t *dimension) {
    /* SIZE_MAX rounds up on a 64-bit host; use the exclusive power-of-two
     * bound on every host so the floating-to-integer conversion is defined. */
    double limit = ((double)(SIZE_MAX / 2u) + 1.0) * 2.0;
    if (!isfinite(n) || n < 0.0 || floor(n) != n || n >= limit)
        return LANA_ERR_INVALID_PARAMETERS;
    *dimension = (size_t)n;
    return LANA_OK;
}

/* Extract a shape from a VAL_ARRAY of numbers into VM-accounted scratch.
 * Returns LANA_OK, or LANA_ERR_TYPE / LANA_ERR_INVALID_PARAMETERS / OOM.
 * The buffer is GC memory; callers must not free it. */
static LanaError tensor_shape_from_array(LanaVM *vm, const Value *v, size_t *ndim, size_t **shape) {
    if (v->type != VAL_ARRAY) return LANA_ERR_TYPE;
    LanaArray *arr = v->as.array;
    if (arr->count > LANA_TENSOR_MAX_RANK) return LANA_ERR_INVALID_PARAMETERS;
    *ndim = arr->count;
    *shape = NULL;
    if (arr->count == 0) return LANA_OK;
    *shape = lana_vm_alloc(vm, arr->count * sizeof(**shape));
    if (*shape == NULL) return LANA_ERR_OOM;
    for (size_t i = 0; i < arr->count; ++i) {
        const Value *item = &arr->items[i];
        if (item->type != VAL_NUMBER) return LANA_ERR_TYPE;
        LanaError error = tensor_dimension(item->as.number, &(*shape)[i]);
        if (error != LANA_OK) return error;
    }
    return LANA_OK;
}

/* Compute the row-major broadcast strides of `t` against an output shape of
 * `out_ndim` dims. A broadcast dimension (size 1 expanded to >1, or a missing
 * leading dimension) gets stride 0. */
static void tensor_broadcast_strides(const LanaTensor *t, size_t out_ndim,
                                     const size_t *out_shape, size_t *strides) {
    size_t offset = out_ndim - t->ndim;
    for (size_t i = 0; i < out_ndim; ++i) {
        if (i < offset) {
            strides[i] = 0;
        } else {
            size_t ti = i - offset;
            strides[i] = (t->shape[ti] == 1 && out_shape[i] != 1) ? 0 : t->strides[ti];
        }
    }
}

/* Element-wise binary op with NumPy broadcasting. op: 0=add 1=sub 2=mul 3=div. */
static LanaError tensor_elementwise(LanaVM *vm, const LanaTensor *a, const LanaTensor *b,
                                    int op, Value *out) {
    /* LIP-027: element-wise requires the same dtype; no silent promotion. */
    if (a->dtype != b->dtype) return LANA_ERR_TYPE;
    size_t out_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    size_t *out_shape = lana_vm_alloc(vm, out_ndim * sizeof(*out_shape));
    if (out_shape == NULL && out_ndim > 0) return LANA_ERR_OOM;
    for (size_t i = 0; i < out_ndim; ++i) {
        size_t ai = (i < out_ndim - a->ndim) ? 1 : a->shape[i - (out_ndim - a->ndim)];
        size_t bi = (i < out_ndim - b->ndim) ? 1 : b->shape[i - (out_ndim - b->ndim)];
        if (ai == bi) out_shape[i] = ai;
        else if (ai == 1) out_shape[i] = bi;
        else if (bi == 1) out_shape[i] = ai;
        else return LANA_ERR_INVALID_PARAMETERS;
    }
    LanaTensor *r = tensor_new_dtype(vm, out_ndim, out_shape, a->dtype);
    if (r == NULL) return LANA_ERR_OOM;
    size_t *a_strides = lana_vm_alloc(vm, out_ndim * sizeof(*a_strides));
    size_t *b_strides = lana_vm_alloc(vm, out_ndim * sizeof(*b_strides));
    if ((a_strides == NULL || b_strides == NULL) && out_ndim > 0) return LANA_ERR_OOM;
    tensor_broadcast_strides(a, out_ndim, out_shape, a_strides);
    tensor_broadcast_strides(b, out_ndim, out_shape, b_strides);
    size_t total = 1;
    for (size_t i = 0; i < out_ndim; ++i) total *= out_shape[i];
    size_t *idx = lana_vm_alloc(vm, out_ndim * sizeof(*idx));
    if (idx == NULL && out_ndim > 0) return LANA_ERR_OOM;
    bool complex = a->is_complex;
    for (size_t lin = 0; lin < total; ++lin) {
        size_t rem = lin;
        for (ssize_t i = (ssize_t)out_ndim - 1; i >= 0; --i) {
            idx[i] = (out_shape[i] == 0) ? 0 : rem % out_shape[i];
            rem /= out_shape[i];
        }
        size_t ai = 0, bi = 0;
        for (size_t i = 0; i < out_ndim; ++i) { ai += idx[i] * a_strides[i]; bi += idx[i] * b_strides[i]; }
        if (complex) {
            double ar = tensor_get_real(a, a->offset + ai), aim = tensor_get_imag(a, a->offset + ai);
            double br = tensor_get_real(b, b->offset + bi), bim = tensor_get_imag(b, b->offset + bi);
            double rr, ri;
            switch (op) {
                case 0: rr = ar + br; ri = aim + bim; break;
                case 1: rr = ar - br; ri = aim - bim; break;
                case 2: rr = ar * br - aim * bim; ri = ar * bim + aim * br; break;
                default: {
                    double den = br * br + bim * bim;
                    if (den == 0.0) return LANA_ERR_INVALID_PARAMETERS;
                    rr = (ar * br + aim * bim) / den;
                    ri = (aim * br - ar * bim) / den;
                    break;
                }
            }
            tensor_set_real(r, lin, rr); tensor_set_imag(r, lin, ri);
        } else {
            double av = tensor_get_real(a, a->offset + ai), bv = tensor_get_real(b, b->offset + bi);
            switch (op) {
                case 0: tensor_set_real(r, lin, av + bv); break;
                case 1: tensor_set_real(r, lin, av - bv); break;
                case 2: tensor_set_real(r, lin, av * bv); break;
                default:
                    if (bv == 0.0) return LANA_ERR_INVALID_PARAMETERS;
                    tensor_set_real(r, lin, av / bv); break;
            }
        }
    }
    *out = lana_value_tensor(r);
    return LANA_OK;
}

/* LIP-027: element-wise between a tensor and a scalar number. The scalar
 * adopts the tensor's dtype (cast to it), so `f16_tensor + 1.0` is "f16". */
static LanaError tensor_elementwise_scalar(LanaVM *vm, const LanaTensor *t, double s,
                                           int op, Value *out) {
    LanaTensor *r = tensor_new_dtype(vm, t->ndim, t->shape, t->dtype);
    if (r == NULL) return LANA_ERR_OOM;
    size_t total = 1;
    for (size_t i = 0; i < t->ndim; ++i) total *= t->shape[i];
    size_t *idx = lana_vm_alloc(vm, t->ndim * sizeof(*idx));
    if (idx == NULL && t->ndim > 0) return LANA_ERR_OOM;
    bool complex = t->is_complex;
    for (size_t lin = 0; lin < total; ++lin) {
        size_t rem = lin;
        for (ssize_t i = (ssize_t)t->ndim - 1; i >= 0; --i) {
            idx[i] = (t->shape[i] == 0) ? 0 : rem % t->shape[i];
            rem /= t->shape[i];
        }
        size_t src = 0;
        for (size_t i = 0; i < t->ndim; ++i) src += idx[i] * t->strides[i];
        if (complex) {
            double ar = tensor_get_real(t, t->offset + src), aim = tensor_get_imag(t, t->offset + src);
            double rr, ri;
            switch (op) {
                case 0: rr = ar + s; ri = aim; break;
                case 1: rr = ar - s; ri = aim; break;
                case 2: rr = ar * s; ri = aim * s; break;
                default:
                    if (s == 0.0) return LANA_ERR_INVALID_PARAMETERS;
                    rr = ar / s; ri = aim / s; break;
            }
            tensor_set_real(r, lin, rr); tensor_set_imag(r, lin, ri);
        } else {
            double v = tensor_get_real(t, t->offset + src);
            double result;
            switch (op) {
                case 0: result = v + s; break;
                case 1: result = v - s; break;
                case 2: result = v * s; break;
                default:
                    if (s == 0.0) return LANA_ERR_INVALID_PARAMETERS;
                    result = v / s; break;
            }
            tensor_set_real(r, lin, result);
        }
    }
    *out = lana_value_tensor(r);
    return LANA_OK;
}

/* General matmul following NumPy semantics: 1d x 1d is a dot product, 1d x Nd
 * and Nd x 1d are vector-matrix, and Nd x Md contracts the last axis of the
 * left with the second-to-last of the right, broadcasting batch dims. */
static LanaError tensor_matmul(LanaVM *vm, const LanaTensor *a, const LanaTensor *b,
                               LanaTensorDtype out_dtype, Value *out) {
    if (a->is_complex != b->is_complex) return LANA_ERR_TYPE;
    if (a->ndim == 0 || b->ndim == 0) return LANA_ERR_INVALID_PARAMETERS;
    bool complex = a->is_complex;
    /* LIP-027: the output dtype must agree with the complexness of the inputs:
     * complex inputs require a complex output, real inputs a real output. */
    if (complex != (out_dtype == LANA_TENSOR_COMPLEX)) return LANA_ERR_TYPE;
    size_t a_ndim = a->ndim, b_ndim = b->ndim;
    size_t a_rows = (a_ndim == 1) ? 1 : a->shape[a_ndim - 2];
    size_t a_cols = a->shape[a_ndim - 1];
    size_t b_rows = (b_ndim == 1) ? b->shape[0] : b->shape[b_ndim - 2];
    size_t b_cols = (b_ndim == 1) ? 1 : b->shape[b_ndim - 1];
    if (a_cols != b_rows) return LANA_ERR_INVALID_PARAMETERS;
    size_t K = a_cols;
    size_t a_batch = (a_ndim <= 2) ? 0 : a_ndim - 2;
    size_t b_batch = (b_ndim <= 2) ? 0 : b_ndim - 2;
    size_t batch_ndim = a_batch > b_batch ? a_batch : b_batch;
    size_t *batch_shape = lana_vm_alloc(vm, batch_ndim * sizeof(*batch_shape));
    if (batch_shape == NULL && batch_ndim > 0) return LANA_ERR_OOM;
    for (size_t i = 0; i < batch_ndim; ++i) {
        size_t ai = (i < batch_ndim - a_batch) ? 1 : a->shape[i - (batch_ndim - a_batch)];
        size_t bi = (i < batch_ndim - b_batch) ? 1 : b->shape[i - (batch_ndim - b_batch)];
        if (ai == bi) batch_shape[i] = ai;
        else if (ai == 1) batch_shape[i] = bi;
        else if (bi == 1) batch_shape[i] = ai;
        else return LANA_ERR_INVALID_PARAMETERS;
    }
    bool a_vec = (a_ndim == 1), b_vec = (b_ndim == 1);
    size_t out_ndim = batch_ndim + (a_vec ? 0 : 1) + (b_vec ? 0 : 1);
    size_t *out_shape = lana_vm_alloc(vm, out_ndim * sizeof(*out_shape));
    if (out_shape == NULL && out_ndim > 0) return LANA_ERR_OOM;
    for (size_t i = 0; i < batch_ndim; ++i) out_shape[i] = batch_shape[i];
    size_t pos = batch_ndim;
    if (!a_vec) out_shape[pos++] = a_rows;
    if (!b_vec) out_shape[pos++] = b_cols;
    LanaTensor *r = tensor_new_dtype(vm, out_ndim, out_shape, out_dtype);
    if (r == NULL) return LANA_ERR_OOM;
    size_t *a_batch_strides = lana_vm_alloc(vm, batch_ndim * sizeof(*a_batch_strides));
    size_t *b_batch_strides = lana_vm_alloc(vm, batch_ndim * sizeof(*b_batch_strides));
    if ((a_batch_strides == NULL || b_batch_strides == NULL) && batch_ndim > 0) return LANA_ERR_OOM;
    for (size_t i = 0; i < batch_ndim; ++i) {
        size_t a_dim = (i < batch_ndim - a_batch) ? 1 : a->shape[i - (batch_ndim - a_batch)];
        size_t a_stride = (i < batch_ndim - a_batch) ? 0 : a->strides[i - (batch_ndim - a_batch)];
        a_batch_strides[i] = (a_dim == 1 && batch_shape[i] != 1) ? 0 : a_stride;
        size_t b_dim = (i < batch_ndim - b_batch) ? 1 : b->shape[i - (batch_ndim - b_batch)];
        size_t b_stride = (i < batch_ndim - b_batch) ? 0 : b->strides[i - (batch_ndim - b_batch)];
        b_batch_strides[i] = (b_dim == 1 && batch_shape[i] != 1) ? 0 : b_stride;
    }
    size_t batch_total = 1;
    for (size_t i = 0; i < batch_ndim; ++i) batch_total *= batch_shape[i];
    size_t a_row_stride = (a_ndim == 1) ? 0 : a->strides[a_ndim - 2];
    size_t a_col_stride = a->strides[a_ndim - 1];
    size_t b_row_stride = (b_ndim == 1) ? b->strides[0] : b->strides[b_ndim - 2];
    size_t b_col_stride = (b_ndim == 1) ? 0 : b->strides[b_ndim - 1];
    size_t m = a_rows, n = b_cols;
    size_t mult = complex ? 2u : 1u;
    /* BLAS needs row-major contiguous cores. A promoted vector is contiguous
     * when its one live stride is 1; a matrix operand when its column stride
     * is 1. Anything else is gathered into accounted scratch per batch
     * element (LIP-004 section 5). */
    /* LIP-027: a non-f64 real dtype stores compact bytes (2/4 per element), so
     * the direct `double*` fast path is invalid; gather it into a double
     * scratch buffer via tensor_get_real. Complex (interleaved doubles) and
     * f64 read directly. */
    bool pack_a = a_col_stride != 1u || (a->dtype != LANA_TENSOR_F64 && a->dtype != LANA_TENSOR_COMPLEX);
    bool pack_b = b_col_stride != 1u || (b->dtype != LANA_TENSOR_F64 && b->dtype != LANA_TENSOR_COMPLEX);
    /* Leading dimensions for direct (non-packed) operands: the true row
     * stride of the view, which BLAS accepts instead of a copy. A promoted
     * vector has one row (lda = K) or one column (ldb = its row stride). */
    size_t a_ld = pack_a ? K : (a_vec ? K : a_row_stride);
    size_t b_ld = pack_b ? n : b_row_stride;
    size_t a_core = m * K * mult;
    size_t b_core = K * n * mult;
    double *a_packed = NULL, *b_packed = NULL;
    if (pack_a && a_core > 0) {
        a_packed = lana_vm_alloc(vm, a_core * sizeof(double));
        if (a_packed == NULL) return LANA_ERR_OOM;
    }
    if (pack_b && b_core > 0) {
        b_packed = lana_vm_alloc(vm, b_core * sizeof(double));
        if (b_packed == NULL) return LANA_ERR_OOM;
    }
    /* LIP-027: the backend gemm accumulates in binary64. When the output dtype
     * is narrower than f64 (f32/f16/bf16), the result is staged in a scratch
     * buffer and converted element-wise into the compact output buffer. */
    size_t c_core = m * n * mult;
    double *c_scratch = NULL;
    if (c_core > 0) {
        c_scratch = lana_vm_alloc(vm, c_core * sizeof(double));
        if (c_scratch == NULL) return LANA_ERR_OOM;
    }
    for (size_t batch = 0; batch < batch_total; ++batch) {
        size_t rem = batch, a_off = 0, b_off = 0;
        for (ssize_t i = (ssize_t)batch_ndim - 1; i >= 0; --i) {
            size_t bi = (batch_shape[i] == 0) ? 0 : rem % batch_shape[i];
            rem /= batch_shape[i];
            a_off += bi * a_batch_strides[i];
            b_off += bi * b_batch_strides[i];
        }
        const double *a_ptr = (const double*)a->data + (a->offset + a_off) * mult;
        const double *b_ptr = (const double*)b->data + (b->offset + b_off) * mult;
        if (a_packed != NULL) {
            for (size_t i = 0; i < m; ++i) {
                for (size_t k = 0; k < K; ++k) {
                    size_t elem = a->offset + a_off + i * a_row_stride + k * a_col_stride;
                    size_t dst = (i * K + k) * mult;
                    a_packed[dst] = tensor_get_real(a, elem);
                    if (mult == 2u) a_packed[dst + 1] = tensor_get_imag(a, elem);
                }
            }
            a_ptr = a_packed;
        }
        if (b_packed != NULL) {
            for (size_t k = 0; k < K; ++k) {
                for (size_t j = 0; j < n; ++j) {
                    size_t elem = b->offset + b_off + k * b_row_stride + j * b_col_stride;
                    size_t dst = (k * n + j) * mult;
                    b_packed[dst] = tensor_get_real(b, elem);
                    if (mult == 2u) b_packed[dst + 1] = tensor_get_imag(b, elem);
                }
            }
            b_ptr = b_packed;
        }
        LanaGemmCall gemm = { m, K, n, complex, matmul_accumulation_fp32(a, b),
                              a_ptr, a_ld, b_ptr, b_ld, c_scratch };
        lana_backend_gemm(&gemm);
        /* Stage the binary64 result into the compact output buffer. */
        for (size_t i = 0; i < c_core; ++i) {
            size_t elem = batch * m * n + i / mult;
            if (mult == 2u) {
                if (i % 2u == 0u) tensor_set_real(r, elem, c_scratch[i]);
                else tensor_set_imag(r, elem, c_scratch[i]);
            } else {
                tensor_set_real(r, elem, c_scratch[i]);
            }
        }
    }
    *out = lana_value_tensor(r);
    return LANA_OK;
}

/* Explicit GPU matmul (LIP-004 section 5). Mirrors `tensor_matmul`'s shape,
 * batch, and broadcasting logic, but dispatches each batch element to the
 * Metal device in float32. Complex operands are rejected (float32 complex
 * Metal is a later extension); the binary64 -> float32 downcast is explicit
 * and the float32 -> binary64 upcast is exact. The caller attaches the
 * APPROXIMATE derivation. */
static LanaError tensor_gpu_matmul(LanaVM *vm, const LanaTensor *a, const LanaTensor *b, Value *out) {
    if (a->is_complex || b->is_complex) return LANA_ERR_TYPE;
    if (a->ndim == 0 || b->ndim == 0) return LANA_ERR_INVALID_PARAMETERS;
    size_t a_ndim = a->ndim, b_ndim = b->ndim;
    size_t a_rows = (a_ndim == 1) ? 1 : a->shape[a_ndim - 2];
    size_t a_cols = a->shape[a_ndim - 1];
    size_t b_rows = (b_ndim == 1) ? b->shape[0] : b->shape[b_ndim - 2];
    size_t b_cols = (b_ndim == 1) ? 1 : b->shape[b_ndim - 1];
    if (a_cols != b_rows) return LANA_ERR_INVALID_PARAMETERS;
    size_t K = a_cols;
    size_t a_batch = (a_ndim <= 2) ? 0 : a_ndim - 2;
    size_t b_batch = (b_ndim <= 2) ? 0 : b_ndim - 2;
    size_t batch_ndim = a_batch > b_batch ? a_batch : b_batch;
    size_t *batch_shape = lana_vm_alloc(vm, batch_ndim * sizeof(*batch_shape));
    if (batch_shape == NULL && batch_ndim > 0) return LANA_ERR_OOM;
    for (size_t i = 0; i < batch_ndim; ++i) {
        size_t ai = (i < batch_ndim - a_batch) ? 1 : a->shape[i - (batch_ndim - a_batch)];
        size_t bi = (i < batch_ndim - b_batch) ? 1 : b->shape[i - (batch_ndim - b_batch)];
        if (ai == bi) batch_shape[i] = ai;
        else if (ai == 1) batch_shape[i] = bi;
        else if (bi == 1) batch_shape[i] = ai;
        else return LANA_ERR_INVALID_PARAMETERS;
    }
    bool a_vec = (a_ndim == 1), b_vec = (b_ndim == 1);
    size_t out_ndim = batch_ndim + (a_vec ? 0 : 1) + (b_vec ? 0 : 1);
    size_t *out_shape = lana_vm_alloc(vm, out_ndim * sizeof(*out_shape));
    if (out_shape == NULL && out_ndim > 0) return LANA_ERR_OOM;
    for (size_t i = 0; i < batch_ndim; ++i) out_shape[i] = batch_shape[i];
    size_t pos = batch_ndim;
    if (!a_vec) out_shape[pos++] = a_rows;
    if (!b_vec) out_shape[pos++] = b_cols;
    LanaTensor *r = tensor_new(vm, out_ndim, out_shape, false);
    if (r == NULL) return LANA_ERR_OOM;
    size_t *a_batch_strides = lana_vm_alloc(vm, batch_ndim * sizeof(*a_batch_strides));
    size_t *b_batch_strides = lana_vm_alloc(vm, batch_ndim * sizeof(*b_batch_strides));
    if ((a_batch_strides == NULL || b_batch_strides == NULL) && batch_ndim > 0) return LANA_ERR_OOM;
    for (size_t i = 0; i < batch_ndim; ++i) {
        size_t a_dim = (i < batch_ndim - a_batch) ? 1 : a->shape[i - (batch_ndim - a_batch)];
        size_t a_stride = (i < batch_ndim - a_batch) ? 0 : a->strides[i - (batch_ndim - a_batch)];
        a_batch_strides[i] = (a_dim == 1 && batch_shape[i] != 1) ? 0 : a_stride;
        size_t b_dim = (i < batch_ndim - b_batch) ? 1 : b->shape[i - (batch_ndim - b_batch)];
        size_t b_stride = (i < batch_ndim - b_batch) ? 0 : b->strides[i - (batch_ndim - b_batch)];
        b_batch_strides[i] = (b_dim == 1 && batch_shape[i] != 1) ? 0 : b_stride;
    }
    size_t batch_total = 1;
    for (size_t i = 0; i < batch_ndim; ++i) batch_total *= batch_shape[i];
    size_t a_row_stride = (a_ndim == 1) ? 0 : a->strides[a_ndim - 2];
    size_t a_col_stride = a->strides[a_ndim - 1];
    size_t b_row_stride = (b_ndim == 1) ? b->strides[0] : b->strides[b_ndim - 2];
    size_t b_col_stride = (b_ndim == 1) ? 0 : b->strides[b_ndim - 1];
    size_t m = a_rows, n = b_cols;
    /* Metal needs contiguous row-major float32 cores, so the binary64 operands
     * are downcast into float32 scratch per batch element (LIP-004 section 5:
     * the downcast is explicit, never silent). */
    float *a_float = lana_vm_alloc(vm, m * K * sizeof(float));
    float *b_float = lana_vm_alloc(vm, K * n * sizeof(float));
    float *c_float = lana_vm_alloc(vm, m * n * sizeof(float));
    if ((a_float == NULL && m * K > 0) || (b_float == NULL && K * n > 0) ||
        (c_float == NULL && m * n > 0)) return LANA_ERR_OOM;
    for (size_t batch = 0; batch < batch_total; ++batch) {
        size_t rem = batch, a_off = 0, b_off = 0;
        for (ssize_t i = (ssize_t)batch_ndim - 1; i >= 0; --i) {
            size_t bi = (batch_shape[i] == 0) ? 0 : rem % batch_shape[i];
            rem /= batch_shape[i];
            a_off += bi * a_batch_strides[i];
            b_off += bi * b_batch_strides[i];
        }
        for (size_t i = 0; i < m; ++i)
            for (size_t k = 0; k < K; ++k)
                a_float[i * K + k] = (float)tensor_get_real(a, a->offset + a_off + i * a_row_stride + k * a_col_stride);
        for (size_t k = 0; k < K; ++k)
            for (size_t j = 0; j < n; ++j)
                b_float[k * n + j] = (float)tensor_get_real(b, b->offset + b_off + k * b_row_stride + j * b_col_stride);
        if (!lana_metal_sgemm(m, K, n, a_float, b_float, c_float))
            return LANA_ERR_UNSUPPORTED_OPERATION;
        for (size_t i = 0; i < m * n; ++i)
            tensor_set_real(r, batch * m * n + i, (double)c_float[i]);
    }
    *out = lana_value_tensor(r);
    return LANA_OK;
}

/* Reduce one strided fiber. op: 0=sum 1=mean 2=max 3=min. `offset` is
 * relative to the tensor's first element; the tensor's own view offset is
 * folded in here so every caller is automatically view-correct. */
/* LIP-027: sum/mean accumulate in binary32 for f16/bf16 inputs. */
static bool reduction_accumulation_fp32(const LanaTensor *t) {
    return t->dtype == LANA_TENSOR_F16 || t->dtype == LANA_TENSOR_BF16;
}

static LanaError tensor_reduce_fiber(const LanaTensor *t, size_t offset,
                                     size_t count, size_t stride, int op,
                                     bool fp32, double *re, double *im) {
    if (t->is_complex && op >= 2) return LANA_ERR_TYPE;
    if (count == 0 && op != 0) return LANA_ERR_INVALID_PARAMETERS;
    double acc = op == 2 ? -INFINITY : op == 3 ? INFINITY : 0.0;
    float acc32 = 0.0f;
    double imaginary = 0.0;
    for (size_t i = 0; i < count; ++i) {
        size_t index = t->offset + offset + i * stride;
        double v = tensor_get_real(t, index);
        if (!isfinite(v)) return LANA_ERR_INVALID_PARAMETERS;
        if (op < 2) {
            if (fp32) acc32 += (float)v;
            else acc += v;
        }
        else if (op == 2 ? v > acc : v < acc) acc = v;
        if (t->is_complex) {
            double component = tensor_get_imag(t, index);
            if (!isfinite(component)) return LANA_ERR_INVALID_PARAMETERS;
            imaginary += component;
        }
    }
    if (fp32 && op < 2) acc = (double)acc32;
    if (op == 1) { acc /= (double)count; imaginary /= (double)count; }
    if (!isfinite(acc) || !isfinite(imaginary)) return LANA_ERR_INVALID_PARAMETERS;
    *re = acc; *im = imaginary;
    return LANA_OK;
}

static LanaError tensor_reduce(LanaVM *vm, const LanaTensor *t, int op,
                               const Value *axis_value, Value *out) {
    if (t->is_complex && op >= 2) return LANA_ERR_TYPE;
    if (axis_value == NULL) {
        /* Full reduction over every element. A view may be non-contiguous, so
         * traverse the strided layout per dimension instead of assuming a
         * flat fiber. */
        size_t total = 1;
        for (size_t i = 0; i < t->ndim; ++i) total *= t->shape[i];
        if (total == 0 && op != 0) return LANA_ERR_INVALID_PARAMETERS;
        bool fp32 = reduction_accumulation_fp32(t);
        double acc = op == 2 ? -INFINITY : op == 3 ? INFINITY : 0.0;
        float acc32 = 0.0f;
        double imaginary = 0.0;
        for (size_t lin = 0; lin < total; ++lin) {
            size_t rem = lin, index = t->offset;
            for (size_t d = t->ndim; d-- > 0;) {
                index += (t->shape[d] == 0 ? 0 : rem % t->shape[d]) * t->strides[d];
                rem /= t->shape[d];
            }
            double v = tensor_get_real(t, index);
            if (!isfinite(v)) return LANA_ERR_INVALID_PARAMETERS;
            if (op < 2) {
                if (fp32) acc32 += (float)v;
                else acc += v;
            }
            else if (op == 2 ? v > acc : v < acc) acc = v;
            if (t->is_complex) {
                double component = tensor_get_imag(t, index);
                if (!isfinite(component)) return LANA_ERR_INVALID_PARAMETERS;
                imaginary += component;
            }
        }
        if (fp32 && op < 2) acc = (double)acc32;
        if (op == 1) { acc /= (double)total; imaginary /= (double)total; }
        if (!isfinite(acc) || !isfinite(imaginary)) return LANA_ERR_INVALID_PARAMETERS;
        if (!t->is_complex) { *out = lana_value_number(acc); return LANA_OK; }
        LanaTensor *r = tensor_new(vm, 0, NULL, true);
        if (r == NULL) return LANA_ERR_OOM;
        tensor_set_real(r, 0, acc); tensor_set_imag(r, 0, imaginary);
        *out = lana_value_tensor(r);
        return LANA_OK;
    }
    if (axis_value->type != VAL_NUMBER) return LANA_ERR_TYPE;
    double axis_number = axis_value->as.number;
    if (!isfinite(axis_number) || floor(axis_number) != axis_number ||
        axis_number < -(double)t->ndim || axis_number >= (double)t->ndim)
        return LANA_ERR_INVALID_PARAMETERS;
    size_t axis = (size_t)(axis_number < 0 ? axis_number + (double)t->ndim : axis_number);
    size_t count = t->shape[axis];
    if (count == 0 && op != 0) return LANA_ERR_INVALID_PARAMETERS;
    size_t shape[LANA_TENSOR_MAX_RANK];
    for (size_t i = 0, j = 0; i < t->ndim; ++i)
        if (i != axis) shape[j++] = t->shape[i];
    LanaTensor *r = tensor_new_dtype(vm, t->ndim - 1, shape, t->dtype);
    if (r == NULL) return LANA_ERR_OOM;
    size_t total = 1;
    for (size_t i = 0; i < r->ndim; ++i) total *= r->shape[i];
    bool fp32 = reduction_accumulation_fp32(t);
    for (size_t i = 0; i < total; ++i) {
        size_t offset = 0, remaining = i;
        for (size_t d = t->ndim; d-- > 0;) {
            if (d == axis) continue;
            offset += (remaining % t->shape[d]) * t->strides[d];
            remaining /= t->shape[d];
        }
        double re, im;
        LanaError error = tensor_reduce_fiber(t, offset, count, t->strides[axis], op, fp32, &re, &im);
        if (error != LANA_OK) return error;
        tensor_set_real(r, i, re);
        if (t->is_complex) tensor_set_imag(r, i, im);
    }
    *out = lana_value_tensor(r);
    return LANA_OK;
}

/* ===== LIP-008 uncertainty-carrying tensor helpers ===== */

/* Detect an uncertain tensor: a VAL_MAP with exactly the "prediction" and
 * "uncertainty" keys, both VAL_TENSOR. A bare VAL_TENSOR is certain (pred set,
 * var NULL, is_uncertain false). Any other value is a type error. */
static LanaError tensor_uncertainty_unpack(const Value *v, const LanaTensor **pred,
                                           const LanaTensor **var, bool *is_uncertain) {
    *is_uncertain = false;
    *pred = NULL;
    *var = NULL;
    if (v->type == VAL_TENSOR) { *pred = v->as.tensor; return LANA_OK; }
    if (v->type != VAL_MAP) return LANA_ERR_TYPE;
    const LanaMap *map = v->as.map;
    if (map->count != 2u) return LANA_ERR_TYPE;
    Value pred_value, var_value;
    if (lana_map_get(map, "prediction", &pred_value) != LANA_OK) return LANA_ERR_TYPE;
    if (lana_map_get(map, "uncertainty", &var_value) != LANA_OK) return LANA_ERR_TYPE;
    if (pred_value.type != VAL_TENSOR || var_value.type != VAL_TENSOR) return LANA_ERR_TYPE;
    *pred = pred_value.as.tensor;
    *var = var_value.as.tensor;
    *is_uncertain = true;
    return LANA_OK;
}

/* Build the { prediction, uncertainty } map result. */
static LanaError tensor_uncertain_result(LanaVM *vm, LanaTensor *pred, LanaTensor *var, Value *out) {
    LanaMap *map;
    LanaError error = lana_map_new(vm, 2u, &map);
    if (error != LANA_OK) return error;
    Value pred_value = lana_value_tensor(pred);
    Value var_value = lana_value_tensor(var);
    error = lana_map_set(vm, map, "prediction", &pred_value, false);
    if (error != LANA_OK) return error;
    error = lana_map_set(vm, map, "uncertainty", &var_value, false);
    if (error != LANA_OK) return error;
    *out = lana_value_map(map);
    return LANA_OK;
}

/* A zero real tensor with the same shape as `t` (the variance of a certain
 * operand). */
static LanaTensor *tensor_zeros_like(LanaVM *vm, const LanaTensor *t) {
    return tensor_new(vm, t->ndim, t->shape, false);
}

/* Whether every element of a freshly-allocated (contiguous) tensor is finite. */
static bool tensor_all_finite(const LanaTensor *t) {
    size_t total = 1;
    for (size_t i = 0; i < t->ndim; ++i) total *= t->shape[i];
    size_t mult = t->is_complex ? 2u : 1u;
    for (size_t i = 0; i < total * mult; ++i) {
        if (!isfinite(tensor_get_real(t, t->offset + i))) return false;
    }
    return true;
}

/* First-order variance propagation for element-wise + - * / (real-only). */
static LanaError tensor_elementwise_uncertain(LanaVM *vm, const LanaTensor *a_pred,
                                              const LanaTensor *a_var, const LanaTensor *b_pred,
                                              const LanaTensor *b_var, int op, Value *out) {
    if (a_pred->is_complex || b_pred->is_complex || a_var->is_complex || b_var->is_complex)
        return LANA_ERR_TYPE;
    Value pred_value;
    LanaError error = tensor_elementwise(vm, a_pred, b_pred, op, &pred_value);
    if (error != LANA_OK) return error;
    Value var_value;
    if (op == 0 || op == 1) {
        /* add/sub: var = var_a + var_b */
        error = tensor_elementwise(vm, a_var, b_var, 0, &var_value);
    } else if (op == 2) {
        /* mul: var = var_a * b^2 + var_b * a^2 */
        Value b_sq, a_sq, t1, t2;
        error = tensor_elementwise(vm, b_pred, b_pred, 2, &b_sq);
        if (error != LANA_OK) return error;
        error = tensor_elementwise(vm, a_pred, a_pred, 2, &a_sq);
        if (error != LANA_OK) return error;
        error = tensor_elementwise(vm, a_var, b_sq.as.tensor, 2, &t1);
        if (error != LANA_OK) return error;
        error = tensor_elementwise(vm, b_var, a_sq.as.tensor, 2, &t2);
        if (error != LANA_OK) return error;
        error = tensor_elementwise(vm, t1.as.tensor, t2.as.tensor, 0, &var_value);
    } else {
        /* div: var = var_a / b^2 + var_b * a^2 / b^4 */
        Value b_sq, b_4, a_sq, t1, t2, t3;
        error = tensor_elementwise(vm, b_pred, b_pred, 2, &b_sq);
        if (error != LANA_OK) return error;
        error = tensor_elementwise(vm, b_sq.as.tensor, b_sq.as.tensor, 2, &b_4);
        if (error != LANA_OK) return error;
        error = tensor_elementwise(vm, a_pred, a_pred, 2, &a_sq);
        if (error != LANA_OK) return error;
        error = tensor_elementwise(vm, a_var, b_sq.as.tensor, 3, &t1);
        if (error != LANA_OK) return error;
        error = tensor_elementwise(vm, b_var, a_sq.as.tensor, 2, &t2);
        if (error != LANA_OK) return error;
        error = tensor_elementwise(vm, t2.as.tensor, b_4.as.tensor, 3, &t3);
        if (error != LANA_OK) return error;
        error = tensor_elementwise(vm, t1.as.tensor, t3.as.tensor, 0, &var_value);
    }
    if (error != LANA_OK) return error;
    if (!tensor_all_finite(var_value.as.tensor)) return LANA_ERR_INVALID_PARAMETERS;
    return tensor_uncertain_result(vm, pred_value.as.tensor, var_value.as.tensor, out);
}

/* First-order variance propagation for matmul (real-only):
 * var = matmul(var_a, b^2) + matmul(a^2, var_b). */
static LanaError tensor_matmul_uncertain(LanaVM *vm, const LanaTensor *a_pred,
                                         const LanaTensor *a_var, const LanaTensor *b_pred,
                                         const LanaTensor *b_var, Value *out) {
    if (a_pred->is_complex || b_pred->is_complex || a_var->is_complex || b_var->is_complex)
        return LANA_ERR_TYPE;
    Value pred_value;
    LanaError error = tensor_matmul(vm, a_pred, b_pred, matmul_default_dtype(a_pred, b_pred), &pred_value);
    if (error != LANA_OK) return error;
    Value b_sq, a_sq, t1, t2, var_value;
    error = tensor_elementwise(vm, b_pred, b_pred, 2, &b_sq);
    if (error != LANA_OK) return error;
    error = tensor_elementwise(vm, a_pred, a_pred, 2, &a_sq);
    if (error != LANA_OK) return error;
    error = tensor_matmul(vm, a_var, b_sq.as.tensor, matmul_default_dtype(a_var, b_sq.as.tensor), &t1);
    if (error != LANA_OK) return error;
    error = tensor_matmul(vm, a_sq.as.tensor, b_var, matmul_default_dtype(a_sq.as.tensor, b_var), &t2);
    if (error != LANA_OK) return error;
    error = tensor_elementwise(vm, t1.as.tensor, t2.as.tensor, 0, &var_value);
    if (error != LANA_OK) return error;
    if (!tensor_all_finite(var_value.as.tensor)) return LANA_ERR_INVALID_PARAMETERS;
    return tensor_uncertain_result(vm, pred_value.as.tensor, var_value.as.tensor, out);
}

/* Reduce to a tensor, wrapping a full-reduction number in a rank-0 tensor. */
static LanaError tensor_reduce_tensor(LanaVM *vm, const LanaTensor *t, int op,
                                      const Value *axis_value, LanaTensor **out) {
    Value result;
    LanaError error = tensor_reduce(vm, t, op, axis_value, &result);
    if (error != LANA_OK) return error;
    if (result.type == VAL_TENSOR) { *out = result.as.tensor; return LANA_OK; }
    LanaTensor *r = tensor_new(vm, 0, NULL, false);
    if (r == NULL) return LANA_ERR_OOM;
    tensor_set_real(r, 0, result.as.number);
    *out = r;
    return LANA_OK;
}

/* Scale a freshly-allocated (contiguous) tensor by a constant factor. */
static LanaError tensor_scale(LanaVM *vm, const LanaTensor *t, double factor, LanaTensor **out) {
    LanaTensor *r = tensor_new(vm, t->ndim, t->shape, t->is_complex);
    if (r == NULL) return LANA_ERR_OOM;
    size_t total = 1;
    for (size_t i = 0; i < t->ndim; ++i) total *= t->shape[i];
    for (size_t e = 0; e < total; ++e) {
        tensor_set_real(r, e, tensor_get_real(t, t->offset + e) * factor);
        tensor_set_imag(r, e, tensor_get_imag(t, t->offset + e) * factor);
    }
    *out = r;
    return LANA_OK;
}

/* First-order variance propagation for sum/mean reductions (real-only):
 * sum: var = sum(var); mean: var = sum(var) / n^2. */
static LanaError tensor_reduce_uncertain(LanaVM *vm, const LanaTensor *pred, const LanaTensor *var,
                                         int op, const Value *axis_value, Value *out) {
    if (pred->is_complex || var->is_complex) return LANA_ERR_TYPE;
    LanaTensor *pred_tensor;
    LanaError error = tensor_reduce_tensor(vm, pred, op, axis_value, &pred_tensor);
    if (error != LANA_OK) return error;
    LanaTensor *var_tensor;
    error = tensor_reduce_tensor(vm, var, 0, axis_value, &var_tensor);
    if (error != LANA_OK) return error;
    if (op == 1) {
        size_t n;
        if (axis_value == NULL) {
            n = 1;
            for (size_t i = 0; i < pred->ndim; ++i) n *= pred->shape[i];
        } else {
            double axis_number = axis_value->as.number;
            size_t axis = (size_t)(axis_number < 0 ? axis_number + (double)pred->ndim : axis_number);
            n = pred->shape[axis];
        }
        double factor = (double)n * (double)n;
        LanaTensor *scaled;
        error = tensor_scale(vm, var_tensor, 1.0 / factor, &scaled);
        if (error != LANA_OK) return error;
        var_tensor = scaled;
    }
    if (!tensor_all_finite(var_tensor)) return LANA_ERR_INVALID_PARAMETERS;
    return tensor_uncertain_result(vm, pred_tensor, var_tensor, out);
}

/* ===== LIP-011 reverse-mode autodiff helpers ===== */

/* Total number of real elements in a tensor (autodiff is real-only). */
static size_t tensor_element_count(const LanaTensor *t) {
    size_t total = 1;
    for (size_t i = 0; i < t->ndim; ++i) total *= t->shape[i];
    return total;
}

static bool tensor_shape_equal(const LanaTensor *a, const LanaTensor *b) {
    if (a->ndim != b->ndim) return false;
    for (size_t i = 0; i < a->ndim; ++i)
        if (a->shape[i] != b->shape[i]) return false;
    return true;
}

/* A view of `t` with its last two axes swapped. Shares the source buffer. */
static LanaTensor *tensor_transpose_last_two(LanaVM *vm, const LanaTensor *t) {
    if (t->ndim < 2) return (LanaTensor *)t;
    LanaTensor *view = lana_vm_alloc(vm, sizeof(*view));
    if (view == NULL) return NULL;
    view->ndim = t->ndim;
    view->is_complex = t->is_complex;
    view->dtype = t->dtype;    /* LIP-027: views inherit the source dtype */
    view->is_state = t->is_state;
    view->shape = lana_vm_alloc(vm, t->ndim * sizeof(*view->shape));
    view->strides = lana_vm_alloc(vm, t->ndim * sizeof(*view->strides));
    if (view->shape == NULL || view->strides == NULL) return NULL;
    for (size_t i = 0; i < t->ndim; ++i) {
        view->shape[i] = t->shape[i];
        view->strides[i] = t->strides[i];
    }
    size_t last = t->ndim - 1u, second = t->ndim - 2u;
    size_t tmp = view->shape[last];
    view->shape[last] = view->shape[second];
    view->shape[second] = tmp;
    tmp = view->strides[last];
    view->strides[last] = view->strides[second];
    view->strides[second] = tmp;
    view->data = t->data;
    view->offset = t->offset;
    view->base = (LanaTensor *)t;
    return view;
}

/* Outer product of two 1-D tensors: out[i,j] = u[i] * v[j]. */
static LanaError tensor_outer(LanaVM *vm, const LanaTensor *u, const LanaTensor *v,
                              LanaTensor **out) {
    size_t k = u->shape[0], n = v->shape[0];
    size_t shape[2] = {k, n};
    LanaTensor *r = tensor_new(vm, 2, shape, false);
    if (r == NULL) return LANA_ERR_OOM;
    for (size_t i = 0; i < k; ++i)
        for (size_t j = 0; j < n; ++j)
            tensor_set_real(r, i * n + j,
                tensor_get_real(u, u->offset + i * u->strides[0]) *
                tensor_get_real(v, v->offset + j * v->strides[0]));
    *out = r;
    return LANA_OK;
}

/* Negate a tensor into a fresh base tensor (view-correct). */
static LanaError tensor_negate(LanaVM *vm, const LanaTensor *t, LanaTensor **out) {
    LanaTensor *r = tensor_new(vm, t->ndim, t->shape, false);
    if (r == NULL) return LANA_ERR_OOM;
    size_t total = tensor_element_count(t);
    size_t *idx = lana_vm_alloc(vm, t->ndim * sizeof(*idx));
    if (idx == NULL && t->ndim > 0) return LANA_ERR_OOM;
    for (size_t lin = 0; lin < total; ++lin) {
        size_t rem = lin, index = t->offset;
        for (size_t d = t->ndim; d-- > 0;) {
            index += (t->shape[d] == 0 ? 0 : rem % t->shape[d]) * t->strides[d];
            rem /= t->shape[d];
        }
        tensor_set_real(r, lin, -tensor_get_real(t, index));
    }
    *out = r;
    return LANA_OK;
}

/* Sum `g` over the broadcast dimensions so the result has `target`'s shape.
 * `g` is a contiguous base tensor; `target` supplies only its shape. */
static LanaError tensor_unbroadcast(LanaVM *vm, const LanaTensor *g,
                                    const LanaTensor *target, LanaTensor **out) {
    size_t out_ndim = target->ndim;
    LanaTensor *r = tensor_new(vm, out_ndim, target->shape, false);
    if (r == NULL) return LANA_ERR_OOM;
    size_t total = tensor_element_count(g);
    size_t *idx = lana_vm_alloc(vm, g->ndim * sizeof(*idx));
    if (idx == NULL && g->ndim > 0) return LANA_ERR_OOM;
    size_t offset = g->ndim - out_ndim;
    for (size_t lin = 0; lin < total; ++lin) {
        size_t rem = lin;
        for (ssize_t i = (ssize_t)g->ndim - 1; i >= 0; --i) {
            idx[i] = (g->shape[i] == 0) ? 0 : rem % g->shape[i];
            rem /= g->shape[i];
        }
        size_t ti = 0;
        for (size_t i = 0; i < out_ndim; ++i) {
            size_t gi = i + offset;
            size_t coord = (target->shape[i] == 1) ? 0 : idx[gi];
            ti = ti * target->shape[i] + coord;
        }
        tensor_set_real(r, ti, tensor_get_real(r, ti) + tensor_get_real(g, g->offset + lin));
    }
    *out = r;
    return LANA_OK;
}

/* Broadcast a reduced gradient `g` back to `input`'s shape, scaled by `scale`.
 * `axis` is the reduced axis (-1 for a full reduction). */
static LanaError tensor_broadcast_reduce(LanaVM *vm, const LanaTensor *g,
                                         const LanaTensor *input, int axis,
                                         double scale, LanaTensor **out) {
    LanaTensor *r = tensor_new(vm, input->ndim, input->shape, false);
    if (r == NULL) return LANA_ERR_OOM;
    size_t total = tensor_element_count(input);
    size_t *idx = lana_vm_alloc(vm, input->ndim * sizeof(*idx));
    if (idx == NULL && input->ndim > 0) return LANA_ERR_OOM;
    size_t g_strides[LANA_TENSOR_MAX_RANK];
    size_t stride = 1;
    for (ssize_t i = (ssize_t)g->ndim - 1; i >= 0; --i) {
        g_strides[i] = stride;
        stride *= g->shape[i];
    }
    for (size_t lin = 0; lin < total; ++lin) {
        size_t rem = lin;
        for (ssize_t i = (ssize_t)input->ndim - 1; i >= 0; --i) {
            idx[i] = (input->shape[i] == 0) ? 0 : rem % input->shape[i];
            rem /= input->shape[i];
        }
        size_t gi = 0;
        if (axis < 0) {
            gi = 0;
        } else {
            size_t gd = 0;
            for (size_t i = 0; i < input->ndim; ++i) {
                if (i == (size_t)axis) continue;
                gi += idx[i] * g_strides[gd++];
            }
        }
        tensor_set_real(r, lin, tensor_get_real(g, g->offset + gi) * scale);
    }
    *out = r;
    return LANA_OK;
}

/* Thin wrappers that return the tensor directly (the public helpers return a
 * Value). */
static LanaError ad_elementwise(LanaVM *vm, const LanaTensor *a, const LanaTensor *b,
                                int op, LanaTensor **out) {
    Value result;
    LanaError error = tensor_elementwise(vm, a, b, op, &result);
    if (error != LANA_OK) return error;
    *out = result.as.tensor;
    return LANA_OK;
}

static LanaError ad_matmul(LanaVM *vm, const LanaTensor *a, const LanaTensor *b,
                           LanaTensor **out) {
    Value result;
    LanaError error = tensor_matmul(vm, a, b, matmul_default_dtype(a, b), &result);
    if (error != LANA_OK) return error;
    *out = result.as.tensor;
    return LANA_OK;
}

/* Forward declaration: read the (k, i, j) entry of a 3-D tensor as a complex
 * number (defined below with the LIP-005 linear-algebra helpers). */
static void linalg_get3(const LanaTensor *t, size_t k, size_t i, size_t j, double *re, double *im);

/* Record a differentiable primitive onto `result`'s derivation. `ad_op` is
 * 0=add 1=sub 2=mul 3=div 4=matmul 5=sum 6=mean. `b` is NULL for reductions. */
static LanaError ad_record(LanaVM *vm, int ad_op, const Value *a, const Value *b,
                           int ad_axis, Value *result) {
    const Value *inputs[2];
    size_t input_count = (b != NULL) ? 2u : 1u;
    inputs[0] = a;
    if (b != NULL) inputs[1] = b;
    LanaDerivation *node = record_derivation(vm, LANA_DERIVATION_OPERATION, "autodiff",
        inputs, input_count, "", 0u, LANA_EXACTNESS_EXACT, "autodiff",
        LANA_DERIVATION_SUCCESS, "none");
    if (node == NULL) return LANA_ERR_OOM;
    node->ad_op = ad_op;
    node->ad_a = a->as.tensor;
    node->ad_b = (b != NULL) ? b->as.tensor : NULL;
    node->ad_a_deriv = a->derivation;
    node->ad_b_deriv = (b != NULL) ? b->derivation : NULL;
    node->ad_axis = ad_axis;
    result->derivation = node;
    return LANA_OK;
}

/* Reverse-mode backward pass. `seed` is the cotangent of `node`'s output,
 * a contiguous base tensor. Accumulates into each node's `ad_grad` and
 * recurses into the input derivations in left-then-right order. */
static LanaError ad_backward(LanaVM *vm, LanaDerivation *node, const LanaTensor *seed) {
    if (node->ad_grad == NULL) {
        node->ad_grad = tensor_new(vm, seed->ndim, seed->shape, seed->is_complex);
        if (node->ad_grad == NULL) return LANA_ERR_OOM;
    }
    size_t count = tensor_element_count(seed);
    for (size_t e = 0; e < count; ++e) {
        tensor_set_real(node->ad_grad, e, tensor_get_real(node->ad_grad, e) + tensor_get_real(seed, seed->offset + e));
        tensor_set_imag(node->ad_grad, e, tensor_get_imag(node->ad_grad, e) + tensor_get_imag(seed, seed->offset + e));
    }

    if (node->ad_op < 0) return LANA_OK;

    switch (node->ad_op) {
        case 0: /* add */
        case 1: /* sub */
        case 2: /* mul */
        case 3: /* div */ {
            LanaTensor *a = node->ad_a, *b = node->ad_b;
            LanaTensor *ga = NULL, *gb = NULL;
            LanaError error = LANA_OK;
            switch (node->ad_op) {
                case 0:
                    ga = (LanaTensor *)seed;
                    gb = (LanaTensor *)seed;
                    break;
                case 1:
                    ga = (LanaTensor *)seed;
                    error = tensor_negate(vm, seed, &gb);
                    break;
                case 2:
                    error = ad_elementwise(vm, seed, b, 2, &ga);
                    if (error == LANA_OK) error = ad_elementwise(vm, seed, a, 2, &gb);
                    break;
                default: {
                    LanaTensor *t1 = NULL, *t2 = NULL, *t3 = NULL;
                    error = ad_elementwise(vm, seed, b, 3, &ga);
                    if (error == LANA_OK) error = ad_elementwise(vm, seed, a, 2, &t1);
                    if (error == LANA_OK) error = ad_elementwise(vm, b, b, 2, &t2);
                    if (error == LANA_OK) error = ad_elementwise(vm, t1, t2, 3, &t3);
                    if (error == LANA_OK) error = tensor_negate(vm, t3, &gb);
                    break;
                }
            }
            if (error != LANA_OK) return error;
            LanaTensor *ga_u = NULL, *gb_u = NULL;
            error = tensor_unbroadcast(vm, ga, a, &ga_u);
            if (error == LANA_OK) error = tensor_unbroadcast(vm, gb, b, &gb_u);
            if (error != LANA_OK) return error;
            if (node->ad_a_deriv != NULL) {
                error = ad_backward(vm, node->ad_a_deriv, ga_u);
                if (error != LANA_OK) return error;
            }
            if (node->ad_b_deriv != NULL) {
                error = ad_backward(vm, node->ad_b_deriv, gb_u);
                if (error != LANA_OK) return error;
            }
            return LANA_OK;
        }
        case 4: { /* matmul */
            LanaTensor *a = node->ad_a, *b = node->ad_b;
            size_t a_ndim = a->ndim, b_ndim = b->ndim;
            LanaTensor *ga = NULL, *gb = NULL;
            LanaError error = LANA_OK;
            if (a_ndim == 1 && b_ndim == 1) {
                error = ad_elementwise(vm, seed, b, 2, &ga);
                if (error == LANA_OK) error = ad_elementwise(vm, seed, a, 2, &gb);
            } else if (a_ndim == 1) {
                LanaTensor *bt = tensor_transpose_last_two(vm, b);
                if (bt == NULL) return LANA_ERR_OOM;
                error = ad_matmul(vm, seed, bt, &ga);
                if (error == LANA_OK) error = tensor_outer(vm, a, seed, &gb);
            } else if (b_ndim == 1) {
                LanaTensor *at = tensor_transpose_last_two(vm, a);
                if (at == NULL) return LANA_ERR_OOM;
                error = tensor_outer(vm, seed, b, &ga);
                if (error == LANA_OK) error = ad_matmul(vm, at, seed, &gb);
            } else {
                LanaTensor *bt = tensor_transpose_last_two(vm, b);
                LanaTensor *at = tensor_transpose_last_two(vm, a);
                if (bt == NULL || at == NULL) return LANA_ERR_OOM;
                LanaTensor *ga_raw = NULL, *gb_raw = NULL;
                error = ad_matmul(vm, seed, bt, &ga_raw);
                if (error == LANA_OK) error = ad_matmul(vm, at, seed, &gb_raw);
                if (error == LANA_OK) error = tensor_unbroadcast(vm, ga_raw, a, &ga);
                if (error == LANA_OK) error = tensor_unbroadcast(vm, gb_raw, b, &gb);
            }
            if (error != LANA_OK) return error;
            if (node->ad_a_deriv != NULL) {
                error = ad_backward(vm, node->ad_a_deriv, ga);
                if (error != LANA_OK) return error;
            }
            if (node->ad_b_deriv != NULL) {
                error = ad_backward(vm, node->ad_b_deriv, gb);
                if (error != LANA_OK) return error;
            }
            return LANA_OK;
        }
        case 5: /* sum */
        case 6: { /* mean */
            LanaTensor *a = node->ad_a;
            size_t n;
            if (node->ad_axis < 0) {
                n = tensor_element_count(a);
            } else {
                n = a->shape[node->ad_axis];
            }
            double scale = (node->ad_op == 6) ? 1.0 / (double)n : 1.0;
            LanaTensor *ga = NULL;
            LanaError error = tensor_broadcast_reduce(vm, seed, a, node->ad_axis, scale, &ga);
            if (error != LANA_OK) return error;
            if (node->ad_a_deriv != NULL) {
                error = ad_backward(vm, node->ad_a_deriv, ga);
                if (error != LANA_OK) return error;
            }
            return LANA_OK;
        }
        case 7: { /* append (LIP-007): mean-state distribution-valued APPEND */
            LanaTensor *a = node->ad_a, *b = node->ad_b;
            size_t d = a->shape[a->ndim - 1];
            size_t batch = 1;
            LanaTensor *ga, *gb;
            for (size_t i = 0; i + 2 < a->ndim; ++i) batch *= a->shape[i];
            ga = tensor_new(vm, a->ndim, a->shape, true);
            gb = tensor_new(vm, b->ndim, b->shape, true);
            if (ga == NULL || gb == NULL) return LANA_ERR_OOM;
            for (size_t bi = 0; bi < batch; ++bi) {
                const double *da = (const double*)a->data + bi * d * d * 2;
                const double *db = (const double*)b->data + bi * d * d * 2;
                const double *dg = (const double*)seed->data + (seed->offset + bi * d * d) * 2;
                double p_a = da[0], c_a_re = da[2], c_a_im = da[3];
                double p_b = db[0], c_b_re = db[2], c_b_im = db[3];
                double s_a = sqrt(p_a * (1.0 - p_a));
                double s_b = sqrt(p_b * (1.0 - p_b));
                double d_a_re = s_a > 0.0 ? c_a_re / s_a : 0.0;
                double d_a_im = s_a > 0.0 ? c_a_im / s_a : 0.0;
                double d_b_re = s_b > 0.0 ? c_b_re / s_b : 0.0;
                double d_b_im = s_b > 0.0 ? c_b_im / s_b : 0.0;
                double p_c = p_a + p_b - p_a * p_b;
                double d_c_re = (d_a_re + d_b_re) / 2.0;
                double d_c_im = (d_a_im + d_b_im) / 2.0;
                double s_c = sqrt(p_c * (1.0 - p_c));
                /* Reduce the seed to (g_p, g_c): the cotangents of p_C and c_C. */
                double g_p = dg[0] - dg[6];
                double g_c_re = dg[2] + dg[4];
                double g_c_im = dg[3] - dg[5];
                double g_dc_re = s_c * g_c_re;
                double g_dc_im = s_c * g_c_im;
                double g_sc = g_c_re * d_c_re + g_c_im * d_c_im;
                double g_pc = g_p;
                if (s_c > 0.0) g_pc += g_sc * (1.0 - 2.0 * p_c) / (2.0 * s_c);
                double g_da_re = g_dc_re / 2.0, g_da_im = g_dc_im / 2.0;
                double g_db_re = g_dc_re / 2.0, g_db_im = g_dc_im / 2.0;
                double g_pa = g_pc * (1.0 - p_b);
                double g_pb = g_pc * (1.0 - p_a);
                double g_ca_re = 0.0, g_ca_im = 0.0, g_cb_re = 0.0, g_cb_im = 0.0;
                if (s_a > 0.0) {
                    g_ca_re = g_da_re / s_a;
                    g_ca_im = g_da_im / s_a;
                    double g_sa = -(g_da_re * c_a_re + g_da_im * c_a_im) / (s_a * s_a);
                    g_pa += g_sa * (1.0 - 2.0 * p_a) / (2.0 * s_a);
                }
                if (s_b > 0.0) {
                    g_cb_re = g_db_re / s_b;
                    g_cb_im = g_db_im / s_b;
                    double g_sb = -(g_db_re * c_b_re + g_db_im * c_b_im) / (s_b * s_b);
                    g_pb += g_sb * (1.0 - 2.0 * p_b) / (2.0 * s_b);
                }
                double *ga_data = (double*)ga->data + bi * d * d * 2;
                ga_data[0] = g_pa; ga_data[1] = 0.0;
                ga_data[2] = g_ca_re; ga_data[3] = g_ca_im;
                ga_data[4] = g_ca_re; ga_data[5] = g_ca_im == 0.0 ? 0.0 : -g_ca_im;
                ga_data[6] = -g_pa; ga_data[7] = 0.0;
                double *gb_data = (double*)gb->data + bi * d * d * 2;
                gb_data[0] = g_pb; gb_data[1] = 0.0;
                gb_data[2] = g_cb_re; gb_data[3] = g_cb_im;
                gb_data[4] = g_cb_re; gb_data[5] = g_cb_im == 0.0 ? 0.0 : -g_cb_im;
                gb_data[6] = -g_pb; gb_data[7] = 0.0;
            }
            if (node->ad_a_deriv != NULL) {
                LanaError error = ad_backward(vm, node->ad_a_deriv, ga);
                if (error != LANA_OK) return error;
            }
            if (node->ad_b_deriv != NULL) {
                LanaError error = ad_backward(vm, node->ad_b_deriv, gb);
                if (error != LANA_OK) return error;
            }
            return LANA_OK;
        }
        case 8: { /* measure (LIP-007): q[..., i] = Tr(ρ E_i) */
            LanaTensor *s = node->ad_a;
            LanaTensor *povm = node->ad_b;
            size_t d = s->shape[s->ndim - 1];
            size_t k = povm->shape[0];
            size_t batch = 1;
            LanaTensor *gs;
            for (size_t i = 0; i + 2 < s->ndim; ++i) batch *= s->shape[i];
            gs = tensor_new(vm, s->ndim, s->shape, true);
            if (gs == NULL) return LANA_ERR_OOM;
            for (size_t bi = 0; bi < batch; ++bi) {
                for (size_t r = 0; r < d; ++r)
                    for (size_t c = 0; c < d; ++c) {
                        double re = 0.0, im = 0.0;
                        for (size_t m = 0; m < k; ++m) {
                            double seed_val = tensor_get_real(seed, seed->offset + bi * k + m);
                            double e_re, e_im;
                            linalg_get3(povm, m, c, r, &e_re, &e_im);
                            re += seed_val * e_re;
                            im += seed_val * (-e_im);
                        }
                        tensor_set_real(gs, bi * d * d + r * d + c, re);
                        tensor_set_imag(gs, bi * d * d + r * d + c, im);
                    }
            }
            if (node->ad_a_deriv != NULL) {
                LanaError error = ad_backward(vm, node->ad_a_deriv, gs);
                if (error != LANA_OK) return error;
            }
            return LANA_OK;
        }
        case 9: { /* transform (LIP-007): Φ(ρ) = Σ_k K_k ρ K_k† */
            LanaTensor *s = node->ad_a;
            LanaTensor *chan = node->ad_b;
            size_t d = s->shape[s->ndim - 1];
            size_t k = chan->shape[0];
            size_t batch = 1;
            LanaTensor *gs;
            for (size_t i = 0; i + 2 < s->ndim; ++i) batch *= s->shape[i];
            gs = tensor_new(vm, s->ndim, s->shape, true);
            if (gs == NULL) return LANA_ERR_OOM;
            for (size_t bi = 0; bi < batch; ++bi) {
                const double *dg = (const double*)seed->data + (seed->offset + bi * d * d) * 2;
                for (size_t m = 0; m < d; ++m)
                    for (size_t n = 0; n < d; ++n) {
                        double re = 0.0, im = 0.0;
                        for (size_t kk = 0; kk < k; ++kk)
                            for (size_t i = 0; i < d; ++i)
                                for (size_t j = 0; j < d; ++j) {
                                    double ki_re, ki_im, g_re, g_im, kj_re, kj_im;
                                    linalg_get3(chan, kk, i, m, &ki_re, &ki_im);
                                    g_re = dg[(i * d + j) * 2]; g_im = dg[(i * d + j) * 2 + 1];
                                    linalg_get3(chan, kk, j, n, &kj_re, &kj_im);
                                    /* conj(K[i][m]) * G[i][j] * K[j][n] */
                                    double t_re = ki_re * g_re + ki_im * g_im;
                                    double t_im = ki_re * g_im - ki_im * g_re;
                                    re += t_re * kj_re - t_im * kj_im;
                                    im += t_re * kj_im + t_im * kj_re;
                                }
                        tensor_set_real(gs, bi * d * d + m * d + n, re);
                        tensor_set_imag(gs, bi * d * d + m * d + n, im);
                    }
            }
            if (node->ad_a_deriv != NULL) {
                LanaError error = ad_backward(vm, node->ad_a_deriv, gs);
                if (error != LANA_OK) return error;
            }
            return LANA_OK;
        }
        default:
            return LANA_ERR_TYPE;
    }
}

/* Recursively infer the shape of a nested array of numbers, one buffer per
 * nesting level. Ragged input raises LANA_ERR_INVALID_PARAMETERS; non-number
 * leaves raise LANA_ERR_TYPE. */
/* Infer the shape of a nested numeric array into VM-accounted scratch.
 * Returns LANA_OK, or LANA_ERR_TYPE / LANA_ERR_INVALID_PARAMETERS / OOM.
 * The buffers are GC memory; callers must not free them. */
static LanaError tensor_infer_shape_at(LanaVM *vm, const Value *v, size_t *ndim, size_t **shape,
                                      size_t depth) {
    if (v->type == VAL_NUMBER) { *ndim = 0; *shape = NULL; return LANA_OK; }
    if (v->type != VAL_ARRAY) return LANA_ERR_TYPE;
    if (depth == LANA_TENSOR_MAX_RANK) return LANA_ERR_INVALID_PARAMETERS;
    LanaArray *arr = v->as.array;
    if (arr->count == 0) {
        *ndim = 1; *shape = lana_vm_alloc(vm, sizeof(**shape));
        if (*shape == NULL) return LANA_ERR_OOM;
        (*shape)[0] = 0; return LANA_OK;
    }
    size_t sub_ndim; size_t *sub_shape;
    LanaError e = tensor_infer_shape_at(vm, &arr->items[0], &sub_ndim, &sub_shape, depth + 1u);
    if (e != LANA_OK) return e;
    for (size_t i = 1; i < arr->count; ++i) {
        size_t s_ndim; size_t *s_shape;
        e = tensor_infer_shape_at(vm, &arr->items[i], &s_ndim, &s_shape, depth + 1u);
        if (e != LANA_OK) return e;
        if (s_ndim != sub_ndim) return LANA_ERR_INVALID_PARAMETERS;
        for (size_t d = 0; d < sub_ndim; ++d) {
            if (s_shape[d] != sub_shape[d]) return LANA_ERR_INVALID_PARAMETERS;
        }
    }
    *ndim = sub_ndim + 1;
    *shape = lana_vm_alloc(vm, (*ndim) * sizeof(**shape));
    if (*shape == NULL) return LANA_ERR_OOM;
    (*shape)[0] = arr->count;
    for (size_t d = 0; d < sub_ndim; ++d) (*shape)[d + 1] = sub_shape[d];
    return LANA_OK;
}

static LanaError tensor_infer_shape(LanaVM *vm, const Value *v, size_t *ndim, size_t **shape) {
    return tensor_infer_shape_at(vm, v, ndim, shape, 0u);
}

/* Recursively fill a tensor's data buffer in row-major order from a nested
 * array of numbers. `offset` is advanced past the written elements. */
static LanaError tensor_fill_data(const Value *v, LanaTensor *t, size_t *offset) {
    if (v->type == VAL_NUMBER) { tensor_set_real(t, (*offset)++, v->as.number); return LANA_OK; }
    if (v->type != VAL_ARRAY) return LANA_ERR_TYPE;
    LanaArray *arr = v->as.array;
    for (size_t i = 0; i < arr->count; ++i) {
        LanaError e = tensor_fill_data(&arr->items[i], t, offset);
        if (e != LANA_OK) return e;
    }
    return LANA_OK;
}

/* Recursively fill a complex tensor's interleaved [re, im] buffer from two
 * parallel nested arrays. Shape mismatch raises LANA_ERR_INVALID_PARAMETERS. */
static LanaError tensor_fill_complex(const Value *re, const Value *im, LanaTensor *t, size_t *offset) {
    if (re->type == VAL_NUMBER) {
        if (im->type != VAL_NUMBER) return LANA_ERR_TYPE;
        tensor_set_real(t, *offset, re->as.number);
        tensor_set_imag(t, *offset, im->as.number);
        (*offset)++;
        return LANA_OK;
    }
    if (re->type != VAL_ARRAY || im->type != VAL_ARRAY) return LANA_ERR_TYPE;
    LanaArray *ra = re->as.array, *ia = im->as.array;
    if (ra->count != ia->count) return LANA_ERR_INVALID_PARAMETERS;
    for (size_t i = 0; i < ra->count; ++i) {
        LanaError e = tensor_fill_complex(&ra->items[i], &ia->items[i], t, offset);
        if (e != LANA_OK) return e;
    }
    return LANA_OK;
}

/* LIP-004 indexing: resolve one integer position against a dimension of
 * length `dim`. Negative counts from the end; a non-integer or non-finite
 * number is LANA_ERR_TYPE, an adjusted position outside [0, dim) is
 * LANA_ERR_KEY. */
static LanaError tensor_resolve_index(double n, size_t dim, size_t *position) {
    if (!isfinite(n) || floor(n) != n) return LANA_ERR_TYPE;
    if (n < -(double)dim || n >= (double)dim) return LANA_ERR_KEY;
    *position = (size_t)(n < 0.0 ? n + (double)dim : n);
    return LANA_OK;
}

/* LIP-004 slicing: resolve one slice bound against a dimension of length
 * `dim`. Negative counts from the end, then clamps into [0, dim]; slice
 * bounds never range-error. Non-integer numbers are LANA_ERR_TYPE. */
static LanaError tensor_resolve_slice_bound(double n, size_t dim, size_t *bound) {
    if (!isfinite(n) || floor(n) != n) return LANA_ERR_TYPE;
    double adjusted = n < 0.0 ? n + (double)dim : n;
    if (adjusted < 0.0) adjusted = 0.0;
    if (adjusted > (double)dim) adjusted = (double)dim;
    *bound = (size_t)adjusted;
    return LANA_OK;
}

/* LIP-004 indexing and slicing: build a view (or scalar) of `t` from a spec.
 * The spec is a single VAL_NUMBER (one integer position) or a VAL_ARRAY of
 * positions, each a VAL_NUMBER (integer position) or a two-element VAL_ARRAY
 * [start, end] (slice). Fewer positions than the rank keep the remaining
 * trailing axes whole. Integer positions drop their axis; slices keep it.
 * Every non-fully-integer result shares the source buffer as a view. */
static LanaError tensor_index(LanaVM *vm, LanaTensor *t, const Value *spec, Value *out) {
    bool is_int[LANA_TENSOR_MAX_RANK];
    size_t start[LANA_TENSOR_MAX_RANK], count[LANA_TENSOR_MAX_RANK];
    size_t positions = 0;
    LanaError error;

    if (spec->type == VAL_NUMBER) {
        if (t->ndim == 0) return LANA_ERR_INVALID_PARAMETERS;
        error = tensor_resolve_index(spec->as.number, t->shape[0], &start[0]);
        if (error != LANA_OK) return error;
        is_int[0] = true; count[0] = 1; positions = 1;
    } else if (spec->type == VAL_ARRAY) {
        LanaArray *arr = spec->as.array;
        if (arr->count > t->ndim) return LANA_ERR_INVALID_PARAMETERS;
        if (arr->count > LANA_TENSOR_MAX_RANK) return LANA_ERR_INVALID_PARAMETERS;
        for (size_t i = 0; i < arr->count; ++i) {
            const Value *item = &arr->items[i];
            if (item->type == VAL_NUMBER) {
                error = tensor_resolve_index(item->as.number, t->shape[i], &start[i]);
                if (error != LANA_OK) return error;
                is_int[i] = true; count[i] = 1;
            } else if (item->type == VAL_ARRAY) {
                LanaArray *pair = item->as.array;
                if (pair->count != 2u || pair->items[0].type != VAL_NUMBER ||
                    pair->items[1].type != VAL_NUMBER) return LANA_ERR_TYPE;
                size_t from, to;
                error = tensor_resolve_slice_bound(pair->items[0].as.number, t->shape[i], &from);
                if (error != LANA_OK) return error;
                error = tensor_resolve_slice_bound(pair->items[1].as.number, t->shape[i], &to);
                if (error != LANA_OK) return error;
                is_int[i] = false; start[i] = from;
                count[i] = to > from ? to - from : 0;
            } else {
                return LANA_ERR_TYPE;
            }
        }
        positions = arr->count;
    } else {
        return LANA_ERR_TYPE;
    }

    /* Fold every axis into the view: integer axes contribute their offset and
     * drop out; slice axes keep their (clamped) extent; trailing axes beyond
     * the position list are full slices. */
    bool all_int = true;
    size_t view_offset = t->offset;
    size_t view_ndim = 0;
    size_t view_shape[LANA_TENSOR_MAX_RANK], view_strides[LANA_TENSOR_MAX_RANK];
    for (size_t i = 0; i < t->ndim; ++i) {
        bool integer = i < positions && is_int[i];
        size_t axis_start = i < positions ? start[i] : 0;
        size_t axis_count = i < positions ? count[i] : t->shape[i];
        view_offset += axis_start * t->strides[i];
        if (integer) continue;
        all_int = false;
        view_shape[view_ndim] = axis_count;
        view_strides[view_ndim] = t->strides[i];
        view_ndim++;
    }

    if (all_int) {
        /* A full set of integer positions selects one element: a number, or
         * the established rank-zero complex tensor for complex tensors. */
        if (!t->is_complex) { *out = lana_value_number(tensor_get_real(t, view_offset)); return LANA_OK; }
        LanaTensor *r = tensor_new(vm, 0, NULL, true);
        if (r == NULL) return LANA_ERR_OOM;
        tensor_set_real(r, 0, tensor_get_real(t, view_offset));
        tensor_set_imag(r, 0, tensor_get_imag(t, view_offset));
        *out = lana_value_tensor(r);
        return LANA_OK;
    }

    LanaTensor *view = lana_vm_alloc(vm, sizeof(*view));
    if (view == NULL) return LANA_ERR_OOM;
    view->ndim = view_ndim;
    view->is_complex = t->is_complex;
    view->dtype = t->dtype;    /* LIP-027: views inherit the source dtype */
    view->is_state = t->is_state;
    view->shape = NULL;
    view->strides = NULL;
    view->data = t->data;      /* shared with the source; accounted once */
    view->offset = view_offset;
    view->base = t;            /* roots the source tensor and its buffer */
    if (view_ndim > 0) {
        view->shape = lana_vm_alloc(vm, view_ndim * sizeof(*view->shape));
        view->strides = lana_vm_alloc(vm, view_ndim * sizeof(*view->strides));
        if (view->shape == NULL || view->strides == NULL) return LANA_ERR_OOM;
        for (size_t i = 0; i < view_ndim; ++i) {
            view->shape[i] = view_shape[i];
            view->strides[i] = view_strides[i];
        }
    }
    *out = lana_value_tensor(view);
    return LANA_OK;
}

/* A set member is an ordinary value: STATE, STATE_DIST, and Information
 * (reactive, possibility, path-set) have no canonical hash and are rejected
 * (LIP-022 §1). */
static bool value_is_set_member(const Value *value) {
    if (value == NULL || value->reactive != NULL) return false;
    switch (value->type) {
        case VAL_STATE:
        case VAL_STATE_DIST:
        case VAL_JOINT_STATE:
        case VAL_POSSIBILITY:
        case VAL_PATH_SET:
            return false;
        default:
            return true;
    }
}

/* Set membership equality: scalars compare by value, containers by pointer
 * identity (matching `values_equal`). Byte-identical with the Rust VM. */
static bool set_value_equal(const Value *left, const Value *right) {
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
        case VAL_DISTRIBUTION:
            return left->as.distribution.p0 == right->as.distribution.p0 &&
                   left->as.distribution.p1 == right->as.distribution.p1;
        case VAL_FUNCTION: return left->as.function == right->as.function;
        case VAL_LAZY:
            return left->as.lazy.function == right->as.lazy.function &&
                   left->as.lazy.bound == right->as.lazy.bound;
        case VAL_DATASET:
            return left->as.dataset == right->as.dataset;
        default: return left->as.array == right->as.array;
    }
}

static bool set_contains_value(const LanaSet *set, const Value *value) {
    size_t index;
    for (index = 0u; index < set->count; ++index)
        if (set_value_equal(&set->items[index], value)) return true;
    return false;
}

static LanaError set_alloc(LanaVM *vm, size_t count, LanaSet **out) {
    LanaSet *set = lana_vm_alloc(vm, sizeof(*set));
    if (set == NULL) return LANA_ERR_OOM;
    set->count = count;
    set->capacity = count;
    set->items = lana_vm_alloc(vm, count * sizeof(*set->items));
    if (set->items == NULL && count > 0u) return LANA_ERR_OOM;
    *out = set;
    return LANA_OK;
}

static LanaError make_result(LanaVM *vm, bool ok, Value value, Value *out);

/* LIP-024 async/await event-loop helpers (defined after make_result). */
static LanaError enqueue_future(LanaVM *vm, LanaFuture *future);
static LanaFuture *dequeue_future(LanaVM *vm);
static void complete_future(LanaVM *vm, LanaFuture *future, Value result);
static LanaError run_composite_future(LanaVM *vm, LanaFuture *future);
static LanaError run_async_function(LanaVM *vm, LanaFuture *future);
static LanaError run_event_loop(LanaVM *vm, LanaFuture *target, Value *out);
static LanaError vm_step(LanaVM *vm);

/* Growable string builder over GC memory, used by the `format` host call. */
typedef struct {
    char *data;
    size_t length;
    size_t capacity;
} FormatBuffer;

static LanaError format_reserve(LanaVM *vm, FormatBuffer *buffer, size_t extra) {
    size_t needed = buffer->length + extra + 1u;
    size_t capacity;
    char *grown;
    if (needed <= buffer->capacity) return LANA_OK;
    capacity = buffer->capacity == 0u ? 64u : buffer->capacity;
    while (capacity < needed) {
        if (capacity > SIZE_MAX / 2u) return LANA_ERR_LIMIT;
        capacity *= 2u;
    }
    grown = lana_vm_alloc(vm, capacity);
    if (grown == NULL) return LANA_ERR_OOM;
    if (buffer->length > 0u) memcpy(grown, buffer->data, buffer->length);
    buffer->data = grown;
    buffer->capacity = capacity;
    return LANA_OK;
}

static LanaError format_add(LanaVM *vm, FormatBuffer *buffer, const char *text, size_t length) {
    LanaError error = format_reserve(vm, buffer, length);
    if (error != LANA_OK) return error;
    memcpy(buffer->data + buffer->length, text, length);
    buffer->length += length;
    return LANA_OK;
}

/* Append the stringified form of a value, mirroring the Rust `format_value`. */
static LanaError format_value(LanaVM *vm, FormatBuffer *buffer, const Value *value) {
    char number[64];
    int written;
    switch (value->type) {
        case VAL_NULL: return format_add(vm, buffer, "null", 4u);
        case VAL_BOOL:
            return format_add(vm, buffer, value->as.boolean ? "true" : "false",
                              value->as.boolean ? 4u : 5u);
        case VAL_NUMBER:
            written = snprintf(number, sizeof(number), "%.17g", value->as.number);
            if (written < 0 || (size_t)written >= sizeof(number)) return LANA_ERR_FORMAT;
            return format_add(vm, buffer, number, (size_t)written);
        case VAL_STRING:
            return format_add(vm, buffer, value->as.string, strlen(value->as.string));
        case VAL_ARRAY:
        case VAL_MAP: {
            Value stringified;
            LanaError error = lana_json_stringify(vm, value, &stringified);
            if (error != LANA_OK) return error;
            return format_add(vm, buffer, stringified.as.string, strlen(stringified.as.string));
        }
        default: return LANA_ERR_TYPE;
    }
}

/* Decode one UTF-8 code point from `s` (of `len` bytes). On success sets
 * `*cp` and `*consumed` and returns LANA_OK; on invalid UTF-8 (overlong
 * encodings, surrogates, out-of-range, truncated sequences) returns
 * LANA_ERR_SCHEMA. Mirrors the Rust `utf8_decode`. */
static LanaError utf8_decode(const unsigned char *s, size_t len, uint32_t *cp,
                             size_t *consumed) {
    unsigned char b0;
    if (len == 0u) return LANA_ERR_SCHEMA;
    b0 = s[0];
    if (b0 < 0x80u) { *cp = b0; *consumed = 1u; return LANA_OK; }
    if (b0 < 0xC2u) return LANA_ERR_SCHEMA;
    if (b0 < 0xE0u) {
        if (len < 2u || (s[1] & 0xC0u) != 0x80u) return LANA_ERR_SCHEMA;
        *cp = ((uint32_t)(b0 & 0x1Fu) << 6u) | (uint32_t)(s[1] & 0x3Fu);
        *consumed = 2u; return LANA_OK;
    }
    if (b0 < 0xF0u) {
        if (len < 3u || (s[1] & 0xC0u) != 0x80u || (s[2] & 0xC0u) != 0x80u)
            return LANA_ERR_SCHEMA;
        if (b0 == 0xE0u && s[1] < 0xA0u) return LANA_ERR_SCHEMA;
        if (b0 == 0xEDu && s[1] >= 0xA0u) return LANA_ERR_SCHEMA;
        *cp = ((uint32_t)(b0 & 0x0Fu) << 12u) | ((uint32_t)(s[1] & 0x3Fu) << 6u) |
              (uint32_t)(s[2] & 0x3Fu);
        *consumed = 3u; return LANA_OK;
    }
    if (b0 < 0xF5u) {
        if (len < 4u || (s[1] & 0xC0u) != 0x80u || (s[2] & 0xC0u) != 0x80u ||
            (s[3] & 0xC0u) != 0x80u)
            return LANA_ERR_SCHEMA;
        if (b0 == 0xF0u && s[1] < 0x90u) return LANA_ERR_SCHEMA;
        if (b0 == 0xF4u && s[1] >= 0x90u) return LANA_ERR_SCHEMA;
        *cp = ((uint32_t)(b0 & 0x07u) << 18u) | ((uint32_t)(s[1] & 0x3Fu) << 12u) |
              ((uint32_t)(s[2] & 0x3Fu) << 6u) | (uint32_t)(s[3] & 0x3Fu);
        *consumed = 4u; return LANA_OK;
    }
    return LANA_ERR_SCHEMA;
}

/* Number of bytes needed to encode `cp` as UTF-8. */
static size_t utf8_encode_len(uint32_t cp) {
    if (cp < 0x80u) return 1u;
    if (cp < 0x800u) return 2u;
    if (cp < 0x10000u) return 3u;
    return 4u;
}

/* Encode `cp` as UTF-8 into `out` (which has room for utf8_encode_len bytes);
 * returns the number of bytes written. */
static size_t utf8_encode(uint32_t cp, unsigned char *out) {
    if (cp < 0x80u) { out[0] = (unsigned char)cp; return 1u; }
    if (cp < 0x800u) {
        out[0] = (unsigned char)(0xC0u | (cp >> 6u));
        out[1] = (unsigned char)(0x80u | (cp & 0x3Fu));
        return 2u;
    }
    if (cp < 0x10000u) {
        out[0] = (unsigned char)(0xE0u | (cp >> 12u));
        out[1] = (unsigned char)(0x80u | ((cp >> 6u) & 0x3Fu));
        out[2] = (unsigned char)(0x80u | (cp & 0x3Fu));
        return 3u;
    }
    out[0] = (unsigned char)(0xF0u | (cp >> 18u));
    out[1] = (unsigned char)(0x80u | ((cp >> 12u) & 0x3Fu));
    out[2] = (unsigned char)(0x80u | ((cp >> 6u) & 0x3Fu));
    out[3] = (unsigned char)(0x80u | (cp & 0x3Fu));
    return 4u;
}

/* ===== Regular expression engine (LIP-021 §2) =====
 * A Thompson NFA engine: parse the pattern into a program of instructions,
 * then simulate the active-state set one byte at a time. Linear in the text
 * length (no backtracking), so pathological patterns cannot blow up. The
 * program and classes are built in a malloc'd growable buffer and copied into
 * GC-tracked memory once complete. */

typedef struct {
    LanaVM *vm;
    const char *pattern;
    size_t len;
    size_t pos;
    LanaRegexInst *insts;
    size_t inst_count;
    size_t inst_cap;
    LanaRegexClass *classes;
    size_t class_count;
    size_t class_cap;
    LanaError error;
    const char *message;
} RegexCompiler;

static void regex_fail(RegexCompiler *c, const char *message) {
    if (c->error == LANA_OK) { c->error = LANA_ERR_SCHEMA; c->message = message; }
}

static bool regex_emit(RegexCompiler *c, LanaRegexOp op, uint32_t a, uint32_t b, uint32_t d) {
    LanaRegexInst inst;
    if (c->inst_count == c->inst_cap) {
        size_t new_cap = c->inst_cap == 0u ? 16u : c->inst_cap * 2u;
        LanaRegexInst *grown = realloc(c->insts, new_cap * sizeof(*grown));
        if (grown == NULL) { c->error = LANA_ERR_OOM; return false; }
        c->insts = grown;
        c->inst_cap = new_cap;
    }
    inst.op = op;
    inst.c = a;
    inst.x = b;
    inst.y = d;
    c->insts[c->inst_count++] = inst;
    return true;
}

static bool regex_add_class(RegexCompiler *c, const uint32_t *bitmap, bool negated) {
    LanaRegexClass cls;
    if (c->class_count == c->class_cap) {
        size_t new_cap = c->class_cap == 0u ? 4u : c->class_cap * 2u;
        LanaRegexClass *grown = realloc(c->classes, new_cap * sizeof(*grown));
        if (grown == NULL) { c->error = LANA_ERR_OOM; return false; }
        c->classes = grown;
        c->class_cap = new_cap;
    }
    memcpy(cls.bitmap, bitmap, sizeof(cls.bitmap));
    cls.negated = negated;
    c->classes[c->class_count++] = cls;
    return true;
}

static void regex_class_set(uint32_t *bitmap, uint32_t b) {
    bitmap[b >> 5u] |= 1u << (b & 31u);
}

static bool regex_class_has(const uint32_t *bitmap, uint32_t b) {
    return (bitmap[b >> 5u] & (1u << (b & 31u))) != 0u;
}

static uint32_t regex_compile_alternation(RegexCompiler *c);
static uint32_t regex_compile_concat(RegexCompiler *c);
static uint32_t regex_compile_repeat(RegexCompiler *c);
static uint32_t regex_compile_atom(RegexCompiler *c);

/* Compile a character class `[...]`. Returns the index of the CLASS
 * instruction. */
static uint32_t regex_compile_class(RegexCompiler *c) {
    uint32_t bitmap[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
    bool negated = false;
    bool first = true;
    uint32_t start;
    c->pos++;  /* skip '[' */
    if (c->pos < c->len && c->pattern[c->pos] == '^') {
        negated = true;
        c->pos++;
    }
    while (c->pos < c->len) {
        char ch = c->pattern[c->pos];
        if (ch == ']' && !first) {
            c->pos++;
            break;
        }
        first = false;
        {
            uint32_t lo = (unsigned char)ch;
            c->pos++;
            if (c->pos + 1u < c->len && c->pattern[c->pos] == '-' &&
                c->pattern[c->pos + 1u] != ']') {
                uint32_t hi, b;
                c->pos++;  /* skip '-' */
                hi = (unsigned char)c->pattern[c->pos];
                c->pos++;
                if (hi < lo) {
                    regex_fail(c, "invalid range in character class");
                    return 0u;
                }
                for (b = lo; b <= hi; ++b) regex_class_set(bitmap, b);
            } else {
                regex_class_set(bitmap, lo);
            }
        }
    }
    if (c->pos >= c->len) {
        regex_fail(c, "unterminated character class");
        return 0u;
    }
    if (!regex_add_class(c, bitmap, negated)) return 0u;
    start = (uint32_t)c->inst_count;
    if (!regex_emit(c, LANA_REGEX_CLASS, (uint32_t)(c->class_count - 1u), 0u, 0u)) return 0u;
    return start;
}

static uint32_t regex_compile_atom(RegexCompiler *c) {
    uint32_t start;
    char ch;
    if (c->pos >= c->len) {
        regex_fail(c, "unexpected end of pattern");
        return 0u;
    }
    ch = c->pattern[c->pos];
    switch (ch) {
        case '.':
            c->pos++;
            start = (uint32_t)c->inst_count;
            if (!regex_emit(c, LANA_REGEX_ANY, 0u, 0u, 0u)) return 0u;
            return start;
        case '^':
            c->pos++;
            start = (uint32_t)c->inst_count;
            if (!regex_emit(c, LANA_REGEX_BOL, 0u, 0u, 0u)) return 0u;
            return start;
        case '$':
            c->pos++;
            start = (uint32_t)c->inst_count;
            if (!regex_emit(c, LANA_REGEX_EOL, 0u, 0u, 0u)) return 0u;
            return start;
        case '[':
            return regex_compile_class(c);
        case '(':
            c->pos++;
            start = regex_compile_alternation(c);
            if (c->error != LANA_OK) return start;
            if (c->pos >= c->len || c->pattern[c->pos] != ')') {
                regex_fail(c, "unterminated group");
                return start;
            }
            c->pos++;  /* skip ')' */
            return start;
        case ')':
            regex_fail(c, "unmatched ')'");
            return 0u;
        case '*':
        case '+':
        case '?':
        case '|':
            regex_fail(c, "dangling metacharacter");
            return 0u;
        case '\\':
            c->pos++;
            if (c->pos >= c->len) {
                regex_fail(c, "trailing backslash");
                return 0u;
            }
            ch = c->pattern[c->pos];
            c->pos++;
            start = (uint32_t)c->inst_count;
            if (!regex_emit(c, LANA_REGEX_CHAR, (unsigned char)ch, 0u, 0u)) return 0u;
            return start;
        default:
            c->pos++;
            start = (uint32_t)c->inst_count;
            if (!regex_emit(c, LANA_REGEX_CHAR, (unsigned char)ch, 0u, 0u)) return 0u;
            return start;
    }
}

/* Compile an atom plus an optional `*`/`+`/`?` suffix. For `*` and `?` the
 * SPLIT must precede the atom, so the atom is shifted down one slot with
 * memmove after emitting the wrapper. */
static uint32_t regex_compile_repeat(RegexCompiler *c) {
    uint32_t atom_start = regex_compile_atom(c);
    uint32_t split;
    if (c->error != LANA_OK || c->pos >= c->len) return atom_start;
    switch (c->pattern[c->pos]) {
        case '*': {
            c->pos++;
            split = (uint32_t)c->inst_count;
            if (!regex_emit(c, LANA_REGEX_SPLIT, 0u, 0u, 0u)) return 0u;
            if (!regex_emit(c, LANA_REGEX_JMP, 0u, 0u, 0u)) return 0u;
            memmove(c->insts + atom_start + 1u, c->insts + atom_start,
                    (split - atom_start) * sizeof(*c->insts));
            c->insts[atom_start].op = LANA_REGEX_SPLIT;
            c->insts[atom_start].c = 0u;
            c->insts[atom_start].x = atom_start + 1u;
            c->insts[atom_start].y = split + 2u;
            c->insts[split + 1u].op = LANA_REGEX_JMP;
            c->insts[split + 1u].c = 0u;
            c->insts[split + 1u].x = atom_start;
            c->insts[split + 1u].y = 0u;
            return atom_start;
        }
        case '+': {
            c->pos++;
            split = (uint32_t)c->inst_count;
            if (!regex_emit(c, LANA_REGEX_SPLIT, 0u, 0u, 0u)) return 0u;
            c->insts[split].x = atom_start;
            c->insts[split].y = split + 1u;
            return atom_start;
        }
        case '?': {
            c->pos++;
            split = (uint32_t)c->inst_count;
            if (!regex_emit(c, LANA_REGEX_SPLIT, 0u, 0u, 0u)) return 0u;
            memmove(c->insts + atom_start + 1u, c->insts + atom_start,
                    (split - atom_start) * sizeof(*c->insts));
            c->insts[atom_start].op = LANA_REGEX_SPLIT;
            c->insts[atom_start].c = 0u;
            c->insts[atom_start].x = atom_start + 1u;
            c->insts[atom_start].y = split + 1u;
            return atom_start;
        }
        default:
            return atom_start;
    }
}

static uint32_t regex_compile_concat(RegexCompiler *c) {
    uint32_t start = (uint32_t)c->inst_count;
    while (c->error == LANA_OK && c->pos < c->len) {
        char ch = c->pattern[c->pos];
        if (ch == '|' || ch == ')') break;
        (void)regex_compile_repeat(c);
    }
    return start;
}

static uint32_t regex_compile_alternation(RegexCompiler *c) {
    uint32_t start = regex_compile_concat(c);
    while (c->error == LANA_OK && c->pos < c->len && c->pattern[c->pos] == '|') {
        uint32_t split, jmp, right;
        c->pos++;  /* skip '|' */
        split = (uint32_t)c->inst_count;
        if (!regex_emit(c, LANA_REGEX_SPLIT, 0u, 0u, 0u)) return start;
        if (!regex_emit(c, LANA_REGEX_JMP, 0u, 0u, 0u)) return start;
        memmove(c->insts + start + 1u, c->insts + start,
                (split - start) * sizeof(*c->insts));
        c->insts[start].op = LANA_REGEX_SPLIT;
        c->insts[start].c = 0u;
        c->insts[start].x = start + 1u;
        c->insts[start].y = 0u;  /* patched below */
        c->insts[split + 1u].op = LANA_REGEX_JMP;
        c->insts[split + 1u].c = 0u;
        c->insts[split + 1u].x = 0u;  /* patched below */
        c->insts[split + 1u].y = 0u;
        right = regex_compile_concat(c);
        if (c->error != LANA_OK) return start;
        jmp = split + 1u;
        c->insts[start].y = right;
        c->insts[jmp].x = (uint32_t)c->inst_count;
    }
    return start;
}

/* Compile a pattern into a GC-tracked LanaRegex. On success returns LANA_OK
 * with `*out` set; on an invalid pattern returns LANA_ERR_SCHEMA with
 * `*error_msg` set to a static string; on allocation failure returns
 * LANA_ERR_OOM. */
static LanaError regex_compile(LanaVM *vm, const char *pattern, LanaRegex **out,
                               const char **error_msg) {
    RegexCompiler c;
    LanaRegex *re;
    memset(&c, 0, sizeof(c));
    c.vm = vm;
    c.pattern = pattern;
    c.len = strlen(pattern);
    c.error = LANA_OK;
    (void)regex_compile_alternation(&c);
    if (c.error != LANA_OK) {
        free(c.insts);
        free(c.classes);
        if (c.error == LANA_ERR_SCHEMA) *error_msg = c.message;
        return c.error;
    }
    if (c.pos < c.len) {
        free(c.insts);
        free(c.classes);
        *error_msg = "unmatched ')'";
        return LANA_ERR_SCHEMA;
    }
    if (!regex_emit(&c, LANA_REGEX_MATCH, 0u, 0u, 0u)) {
        free(c.insts);
        free(c.classes);
        return LANA_ERR_OOM;
    }
    re = lana_vm_alloc(vm, sizeof(*re));
    if (re == NULL) { free(c.insts); free(c.classes); return LANA_ERR_OOM; }
    re->insts = lana_vm_alloc(vm, c.inst_count * sizeof(*re->insts));
    re->classes = lana_vm_alloc(vm, c.class_count * sizeof(*re->classes));
    if (re->insts == NULL || re->classes == NULL) {
        free(c.insts); free(c.classes); return LANA_ERR_OOM;
    }
    memcpy(re->insts, c.insts, c.inst_count * sizeof(*re->insts));
    memcpy(re->classes, c.classes, c.class_count * sizeof(*re->classes));
    re->inst_count = c.inst_count;
    re->class_count = c.class_count;
    free(c.insts);
    free(c.classes);
    *out = re;
    *error_msg = NULL;
    return LANA_OK;
}

/* Add `pc` to the active-state list, following epsilon transitions (SPLIT,
 * JMP) and zero-width assertions (BOL/EOL) that hold at position `pos`. */
static void regex_addstate(const LanaRegex *re, uint32_t *list, size_t *count,
                           bool *seen, uint32_t pc, size_t pos, size_t len) {
    const LanaRegexInst *inst;
    if (seen[pc]) return;
    seen[pc] = true;
    inst = &re->insts[pc];
    switch (inst->op) {
        case LANA_REGEX_SPLIT:
            regex_addstate(re, list, count, seen, inst->x, pos, len);
            regex_addstate(re, list, count, seen, inst->y, pos, len);
            break;
        case LANA_REGEX_JMP:
            regex_addstate(re, list, count, seen, inst->x, pos, len);
            break;
        case LANA_REGEX_BOL:
            if (pos == 0u) regex_addstate(re, list, count, seen, pc + 1u, pos, len);
            break;
        case LANA_REGEX_EOL:
            if (pos == len) regex_addstate(re, list, count, seen, pc + 1u, pos, len);
            break;
        default:
            list[(*count)++] = pc;
            break;
    }
}

/* Advance the active set by one byte `c` at position `pos`. */
static void regex_step(const LanaRegex *re, const uint32_t *clist, size_t clist_count,
                       uint32_t *nlist, size_t *nlist_count, bool *seen,
                       unsigned char c, size_t pos, size_t len) {
    size_t i;
    for (i = 0u; i < clist_count; ++i) {
        uint32_t pc = clist[i];
        const LanaRegexInst *inst = &re->insts[pc];
        switch (inst->op) {
            case LANA_REGEX_CHAR:
                if (inst->c == (uint32_t)c)
                    regex_addstate(re, nlist, nlist_count, seen, pc + 1u, pos + 1u, len);
                break;
            case LANA_REGEX_ANY:
                if (c != '\n')
                    regex_addstate(re, nlist, nlist_count, seen, pc + 1u, pos + 1u, len);
                break;
            case LANA_REGEX_CLASS: {
                const LanaRegexClass *cls = &re->classes[inst->c];
                bool in = regex_class_has(cls->bitmap, (uint32_t)c);
                if (cls->negated) in = !in;
                if (in)
                    regex_addstate(re, nlist, nlist_count, seen, pc + 1u, pos + 1u, len);
                break;
            }
            default:
                break;
        }
    }
}

static bool regex_has_match(const LanaRegex *re, const uint32_t *list, size_t count) {
    size_t i;
    for (i = 0u; i < count; ++i)
        if (re->insts[list[i]].op == LANA_REGEX_MATCH) return true;
    return false;
}

/* Run the NFA from `start`, returning the longest (greedy) match end. Returns
 * LANA_OK (match, `*end` set), LANA_ERR_KEY (no match), or LANA_ERR_OOM. */
static LanaError regex_match_from(LanaVM *vm, const LanaRegex *re, const char *text,
                                  size_t len, size_t start, size_t *end) {
    uint32_t *cur = lana_vm_alloc(vm, re->inst_count * sizeof(*cur));
    uint32_t *next = lana_vm_alloc(vm, re->inst_count * sizeof(*next));
    bool *seen = lana_vm_alloc(vm, re->inst_count * sizeof(*seen));
    size_t cur_count, next_count, pos;
    bool matched = false;
    if (cur == NULL || next == NULL || seen == NULL) return LANA_ERR_OOM;
    memset(seen, 0, re->inst_count * sizeof(*seen));
    cur_count = 0u;
    regex_addstate(re, cur, &cur_count, seen, 0u, start, len);
    if (regex_has_match(re, cur, cur_count)) { *end = start; matched = true; }
    for (pos = start; pos < len; ++pos) {
        uint32_t *tmp;
        memset(seen, 0, re->inst_count * sizeof(*seen));
        next_count = 0u;
        regex_step(re, cur, cur_count, next, &next_count, seen,
                   (unsigned char)text[pos], pos, len);
        tmp = cur; cur = next; next = tmp;
        cur_count = next_count;
        if (regex_has_match(re, cur, cur_count)) { *end = pos + 1u; matched = true; }
    }
    return matched ? LANA_OK : LANA_ERR_KEY;
}

/* Unanchored search: the leftmost-longest (greedy) match. Returns LANA_OK
 * (match, `*start`/`*end` set) or LANA_ERR_KEY (no match). */
static LanaError regex_search(LanaVM *vm, const LanaRegex *re, const char *text,
                              size_t len, size_t from, size_t *start, size_t *end) {
    size_t s;
    for (s = from; s <= len; ++s) {
        LanaError error = regex_match_from(vm, re, text, len, s, end);
        if (error == LANA_OK) { *start = s; return LANA_OK; }
        if (error == LANA_ERR_OOM) return LANA_ERR_OOM;
    }
    return LANA_ERR_KEY;
}

/* Build a Match map {start, end, text} for a match over `text`. */
static LanaError regex_make_match(LanaVM *vm, const char *text, size_t start,
                                  size_t end, Value *out) {
    LanaMap *map;
    LanaError error;
    char *sub = lana_vm_alloc(vm, end - start + 1u);
    if (sub == NULL) return LANA_ERR_OOM;
    memcpy(sub, text + start, end - start);
    sub[end - start] = '\0';
    if ((error = lana_map_new(vm, 3u, &map)) != LANA_OK) return error;
    if ((error = map_put(vm, map, "start", lana_value_number((double)start))) != LANA_OK ||
        (error = map_put(vm, map, "end", lana_value_number((double)end))) != LANA_OK ||
        (error = map_put(vm, map, "text", lana_value_string(sub))) != LANA_OK)
        return error;
    *out = lana_value_map(map);
    return LANA_OK;
}

/* ---------------------------------------------------------------------------
 * LIP-005 linear algebra on STATEs. The four first-class value types
 * (VAL_NQUBIT_STATE, VAL_POVM, VAL_CHANNEL, VAL_OBSERVABLE) are thin wrappers
 * over a complex VAL_TENSOR: a density operator is a d×d matrix, a POVM or
 * channel is a [k, d, d] stack of operators, and an observable is a d×d
 * Hermitian matrix. The arithmetic below is scalar double loops; the tensor
 * backend (LIP-004) supplies matmul/trace/reductions where a host call
 * composes them.
 * ------------------------------------------------------------------------- */

/* Read the (i, j) entry of a 2-D tensor as a complex number. A real tensor
 * has zero imaginary part. */
static void linalg_get2(const LanaTensor *t, size_t i, size_t j, double *re, double *im) {
    size_t lin = t->offset + i * t->strides[0] + j * t->strides[1];
    if (t->is_complex) {
        *re = tensor_get_real(t, lin);
        *im = tensor_get_imag(t, lin);
    } else {
        *re = tensor_get_real(t, lin);
        *im = 0.0;
    }
}

/* Write the (i, j) entry of a 2-D complex tensor. */
static void linalg_set2(LanaTensor *t, size_t i, size_t j, double re, double im) {
    size_t lin = t->offset + i * t->strides[0] + j * t->strides[1];
    tensor_set_real(t, lin, re);
    tensor_set_imag(t, lin, im);
}

/* Read the (k, i, j) entry of a 3-D tensor as a complex number. */
static void linalg_get3(const LanaTensor *t, size_t k, size_t i, size_t j, double *re, double *im) {
    size_t lin = t->offset + k * t->strides[0] + i * t->strides[1] + j * t->strides[2];
    if (t->is_complex) {
        *re = tensor_get_real(t, lin);
        *im = tensor_get_imag(t, lin);
    } else {
        *re = tensor_get_real(t, lin);
        *im = 0.0;
    }
}

/* Write the (k, i, j) entry of a 3-D complex tensor. */
static void linalg_set3(LanaTensor *t, size_t k, size_t i, size_t j, double re, double im) {
    size_t lin = t->offset + k * t->strides[0] + i * t->strides[1] + j * t->strides[2];
    tensor_set_real(t, lin, re);
    tensor_set_imag(t, lin, im);
}

/* Copy a 2-D tensor (real or complex) into a fresh complex tensor. */
static LanaTensor *linalg_copy_complex2(LanaVM *vm, const LanaTensor *t) {
    size_t shape[2] = { t->shape[0], t->shape[1] };
    LanaTensor *r = tensor_new(vm, 2, shape, true);
    size_t i, j;
    if (r == NULL) return NULL;
    for (i = 0; i < t->shape[0]; ++i)
        for (j = 0; j < t->shape[1]; ++j) {
            double re, im;
            linalg_get2(t, i, j, &re, &im);
            linalg_set2(r, i, j, re, im);
        }
    return r;
}

/* The base-2 logarithm of a power-of-two dimension (the qubit count). */
static size_t linalg_qubits(size_t d) {
    size_t n = 0;
    while (d > 1u) { d >>= 1u; ++n; }
    return n;
}

/* Jacobi eigenvalue algorithm for a Hermitian d×d matrix stored as an
 * interleaved [re, im] row-major array. On return the diagonal holds the real
 * eigenvalues (the off-diagonal has been driven to ~0). */
static void linalg_jacobi(double *a, size_t d, double *eigenvalues) {
    const double tol = 1e-12;
    const size_t max_sweeps = 50;
    size_t sweep, p, q, k;
    for (sweep = 0; sweep < max_sweeps; ++sweep) {
        double off = 0.0;
        for (p = 0; p < d; ++p)
            for (q = p + 1; q < d; ++q) {
                double re = a[(p * d + q) * 2];
                double im = a[(p * d + q) * 2 + 1];
                off += re * re + im * im;
            }
        if (off <= tol * tol) break;
        for (p = 0; p < d; ++p) {
            for (q = p + 1; q < d; ++q) {
                double apq_re = a[(p * d + q) * 2];
                double apq_im = a[(p * d + q) * 2 + 1];
                double m = hypot(apq_re, apq_im);
                if (m <= tol) continue;
                double app = a[(p * d + p) * 2];
                double aqq = a[(q * d + q) * 2];
                double theta = 0.5 * atan2(2.0 * m, app - aqq);
                double c = cos(theta);
                double s = sin(theta);
                double cos_phi = apq_re / m;
                double sin_phi = apq_im / m;
                a[(p * d + p) * 2] = c * c * app + 2.0 * c * s * m + s * s * aqq;
                a[(p * d + p) * 2 + 1] = 0.0;
                a[(q * d + q) * 2] = s * s * app - 2.0 * c * s * m + c * c * aqq;
                a[(q * d + q) * 2 + 1] = 0.0;
                a[(p * d + q) * 2] = 0.0;
                a[(p * d + q) * 2 + 1] = 0.0;
                a[(q * d + p) * 2] = 0.0;
                a[(q * d + p) * 2 + 1] = 0.0;
                for (k = 0; k < d; ++k) {
                    if (k == p || k == q) continue;
                    double akp_re = a[(k * d + p) * 2];
                    double akp_im = a[(k * d + p) * 2 + 1];
                    double akq_re = a[(k * d + q) * 2];
                    double akq_im = a[(k * d + q) * 2 + 1];
                    /* new_akp = c*akp + s*e^{-iφ}*akq */
                    double t_re = cos_phi * akq_re + sin_phi * akq_im;
                    double t_im = cos_phi * akq_im - sin_phi * akq_re;
                    double new_akp_re = c * akp_re + s * t_re;
                    double new_akp_im = c * akp_im + s * t_im;
                    /* new_akq = -s*e^{iφ}*akp + c*akq */
                    double u_re = cos_phi * akp_re - sin_phi * akp_im;
                    double u_im = cos_phi * akp_im + sin_phi * akp_re;
                    double new_akq_re = -s * u_re + c * akq_re;
                    double new_akq_im = -s * u_im + c * akq_im;
                    a[(k * d + p) * 2] = new_akp_re;
                    a[(k * d + p) * 2 + 1] = new_akp_im;
                    a[(p * d + k) * 2] = new_akp_re;
                    a[(p * d + k) * 2 + 1] = -new_akp_im;
                    a[(k * d + q) * 2] = new_akq_re;
                    a[(k * d + q) * 2 + 1] = new_akq_im;
                    a[(q * d + k) * 2] = new_akq_re;
                    a[(q * d + k) * 2 + 1] = -new_akq_im;
                }
            }
        }
    }
    for (p = 0; p < d; ++p) eigenvalues[p] = a[(p * d + p) * 2];
}

/* Whether a 2-D tensor is Hermitian (A[i][j] == conj(A[j][i])) within tol. */
static bool linalg_is_hermitian(const LanaTensor *t, double tol) {
    size_t i, j;
    size_t d = t->shape[0];
    for (i = 0; i < d; ++i)
        for (j = i; j < d; ++j) {
            double a_re, a_im, b_re, b_im;
            linalg_get2(t, i, j, &a_re, &a_im);
            linalg_get2(t, j, i, &b_re, &b_im);
            if (fabs(a_re - b_re) > tol || fabs(a_im + b_im) > tol) return false;
        }
    return true;
}

/* Trace of a 2-D tensor (the real part; the imaginary part is ~0 for a
 * Hermitian matrix). */
static double linalg_trace(const LanaTensor *t) {
    size_t i;
    double sum = 0.0;
    for (i = 0; i < t->shape[0]; ++i) {
        double re, im;
        linalg_get2(t, i, i, &re, &im);
        sum += re;
    }
    return sum;
}

/* Eigenvalues of a 2-D tensor (assumed Hermitian), in GC scratch. */
static LanaError linalg_eigenvalues(LanaVM *vm, const LanaTensor *t, double **out) {
    size_t d = t->shape[0];
    double *a = lana_vm_alloc(vm, d * d * 2 * sizeof(*a));
    double *eig = lana_vm_alloc(vm, d * sizeof(*eig));
    size_t i, j;
    if (a == NULL || eig == NULL) return LANA_ERR_OOM;
    for (i = 0; i < d; ++i)
        for (j = 0; j < d; ++j) {
            double re, im;
            linalg_get2(t, i, j, &re, &im);
            a[(i * d + j) * 2] = re;
            a[(i * d + j) * 2 + 1] = im;
        }
    linalg_jacobi(a, d, eig);
    *out = eig;
    return LANA_OK;
}

/* Whether a 2-D tensor is positive semidefinite (all eigenvalues >= -1e-9). */
static bool linalg_is_psd(LanaVM *vm, const LanaTensor *t) {
    double *eig;
    size_t i;
    if (linalg_eigenvalues(vm, t, &eig) != LANA_OK) return false;
    for (i = 0; i < t->shape[0]; ++i)
        if (eig[i] < -1e-9) return false;
    return true;
}

/* density_operator from an N=1 STATE: the 2×2 matrix [[p, c], [c*, 1-p]]. */
static LanaError linalg_density_from_state(LanaVM *vm, const LanaState *state, Value *out) {
    double c_re, c_im;
    size_t shape[2] = { 2, 2 };
    LanaTensor *r = tensor_new(vm, 2, shape, true);
    if (r == NULL) return LANA_ERR_OOM;
    lana_state_reconstruct_c(state, &c_re, &c_im);
    linalg_set2(r, 0, 0, state->p, 0.0);
    linalg_set2(r, 0, 1, c_re, c_im);
    linalg_set2(r, 1, 0, c_re, c_im == 0.0 ? 0.0 : -c_im); /* normalize -0.0 */
    linalg_set2(r, 1, 1, 1.0 - state->p, 0.0);
    *out = lana_value_nqubit_state(r);
    return LANA_OK;
}

/* density_operator from a tensor: validate Hermitian, PSD, unit trace, and
 * that d = 2^N with N <= 10. */
static LanaError linalg_density_from_tensor(LanaVM *vm, const LanaTensor *t, Value *out) {
    size_t d, n;
    LanaTensor *r;
    if (t->ndim != 2 || t->shape[0] != t->shape[1]) return LANA_ERR_INVALID_STATE;
    d = t->shape[0];
    if (d == 0 || (d & (d - 1u)) != 0u) return LANA_ERR_INVALID_STATE;
    n = linalg_qubits(d);
    if (n > 10) return LANA_ERR_INVALID_PARAMETERS;
    if (!linalg_is_hermitian(t, 1e-9)) return LANA_ERR_INVALID_STATE;
    if (!linalg_is_psd(vm, t)) return LANA_ERR_INVALID_STATE;
    if (fabs(linalg_trace(t) - 1.0) > 1e-9) return LANA_ERR_INVALID_STATE;
    r = linalg_copy_complex2(vm, t);
    if (r == NULL) return LANA_ERR_OOM;
    *out = lana_value_nqubit_state(r);
    return LANA_OK;
}

/* povm([E...]): stack the operators and validate each PSD and Σ E_i = I. */
static LanaError linalg_povm(LanaVM *vm, const Value *arg, Value *out) {
    LanaArray *arr;
    size_t k, d, i, r, c;
    LanaTensor *stack;
    if (arg->type != VAL_ARRAY) return LANA_ERR_TYPE;
    arr = arg->as.array;
    k = arr->count;
    if (k == 0) return LANA_ERR_INVALID_PARAMETERS;
    if (arr->items[0].type != VAL_TENSOR) return LANA_ERR_TYPE;
    if (arr->items[0].as.tensor->ndim != 2 ||
        arr->items[0].as.tensor->shape[0] != arr->items[0].as.tensor->shape[1])
        return LANA_ERR_INVALID_PARAMETERS;
    d = arr->items[0].as.tensor->shape[0];
    {
        size_t shape[3] = { k, d, d };
        stack = tensor_new(vm, 3, shape, true);
    }
    if (stack == NULL) return LANA_ERR_OOM;
    for (i = 0; i < k; ++i) {
        const LanaTensor *e;
        if (arr->items[i].type != VAL_TENSOR) return LANA_ERR_TYPE;
        e = arr->items[i].as.tensor;
        if (e->ndim != 2 || e->shape[0] != d || e->shape[1] != d)
            return LANA_ERR_INVALID_PARAMETERS;
        if (!linalg_is_psd(vm, e)) return LANA_ERR_INVALID_PARAMETERS;
        for (r = 0; r < d; ++r)
            for (c = 0; c < d; ++c) {
                double re, im;
                linalg_get2(e, r, c, &re, &im);
                linalg_set3(stack, i, r, c, re, im);
            }
    }
    for (r = 0; r < d; ++r)
        for (c = 0; c < d; ++c) {
            double re = 0.0, im = 0.0;
            for (i = 0; i < k; ++i) {
                double e_re, e_im;
                linalg_get3(stack, i, r, c, &e_re, &e_im);
                re += e_re;
                im += e_im;
            }
            double want_re = (r == c) ? 1.0 : 0.0;
            if (fabs(re - want_re) > 1e-9 || fabs(im) > 1e-9)
                return LANA_ERR_INVALID_PARAMETERS;
        }
    *out = lana_value_povm(stack);
    return LANA_OK;
}

/* channel([K...]): stack the Kraus operators and validate Σ K_k† K_k = I. */
static LanaError linalg_channel(LanaVM *vm, const Value *arg, Value *out) {
    LanaArray *arr;
    size_t k, d, i, r, c, m;
    LanaTensor *stack;
    if (arg->type != VAL_ARRAY) return LANA_ERR_TYPE;
    arr = arg->as.array;
    k = arr->count;
    if (k == 0) return LANA_ERR_INVALID_PARAMETERS;
    if (arr->items[0].type != VAL_TENSOR) return LANA_ERR_TYPE;
    if (arr->items[0].as.tensor->ndim != 2 ||
        arr->items[0].as.tensor->shape[0] != arr->items[0].as.tensor->shape[1])
        return LANA_ERR_INVALID_PARAMETERS;
    d = arr->items[0].as.tensor->shape[0];
    {
        size_t shape[3] = { k, d, d };
        stack = tensor_new(vm, 3, shape, true);
    }
    if (stack == NULL) return LANA_ERR_OOM;
    for (i = 0; i < k; ++i) {
        const LanaTensor *e;
        if (arr->items[i].type != VAL_TENSOR) return LANA_ERR_TYPE;
        e = arr->items[i].as.tensor;
        if (e->ndim != 2 || e->shape[0] != d || e->shape[1] != d)
            return LANA_ERR_INVALID_PARAMETERS;
        for (r = 0; r < d; ++r)
            for (c = 0; c < d; ++c) {
                double re, im;
                linalg_get2(e, r, c, &re, &im);
                linalg_set3(stack, i, r, c, re, im);
            }
    }
    for (r = 0; r < d; ++r)
        for (c = 0; c < d; ++c) {
            double re = 0.0, im = 0.0;
            for (i = 0; i < k; ++i)
                for (m = 0; m < d; ++m) {
                    double k_mr_re, k_mr_im, k_mc_re, k_mc_im;
                    linalg_get3(stack, i, m, r, &k_mr_re, &k_mr_im);
                    linalg_get3(stack, i, m, c, &k_mc_re, &k_mc_im);
                    /* conj(K[m][r]) * K[m][c] */
                    re += k_mr_re * k_mc_re + k_mr_im * k_mc_im;
                    im += k_mr_re * k_mc_im - k_mr_im * k_mc_re;
                }
            double want_re = (r == c) ? 1.0 : 0.0;
            if (fabs(re - want_re) > 1e-9 || fabs(im) > 1e-9)
                return LANA_ERR_INVALID_PARAMETERS;
        }
    *out = lana_value_channel(stack);
    return LANA_OK;
}

/* observable(A): validate Hermitian. */
static LanaError linalg_observable(LanaVM *vm, const Value *arg, Value *out) {
    const LanaTensor *t;
    LanaTensor *r;
    if (arg->type != VAL_TENSOR) return LANA_ERR_TYPE;
    t = arg->as.tensor;
    if (t->ndim != 2 || t->shape[0] != t->shape[1]) return LANA_ERR_INVALID_PARAMETERS;
    if (!linalg_is_hermitian(t, 1e-9)) return LANA_ERR_INVALID_PARAMETERS;
    r = linalg_copy_complex2(vm, t);
    if (r == NULL) return LANA_ERR_OOM;
    *out = lana_value_observable(r);
    return LANA_OK;
}

/* tensor_product(a, b): the Kronecker product ρ_A ⊗ ρ_B. */
static LanaError linalg_tensor_product(LanaVM *vm, const LanaTensor *a, const LanaTensor *b, Value *out) {
    size_t da = a->shape[0], db = b->shape[0];
    size_t shape[2] = { da * db, da * db };
    LanaTensor *r = tensor_new(vm, 2, shape, true);
    size_t i1, i2, j1, j2;
    if (r == NULL) return LANA_ERR_OOM;
    for (i1 = 0; i1 < da; ++i1)
        for (i2 = 0; i2 < da; ++i2)
            for (j1 = 0; j1 < db; ++j1)
                for (j2 = 0; j2 < db; ++j2) {
                    double a_re, a_im, b_re, b_im;
                    linalg_get2(a, i1, i2, &a_re, &a_im);
                    linalg_get2(b, j1, j2, &b_re, &b_im);
                    linalg_set2(r, i1 * db + j1, i2 * db + j2,
                                a_re * b_re - a_im * b_im, a_re * b_im + a_im * b_re);
                }
    *out = lana_value_nqubit_state(r);
    return LANA_OK;
}

/* partial_trace(ab, subsystem): trace out the second subsystem, keeping the
 * first `subsystem` qubits. `subsystem` is the qubit count of the kept
 * subsystem A (1 <= subsystem < N). */
static LanaError linalg_partial_trace(LanaVM *vm, const LanaTensor *ab, double subsystem, Value *out) {
    size_t d = ab->shape[0];
    size_t n = linalg_qubits(d);
    size_t k, da, db, i, j, m;
    LanaTensor *r;
    if (!isfinite(subsystem) || subsystem < 1.0 || floor(subsystem) != subsystem)
        return LANA_ERR_INVALID_PARAMETERS;
    k = (size_t)subsystem;
    if (k >= n) return LANA_ERR_INVALID_PARAMETERS;
    da = (size_t)1u << k;
    db = d / da;
    {
        size_t shape[2] = { da, da };
        r = tensor_new(vm, 2, shape, true);
    }
    if (r == NULL) return LANA_ERR_OOM;
    for (i = 0; i < da; ++i)
        for (j = 0; j < da; ++j) {
            double re = 0.0, im = 0.0;
            for (m = 0; m < db; ++m) {
                double e_re, e_im;
                linalg_get2(ab, i * db + m, j * db + m, &e_re, &e_im);
                re += e_re;
                im += e_im;
            }
            linalg_set2(r, i, j, re, im);
        }
    *out = lana_value_nqubit_state(r);
    return LANA_OK;
}

/* measure_with(rho, povm): the outcome distribution p(i) = Tr(ρ E_i). */
static LanaError linalg_measure_with(LanaVM *vm, const LanaTensor *rho, const LanaTensor *povm, Value *out) {
    size_t k = povm->shape[0];
    size_t d = povm->shape[1];
    LanaArray *arr = lana_vm_alloc(vm, sizeof(*arr));
    size_t i, r, c;
    if (arr == NULL) return LANA_ERR_OOM;
    arr->count = k;
    arr->capacity = k;
    arr->items = lana_vm_alloc(vm, k * sizeof(*arr->items));
    if (arr->items == NULL && k > 0) return LANA_ERR_OOM;
    for (i = 0; i < k; ++i) {
        double re = 0.0;
        for (r = 0; r < d; ++r)
            for (c = 0; c < d; ++c) {
                double rho_re, rho_im, e_re, e_im;
                linalg_get2(rho, r, c, &rho_re, &rho_im);
                linalg_get3(povm, i, c, r, &e_re, &e_im);
                re += rho_re * e_re - rho_im * e_im;
            }
        arr->items[i] = lana_value_number(re);
    }
    *out = lana_value_array(arr);
    return LANA_OK;
}

/* apply_to(chan, rho): Φ(ρ) = Σ_k K_k ρ K_k†. */
static LanaError linalg_apply_to(LanaVM *vm, const LanaTensor *chan, const LanaTensor *rho, Value *out) {
    size_t k = chan->shape[0];
    size_t d = chan->shape[1];
    size_t shape[2] = { d, d };
    LanaTensor *r = tensor_new(vm, 2, shape, true);
    size_t i, j, m, n, kk;
    if (r == NULL) return LANA_ERR_OOM;
    for (kk = 0; kk < k; ++kk)
        for (i = 0; i < d; ++i)
            for (j = 0; j < d; ++j) {
                double re = 0.0, im = 0.0;
                for (m = 0; m < d; ++m)
                    for (n = 0; n < d; ++n) {
                        double k_re, k_im, rho_re, rho_im, kd_re, kd_im;
                        linalg_get3(chan, kk, i, m, &k_re, &k_im);
                        linalg_get2(rho, m, n, &rho_re, &rho_im);
                        linalg_get3(chan, kk, j, n, &kd_re, &kd_im);
                        /* K[i][m] * ρ[m][n] * conj(K[j][n]) */
                        double t_re = k_re * rho_re - k_im * rho_im;
                        double t_im = k_re * rho_im + k_im * rho_re;
                        re += t_re * kd_re + t_im * kd_im;
                        im += t_im * kd_re - t_re * kd_im;
                    }
                double cur_re, cur_im;
                linalg_get2(r, i, j, &cur_re, &cur_im);
                linalg_set2(r, i, j, cur_re + re, cur_im + im);
            }
    *out = lana_value_nqubit_state(r);
    return LANA_OK;
}

/* expect(rho, obs): ⟨A⟩ = Tr(ρ A). */
static LanaError linalg_expect(const LanaTensor *rho, const LanaTensor *obs, Value *out) {
    size_t d = rho->shape[0];
    double re = 0.0;
    size_t r, c;
    for (r = 0; r < d; ++r)
        for (c = 0; c < d; ++c) {
            double rho_re, rho_im, a_re, a_im;
            linalg_get2(rho, r, c, &rho_re, &rho_im);
            linalg_get2(obs, c, r, &a_re, &a_im);
            re += rho_re * a_re - rho_im * a_im;
        }
    *out = lana_value_number(re);
    return LANA_OK;
}

/* mix(a, b, w): the convex mixture w·a + (1-w)·b. */
static LanaError linalg_mix(LanaVM *vm, const LanaTensor *a, const LanaTensor *b, double w, Value *out) {
    size_t d = a->shape[0];
    size_t shape[2] = { d, d };
    LanaTensor *r = tensor_new(vm, 2, shape, true);
    size_t i, j;
    if (r == NULL) return LANA_ERR_OOM;
    for (i = 0; i < d; ++i)
        for (j = 0; j < d; ++j) {
            double a_re, a_im, b_re, b_im;
            linalg_get2(a, i, j, &a_re, &a_im);
            linalg_get2(b, i, j, &b_re, &b_im);
            linalg_set2(r, i, j, w * a_re + (1.0 - w) * b_re, w * a_im + (1.0 - w) * b_im);
        }
    *out = lana_value_nqubit_state(r);
    return LANA_OK;
}

/* trace_distance(a, b): ½‖ρ − σ‖₁ = ½ Σ |λ_i(ρ − σ)|. */
static LanaError linalg_trace_distance(LanaVM *vm, const LanaTensor *a, const LanaTensor *b, Value *out) {
    size_t d = a->shape[0];
    double *diff = lana_vm_alloc(vm, d * d * 2 * sizeof(*diff));
    double *eig = lana_vm_alloc(vm, d * sizeof(*eig));
    size_t i, j;
    double sum = 0.0;
    if (diff == NULL || eig == NULL) return LANA_ERR_OOM;
    for (i = 0; i < d; ++i)
        for (j = 0; j < d; ++j) {
            double a_re, a_im, b_re, b_im;
            linalg_get2(a, i, j, &a_re, &a_im);
            linalg_get2(b, i, j, &b_re, &b_im);
            diff[(i * d + j) * 2] = a_re - b_re;
            diff[(i * d + j) * 2 + 1] = a_im - b_im;
        }
    linalg_jacobi(diff, d, eig);
    for (i = 0; i < d; ++i) sum += fabs(eig[i]);
    *out = lana_value_number(0.5 * sum);
    return LANA_OK;
}

/* is_separable(ab, bipartition): the PPT (Peres-Horodecki) criterion. A
 * negative partial-transpose eigenvalue proves entanglement; a PSD partial
 * transpose proves separability for 2×2 and 2×3 systems and is otherwise
 * inconclusive. `bipartition` is the qubit count of the first subsystem. */
static LanaError linalg_is_separable(LanaVM *vm, const LanaTensor *ab, double bipartition, Value *out) {
    size_t d = ab->shape[0];
    size_t n = linalg_qubits(d);
    size_t k, da, db, i1, i2, j1, j2;
    double *pt, *eig;
    bool negative = false;
    bool provably_separable;
    if (!isfinite(bipartition) || bipartition < 1.0 || floor(bipartition) != bipartition)
        return LANA_ERR_INVALID_PARAMETERS;
    k = (size_t)bipartition;
    if (k >= n) return LANA_ERR_INVALID_PARAMETERS;
    da = (size_t)1u << k;
    db = d / da;
    pt = lana_vm_alloc(vm, d * d * 2 * sizeof(*pt));
    eig = lana_vm_alloc(vm, d * sizeof(*eig));
    if (pt == NULL || eig == NULL) return LANA_ERR_OOM;
    for (i1 = 0; i1 < da; ++i1)
        for (i2 = 0; i2 < da; ++i2)
            for (j1 = 0; j1 < db; ++j1)
                for (j2 = 0; j2 < db; ++j2) {
                    double re, im;
                    linalg_get2(ab, i1 * db + j2, i2 * db + j1, &re, &im);
                    pt[((i1 * db + j1) * d + (i2 * db + j2)) * 2] = re;
                    pt[((i1 * db + j1) * d + (i2 * db + j2)) * 2 + 1] = im;
                }
    linalg_jacobi(pt, d, eig);
    for (i1 = 0; i1 < d; ++i1)
        if (eig[i1] < -1e-9) { negative = true; break; }
    provably_separable = (da == 1u || db == 1u) ||
        (da == 2u && db == 2u) || (da == 2u && db == 3u) || (da == 3u && db == 2u);
    if (negative) *out = lana_value_string("entangled");
    else if (provably_separable) *out = lana_value_string("separable");
    else *out = lana_value_string("inconclusive");
    return LANA_OK;
}

/* to_state(rho): recover the N=1 (p, d_re, d_im) form from a 1-qubit density
 * operator. */
static LanaError linalg_to_state(const LanaTensor *rho, Value *out) {
    double p, c_re, c_im, scale, dummy;
    LanaState state;
    LanaError error;
    if (rho->shape[0] != 2) return LANA_ERR_INVALID_PARAMETERS;
    linalg_get2(rho, 0, 0, &p, &dummy);
    linalg_get2(rho, 0, 1, &c_re, &c_im);
    scale = sqrt(p * (1.0 - p));
    if (scale > 0.0)
        error = lana_state_make_complex(p, c_re / scale, c_im / scale, &state);
    else
        error = lana_state_make_complex(p, 0.0, 0.0, &state);
    if (error != LANA_OK) return error;
    *out = lana_value_state(state);
    return LANA_OK;
}

/* ===== LIP-007 differentiable STATE tensors ===== */

/* Validate a d×d density matrix stored as interleaved [re, im] row-major data.
 * Checks Hermitian, PSD, and unit trace (LIP-005 §1.2). */
static LanaError linalg_validate_density_data(LanaVM *vm, const double *data, size_t d) {
    size_t i, j;
    for (i = 0; i < d; ++i)
        for (j = i; j < d; ++j) {
            double a_re = data[(i * d + j) * 2], a_im = data[(i * d + j) * 2 + 1];
            double b_re = data[(j * d + i) * 2], b_im = data[(j * d + i) * 2 + 1];
            if (fabs(a_re - b_re) > 1e-9 || fabs(a_im + b_im) > 1e-9)
                return LANA_ERR_INVALID_STATE;
        }
    {
        double *a = lana_vm_alloc(vm, d * d * 2 * sizeof(*a));
        double *eig = lana_vm_alloc(vm, d * sizeof(*eig));
        if (a == NULL || eig == NULL) return LANA_ERR_OOM;
        memcpy(a, data, d * d * 2 * sizeof(*a));
        linalg_jacobi(a, d, eig);
        for (i = 0; i < d; ++i)
            if (eig[i] < -1e-9) return LANA_ERR_INVALID_STATE;
    }
    {
        double trace = 0.0;
        for (i = 0; i < d; ++i) trace += data[(i * d + i) * 2];
        if (fabs(trace - 1.0) > 1e-9) return LANA_ERR_INVALID_STATE;
    }
    return LANA_OK;
}

/* Recursively fill a complex tensor's interleaved [re, im] buffer from a nested
 * array of real numbers (imaginary parts are zero). */
static LanaError tensor_fill_state(const Value *v, double *data, size_t *offset) {
    if (v->type == VAL_NUMBER) {
        data[2 * (*offset)] = v->as.number;
        data[2 * (*offset) + 1] = 0.0;
        (*offset)++;
        return LANA_OK;
    }
    if (v->type != VAL_ARRAY) return LANA_ERR_TYPE;
    LanaArray *arr = v->as.array;
    for (size_t i = 0; i < arr->count; ++i) {
        LanaError e = tensor_fill_state(&arr->items[i], data, offset);
        if (e != LANA_OK) return e;
    }
    return LANA_OK;
}

/* state_tensor(literal): construct a STATE tensor from a nested literal of
 * density matrices (real entries; imaginary parts are zero). The literal shape
 * is [s_1, ..., s_k, d, d]; each d×d element is validated as a density
 * operator. */
static LanaError linalg_state_tensor(LanaVM *vm, const Value *arg, Value *out) {
    size_t ndim, *shape, d, n, batch, i;
    LanaTensor *t;
    LanaError error;
    if (arg->type != VAL_ARRAY) return LANA_ERR_TYPE;
    error = tensor_infer_shape(vm, arg, &ndim, &shape);
    if (error != LANA_OK) return error;
    if (ndim < 2) return LANA_ERR_INVALID_PARAMETERS;
    if (shape[ndim - 1] != shape[ndim - 2]) return LANA_ERR_INVALID_PARAMETERS;
    d = shape[ndim - 1];
    if (d < 2 || (d & (d - 1u)) != 0u) return LANA_ERR_INVALID_PARAMETERS;
    n = linalg_qubits(d);
    if (n > 10) return LANA_ERR_INVALID_PARAMETERS;
    t = tensor_new(vm, ndim, shape, true);
    if (t == NULL) return LANA_ERR_OOM;
    t->is_state = true;
    {
        size_t offset = 0;
        error = tensor_fill_state(arg, (double*)t->data, &offset);
        if (error != LANA_OK) return error;
    }
    batch = 1;
    for (i = 0; i + 2 < ndim; ++i) batch *= shape[i];
    for (i = 0; i < batch; ++i) {
        error = linalg_validate_density_data(vm, (double*)t->data + i * d * d * 2, d);
        if (error != LANA_OK) return error;
    }
    *out = lana_value_tensor(t);
    return LANA_OK;
}

/* append(a, b): element-wise distribution-valued APPEND (mean state). N=1
 * (single-qubit) only. */
static LanaError linalg_state_append(LanaVM *vm, const LanaTensor *a, const LanaTensor *b, Value *out) {
    size_t d, batch, i;
    LanaTensor *r;
    if (a->ndim != b->ndim) return LANA_ERR_INVALID_PARAMETERS;
    for (i = 0; i < a->ndim; ++i)
        if (a->shape[i] != b->shape[i]) return LANA_ERR_INVALID_PARAMETERS;
    if (a->ndim < 2) return LANA_ERR_INVALID_PARAMETERS;
    d = a->shape[a->ndim - 1];
    if (d != 2) return LANA_ERR_INVALID_PARAMETERS;
    r = tensor_new(vm, a->ndim, a->shape, true);
    if (r == NULL) return LANA_ERR_OOM;
    r->is_state = true;
    batch = 1;
    for (i = 0; i + 2 < a->ndim; ++i) batch *= a->shape[i];
    for (i = 0; i < batch; ++i) {
        const double *da = (const double*)a->data + i * d * d * 2;
        const double *db = (const double*)b->data + i * d * d * 2;
        double *dr = (double*)r->data + i * d * d * 2;
        double p_a = da[0], c_a_re = da[2], c_a_im = da[3];
        double p_b = db[0], c_b_re = db[2], c_b_im = db[3];
        double s_a = sqrt(p_a * (1.0 - p_a));
        double s_b = sqrt(p_b * (1.0 - p_b));
        double d_a_re = s_a > 0.0 ? c_a_re / s_a : 0.0;
        double d_a_im = s_a > 0.0 ? c_a_im / s_a : 0.0;
        double d_b_re = s_b > 0.0 ? c_b_re / s_b : 0.0;
        double d_b_im = s_b > 0.0 ? c_b_im / s_b : 0.0;
        double p_c = p_a + p_b - p_a * p_b;
        double d_c_re = (d_a_re + d_b_re) / 2.0;
        double d_c_im = (d_a_im + d_b_im) / 2.0;
        double s_c = sqrt(p_c * (1.0 - p_c));
        double c_c_re = d_c_re * s_c;
        double c_c_im = d_c_im * s_c;
        /* ρ_C = [[p_C, c_C], [c_C*, 1-p_C]]. */
        dr[0] = p_c; dr[1] = 0.0;
        dr[2] = c_c_re; dr[3] = c_c_im;
        dr[4] = c_c_re; dr[5] = c_c_im == 0.0 ? 0.0 : -c_c_im;
        dr[6] = 1.0 - p_c; dr[7] = 0.0;
    }
    *out = lana_value_tensor(r);
    return LANA_OK;
}

/* measure(s, povm): element-wise outcome probability q[..., i] = Tr(ρ E_i). */
static LanaError linalg_state_measure(LanaVM *vm, const LanaTensor *s, const LanaTensor *povm, Value *out) {
    size_t d = s->shape[s->ndim - 1];
    size_t k = povm->shape[0];
    size_t batch, i, r, c, m;
    size_t out_ndim = s->ndim - 1;
    size_t *out_shape;
    LanaTensor *res;
    if (povm->shape[1] != d || povm->shape[2] != d) return LANA_ERR_INVALID_PARAMETERS;
    batch = 1;
    for (i = 0; i + 2 < s->ndim; ++i) batch *= s->shape[i];
    out_shape = lana_vm_alloc(vm, out_ndim * sizeof(*out_shape));
    if (out_shape == NULL && out_ndim > 0) return LANA_ERR_OOM;
    for (i = 0; i + 2 < s->ndim; ++i) out_shape[i] = s->shape[i];
    out_shape[out_ndim - 1] = k;
    res = tensor_new(vm, out_ndim, out_shape, false);
    if (res == NULL) return LANA_ERR_OOM;
    for (i = 0; i < batch; ++i) {
        const double *ds = (const double*)s->data + i * d * d * 2;
        for (m = 0; m < k; ++m) {
            double re = 0.0;
            for (r = 0; r < d; ++r)
                for (c = 0; c < d; ++c) {
                    double rho_re = ds[(r * d + c) * 2], rho_im = ds[(r * d + c) * 2 + 1];
                    double e_re, e_im;
                    linalg_get3(povm, m, c, r, &e_re, &e_im);
                    re += rho_re * e_re - rho_im * e_im;
                }
            tensor_set_real(res, i * k + m, re);
        }
    }
    *out = lana_value_tensor(res);
    return LANA_OK;
}

/* transform(s, chan): element-wise channel application Φ(ρ) = Σ_k K_k ρ K_k†. */
static LanaError linalg_state_transform(LanaVM *vm, const LanaTensor *s, const LanaTensor *chan, Value *out) {
    size_t d = s->shape[s->ndim - 1];
    size_t k = chan->shape[0];
    size_t batch, i, r, c, m, n, kk;
    LanaTensor *res;
    if (chan->shape[1] != d || chan->shape[2] != d) return LANA_ERR_INVALID_PARAMETERS;
    batch = 1;
    for (i = 0; i + 2 < s->ndim; ++i) batch *= s->shape[i];
    res = tensor_new(vm, s->ndim, s->shape, true);
    if (res == NULL) return LANA_ERR_OOM;
    res->is_state = true;
    for (i = 0; i < batch; ++i) {
        const double *ds = (const double*)s->data + i * d * d * 2;
        double *dr = (double*)res->data + i * d * d * 2;
        for (r = 0; r < d; ++r)
            for (c = 0; c < d; ++c) {
                double re = 0.0, im = 0.0;
                for (kk = 0; kk < k; ++kk)
                    for (m = 0; m < d; ++m)
                        for (n = 0; n < d; ++n) {
                            double k_re, k_im, rho_re, rho_im, kd_re, kd_im;
                            linalg_get3(chan, kk, r, m, &k_re, &k_im);
                            rho_re = ds[(m * d + n) * 2]; rho_im = ds[(m * d + n) * 2 + 1];
                            linalg_get3(chan, kk, c, n, &kd_re, &kd_im);
                            /* K[r][m] * ρ[m][n] * conj(K[c][n]) */
                            double t_re = k_re * rho_re - k_im * rho_im;
                            double t_im = k_re * rho_im + k_im * rho_re;
                            re += t_re * kd_re + t_im * kd_im;
                            im += t_im * kd_re - t_re * kd_im;
                        }
                dr[(r * d + c) * 2] = re;
                dr[(r * d + c) * 2 + 1] = im;
            }
    }
    *out = lana_value_tensor(res);
    return LANA_OK;
}

/* ---------------------------------------------------------------------------
 * LIP-018 two-way FFI.
 *
 * `ffi_declare` parses a C-style signature string and stores it in a per-VM
 * table; `ffi_load` dlopens a shared library; `ffi_call` marshals a bounded
 * set of argument types, invokes the symbol through libffi, and unmarshals the
 * return. A faulting callee is contained by a sigsetjmp guard and reported as
 * `LANA_ERR_EXTERNAL` rather than taking down the VM.
 * ------------------------------------------------------------------------- */

typedef enum {
    LANA_FFI_VOID = 0,
    LANA_FFI_INT,
    LANA_FFI_DOUBLE,
    LANA_FFI_STRING,
    LANA_FFI_ARRAY
} FfiType;

typedef struct {
    FfiType ret;
    char name[128];
    FfiType args[16];
    size_t arg_count;
} FfiSignature;

static bool ffi_type_end(char c) {
    return c == ' ' || c == '\t' || c == ')' || c == ',' || c == '\0';
}

static bool ffi_parse_type(const char **cursor, FfiType *out) {
    const char *p = *cursor;
    while (*p == ' ' || *p == '\t') ++p;
    if (strncmp(p, "void", 4) == 0 && ffi_type_end(p[4])) { *cursor = p + 4; *out = LANA_FFI_VOID; return true; }
    if (strncmp(p, "int", 3) == 0 && ffi_type_end(p[3])) { *cursor = p + 3; *out = LANA_FFI_INT; return true; }
    if (strncmp(p, "double", 6) == 0 && ffi_type_end(p[6])) { *cursor = p + 6; *out = LANA_FFI_DOUBLE; return true; }
    if (strncmp(p, "const char *", 12) == 0) { *cursor = p + 12; *out = LANA_FFI_STRING; return true; }
    if (strncmp(p, "array", 5) == 0 && ffi_type_end(p[5])) { *cursor = p + 5; *out = LANA_FFI_ARRAY; return true; }
    return false;
}

static bool ffi_parse_signature(const char *sig, FfiSignature *out) {
    const char *p;
    size_t i;
    if (sig == NULL) return false;
    memset(out, 0, sizeof(*out));
    p = sig;
    if (!ffi_parse_type(&p, &out->ret)) return false;
    while (*p == ' ' || *p == '\t') ++p;
    i = 0u;
    while (*p != '\0' && *p != '(' && i + 1u < sizeof(out->name)) out->name[i++] = *p++;
    out->name[i] = '\0';
    if (*p != '(') return false;
    ++p;
    while (*p == ' ' || *p == '\t') ++p;
    if (*p == ')') return true; /* no arguments */
    /* C's `(void)` means no arguments. */
    if (strncmp(p, "void", 4) == 0 && p[4] == ')') return true;
    for (;;) {
        if (out->arg_count >= 16u) return false;
        if (!ffi_parse_type(&p, &out->args[out->arg_count])) return false;
        ++out->arg_count;
        while (*p == ' ' || *p == '\t') ++p;
        if (*p == ',') { ++p; continue; }
        if (*p == ')') return true;
        return false;
    }
}

/* Crash containment: the single VM currently under the FFI guard. A fault in
 * the callee longjmps back to the guard, which reports LANA_ERR_EXTERNAL. */
static LanaVM *ffi_guarded_vm = NULL;

static void ffi_fault_handler(int signal_number) {
    (void)signal_number;
    if (ffi_guarded_vm != NULL) {
        ffi_guarded_vm->ffi_faulted = true;
        siglongjmp(ffi_guarded_vm->ffi_jmp, 1);
    }
}

static bool ffi_validate_args(const FfiSignature *sig, const Value *args, size_t argc) {
    size_t i;
    if (argc != sig->arg_count) return false;
    for (i = 0u; i < argc; ++i) {
        switch (sig->args[i]) {
            case LANA_FFI_INT:
            case LANA_FFI_DOUBLE:
                if (args[i].type != VAL_NUMBER) return false;
                break;
            case LANA_FFI_STRING:
                if (args[i].type != VAL_STRING) return false;
                break;
            case LANA_FFI_ARRAY:
                if (args[i].type != VAL_ARRAY) return false;
                break;
            default:
                return false;
        }
    }
    return true;
}

static LanaError ffi_result_map(LanaVM *vm, const char *key, const Value *value,
                                Value *out) {
    LanaMap *map;
    LanaError error = lana_map_new(vm, 1u, &map);
    if (error != LANA_OK) return error;
    error = lana_map_set(vm, map, key, value, true);
    if (error != LANA_OK) return error;
    *out = lana_value_map(map);
    return LANA_OK;
}

static LanaError ffi_call_impl(LanaVM *vm, const FfiSignature *sig, void *lib,
                               const Value *args, size_t argc, Value *out) {
    ffi_cif cif;
    ffi_type *arg_types[16];
    void *arg_values[16];
    double dargs[16];
    int iargs[16];
    const char *sargs[16];
    struct { void *data; size_t len; } aargs[16];
    union { void *p; void (*f)(void); } sym;
    ffi_type *ret_type;
    union { double d; int i; } ret;
    struct sigaction old_segv, old_bus, act;
    size_t i;

    if (argc != sig->arg_count) return LANA_ERR_TYPE;
    if (lib == NULL) return LANA_ERR_INVALID_STATE;

    for (i = 0u; i < argc; ++i) {
        switch (sig->args[i]) {
            case LANA_FFI_INT:
                if (args[i].type != VAL_NUMBER) return LANA_ERR_TYPE;
                iargs[i] = (int)args[i].as.number;
                arg_types[i] = &ffi_type_sint;
                arg_values[i] = &iargs[i];
                break;
            case LANA_FFI_DOUBLE:
                if (args[i].type != VAL_NUMBER) return LANA_ERR_TYPE;
                dargs[i] = args[i].as.number;
                arg_types[i] = &ffi_type_double;
                arg_values[i] = &dargs[i];
                break;
            case LANA_FFI_STRING:
                if (args[i].type != VAL_STRING) return LANA_ERR_TYPE;
                sargs[i] = args[i].as.string;
                arg_types[i] = &ffi_type_pointer;
                arg_values[i] = &sargs[i];
                break;
            case LANA_FFI_ARRAY:
                if (args[i].type != VAL_ARRAY) return LANA_ERR_TYPE;
                aargs[i].data = args[i].as.array->items;
                aargs[i].len = args[i].as.array->count;
                arg_types[i] = &ffi_type_pointer;
                arg_values[i] = &aargs[i];
                break;
            default:
                return LANA_ERR_TYPE;
        }
    }

    switch (sig->ret) {
        case LANA_FFI_VOID: ret_type = &ffi_type_void; break;
        case LANA_FFI_INT: ret_type = &ffi_type_sint; break;
        case LANA_FFI_DOUBLE: ret_type = &ffi_type_double; break;
        default: return LANA_ERR_TYPE;
    }

    if (ffi_prep_cif(&cif, FFI_DEFAULT_ABI, (unsigned)argc, ret_type, arg_types) != FFI_OK)
        return LANA_ERR_EXTERNAL;

    sym.p = dlsym(lib, sig->name);
    if (sym.p == NULL) return LANA_ERR_EXTERNAL;

    vm->ffi_faulted = false;
    ffi_guarded_vm = vm;
    memset(&act, 0, sizeof(act));
    act.sa_handler = ffi_fault_handler;
    sigemptyset(&act.sa_mask);
    act.sa_flags = 0;
    sigaction(SIGSEGV, &act, &old_segv);
    sigaction(SIGBUS, &act, &old_bus);
    if (sigsetjmp(vm->ffi_jmp, 1) == 0) {
        ffi_call(&cif, sym.f, &ret, arg_values);
    }
    sigaction(SIGSEGV, &old_segv, NULL);
    sigaction(SIGBUS, &old_bus, NULL);
    ffi_guarded_vm = NULL;
    if (vm->ffi_faulted) return LANA_ERR_EXTERNAL;

    switch (sig->ret) {
        case LANA_FFI_VOID: *out = lana_value_null(); break;
        case LANA_FFI_INT: *out = lana_value_number((double)ret.i); break;
        case LANA_FFI_DOUBLE: *out = lana_value_number(ret.d); break;
        default: return LANA_ERR_TYPE;
    }
    return LANA_OK;
}

/* ---- LIP-019 networking ---- */

/* Close a socket (fd + TLS). */
static void net_socket_close(LanaSocket *sock) {
    if (sock->ssl != NULL) {
        SSL_shutdown((SSL *)sock->ssl);
        SSL_free((SSL *)sock->ssl);
        sock->ssl = NULL;
    }
    if (sock->fd >= 0) { close(sock->fd); sock->fd = -1; }
    sock->is_tls = false;
}

/* Result map: {"ok": value} or {"error": reason}. */
static LanaError net_result_map(LanaVM *vm, const char *key, const Value *value,
                                Value *out) {
    LanaMap *map;
    LanaError error = lana_map_new(vm, 1u, &map);
    if (error != LANA_OK) return error;
    error = lana_map_set(vm, map, key, value, true);
    if (error != LANA_OK) return error;
    *out = lana_value_map(map);
    return LANA_OK;
}

static LanaError net_error_result(LanaVM *vm, const char *reason, Value *out) {
    Value e = lana_value_string(reason);
    return net_result_map(vm, "error", &e, out);
}

/* Parse a URL into scheme, host, port, path. Returns false on malformed. */
static bool net_parse_url(const char *url, char *scheme, size_t scheme_cap,
                          char *host, size_t host_cap, int *port,
                          char *path, size_t path_cap) {
    const char *scheme_end = strstr(url, "://");
    const char *host_start, *host_end, *path_start;
    size_t slen, hlen, plen;
    if (scheme_end == NULL) return false;
    slen = (size_t)(scheme_end - url);
    if (slen == 0 || slen >= scheme_cap) return false;
    memcpy(scheme, url, slen); scheme[slen] = '\0';
    host_start = scheme_end + 3;
    host_end = host_start;
    while (*host_end != '\0' && *host_end != ':' && *host_end != '/') ++host_end;
    hlen = (size_t)(host_end - host_start);
    if (hlen == 0 || hlen >= host_cap) return false;
    memcpy(host, host_start, hlen); host[hlen] = '\0';
    *port = 0;
    if (*host_end == ':') {
        const char *port_start = host_end + 1;
        const char *port_end = port_start;
        while (*port_end >= '0' && *port_end <= '9') ++port_end;
        if (port_end == port_start) return false;
        *port = atoi(port_start);
        host_end = port_end;
    }
    path_start = host_end;
    if (*path_start == '\0') path_start = "/";
    plen = strlen(path_start);
    if (plen >= path_cap) return false;
    memcpy(path, path_start, plen + 1);
    if (*port == 0) *port = strcmp(scheme, "https") == 0 ? 443 : 80;
    return true;
}

/* Connect a TCP socket to host:port with a timeout. Returns fd or -1. */
static int net_connect(const char *host, int port, int timeout_ms, bool *timed_out) {
    struct addrinfo hints, *res = NULL, *rp;
    char port_str[16];
    int fd = -1;
    int rc;
    *timed_out = false;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    snprintf(port_str, sizeof(port_str), "%d", port);
    rc = getaddrinfo(host, port_str, &hints, &res);
    if (rc != 0) return -1;
    for (rp = res; rp != NULL; rp = rp->ai_next) {
        int flags;
        struct pollfd pfd;
        int so_error = 0;
        socklen_t len = sizeof(so_error);
        fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (fd < 0) continue;
        flags = fcntl(fd, F_GETFL, 0);
        fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        if (connect(fd, rp->ai_addr, rp->ai_addrlen) == 0) {
            fcntl(fd, F_SETFL, flags);
            break;
        }
        if (errno != EINPROGRESS) { close(fd); fd = -1; continue; }
        pfd.fd = fd; pfd.events = POLLOUT;
        rc = poll(&pfd, 1, timeout_ms);
        if (rc <= 0) {
            if (rc == 0) *timed_out = true;
            close(fd); fd = -1; continue;
        }
        if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_error, &len) != 0 || so_error != 0) {
            close(fd); fd = -1; continue;
        }
        fcntl(fd, F_SETFL, flags);
        break;
    }
    freeaddrinfo(res);
    return fd;
}

/* Wrap an fd in TLS. Returns SSL* or NULL on failure. */
static SSL *net_tls_wrap(int fd, const char *host, bool verify) {
    SSL_CTX *ctx;
    SSL *ssl;
    ctx = SSL_CTX_new(TLS_client_method());
    if (ctx == NULL) return NULL;
    if (verify) {
        SSL_CTX_set_default_verify_paths(ctx);
        SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER, NULL);
    } else {
        SSL_CTX_set_verify(ctx, SSL_VERIFY_NONE, NULL);
    }
    ssl = SSL_new(ctx);
    SSL_CTX_free(ctx);
    if (ssl == NULL) return NULL;
    SSL_set_fd(ssl, fd);
    SSL_set_tlsext_host_name(ssl, host);
    if (SSL_connect(ssl) != 1) {
        SSL_free(ssl);
        return NULL;
    }
    return ssl;
}

/* Read up to `cap` bytes from a socket (TLS-aware) with a timeout. */
static ssize_t net_read(LanaSocket *sock, char *buf, size_t cap, int timeout_ms,
                        bool *timed_out) {
    struct pollfd pfd;
    int rc;
    *timed_out = false;
    pfd.fd = sock->fd; pfd.events = POLLIN;
    rc = poll(&pfd, 1, timeout_ms);
    if (rc == 0) { *timed_out = true; return -1; }
    if (rc < 0) return -1;
    if (sock->ssl != NULL) return (ssize_t)SSL_read((SSL *)sock->ssl, buf, (int)cap);
    return recv(sock->fd, buf, cap, 0);
}

/* Write all bytes to a socket (TLS-aware). Returns bytes written or -1. */
static ssize_t net_write_all(LanaSocket *sock, const char *buf, size_t len) {
    size_t off = 0;
    while (off < len) {
        ssize_t n;
        if (sock->ssl != NULL) n = (ssize_t)SSL_write((SSL *)sock->ssl, buf + off, (int)(len - off));
        else n = send(sock->fd, buf + off, len - off, 0);
        if (n <= 0) return -1;
        off += (size_t)n;
    }
    return (ssize_t)off;
}

/* Perform an HTTP request and build the Result map. */
static LanaError net_http_request(LanaVM *vm, const char *method, const char *url,
                                  const char *body, double timeout_ms, Value *out) {
    char scheme[16], host[256], path[1024];
    char request[16384];
    char response[65536];
    int port, timeout = (int)timeout_ms;
    int fd;
    bool timed_out = false, is_tls;
    SSL *ssl = NULL;
    LanaSocket sock;
    size_t req_len = 0, resp_len = 0;
    ssize_t n;
    LanaError error;
    Value status_value, body_value, headers_map, resp_map;
    LanaMap *header_map, *headers_map_ptr;
    const char *status_line, *header_end, *body_start;
    long http_status = 0;
    if (timeout <= 0) timeout = 5000;
    if (!net_parse_url(url, scheme, sizeof(scheme), host, sizeof(host), &port,
                       path, sizeof(path))) {
        return net_error_result(vm, "url", out);
    }
    is_tls = strcmp(scheme, "https") == 0;
    fd = net_connect(host, port, timeout, &timed_out);
    if (fd < 0) {
        if (timed_out) return net_error_result(vm, "timeout", out);
        return net_error_result(vm, "connect", out);
    }
    if (is_tls) {
        ssl = net_tls_wrap(fd, host, true);
        if (ssl == NULL) { close(fd); return net_error_result(vm, "tls", out); }
    }
    sock.fd = fd; sock.ssl = ssl; sock.is_tls = is_tls;
    req_len = (size_t)snprintf(request, sizeof(request),
        "%s %s HTTP/1.1\r\nHost: %s\r\nConnection: close\r\n"
        "Content-Length: %zu\r\n\r\n%s",
        method, path, host, body == NULL ? 0u : strlen(body),
        body == NULL ? "" : body);
    if (req_len >= sizeof(request)) { net_socket_close(&sock); return net_error_result(vm, "request", out); }
    if (net_write_all(&sock, request, req_len) < 0) {
        net_socket_close(&sock); return net_error_result(vm, "send", out);
    }
    while (resp_len < sizeof(response) - 1u) {
        n = net_read(&sock, response + resp_len, sizeof(response) - 1u - resp_len, timeout, &timed_out);
        if (n < 0) break;
        if (n == 0) break;
        resp_len += (size_t)n;
    }
    net_socket_close(&sock);
    if (timed_out) return net_error_result(vm, "timeout", out);
    if (resp_len == 0) return net_error_result(vm, "response", out);
    response[resp_len] = '\0';
    status_line = response;
    header_end = strstr(response, "\r\n\r\n");
    if (header_end == NULL) return net_error_result(vm, "response", out);
    if (strncmp(status_line, "HTTP/1.", 7) == 0) {
        const char *sp = strchr(status_line, ' ');
        if (sp != NULL) http_status = strtol(sp + 1, NULL, 10);
    }
    body_start = header_end + 4;
    error = lana_map_new(vm, 3u, &header_map);
    if (error != LANA_OK) return error;
    /* The "headers" value is a distinct empty map, not a self-reference. */
    error = lana_map_new(vm, 0u, &headers_map_ptr);
    if (error != LANA_OK) return error;
    headers_map = lana_value_map(headers_map_ptr);
    status_value = lana_value_number((double)http_status);
    body_value = lana_value_string(body_start);
    error = lana_map_set(vm, header_map, "status", &status_value, true);
    if (error != LANA_OK) return error;
    error = lana_map_set(vm, header_map, "headers", &headers_map, true);
    if (error != LANA_OK) return error;
    error = lana_map_set(vm, header_map, "body", &body_value, true);
    if (error != LANA_OK) return error;
    resp_map = lana_value_map(header_map);
    return net_result_map(vm, "ok", &resp_map, out);
}

/* Store a socket in the VM table, returning its handle. */
static LanaError net_socket_store(LanaVM *vm, LanaSocket *sock, Value *out) {
    LanaSocket *new_sockets;
    if (vm->socket_count == vm->socket_capacity) {
        size_t capacity = vm->socket_capacity == 0 ? 8u : vm->socket_capacity * 2u;
        if (capacity < vm->socket_capacity || capacity > SIZE_MAX / sizeof(*new_sockets))
            return LANA_ERR_OOM;
        new_sockets = realloc(vm->sockets, capacity * sizeof(*new_sockets));
        if (new_sockets == NULL) return LANA_ERR_OOM;
        vm->sockets = new_sockets;
        vm->socket_capacity = capacity;
    }
    vm->sockets[vm->socket_count] = *sock;
    *out = lana_value_number((double)vm->socket_count);
    vm->socket_count += 1u;
    return LANA_OK;
}

static LanaSocket *net_socket_get(LanaVM *vm, double handle) {
    size_t index = (size_t)handle;
    if (handle < 0 || index >= vm->socket_count) return NULL;
    return &vm->sockets[index];
}

static LanaError execute_host_call(LanaVM *vm, uint32_t host_id, const Value *arguments,
                                 size_t argc, uint32_t scratch_register, Value *out) {
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
        case LANA_HOST_LAZY_BOUND:
            if (argc != 1u || arguments[0].type != VAL_LAZY) return LANA_ERR_TYPE;
            *out = lana_value_number((double)arguments[0].as.lazy.bound);
            return LANA_OK;
        case LANA_HOST_CORRELATED:
            if (argc != 3u) return LANA_ERR_TYPE;
            return host_correlated(vm, &arguments[0], &arguments[1],
                                   &arguments[2], out);
        case LANA_HOST_SURPRISAL: {
            double probability, result;
            if (argc != 1u || arguments[0].type != VAL_NUMBER) return LANA_ERR_TYPE;
            probability = arguments[0].as.number;
            if (probability < 0.0) return LANA_ERR_INVALID_PARAMETERS;
            result = -log(probability);
            if (result == 0.0) result = 0.0; /* normalize -0.0 to 0.0 */
            *out = lana_value_number(result);
            return LANA_OK;
        }
        case LANA_HOST_NOW: {
            struct timespec now;
            if (argc != 0u || timespec_get(&now, TIME_UTC) != TIME_UTC) return LANA_ERR_TYPE;
            *out = lana_value_number((double)now.tv_sec + (double)now.tv_nsec / 1000000000.0);
            return LANA_OK;
        }
        case LANA_HOST_RANDOM:
            if (argc != 0u) return LANA_ERR_TYPE;
            *out = lana_value_number((double)lana_vm_random(vm) / 4294967296.0); return LANA_OK;
        case LANA_HOST_TENSOR_ALLOC: {
            // Arguments: shape array (VAL_ARRAY), optional is_complex (VAL_BOOL).
            if (argc < 1u || argc > 2u) return LANA_ERR_TYPE;
            bool is_complex = false;
            if (argc == 2u) {
                if (arguments[1].type != VAL_BOOL) return LANA_ERR_TYPE;
                is_complex = arguments[1].as.boolean;
            }
            size_t ndim; size_t *shape;
            LanaError e = tensor_shape_from_array(vm, &arguments[0], &ndim, &shape);
            if (e != LANA_OK) return e;
            LanaTensor *t = tensor_new(vm, ndim, shape, is_complex);
            if (t == NULL) return LANA_ERR_OOM;
            *out = lana_value_tensor(t);
            return LANA_OK;
        }
        case LANA_HOST_TENSOR_ZEROS: {
            if (argc != 1u && argc != 2u) return LANA_ERR_TYPE;
            int dtype = tensor_optional_dtype(arguments, argc);
            if (dtype < 0) return LANA_ERR_INVALID_PARAMETERS;
            size_t ndim; size_t *shape;
            LanaError e = tensor_shape_from_array(vm, &arguments[0], &ndim, &shape);
            if (e != LANA_OK) return e;
            LanaTensor *t = tensor_new_dtype(vm, ndim, shape, (LanaTensorDtype)dtype);
            if (t == NULL) return LANA_ERR_OOM;
            *out = lana_value_tensor(t);
            return LANA_OK;
        }
        case LANA_HOST_TENSOR_ONES: {
            if (argc != 1u && argc != 2u) return LANA_ERR_TYPE;
            int dtype = tensor_optional_dtype(arguments, argc);
            if (dtype < 0) return LANA_ERR_INVALID_PARAMETERS;
            size_t ndim; size_t *shape;
            LanaError e = tensor_shape_from_array(vm, &arguments[0], &ndim, &shape);
            if (e != LANA_OK) return e;
            LanaTensor *t = tensor_new_dtype(vm, ndim, shape, (LanaTensorDtype)dtype);
            if (t == NULL) return LANA_ERR_OOM;
            size_t total = 1;
            for (size_t i = 0; i < ndim; ++i) total *= t->shape[i];
            for (size_t i = 0; i < total; ++i) tensor_set_real(t, i, 1.0);
            *out = lana_value_tensor(t);
            return LANA_OK;
        }
        case LANA_HOST_TENSOR_EYE: {
            if (argc != 1u && argc != 2u) return LANA_ERR_TYPE;
            if (arguments[0].type != VAL_NUMBER) return LANA_ERR_TYPE;
            int dtype = tensor_optional_dtype(arguments, argc);
            if (dtype < 0) return LANA_ERR_INVALID_PARAMETERS;
            size_t n;
            LanaError error = tensor_dimension(arguments[0].as.number, &n);
            if (error != LANA_OK) return error;
            size_t shape[2] = {n, n};
            LanaTensor *t = tensor_new_dtype(vm, 2, shape, (LanaTensorDtype)dtype);
            if (t == NULL) return LANA_ERR_OOM;
            for (size_t i = 0; i < n; ++i) tensor_set_real(t, i * n + i, 1.0);
            *out = lana_value_tensor(t);
            return LANA_OK;
        }
        case LANA_HOST_TENSOR_DTYPE: {
            if (argc != 1u) return LANA_ERR_TYPE;
            if (arguments[0].type != VAL_TENSOR) return LANA_ERR_TYPE;
            *out = lana_value_string(dtype_to_string(arguments[0].as.tensor->dtype));
            return LANA_OK;
        }
        case LANA_HOST_TENSOR_CAST: {
            if (argc != 2u) return LANA_ERR_TYPE;
            if (arguments[0].type != VAL_TENSOR || arguments[1].type != VAL_STRING)
                return LANA_ERR_TYPE;
            int dtype = dtype_from_string(arguments[1].as.string);
            if (dtype < 0) return LANA_ERR_INVALID_PARAMETERS;
            return tensor_cast(vm, arguments[0].as.tensor, (LanaTensorDtype)dtype, out);
        }
        case LANA_HOST_TENSOR_SHAPE: {
            if (argc != 1u) return LANA_ERR_TYPE;
            if (arguments[0].type != VAL_TENSOR) return LANA_ERR_TYPE;
            const LanaTensor *t = arguments[0].as.tensor;
            LanaArray *arr = lana_vm_alloc(vm, sizeof(*arr));
            if (arr == NULL) return LANA_ERR_OOM;
            arr->count = t->ndim;
            arr->capacity = t->ndim;
            arr->items = lana_vm_alloc(vm, t->ndim * sizeof(*arr->items));
            if (arr->items == NULL && t->ndim > 0) return LANA_ERR_OOM;
            for (size_t i = 0; i < t->ndim; ++i) {
                arr->items[i] = lana_value_number((double)t->shape[i]);
            }
            *out = lana_value_array(arr);
            return LANA_OK;
        }
        case LANA_HOST_TENSOR_NDIM: {
            if (argc != 1u) return LANA_ERR_TYPE;
            if (arguments[0].type != VAL_TENSOR) return LANA_ERR_TYPE;
            *out = lana_value_number((double)arguments[0].as.tensor->ndim);
            return LANA_OK;
        }
        case LANA_HOST_TENSOR_ADD:
        case LANA_HOST_TENSOR_SUB:
        case LANA_HOST_TENSOR_MUL:
        case LANA_HOST_TENSOR_DIV: {
            if (argc != 2u) return LANA_ERR_TYPE;
            int op = (int)(host_id - LANA_HOST_TENSOR_ADD);
            const LanaTensor *a_pred = NULL, *a_var = NULL, *b_pred = NULL, *b_var = NULL;
            bool a_unc = false, b_unc = false;
            LanaError unpack_error = tensor_uncertainty_unpack(&arguments[0], &a_pred, &a_var, &a_unc);
            if (unpack_error != LANA_OK) return unpack_error;
            unpack_error = tensor_uncertainty_unpack(&arguments[1], &b_pred, &b_var, &b_unc);
            if (unpack_error != LANA_OK) return unpack_error;
            if (a_unc || b_unc) {
                if (!a_unc) { a_var = tensor_zeros_like(vm, a_pred); if (a_var == NULL) return LANA_ERR_OOM; }
                if (!b_unc) { b_var = tensor_zeros_like(vm, b_pred); if (b_var == NULL) return LANA_ERR_OOM; }
                return tensor_elementwise_uncertain(vm, a_pred, a_var, b_pred, b_var, op, out);
            }
            LanaError error = tensor_elementwise(vm, a_pred, b_pred, op, out);
            if (error != LANA_OK) return error;
            if (vm->ad_recording)
                return ad_record(vm, op, &arguments[0], &arguments[1], -1, out);
            return LANA_OK;
        }
        case LANA_HOST_TENSOR_MATMUL: {
            if (argc != 2u && argc != 3u) return LANA_ERR_TYPE;
            if (argc == 3u && arguments[2].type != VAL_STRING) return LANA_ERR_TYPE;
            const LanaTensor *a_pred = NULL, *a_var = NULL, *b_pred = NULL, *b_var = NULL;
            bool a_unc = false, b_unc = false;
            LanaError unpack_error = tensor_uncertainty_unpack(&arguments[0], &a_pred, &a_var, &a_unc);
            if (unpack_error != LANA_OK) return unpack_error;
            unpack_error = tensor_uncertainty_unpack(&arguments[1], &b_pred, &b_var, &b_unc);
            if (unpack_error != LANA_OK) return unpack_error;
            if (a_unc || b_unc) {
                if (!a_unc) { a_var = tensor_zeros_like(vm, a_pred); if (a_var == NULL) return LANA_ERR_OOM; }
                if (!b_unc) { b_var = tensor_zeros_like(vm, b_pred); if (b_var == NULL) return LANA_ERR_OOM; }
                return tensor_matmul_uncertain(vm, a_pred, a_var, b_pred, b_var, out);
            }
            /* LIP-027: optional out_dtype: named parameter. Defaults to the
             * input dtype when both match, else the higher-precision operand. */
            LanaTensorDtype out_dtype;
            if (argc == 3u) {
                int d = dtype_from_string(arguments[2].as.string);
                if (d < 0) return LANA_ERR_INVALID_PARAMETERS;
                out_dtype = (LanaTensorDtype)d;
            } else {
                out_dtype = matmul_default_dtype(a_pred, b_pred);
            }
            LanaError error = tensor_matmul(vm, a_pred, b_pred, out_dtype, out);
            if (error != LANA_OK) return error;
            if (vm->ad_recording)
                return ad_record(vm, 4, &arguments[0], &arguments[1], -1, out);
            return LANA_OK;
        }
        case LANA_HOST_GPU_MATMUL: {
            if (argc != 3u || arguments[0].type != VAL_TENSOR || arguments[1].type != VAL_TENSOR ||
                arguments[2].type != VAL_STRING)
                return LANA_ERR_TYPE;
            if (strcmp(arguments[2].as.string, "float32") != 0)
                return LANA_ERR_TYPE;
            return tensor_gpu_matmul(vm, arguments[0].as.tensor, arguments[1].as.tensor, out);
        }
        case LANA_HOST_TENSOR_SUM:
        case LANA_HOST_TENSOR_MEAN:
        case LANA_HOST_TENSOR_MAX:
        case LANA_HOST_TENSOR_MIN: {
            if (argc != 1u && argc != 2u) return LANA_ERR_TYPE;
            int op = (int)(host_id - LANA_HOST_TENSOR_SUM);
            const LanaTensor *pred = NULL, *var = NULL;
            bool unc = false;
            LanaError unpack_error = tensor_uncertainty_unpack(&arguments[0], &pred, &var, &unc);
            if (unpack_error != LANA_OK) return unpack_error;
            if (unc) {
                if (op >= 2) return LANA_ERR_TYPE;
                return tensor_reduce_uncertain(vm, pred, var, op,
                                               argc == 2u ? &arguments[1] : NULL, out);
            }
            LanaError error = tensor_reduce(vm, pred, op,
                                            argc == 2u ? &arguments[1] : NULL, out);
            if (error != LANA_OK) return error;
            if (vm->ad_recording && op < 2) {
                int ad_axis = -1;
                if (argc == 2u) {
                    double axis_number = arguments[1].as.number;
                    size_t ndim = pred->ndim;
                    ad_axis = (int)(axis_number < 0 ? axis_number + (double)ndim : axis_number);
                }
                return ad_record(vm, 5 + op, &arguments[0], NULL, ad_axis, out);
            }
            return LANA_OK;
        }
        case LANA_HOST_TENSOR: {
            if (argc != 1u && argc != 2u) return LANA_ERR_TYPE;
            int dtype = tensor_optional_dtype(arguments, argc);
            if (dtype < 0) return LANA_ERR_INVALID_PARAMETERS;
            size_t ndim; size_t *shape;
            LanaError e = tensor_infer_shape(vm, &arguments[0], &ndim, &shape);
            if (e != LANA_OK) return e;
            LanaTensor *t = tensor_new_dtype(vm, ndim, shape, (LanaTensorDtype)dtype);
            if (t == NULL) return LANA_ERR_OOM;
            size_t offset = 0;
            /* tensor_set_real rounds to the target precision at construction. */
            e = tensor_fill_data(&arguments[0], t, &offset);
            if (e != LANA_OK) return e;
            *out = lana_value_tensor(t);
            return LANA_OK;
        }
        case LANA_HOST_TENSOR_COMPLEX: {
            if (argc != 2u) return LANA_ERR_TYPE;
            size_t ndim; size_t *shape;
            LanaError e = tensor_infer_shape(vm, &arguments[0], &ndim, &shape);
            if (e != LANA_OK) return e;
            LanaTensor *t = tensor_new(vm, ndim, shape, true);
            if (t == NULL) return LANA_ERR_OOM;
            size_t offset = 0;
            e = tensor_fill_complex(&arguments[0], &arguments[1], t, &offset);
            if (e != LANA_OK) return e;
            *out = lana_value_tensor(t);
            return LANA_OK;
        }
        case LANA_HOST_DENSITY_OPERATOR: {
            if (argc != 1u) return LANA_ERR_TYPE;
            if (arguments[0].type == VAL_STATE)
                return linalg_density_from_state(vm, &arguments[0].as.state.state, out);
            if (arguments[0].type != VAL_TENSOR) return LANA_ERR_TYPE;
            return linalg_density_from_tensor(vm, arguments[0].as.tensor, out);
        }
        case LANA_HOST_POVM:
            if (argc != 1u) return LANA_ERR_TYPE;
            return linalg_povm(vm, &arguments[0], out);
        case LANA_HOST_CHANNEL:
            if (argc != 1u) return LANA_ERR_TYPE;
            return linalg_channel(vm, &arguments[0], out);
        case LANA_HOST_OBSERVABLE:
            if (argc != 1u) return LANA_ERR_TYPE;
            return linalg_observable(vm, &arguments[0], out);
        case LANA_HOST_TENSOR_PRODUCT: {
            if (argc != 2u || arguments[0].type != VAL_NQUBIT_STATE ||
                arguments[1].type != VAL_NQUBIT_STATE)
                return LANA_ERR_TYPE;
            return linalg_tensor_product(vm, arguments[0].as.tensor, arguments[1].as.tensor, out);
        }
        case LANA_HOST_PARTIAL_TRACE: {
            if (argc != 2u || arguments[0].type != VAL_NQUBIT_STATE ||
                arguments[1].type != VAL_NUMBER)
                return LANA_ERR_TYPE;
            return linalg_partial_trace(vm, arguments[0].as.tensor, arguments[1].as.number, out);
        }
        case LANA_HOST_MEASURE_WITH: {
            if (argc != 2u || arguments[0].type != VAL_NQUBIT_STATE ||
                arguments[1].type != VAL_POVM)
                return LANA_ERR_TYPE;
            return linalg_measure_with(vm, arguments[0].as.tensor, arguments[1].as.tensor, out);
        }
        case LANA_HOST_APPLY_TO: {
            if (argc != 2u || arguments[0].type != VAL_CHANNEL ||
                arguments[1].type != VAL_NQUBIT_STATE)
                return LANA_ERR_TYPE;
            return linalg_apply_to(vm, arguments[0].as.tensor, arguments[1].as.tensor, out);
        }
        case LANA_HOST_EXPECT: {
            if (argc != 2u || arguments[0].type != VAL_NQUBIT_STATE ||
                arguments[1].type != VAL_OBSERVABLE)
                return LANA_ERR_TYPE;
            return linalg_expect(arguments[0].as.tensor, arguments[1].as.tensor, out);
        }
        case LANA_HOST_MIX: {
            if (argc != 3u || arguments[0].type != VAL_NQUBIT_STATE ||
                arguments[1].type != VAL_NQUBIT_STATE || arguments[2].type != VAL_NUMBER)
                return LANA_ERR_TYPE;
            if (!isfinite(arguments[2].as.number) || arguments[2].as.number < 0.0 ||
                arguments[2].as.number > 1.0)
                return LANA_ERR_INVALID_PARAMETERS;
            return linalg_mix(vm, arguments[0].as.tensor, arguments[1].as.tensor,
                              arguments[2].as.number, out);
        }
        case LANA_HOST_TRACE_DISTANCE: {
            if (argc != 2u || arguments[0].type != VAL_NQUBIT_STATE ||
                arguments[1].type != VAL_NQUBIT_STATE)
                return LANA_ERR_TYPE;
            return linalg_trace_distance(vm, arguments[0].as.tensor, arguments[1].as.tensor, out);
        }
        case LANA_HOST_IS_SEPARABLE: {
            if (argc != 2u || arguments[0].type != VAL_NQUBIT_STATE ||
                arguments[1].type != VAL_NUMBER)
                return LANA_ERR_TYPE;
            return linalg_is_separable(vm, arguments[0].as.tensor, arguments[1].as.number, out);
        }
        case LANA_HOST_TO_STATE: {
            if (argc != 1u || arguments[0].type != VAL_NQUBIT_STATE) return LANA_ERR_TYPE;
            return linalg_to_state(arguments[0].as.tensor, out);
        }
        case LANA_HOST_STATE_TENSOR: {
            if (argc != 1u) return LANA_ERR_TYPE;
            return linalg_state_tensor(vm, &arguments[0], out);
        }
        case LANA_HOST_APPEND: {
            if (argc != 2u || arguments[0].type != VAL_TENSOR || arguments[1].type != VAL_TENSOR)
                return LANA_ERR_TYPE;
            if (!arguments[0].as.tensor->is_state || !arguments[1].as.tensor->is_state)
                return LANA_ERR_TYPE;
            LanaError error = linalg_state_append(vm, arguments[0].as.tensor, arguments[1].as.tensor, out);
            if (error != LANA_OK) return error;
            if (vm->ad_recording)
                return ad_record(vm, 7, &arguments[0], &arguments[1], -1, out);
            return LANA_OK;
        }
        case LANA_HOST_MEASURE: {
            if (argc != 2u || arguments[0].type != VAL_TENSOR || arguments[1].type != VAL_POVM)
                return LANA_ERR_TYPE;
            if (!arguments[0].as.tensor->is_state) return LANA_ERR_TYPE;
            LanaError error = linalg_state_measure(vm, arguments[0].as.tensor, arguments[1].as.tensor, out);
            if (error != LANA_OK) return error;
            if (vm->ad_recording)
                return ad_record(vm, 8, &arguments[0], &arguments[1], -1, out);
            return LANA_OK;
        }
        case LANA_HOST_TRANSFORM: {
            if (argc != 2u || arguments[0].type != VAL_TENSOR || arguments[1].type != VAL_CHANNEL)
                return LANA_ERR_TYPE;
            if (!arguments[0].as.tensor->is_state) return LANA_ERR_TYPE;
            LanaError error = linalg_state_transform(vm, arguments[0].as.tensor, arguments[1].as.tensor, out);
            if (error != LANA_OK) return error;
            if (vm->ad_recording)
                return ad_record(vm, 9, &arguments[0], &arguments[1], -1, out);
            return LANA_OK;
        }
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
            if (arguments[0].type == VAL_TENSOR)
                return tensor_index(vm, arguments[0].as.tensor, &arguments[1], out);
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
        case LANA_HOST_JSON_PARSE: {
            Value value; size_t offset = 0u; LanaError error; char message[64]; char *stored;
            if (argc != 1u || arguments[0].type != VAL_STRING) return LANA_ERR_TYPE;
            error = lana_json_parse_offset(vm, arguments[0].as.string, &value, &offset);
            if (error != LANA_OK) {
                (void)snprintf(message, sizeof(message), "invalid JSON at byte %zu", offset);
                stored = lana_vm_alloc(vm, strlen(message) + 1u);
                if (stored == NULL) return LANA_ERR_OOM;
                (void)strcpy(stored, message);
                return make_result(vm, false, lana_value_string(stored), out);
            }
            /* LIP-023 §4: root the parsed value as Information and record the
             * source-text identity (SHA-256) so a decision that consumed the
             * record can be audited and replayed against the exact input. */
            {
                unsigned char digest[LANA_SHA256_DIGEST_SIZE];
                char hex[LANA_SHA256_DIGEST_SIZE * 2u + 1u];
                static const char digits[] = "0123456789abcdef";
                size_t i;
                Value rooted;
                lana_sha256(arguments[0].as.string, strlen(arguments[0].as.string), digest);
                for (i = 0u; i < LANA_SHA256_DIGEST_SIZE; ++i) {
                    hex[i * 2u] = digits[digest[i] >> 4u];
                    hex[i * 2u + 1u] = digits[digest[i] & 15u];
                }
                hex[LANA_SHA256_DIGEST_SIZE * 2u] = '\0';
                error = lana_vm_reactive_root(vm, &value, LANA_EXACTNESS_EXACT, &rooted);
                if (error != LANA_OK) return error;
                error = attach_derivation(vm, &rooted, LANA_DERIVATION_EVIDENCE,
                                          "json_parse", NULL, 0u, hex, 0u,
                                          LANA_EXACTNESS_EXACT, "root");
                if (error != LANA_OK) return error;
                return make_result(vm, true, rooted, out);
            }
        }
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
        case LANA_HOST_GRANT: {
            LanaCapabilityToken *capability;
            uint32_t permission;
            if (argc != 2u || arguments[0].type != VAL_SHARED_CAPABILITY)
                return LANA_ERR_TYPE;
            permission = grant_permission(&arguments[1]);
            if (permission == 0u) return LANA_ERR_TYPE;
            {
                LanaError error = lana_shared_capability_grant(
                    arguments[0].as.capability, permission, &capability);
                if (error != LANA_OK) return error;
            }
            *out = lana_value_shared_capability(capability);
            return LANA_OK;
        }
        case LANA_HOST_REVOKE:
            if (argc != 1u || arguments[0].type != VAL_SHARED_CAPABILITY)
                return LANA_ERR_TYPE;
            return lana_shared_capability_invalidate(arguments[0].as.capability);
        case LANA_HOST_SET_NEW: {
            LanaSet *set;
            LanaError error;
            if (argc != 0u) return LANA_ERR_TYPE;
            error = set_alloc(vm, 0u, &set);
            if (error == LANA_OK) *out = lana_value_set(set);
            return error;
        }
        case LANA_HOST_SET_ADD: {
            LanaSet *set;
            LanaError error;
            size_t count;
            if (argc != 2u || arguments[0].type != VAL_SET) return LANA_ERR_TYPE;
            if (!value_is_set_member(&arguments[1])) return LANA_ERR_TYPE;
            count = arguments[0].as.set->count +
                    (set_contains_value(arguments[0].as.set, &arguments[1]) ? 0u : 1u);
            error = set_alloc(vm, count, &set);
            if (error != LANA_OK) return error;
            for (index = 0u; index < arguments[0].as.set->count; ++index)
                set->items[index] = arguments[0].as.set->items[index];
            if (count > arguments[0].as.set->count)
                set->items[arguments[0].as.set->count] = arguments[1];
            *out = lana_value_set(set);
            return LANA_OK;
        }
        case LANA_HOST_SET_CONTAINS: {
            if (argc != 2u || arguments[0].type != VAL_SET) return LANA_ERR_TYPE;
            if (!value_is_set_member(&arguments[1])) return LANA_ERR_TYPE;
            *out = lana_value_bool(set_contains_value(arguments[0].as.set, &arguments[1]));
            return LANA_OK;
        }
        case LANA_HOST_SET_UNION: {
            LanaSet *set;
            LanaError error;
            size_t count;
            if (argc != 2u || arguments[0].type != VAL_SET || arguments[1].type != VAL_SET)
                return LANA_ERR_TYPE;
            count = arguments[0].as.set->count;
            for (index = 0u; index < arguments[1].as.set->count; ++index)
                if (!set_contains_value(arguments[0].as.set, &arguments[1].as.set->items[index]))
                    ++count;
            error = set_alloc(vm, count, &set);
            if (error != LANA_OK) return error;
            for (index = 0u; index < arguments[0].as.set->count; ++index)
                set->items[index] = arguments[0].as.set->items[index];
            count = arguments[0].as.set->count;
            for (index = 0u; index < arguments[1].as.set->count; ++index) {
                if (!set_contains_value(arguments[0].as.set, &arguments[1].as.set->items[index]))
                    set->items[count++] = arguments[1].as.set->items[index];
            }
            *out = lana_value_set(set);
            return LANA_OK;
        }
        case LANA_HOST_SET_INTERSECT: {
            LanaSet *set;
            LanaError error;
            size_t count = 0u;
            if (argc != 2u || arguments[0].type != VAL_SET || arguments[1].type != VAL_SET)
                return LANA_ERR_TYPE;
            for (index = 0u; index < arguments[0].as.set->count; ++index)
                if (set_contains_value(arguments[1].as.set, &arguments[0].as.set->items[index]))
                    ++count;
            error = set_alloc(vm, count, &set);
            if (error != LANA_OK) return error;
            count = 0u;
            for (index = 0u; index < arguments[0].as.set->count; ++index) {
                if (set_contains_value(arguments[1].as.set, &arguments[0].as.set->items[index]))
                    set->items[count++] = arguments[0].as.set->items[index];
            }
            *out = lana_value_set(set);
            return LANA_OK;
        }
        case LANA_HOST_SET_DIFFERENCE: {
            LanaSet *set;
            LanaError error;
            size_t count = 0u;
            if (argc != 2u || arguments[0].type != VAL_SET || arguments[1].type != VAL_SET)
                return LANA_ERR_TYPE;
            for (index = 0u; index < arguments[0].as.set->count; ++index)
                if (!set_contains_value(arguments[1].as.set, &arguments[0].as.set->items[index]))
                    ++count;
            error = set_alloc(vm, count, &set);
            if (error != LANA_OK) return error;
            count = 0u;
            for (index = 0u; index < arguments[0].as.set->count; ++index) {
                if (!set_contains_value(arguments[1].as.set, &arguments[0].as.set->items[index]))
                    set->items[count++] = arguments[0].as.set->items[index];
            }
            *out = lana_value_set(set);
            return LANA_OK;
        }
        case LANA_HOST_GETENV: {
            const char *value;
            char *copy;
            if (argc != 1u || arguments[0].type != VAL_STRING) return LANA_ERR_TYPE;
            value = getenv(arguments[0].as.string);
            if (value == NULL) value = "";
            copy = lana_vm_alloc(vm, strlen(value) + 1u);
            if (copy == NULL) return LANA_ERR_OOM;
            (void)strcpy(copy, value);
            *out = lana_value_string(copy);
            return LANA_OK;
        }
        case LANA_HOST_RANDOM_SEED: {
            if (argc != 1u || arguments[0].type != VAL_NUMBER) return LANA_ERR_TYPE;
            lana_vm_seed(vm, (uint64_t)arguments[0].as.number);
            *out = lana_value_null();
            return LANA_OK;
        }
        case LANA_HOST_FLOOR: {
            if (argc != 1u || arguments[0].type != VAL_NUMBER) return LANA_ERR_TYPE;
            *out = lana_value_number(floor(arguments[0].as.number));
            return LANA_OK;
        }
        case LANA_HOST_STRING_TO_NUMBER: {
            char *end;
            double number;
            if (argc != 1u || arguments[0].type != VAL_STRING) return LANA_ERR_TYPE;
            number = strtod(arguments[0].as.string, &end);
            if (end == arguments[0].as.string || *end != '\0') {
                char *message = lana_vm_alloc(vm, 15u);
                if (message == NULL) return LANA_ERR_OOM;
                (void)strcpy(message, "invalid number");
                return make_result(vm, false, lana_value_string(message), out);
            }
            return make_result(vm, true, lana_value_number(number), out);
        }
        case LANA_HOST_TYPE_OF: {
            const char *name;
            char *copy;
            if (argc != 1u) return LANA_ERR_TYPE;
            switch (arguments[0].type) {
                case VAL_NULL: name = "null"; break;
                case VAL_NUMBER: name = "number"; break;
                case VAL_BOOL: name = "bool"; break;
                case VAL_STRING: name = "string"; break;
                case VAL_STATE: name = "state"; break;
                case VAL_DISTRIBUTION: name = "distribution"; break;
                case VAL_SAMPLE: name = "sample"; break;
                case VAL_JOINT_STATE: name = "joint_state"; break;
                case VAL_ARRAY: name = "array"; break;
                case VAL_FUNCTION: name = "function"; break;
                case VAL_TASK: name = "task"; break;
                case VAL_STATE_DIST: name = "state_dist"; break;
                case VAL_MAP: name = "map"; break;
                case VAL_POSSIBILITY: name = "possibility"; break;
                case VAL_PATH_SET: name = "path_set"; break;
                case VAL_SHARED_CAPABILITY: name = "shared_capability"; break;
                case VAL_ADT: name = "adt"; break;
                case VAL_TENSOR: name = "tensor"; break;
                case VAL_NQUBIT_STATE: name = "nqubit_state"; break;
                case VAL_POVM: name = "povm"; break;
                case VAL_CHANNEL: name = "channel"; break;
                case VAL_OBSERVABLE: name = "observable"; break;
                case VAL_LAZY: name = "lazy"; break;
                case VAL_GENERATOR: name = "generator"; break;
                case VAL_FUTURE: name = "future"; break;
                case VAL_SET: name = "set"; break;
                case VAL_DATASET: name = "dataset"; break;
                default: name = "unknown"; break;
            }
            copy = lana_vm_alloc(vm, strlen(name) + 1u);
            if (copy == NULL) return LANA_ERR_OOM;
            (void)strcpy(copy, name);
            *out = lana_value_string(copy);
            return LANA_OK;
        }
        case LANA_HOST_FORMAT: {
            const char *format;
            size_t format_length, arg_index = 1u, i;
            FormatBuffer buffer = {0};
            LanaError error;
            if (argc < 1u || arguments[0].type != VAL_STRING) return LANA_ERR_TYPE;
            format = arguments[0].as.string;
            format_length = strlen(format);
            for (i = 0u; i < format_length; ++i) {
                if (format[i] == '{' && i + 1u < format_length && format[i + 1u] == '}') {
                    if (arg_index >= argc) return LANA_ERR_FORMAT;
                    error = format_value(vm, &buffer, &arguments[arg_index]);
                    if (error != LANA_OK) return error;
                    arg_index += 1u;
                    i += 1u;
                } else {
                    error = format_add(vm, &buffer, &format[i], 1u);
                    if (error != LANA_OK) return error;
                }
            }
            if (arg_index != argc) return LANA_ERR_FORMAT;
            error = format_reserve(vm, &buffer, 1u);
            if (error != LANA_OK) return error;
            buffer.data[buffer.length] = '\0';
            *out = lana_value_string(buffer.data);
            return LANA_OK;
        }
        case LANA_HOST_FORMAT_NUMBER: {
            char *copy;
            int written;
            if (argc == 1u) {
                if (arguments[0].type != VAL_NUMBER) return LANA_ERR_TYPE;
                written = snprintf(NULL, 0, "%.17g", arguments[0].as.number);
            } else if (argc == 2u) {
                int precision;
                if (arguments[0].type != VAL_NUMBER || arguments[1].type != VAL_NUMBER ||
                    arguments[1].as.number < 0.0 ||
                    floor(arguments[1].as.number) != arguments[1].as.number ||
                    arguments[1].as.number > 1000.0) return LANA_ERR_TYPE;
                precision = (int)arguments[1].as.number;
                written = snprintf(NULL, 0, "%.*f", precision, arguments[0].as.number);
            } else {
                return LANA_ERR_TYPE;
            }
            if (written < 0) return LANA_ERR_FORMAT;
            copy = lana_vm_alloc(vm, (size_t)written + 1u);
            if (copy == NULL) return LANA_ERR_OOM;
            if (argc == 1u) {
                (void)snprintf(copy, (size_t)written + 1u, "%.17g", arguments[0].as.number);
            } else {
                (void)snprintf(copy, (size_t)written + 1u, "%.*f",
                               (int)arguments[1].as.number, arguments[0].as.number);
            }
            *out = lana_value_string(copy);
            return LANA_OK;
        }
        case LANA_HOST_CHAR_LENGTH: {
            const unsigned char *s; size_t len, i = 0u, count = 0u;
            if (argc != 1u || arguments[0].type != VAL_STRING) return LANA_ERR_TYPE;
            s = (const unsigned char *)arguments[0].as.string;
            len = strlen(arguments[0].as.string);
            while (i < len) {
                uint32_t cp; size_t consumed;
                if (utf8_decode(s + i, len - i, &cp, &consumed) != LANA_OK)
                    return LANA_ERR_SCHEMA;
                i += consumed; count += 1u;
            }
            *out = lana_value_number((double)count); return LANA_OK;
        }
        case LANA_HOST_STRING_CODEPOINT_SLICE: {
            const unsigned char *s; size_t len, i, cp_index;
            size_t start, end, start_byte, end_byte, cp_count = 0u;
            char *copy;
            if (argc != 3u || arguments[0].type != VAL_STRING ||
                arguments[1].type != VAL_NUMBER || arguments[2].type != VAL_NUMBER ||
                arguments[1].as.number < 0.0 || arguments[2].as.number < 0.0 ||
                floor(arguments[1].as.number) != arguments[1].as.number ||
                floor(arguments[2].as.number) != arguments[2].as.number)
                return LANA_ERR_TYPE;
            start = (size_t)arguments[1].as.number;
            end = (size_t)arguments[2].as.number;
            if (start > end) return LANA_ERR_LIMIT;
            s = (const unsigned char *)arguments[0].as.string;
            len = strlen(arguments[0].as.string);
            for (i = 0u; i < len;) {
                uint32_t cp; size_t consumed;
                if (utf8_decode(s + i, len - i, &cp, &consumed) != LANA_OK)
                    return LANA_ERR_SCHEMA;
                i += consumed; cp_count += 1u;
            }
            if (end > cp_count) return LANA_ERR_LIMIT;
            start_byte = len; end_byte = len;
            for (i = 0u, cp_index = 0u; i < len;) {
                uint32_t cp; size_t consumed;
                (void)utf8_decode(s + i, len - i, &cp, &consumed);
                if (cp_index == start) start_byte = i;
                if (cp_index == end) end_byte = i;
                i += consumed; cp_index += 1u;
            }
            copy = lana_vm_alloc(vm, end_byte - start_byte + 1u);
            if (copy == NULL) return LANA_ERR_OOM;
            memcpy(copy, s + start_byte, end_byte - start_byte);
            copy[end_byte - start_byte] = '\0';
            *out = lana_value_string(copy); return LANA_OK;
        }
        case LANA_HOST_TO_UPPER:
        case LANA_HOST_TO_LOWER: {
            const unsigned char *s; size_t len, i, out_len = 0u;
            char *result;
            bool upper = (host_id == LANA_HOST_TO_UPPER);
            if (argc != 1u || arguments[0].type != VAL_STRING) return LANA_ERR_TYPE;
            s = (const unsigned char *)arguments[0].as.string;
            len = strlen(arguments[0].as.string);
            for (i = 0u; i < len;) {
                uint32_t cp, mapped; size_t consumed;
                if (utf8_decode(s + i, len - i, &cp, &consumed) != LANA_OK)
                    return LANA_ERR_SCHEMA;
                mapped = upper ? lana_unicode_upper(cp) : lana_unicode_lower(cp);
                out_len += utf8_encode_len(mapped);
                i += consumed;
            }
            result = lana_vm_alloc(vm, out_len + 1u);
            if (result == NULL) return LANA_ERR_OOM;
            out_len = 0u;
            for (i = 0u; i < len;) {
                uint32_t cp, mapped; size_t consumed;
                (void)utf8_decode(s + i, len - i, &cp, &consumed);
                mapped = upper ? lana_unicode_upper(cp) : lana_unicode_lower(cp);
                out_len += utf8_encode(mapped, (unsigned char *)result + out_len);
                i += consumed;
            }
            result[out_len] = '\0';
            *out = lana_value_string(result); return LANA_OK;
        }
        case LANA_HOST_REGEX_COMPILE: {
            LanaRegex *re = NULL;
            const char *error_msg = NULL;
            LanaError error;
            if (argc != 1u || arguments[0].type != VAL_STRING) return LANA_ERR_TYPE;
            error = regex_compile(vm, arguments[0].as.string, &re, &error_msg);
            if (error == LANA_ERR_OOM) return LANA_ERR_OOM;
            if (error != LANA_OK) {
                char *msg = host_string_copy(vm, error_msg);
                if (msg == NULL) return LANA_ERR_OOM;
                return make_result(vm, false, lana_value_string(msg), out);
            }
            return make_result(vm, true, lana_value_regex(re), out);
        }
        case LANA_HOST_REGEX_MATCH:
        case LANA_HOST_REGEX_SEARCH: {
            const LanaRegex *re;
            const char *text;
            size_t len, start, end;
            LanaError error;
            if (argc != 2u || arguments[0].type != VAL_REGEX ||
                arguments[1].type != VAL_STRING) return LANA_ERR_TYPE;
            re = arguments[0].as.regex;
            text = arguments[1].as.string;
            len = strlen(text);
            if (host_id == LANA_HOST_REGEX_MATCH) {
                error = regex_match_from(vm, re, text, len, 0u, &end);
                if (error == LANA_ERR_OOM) return LANA_ERR_OOM;
                if (error != LANA_OK || end != len) {
                    char *msg = host_string_copy(vm, "no match");
                    if (msg == NULL) return LANA_ERR_OOM;
                    return make_result(vm, false, lana_value_string(msg), out);
                }
                {
                    Value match_value;
                    error = regex_make_match(vm, text, 0u, len, &match_value);
                    if (error != LANA_OK) return error;
                    return make_result(vm, true, match_value, out);
                }
            }
            error = regex_search(vm, re, text, len, 0u, &start, &end);
            if (error == LANA_ERR_OOM) return LANA_ERR_OOM;
            if (error != LANA_OK) {
                char *msg = host_string_copy(vm, "no match");
                if (msg == NULL) return LANA_ERR_OOM;
                return make_result(vm, false, lana_value_string(msg), out);
            }
            {
                Value match_value;
                error = regex_make_match(vm, text, start, end, &match_value);
                if (error != LANA_OK) return error;
                return make_result(vm, true, match_value, out);
            }
        }
        case LANA_HOST_REGEX_REPLACE: {
            const LanaRegex *re;
            const char *text, *replacement;
            size_t len, repl_len, pos = 0u;
            FormatBuffer buffer = {0};
            LanaError error;
            if (argc != 3u || arguments[0].type != VAL_REGEX ||
                arguments[1].type != VAL_STRING || arguments[2].type != VAL_STRING)
                return LANA_ERR_TYPE;
            re = arguments[0].as.regex;
            text = arguments[1].as.string;
            replacement = arguments[2].as.string;
            len = strlen(text);
            repl_len = strlen(replacement);
            while (pos <= len) {
                size_t start, end;
                error = regex_search(vm, re, text, len, pos, &start, &end);
                if (error == LANA_ERR_OOM) return LANA_ERR_OOM;
                if (error != LANA_OK) {
                    error = format_add(vm, &buffer, text + pos, len - pos);
                    if (error != LANA_OK) return error;
                    break;
                }
                error = format_add(vm, &buffer, text + pos, start - pos);
                if (error != LANA_OK) return error;
                error = format_add(vm, &buffer, replacement, repl_len);
                if (error != LANA_OK) return error;
                pos = end;
                if (start == end) {
                    /* Empty match: advance one byte to avoid an infinite loop. */
                    if (pos < len) {
                        error = format_add(vm, &buffer, text + pos, 1u);
                        if (error != LANA_OK) return error;
                        pos += 1u;
                    } else {
                        break;
                    }
                }
            }
            error = format_reserve(vm, &buffer, 1u);
            if (error != LANA_OK) return error;
            buffer.data[buffer.length] = '\0';
            *out = lana_value_string(buffer.data);
            return LANA_OK;
        }
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
        case LANA_HOST_SGD:
            return host_sgd(vm, arguments, argc, out);
        case LANA_HOST_ADAM:
            return host_adam(vm, arguments, argc, out);
        case LANA_HOST_MCMC:
            return host_mcmc(vm, arguments, argc, out);
        case LANA_HOST_VI:
            return host_vi(vm, arguments, argc, out);
        case LANA_HOST_SMC:
            return host_smc(vm, arguments, argc, out);
        case LANA_HOST_RUN_ASYNC: {
            /* Run the event loop to completion on a future and return its
             * result. Nested invocations run a nested loop. */
            LanaFuture *target;
            if (argc != 1u || arguments[0].type != VAL_FUTURE) return LANA_ERR_TYPE;
            target = arguments[0].as.future;
            if (target->exhausted) { *out = target->registers[0]; return LANA_OK; }
            {
                LanaError error = enqueue_future(vm, target);
                if (error != LANA_OK) return error;
                return run_event_loop(vm, target, out);
            }
        }
        case LANA_HOST_FUTURE_ALL: {
            /* A future that completes when all input futures complete,
             * yielding an array of their results in input order. */
            LanaFuture *future;
            size_t index;
            if (argc != 1u || arguments[0].type != VAL_ARRAY) return LANA_ERR_TYPE;
            future = lana_vm_alloc(vm, sizeof(*future));
            if (future == NULL) return LANA_ERR_OOM;
            future->function = UINT32_MAX;
            future->ip = 0u;
            future->register_count = 1u;
            future->exhausted = false;
            future->ready = false;
            future->is_composite = true;
            future->composite_kind = LANA_FUTURE_ALL;
            future->input_count = arguments[0].as.array->count;
            future->inputs = lana_vm_alloc(vm, future->input_count * sizeof(*future->inputs));
            if (future->inputs == NULL) return LANA_ERR_OOM;
            for (index = 0u; index < future->input_count; ++index) {
                if (arguments[0].as.array->items[index].type != VAL_FUTURE) return LANA_ERR_TYPE;
                future->inputs[index] = arguments[0].as.array->items[index].as.future;
            }
            future->wake_time = 0.0;
            future->registers = lana_vm_alloc(vm, sizeof(Value));
            if (future->registers == NULL) return LANA_ERR_OOM;
            future->registers[0] = lana_value_null();
            *out = lana_value_future(future);
            return LANA_OK;
        }
        case LANA_HOST_FUTURE_RACE: {
            /* A future that completes with the first input future to complete,
             * yielding that future's result. */
            LanaFuture *future;
            size_t index;
            if (argc != 1u || arguments[0].type != VAL_ARRAY) return LANA_ERR_TYPE;
            future = lana_vm_alloc(vm, sizeof(*future));
            if (future == NULL) return LANA_ERR_OOM;
            future->function = UINT32_MAX;
            future->ip = 0u;
            future->register_count = 1u;
            future->exhausted = false;
            future->ready = false;
            future->is_composite = true;
            future->composite_kind = LANA_FUTURE_RACE;
            future->input_count = arguments[0].as.array->count;
            future->inputs = lana_vm_alloc(vm, future->input_count * sizeof(*future->inputs));
            if (future->inputs == NULL) return LANA_ERR_OOM;
            for (index = 0u; index < future->input_count; ++index) {
                if (arguments[0].as.array->items[index].type != VAL_FUTURE) return LANA_ERR_TYPE;
                future->inputs[index] = arguments[0].as.array->items[index].as.future;
            }
            future->wake_time = 0.0;
            future->registers = lana_vm_alloc(vm, sizeof(Value));
            if (future->registers == NULL) return LANA_ERR_OOM;
            future->registers[0] = lana_value_null();
            *out = lana_value_future(future);
            return LANA_OK;
        }
        case LANA_HOST_SLEEP: {
            /* A future that completes after `ms` milliseconds, yielding null. */
            LanaFuture *future;
            struct timespec now;
            double ms;
            if (argc != 1u || arguments[0].type != VAL_NUMBER) return LANA_ERR_TYPE;
            ms = arguments[0].as.number;
            if (ms < 0.0) return LANA_ERR_INVALID_PARAMETERS;
            if (timespec_get(&now, TIME_UTC) != TIME_UTC) return LANA_ERR_TYPE;
            future = lana_vm_alloc(vm, sizeof(*future));
            if (future == NULL) return LANA_ERR_OOM;
            future->function = UINT32_MAX;
            future->ip = 0u;
            future->register_count = 1u;
            future->exhausted = false;
            future->ready = false;
            future->is_composite = true;
            future->composite_kind = LANA_FUTURE_SLEEP;
            future->inputs = NULL;
            future->input_count = 0u;
            future->wake_time = (double)now.tv_sec + (double)now.tv_nsec / 1000000000.0
                                + ms / 1000.0;
            future->registers = lana_vm_alloc(vm, sizeof(Value));
            if (future->registers == NULL) return LANA_ERR_OOM;
            future->registers[0] = lana_value_null();
            *out = lana_value_future(future);
            return LANA_OK;
        }
        case LANA_HOST_DATASET: {
            LanaDataset *ds;
            /* LIP-015 §3: accept either a lazy generator (the lazy path) or an
             * in-memory array of rows (the persistence / adapter load path). */
            if (argc != 1u || (arguments[0].type != VAL_LAZY &&
                               arguments[0].type != VAL_ARRAY)) return LANA_ERR_TYPE;
            if (dataset_new(vm, LANA_DATASET_SOURCE, arguments[0], 0u,
                            lana_value_null(), lana_value_null(), lana_value_null(),
                            lana_value_null(), lana_value_null(), &ds) != LANA_OK)
                return LANA_ERR_OOM;
            *out = lana_value_dataset(ds);
            return LANA_OK;
        }
        case LANA_HOST_DATASET_FILTER: {
            LanaDataset *ds;
            if (argc != 2u || arguments[0].type != VAL_DATASET ||
                arguments[1].type != VAL_FUNCTION) return LANA_ERR_TYPE;
            if (dataset_new(vm, LANA_DATASET_FILTER, arguments[0],
                            arguments[1].as.function, lana_value_null(),
                            lana_value_null(), lana_value_null(), lana_value_null(),
                            lana_value_null(), &ds) != LANA_OK)
                return LANA_ERR_OOM;
            *out = lana_value_dataset(ds);
            return LANA_OK;
        }
        case LANA_HOST_DATASET_MAP: {
            LanaDataset *ds;
            if (argc != 2u || arguments[0].type != VAL_DATASET ||
                arguments[1].type != VAL_FUNCTION) return LANA_ERR_TYPE;
            if (dataset_new(vm, LANA_DATASET_MAP, arguments[0],
                            arguments[1].as.function, lana_value_null(),
                            lana_value_null(), lana_value_null(), lana_value_null(),
                            lana_value_null(), &ds) != LANA_OK)
                return LANA_ERR_OOM;
            *out = lana_value_dataset(ds);
            return LANA_OK;
        }
        case LANA_HOST_DATASET_SELECT: {
            LanaDataset *ds;
            if (argc != 2u || arguments[0].type != VAL_DATASET ||
                arguments[1].type != VAL_ARRAY) return LANA_ERR_TYPE;
            if (dataset_new(vm, LANA_DATASET_SELECT, arguments[0], 0u,
                            arguments[1], lana_value_null(), lana_value_null(),
                            lana_value_null(), lana_value_null(), &ds) != LANA_OK)
                return LANA_ERR_OOM;
            *out = lana_value_dataset(ds);
            return LANA_OK;
        }
        case LANA_HOST_DATASET_LIMIT: {
            LanaDataset *ds;
            if (argc != 2u || arguments[0].type != VAL_DATASET ||
                arguments[1].type != VAL_NUMBER || arguments[1].as.number < 0.0)
                return LANA_ERR_TYPE;
            if (dataset_new(vm, LANA_DATASET_LIMIT, arguments[0], 0u,
                            lana_value_null(), lana_value_null(), arguments[1],
                            lana_value_null(), lana_value_null(), &ds) != LANA_OK)
                return LANA_ERR_OOM;
            *out = lana_value_dataset(ds);
            return LANA_OK;
        }
        case LANA_HOST_DATASET_SORT: {
            LanaDataset *ds;
            if (argc != 2u || arguments[0].type != VAL_DATASET ||
                arguments[1].type != VAL_STRING) return LANA_ERR_TYPE;
            if (dataset_new(vm, LANA_DATASET_SORT, arguments[0], 0u,
                            lana_value_null(), arguments[1], lana_value_null(),
                            lana_value_null(), lana_value_null(), &ds) != LANA_OK)
                return LANA_ERR_OOM;
            *out = lana_value_dataset(ds);
            return LANA_OK;
        }
        case LANA_HOST_DATASET_GROUP_BY: {
            LanaDataset *ds;
            if (argc != 2u || arguments[0].type != VAL_DATASET ||
                arguments[1].type != VAL_STRING) return LANA_ERR_TYPE;
            if (dataset_new(vm, LANA_DATASET_GROUP_BY, arguments[0], 0u,
                            lana_value_null(), arguments[1], lana_value_null(),
                            lana_value_null(), lana_value_null(), &ds) != LANA_OK)
                return LANA_ERR_OOM;
            *out = lana_value_dataset(ds);
            return LANA_OK;
        }
        case LANA_HOST_DATASET_AGGREGATE: {
            LanaDataset *ds;
            if (argc != 2u || arguments[0].type != VAL_DATASET ||
                arguments[1].type != VAL_ARRAY) return LANA_ERR_TYPE;
            if (dataset_new(vm, LANA_DATASET_AGGREGATE, arguments[0], 0u,
                            lana_value_null(), lana_value_null(), lana_value_null(),
                            lana_value_null(), arguments[1], &ds) != LANA_OK)
                return LANA_ERR_OOM;
            *out = lana_value_dataset(ds);
            return LANA_OK;
        }
        case LANA_HOST_DATASET_JOIN: {
            LanaDataset *ds;
            if (argc != 3u || arguments[0].type != VAL_DATASET ||
                arguments[1].type != VAL_DATASET || arguments[2].type != VAL_STRING)
                return LANA_ERR_TYPE;
            if (dataset_new(vm, LANA_DATASET_JOIN, arguments[0], 0u,
                            lana_value_null(), arguments[2], lana_value_null(),
                            arguments[1], lana_value_null(), &ds) != LANA_OK)
                return LANA_ERR_OOM;
            *out = lana_value_dataset(ds);
            return LANA_OK;
        }
        case LANA_HOST_DATASET_MATERIALIZE: {
            LanaArray *rows;
            LanaError err;
            if (argc != 1u || arguments[0].type != VAL_DATASET) return LANA_ERR_TYPE;
            err = dataset_materialize(vm, arguments[0].as.dataset, scratch_register,
                                      &rows);
            if (err != LANA_OK) return err;
            *out = lana_value_array(rows);
            return LANA_OK;
        }
        case LANA_HOST_DATASET_EXPLAIN: {
            Value plan;
            if (argc != 1u || arguments[0].type != VAL_DATASET) return LANA_ERR_TYPE;
            if (dataset_explain(vm, arguments[0].as.dataset, &plan) != LANA_OK)
                return LANA_ERR_TYPE;
            *out = plan;
            return LANA_OK;
        }
        case LANA_HOST_STORE_OPEN: {
            LanaStoreOptions options;
            LanaStore *store;
            if (argc != 1u || arguments[0].type != VAL_STRING) return LANA_ERR_TYPE;
            if (vm->store != NULL) return LANA_ERR_CONFLICT;
            memset(&options, 0, sizeof(options));
            options.struct_size = sizeof(options);
            options.schema_version = 1u;
            options.path = arguments[0].as.string;
            options.timeout_ms = 0u;
            if (lana_store_open(&options, &store) != LANA_OK) return LANA_ERR_IO;
            vm->store = store;
            *out = lana_value_null();
            return LANA_OK;
        }
        case LANA_HOST_STORE_PUT: {
            if (argc != 2u || arguments[0].type != VAL_STRING) return LANA_ERR_TYPE;
            if (vm->store == NULL) return LANA_ERR_INVALID_STATE;
            return lana_store_put(vm->store, arguments[0].as.string, arguments[1]);
        }
        case LANA_HOST_STORE_GET: {
            Value value;
            LanaError error;
            if (argc != 1u || arguments[0].type != VAL_STRING) return LANA_ERR_TYPE;
            if (vm->store == NULL) return LANA_ERR_INVALID_STATE;
            error = lana_store_get(vm->store, vm, arguments[0].as.string, &value);
            if (error != LANA_OK) return error;
            *out = value;
            return LANA_OK;
        }
        case LANA_HOST_STORE_DELETE: {
            if (argc != 1u || arguments[0].type != VAL_STRING) return LANA_ERR_TYPE;
            if (vm->store == NULL) return LANA_ERR_INVALID_STATE;
            return lana_store_delete(vm->store, arguments[0].as.string);
        }
        case LANA_HOST_STORE_COMMIT: {
            LanaStoreRevisionInfo info;
            if (argc != 0u) return LANA_ERR_TYPE;
            if (vm->store == NULL) return LANA_ERR_INVALID_STATE;
            if (lana_store_commit(vm->store, &info) != LANA_OK) return LANA_ERR_UNSUPPORTED_OPERATION;
            *out = lana_value_number((double)info.revision_id);
            return LANA_OK;
        }
        case LANA_HOST_STORE_SCAN: {
            LanaStoreScanRecord *records = NULL;
            LanaArray *array;
            size_t count = 0u, index;
            LanaError error;
            if (argc != 1u || arguments[0].type != VAL_STRING) return LANA_ERR_TYPE;
            if (vm->store == NULL) return LANA_ERR_INVALID_STATE;
            error = lana_store_scan(vm->store, vm, arguments[0].as.string, &records, &count);
            if (error != LANA_OK) return error;
            if (dataset_array_new(vm, &array) != LANA_OK) {
                lana_store_scan_free(records, count); return LANA_ERR_OOM;
            }
            for (index = 0u; index < count; ++index) {
                LanaMap *map;
                Value map_value, key_value;
                if (lana_map_new(vm, 2u, &map) != LANA_OK) {
                    lana_store_scan_free(records, count); return LANA_ERR_OOM;
                }
                key_value = lana_value_string(records[index].key);
                if (lana_map_set(vm, map, "key", &key_value, true) != LANA_OK ||
                    lana_map_set(vm, map, "value", &records[index].value, true) != LANA_OK) {
                    lana_store_scan_free(records, count); return LANA_ERR_OOM;
                }
                map_value = lana_value_map(map);
                if (dataset_array_push(vm, array, &map_value) != LANA_OK) {
                    lana_store_scan_free(records, count); return LANA_ERR_OOM;
                }
            }
            lana_store_scan_free(records, count);
            *out = lana_value_array(array);
            return LANA_OK;
        }
        case LANA_HOST_STORE_CURRENT_REVISION: {
            LanaStoreRevisionInfo info;
            if (argc != 0u) return LANA_ERR_TYPE;
            if (vm->store == NULL) return LANA_ERR_INVALID_STATE;
            if (lana_store_current_revision(vm->store, &info) != LANA_OK) return LANA_ERR_IO;
            *out = lana_value_number((double)info.revision_id);
            return LANA_OK;
        }
        case LANA_HOST_STORE_GET_AT: {
            Value value;
            LanaError error;
            if (argc != 2u || arguments[0].type != VAL_NUMBER ||
                arguments[1].type != VAL_STRING) return LANA_ERR_TYPE;
            if (vm->store == NULL) return LANA_ERR_INVALID_STATE;
            error = lana_store_get_at(vm->store, vm, (uint64_t)arguments[0].as.number,
                                      arguments[1].as.string, &value);
            if (error != LANA_OK) return error;
            *out = value;
            return LANA_OK;
        }
        case LANA_HOST_STORE_SNAPSHOT: {
            Value value;
            LanaStoreRevisionInfo info;
            if (argc != 0u) return LANA_ERR_TYPE;
            if (vm->store == NULL) return LANA_ERR_INVALID_STATE;
            if (lana_store_snapshot(vm->store, vm, &value, &info) != LANA_OK) return LANA_ERR_IO;
            *out = value;
            return LANA_OK;
        }
        case LANA_HOST_STORE_COMMIT_IF: {
            LanaStoreRevisionInfo current, info;
            if (argc != 1u || arguments[0].type != VAL_NUMBER) return LANA_ERR_TYPE;
            if (vm->store == NULL) return LANA_ERR_INVALID_STATE;
            if (lana_store_current_revision(vm->store, &current) != LANA_OK) return LANA_ERR_IO;
            if (current.revision_id != (uint64_t)arguments[0].as.number) return LANA_ERR_CONFLICT;
            if (lana_store_commit(vm->store, &info) != LANA_OK) return LANA_ERR_UNSUPPORTED_OPERATION;
            *out = lana_value_number((double)info.revision_id);
            return LANA_OK;
        }
        case LANA_HOST_ADAPTER_LOAD: {
            LanaAdapterOptions options;
            void *adapter;
            if (argc != 2u || arguments[0].type != VAL_NUMBER ||
                arguments[1].type != VAL_STRING) return LANA_ERR_TYPE;
            memset(&options, 0, sizeof(options));
            options.struct_size = sizeof(options);
            options.schema_version = 1u;
            options.kind = (LanaAdapterKind)(uint32_t)arguments[0].as.number;
            options.config = arguments[1].as.string;
            if (lana_adapter_load(&options, &adapter) != LANA_OK) return LANA_ERR_IO;
            if (vm->adapter != NULL) lana_adapter_close(vm->adapter);
            vm->adapter = adapter;
            *out = lana_value_null();
            return LANA_OK;
        }
        case LANA_HOST_ADAPTER_FETCH: {
            Value value;
            LanaError error;
            if (argc != 1u || arguments[0].type != VAL_STRING) return LANA_ERR_TYPE;
            if (vm->adapter == NULL) return LANA_ERR_INVALID_STATE;
            error = lana_adapter_fetch(vm->adapter, vm, arguments[0].as.string, &value);
            if (error != LANA_OK) return error;
            *out = value;
            return LANA_OK;
        }
        case LANA_HOST_FFI_DECLARE: {
            FfiSignature sig;
            char *copy;
            char **new_sigs;
            size_t index;
            if (argc != 1u || arguments[0].type != VAL_STRING) return LANA_ERR_TYPE;
            if (!ffi_parse_signature(arguments[0].as.string, &sig)) return LANA_ERR_EXTERNAL;
            copy = strdup(arguments[0].as.string);
            if (copy == NULL) return LANA_ERR_OOM;
            new_sigs = realloc(vm->ffi_sigs, (vm->ffi_sig_count + 1u) * sizeof(*new_sigs));
            if (new_sigs == NULL) { free(copy); return LANA_ERR_OOM; }
            vm->ffi_sigs = new_sigs;
            index = vm->ffi_sig_count++;
            vm->ffi_sigs[index] = copy;
            *out = lana_value_number((double)index);
            return LANA_OK;
        }
        case LANA_HOST_FFI_LOAD: {
            void *lib;
            if (argc != 1u || arguments[0].type != VAL_STRING) return LANA_ERR_TYPE;
            if (!vm_has_named_capability(vm, "ffi")) return LANA_ERR_EXTERNAL;
            lib = dlopen(arguments[0].as.string, RTLD_NOW);
            if (lib == NULL) return LANA_ERR_EXTERNAL;
            if (vm->ffi_lib != NULL) dlclose(vm->ffi_lib);
            vm->ffi_lib = lib;
            *out = lana_value_null();
            return LANA_OK;
        }
        case LANA_HOST_FFI_CALL: {
            FfiSignature sig;
            Value result, error_value;
            LanaError error;
            if (argc != 3u || arguments[0].type != VAL_NUMBER ||
                arguments[1].type != VAL_NUMBER || arguments[2].type != VAL_ARRAY)
                return LANA_ERR_TYPE;
            if (!vm_has_named_capability(vm, "ffi")) return LANA_ERR_EXTERNAL;
            if ((size_t)arguments[1].as.number >= vm->ffi_sig_count) return LANA_ERR_EXTERNAL;
            if (!ffi_parse_signature(vm->ffi_sigs[(size_t)arguments[1].as.number], &sig))
                return LANA_ERR_EXTERNAL;
            /* Validate the argument types before the library check so a bad
             * argument is reported as a Result error even with no library
             * loaded (deterministic across both VMs). */
            if (!ffi_validate_args(&sig, arguments[2].as.array->items,
                                   arguments[2].as.array->count)) {
                error_value = lana_value_string("type");
                return ffi_result_map(vm, "error", &error_value, out);
            }
            if (vm->ffi_lib == NULL) return LANA_ERR_INVALID_STATE;
            error = ffi_call_impl(vm, &sig, vm->ffi_lib, arguments[2].as.array->items,
                                  arguments[2].as.array->count, &result);
            if (error == LANA_ERR_TYPE) {
                error_value = lana_value_string("type");
                return ffi_result_map(vm, "error", &error_value, out);
            }
            if (error == LANA_ERR_EXTERNAL) {
                error_value = lana_value_string("external");
                return ffi_result_map(vm, "error", &error_value, out);
            }
            if (error != LANA_OK) return error;
            return ffi_result_map(vm, "ok", &result, out);
        }
        case LANA_HOST_HTTP_GET: {
            const char *url;
            double timeout_ms;
            if (argc != 3u || arguments[0].type != VAL_STRING ||
                arguments[1].type != VAL_MAP || arguments[2].type != VAL_NUMBER)
                return LANA_ERR_TYPE;
            if (!vm_has_named_capability(vm, "net")) return LANA_ERR_CAPABILITY;
            url = arguments[0].as.string;
            timeout_ms = arguments[2].as.number;
            return net_http_request(vm, "GET", url, NULL, timeout_ms, out);
        }
        case LANA_HOST_HTTP_POST: {
            const char *url, *body;
            double timeout_ms;
            if (argc != 4u || arguments[0].type != VAL_STRING ||
                arguments[1].type != VAL_STRING || arguments[2].type != VAL_MAP ||
                arguments[3].type != VAL_NUMBER)
                return LANA_ERR_TYPE;
            if (!vm_has_named_capability(vm, "net")) return LANA_ERR_CAPABILITY;
            url = arguments[0].as.string;
            body = arguments[1].as.string;
            timeout_ms = arguments[3].as.number;
            return net_http_request(vm, "POST", url, body, timeout_ms, out);
        }
        case LANA_HOST_SOCKET_CONNECT: {
            const char *host;
            int port;
            int fd;
            bool timed_out = false;
            LanaSocket sock;
            if (argc != 2u || arguments[0].type != VAL_STRING ||
                arguments[1].type != VAL_NUMBER)
                return LANA_ERR_TYPE;
            if (!vm_has_named_capability(vm, "net")) return LANA_ERR_CAPABILITY;
            host = arguments[0].as.string;
            port = (int)arguments[1].as.number;
            fd = net_connect(host, port, 5000, &timed_out);
            if (fd < 0) {
                if (timed_out) return net_error_result(vm, "timeout", out);
                return net_error_result(vm, "connect", out);
            }
            sock.fd = fd; sock.ssl = NULL; sock.is_tls = false;
            return net_socket_store(vm, &sock, out);
        }
        case LANA_HOST_SOCKET_SEND: {
            LanaSocket *sock;
            ssize_t n;
            if (argc != 2u || arguments[0].type != VAL_NUMBER ||
                arguments[1].type != VAL_STRING)
                return LANA_ERR_TYPE;
            if (!vm_has_named_capability(vm, "net")) return LANA_ERR_CAPABILITY;
            sock = net_socket_get(vm, arguments[0].as.number);
            if (sock == NULL) return LANA_ERR_INVALID_STATE;
            n = net_write_all(sock, arguments[1].as.string, strlen(arguments[1].as.string));
            if (n < 0) return net_error_result(vm, "send", out);
            *out = lana_value_number((double)n);
            return LANA_OK;
        }
        case LANA_HOST_SOCKET_RECV: {
            LanaSocket *sock;
            char buf[65536];
            ssize_t n;
            bool timed_out = false;
            size_t max_bytes;
            if (argc != 2u || arguments[0].type != VAL_NUMBER ||
                arguments[1].type != VAL_NUMBER)
                return LANA_ERR_TYPE;
            if (!vm_has_named_capability(vm, "net")) return LANA_ERR_CAPABILITY;
            sock = net_socket_get(vm, arguments[0].as.number);
            if (sock == NULL) return LANA_ERR_INVALID_STATE;
            max_bytes = (size_t)arguments[1].as.number;
            if (max_bytes > sizeof(buf)) max_bytes = sizeof(buf);
            n = net_read(sock, buf, max_bytes, 5000, &timed_out);
            if (n < 0) {
                if (timed_out) return net_error_result(vm, "timeout", out);
                return net_error_result(vm, "recv", out);
            }
            buf[n] = '\0';
            *out = lana_value_string(buf);
            return LANA_OK;
        }
        case LANA_HOST_SOCKET_CLOSE: {
            LanaSocket *sock;
            if (argc != 1u || arguments[0].type != VAL_NUMBER) return LANA_ERR_TYPE;
            if (!vm_has_named_capability(vm, "net")) return LANA_ERR_CAPABILITY;
            sock = net_socket_get(vm, arguments[0].as.number);
            if (sock == NULL) return LANA_ERR_INVALID_STATE;
            net_socket_close(sock);
            *out = lana_value_null();
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
    frame->registers[reg] = (Value){.type = VAL_STATE, .as.state = state};
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

static LanaError distribution_from_value(const Value *value, LanaDistOperand *out) {
    if (value == NULL || out == NULL) return LANA_ERR_TYPE;
    if (value->type == VAL_STATE) {
        out->is_inline = true;
        out->as.state = value->as.state;
        return LANA_OK;
    }
    if (value->type == VAL_STATE_DIST && value->as.state_dist != NULL) {
        out->is_inline = false;
        out->as.node = value->as.state_dist;
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
    error = distribution_from_value(left, &distribution->as.append.left);
    if (error == LANA_OK)
        error = distribution_from_value(right, &distribution->as.append.right);
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

LanaError lana_vm_state_dist_attenuate(LanaVM *vm, LanaStateDist *child, double factor,
                                   LanaStateDist **out) {
    LanaStateDist *distribution;
    if (vm == NULL || child == NULL || out == NULL) return LANA_ERR_INVALID_DISTRIBUTION;
    if (!isfinite(factor) || factor < 0.0 || factor > 1.0)
        return LANA_ERR_INVALID_PARAMETERS;
    distribution = lana_vm_alloc(vm, sizeof(*distribution));
    if (distribution == NULL) return LANA_ERR_OOM;
    distribution->kind = LANA_DIST_ATTENUATE;
    distribution->as.attenuate.child = child;
    distribution->as.attenuate.factor = factor;
    *out = distribution;
    return LANA_OK;
}

LanaError lana_vm_state_dist_append_relationship(LanaVM *vm, const Value *left,
                                   const Value *right, uint32_t mode, double strength,
                                   LanaStateDist **out) {
    LanaStateDist *distribution;
    LanaError error;
    if (vm == NULL || out == NULL) return LANA_ERR_INVALID_DISTRIBUTION;
    if (left->type == VAL_STATE_DIST || right->type == VAL_STATE_DIST)
        return LANA_ERR_UNSUPPORTED_OPERATION;
    if (left->type != VAL_STATE || right->type != VAL_STATE) return LANA_ERR_TYPE;
    distribution = lana_vm_alloc(vm, sizeof(*distribution));
    if (distribution == NULL) return LANA_ERR_OOM;
    distribution->kind = LANA_DIST_APPEND;
    error = distribution_from_value(left, &distribution->as.append.left);
    if (error == LANA_OK)
        error = distribution_from_value(right, &distribution->as.append.right);
    if (error != LANA_OK) return error;
    distribution->as.append.has_cached_parameters = false;
    error = lana_state_append_relationship_parameters(&left->as.state.state,
                                                      &right->as.state.state, mode,
                                                      strength,
                                                      &distribution->as.append.p,
                                                      &distribution->as.append.m_re,
                                                      &distribution->as.append.m_im,
                                                      &distribution->as.append.sigma);
    if (error != LANA_OK) return error;
    distribution->as.append.has_cached_parameters = true;
    *out = distribution;
    return LANA_OK;
}

typedef struct {
    const LanaStateDist *node;
    unsigned stage;
    double left;
    double right;
    LanaStateValue left_state;
    LanaStateValue right_state;
} LanaDistEvalFrame;

static LanaDistEvalFrame dist_eval_frame(const LanaStateDist *node) {
    LanaDistEvalFrame frame = {0};
    frame.node = node;
    return frame;
}

static LanaError expected_probability_operand(const LanaDistOperand *operand,
                                              double *out) {
    if (operand == NULL || out == NULL) return LANA_ERR_INVALID_DISTRIBUTION;
    if (!operand->is_inline)
        return LANA_ERR_INVALID_DISTRIBUTION;
    if (!lana_state_valid(&operand->as.state.state))
        return LANA_ERR_INVALID_DISTRIBUTION;
    *out = operand->as.state.state.p;
    return LANA_OK;
}

LanaError lana_vm_state_dist_expected_probability(const LanaStateDist *distribution,
                                              double *out) {
    LanaDistEvalFrame stack[LANA_STATE_DIST_DEPTH_LIMIT + 1u];
    size_t top = 0u;
    double result = 0.0;
    if (distribution == NULL || out == NULL) return LANA_ERR_INVALID_DISTRIBUTION;
    stack[top++] = dist_eval_frame(distribution);
    while (top > 0u) {
        LanaDistEvalFrame *frame = &stack[top - 1u];
        LanaError error;
        if (frame->node == NULL) return LANA_ERR_INVALID_DISTRIBUTION;
        if (frame->stage == 0u) {
            if (frame->node->kind == LANA_DIST_DIRAC) {
                if (!lana_state_valid(&frame->node->as.dirac.state))
                    return LANA_ERR_INVALID_DISTRIBUTION;
                result = frame->node->as.dirac.state.p;
                --top;
            } else if (frame->node->kind == LANA_DIST_APPEND) {
                if (frame->node->as.append.has_cached_parameters) {
                    result = frame->node->as.append.p;
                    --top;
                } else {
                    frame->stage = 1u;
                    if (frame->node->as.append.left.is_inline) {
                        error = expected_probability_operand(&frame->node->as.append.left,
                                                             &frame->left);
                        if (error != LANA_OK) return error;
                        frame->stage = 2u;
                    } else {
                        if (top >= LANA_STATE_DIST_DEPTH_LIMIT + 1u) return LANA_ERR_INVALID_DISTRIBUTION;
                        stack[top++] = dist_eval_frame(frame->node->as.append.left.as.node);
                    }
                }
            } else if (frame->node->kind == LANA_DIST_TRANSFORM) {
                frame->stage = 4u;
                if (top >= LANA_STATE_DIST_DEPTH_LIMIT + 1u) return LANA_ERR_INVALID_DISTRIBUTION;
                stack[top++] = dist_eval_frame(frame->node->as.transform.child);
            } else if (frame->node->kind == LANA_DIST_ATTENUATE) {
                frame->stage = 5u;
                if (top >= LANA_STATE_DIST_DEPTH_LIMIT + 1u) return LANA_ERR_INVALID_DISTRIBUTION;
                stack[top++] = dist_eval_frame(frame->node->as.attenuate.child);
            } else return LANA_ERR_INVALID_DISTRIBUTION;
        } else if (frame->stage == 1u) {
            frame->left = result;
            frame->stage = 2u;
        } else if (frame->stage == 2u) {
            if (frame->node->as.append.right.is_inline) {
                error = expected_probability_operand(&frame->node->as.append.right,
                                                     &frame->right);
                if (error != LANA_OK) return error;
                frame->stage = 3u;
            } else {
                frame->stage = 3u;
                if (top >= LANA_STATE_DIST_DEPTH_LIMIT + 1u) return LANA_ERR_INVALID_DISTRIBUTION;
                stack[top++] = dist_eval_frame(frame->node->as.append.right.as.node);
            }
        } else if (frame->stage == 3u) {
            if (!frame->node->as.append.right.is_inline) frame->right = result;
            result = 1.0 - (1.0 - frame->left) * (1.0 - frame->right);
            if (!isfinite(result) || result < 0.0 || result > 1.0)
                return LANA_ERR_INVALID_DISTRIBUTION;
            --top;
        } else if (frame->stage == 4u) {
            error = lana_transform_expected_probability(
                frame->node->as.transform.transform_id, result, &result);
            if (error != LANA_OK) return error;
            --top;
        } else {
            /* ATTENUATE is the identity on probability: the child's result is
             * already the expected probability. */
            --top;
        }
    }
    *out = result;
    return LANA_OK;
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

static LanaError sample_append_parameters(LanaVM *vm, double p, double m_re,
                                          double m_im, double sigma,
                                          LanaStateValue *out) {
    memset(&out->indexes, 0, sizeof(out->indexes));
    if (p == 0.0 || p == 1.0)
        return lana_state_make_complex(p, 0.0, 0.0, &out->state);
    if (sigma == 0.0)
        return lana_state_make_complex(p, m_re, m_im, &out->state);
    for (;;) {
        double x, y, radius_squared, factor, d_re, d_im;
        LanaError error = consume_sampling_budget(vm);
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

static LanaError sample_append_kernel(LanaVM *vm, const LanaStateValue *left,
                                    const LanaStateValue *right, LanaStateValue *out) {
    double p, m_re, m_im, sigma;
    LanaError error = lana_state_append_parameters(&left->state, &right->state,
                                               &p, &m_re, &m_im, &sigma);
    if (error != LANA_OK) return LANA_ERR_INVALID_DISTRIBUTION;
    return sample_append_parameters(vm, p, m_re, m_im, sigma, out);
}

static LanaError sample_distribution_recursive(LanaVM *vm, const LanaStateDist *distribution,
                                             LanaStateValue *out, size_t depth) {
    LanaDistEvalFrame stack[LANA_STATE_DIST_DEPTH_LIMIT + 1u];
    LanaStateValue result_state;
    size_t top = 0u;
    if (distribution == NULL || out == NULL || depth > LANA_STATE_DIST_DEPTH_LIMIT)
        return LANA_ERR_INVALID_DISTRIBUTION;
    stack[top++] = dist_eval_frame(distribution);
    while (top > 0u) {
        LanaDistEvalFrame *frame = &stack[top - 1u];
        LanaError error;
        if (frame->node == NULL) return LANA_ERR_INVALID_DISTRIBUTION;
        if (frame->stage == 0u) {
            if (frame->node->kind == LANA_DIST_DIRAC) {
                if (!lana_state_valid(&frame->node->as.dirac.state))
                    return LANA_ERR_INVALID_DISTRIBUTION;
                result_state = frame->node->as.dirac;
                --top;
            } else if (frame->node->kind == LANA_DIST_APPEND) {
                if (frame->node->as.append.has_cached_parameters) {
                    error = sample_append_parameters(vm, frame->node->as.append.p,
                                                     frame->node->as.append.m_re,
                                                     frame->node->as.append.m_im,
                                                     frame->node->as.append.sigma,
                                                     &result_state);
                    if (error != LANA_OK) return error;
                    --top;
                } else {
                    frame->stage = 1u;
                    if (frame->node->as.append.left.is_inline) {
                        frame->left_state = frame->node->as.append.left.as.state;
                        if (!lana_state_valid(&frame->left_state.state)) return LANA_ERR_INVALID_DISTRIBUTION;
                        frame->stage = 2u;
                    } else {
                        if (top >= LANA_STATE_DIST_DEPTH_LIMIT + 1u) return LANA_ERR_INVALID_DISTRIBUTION;
                        stack[top++] = dist_eval_frame(frame->node->as.append.left.as.node);
                    }
                }
            } else if (frame->node->kind == LANA_DIST_TRANSFORM) {
                frame->stage = 4u;
                if (top >= LANA_STATE_DIST_DEPTH_LIMIT + 1u) return LANA_ERR_INVALID_DISTRIBUTION;
                stack[top++] = dist_eval_frame(frame->node->as.transform.child);
            } else if (frame->node->kind == LANA_DIST_ATTENUATE) {
                frame->stage = 5u;
                if (top >= LANA_STATE_DIST_DEPTH_LIMIT + 1u) return LANA_ERR_INVALID_DISTRIBUTION;
                stack[top++] = dist_eval_frame(frame->node->as.attenuate.child);
            } else return LANA_ERR_INVALID_DISTRIBUTION;
        } else if (frame->stage == 1u) {
            frame->left_state = result_state;
            frame->stage = 2u;
        } else if (frame->stage == 2u) {
            if (frame->node->as.append.right.is_inline) {
                frame->right_state = frame->node->as.append.right.as.state;
                if (!lana_state_valid(&frame->right_state.state)) return LANA_ERR_INVALID_DISTRIBUTION;
                frame->stage = 3u;
            } else {
                frame->stage = 3u;
                if (top >= LANA_STATE_DIST_DEPTH_LIMIT + 1u) return LANA_ERR_INVALID_DISTRIBUTION;
                stack[top++] = dist_eval_frame(frame->node->as.append.right.as.node);
            }
        } else if (frame->stage == 3u) {
            if (!frame->node->as.append.right.is_inline) frame->right_state = result_state;
            error = sample_append_kernel(vm, &frame->left_state, &frame->right_state,
                                         &result_state);
            if (error != LANA_OK) return error;
            --top;
        } else if (frame->stage == 4u) {
            error = lana_transform_apply(frame->node->as.transform.transform_id,
                                         &result_state.state, &result_state.state);
            if (error != LANA_OK) return error;
            --top;
        } else {
            /* ATTENUATE: scale the disposition of the sampled child state. */
            error = lana_state_attenuate(&result_state.state,
                                         frame->node->as.attenuate.factor,
                                         &result_state.state);
            if (error != LANA_OK) return error;
            --top;
        }
    }
    *out = result_state;
    return LANA_OK;
}

LanaError lana_vm_state_dist_sample(LanaVM *vm, const LanaStateDist *distribution,
                                LanaStateValue *out) {
    if (vm == NULL) return LANA_ERR_INVALID_DISTRIBUTION;
    return sample_distribution_recursive(vm, distribution, out, 0u);
}

static LanaError support_array_push(LanaVM *vm, LanaArray *array, LanaStateValue state) {
    if (array->count == array->capacity) {
        size_t capacity = array->capacity == 0u ? 4u : array->capacity * 2u;
        Value *items = lana_vm_alloc(vm, capacity * sizeof(*items));
        if (items == NULL) return LANA_ERR_OOM;
        if (array->count > 0u) memcpy(items, array->items, array->count * sizeof(*items));
        array->items = items;
        array->capacity = capacity;
    }
    array->items[array->count++] = lana_value_state(state.state);
    return LANA_OK;
}

static int support_compare(const void *left, const void *right) {
    const LanaState *a = &((const Value *)left)->as.state.state;
    const LanaState *b = &((const Value *)right)->as.state.state;
    if (a->p < b->p) return -1;
    if (a->p > b->p) return 1;
    if (a->d_re < b->d_re) return -1;
    if (a->d_re > b->d_re) return 1;
    if (a->d_im < b->d_im) return -1;
    if (a->d_im > b->d_im) return 1;
    return 0;
}

static void support_sort(LanaArray *array) {
    qsort(array->items, array->count, sizeof(*array->items), support_compare);
}

static LanaError lana_vm_state_dist_support(LanaVM *vm, const LanaStateDist *distribution,
                                            uint32_t limit, LanaArray **out);

static LanaError support_collect(LanaVM *vm, const LanaStateDist *node,
                                 LanaArray *array, uint32_t limit, size_t depth) {
    LanaError error;
    if (node == NULL || depth > LANA_STATE_DIST_DEPTH_LIMIT)
        return LANA_ERR_INVALID_DISTRIBUTION;
    if (node->kind == LANA_DIST_DIRAC) {
        if (!lana_state_valid(&node->as.dirac.state))
            return LANA_ERR_INVALID_DISTRIBUTION;
        if (array->count >= limit) return LANA_ERR_LIMIT;
        return support_array_push(vm, array, node->as.dirac);
    }
    if (node->kind == LANA_DIST_TRANSFORM) {
        LanaArray *child_support;
        size_t index;
        error = lana_vm_state_dist_support(vm, node->as.transform.child, limit,
                                           &child_support);
        if (error != LANA_OK) return error;
        for (index = 0u; index < child_support->count; ++index) {
            LanaStateValue transformed;
            if (array->count >= limit) return LANA_ERR_LIMIT;
            transformed.state = child_support->items[index].as.state.state;
            memset(&transformed.indexes, 0, sizeof(transformed.indexes));
            error = lana_transform_apply(node->as.transform.transform_id,
                                         &transformed.state, &transformed.state);
            if (error != LANA_OK) return error;
            error = support_array_push(vm, array, transformed);
            if (error != LANA_OK) return error;
        }
        return LANA_OK;
    }
    if (node->kind == LANA_DIST_ATTENUATE) {
        LanaArray *child_support;
        size_t index;
        error = lana_vm_state_dist_support(vm, node->as.attenuate.child, limit,
                                           &child_support);
        if (error != LANA_OK) return error;
        for (index = 0u; index < child_support->count; ++index) {
            LanaStateValue attenuated;
            if (array->count >= limit) return LANA_ERR_LIMIT;
            attenuated.state = child_support->items[index].as.state.state;
            memset(&attenuated.indexes, 0, sizeof(attenuated.indexes));
            error = lana_state_attenuate(&attenuated.state,
                                         node->as.attenuate.factor,
                                         &attenuated.state);
            if (error != LANA_OK) return error;
            error = support_array_push(vm, array, attenuated);
            if (error != LANA_OK) return error;
        }
        return LANA_OK;
    }
    if (node->kind == LANA_DIST_APPEND) {
        if (node->as.append.has_cached_parameters && node->as.append.sigma == 0.0) {
            LanaStateValue result;
            if (array->count >= limit) return LANA_ERR_LIMIT;
            memset(&result.indexes, 0, sizeof(result.indexes));
            error = lana_state_make_complex(node->as.append.p, node->as.append.m_re,
                                            node->as.append.m_im, &result.state);
            if (error != LANA_OK) return LANA_ERR_INVALID_DISTRIBUTION;
            return support_array_push(vm, array, result);
        }
        return LANA_ERR_UNSUPPORTED_OPERATION;
    }
    return LANA_ERR_INVALID_DISTRIBUTION;
}

LanaError lana_vm_state_dist_support(LanaVM *vm, const LanaStateDist *distribution,
                                    uint32_t limit, LanaArray **out) {
    LanaArray *array;
    LanaError error;
    if (vm == NULL || distribution == NULL || out == NULL || limit == 0u)
        return LANA_ERR_INVALID_DISTRIBUTION;
    array = lana_vm_alloc(vm, sizeof(*array));
    if (array == NULL) return LANA_ERR_OOM;
    array->count = 0u; array->capacity = 0u; array->items = NULL;
    error = support_collect(vm, distribution, array, limit, 0u);
    if (error != LANA_OK) return error;
    support_sort(array);
    *out = array;
    return LANA_OK;
}

/* ---- LIP-002: lazy state-dist inspection ---- */

typedef struct {
    char *data;
    size_t length;
    size_t capacity;
} InspectBuffer;

static bool inspect_buffer_reserve(InspectBuffer *buffer, size_t extra) {
    size_t needed = buffer->length + extra + 1u;
    char *grown;
    if (needed <= buffer->capacity) return true;
    if (needed > SIZE_MAX / 2u) return false;
    buffer->capacity = buffer->capacity == 0u ? 256u : buffer->capacity;
    while (buffer->capacity < needed) buffer->capacity *= 2u;
    grown = realloc(buffer->data, buffer->capacity);
    if (grown == NULL) return false;
    buffer->data = grown;
    return true;
}

static bool inspect_buffer_add(InspectBuffer *buffer, const char *text, size_t length) {
    if (!inspect_buffer_reserve(buffer, length)) return false;
    memcpy(buffer->data + buffer->length, text, length);
    buffer->length += length;
    buffer->data[buffer->length] = '\0';
    return true;
}

static bool inspect_buffer_add_str(InspectBuffer *buffer, const char *text) {
    return inspect_buffer_add(buffer, text, strlen(text));
}

static bool inspect_buffer_add_size(InspectBuffer *buffer, size_t value) {
    char text[32];
    int length = snprintf(text, sizeof(text), "%zu", value);
    if (length < 0 || (size_t)length >= (int)sizeof(text)) return false;
    return inspect_buffer_add(buffer, text, (size_t)length);
}

static bool inspect_buffer_add_double(InspectBuffer *buffer, double value) {
    char text[64];
    int length = snprintf(text, sizeof(text), "%.12g", value);
    if (length < 0 || (size_t)length >= (int)sizeof(text)) return false;
    return inspect_buffer_add(buffer, text, (size_t)length);
}

typedef struct {
    const LanaStateDist **nodes;
    size_t *heights;
    size_t count;
    size_t capacity;
    size_t append_count;
    size_t transform_count;
    size_t dirac_count;
    bool sampling_required;
} InspectGraph;

static bool inspect_graph_reserve(InspectGraph *graph, size_t extra) {
    size_t needed = graph->count + extra;
    const LanaStateDist **nodes;
    size_t *heights;
    if (needed <= graph->capacity) return true;
    if (needed > SIZE_MAX / sizeof(*graph->nodes)) return false;
    graph->capacity = graph->capacity == 0u ? 16u : graph->capacity;
    while (graph->capacity < needed) graph->capacity *= 2u;
    nodes = realloc(graph->nodes, graph->capacity * sizeof(*graph->nodes));
    if (nodes == NULL) return false;
    graph->nodes = nodes;
    heights = realloc(graph->heights, graph->capacity * sizeof(*graph->heights));
    if (heights == NULL) return false;
    graph->heights = heights;
    return true;
}

static size_t inspect_node_id(const InspectGraph *graph, const LanaStateDist *node) {
    size_t index;
    for (index = 0u; index < graph->count; ++index)
        if (graph->nodes[index] == node) return index;
    return SIZE_MAX;
}

/* Assign a stable id to every reachable node, record each node's height (max
 * edges to a leaf), and accumulate the structural counters. Shared subgraphs
 * are visited once, so the reported node count matches the DAG, not the tree
 * expansion. */
static LanaError inspect_visit(const LanaStateDist *node, InspectGraph *graph,
                               size_t *out_id) {
    size_t index, id, height = 0u;
    if (node == NULL) return LANA_ERR_INVALID_DISTRIBUTION;
    for (index = 0u; index < graph->count; ++index)
        if (graph->nodes[index] == node) { *out_id = index; return LANA_OK; }
    if (!inspect_graph_reserve(graph, 1u)) return LANA_ERR_OOM;
    id = graph->count;
    graph->nodes[id] = node;
    graph->heights[id] = 0u;
    ++graph->count;
    switch (node->kind) {
        case LANA_DIST_DIRAC:
            ++graph->dirac_count;
            break;
        case LANA_DIST_APPEND: {
            size_t child_id;
            ++graph->append_count;
            if (node->as.append.sigma != 0.0) graph->sampling_required = true;
            if (node->as.append.left.is_inline) {
                if (height < 1u) height = 1u;
            } else {
                LanaError error = inspect_visit(node->as.append.left.as.node, graph, &child_id);
                if (error != LANA_OK) return error;
                if (graph->heights[child_id] + 1u > height) height = graph->heights[child_id] + 1u;
            }
            if (node->as.append.right.is_inline) {
                if (height < 1u) height = 1u;
            } else {
                LanaError error = inspect_visit(node->as.append.right.as.node, graph, &child_id);
                if (error != LANA_OK) return error;
                if (graph->heights[child_id] + 1u > height) height = graph->heights[child_id] + 1u;
            }
            break;
        }
        case LANA_DIST_TRANSFORM: {
            size_t child_id;
            LanaError error;
            ++graph->transform_count;
            error = inspect_visit(node->as.transform.child, graph, &child_id);
            if (error != LANA_OK) return error;
            height = graph->heights[child_id] + 1u;
            break;
        }
        case LANA_DIST_ATTENUATE: {
            size_t child_id;
            LanaError error;
            ++graph->transform_count;
            error = inspect_visit(node->as.attenuate.child, graph, &child_id);
            if (error != LANA_OK) return error;
            height = graph->heights[child_id] + 1u;
            break;
        }
        default:
            return LANA_ERR_INVALID_DISTRIBUTION;
    }
    graph->heights[id] = height;
    *out_id = id;
    return LANA_OK;
}

static bool inspect_emit_state(InspectBuffer *buffer, const LanaState *state) {
    return inspect_buffer_add_str(buffer, "{\"p\":")
        && inspect_buffer_add_double(buffer, state->p)
        && inspect_buffer_add_str(buffer, ",\"d_re\":")
        && inspect_buffer_add_double(buffer, state->d_re)
        && inspect_buffer_add_str(buffer, ",\"d_im\":")
        && inspect_buffer_add_double(buffer, state->d_im)
        && inspect_buffer_add_str(buffer, "}");
}

static bool inspect_emit_operand_json(const InspectGraph *graph,
                                      const LanaDistOperand *operand,
                                      InspectBuffer *buffer) {
    if (operand->is_inline) {
        if (!inspect_buffer_add_str(buffer, "{\"inline\":")) return false;
        if (!inspect_emit_state(buffer, &operand->as.state.state)) return false;
        return inspect_buffer_add_str(buffer, "}");
    }
    if (!inspect_buffer_add_str(buffer, "{\"node\":")) return false;
    if (!inspect_buffer_add_size(buffer, inspect_node_id(graph, operand->as.node))) return false;
    return inspect_buffer_add_str(buffer, "}");
}

static bool inspect_emit_node_json(const InspectGraph *graph, size_t id,
                                   InspectBuffer *buffer) {
    const LanaStateDist *node = graph->nodes[id];
    if (!inspect_buffer_add_str(buffer, "{\"id\":")) return false;
    if (!inspect_buffer_add_size(buffer, id)) return false;
    switch (node->kind) {
        case LANA_DIST_DIRAC:
            if (!inspect_buffer_add_str(buffer, ",\"kind\":\"dirac\",\"state\":")) return false;
            if (!inspect_emit_state(buffer, &node->as.dirac.state)) return false;
            break;
        case LANA_DIST_APPEND:
            if (!inspect_buffer_add_str(buffer, ",\"kind\":\"append\",\"left\":")) return false;
            if (!inspect_emit_operand_json(graph, &node->as.append.left, buffer)) return false;
            if (!inspect_buffer_add_str(buffer, ",\"right\":")) return false;
            if (!inspect_emit_operand_json(graph, &node->as.append.right, buffer)) return false;
            if (!inspect_buffer_add_str(buffer, ",\"sigma\":")) return false;
            if (!inspect_buffer_add_double(buffer, node->as.append.sigma)) return false;
            break;
        case LANA_DIST_TRANSFORM:
            if (!inspect_buffer_add_str(buffer, ",\"kind\":\"transform\",\"transform_id\":")) return false;
            if (!inspect_buffer_add_size(buffer, node->as.transform.transform_id)) return false;
            if (!inspect_buffer_add_str(buffer, ",\"child\":")) return false;
            if (!inspect_buffer_add_size(buffer, inspect_node_id(graph, node->as.transform.child))) return false;
            break;
        case LANA_DIST_ATTENUATE:
            if (!inspect_buffer_add_str(buffer, ",\"kind\":\"attenuate\",\"factor\":")) return false;
            if (!inspect_buffer_add_double(buffer, node->as.attenuate.factor)) return false;
            if (!inspect_buffer_add_str(buffer, ",\"child\":")) return false;
            if (!inspect_buffer_add_size(buffer, inspect_node_id(graph, node->as.attenuate.child))) return false;
            break;
        default:
            return false;
    }
    return inspect_buffer_add_str(buffer, "}");
}

static bool inspect_emit_json(const InspectGraph *graph, size_t root_id,
                              InspectBuffer *buffer) {
    size_t index;
    if (!inspect_buffer_add_str(buffer, "{\"kind\":\"state_dist\"")) return false;
    if (!inspect_buffer_add_str(buffer, ",\"node_count\":")) return false;
    if (!inspect_buffer_add_size(buffer, graph->count)) return false;
    if (!inspect_buffer_add_str(buffer, ",\"max_depth\":")) return false;
    if (!inspect_buffer_add_size(buffer, graph->heights[root_id])) return false;
    if (!inspect_buffer_add_str(buffer, ",\"append_count\":")) return false;
    if (!inspect_buffer_add_size(buffer, graph->append_count)) return false;
    if (!inspect_buffer_add_str(buffer, ",\"transform_count\":")) return false;
    if (!inspect_buffer_add_size(buffer, graph->transform_count)) return false;
    if (!inspect_buffer_add_str(buffer, ",\"dirac_count\":")) return false;
    if (!inspect_buffer_add_size(buffer, graph->dirac_count)) return false;
    if (!inspect_buffer_add_str(buffer, ",\"exact_measurement\":")) return false;
    if (!inspect_buffer_add_str(buffer, graph->sampling_required ? "false" : "true")) return false;
    if (!inspect_buffer_add_str(buffer, ",\"sampling_required\":")) return false;
    if (!inspect_buffer_add_str(buffer, graph->sampling_required ? "true" : "false")) return false;
    if (!inspect_buffer_add_str(buffer, ",\"provenance\":{}")) return false;
    if (!inspect_buffer_add_str(buffer, ",\"root\":")) return false;
    if (!inspect_buffer_add_size(buffer, root_id)) return false;
    if (!inspect_buffer_add_str(buffer, ",\"nodes\":[")) return false;
    for (index = 0u; index < graph->count; ++index) {
        if (index > 0u && !inspect_buffer_add_str(buffer, ",")) return false;
        if (!inspect_emit_node_json(graph, index, buffer)) return false;
    }
    return inspect_buffer_add_str(buffer, "]}");
}

static bool inspect_emit_dot_node(const InspectGraph *graph, size_t id,
                                  InspectBuffer *buffer) {
    const LanaStateDist *node = graph->nodes[id];
    if (!inspect_buffer_add_str(buffer, "  n")) return false;
    if (!inspect_buffer_add_size(buffer, id)) return false;
    if (!inspect_buffer_add_str(buffer, " [label=\"")) return false;
    switch (node->kind) {
        case LANA_DIST_DIRAC:
            if (!inspect_buffer_add_str(buffer, "dirac\\np=")) return false;
            if (!inspect_buffer_add_double(buffer, node->as.dirac.state.p)) return false;
            break;
        case LANA_DIST_APPEND:
            if (!inspect_buffer_add_str(buffer, "append\\nsigma=")) return false;
            if (!inspect_buffer_add_double(buffer, node->as.append.sigma)) return false;
            break;
        case LANA_DIST_TRANSFORM: {
            const LanaTransformSpec *spec = lana_transform_spec(node->as.transform.transform_id);
            if (!inspect_buffer_add_str(buffer, "transform\\n")) return false;
            if (!inspect_buffer_add_str(buffer, spec == NULL ? "?" : spec->name)) return false;
            break;
        }
        case LANA_DIST_ATTENUATE:
            if (!inspect_buffer_add_str(buffer, "attenuate\\nfactor=")) return false;
            if (!inspect_buffer_add_double(buffer, node->as.attenuate.factor)) return false;
            break;
        default:
            return false;
    }
    return inspect_buffer_add_str(buffer, "\"];\n");
}

static bool inspect_emit_dot_edge(const InspectGraph *graph, size_t from,
                                  const char *label, const LanaDistOperand *operand,
                                  InspectBuffer *buffer) {
    if (!inspect_buffer_add_str(buffer, "  n")) return false;
    if (!inspect_buffer_add_size(buffer, from)) return false;
    if (!inspect_buffer_add_str(buffer, " -> ")) return false;
    if (operand->is_inline) {
        if (!inspect_buffer_add_str(buffer, "i")) return false;
        if (!inspect_buffer_add_size(buffer, from)) return false;
        if (!inspect_buffer_add_str(buffer, "_")) return false;
        if (!inspect_buffer_add_str(buffer, label)) return false;
        if (!inspect_buffer_add_str(buffer, " [label=\"")) return false;
        if (!inspect_buffer_add_str(buffer, label)) return false;
        if (!inspect_buffer_add_str(buffer, ": state(p=")) return false;
        if (!inspect_buffer_add_double(buffer, operand->as.state.state.p)) return false;
        if (!inspect_buffer_add_str(buffer, ")\"];\n")) return false;
        if (!inspect_buffer_add_str(buffer, "  i")) return false;
        if (!inspect_buffer_add_size(buffer, from)) return false;
        if (!inspect_buffer_add_str(buffer, "_")) return false;
        if (!inspect_buffer_add_str(buffer, label)) return false;
        if (!inspect_buffer_add_str(buffer, " [label=\"state(p=")) return false;
        if (!inspect_buffer_add_double(buffer, operand->as.state.state.p)) return false;
        return inspect_buffer_add_str(buffer, ")\"];\n");
    }
    if (!inspect_buffer_add_str(buffer, "n")) return false;
    if (!inspect_buffer_add_size(buffer, inspect_node_id(graph, operand->as.node))) return false;
    if (!inspect_buffer_add_str(buffer, " [label=\"")) return false;
    if (!inspect_buffer_add_str(buffer, label)) return false;
    return inspect_buffer_add_str(buffer, "\"];\n");
}

static bool inspect_emit_dot(const InspectGraph *graph, InspectBuffer *buffer) {
    size_t index;
    if (!inspect_buffer_add_str(buffer, "digraph state_dist {\n")) return false;
    for (index = 0u; index < graph->count; ++index)
        if (!inspect_emit_dot_node(graph, index, buffer)) return false;
    for (index = 0u; index < graph->count; ++index) {
        const LanaStateDist *node = graph->nodes[index];
        if (node->kind == LANA_DIST_APPEND) {
            if (!inspect_emit_dot_edge(graph, index, "left", &node->as.append.left, buffer)) return false;
            if (!inspect_emit_dot_edge(graph, index, "right", &node->as.append.right, buffer)) return false;
        } else if (node->kind == LANA_DIST_TRANSFORM) {
            LanaDistOperand child;
            child.is_inline = false;
            child.as.node = node->as.transform.child;
            if (!inspect_emit_dot_edge(graph, index, "child", &child, buffer)) return false;
        } else if (node->kind == LANA_DIST_ATTENUATE) {
            LanaDistOperand child;
            child.is_inline = false;
            child.as.node = node->as.attenuate.child;
            if (!inspect_emit_dot_edge(graph, index, "child", &child, buffer)) return false;
        }
    }
    return inspect_buffer_add_str(buffer, "}\n");
}

LanaError lana_vm_state_dist_inspect(const LanaStateDist *distribution,
                                     LanaInspectFormat format, char **out) {
    InspectGraph graph = {0};
    InspectBuffer buffer = {0};
    size_t root_id = 0u;
    LanaError error;
    if (distribution == NULL || out == NULL) return LANA_ERR_INVALID_DISTRIBUTION;
    error = inspect_visit(distribution, &graph, &root_id);
    if (error != LANA_OK) {
        free(graph.nodes); free(graph.heights);
        return error;
    }
    if (format == LANA_INSPECT_JSON) {
        if (!inspect_emit_json(&graph, root_id, &buffer)) error = LANA_ERR_OOM;
    } else if (format == LANA_INSPECT_DOT) {
        if (!inspect_emit_dot(&graph, &buffer)) error = LANA_ERR_OOM;
    } else {
        error = LANA_ERR_INVALID_DISTRIBUTION;
    }
    free(graph.nodes); free(graph.heights);
    if (error != LANA_OK) { free(buffer.data); return error; }
    *out = buffer.data;
    return LANA_OK;
}

static LanaError build_statistical_result(LanaVM *vm, const char *method,
                                          double value, uint32_t observable,
                                          LanaMap **out) {
    LanaMap *map;
    LanaError error;
    if (vm == NULL || method == NULL || out == NULL) return LANA_ERR_FORMAT;
    if ((error = lana_map_new(vm, 6u, &map)) != LANA_OK) return error;
    if ((error = map_put(vm, map, "method", lana_value_string(method))) != LANA_OK)
        return error;
    if ((error = map_put(vm, map, "value", lana_value_number(value))) != LANA_OK)
        return error;
    if ((error = map_put(vm, map, "observable",
                         lana_value_string(observable == LANA_OBSERVABLE_PROBABILITY
                                           ? "probability" : "unknown"))) != LANA_OK)
        return error;
    if ((error = map_put(vm, map, "provenance", lana_value_string("exact"))) != LANA_OK)
        return error;
    if ((error = map_put(vm, map, "sample_count", lana_value_null())) != LANA_OK)
        return error;
    if ((error = map_put(vm, map, "seed", lana_value_null())) != LANA_OK)
        return error;
    *out = map;
    return LANA_OK;
}

static LanaError validate_result(LanaVM *vm, const char *status, const char *reason,
                                 LanaMap **out) {
    LanaMap *map;
    LanaError error;
    if (vm == NULL || status == NULL || reason == NULL || out == NULL)
        return LANA_ERR_FORMAT;
    if ((error = lana_map_new(vm, 3u, &map)) != LANA_OK) return error;
    if ((error = map_put(vm, map, "status", lana_value_string(status))) != LANA_OK)
        return error;
    if ((error = map_put(vm, map, "reason", lana_value_string(reason))) != LANA_OK)
        return error;
    if ((error = map_put(vm, map, "schema_version", lana_value_number(1.0))) != LANA_OK)
        return error;
    *out = map;
    return LANA_OK;
}

static LanaError lana_vm_validate(LanaVM *vm, const Value *value, const Value *schema,
                                 LanaMap **out) {
    Value type_value, required_value, constraints_value, exactness_value;
    const char *type_name;
    ValueType expected;
    if (vm == NULL || value == NULL || schema == NULL || out == NULL)
        return LANA_ERR_FORMAT;
    if (schema->type != VAL_MAP) return LANA_ERR_SCHEMA;
    if (lana_map_get(schema->as.map, "type", &type_value) != LANA_OK ||
        type_value.type != VAL_STRING)
        return LANA_ERR_SCHEMA;
    type_name = type_value.as.string;
    if (strcmp(type_name, "number") == 0) expected = VAL_NUMBER;
    else if (strcmp(type_name, "bool") == 0) expected = VAL_BOOL;
    else if (strcmp(type_name, "string") == 0) expected = VAL_STRING;
    else if (strcmp(type_name, "state") == 0) expected = VAL_STATE;
    else if (strcmp(type_name, "array") == 0) expected = VAL_ARRAY;
    else if (strcmp(type_name, "map") == 0) expected = VAL_MAP;
    else if (strcmp(type_name, "any") == 0) expected = value->type;
    else return LANA_ERR_SCHEMA;

    if (value->derivation != NULL &&
        value->derivation->outcome == LANA_DERIVATION_UNRESOLVED)
        return validate_result(vm, "insufficient_evidence", "unresolved_derivation", out);

    if (value->type != expected)
        return validate_result(vm, "invalid", "type_mismatch", out);

    if (expected == VAL_MAP &&
        lana_map_get(schema->as.map, "required", &required_value) == LANA_OK &&
        required_value.type == VAL_ARRAY) {
        size_t index;
        for (index = 0u; index < required_value.as.array->count; ++index) {
            Value field = required_value.as.array->items[index];
            if (field.type != VAL_STRING) return LANA_ERR_SCHEMA;
            if (lana_map_has(value->as.map, field.as.string) < 0)
                return validate_result(vm, "invalid", "missing_required_field", out);
        }
    }

    if (lana_map_get(schema->as.map, "constraints", &constraints_value) == LANA_OK &&
        constraints_value.type == VAL_MAP) {
        if (expected == VAL_NUMBER) {
            Value min_value, max_value;
            if (lana_map_get(constraints_value.as.map, "min", &min_value) == LANA_OK &&
                min_value.type == VAL_NUMBER &&
                value->as.number < min_value.as.number)
                return validate_result(vm, "invalid", "below_minimum", out);
            if (lana_map_get(constraints_value.as.map, "max", &max_value) == LANA_OK &&
                max_value.type == VAL_NUMBER &&
                value->as.number > max_value.as.number)
                return validate_result(vm, "invalid", "above_maximum", out);
        } else if (expected == VAL_STRING) {
            Value min_length_value, max_length_value;
            size_t length = strlen(value->as.string);
            if (lana_map_get(constraints_value.as.map, "min_length", &min_length_value) == LANA_OK &&
                min_length_value.type == VAL_NUMBER &&
                length < (size_t)min_length_value.as.number)
                return validate_result(vm, "invalid", "below_minimum_length", out);
            if (lana_map_get(constraints_value.as.map, "max_length", &max_length_value) == LANA_OK &&
                max_length_value.type == VAL_NUMBER &&
                length > (size_t)max_length_value.as.number)
                return validate_result(vm, "invalid", "above_maximum_length", out);
        }
    }

    if (lana_map_get(schema->as.map, "exactness", &exactness_value) == LANA_OK &&
        exactness_value.type == VAL_STRING &&
        strcmp(exactness_value.as.string, "exact") == 0 &&
        value->derivation != NULL &&
        value->derivation->exactness != LANA_EXACTNESS_EXACT)
        return validate_result(vm, "invalid", "exactness_mismatch", out);

    return validate_result(vm, "valid", "none", out);
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
        left->type == VAL_SET || right->type == VAL_SET ||
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
    if (left->type == VAL_TENSOR || right->type == VAL_TENSOR ||
        left->type == VAL_MAP || right->type == VAL_MAP) {
        if (kind != LANA_PURE_BINARY) return LANA_ERR_TYPE;
        /* LIP-027: a scalar operand adopts the tensor's dtype. */
        if (left->type == VAL_TENSOR && right->type == VAL_NUMBER)
            return tensor_elementwise_scalar(vm, left->as.tensor, right->as.number, (int)operation, out);
        if (left->type == VAL_NUMBER && right->type == VAL_TENSOR)
            return tensor_elementwise_scalar(vm, right->as.tensor, left->as.number, (int)operation, out);
        const LanaTensor *a_pred = NULL, *a_var = NULL, *b_pred = NULL, *b_var = NULL;
        bool a_unc = false, b_unc = false;
        LanaError unpack_error = tensor_uncertainty_unpack(left, &a_pred, &a_var, &a_unc);
        if (unpack_error != LANA_OK) return unpack_error;
        unpack_error = tensor_uncertainty_unpack(right, &b_pred, &b_var, &b_unc);
        if (unpack_error != LANA_OK) return unpack_error;
        if (a_unc || b_unc) {
            if (!a_unc) { a_var = tensor_zeros_like(vm, a_pred); if (a_var == NULL) return LANA_ERR_OOM; }
            if (!b_unc) { b_var = tensor_zeros_like(vm, b_pred); if (b_var == NULL) return LANA_ERR_OOM; }
            return tensor_elementwise_uncertain(vm, a_pred, a_var, b_pred, b_var, (int)operation, out);
        }
        return tensor_elementwise(vm, a_pred, b_pred, (int)operation, out);
    }
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
                                                const Value *replacement,
                                                uint32_t scratch_register) {
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
            else if (node->kind == LANA_REACTIVE_TRAIN)
                error = reactive_train_recompute(vm, node, left, scratch_register,
                                                 staged[index]);
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

static int compare_double(const void *a, const void *b) {
    double x = *(const double *)a;
    double y = *(const double *)b;
    if (x < y) return -1;
    if (x > y) return 1;
    return 0;
}

static LanaError bootstrap_resample(LanaVM *vm, const LanaArray *data,
                                    LanaArray **out) {
    LanaArray *resampled;
    size_t n = data->count;
    size_t index;
    resampled = lana_vm_alloc(vm, sizeof(*resampled));
    if (resampled == NULL) return LANA_ERR_OOM;
    resampled->count = n;
    resampled->capacity = n;
    resampled->items = lana_vm_alloc(vm, n * sizeof(*resampled->items));
    if (resampled->items == NULL && n > 0u) return LANA_ERR_OOM;
    for (index = 0; index < n; ++index) {
        size_t draw = (size_t)lana_vm_random(vm) % n;
        resampled->items[index] = data->items[draw];
    }
    *out = resampled;
    return LANA_OK;
}

static LanaError run_function(LanaVM *vm, uint32_t function_index,
                              const Value *arg, uint32_t scratch_register,
                              Value *result);

/* Build a Result tagged pair [tag, value] as a 2-element array, mirroring the
 * compiler's `emit_tagged_pair` encoding (`result_ok` -> [true, v],
 * `result_error` -> [false, e]). */
static LanaError make_result(LanaVM *vm, bool ok, Value value, Value *out) {
    LanaArray *array = lana_vm_alloc(vm, sizeof(*array));
    if (array == NULL) return LANA_ERR_OOM;
    array->count = array->capacity = 2u;
    array->items = lana_vm_alloc(vm, 2u * sizeof(*array->items));
    if (array->items == NULL) return LANA_ERR_OOM;
    array->items[0] = lana_value_bool(ok);
    array->items[1] = value;
    *out = lana_value_array(array);
    return LANA_OK;
}

/* LIP-024 async/await event loop. The ready queue is a FIFO of runnable
 * futures; `ready` on a future mirrors membership in the queue. Scheduling is
 * deterministic: the oldest runnable future runs next. */

static LanaError enqueue_future(LanaVM *vm, LanaFuture *future) {
    if (future->ready) return LANA_OK;
    if (vm->ready_count >= vm->ready_capacity) {
        size_t new_capacity = vm->ready_capacity == 0u ? 8u : vm->ready_capacity * 2u;
        LanaFuture **new_queue = lana_vm_alloc(vm, new_capacity * sizeof(*new_queue));
        if (new_queue == NULL) return LANA_ERR_OOM;
        if (vm->ready_queue != NULL)
            memcpy(new_queue, vm->ready_queue, vm->ready_count * sizeof(*new_queue));
        vm->ready_queue = new_queue;
        vm->ready_capacity = new_capacity;
    }
    vm->ready_queue[vm->ready_count++] = future;
    future->ready = true;
    return LANA_OK;
}

static LanaFuture *dequeue_future(LanaVM *vm) {
    LanaFuture *future = vm->ready_queue[0];
    --vm->ready_count;
    memmove(vm->ready_queue, &vm->ready_queue[1],
            vm->ready_count * sizeof(*vm->ready_queue));
    future->ready = false;
    return future;
}

static void complete_future(LanaVM *vm, LanaFuture *future, Value result) {
    (void)vm;
    future->exhausted = true;
    future->ready = false;
    future->registers[0] = result;
}

/* Run a composite future's completion check. If its condition is met it
 * completes; otherwise it re-queues itself (and any not-yet-run inputs) so the
 * loop re-checks it later. */
static LanaError run_composite_future(LanaVM *vm, LanaFuture *future) {
    size_t index;
    switch (future->composite_kind) {
        case LANA_FUTURE_ALL: {
            bool all_done = true;
            for (index = 0u; index < future->input_count; ++index)
                if (!future->inputs[index]->exhausted) { all_done = false; break; }
            if (all_done) {
                LanaArray *array = lana_vm_alloc(vm, sizeof(*array));
                if (array == NULL) return LANA_ERR_OOM;
                array->count = array->capacity = future->input_count;
                array->items = lana_vm_alloc(vm, future->input_count * sizeof(*array->items));
                if (array->items == NULL) return LANA_ERR_OOM;
                for (index = 0u; index < future->input_count; ++index)
                    array->items[index] = future->inputs[index]->registers[0];
                complete_future(vm, future, lana_value_array(array));
            } else {
                for (index = 0u; index < future->input_count; ++index) {
                    LanaFuture *input = future->inputs[index];
                    if (!input->exhausted) {
                        LanaError error = enqueue_future(vm, input);
                        if (error != LANA_OK) return error;
                    }
                }
                return enqueue_future(vm, future);
            }
            break;
        }
        case LANA_FUTURE_RACE: {
            LanaFuture *winner = NULL;
            for (index = 0u; index < future->input_count; ++index)
                if (future->inputs[index]->exhausted) { winner = future->inputs[index]; break; }
            if (winner != NULL) {
                complete_future(vm, future, winner->registers[0]);
            } else {
                for (index = 0u; index < future->input_count; ++index) {
                    LanaFuture *input = future->inputs[index];
                    if (!input->exhausted) {
                        LanaError error = enqueue_future(vm, input);
                        if (error != LANA_OK) return error;
                    }
                }
                return enqueue_future(vm, future);
            }
            break;
        }
        case LANA_FUTURE_SLEEP: {
            struct timespec now;
            double now_seconds;
            if (timespec_get(&now, TIME_UTC) != TIME_UTC) return LANA_ERR_TYPE;
            now_seconds = (double)now.tv_sec + (double)now.tv_nsec / 1000000000.0;
            if (now_seconds >= future->wake_time) {
                complete_future(vm, future, lana_value_null());
            } else {
                return enqueue_future(vm, future);
            }
            break;
        }
        default:
            return LANA_ERR_TYPE;
    }
    return LANA_OK;
}

/* Push a fresh frame for a function future and run it until it suspends on an
 * OP_AWAIT or returns (both pop the frame). Mirrors OP_NEXT's frame setup. */
static LanaError run_async_function(LanaVM *vm, LanaFuture *future) {
    LanaFrame *callee;
    size_t saved_frame_count = vm->frame_count;
    size_t index;
    if (vm->frame_count >= LANA_MAX_CALL_FRAMES) return LANA_ERR_LIMIT;
    callee = &vm->frames[vm->frame_count++];
    for (index = 0u; index < future->register_count; ++index) {
        callee->registers[index] = future->registers[index];
        memset(&callee->histories[index], 0, sizeof(callee->histories[index]));
    }
    callee->registers[0] = lana_value_future(future);
    callee->return_ip = vm->ip;
    callee->return_register = 0u;
    callee->function = future->function;
    callee->is_generator = false;
    callee->is_async = true;
    vm->ip = future->ip;
    while (vm->frame_count > saved_frame_count && vm->running) {
        LanaError error = vm_step(vm);
        if (error != LANA_OK) return error;
    }
    return LANA_OK;
}

/* Drive the single-threaded cooperative event loop until `target` is exhausted,
 * then return its result. Nested invocations simply run a nested loop. */
static LanaError run_event_loop(LanaVM *vm, LanaFuture *target, Value *out) {
    size_t saved_ip = vm->ip;
    LanaError error = LANA_OK;
    while (vm->ready_count > 0u && !target->exhausted) {
        LanaFuture *future = dequeue_future(vm);
        if (future->is_composite)
            error = run_composite_future(vm, future);
        else
            error = run_async_function(vm, future);
        if (error != LANA_OK) break;
    }
    vm->ip = saved_ip;
    if (error != LANA_OK) return error;
    if (!target->exhausted) return LANA_ERR_TYPE;
    *out = target->registers[0];
    return LANA_OK;
}

static LanaError vm_step(LanaVM *vm) {
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
        if (vm->profile_opcodes) ++vm->opcode_counts[ins->opcode];
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
            case OP_MIX: {
                const Value *left = &frame->registers[ins->b];
                const Value *right = &frame->registers[ins->c];
                const Value *weight = &frame->registers[ins->imm];
                LanaState state;
                if (left->type == VAL_STATE_DIST || right->type == VAL_STATE_DIST)
                    error = LANA_ERR_UNSUPPORTED_OPERATION;
                else if (left->type != VAL_STATE || right->type != VAL_STATE)
                    error = LANA_ERR_TYPE;
                else if (weight->type != VAL_NUMBER) error = LANA_ERR_TYPE;
                else error = lana_state_mix(&left->as.state.state,
                                            &right->as.state.state,
                                            weight->as.number, &state);
                if (error == LANA_OK) {
                    char details[64];
                    const Value *inputs[] = {left, right};
                    (void)snprintf(details, sizeof(details), "w=%.17g",
                                   weight->as.number);
                    error = store_state(vm, ins->a, lana_value_state(state).as.state);
                    if (error == LANA_OK)
                        error = attach_combine_derivation(vm, &frame->registers[ins->a],
                            "mix", inputs, 2u, ins->line, details);
                }
                break;
            }
            case OP_MAP: {
                const Value *source = &frame->registers[ins->b];
                if (source->type != VAL_STATE_DIST) error = LANA_ERR_TYPE;
                else if (lana_transform_spec(ins->c) == NULL) error = LANA_ERR_TRANSFORM;
                else {
                    LanaStateDist *distribution;
                    error = lana_vm_state_dist_transform(vm, ins->c,
                                                       source->as.state_dist,
                                                       &distribution);
                    if (error == LANA_OK) {
                        const Value *inputs[] = {source};
                        frame->registers[ins->a] = lana_value_state_dist(distribution);
                        error = attach_derivation(vm, &frame->registers[ins->a],
                            LANA_DERIVATION_OPERATION, "map", inputs, 1u, "",
                            ins->line, LANA_EXACTNESS_EXACT,
                            lana_transform_spec(ins->c)->name);
                    }
                }
                break;
            }
            case OP_SUPPORT: {
                const Value *source = &frame->registers[ins->b];
                if (source->type != VAL_STATE_DIST) error = LANA_ERR_TYPE;
                else {
                    LanaArray *array;
                    error = lana_vm_state_dist_support(vm, source->as.state_dist,
                                                       ins->imm, &array);
                    if (error == LANA_OK) {
                        const Value *inputs[] = {source};
                        frame->registers[ins->a] = lana_value_array(array);
                        error = attach_derivation(vm, &frame->registers[ins->a],
                            LANA_DERIVATION_OPERATION, "support", inputs, 1u, "",
                            ins->line, LANA_EXACTNESS_EXACT, NULL);
                    }
                }
                break;
            }
            case OP_EXPECT: {
                const Value *source = &frame->registers[ins->b];
                if (source->type != VAL_STATE_DIST) error = LANA_ERR_TYPE;
                else if (ins->imm != LANA_OBSERVABLE_PROBABILITY)
                    error = LANA_ERR_UNSUPPORTED_OPERATION;
                else {
                    double expected;
                    LanaMap *result;
                    error = lana_vm_state_dist_expected_probability(source->as.state_dist,
                                                                    &expected);
                    if (error == LANA_OK)
                        error = build_statistical_result(vm, "exact", expected,
                                                         ins->imm, &result);
                    if (error == LANA_OK) {
                        const Value *inputs[] = {source};
                        frame->registers[ins->a] = lana_value_map(result);
                        error = attach_derivation(vm, &frame->registers[ins->a],
                            LANA_DERIVATION_OPERATION, "expect", inputs, 1u, "",
                            ins->line, LANA_EXACTNESS_EXACT, "exact");
                    }
                }
                break;
            }
            case OP_VALIDATE: {
                const Value *value = &frame->registers[ins->b];
                const Value *schema = &frame->registers[ins->c];
                LanaMap *result;
                error = lana_vm_validate(vm, value, schema, &result);
                if (error == LANA_OK) {
                    const Value *inputs[] = {value, schema};
                    frame->registers[ins->a] = lana_value_map(result);
                    error = attach_derivation(vm, &frame->registers[ins->a],
                        LANA_DERIVATION_OPERATION, "validate", inputs, 2u, "",
                        ins->line, LANA_EXACTNESS_EXACT, NULL);
                }
                break;
            }
            case OP_REVISION: {
                const Value *source = &frame->registers[ins->b];
                uint64_t revision;
                if (source->type == VAL_SHARED_CAPABILITY) {
                    LanaSharedInformation *shared =
                        lana_shared_capability_information(source->as.capability);
                    if (shared == NULL) error = LANA_ERR_TYPE;
                    else revision = lana_shared_information_revision(shared);
                } else if (source->derivation != NULL) {
                    revision = source->derivation->revision;
                } else if (source->reactive != NULL) {
                    revision = source->reactive->revision;
                } else error = LANA_ERR_TYPE;
                if (error == LANA_OK) {
                    const Value *inputs[] = {source};
                    frame->registers[ins->a] = lana_value_number((double)revision);
                    error = attach_derivation(vm, &frame->registers[ins->a],
                        LANA_DERIVATION_OPERATION, "revision", inputs, 1u, "",
                        ins->line, LANA_EXACTNESS_EXACT, NULL);
                }
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
            case OP_ATTENUATE: {
                const Value *source = &frame->registers[ins->b];
                const Value *factor = &frame->registers[ins->c];
                if (factor->type != VAL_NUMBER) error = LANA_ERR_TYPE;
                else if (source->type == VAL_STATE) {
                    LanaStateValue attenuated = source->as.state;
                    error = lana_state_attenuate(&source->as.state.state,
                                                 factor->as.number,
                                                 &attenuated.state);
                    if (error == LANA_OK) {
                        const Value *inputs[] = {source, factor};
                        error = store_state(vm, ins->a, attenuated);
                        if (error == LANA_OK)
                            error = attach_derivation(vm, &frame->registers[ins->a],
                                LANA_DERIVATION_OPERATION, "attenuate", inputs, 2u, "",
                                ins->line, LANA_EXACTNESS_EXACT, NULL);
                    }
                } else if (source->type == VAL_STATE_DIST) {
                    LanaStateDist *distribution;
                    error = lana_vm_state_dist_attenuate(vm, source->as.state_dist,
                                                         factor->as.number,
                                                         &distribution);
                    if (error == LANA_OK)
                        frame->registers[ins->a] = lana_value_state_dist(distribution);
                } else error = LANA_ERR_TYPE;
                break;
            }
            case OP_TRACE_DISTANCE: {
                const Value *left = &frame->registers[ins->b];
                const Value *right = &frame->registers[ins->c];
                double distance;
                if (left->type == VAL_STATE_DIST || right->type == VAL_STATE_DIST)
                    error = LANA_ERR_UNSUPPORTED_OPERATION;
                else if (left->type != VAL_STATE || right->type != VAL_STATE)
                    error = LANA_ERR_TYPE;
                else error = lana_state_trace_distance(&left->as.state.state,
                                                       &right->as.state.state,
                                                       &distance);
                if (error == LANA_OK) {
                    const Value *inputs[] = {left, right};
                    frame->registers[ins->a] = lana_value_number(distance);
                    error = attach_combine_derivation(vm, &frame->registers[ins->a],
                        "trace_distance", inputs, 2u, ins->line, NULL);
                }
                break;
            }
            case OP_APPEND_REDUNDANT:
            case OP_APPEND_COMPLEMENTARY: {
                const Value *left = &frame->registers[ins->a];
                const Value *right = &frame->registers[ins->b];
                const Value *strength = &frame->registers[ins->imm];
                LanaStateDist *distribution;
                uint32_t mode = ins->opcode == OP_APPEND_REDUNDANT
                    ? LANA_APPEND_REDUNDANT : LANA_APPEND_COMPLEMENTARY;
                if (strength->type != VAL_NUMBER) error = LANA_ERR_TYPE;
                else error = lana_vm_state_dist_append_relationship(vm, left, right,
                                                                    mode,
                                                                    strength->as.number,
                                                                    &distribution);
                if (error == LANA_OK)
                    frame->registers[ins->c] = lana_value_state_dist(distribution);
                break;
            }
            case OP_APPEND_FULL_REDUNDANCY: {
                const Value *left = &frame->registers[ins->a];
                const Value *right = &frame->registers[ins->b];
                LanaStateDist *distribution;
                error = lana_vm_state_dist_append_relationship(vm, left, right,
                                                               LANA_APPEND_FULL_REDUNDANCY,
                                                               0.0, &distribution);
                if (error == LANA_OK)
                    frame->registers[ins->c] = lana_value_state_dist(distribution);
                break;
            }
            case OP_ADT_BUILD: {
                LanaAdt *adt;
                const Value *tag = &vm->chunk->constants[ins->imm];
                if (tag->type != VAL_NUMBER) { error = LANA_ERR_TYPE; break; }
                adt = lana_vm_alloc(vm, sizeof(*adt));
                if (adt == NULL) { error = LANA_ERR_OOM; break; }
                adt->variant = (uint32_t)tag->as.number;
                adt->field_count = ins->c;
                adt->fields = lana_vm_alloc(vm, ins->c * sizeof(*adt->fields));
                if (adt->fields == NULL && ins->c > 0u) { error = LANA_ERR_OOM; break; }
                if (ins->c > 0u)
                    memcpy(adt->fields, &frame->registers[ins->b],
                           ins->c * sizeof(*adt->fields));
                frame->registers[ins->a] = lana_value_adt(adt);
                break;
            }
            case OP_ADT_CASE: {
                const Value *value = &frame->registers[ins->a];
                const Value *tag = &vm->chunk->constants[ins->b];
                if (tag->type != VAL_NUMBER) { error = LANA_ERR_TYPE; break; }
                if (value->type != VAL_ADT) { error = LANA_ERR_TYPE; break; }
                if (value->as.adt->variant == (uint32_t)tag->as.number)
                    vm->ip = ins->imm;
                break;
            }
            case OP_ADT_GET: {
                const Value *value = &frame->registers[ins->b];
                if (value->type != VAL_ADT) { error = LANA_ERR_TYPE; break; }
                if (ins->c >= value->as.adt->field_count) {
                    error = LANA_ERR_INVALID_PARAMETERS; break;
                }
                frame->registers[ins->a] = value->as.adt->fields[ins->c];
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
                    error = reactive_observe_scratch(
                        vm, source, &frame->registers[ins->imm],
                        ins->b, &frame->registers[ins->b]);
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
                else if (source->type == VAL_TRAINING_RESULT &&
                         source->as.training_result != NULL && ins->c <= 1u) {
                    /* LIP-010: reactive parameters resolve to the current
                     * training result, not the frozen one captured at `train`. */
                    const Value *effective = reactive_value(source);
                    if (effective->type != VAL_TRAINING_RESULT ||
                        effective->as.training_result == NULL)
                        error = LANA_ERR_TYPE;
                    else if (ins->c == 0u)
                        frame->registers[ins->b] = lana_value_tensor(
                            effective->as.training_result->params);
                    else
                        frame->registers[ins->b] = lana_value_array(
                            effective->as.training_result->steps);
                }
                else if (source->type == VAL_POSTERIOR &&
                         source->as.posterior != NULL && ins->c <= 3u) {
                    if (ins->c == 0u)
                        frame->registers[ins->b] = lana_value_tensor(
                            source->as.posterior->mean);
                    else if (ins->c == 1u)
                        frame->registers[ins->b] = lana_value_tensor(
                            source->as.posterior->variance);
                    else if (ins->c == 2u)
                        frame->registers[ins->b] = lana_value_tensor(
                            source->as.posterior->samples);
                    else
                        frame->registers[ins->b] = lana_value_array(
                            source->as.posterior->steps);
                }
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
                if (error == LANA_OK && vm->ad_recording &&
                    left->type == VAL_TENSOR && right->type == VAL_TENSOR) {
                    error = ad_record(vm, (int)ins->imm, left, right, -1,
                                      &frame->registers[ins->c]);
                } else if (error == LANA_OK && (left->derivation != NULL || right->derivation != NULL)) {
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
                frame->registers[ins->a] = lana_value_array(array); break;
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
                callee->return_ip = vm->ip; callee->return_register = ins->a; callee->function = ins->b; callee->is_generator = false;
                for (index = 0; index < ins->imm; ++index) {
                    callee->registers[index] = frame->registers[ins->c + index];
                    error = clone_history(vm, &frame->histories[ins->c + index],
                                          &callee->histories[index]);
                    if (error != LANA_OK) break;
                }
                if (error != LANA_OK) break;
                vm->ip = function->entry; break;
            }
            case OP_LAZY: {
                const Value *bound_value = &frame->registers[ins->c];
                LanaLazy lazy;
                if (bound_value->type != VAL_NUMBER ||
                    bound_value->as.number < 0.0 ||
                    bound_value->as.number > (double)SIZE_MAX ||
                    !isfinite(bound_value->as.number)) {
                    error = LANA_ERR_TYPE; break;
                }
                lazy.function = ins->b;
                lazy.bound = (size_t)bound_value->as.number;
                frame->registers[ins->a] = lana_value_lazy(lazy);
                break;
            }
            case OP_FORCE: {
                const Value *lazy_value = &frame->registers[ins->b];
                const Value *index_value = &frame->registers[ins->c];
                const LanaFunction *function;
                LanaFrame *callee;
                size_t index;
                if (lazy_value->type != VAL_LAZY ||
                    index_value->type != VAL_NUMBER ||
                    index_value->as.number < 0.0 ||
                    floor(index_value->as.number) != index_value->as.number) {
                    error = LANA_ERR_TYPE; break;
                }
                index = (size_t)index_value->as.number;
                if (index >= lazy_value->as.lazy.bound) { error = LANA_ERR_LIMIT; break; }
                function = &vm->chunk->functions[lazy_value->as.lazy.function];
                if (function->arity != 1u) { error = LANA_ERR_TYPE; break; }
                if (vm->frame_count >= LANA_MAX_CALL_FRAMES) { error = LANA_ERR_LIMIT; break; }
                callee = &vm->frames[vm->frame_count++];
                for (index = 0; index < function->register_count; ++index) {
                    callee->registers[index] = lana_value_null();
                    memset(&callee->histories[index], 0, sizeof(callee->histories[index]));
                }
                callee->return_ip = vm->ip; callee->return_register = ins->a;
                callee->function = lazy_value->as.lazy.function;
                callee->is_generator = false;
                callee->registers[0] = *index_value;
                error = clone_history(vm, &frame->histories[ins->c],
                                      &callee->histories[0]);
                if (error != LANA_OK) break;
                vm->ip = function->entry; break;
            }
            case OP_BOOTSTRAP: {
                const Value *data_value = &frame->registers[ins->imm];
                const Value *b_value = &frame->registers[ins->c];
                const LanaFunction *function;
                LanaArray *data;
                size_t n, b, index;
                double estimate, ci_low, ci_high;
                double *resamples;
                size_t lo, hi;
                LanaMap *map;
                if (data_value->type != VAL_ARRAY || b_value->type != VAL_NUMBER) {
                    error = LANA_ERR_TYPE; break;
                }
                if (b_value->as.number < 1.0 ||
                    floor(b_value->as.number) != b_value->as.number ||
                    b_value->as.number > (double)SIZE_MAX) {
                    error = LANA_ERR_INVALID_PARAMETERS; break;
                }
                data = data_value->as.array;
                n = data->count;
                if (n == 0u) { error = LANA_ERR_INVALID_PARAMETERS; break; }
                b = (size_t)b_value->as.number;
                if (ins->b >= vm->chunk->function_count) { error = LANA_ERR_OPCODE; break; }
                function = &vm->chunk->functions[ins->b];
                if (function->arity != 1u) { error = LANA_ERR_TYPE; break; }
                resamples = malloc(b * sizeof(*resamples));
                if (resamples == NULL) { error = LANA_ERR_OOM; break; }
                {
                    Value result;
                    error = run_function(vm, ins->b, data_value, ins->a, &result);
                    if (error != LANA_OK) { free(resamples); break; }
                    if (result.type != VAL_NUMBER) { free(resamples); error = LANA_ERR_TYPE; break; }
                    estimate = result.as.number;
                }
                for (index = 0; index < b && error == LANA_OK; ++index) {
                    LanaArray *resampled;
                    Value resampled_value;
                    Value result;
                    error = bootstrap_resample(vm, data, &resampled);
                    if (error != LANA_OK) break;
                    resampled_value = lana_value_array(resampled);
                    error = run_function(vm, ins->b, &resampled_value, ins->a, &result);
                    if (error != LANA_OK) break;
                    if (result.type != VAL_NUMBER) { error = LANA_ERR_TYPE; break; }
                    resamples[index] = result.as.number;
                }
                if (error != LANA_OK) { free(resamples); break; }
                qsort(resamples, b, sizeof(*resamples), compare_double);
                lo = (size_t)(0.025 * (double)b);
                hi = (size_t)(0.975 * (double)b);
                if (lo >= b) lo = b - 1u;
                if (hi >= b) hi = b - 1u;
                ci_low = resamples[lo];
                ci_high = resamples[hi];
                free(resamples);
                if ((error = lana_map_new(vm, 7u, &map)) != LANA_OK) break;
                if ((error = map_put(vm, map, "estimate", lana_value_number(estimate))) != LANA_OK ||
                    (error = map_put(vm, map, "ci_low", lana_value_number(ci_low))) != LANA_OK ||
                    (error = map_put(vm, map, "ci_high", lana_value_number(ci_high))) != LANA_OK ||
                    (error = map_put(vm, map, "method", lana_value_string("sampled"))) != LANA_OK ||
                    (error = map_put(vm, map, "procedure", lana_value_string("bootstrap"))) != LANA_OK ||
                    (error = map_put(vm, map, "sample_count", lana_value_number((double)b))) != LANA_OK ||
                    (error = map_put(vm, map, "seed", lana_value_number((double)vm->root_seed))) != LANA_OK)
                    break;
                frame->registers[ins->a] = lana_value_map(map);
                break;
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
                    frame->registers[ins->a] = (Value){.type = VAL_TASK, .as.task = task};
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
                    frame->registers[ins->b] = lana_value_array(results);
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
                Value result = lana_value_null();
                if (error == LANA_OK) {
                    if (ins->b == LANA_HOST_GRAD) {
                        if (ins->imm != 2u) error = LANA_ERR_TYPE;
                        else error = ad_grad(vm, &arguments[0], &arguments[1],
                                             ins->a, &result);
                    } else if (ins->b == LANA_HOST_VJP) {
                        if (ins->imm != 3u) error = LANA_ERR_TYPE;
                        else error = ad_vjp(vm, &arguments[0], &arguments[1],
                                            &arguments[2], ins->a, &result);
                    } else if (ins->b == LANA_HOST_TRAIN) {
                        error = host_train(vm, arguments, ins->imm, ins->a, &result);
                    } else if (ins->b == LANA_HOST_UPDATE) {
                        error = host_update(vm, arguments, ins->imm, ins->a, &result);
                    } else if (ins->b == LANA_HOST_RESUME) {
                        error = host_resume(vm, arguments, ins->imm, ins->a, &result);
                    } else if (ins->b == LANA_HOST_INFER) {
                        error = host_infer(vm, arguments, ins->imm, ins->a, &result);
                    } else {
                        error = execute_host_call(vm, ins->b, arguments, ins->imm,
                                                  ins->a, &result);
                    }
                }
                if (error == LANA_ERR_ASSERTION && ins->b == LANA_HOST_ASSERT &&
                    ins->imm == 2u && frame->registers[ins->c + 1u].type == VAL_STRING)
                    error_message = frame->registers[ins->c + 1u].as.string;
                if (error == LANA_OK && ins->b == LANA_HOST_GPU_MATMUL) {
                    const Value *inputs[] = { &arguments[0], &arguments[1] };
                    error = attach_derivation(vm, &result, LANA_DERIVATION_APPROXIMATION,
                        "gpu_matmul", inputs, 2u, "", ins->line,
                        LANA_EXACTNESS_APPROXIMATE, "backend=metal precision=float32");
                }
                if (error == LANA_OK) frame->registers[ins->a] = result;
                break;
            }
            case OP_GENERATOR: {
                const LanaFunction *function = &vm->chunk->functions[ins->b];
                LanaGenerator *generator;
                size_t index;
                if (ins->imm != function->arity) { error = LANA_ERR_TYPE; break; }
                generator = lana_vm_alloc(vm, sizeof(*generator));
                if (generator == NULL) { error = LANA_ERR_OOM; break; }
                generator->function = ins->b;
                generator->ip = function->entry;
                generator->register_count = function->register_count;
                generator->exhausted = false;
                generator->registers = lana_vm_alloc(vm, function->register_count * sizeof(Value));
                if (generator->registers == NULL) { error = LANA_ERR_OOM; break; }
                for (index = 0; index < function->register_count; ++index)
                    generator->registers[index] = lana_value_null();
                for (index = 0; index < ins->imm; ++index)
                    generator->registers[1u + index] = frame->registers[ins->c + index];
                frame->registers[ins->a] = lana_value_generator(generator);
                break;
            }
            case OP_YIELD: {
                Value *gen_value = &frame->registers[ins->a];
                LanaGenerator *generator;
                Value yielded = frame->registers[ins->b];
                size_t index;
                if (gen_value->type != VAL_GENERATOR) { error = LANA_ERR_TYPE; break; }
                generator = gen_value->as.generator;
                generator->ip = vm->ip;
                for (index = 1u; index < generator->register_count; ++index)
                    generator->registers[index] = frame->registers[index];
                generator->registers[0] = lana_value_null();
                if (vm->frame_count == 1u) { error = LANA_ERR_TYPE; break; }
                {
                    size_t return_ip = frame->return_ip;
                    uint32_t destination = frame->return_register;
                    --vm->frame_count;
                    error = make_result(vm, true, yielded,
                                        &current_frame(vm)->registers[destination]);
                    if (error == LANA_OK) vm->ip = return_ip;
                }
                break;
            }
            case OP_NEXT: {
                Value *gen_value = &frame->registers[ins->a];
                LanaGenerator *generator;
                LanaFrame *callee;
                size_t index;
                if (gen_value->type != VAL_GENERATOR) { error = LANA_ERR_TYPE; break; }
                generator = gen_value->as.generator;
                if (generator->exhausted) {
                    char *exhausted = lana_vm_alloc(vm, sizeof("exhausted"));
                    if (exhausted == NULL) { error = LANA_ERR_OOM; break; }
                    memcpy(exhausted, "exhausted", sizeof("exhausted"));
                    error = make_result(vm, false, lana_value_string(exhausted),
                                        &frame->registers[ins->b]);
                    break;
                }
                if (vm->frame_count >= LANA_MAX_CALL_FRAMES) { error = LANA_ERR_LIMIT; break; }
                callee = &vm->frames[vm->frame_count++];
                for (index = 0; index < generator->register_count; ++index) {
                    callee->registers[index] = generator->registers[index];
                    memset(&callee->histories[index], 0, sizeof(callee->histories[index]));
                }
                callee->registers[0] = *gen_value;
                callee->return_ip = vm->ip;
                callee->return_register = ins->b;
                callee->function = generator->function;
                callee->is_generator = true;
                vm->ip = generator->ip;
                break;
            }
            case OP_ASYNC: {
                /* Create a cold future: allocate the frame but do not execute
                 * the body (mirrors OP_GENERATOR). */
                const LanaFunction *function = &vm->chunk->functions[ins->b];
                LanaFuture *future;
                size_t index;
                if (ins->imm != function->arity) { error = LANA_ERR_TYPE; break; }
                future = lana_vm_alloc(vm, sizeof(*future));
                if (future == NULL) { error = LANA_ERR_OOM; break; }
                future->function = ins->b;
                future->ip = function->entry;
                future->register_count = function->register_count;
                future->exhausted = false;
                future->ready = false;
                future->is_composite = false;
                future->composite_kind = 0u;
                future->inputs = NULL;
                future->input_count = 0u;
                future->wake_time = 0.0;
                future->registers = lana_vm_alloc(vm, function->register_count * sizeof(Value));
                if (future->registers == NULL) { error = LANA_ERR_OOM; break; }
                for (index = 0u; index < function->register_count; ++index)
                    future->registers[index] = lana_value_null();
                for (index = 0u; index < ins->imm; ++index)
                    future->registers[1u + index] = frame->registers[ins->c + index];
                frame->registers[ins->a] = lana_value_future(future);
                break;
            }
            case OP_AWAIT: {
                /* Suspend the current async frame until the awaited future
                 * completes, then store its result and yield to the loop. */
                Value *future_value = &frame->registers[ins->a];
                LanaFuture *current, *awaited;
                size_t index;
                if (future_value->type != VAL_FUTURE) { error = LANA_ERR_TYPE; break; }
                awaited = future_value->as.future;
                if (awaited->exhausted) {
                    frame->registers[ins->b] = awaited->registers[0];
                    break;
                }
                if (frame->registers[0].type != VAL_FUTURE) { error = LANA_ERR_TYPE; break; }
                current = frame->registers[0].as.future;
                /* Resume at the AWAIT instruction itself (not the next one) so
                 * the re-executed AWAIT observes the awaited future's result. */
                current->ip = instruction_ip;
                for (index = 1u; index < current->register_count; ++index)
                    current->registers[index] = frame->registers[index];
                current->registers[0] = lana_value_null();
                if (vm->frame_count == 1u) { error = LANA_ERR_TYPE; break; }
                --vm->frame_count;
                error = enqueue_future(vm, awaited);
                if (error == LANA_OK) error = enqueue_future(vm, current);
                break;
            }
            case OP_RUN_ASYNC: {
                /* Run the event loop to completion on the target future. */
                Value *future_value = &frame->registers[ins->a];
                LanaFuture *target;
                if (future_value->type != VAL_FUTURE) { error = LANA_ERR_TYPE; break; }
                target = future_value->as.future;
                if (target->exhausted) {
                    frame->registers[ins->b] = target->registers[0];
                    break;
                }
                error = enqueue_future(vm, target);
                if (error == LANA_OK)
                    error = run_event_loop(vm, target, &frame->registers[ins->b]);
                break;
            }
            case OP_LOAD_FUNCTION:
                frame->registers[ins->a] = lana_value_function(ins->b);
                break;
            case OP_RETURN: {
                Value returned = frame->registers[ins->a];
                if (frame->is_generator) {
                    Value *gen_value = &frame->registers[0];
                    char *exhausted;
                    if (gen_value->type != VAL_GENERATOR) { error = LANA_ERR_TYPE; break; }
                    gen_value->as.generator->exhausted = true;
                    if (vm->frame_count == 1u) { error = LANA_ERR_TYPE; break; }
                    exhausted = lana_vm_alloc(vm, sizeof("exhausted"));
                    if (exhausted == NULL) { error = LANA_ERR_OOM; break; }
                    memcpy(exhausted, "exhausted", sizeof("exhausted"));
                    {
                        size_t return_ip = frame->return_ip;
                        uint32_t destination = frame->return_register;
                        --vm->frame_count;
                        error = make_result(vm, false, lana_value_string(exhausted),
                                            &current_frame(vm)->registers[destination]);
                        if (error == LANA_OK) vm->ip = return_ip;
                    }
                } else if (frame->is_async) {
                    /* An async frame returning completes its future. The
                     * event loop's run_async_function observes the frame pop
                     * and continues with the next ready future. */
                    Value *future_value = &frame->registers[0];
                    LanaFuture *future;
                    if (future_value->type != VAL_FUTURE) { error = LANA_ERR_TYPE; break; }
                    future = future_value->as.future;
                    complete_future(vm, future, returned);
                    if (vm->frame_count == 1u) { error = LANA_ERR_TYPE; break; }
                    --vm->frame_count;
                } else if (vm->frame_count == 1u) { vm->result = returned; vm->running = false; }
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
    return LANA_OK;
}

static LanaError run_function(LanaVM *vm, uint32_t function_index,
                              const Value *arg, uint32_t scratch_register,
                              Value *result) {
    const LanaFunction *function = &vm->chunk->functions[function_index];
    LanaFrame *caller = current_frame(vm);
    LanaFrame *callee;
    size_t saved_frame_count = vm->frame_count;
    size_t index;
    if (function->arity != 1u) return LANA_ERR_TYPE;
    if (vm->frame_count >= LANA_MAX_CALL_FRAMES) return LANA_ERR_LIMIT;
    callee = &vm->frames[vm->frame_count++];
    for (index = 0; index < function->register_count; ++index) {
        callee->registers[index] = lana_value_null();
        memset(&callee->histories[index], 0, sizeof(callee->histories[index]));
    }
    callee->return_ip = vm->ip;
    callee->return_register = scratch_register;
    callee->function = function_index;
    callee->is_generator = false;
    callee->registers[0] = *arg;
    vm->ip = function->entry;
    while (vm->frame_count > saved_frame_count && vm->running) {
        LanaError error = vm_step(vm);
        if (error != LANA_OK) {
            vm->frame_count = saved_frame_count;
            return error;
        }
    }
    *result = caller->registers[scratch_register];
    return LANA_OK;
}

/* Run a two-argument Lana function to completion, mirroring `run_function`.
 * Used by `train` to invoke the model (params, x) and loss (y, target)
 * functions, which are ordinary arity-2 functions over tensors. */
static LanaError run_function2(LanaVM *vm, uint32_t function_index,
                               const Value *arg0, const Value *arg1,
                               uint32_t scratch_register, Value *result) {
    const LanaFunction *function = &vm->chunk->functions[function_index];
    LanaFrame *caller = current_frame(vm);
    LanaFrame *callee;
    size_t saved_frame_count = vm->frame_count;
    size_t index;
    if (function->arity != 2u) return LANA_ERR_TYPE;
    if (vm->frame_count >= LANA_MAX_CALL_FRAMES) return LANA_ERR_LIMIT;
    callee = &vm->frames[vm->frame_count++];
    for (index = 0; index < function->register_count; ++index) {
        callee->registers[index] = lana_value_null();
        memset(&callee->histories[index], 0, sizeof(callee->histories[index]));
    }
    callee->return_ip = vm->ip;
    callee->return_register = scratch_register;
    callee->function = function_index;
    callee->is_generator = false;
    callee->registers[0] = *arg0;
    callee->registers[1] = *arg1;
    vm->ip = function->entry;
    while (vm->frame_count > saved_frame_count && vm->running) {
        LanaError error = vm_step(vm);
        if (error != LANA_OK) {
            vm->frame_count = saved_frame_count;
            return error;
        }
    }
    *result = caller->registers[scratch_register];
    return LANA_OK;
}

/* ===== LIP-015: lazy relational-algebra dataset engine ===== */

static LanaError dataset_new(LanaVM *vm, LanaDatasetOp op, Value source,
                             uint32_t function, Value columns, Value key,
                             Value limit, Value other, Value aggregate,
                             LanaDataset **out) {
    LanaDataset *dataset = lana_vm_alloc(vm, sizeof(*dataset));
    if (dataset == NULL) return LANA_ERR_OOM;
    dataset->op = op;
    dataset->source = source;
    dataset->function = function;
    dataset->columns = columns;
    dataset->key = key;
    dataset->limit = limit;
    dataset->other = other;
    dataset->aggregate = aggregate;
    *out = dataset;
    return LANA_OK;
}

static LanaError dataset_array_new(LanaVM *vm, LanaArray **out) {
    LanaArray *array = lana_vm_alloc(vm, sizeof(*array));
    if (array == NULL) return LANA_ERR_OOM;
    array->count = 0u;
    array->capacity = 0u;
    array->items = NULL;
    *out = array;
    return LANA_OK;
}

static LanaError dataset_array_push(LanaVM *vm, LanaArray *array, const Value *value) {
    Value *items;
    if (array->count == SIZE_MAX / sizeof(*items)) return LANA_ERR_LIMIT;
    if (array->count == array->capacity) {
        size_t capacity = array->capacity == 0u ? 8u : array->capacity * 2u;
        if (capacity <= array->capacity) return LANA_ERR_LIMIT;
        items = lana_vm_alloc(vm, capacity * sizeof(*items));
        if (items == NULL) return LANA_ERR_OOM;
        if (array->count > 0u)
            memcpy(items, array->items, array->count * sizeof(*items));
        array->items = items;
        array->capacity = capacity;
    }
    array->items[array->count++] = *value;
    return LANA_OK;
}

/* Materialize a source into an array of rows. A lazy source is materialized by
 * invoking the generator function for each index in [0, bound); an in-memory
 * array source (LIP-015 §3 persistence / adapter load path) is returned as-is.
 * The output array is GC-rooted for the duration so a collection triggered by
 * a generator call cannot free it. */
static LanaError dataset_materialize_source(LanaVM *vm, const Value *lazy_value,
                                            uint32_t scratch, LanaArray **out) {
    LanaArray *rows;
    Value rows_value;
    size_t bound, i;
    if (lazy_value->type == VAL_ARRAY) {
        *out = lazy_value->as.array;
        return LANA_OK;
    }
    if (lazy_value->type != VAL_LAZY) return LANA_ERR_TYPE;
    bound = lazy_value->as.lazy.bound;
    if (dataset_array_new(vm, &rows) != LANA_OK) return LANA_ERR_OOM;
    rows_value = lana_value_array(rows);
    size_t root_base = lana_vm_root_push(vm, &rows_value);
    for (i = 0u; i < bound; ++i) {
        Value index_value = lana_value_number((double)i);
        Value row;
        LanaError error = run_function(vm, lazy_value->as.lazy.function,
                                       &index_value, scratch, &row);
        if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }
        error = dataset_array_push(vm, rows, &row);
        if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }
    }
    lana_vm_root_pop(vm, root_base);
    *out = rows;
    return LANA_OK;
}

/* Read the value of `key` from a row (a map). */
static LanaError dataset_row_key(const Value *row, const char *key, Value *out) {
    if (row->type != VAL_MAP || row->as.map == NULL) return LANA_ERR_TYPE;
    return lana_map_get(row->as.map, key, out);
}

/* Compare two values for sort ordering. Returns <0, 0, >0. Numbers compare
 * numerically; strings byte-wise; bools false<true; everything else by type
 * tag then pointer. */
static int dataset_compare_values(const Value *a, const Value *b) {
    if (a->type == VAL_NUMBER && b->type == VAL_NUMBER) {
        if (a->as.number < b->as.number) return -1;
        if (a->as.number > b->as.number) return 1;
        return 0;
    }
    if (a->type == VAL_STRING && b->type == VAL_STRING)
        return strcmp(a->as.string, b->as.string);
    if (a->type == VAL_BOOL && b->type == VAL_BOOL)
        return (a->as.boolean ? 1 : 0) - (b->as.boolean ? 1 : 0);
    if (a->type != b->type) return (int)a->type - (int)b->type;
    return (a->as.array > b->as.array) - (a->as.array < b->as.array);
}

/* Materialize a dataset plan to an array of rows. `scratch` is a caller
 * register used to run predicate/transform functions. */
static LanaError dataset_materialize(LanaVM *vm, const LanaDataset *dataset,
                                     uint32_t scratch, LanaArray **out) {
    LanaError error;
    switch (dataset->op) {
        case LANA_DATASET_SOURCE:
            return dataset_materialize_source(vm, &dataset->source, scratch, out);

        case LANA_DATASET_FILTER: {
            LanaArray *source_rows, *result;
            Value result_value;
            size_t i;
            error = dataset_materialize(vm, dataset->source.as.dataset, scratch,
                                        &source_rows);
            if (error != LANA_OK) return error;
            if (dataset_array_new(vm, &result) != LANA_OK) return LANA_ERR_OOM;
            result_value = lana_value_array(result);
            size_t root_base = lana_vm_root_push(vm, &result_value);
            for (i = 0u; i < source_rows->count; ++i) {
                Value pred_result;
                error = run_function(vm, dataset->function,
                                      &source_rows->items[i], scratch, &pred_result);
                if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }
                if (pred_result.type == VAL_BOOL && pred_result.as.boolean) {
                    error = dataset_array_push(vm, result, &source_rows->items[i]);
                    if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }
                }
            }
            lana_vm_root_pop(vm, root_base);
            *out = result;
            return LANA_OK;
        }

        case LANA_DATASET_MAP: {
            LanaArray *source_rows, *result;
            Value result_value;
            size_t i;
            error = dataset_materialize(vm, dataset->source.as.dataset, scratch,
                                        &source_rows);
            if (error != LANA_OK) return error;
            if (dataset_array_new(vm, &result) != LANA_OK) return LANA_ERR_OOM;
            result_value = lana_value_array(result);
            size_t root_base = lana_vm_root_push(vm, &result_value);
            for (i = 0u; i < source_rows->count; ++i) {
                Value mapped;
                error = run_function(vm, dataset->function,
                                     &source_rows->items[i], scratch, &mapped);
                if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }
                error = dataset_array_push(vm, result, &mapped);
                if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }
            }
            lana_vm_root_pop(vm, root_base);
            *out = result;
            return LANA_OK;
        }

        case LANA_DATASET_SELECT: {
            LanaArray *source_rows, *result;
            Value result_value;
            size_t i, c;
            if (dataset->columns.type != VAL_ARRAY) return LANA_ERR_TYPE;
            error = dataset_materialize(vm, dataset->source.as.dataset, scratch,
                                        &source_rows);
            if (error != LANA_OK) return error;
            if (dataset_array_new(vm, &result) != LANA_OK) return LANA_ERR_OOM;
            result_value = lana_value_array(result);
            size_t root_base = lana_vm_root_push(vm, &result_value);
            for (i = 0u; i < source_rows->count; ++i) {
                const Value *row = &source_rows->items[i];
                LanaMap *projected;
                if (row->type != VAL_MAP) { lana_vm_root_pop(vm, root_base); return LANA_ERR_TYPE; }
                if (lana_map_new(vm, dataset->columns.as.array->count, &projected) != LANA_OK) {
                    lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM;
                }
                for (c = 0u; c < dataset->columns.as.array->count; ++c) {
                    const Value *col = &dataset->columns.as.array->items[c];
                    Value col_value;
                    if (col->type != VAL_STRING) { lana_vm_root_pop(vm, root_base); return LANA_ERR_TYPE; }
                    if (lana_map_get(row->as.map, col->as.string, &col_value) != LANA_OK) {
                        lana_vm_root_pop(vm, root_base); return LANA_ERR_TYPE;
                    }
                    if (lana_map_set(vm, projected, col->as.string, &col_value, false) != LANA_OK) {
                        lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM;
                    }
                }
                Value projected_value = lana_value_map(projected);
                error = dataset_array_push(vm, result, &projected_value);
                if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }
            }
            lana_vm_root_pop(vm, root_base);
            *out = result;
            return LANA_OK;
        }

        case LANA_DATASET_LIMIT: {
            LanaArray *source_rows, *result;
            size_t n, i;
            if (dataset->limit.type != VAL_NUMBER || dataset->limit.as.number < 0.0)
                return LANA_ERR_TYPE;
            n = (size_t)dataset->limit.as.number;
            error = dataset_materialize(vm, dataset->source.as.dataset, scratch,
                                        &source_rows);
            if (error != LANA_OK) return error;
            if (dataset_array_new(vm, &result) != LANA_OK) return LANA_ERR_OOM;
            if (n > source_rows->count) n = source_rows->count;
            for (i = 0u; i < n; ++i) {
                error = dataset_array_push(vm, result, &source_rows->items[i]);
                if (error != LANA_OK) return error;
            }
            *out = result;
            return LANA_OK;
        }

        case LANA_DATASET_SORT: {
            LanaArray *source_rows, *result;
            size_t i, j;
            if (dataset->key.type != VAL_STRING) return LANA_ERR_TYPE;
            error = dataset_materialize(vm, dataset->source.as.dataset, scratch,
                                        &source_rows);
            if (error != LANA_OK) return error;
            if (dataset_array_new(vm, &result) != LANA_OK) return LANA_ERR_OOM;
            for (i = 0u; i < source_rows->count; ++i) {
                error = dataset_array_push(vm, result, &source_rows->items[i]);
                if (error != LANA_OK) return error;
            }
            /* Insertion sort by key value (rows are maps). */
            for (i = 1u; i < result->count; ++i) {
                Value key_i, key_j;
                if (dataset_row_key(&result->items[i], dataset->key.as.string, &key_i) != LANA_OK)
                    return LANA_ERR_TYPE;
                Value pivot = result->items[i];
                j = i;
                while (j > 0u) {
                    if (dataset_row_key(&result->items[j - 1u], dataset->key.as.string, &key_j) != LANA_OK)
                        return LANA_ERR_TYPE;
                    if (dataset_compare_values(&key_j, &key_i) <= 0) break;
                    result->items[j] = result->items[j - 1u];
                    --j;
                }
                result->items[j] = pivot;
            }
            *out = result;
            return LANA_OK;
        }

        case LANA_DATASET_GROUP_BY: {
            LanaArray *source_rows, *result;
            Value result_value;
            size_t i, g;
            if (dataset->key.type != VAL_STRING) return LANA_ERR_TYPE;
            error = dataset_materialize(vm, dataset->source.as.dataset, scratch,
                                        &source_rows);
            if (error != LANA_OK) return error;
            if (dataset_array_new(vm, &result) != LANA_OK) return LANA_ERR_OOM;
            result_value = lana_value_array(result);
            size_t root_base = lana_vm_root_push(vm, &result_value);
            for (i = 0u; i < source_rows->count; ++i) {
                Value key_value;
                LanaMap *group_map = NULL;
                Value group_value;
                LanaArray *group_rows;
                if (dataset_row_key(&source_rows->items[i], dataset->key.as.string,
                                    &key_value) != LANA_OK) {
                    lana_vm_root_pop(vm, root_base); return LANA_ERR_TYPE;
                }
                /* Find an existing group record with this key. */
                for (g = 0u; g < result->count; ++g) {
                    Value existing_key;
                    if (result->items[g].type != VAL_MAP) { lana_vm_root_pop(vm, root_base); return LANA_ERR_TYPE; }
                    if (lana_map_get(result->items[g].as.map, "key", &existing_key) != LANA_OK) {
                        lana_vm_root_pop(vm, root_base); return LANA_ERR_TYPE;
                    }
                    if (set_value_equal(&existing_key, &key_value)) {
                        group_map = result->items[g].as.map;
                        break;
                    }
                }
                if (group_map == NULL) {
                    if (lana_map_new(vm, 2u, &group_map) != LANA_OK) {
                        lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM;
                    }
                    if (dataset_array_new(vm, &group_rows) != LANA_OK) {
                        lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM;
                    }
                    Value group_rows_value = lana_value_array(group_rows);
                    if (lana_map_set(vm, group_map, "key", &key_value, false) != LANA_OK ||
                        lana_map_set(vm, group_map, "rows", &group_rows_value, false) != LANA_OK) {
                        lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM;
                    }
                    group_value = lana_value_map(group_map);
                    error = dataset_array_push(vm, result, &group_value);
                    if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }
                } else {
                    if (lana_map_get(group_map, "rows", &group_value) != LANA_OK) {
                        lana_vm_root_pop(vm, root_base); return LANA_ERR_TYPE;
                    }
                    group_rows = group_value.as.array;
                }
                error = dataset_array_push(vm, group_rows, &source_rows->items[i]);
                if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }
            }
            lana_vm_root_pop(vm, root_base);
            *out = result;
            return LANA_OK;
        }

        case LANA_DATASET_AGGREGATE: {
            LanaArray *group_records, *result;
            Value result_value;
            size_t i, r;
            const char *agg_op, *agg_col = NULL;
            if (dataset->aggregate.type != VAL_ARRAY ||
                dataset->aggregate.as.array->count < 1u ||
                dataset->aggregate.as.array->items[0].type != VAL_STRING)
                return LANA_ERR_TYPE;
            agg_op = dataset->aggregate.as.array->items[0].as.string;
            if (dataset->aggregate.as.array->count >= 2u) {
                if (dataset->aggregate.as.array->items[1].type != VAL_STRING)
                    return LANA_ERR_TYPE;
                agg_col = dataset->aggregate.as.array->items[1].as.string;
            }
            error = dataset_materialize(vm, dataset->source.as.dataset, scratch,
                                        &group_records);
            if (error != LANA_OK) return error;
            if (dataset_array_new(vm, &result) != LANA_OK) return LANA_ERR_OOM;
            result_value = lana_value_array(result);
            size_t root_base = lana_vm_root_push(vm, &result_value);
            for (i = 0u; i < group_records->count; ++i) {
                const Value *record = &group_records->items[i];
                Value key_value, rows_value;
                LanaArray *rows;
                LanaMap *out_row;
                Value agg_value;
                if (record->type != VAL_MAP ||
                    lana_map_get(record->as.map, "key", &key_value) != LANA_OK ||
                    lana_map_get(record->as.map, "rows", &rows_value) != LANA_OK ||
                    rows_value.type != VAL_ARRAY) {
                    lana_vm_root_pop(vm, root_base); return LANA_ERR_TYPE;
                }
                rows = rows_value.as.array;
                if (strcmp(agg_op, "count") == 0) {
                    agg_value = lana_value_number((double)rows->count);
                } else {
                    double acc = 0.0;
                    bool have = false;
                    if (agg_col == NULL) { lana_vm_root_pop(vm, root_base); return LANA_ERR_TYPE; }
                    for (r = 0u; r < rows->count; ++r) {
                        Value cell;
                        if (dataset_row_key(&rows->items[r], agg_col, &cell) != LANA_OK ||
                            cell.type != VAL_NUMBER) {
                            lana_vm_root_pop(vm, root_base); return LANA_ERR_TYPE;
                        }
                        if (!have) { acc = cell.as.number; have = true; }
                        else if (strcmp(agg_op, "sum") == 0 || strcmp(agg_op, "mean") == 0)
                            acc += cell.as.number;
                        else if (strcmp(agg_op, "max") == 0) { if (cell.as.number > acc) acc = cell.as.number; }
                        else if (strcmp(agg_op, "min") == 0) { if (cell.as.number < acc) acc = cell.as.number; }
                        else { lana_vm_root_pop(vm, root_base); return LANA_ERR_TYPE; }
                    }
                    if (strcmp(agg_op, "mean") == 0) {
                        if (rows->count == 0u) { lana_vm_root_pop(vm, root_base); return LANA_ERR_TYPE; }
                        acc /= (double)rows->count;
                    }
                    agg_value = lana_value_number(acc);
                }
                if (lana_map_new(vm, 2u, &out_row) != LANA_OK) {
                    lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM;
                }
                /* Output row: {<group key name>: key_value, <op>: agg_value}. */
                const char *key_name = "key";
                if (dataset->source.type == VAL_DATASET &&
                    dataset->source.as.dataset->op == LANA_DATASET_GROUP_BY &&
                    dataset->source.as.dataset->key.type == VAL_STRING)
                    key_name = dataset->source.as.dataset->key.as.string;
                if (lana_map_set(vm, out_row, key_name, &key_value, false) != LANA_OK ||
                    lana_map_set(vm, out_row, agg_op, &agg_value, false) != LANA_OK) {
                    lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM;
                }
                Value out_value = lana_value_map(out_row);
                error = dataset_array_push(vm, result, &out_value);
                if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }
            }
            lana_vm_root_pop(vm, root_base);
            *out = result;
            return LANA_OK;
        }

        case LANA_DATASET_JOIN: {
            LanaArray *left_rows, *right_rows, *result;
            Value result_value;
            size_t i, j;
            if (dataset->key.type != VAL_STRING) return LANA_ERR_TYPE;
            error = dataset_materialize(vm, dataset->source.as.dataset, scratch,
                                        &left_rows);
            if (error != LANA_OK) return error;
            error = dataset_materialize(vm, dataset->other.as.dataset, scratch,
                                        &right_rows);
            if (error != LANA_OK) return error;
            if (dataset_array_new(vm, &result) != LANA_OK) return LANA_ERR_OOM;
            result_value = lana_value_array(result);
            size_t root_base = lana_vm_root_push(vm, &result_value);
            for (i = 0u; i < left_rows->count; ++i) {
                Value left_key;
                if (dataset_row_key(&left_rows->items[i], dataset->key.as.string,
                                    &left_key) != LANA_OK) {
                    lana_vm_root_pop(vm, root_base); return LANA_ERR_TYPE;
                }
                for (j = 0u; j < right_rows->count; ++j) {
                    Value right_key;
                    if (dataset_row_key(&right_rows->items[j], dataset->key.as.string,
                                        &right_key) != LANA_OK) {
                        lana_vm_root_pop(vm, root_base); return LANA_ERR_TYPE;
                    }
                    if (!set_value_equal(&left_key, &right_key)) continue;
                    /* Merge left and right rows into one map. */
                    LanaMap *merged;
                    size_t e;
                    if (lana_map_new(vm, left_rows->items[i].as.map->count +
                                        right_rows->items[j].as.map->count, &merged) != LANA_OK) {
                        lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM;
                    }
                    for (e = 0u; e < left_rows->items[i].as.map->count; ++e) {
                        const LanaMapEntry *entry = &left_rows->items[i].as.map->entries[e];
                        if (lana_map_set(vm, merged, entry->key, entry->value, false) != LANA_OK) {
                            lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM;
                        }
                    }
                    for (e = 0u; e < right_rows->items[j].as.map->count; ++e) {
                        const LanaMapEntry *entry = &right_rows->items[j].as.map->entries[e];
                        if (lana_map_set(vm, merged, entry->key, entry->value, false) != LANA_OK) {
                            lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM;
                        }
                    }
                    Value merged_value = lana_value_map(merged);
                    error = dataset_array_push(vm, result, &merged_value);
                    if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }
                }
            }
            lana_vm_root_pop(vm, root_base);
            *out = result;
            return LANA_OK;
        }

        default:
            return LANA_ERR_TYPE;
    }
}

/* Build an inspectable plan value for `explain(ds)`. */
static LanaError dataset_explain(LanaVM *vm, const LanaDataset *dataset, Value *out) {
    static const char *op_names[] = {
        "source", "filter", "map", "select", "limit", "sort",
        "group_by", "aggregate", "join"
    };
    LanaMap *map;
    Value op_value, source_value;
    if (dataset == NULL) return LANA_ERR_TYPE;
    if (lana_map_new(vm, 4u, &map) != LANA_OK) return LANA_ERR_OOM;
    op_value = lana_value_string(op_names[dataset->op]);
    if (lana_map_set(vm, map, "op", &op_value, false) != LANA_OK) return LANA_ERR_OOM;
    if (dataset->op == LANA_DATASET_SOURCE) {
        /* LIP-015 §3: a source is either a lazy generator (report its bound) or
         * an in-memory array of rows (report the row count). */
        if (dataset->source.type == VAL_LAZY) {
            Value bound_value = lana_value_number((double)dataset->source.as.lazy.bound);
            if (lana_map_set(vm, map, "bound", &bound_value, false) != LANA_OK) return LANA_ERR_OOM;
        } else if (dataset->source.type == VAL_ARRAY) {
            Value count_value = lana_value_number((double)dataset->source.as.array->count);
            if (lana_map_set(vm, map, "rows", &count_value, false) != LANA_OK) return LANA_ERR_OOM;
        } else {
            return LANA_ERR_TYPE;
        }
    } else {
        if (dataset->source.type != VAL_DATASET) return LANA_ERR_TYPE;
        if (dataset_explain(vm, dataset->source.as.dataset, &source_value) != LANA_OK)
            return LANA_ERR_TYPE;
        if (lana_map_set(vm, map, "source", &source_value, false) != LANA_OK) return LANA_ERR_OOM;
    }
    if (dataset->op == LANA_DATASET_FILTER || dataset->op == LANA_DATASET_MAP) {
        Value fn_value = lana_value_function(dataset->function);
        if (lana_map_set(vm, map, "function", &fn_value, false) != LANA_OK) return LANA_ERR_OOM;
    }
    if (dataset->op == LANA_DATASET_SELECT) {
        if (lana_map_set(vm, map, "columns", &dataset->columns, false) != LANA_OK) return LANA_ERR_OOM;
    }
    if (dataset->op == LANA_DATASET_LIMIT) {
        if (lana_map_set(vm, map, "limit", &dataset->limit, false) != LANA_OK) return LANA_ERR_OOM;
    }
    if (dataset->op == LANA_DATASET_SORT || dataset->op == LANA_DATASET_GROUP_BY ||
        dataset->op == LANA_DATASET_JOIN) {
        if (lana_map_set(vm, map, "key", &dataset->key, false) != LANA_OK) return LANA_ERR_OOM;
    }
    if (dataset->op == LANA_DATASET_AGGREGATE) {
        if (lana_map_set(vm, map, "aggregate", &dataset->aggregate, false) != LANA_OK) return LANA_ERR_OOM;
    }
    if (dataset->op == LANA_DATASET_JOIN) {
        Value other_value;
        if (dataset->other.type != VAL_DATASET) return LANA_ERR_TYPE;
        if (dataset_explain(vm, dataset->other.as.dataset, &other_value) != LANA_OK)
            return LANA_ERR_TYPE;
        if (lana_map_set(vm, map, "other", &other_value, false) != LANA_OK) return LANA_ERR_OOM;
    }
    *out = lana_value_map(map);
    return LANA_OK;
}

/* ===== LIP-011 grad / vjp host calls ===== */

/* Validate `f`/`x`, create the input leaf, and run `f(x)` with recording on.
 * On success `*leaf_out` is the input leaf and `*result_out` is `f`'s output. */
static LanaError ad_run(LanaVM *vm, const Value *function_value, const Value *x,
                        uint32_t scratch_register, LanaDerivation **leaf_out,
                        Value *result_out) {
    if (function_value->type != VAL_FUNCTION) return LANA_ERR_TYPE;
    if (x->type != VAL_TENSOR) return LANA_ERR_TYPE;
    if (x->as.tensor->is_complex && !x->as.tensor->is_state) return LANA_ERR_TYPE;
    uint32_t function_index = function_value->as.function;
    if (function_index >= vm->chunk->function_count) return LANA_ERR_TYPE;
    if (vm->chunk->functions[function_index].arity != 1u) return LANA_ERR_TYPE;
    LanaDerivation *leaf = record_derivation(vm, LANA_DERIVATION_OPERATION, "input",
        NULL, 0u, "", 0u, LANA_EXACTNESS_EXACT, "autodiff",
        LANA_DERIVATION_SUCCESS, "none");
    if (leaf == NULL) return LANA_ERR_OOM;
    leaf->ad_a = x->as.tensor;
    Value x_with_deriv = *x;
    x_with_deriv.derivation = leaf;
    vm->ad_recording = true;
    LanaError error = run_function(vm, function_index, &x_with_deriv,
                                   scratch_register, result_out);
    vm->ad_recording = false;
    if (error != LANA_OK) return error;
    *leaf_out = leaf;
    return LANA_OK;
}

/* Read the leaf's accumulated gradient, check finiteness, and return a fresh
 * tensor carrying a provenance derivation that traces to the input leaf. */
static LanaError ad_finish(LanaVM *vm, LanaDerivation *leaf, const char *operation,
                           Value *out) {
    LanaTensor *grad = leaf->ad_grad;
    if (grad == NULL) {
        grad = tensor_new(vm, leaf->ad_a->ndim, leaf->ad_a->shape, leaf->ad_a->is_complex);
        if (grad == NULL) return LANA_ERR_OOM;
    }
    size_t count = tensor_element_count(grad);
    size_t components = grad->is_complex ? 2u : 1u;
    for (size_t i = 0; i < count * components; ++i)
        if (!isfinite(tensor_get_real(grad, i))) return LANA_ERR_INVALID_PARAMETERS;
    LanaTensor *result_tensor = tensor_new(vm, grad->ndim, grad->shape, grad->is_complex);
    if (result_tensor == NULL) return LANA_ERR_OOM;
    memcpy(result_tensor->data, grad->data, count * components * sizeof(double));
    Value leaf_value = lana_value_tensor(leaf->ad_a);
    leaf_value.derivation = leaf;
    const Value *inputs[] = { &leaf_value };
    LanaDerivation *grad_deriv = record_derivation(vm, LANA_DERIVATION_OPERATION,
        operation, inputs, 1u, "", 0u, LANA_EXACTNESS_EXACT, "autodiff",
        LANA_DERIVATION_SUCCESS, "none");
    if (grad_deriv == NULL) return LANA_ERR_OOM;
    Value grad_value = lana_value_tensor(result_tensor);
    grad_value.derivation = grad_deriv;
    *out = grad_value;
    return LANA_OK;
}

static LanaError ad_grad(LanaVM *vm, const Value *function_value, const Value *x,
                         uint32_t scratch_register, Value *out) {
    LanaDerivation *leaf = NULL;
    Value result;
    LanaError error = ad_run(vm, function_value, x, scratch_register, &leaf, &result);
    if (error != LANA_OK) return error;
    if (result.type != VAL_NUMBER) return LANA_ERR_TYPE;
    LanaTensor *seed = tensor_new(vm, 0, NULL, false);
    if (seed == NULL) return LANA_ERR_OOM;
    tensor_set_real(seed, 0, 1.0);
    if (result.derivation != NULL) {
        error = ad_backward(vm, result.derivation, seed);
        if (error != LANA_OK) return error;
    }
    return ad_finish(vm, leaf, "grad", out);
}

static LanaError ad_vjp(LanaVM *vm, const Value *function_value, const Value *x,
                        const Value *v, uint32_t scratch_register, Value *out) {
    if (v->type != VAL_TENSOR) return LANA_ERR_TYPE;
    if (v->as.tensor->is_complex) return LANA_ERR_TYPE;
    LanaDerivation *leaf = NULL;
    Value result;
    LanaError error = ad_run(vm, function_value, x, scratch_register, &leaf, &result);
    if (error != LANA_OK) return error;
    if (result.type != VAL_TENSOR) return LANA_ERR_TYPE;
    if (!tensor_shape_equal(v->as.tensor, result.as.tensor)) return LANA_ERR_TYPE;
    LanaTensor *seed = tensor_new(vm, v->as.tensor->ndim, v->as.tensor->shape, false);
    if (seed == NULL) return LANA_ERR_OOM;
    size_t count = tensor_element_count(v->as.tensor);
    size_t *idx = lana_vm_alloc(vm, v->as.tensor->ndim * sizeof(*idx));
    if (idx == NULL && v->as.tensor->ndim > 0) return LANA_ERR_OOM;
    for (size_t lin = 0; lin < count; ++lin) {
        size_t rem = lin, index = v->as.tensor->offset;
        for (size_t d = v->as.tensor->ndim; d-- > 0;) {
            index += (v->as.tensor->shape[d] == 0 ? 0 : rem % v->as.tensor->shape[d]) *
                     v->as.tensor->strides[d];
            rem /= v->as.tensor->shape[d];
        }
        tensor_set_real(seed, lin, tensor_get_real(v->as.tensor, index));
    }
    if (result.derivation != NULL) {
        error = ad_backward(vm, result.derivation, seed);
        if (error != LANA_OK) return error;
    }
    return ad_finish(vm, leaf, "vjp", out);
}

/* ===== LIP-006 sgd / adam / train host calls ===== */

/* Copy a tensor (possibly a view) into a fresh contiguous base tensor. */
static LanaTensor *tensor_copy_contiguous(LanaVM *vm, const LanaTensor *t) {
    LanaTensor *copy = tensor_new(vm, t->ndim, t->shape, t->is_complex);
    if (copy == NULL) return NULL;
    copy->is_state = t->is_state;
    size_t count = tensor_element_count(t);
    size_t *idx = lana_vm_alloc(vm, t->ndim * sizeof(*idx));
    if (idx == NULL && t->ndim > 0) return NULL;
    for (size_t lin = 0; lin < count; ++lin) {
        size_t rem = lin, index = t->offset;
        for (size_t d = t->ndim; d-- > 0;) {
            index += (t->shape[d] == 0 ? 0 : rem % t->shape[d]) * t->strides[d];
            rem /= t->shape[d];
        }
        if (t->is_complex) {
            tensor_set_real(copy, lin, tensor_get_real(t, index));
            tensor_set_imag(copy, lin, tensor_get_imag(t, index));
        } else {
            tensor_set_real(copy, lin, tensor_get_real(t, index));
        }
    }
    return copy;
}

static LanaError host_sgd(LanaVM *vm, const Value *arguments, size_t argc, Value *out) {
    double learning_rate = 0.01;
    double momentum = 0.9;
    if (argc > 2u) return LANA_ERR_TYPE;
    if (argc >= 1u) {
        if (arguments[0].type != VAL_NUMBER) return LANA_ERR_TYPE;
        learning_rate = arguments[0].as.number;
    }
    if (argc >= 2u) {
        if (arguments[1].type != VAL_NUMBER) return LANA_ERR_TYPE;
        momentum = arguments[1].as.number;
    }
    if (!isfinite(learning_rate) || learning_rate <= 0.0 ||
        !isfinite(momentum) || momentum < 0.0 || momentum >= 1.0)
        return LANA_ERR_INVALID_PARAMETERS;
    LanaOptimizer *optimizer = lana_vm_alloc(vm, sizeof(*optimizer));
    if (optimizer == NULL) return LANA_ERR_OOM;
    optimizer->name = derivation_string(vm, "sgd");
    if (optimizer->name == NULL) return LANA_ERR_OOM;
    optimizer->learning_rate = learning_rate;
    optimizer->momentum = momentum;
    optimizer->beta1 = 0.0;
    optimizer->beta2 = 0.0;
    optimizer->epsilon = 0.0;
    *out = lana_value_optimizer(optimizer);
    return LANA_OK;
}

static LanaError host_adam(LanaVM *vm, const Value *arguments, size_t argc, Value *out) {
    double learning_rate = 0.001;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
    if (argc > 4u) return LANA_ERR_TYPE;
    if (argc >= 1u) {
        if (arguments[0].type != VAL_NUMBER) return LANA_ERR_TYPE;
        learning_rate = arguments[0].as.number;
    }
    if (argc >= 2u) {
        if (arguments[1].type != VAL_NUMBER) return LANA_ERR_TYPE;
        beta1 = arguments[1].as.number;
    }
    if (argc >= 3u) {
        if (arguments[2].type != VAL_NUMBER) return LANA_ERR_TYPE;
        beta2 = arguments[2].as.number;
    }
    if (argc >= 4u) {
        if (arguments[3].type != VAL_NUMBER) return LANA_ERR_TYPE;
        epsilon = arguments[3].as.number;
    }
    if (!isfinite(learning_rate) || learning_rate <= 0.0 ||
        !isfinite(beta1) || beta1 < 0.0 || beta1 >= 1.0 ||
        !isfinite(beta2) || beta2 < 0.0 || beta2 >= 1.0 ||
        !isfinite(epsilon) || epsilon <= 0.0)
        return LANA_ERR_INVALID_PARAMETERS;
    LanaOptimizer *optimizer = lana_vm_alloc(vm, sizeof(*optimizer));
    if (optimizer == NULL) return LANA_ERR_OOM;
    optimizer->name = derivation_string(vm, "adam");
    if (optimizer->name == NULL) return LANA_ERR_OOM;
    optimizer->learning_rate = learning_rate;
    optimizer->momentum = 0.0;
    optimizer->beta1 = beta1;
    optimizer->beta2 = beta2;
    optimizer->epsilon = epsilon;
    *out = lana_value_optimizer(optimizer);
    return LANA_OK;
}

static LanaError host_train(LanaVM *vm, const Value *arguments, size_t argc,
                            uint32_t scratch_register, Value *out) {
    /* train(model, data, loss, optimizer, initial_params, [epochs], [batch_size]) */
    if (argc < 5u || argc > 7u) return LANA_ERR_TYPE;
    const Value *model = &arguments[0];
    const Value *data = &arguments[1];
    const Value *loss = &arguments[2];
    const Value *optimizer_value = &arguments[3];
    const Value *initial_params = &arguments[4];
    /* LIP-010: a live data root produces reactive parameters. Resolve the
     * current dataset and remember whether the data was reactive so the result
     * can be wired into the reactive DAG. */
    bool data_is_reactive = data->reactive != NULL;
    const Value *data_current = reactive_value(data);

    if (model->type != VAL_FUNCTION || loss->type != VAL_FUNCTION)
        return LANA_ERR_TYPE;
    if (model->as.function >= vm->chunk->function_count ||
        loss->as.function >= vm->chunk->function_count)
        return LANA_ERR_TYPE;
    if (vm->chunk->functions[model->as.function].arity != 2u ||
        vm->chunk->functions[loss->as.function].arity != 2u)
        return LANA_ERR_TYPE;

    if (optimizer_value->type != VAL_OPTIMIZER || optimizer_value->as.optimizer == NULL)
        return LANA_ERR_TYPE;
    const LanaOptimizer *optimizer = optimizer_value->as.optimizer;

    if (initial_params->type != VAL_TENSOR ||
        (initial_params->as.tensor->is_complex && !initial_params->as.tensor->is_state))
        return LANA_ERR_TYPE;

    double epochs = 10.0;
    double batch_size = 0.0; /* 0 = full dataset */
    if (argc >= 6u) {
        if (arguments[5].type != VAL_NUMBER) return LANA_ERR_TYPE;
        epochs = arguments[5].as.number;
    }
    if (argc >= 7u) {
        if (arguments[6].type != VAL_NUMBER) return LANA_ERR_TYPE;
        batch_size = arguments[6].as.number;
    }
    if (!isfinite(epochs) || epochs < 1.0 || floor(epochs) != epochs ||
        epochs > (double)SIZE_MAX)
        return LANA_ERR_INVALID_PARAMETERS;
    if (!isfinite(batch_size) || batch_size < 0.0 || floor(batch_size) != batch_size ||
        batch_size > (double)SIZE_MAX)
        return LANA_ERR_INVALID_PARAMETERS;

    /* Authorized execution: `train` requires the `train` capability (LIP-012). */
    if (!vm_has_named_capability(vm, "train")) return LANA_ERR_CAPABILITY;

    size_t dataset_size;
    if (data_current->type == VAL_ARRAY) {
        dataset_size = data_current->as.array->count;
    } else if (data_current->type == VAL_LAZY) {
        if (data_current->as.lazy.function >= vm->chunk->function_count ||
            vm->chunk->functions[data_current->as.lazy.function].arity != 1u)
            return LANA_ERR_TYPE;
        dataset_size = data_current->as.lazy.bound;
    } else {
        return LANA_ERR_TYPE;
    }
    if (dataset_size == 0u) return LANA_ERR_INVALID_PARAMETERS;

    size_t batch = (size_t)batch_size;
    if (batch == 0u || batch > dataset_size) batch = dataset_size;
    size_t epoch_count = (size_t)epochs;

    LanaTensor *params = tensor_copy_contiguous(vm, initial_params->as.tensor);
    if (params == NULL) return LANA_ERR_OOM;
    size_t param_count = tensor_element_count(params);
    size_t param_components = params->is_complex ? 2u : 1u;

    bool is_adam = strcmp(optimizer->name, "adam") == 0;
    LanaTensor *velocity = NULL;
    LanaTensor *m = NULL;
    LanaTensor *v = NULL;
    size_t adam_t = 0u;
    if (is_adam) {
        m = tensor_new(vm, params->ndim, params->shape, params->is_complex);
        v = tensor_new(vm, params->ndim, params->shape, params->is_complex);
        if (m == NULL || v == NULL) return LANA_ERR_OOM;
    } else if (optimizer->momentum != 0.0) {
        velocity = tensor_new(vm, params->ndim, params->shape, params->is_complex);
        if (velocity == NULL) return LANA_ERR_OOM;
    }

    LanaTensor *batch_grad = tensor_new(vm, params->ndim, params->shape, params->is_complex);
    if (batch_grad == NULL) return LANA_ERR_OOM;

    size_t total_steps = epoch_count * ((dataset_size + batch - 1u) / batch);
    LanaArray *steps = lana_vm_alloc(vm, sizeof(*steps));
    if (steps == NULL) return LANA_ERR_OOM;
    steps->count = 0u;
    steps->capacity = total_steps;
    steps->items = lana_vm_alloc(vm, total_steps * sizeof(*steps->items));
    if (steps->items == NULL && total_steps > 0u) return LANA_ERR_OOM;

    /* Root mutable locals so they survive function calls (which may collect). */
    Value params_root = lana_value_tensor(params);
    size_t root_base = lana_vm_root_push(vm, &params_root);
    Value velocity_root = lana_value_null();
    Value m_root = lana_value_null();
    Value v_root = lana_value_null();
    Value steps_root = lana_value_array(steps);
    Value batch_grad_root = lana_value_tensor(batch_grad);
    if (velocity != NULL) {
        velocity_root = lana_value_tensor(velocity);
        (void)lana_vm_root_push(vm, &velocity_root);
    }
    if (m != NULL) {
        m_root = lana_value_tensor(m);
        (void)lana_vm_root_push(vm, &m_root);
    }
    if (v != NULL) {
        v_root = lana_value_tensor(v);
        (void)lana_vm_root_push(vm, &v_root);
    }
    (void)lana_vm_root_push(vm, &steps_root);
    (void)lana_vm_root_push(vm, &batch_grad_root);

    LanaDerivation *params_deriv = NULL;

    for (size_t epoch = 0; epoch < epoch_count; ++epoch) {
        for (size_t batch_start = 0; batch_start < dataset_size; batch_start += batch) {
            size_t batch_end = batch_start + batch;
            if (batch_end > dataset_size) batch_end = dataset_size;
            size_t batch_actual = batch_end - batch_start;

            memset(batch_grad->data, 0, param_count * param_components * tensor_elem_bytes(batch_grad));

            for (size_t i = batch_start; i < batch_end; ++i) {
                Value pair;
                LanaError error;
                if (data_current->type == VAL_ARRAY) {
                    pair = data_current->as.array->items[i];
                } else {
                    Value index_value = lana_value_number((double)i);
                    error = run_function(vm, data_current->as.lazy.function, &index_value,
                                         scratch_register, &pair);
                    if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }
                }
                if (pair.type != VAL_ARRAY || pair.as.array->count != 2u) {
                    lana_vm_root_pop(vm, root_base); return LANA_ERR_TYPE;
                }
                const Value *x = &pair.as.array->items[0];
                const Value *target = &pair.as.array->items[1];

                LanaDerivation *leaf = record_derivation(vm, LANA_DERIVATION_OPERATION,
                    "input", NULL, 0u, "", 0u, LANA_EXACTNESS_EXACT, "autodiff",
                    LANA_DERIVATION_SUCCESS, "none");
                if (leaf == NULL) { lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM; }
                leaf->ad_a = params;
                Value params_with_deriv = lana_value_tensor(params);
                params_with_deriv.derivation = leaf;

                vm->ad_recording = true;
                Value y;
                error = run_function2(vm, model->as.function, &params_with_deriv, x,
                                      scratch_register, &y);
                if (error == LANA_OK)
                    error = run_function2(vm, loss->as.function, &y, target,
                                          scratch_register, &y);
                vm->ad_recording = false;
                if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }

                if (y.type != VAL_NUMBER) { lana_vm_root_pop(vm, root_base); return LANA_ERR_TYPE; }
                if (!isfinite(y.as.number)) { lana_vm_root_pop(vm, root_base); return LANA_ERR_INVALID_PARAMETERS; }

                LanaTensor *seed = tensor_new(vm, 0, NULL, false);
                if (seed == NULL) { lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM; }
                tensor_set_real(seed, 0, 1.0 / (double)batch_actual);
                if (y.derivation != NULL) {
                    error = ad_backward(vm, y.derivation, seed);
                    if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }
                }

                LanaTensor *grad = leaf->ad_grad;
                if (grad == NULL) {
                    grad = tensor_new(vm, params->ndim, params->shape, params->is_complex);
                    if (grad == NULL) { lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM; }
                }
                for (size_t k = 0; k < param_count * param_components; ++k) {
                    if (!isfinite(tensor_get_real(grad, k))) { lana_vm_root_pop(vm, root_base); return LANA_ERR_INVALID_PARAMETERS; }
                    tensor_set_real(batch_grad, k, tensor_get_real(batch_grad, k) + tensor_get_real(grad, k));
                }
            }

            /* Update params. */
            LanaTensor *new_params = tensor_new(vm, params->ndim, params->shape, params->is_complex);
            if (new_params == NULL) { lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM; }
            new_params->is_state = params->is_state;
            if (is_adam) {
                ++adam_t;
                double bc1 = 1.0 - pow(optimizer->beta1, (double)adam_t);
                double bc2 = 1.0 - pow(optimizer->beta2, (double)adam_t);
                for (size_t k = 0; k < param_count * param_components; ++k) {
                    double g = tensor_get_real(batch_grad, k);
                    tensor_set_real(m, k, optimizer->beta1 * tensor_get_real(m, k) + (1.0 - optimizer->beta1) * g);
                    tensor_set_real(v, k, optimizer->beta2 * tensor_get_real(v, k) + (1.0 - optimizer->beta2) * g * g);
                    double m_hat = tensor_get_real(m, k) / bc1;
                    double v_hat = tensor_get_real(v, k) / bc2;
                    tensor_set_real(new_params, k, tensor_get_real(params, k) -
                        optimizer->learning_rate * m_hat / (sqrt(v_hat) + optimizer->epsilon));
                }
            } else if (velocity != NULL) {
                for (size_t k = 0; k < param_count * param_components; ++k) {
                    tensor_set_real(velocity, k, optimizer->momentum * tensor_get_real(velocity, k) -
                        optimizer->learning_rate * tensor_get_real(batch_grad, k));
                    tensor_set_real(new_params, k, tensor_get_real(params, k) + tensor_get_real(velocity, k));
                }
            } else {
                for (size_t k = 0; k < param_count * param_components; ++k) {
                    tensor_set_real(new_params, k, tensor_get_real(params, k) -
                        optimizer->learning_rate * tensor_get_real(batch_grad, k));
                }
            }

            /* Attach provenance to the post-update params: a derivation node
             * whose inputs are the pre-update params and the batch gradient,
             * with the batch identity in the details string. */
            Value pre_value = lana_value_tensor(params);
            pre_value.derivation = params_deriv;
            LanaTensor *grad_snap = tensor_copy_contiguous(vm, batch_grad);
            if (grad_snap == NULL) { lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM; }
            Value grad_value = lana_value_tensor(grad_snap);
            const Value *inputs[2] = { &pre_value, &grad_value };
            char details[64];
            (void)snprintf(details, sizeof(details), "epoch=%zu batch=%zu",
                           epoch, batch_start / batch);
            LanaDerivation *step_deriv = record_derivation(vm, LANA_DERIVATION_OPERATION,
                "train_step", inputs, 2u, "", 0u, LANA_EXACTNESS_EXACT, details,
                LANA_DERIVATION_SUCCESS, "none");
            if (step_deriv == NULL) { lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM; }
            Value new_params_value = lana_value_tensor(new_params);
            new_params_value.derivation = step_deriv;

            /* Build the step map. */
            LanaMap *step_map;
            LanaError error = lana_map_new(vm, 5u, &step_map);
            if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }
            if ((error = map_put(vm, step_map, "epoch", lana_value_number((double)epoch))) != LANA_OK ||
                (error = map_put(vm, step_map, "batch", lana_value_number((double)(batch_start / batch)))) != LANA_OK ||
                (error = map_put(vm, step_map, "parameters", new_params_value)) != LANA_OK ||
                (error = map_put(vm, step_map, "gradient", grad_value)) != LANA_OK) {
                lana_vm_root_pop(vm, root_base); return error;
            }

            /* Optimizer state snapshot. */
            LanaMap *state_map;
            error = lana_map_new(vm, 3u, &state_map);
            if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }
            if (is_adam) {
                LanaTensor *m_snap = tensor_copy_contiguous(vm, m);
                LanaTensor *v_snap = tensor_copy_contiguous(vm, v);
                if (m_snap == NULL || v_snap == NULL) { lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM; }
                if ((error = map_put(vm, state_map, "m", lana_value_tensor(m_snap))) != LANA_OK ||
                    (error = map_put(vm, state_map, "v", lana_value_tensor(v_snap))) != LANA_OK ||
                    (error = map_put(vm, state_map, "t", lana_value_number((double)adam_t))) != LANA_OK) {
                    lana_vm_root_pop(vm, root_base); return error;
                }
            } else if (velocity != NULL) {
                LanaTensor *vel_snap = tensor_copy_contiguous(vm, velocity);
                if (vel_snap == NULL) { lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM; }
                if ((error = map_put(vm, state_map, "velocity", lana_value_tensor(vel_snap))) != LANA_OK) {
                    lana_vm_root_pop(vm, root_base); return error;
                }
            }
            if ((error = map_put(vm, step_map, "optimizer_state", lana_value_map(state_map))) != LANA_OK) {
                lana_vm_root_pop(vm, root_base); return error;
            }

            steps->items[steps->count++] = lana_value_map(step_map);

            params = new_params;
            params_root.as.tensor = new_params;
            params_deriv = step_deriv;
        }
    }

    lana_vm_root_pop(vm, root_base);

    LanaTrainingResult *result = lana_vm_alloc(vm, sizeof(*result));
    if (result == NULL) return LANA_ERR_OOM;
    result->params = params;
    result->steps = steps;
    result->model_function = model->as.function;
    result->loss_function = loss->as.function;
    result->optimizer = (LanaOptimizer *)optimizer;
    /* LIP-014: keep the resolved dataset and effective batch size so `resume`
     * can continue the run from any step. The dataset is immutable, so a
     * shallow copy (sharing the array/lazy payload) is sufficient. */
    Value *data_copy = lana_vm_alloc(vm, sizeof(*data_copy));
    if (data_copy == NULL) return LANA_ERR_OOM;
    *data_copy = *data_current;
    data_copy->reactive = NULL;
    data_copy->claim = NULL;
    data_copy->planned_effect = NULL;
    result->data = data_copy;
    result->batch_size = batch;
    Value result_value = lana_value_training_result(result);
    if (data_is_reactive) {
        /* Wire the training result into the reactive DAG: a TRAIN node whose
         * input is the data root. On `observe` the node is recomputed with one
         * incremental optimizer step over the new observation. */
        LanaReactive *node = lana_vm_alloc(vm, sizeof(*node));
        if (node == NULL) return LANA_ERR_OOM;
        memset(node, 0, sizeof(*node));
        node->id = vm->next_reactive_id++;
        node->kind = LANA_REACTIVE_TRAIN;
        node->revision = vm->revision;
        node->exactness = data->reactive->exactness;
        node->relationship = LANA_RELATION_EXACT;
        node->dependency_id = data->reactive->dependency_id;
        node->inputs[0] = data->reactive;
        data->reactive->is_training_data = true;
        LanaError error = allocate_plain_value(vm, &result_value, &node->current);
        if (error != LANA_OK) return error;
        result_value.reactive = node;
    }
    *out = result_value;
    return LANA_OK;
}

/* ===== LIP-010 incremental / online learning ===== */

/* Run `step_count` incremental optimizer steps over `data_current` (an array
 * of [x, target] pairs or a lazy dataset), extending the prior training
 * result's step history. The prior result is unchanged; a new
 * VAL_TRAINING_RESULT is returned. Shared by `update` (explicit) and the
 * reactive `observe` path. */
static LanaError incremental_train(LanaVM *vm, const LanaTrainingResult *prior,
                                   const Value *data_current, size_t dataset_size,
                                   size_t step_count, uint32_t scratch_register,
                                   Value *out) {
    if (prior == NULL || prior->params == NULL || prior->steps == NULL ||
        prior->optimizer == NULL)
        return LANA_ERR_TYPE;
    if (prior->model_function >= vm->chunk->function_count ||
        prior->loss_function >= vm->chunk->function_count)
        return LANA_ERR_TYPE;
    if (vm->chunk->functions[prior->model_function].arity != 2u ||
        vm->chunk->functions[prior->loss_function].arity != 2u)
        return LANA_ERR_TYPE;
    if (dataset_size == 0u) return LANA_ERR_INVALID_PARAMETERS;

    const LanaOptimizer *optimizer = prior->optimizer;
    bool is_adam = strcmp(optimizer->name, "adam") == 0;
    size_t prior_count = prior->steps->count;

    /* Recover the optimizer state and params provenance from the last step
     * map, so the incremental run resumes exactly where batch training left
     * off. */
    LanaTensor *m = NULL;
    LanaTensor *v = NULL;
    LanaTensor *velocity = NULL;
    size_t adam_t = 0u;
    LanaDerivation *params_deriv = NULL;
    if (prior_count > 0u) {
        Value last_step = prior->steps->items[prior_count - 1u];
        if (last_step.type != VAL_MAP || last_step.as.map == NULL)
            return LANA_ERR_TYPE;
        Value state_value;
        if (lana_map_get(last_step.as.map, "optimizer_state", &state_value) != LANA_OK ||
            state_value.type != VAL_MAP || state_value.as.map == NULL)
            return LANA_ERR_TYPE;
        if (is_adam) {
            Value m_value, v_value, t_value;
            if (lana_map_get(state_value.as.map, "m", &m_value) != LANA_OK ||
                lana_map_get(state_value.as.map, "v", &v_value) != LANA_OK ||
                lana_map_get(state_value.as.map, "t", &t_value) != LANA_OK)
                return LANA_ERR_TYPE;
            if (m_value.type != VAL_TENSOR || v_value.type != VAL_TENSOR ||
                t_value.type != VAL_NUMBER)
                return LANA_ERR_TYPE;
            m = tensor_copy_contiguous(vm, m_value.as.tensor);
            v = tensor_copy_contiguous(vm, v_value.as.tensor);
            if (m == NULL || v == NULL) return LANA_ERR_OOM;
            adam_t = (size_t)t_value.as.number;
        } else if (optimizer->momentum != 0.0) {
            Value vel_value;
            if (lana_map_get(state_value.as.map, "velocity", &vel_value) != LANA_OK ||
                vel_value.type != VAL_TENSOR)
                return LANA_ERR_TYPE;
            velocity = tensor_copy_contiguous(vm, vel_value.as.tensor);
            if (velocity == NULL) return LANA_ERR_OOM;
        }
        Value params_value;
        if (lana_map_get(last_step.as.map, "parameters", &params_value) != LANA_OK)
            return LANA_ERR_TYPE;
        params_deriv = params_value.derivation;
    } else if (is_adam) {
        m = tensor_new(vm, prior->params->ndim, prior->params->shape, false);
        v = tensor_new(vm, prior->params->ndim, prior->params->shape, false);
        if (m == NULL || v == NULL) return LANA_ERR_OOM;
    } else if (optimizer->momentum != 0.0) {
        velocity = tensor_new(vm, prior->params->ndim, prior->params->shape, false);
        if (velocity == NULL) return LANA_ERR_OOM;
    }

    LanaTensor *params = prior->params;
    size_t param_count = tensor_element_count(params);

    LanaTensor *batch_grad = tensor_new(vm, params->ndim, params->shape, false);
    if (batch_grad == NULL) return LANA_ERR_OOM;

    LanaArray *steps = lana_vm_alloc(vm, sizeof(*steps));
    if (steps == NULL) return LANA_ERR_OOM;
    steps->count = 0u;
    steps->capacity = prior_count + step_count;
    steps->items = lana_vm_alloc(vm, (prior_count + step_count) * sizeof(*steps->items));
    if (steps->items == NULL && prior_count + step_count > 0u) return LANA_ERR_OOM;
    for (size_t i = 0u; i < prior_count; ++i)
        steps->items[steps->count++] = prior->steps->items[i];

    /* Root mutable locals so they survive function calls (which may collect). */
    Value params_root = lana_value_tensor(params);
    size_t root_base = lana_vm_root_push(vm, &params_root);
    Value velocity_root = lana_value_null();
    Value m_root = lana_value_null();
    Value v_root = lana_value_null();
    Value steps_root = lana_value_array(steps);
    Value batch_grad_root = lana_value_tensor(batch_grad);
    Value data_root = *data_current;
    if (velocity != NULL) {
        velocity_root = lana_value_tensor(velocity);
        (void)lana_vm_root_push(vm, &velocity_root);
    }
    if (m != NULL) {
        m_root = lana_value_tensor(m);
        (void)lana_vm_root_push(vm, &m_root);
    }
    if (v != NULL) {
        v_root = lana_value_tensor(v);
        (void)lana_vm_root_push(vm, &v_root);
    }
    (void)lana_vm_root_push(vm, &steps_root);
    (void)lana_vm_root_push(vm, &batch_grad_root);
    (void)lana_vm_root_push(vm, &data_root);

    for (size_t step = 0u; step < step_count; ++step) {
        memset(batch_grad->data, 0, param_count * tensor_elem_bytes(batch_grad));

        for (size_t i = 0u; i < dataset_size; ++i) {
            Value pair;
            LanaError error;
            if (data_current->type == VAL_ARRAY) {
                pair = data_current->as.array->items[i];
            } else {
                Value index_value = lana_value_number((double)i);
                error = run_function(vm, data_current->as.lazy.function, &index_value,
                                     scratch_register, &pair);
                if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }
            }
            if (pair.type != VAL_ARRAY || pair.as.array->count != 2u) {
                lana_vm_root_pop(vm, root_base); return LANA_ERR_TYPE;
            }
            const Value *x = &pair.as.array->items[0];
            const Value *target = &pair.as.array->items[1];

            LanaDerivation *leaf = record_derivation(vm, LANA_DERIVATION_OPERATION,
                "input", NULL, 0u, "", 0u, LANA_EXACTNESS_EXACT, "autodiff",
                LANA_DERIVATION_SUCCESS, "none");
            if (leaf == NULL) { lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM; }
            leaf->ad_a = params;
            Value params_with_deriv = lana_value_tensor(params);
            params_with_deriv.derivation = leaf;

            vm->ad_recording = true;
            Value y;
            error = run_function2(vm, prior->model_function, &params_with_deriv, x,
                                  scratch_register, &y);
            if (error == LANA_OK)
                error = run_function2(vm, prior->loss_function, &y, target,
                                      scratch_register, &y);
            vm->ad_recording = false;
            if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }

            if (y.type != VAL_NUMBER) { lana_vm_root_pop(vm, root_base); return LANA_ERR_TYPE; }
            if (!isfinite(y.as.number)) { lana_vm_root_pop(vm, root_base); return LANA_ERR_INVALID_PARAMETERS; }

            LanaTensor *seed = tensor_new(vm, 0, NULL, false);
            if (seed == NULL) { lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM; }
            tensor_set_real(seed, 0, 1.0 / (double)dataset_size);
            if (y.derivation != NULL) {
                error = ad_backward(vm, y.derivation, seed);
                if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }
            }

            LanaTensor *grad = leaf->ad_grad;
            if (grad == NULL) {
                grad = tensor_new(vm, params->ndim, params->shape, false);
                if (grad == NULL) { lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM; }
            }
            for (size_t k = 0; k < param_count; ++k) {
                if (!isfinite(tensor_get_real(grad, k))) { lana_vm_root_pop(vm, root_base); return LANA_ERR_INVALID_PARAMETERS; }
                tensor_set_real(batch_grad, k, tensor_get_real(batch_grad, k) + tensor_get_real(grad, k));
            }
        }

        /* Update params. */
        LanaTensor *new_params = tensor_new(vm, params->ndim, params->shape, false);
        if (new_params == NULL) { lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM; }
        if (is_adam) {
            ++adam_t;
            double bc1 = 1.0 - pow(optimizer->beta1, (double)adam_t);
            double bc2 = 1.0 - pow(optimizer->beta2, (double)adam_t);
            for (size_t k = 0; k < param_count; ++k) {
                double g = tensor_get_real(batch_grad, k);
                tensor_set_real(m, k, optimizer->beta1 * tensor_get_real(m, k) + (1.0 - optimizer->beta1) * g);
                tensor_set_real(v, k, optimizer->beta2 * tensor_get_real(v, k) + (1.0 - optimizer->beta2) * g * g);
                double m_hat = tensor_get_real(m, k) / bc1;
                double v_hat = tensor_get_real(v, k) / bc2;
                tensor_set_real(new_params, k, tensor_get_real(params, k) -
                    optimizer->learning_rate * m_hat / (sqrt(v_hat) + optimizer->epsilon));
            }
        } else if (velocity != NULL) {
            for (size_t k = 0; k < param_count; ++k) {
                tensor_set_real(velocity, k, optimizer->momentum * tensor_get_real(velocity, k) -
                    optimizer->learning_rate * tensor_get_real(batch_grad, k));
                tensor_set_real(new_params, k, tensor_get_real(params, k) + tensor_get_real(velocity, k));
            }
        } else {
            for (size_t k = 0; k < param_count; ++k) {
                tensor_set_real(new_params, k, tensor_get_real(params, k) -
                    optimizer->learning_rate * tensor_get_real(batch_grad, k));
            }
        }

        /* Attach provenance to the post-update params. */
        Value pre_value = lana_value_tensor(params);
        pre_value.derivation = params_deriv;
        LanaTensor *grad_snap = tensor_copy_contiguous(vm, batch_grad);
        if (grad_snap == NULL) { lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM; }
        Value grad_value = lana_value_tensor(grad_snap);
        const Value *inputs[2] = { &pre_value, &grad_value };
        char details[64];
        (void)snprintf(details, sizeof(details), "epoch=%zu batch=%zu",
                       prior_count + step, (size_t)0u);
        LanaDerivation *step_deriv = record_derivation(vm, LANA_DERIVATION_OPERATION,
            "train_step", inputs, 2u, "", 0u, LANA_EXACTNESS_EXACT, details,
            LANA_DERIVATION_SUCCESS, "none");
        if (step_deriv == NULL) { lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM; }
        Value new_params_value = lana_value_tensor(new_params);
        new_params_value.derivation = step_deriv;

        /* Build the step map. */
        LanaMap *step_map;
        LanaError error = lana_map_new(vm, 5u, &step_map);
        if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }
        if ((error = map_put(vm, step_map, "epoch", lana_value_number((double)(prior_count + step)))) != LANA_OK ||
            (error = map_put(vm, step_map, "batch", lana_value_number(0.0))) != LANA_OK ||
            (error = map_put(vm, step_map, "parameters", new_params_value)) != LANA_OK ||
            (error = map_put(vm, step_map, "gradient", grad_value)) != LANA_OK) {
            lana_vm_root_pop(vm, root_base); return error;
        }

        /* Optimizer state snapshot. */
        LanaMap *state_map;
        error = lana_map_new(vm, 3u, &state_map);
        if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }
        if (is_adam) {
            LanaTensor *m_snap = tensor_copy_contiguous(vm, m);
            LanaTensor *v_snap = tensor_copy_contiguous(vm, v);
            if (m_snap == NULL || v_snap == NULL) { lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM; }
            if ((error = map_put(vm, state_map, "m", lana_value_tensor(m_snap))) != LANA_OK ||
                (error = map_put(vm, state_map, "v", lana_value_tensor(v_snap))) != LANA_OK ||
                (error = map_put(vm, state_map, "t", lana_value_number((double)adam_t))) != LANA_OK) {
                lana_vm_root_pop(vm, root_base); return error;
            }
        } else if (velocity != NULL) {
            LanaTensor *vel_snap = tensor_copy_contiguous(vm, velocity);
            if (vel_snap == NULL) { lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM; }
            if ((error = map_put(vm, state_map, "velocity", lana_value_tensor(vel_snap))) != LANA_OK) {
                lana_vm_root_pop(vm, root_base); return error;
            }
        }
        if ((error = map_put(vm, step_map, "optimizer_state", lana_value_map(state_map))) != LANA_OK) {
            lana_vm_root_pop(vm, root_base); return error;
        }

        steps->items[steps->count++] = lana_value_map(step_map);

        params = new_params;
        params_root.as.tensor = new_params;
        params_deriv = step_deriv;
    }

    lana_vm_root_pop(vm, root_base);

    LanaTrainingResult *result = lana_vm_alloc(vm, sizeof(*result));
    if (result == NULL) return LANA_ERR_OOM;
    result->params = params;
    result->steps = steps;
    result->model_function = prior->model_function;
    result->loss_function = prior->loss_function;
    result->optimizer = prior->optimizer;
    result->data = prior->data;
    result->batch_size = prior->batch_size;
    *out = lana_value_training_result(result);
    return LANA_OK;
}

/* update(model, new_data, steps) -> VAL_TRAINING_RESULT. The prior result is
 * unchanged; the returned result extends its step history with `steps`
 * incremental optimizer steps over `new_data`. */
static LanaError host_update(LanaVM *vm, const Value *arguments, size_t argc,
                             uint32_t scratch_register, Value *out) {
    if (argc != 3u) return LANA_ERR_TYPE;
    const Value *model = &arguments[0];
    const Value *new_data = &arguments[1];
    const Value *steps_value = &arguments[2];

    if (model->type != VAL_TRAINING_RESULT || model->as.training_result == NULL)
        return LANA_ERR_TYPE;
    const LanaTrainingResult *prior = model->as.training_result;

    if (steps_value->type != VAL_NUMBER) return LANA_ERR_TYPE;
    double steps = steps_value->as.number;
    if (!isfinite(steps) || steps < 1.0 || floor(steps) != steps ||
        steps > (double)SIZE_MAX)
        return LANA_ERR_INVALID_PARAMETERS;
    size_t step_count = (size_t)steps;

    /* Authorized execution: `update` is the non-reactive form of `train`. */
    if (!vm_has_named_capability(vm, "train")) return LANA_ERR_CAPABILITY;

    const Value *data_current = reactive_value(new_data);
    size_t dataset_size;
    if (data_current->type == VAL_ARRAY) {
        dataset_size = data_current->as.array->count;
    } else if (data_current->type == VAL_LAZY) {
        if (data_current->as.lazy.function >= vm->chunk->function_count ||
            vm->chunk->functions[data_current->as.lazy.function].arity != 1u)
            return LANA_ERR_TYPE;
        dataset_size = data_current->as.lazy.bound;
    } else {
        return LANA_ERR_TYPE;
    }
    if (dataset_size == 0u) return LANA_ERR_INVALID_PARAMETERS;

    return incremental_train(vm, prior, data_current, dataset_size, step_count,
                             scratch_register, out);
}

/* Recompute a LANA_REACTIVE_TRAIN node on `observe`: run one incremental
 * optimizer step over the new observation (a [x, target] point). */
static LanaError reactive_train_recompute(LanaVM *vm, LanaReactive *node,
                                          const Value *observation,
                                          uint32_t scratch_register, Value *out) {
    if (node->current == NULL || node->current->type != VAL_TRAINING_RESULT ||
        node->current->as.training_result == NULL)
        return LANA_ERR_TYPE;
    const LanaTrainingResult *prior = node->current->as.training_result;

    LanaArray *single = lana_vm_alloc(vm, sizeof(*single));
    if (single == NULL) return LANA_ERR_OOM;
    single->count = single->capacity = 1u;
    single->items = lana_vm_alloc(vm, sizeof(*single->items));
    if (single->items == NULL) return LANA_ERR_OOM;
    single->items[0] = *observation;
    Value single_value = lana_value_array(single);

    size_t root_base = lana_vm_root_push(vm, &single_value);
    LanaError error = incremental_train(vm, prior, &single_value, 1u, 1u,
                                        scratch_register, out);
    lana_vm_root_pop(vm, root_base);
    return error;
}

/* ===== LIP-014 whole-run reproducibility and resumability ===== */

/* resume(run, i) -> VAL_TRAINING_RESULT. The original run is unchanged; the
 * returned run continues from step `i` with the parameters and optimizer state
 * the original run had at that step, recomputing steps `i+1..` byte-identically
 * to the original. The training loop is deterministic (no RNG consumption), so
 * the continuation is byte-identical by construction given the same data, batch
 * size, and recovered state. */
static LanaError host_resume(LanaVM *vm, const Value *arguments, size_t argc,
                             uint32_t scratch_register, Value *out) {
    if (argc != 2u) return LANA_ERR_TYPE;
    const Value *run = &arguments[0];
    const Value *index_value = &arguments[1];

    if (run->type != VAL_TRAINING_RESULT || run->as.training_result == NULL)
        return LANA_ERR_TYPE;
    const LanaTrainingResult *prior = run->as.training_result;

    /* Step index must be a nonnegative integer. */
    if (index_value->type != VAL_NUMBER) return LANA_ERR_INVALID_PARAMETERS;
    double index = index_value->as.number;
    if (!isfinite(index) || index < 0.0 || floor(index) != index ||
        index > (double)SIZE_MAX)
        return LANA_ERR_INVALID_PARAMETERS;
    size_t step_index = (size_t)index;

    if (prior->params == NULL || prior->steps == NULL || prior->optimizer == NULL)
        return LANA_ERR_TYPE;
    if (prior->model_function >= vm->chunk->function_count ||
        prior->loss_function >= vm->chunk->function_count)
        return LANA_ERR_TYPE;
    if (vm->chunk->functions[prior->model_function].arity != 2u ||
        vm->chunk->functions[prior->loss_function].arity != 2u)
        return LANA_ERR_TYPE;

    /* Authorized execution: `resume` is the batch counterpart of `train`. */
    if (!vm_has_named_capability(vm, "train")) return LANA_ERR_CAPABILITY;

    size_t total_steps = prior->steps->count;
    if (step_index >= total_steps) return LANA_ERR_KEY;

    /* Resolve the stored dataset and effective batch size. */
    const Value *data_current = prior->data;
    if (data_current == NULL) return LANA_ERR_TYPE;
    size_t dataset_size;
    if (data_current->type == VAL_ARRAY) {
        dataset_size = data_current->as.array->count;
    } else if (data_current->type == VAL_LAZY) {
        if (data_current->as.lazy.function >= vm->chunk->function_count ||
            vm->chunk->functions[data_current->as.lazy.function].arity != 1u)
            return LANA_ERR_TYPE;
        dataset_size = data_current->as.lazy.bound;
    } else {
        return LANA_ERR_TYPE;
    }
    if (dataset_size == 0u) return LANA_ERR_INVALID_PARAMETERS;

    size_t batch = prior->batch_size;
    if (batch == 0u || batch > dataset_size) batch = dataset_size;
    size_t batches_per_epoch = (dataset_size + batch - 1u) / batch;

    const LanaOptimizer *optimizer = prior->optimizer;
    bool is_adam = strcmp(optimizer->name, "adam") == 0;

    /* Recover the parameters and optimizer state from the step map at
     * `step_index`, so the continuation resumes exactly where the original run
     * was at that step. */
    LanaTensor *m = NULL;
    LanaTensor *v = NULL;
    LanaTensor *velocity = NULL;
    size_t adam_t = 0u;
    LanaDerivation *params_deriv = NULL;
    LanaTensor *params = NULL;

    Value step_value = prior->steps->items[step_index];
    if (step_value.type != VAL_MAP || step_value.as.map == NULL)
        return LANA_ERR_TYPE;
    Value state_value;
    if (lana_map_get(step_value.as.map, "optimizer_state", &state_value) != LANA_OK ||
        state_value.type != VAL_MAP || state_value.as.map == NULL)
        return LANA_ERR_TYPE;
    if (is_adam) {
        Value m_value, v_value, t_value;
        if (lana_map_get(state_value.as.map, "m", &m_value) != LANA_OK ||
            lana_map_get(state_value.as.map, "v", &v_value) != LANA_OK ||
            lana_map_get(state_value.as.map, "t", &t_value) != LANA_OK)
            return LANA_ERR_TYPE;
        if (m_value.type != VAL_TENSOR || v_value.type != VAL_TENSOR ||
            t_value.type != VAL_NUMBER)
            return LANA_ERR_TYPE;
        m = tensor_copy_contiguous(vm, m_value.as.tensor);
        v = tensor_copy_contiguous(vm, v_value.as.tensor);
        if (m == NULL || v == NULL) return LANA_ERR_OOM;
        adam_t = (size_t)t_value.as.number;
    } else if (optimizer->momentum != 0.0) {
        Value vel_value;
        if (lana_map_get(state_value.as.map, "velocity", &vel_value) != LANA_OK ||
            vel_value.type != VAL_TENSOR)
            return LANA_ERR_TYPE;
        velocity = tensor_copy_contiguous(vm, vel_value.as.tensor);
        if (velocity == NULL) return LANA_ERR_OOM;
    }
    Value params_value;
    if (lana_map_get(step_value.as.map, "parameters", &params_value) != LANA_OK ||
        params_value.type != VAL_TENSOR)
        return LANA_ERR_TYPE;
    params = tensor_copy_contiguous(vm, params_value.as.tensor);
    if (params == NULL) return LANA_ERR_OOM;
    params_deriv = params_value.derivation;

    size_t param_count = tensor_element_count(params);

    LanaTensor *batch_grad = tensor_new(vm, params->ndim, params->shape, false);
    if (batch_grad == NULL) return LANA_ERR_OOM;

    /* New step history: copy steps 0..step_index, recompute step_index+1.. */
    LanaArray *steps = lana_vm_alloc(vm, sizeof(*steps));
    if (steps == NULL) return LANA_ERR_OOM;
    steps->count = 0u;
    steps->capacity = total_steps;
    steps->items = lana_vm_alloc(vm, total_steps * sizeof(*steps->items));
    if (steps->items == NULL && total_steps > 0u) return LANA_ERR_OOM;
    for (size_t i = 0u; i <= step_index; ++i)
        steps->items[steps->count++] = prior->steps->items[i];

    /* Root mutable locals so they survive function calls (which may collect). */
    Value params_root = lana_value_tensor(params);
    size_t root_base = lana_vm_root_push(vm, &params_root);
    Value velocity_root = lana_value_null();
    Value m_root = lana_value_null();
    Value v_root = lana_value_null();
    Value steps_root = lana_value_array(steps);
    Value batch_grad_root = lana_value_tensor(batch_grad);
    Value data_root = *data_current;
    if (velocity != NULL) {
        velocity_root = lana_value_tensor(velocity);
        (void)lana_vm_root_push(vm, &velocity_root);
    }
    if (m != NULL) {
        m_root = lana_value_tensor(m);
        (void)lana_vm_root_push(vm, &m_root);
    }
    if (v != NULL) {
        v_root = lana_value_tensor(v);
        (void)lana_vm_root_push(vm, &v_root);
    }
    (void)lana_vm_root_push(vm, &steps_root);
    (void)lana_vm_root_push(vm, &batch_grad_root);
    (void)lana_vm_root_push(vm, &data_root);

    for (size_t j = step_index + 1u; j < total_steps; ++j) {
        size_t epoch = j / batches_per_epoch;
        size_t batch_index = j % batches_per_epoch;
        size_t batch_start = batch_index * batch;
        size_t batch_end = batch_start + batch;
        if (batch_end > dataset_size) batch_end = dataset_size;
        size_t batch_actual = batch_end - batch_start;

        memset(batch_grad->data, 0, param_count * tensor_elem_bytes(batch_grad));

        for (size_t i = batch_start; i < batch_end; ++i) {
            Value pair;
            LanaError error;
            if (data_current->type == VAL_ARRAY) {
                pair = data_current->as.array->items[i];
            } else {
                Value index_value = lana_value_number((double)i);
                error = run_function(vm, data_current->as.lazy.function, &index_value,
                                     scratch_register, &pair);
                if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }
            }
            if (pair.type != VAL_ARRAY || pair.as.array->count != 2u) {
                lana_vm_root_pop(vm, root_base); return LANA_ERR_TYPE;
            }
            const Value *x = &pair.as.array->items[0];
            const Value *target = &pair.as.array->items[1];

            LanaDerivation *leaf = record_derivation(vm, LANA_DERIVATION_OPERATION,
                "input", NULL, 0u, "", 0u, LANA_EXACTNESS_EXACT, "autodiff",
                LANA_DERIVATION_SUCCESS, "none");
            if (leaf == NULL) { lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM; }
            leaf->ad_a = params;
            Value params_with_deriv = lana_value_tensor(params);
            params_with_deriv.derivation = leaf;

            vm->ad_recording = true;
            Value y;
            error = run_function2(vm, prior->model_function, &params_with_deriv, x,
                                  scratch_register, &y);
            if (error == LANA_OK)
                error = run_function2(vm, prior->loss_function, &y, target,
                                      scratch_register, &y);
            vm->ad_recording = false;
            if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }

            if (y.type != VAL_NUMBER) { lana_vm_root_pop(vm, root_base); return LANA_ERR_TYPE; }
            if (!isfinite(y.as.number)) { lana_vm_root_pop(vm, root_base); return LANA_ERR_INVALID_PARAMETERS; }

            LanaTensor *seed = tensor_new(vm, 0, NULL, false);
            if (seed == NULL) { lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM; }
            tensor_set_real(seed, 0, 1.0 / (double)batch_actual);
            if (y.derivation != NULL) {
                error = ad_backward(vm, y.derivation, seed);
                if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }
            }

            LanaTensor *grad = leaf->ad_grad;
            if (grad == NULL) {
                grad = tensor_new(vm, params->ndim, params->shape, false);
                if (grad == NULL) { lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM; }
            }
            for (size_t k = 0; k < param_count; ++k) {
                if (!isfinite(tensor_get_real(grad, k))) { lana_vm_root_pop(vm, root_base); return LANA_ERR_INVALID_PARAMETERS; }
                tensor_set_real(batch_grad, k, tensor_get_real(batch_grad, k) + tensor_get_real(grad, k));
            }
        }

        /* Update params. */
        LanaTensor *new_params = tensor_new(vm, params->ndim, params->shape, false);
        if (new_params == NULL) { lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM; }
        if (is_adam) {
            ++adam_t;
            double bc1 = 1.0 - pow(optimizer->beta1, (double)adam_t);
            double bc2 = 1.0 - pow(optimizer->beta2, (double)adam_t);
            for (size_t k = 0; k < param_count; ++k) {
                double g = tensor_get_real(batch_grad, k);
                tensor_set_real(m, k, optimizer->beta1 * tensor_get_real(m, k) + (1.0 - optimizer->beta1) * g);
                tensor_set_real(v, k, optimizer->beta2 * tensor_get_real(v, k) + (1.0 - optimizer->beta2) * g * g);
                double m_hat = tensor_get_real(m, k) / bc1;
                double v_hat = tensor_get_real(v, k) / bc2;
                tensor_set_real(new_params, k, tensor_get_real(params, k) -
                    optimizer->learning_rate * m_hat / (sqrt(v_hat) + optimizer->epsilon));
            }
        } else if (velocity != NULL) {
            for (size_t k = 0; k < param_count; ++k) {
                tensor_set_real(velocity, k, optimizer->momentum * tensor_get_real(velocity, k) -
                    optimizer->learning_rate * tensor_get_real(batch_grad, k));
                tensor_set_real(new_params, k, tensor_get_real(params, k) + tensor_get_real(velocity, k));
            }
        } else {
            for (size_t k = 0; k < param_count; ++k) {
                tensor_set_real(new_params, k, tensor_get_real(params, k) -
                    optimizer->learning_rate * tensor_get_real(batch_grad, k));
            }
        }

        /* Non-finite optimizer state after a step → INVALID_PARAMETERS. */
        if (is_adam) {
            for (size_t k = 0; k < param_count; ++k) {
                if (!isfinite(tensor_get_real(m, k)) || !isfinite(tensor_get_real(v, k))) {
                    lana_vm_root_pop(vm, root_base); return LANA_ERR_INVALID_PARAMETERS;
                }
            }
        } else if (velocity != NULL) {
            for (size_t k = 0; k < param_count; ++k) {
                if (!isfinite(tensor_get_real(velocity, k))) {
                    lana_vm_root_pop(vm, root_base); return LANA_ERR_INVALID_PARAMETERS;
                }
            }
        }

        /* Attach provenance to the post-update params. */
        Value pre_value = lana_value_tensor(params);
        pre_value.derivation = params_deriv;
        LanaTensor *grad_snap = tensor_copy_contiguous(vm, batch_grad);
        if (grad_snap == NULL) { lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM; }
        Value grad_value = lana_value_tensor(grad_snap);
        const Value *inputs[2] = { &pre_value, &grad_value };
        char details[64];
        (void)snprintf(details, sizeof(details), "epoch=%zu batch=%zu",
                       epoch, batch_index);
        LanaDerivation *step_deriv = record_derivation(vm, LANA_DERIVATION_OPERATION,
            "train_step", inputs, 2u, "", 0u, LANA_EXACTNESS_EXACT, details,
            LANA_DERIVATION_SUCCESS, "none");
        if (step_deriv == NULL) { lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM; }
        Value new_params_value = lana_value_tensor(new_params);
        new_params_value.derivation = step_deriv;

        /* Build the step map. */
        LanaMap *step_map;
        LanaError error = lana_map_new(vm, 5u, &step_map);
        if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }
        if ((error = map_put(vm, step_map, "epoch", lana_value_number((double)epoch))) != LANA_OK ||
            (error = map_put(vm, step_map, "batch", lana_value_number((double)batch_index))) != LANA_OK ||
            (error = map_put(vm, step_map, "parameters", new_params_value)) != LANA_OK ||
            (error = map_put(vm, step_map, "gradient", grad_value)) != LANA_OK) {
            lana_vm_root_pop(vm, root_base); return error;
        }

        /* Optimizer state snapshot. */
        LanaMap *state_map;
        error = lana_map_new(vm, 3u, &state_map);
        if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }
        if (is_adam) {
            LanaTensor *m_snap = tensor_copy_contiguous(vm, m);
            LanaTensor *v_snap = tensor_copy_contiguous(vm, v);
            if (m_snap == NULL || v_snap == NULL) { lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM; }
            if ((error = map_put(vm, state_map, "m", lana_value_tensor(m_snap))) != LANA_OK ||
                (error = map_put(vm, state_map, "v", lana_value_tensor(v_snap))) != LANA_OK ||
                (error = map_put(vm, state_map, "t", lana_value_number((double)adam_t))) != LANA_OK) {
                lana_vm_root_pop(vm, root_base); return error;
            }
        } else if (velocity != NULL) {
            LanaTensor *vel_snap = tensor_copy_contiguous(vm, velocity);
            if (vel_snap == NULL) { lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM; }
            if ((error = map_put(vm, state_map, "velocity", lana_value_tensor(vel_snap))) != LANA_OK) {
                lana_vm_root_pop(vm, root_base); return error;
            }
        }
        if ((error = map_put(vm, step_map, "optimizer_state", lana_value_map(state_map))) != LANA_OK) {
            lana_vm_root_pop(vm, root_base); return error;
        }

        steps->items[steps->count++] = lana_value_map(step_map);

        params = new_params;
        params_root.as.tensor = new_params;
        params_deriv = step_deriv;
    }

    lana_vm_root_pop(vm, root_base);

    LanaTrainingResult *result = lana_vm_alloc(vm, sizeof(*result));
    if (result == NULL) return LANA_ERR_OOM;
    result->params = params;
    result->steps = steps;
    result->model_function = prior->model_function;
    result->loss_function = prior->loss_function;
    result->optimizer = prior->optimizer;
    result->data = prior->data;
    result->batch_size = prior->batch_size;
    Value result_value = lana_value_training_result(result);

    /* Record the resumption point as a run-level derivation. */
    const Value *resume_inputs[1] = { run };
    char resume_details[64];
    (void)snprintf(resume_details, sizeof(resume_details), "step=%zu", step_index);
    LanaError error = attach_derivation(vm, &result_value, LANA_DERIVATION_OPERATION,
        "resume", resume_inputs, 1u, "", 0u, LANA_EXACTNESS_EXACT, resume_details);
    if (error != LANA_OK) return error;

    *out = result_value;
    return LANA_OK;
}

/* ===== LIP-009 Bayesian inference host calls ===== */

/* A standard-normal draw via the Box-Muller transform, mirroring the cached
 * append-parameter sampler. Deterministic given the VM RNG. */
static double gaussian_sample(LanaVM *vm) {
    for (;;) {
        double x = uniform_signed(vm);
        double y = uniform_signed(vm);
        double radius_squared = x * x + y * y;
        if (radius_squared <= 0.0 || radius_squared >= 1.0) continue;
        return x * sqrt(-2.0 * log(radius_squared) / radius_squared);
    }
}

/* A uniform draw in [0, 1). */
static double uniform01(LanaVM *vm) {
    return (double)lana_vm_random(vm) / 4294967296.0;
}

/* The Gaussian log-likelihood of `data` under `model(params)` with unit
 * variance: -0.5 * sum((model(params) - data)^2). The model is an arity-1
 * function returning a real tensor of the same shape as `data`. */
static LanaError infer_log_likelihood(LanaVM *vm, uint32_t model_fn,
                                      const LanaTensor *params, const LanaTensor *data,
                                      uint32_t scratch, double *out) {
    Value params_value = lana_value_tensor((LanaTensor *)params);
    Value pred;
    LanaError error = run_function(vm, model_fn, &params_value, scratch, &pred);
    if (error != LANA_OK) return error;
    if (pred.type != VAL_TENSOR || pred.as.tensor->is_complex) return LANA_ERR_TYPE;
    const LanaTensor *p = pred.as.tensor;
    if (!tensor_shape_equal(p, data)) return LANA_ERR_TYPE;
    size_t count = tensor_element_count(data);
    double sse = 0.0;
    for (size_t lin = 0; lin < count; ++lin) {
        size_t rem = lin, ip = p->offset, id = data->offset;
        for (size_t d = p->ndim; d-- > 0;) {
            ip += (p->shape[d] == 0 ? 0 : rem % p->shape[d]) * p->strides[d];
            id += (data->shape[d] == 0 ? 0 : rem % data->shape[d]) * data->strides[d];
            rem /= p->shape[d];
        }
        double diff = tensor_get_real(p, ip) - tensor_get_real(data, id);
        sse += diff * diff;
    }
    *out = -0.5 * sse;
    return LANA_OK;
}

/* Append a step map { step, parameters, log_likelihood } to `steps`. The
 * `parameters` value carries a derivation node tracing to the prior and the
 * observed data, mirroring LIP-006's per-step provenance. */
static LanaError infer_record_step(LanaVM *vm, LanaArray *steps, size_t step,
                                   const LanaTensor *params, double log_likelihood,
                                   const LanaTensor *prior, const LanaTensor *data) {
    LanaMap *map;
    LanaError error = lana_map_new(vm, 3u, &map);
    if (error != LANA_OK) return error;
    LanaTensor *snap = tensor_copy_contiguous(vm, params);
    if (snap == NULL) return LANA_ERR_OOM;
    Value prior_value = lana_value_tensor((LanaTensor *)prior);
    Value data_value = lana_value_tensor((LanaTensor *)data);
    const Value *inputs[2] = { &prior_value, &data_value };
    LanaDerivation *step_deriv = record_derivation(vm, LANA_DERIVATION_OPERATION,
        "infer_step", inputs, 2u, "", 0u, LANA_EXACTNESS_SAMPLE, "infer",
        LANA_DERIVATION_SUCCESS, "none");
    Value params_value = lana_value_tensor(snap);
    if (step_deriv != NULL) params_value.derivation = step_deriv;
    if ((error = map_put(vm, map, "step", lana_value_number((double)step))) != LANA_OK ||
        (error = map_put(vm, map, "parameters", params_value)) != LANA_OK ||
        (error = map_put(vm, map, "log_likelihood", lana_value_number(log_likelihood))) != LANA_OK)
        return error;
    steps->items[steps->count++] = lana_value_map(map);
    return LANA_OK;
}

/* mcmc(samples, burn_in) -> VAL_INFERENCE_ALGORITHM. */
static LanaError host_mcmc(LanaVM *vm, const Value *arguments, size_t argc, Value *out) {
    double samples = 10000.0;
    double burn_in = 1000.0;
    if (argc > 2u) return LANA_ERR_TYPE;
    if (argc >= 1u) {
        if (arguments[0].type != VAL_NUMBER) return LANA_ERR_TYPE;
        samples = arguments[0].as.number;
    }
    if (argc >= 2u) {
        if (arguments[1].type != VAL_NUMBER) return LANA_ERR_TYPE;
        burn_in = arguments[1].as.number;
    }
    if (!isfinite(samples) || samples < 1.0 || floor(samples) != samples ||
        samples > (double)SIZE_MAX)
        return LANA_ERR_INVALID_PARAMETERS;
    if (!isfinite(burn_in) || burn_in < 0.0 || floor(burn_in) != burn_in ||
        burn_in > (double)SIZE_MAX)
        return LANA_ERR_INVALID_PARAMETERS;
    if (burn_in >= samples) return LANA_ERR_INVALID_PARAMETERS;
    LanaInferenceAlgorithm *algorithm = lana_vm_alloc(vm, sizeof(*algorithm));
    if (algorithm == NULL) return LANA_ERR_OOM;
    algorithm->name = derivation_string(vm, "mcmc");
    if (algorithm->name == NULL) return LANA_ERR_OOM;
    algorithm->family = NULL;
    algorithm->samples = samples;
    algorithm->burn_in = burn_in;
    algorithm->iterations = 0.0;
    *out = lana_value_inference_algorithm(algorithm);
    return LANA_OK;
}

/* vi(family, iterations) -> VAL_INFERENCE_ALGORITHM. */
static LanaError host_vi(LanaVM *vm, const Value *arguments, size_t argc, Value *out) {
    const char *family = "gaussian";
    double iterations = 1000.0;
    if (argc > 2u) return LANA_ERR_TYPE;
    if (argc >= 1u) {
        if (arguments[0].type != VAL_STRING) return LANA_ERR_TYPE;
        family = arguments[0].as.string;
    }
    if (argc >= 2u) {
        if (arguments[1].type != VAL_NUMBER) return LANA_ERR_TYPE;
        iterations = arguments[1].as.number;
    }
    if (strcmp(family, "gaussian") != 0 && strcmp(family, "mean_field") != 0)
        return LANA_ERR_INVALID_PARAMETERS;
    if (!isfinite(iterations) || iterations < 1.0 || floor(iterations) != iterations ||
        iterations > (double)SIZE_MAX)
        return LANA_ERR_INVALID_PARAMETERS;
    LanaInferenceAlgorithm *algorithm = lana_vm_alloc(vm, sizeof(*algorithm));
    if (algorithm == NULL) return LANA_ERR_OOM;
    algorithm->name = derivation_string(vm, "vi");
    if (algorithm->name == NULL) return LANA_ERR_OOM;
    algorithm->family = derivation_string(vm, family);
    if (algorithm->family == NULL) return LANA_ERR_OOM;
    algorithm->samples = 0.0;
    algorithm->burn_in = 0.0;
    algorithm->iterations = iterations;
    *out = lana_value_inference_algorithm(algorithm);
    return LANA_OK;
}

/* smc(particles) -> VAL_INFERENCE_ALGORITHM. */
static LanaError host_smc(LanaVM *vm, const Value *arguments, size_t argc, Value *out) {
    double particles = 1000.0;
    if (argc > 1u) return LANA_ERR_TYPE;
    if (argc >= 1u) {
        if (arguments[0].type != VAL_NUMBER) return LANA_ERR_TYPE;
        particles = arguments[0].as.number;
    }
    if (!isfinite(particles) || particles < 1.0 || floor(particles) != particles ||
        particles > (double)SIZE_MAX)
        return LANA_ERR_INVALID_PARAMETERS;
    LanaInferenceAlgorithm *algorithm = lana_vm_alloc(vm, sizeof(*algorithm));
    if (algorithm == NULL) return LANA_ERR_OOM;
    algorithm->name = derivation_string(vm, "smc");
    if (algorithm->name == NULL) return LANA_ERR_OOM;
    algorithm->family = NULL;
    algorithm->samples = particles;
    algorithm->burn_in = 0.0;
    algorithm->iterations = 0.0;
    *out = lana_value_inference_algorithm(algorithm);
    return LANA_OK;
}

/* Metropolis-Hastings random walk. `params` is the current chain state (a
 * contiguous copy of the prior); `samples`/`burn_in` are the algorithm
 * hyperparameters. Fills `posterior` with the sample mean, variance, the
 * sample matrix, and the per-step provenance. */
static LanaError infer_mcmc(LanaVM *vm, uint32_t model_fn, const LanaTensor *prior,
                            const LanaTensor *data, size_t samples, size_t burn_in,
                            uint32_t scratch, LanaPosterior *posterior) {
    size_t param_count = tensor_element_count(prior);
    size_t kept = samples - burn_in;
    LanaTensor *params = tensor_copy_contiguous(vm, prior);
    if (params == NULL) return LANA_ERR_OOM;
    size_t sample_shape[2] = { kept, param_count };
    LanaTensor *sample_matrix = tensor_new(vm, 2, sample_shape, false);
    if (sample_matrix == NULL) return LANA_ERR_OOM;
    LanaArray *steps = lana_vm_alloc(vm, sizeof(*steps));
    if (steps == NULL) return LANA_ERR_OOM;
    steps->count = 0u;
    steps->capacity = samples;
    steps->items = lana_vm_alloc(vm, samples * sizeof(*steps->items));
    if (steps->items == NULL && samples > 0u) return LANA_ERR_OOM;

    Value params_root = lana_value_tensor(params);
    size_t root_base = lana_vm_root_push(vm, &params_root);
    Value sample_root = lana_value_tensor(sample_matrix);
    (void)lana_vm_root_push(vm, &sample_root);
    Value steps_root = lana_value_array(steps);
    (void)lana_vm_root_push(vm, &steps_root);

    double current_ll;
    LanaError error = infer_log_likelihood(vm, model_fn, params, data, scratch, &current_ll);
    if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }

    size_t kept_index = 0u;
    for (size_t i = 0u; i < samples; ++i) {
        error = consume_sampling_budget(vm);
        if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }
        LanaTensor *proposal = tensor_new(vm, prior->ndim, prior->shape, false);
        if (proposal == NULL) { lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM; }
        for (size_t k = 0u; k < param_count; ++k)
            tensor_set_real(proposal, k, tensor_get_real(params, k) + 0.1 * gaussian_sample(vm));
        double proposal_ll;
        error = infer_log_likelihood(vm, model_fn, proposal, data, scratch, &proposal_ll);
        if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }
        double log_alpha = proposal_ll - current_ll;
        bool accept = log_alpha >= 0.0 || uniform01(vm) < exp(log_alpha);
        if (accept) {
            params = proposal;
            params_root.as.tensor = proposal;
            current_ll = proposal_ll;
        }
        if (i >= burn_in) {
            for (size_t k = 0u; k < param_count; ++k)
                tensor_set_real(sample_matrix, kept_index * param_count + k, tensor_get_real(params, k));
            ++kept_index;
        }
        error = infer_record_step(vm, steps, i, params, current_ll, prior, data);
        if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }
    }

    LanaTensor *mean = tensor_new(vm, prior->ndim, prior->shape, false);
    LanaTensor *variance = tensor_new(vm, prior->ndim, prior->shape, false);
    if (mean == NULL || variance == NULL) { lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM; }
    for (size_t k = 0u; k < param_count; ++k) {
        double sum = 0.0;
        for (size_t s = 0u; s < kept; ++s)
            sum += tensor_get_real(sample_matrix, s * param_count + k);
        double m = sum / (double)kept;
        double var = 0.0;
        for (size_t s = 0u; s < kept; ++s) {
            double d = tensor_get_real(sample_matrix, s * param_count + k) - m;
            var += d * d;
        }
        tensor_set_real(mean, k, m);
        tensor_set_real(variance, k, var / (double)kept);
    }

    lana_vm_root_pop(vm, root_base);
    posterior->mean = mean;
    posterior->variance = variance;
    posterior->samples = sample_matrix;
    posterior->steps = steps;
    return LANA_OK;
}

/* Mean-field Gaussian variational inference via the reparameterization trick
 * and LIP-011 autodiff. `mu`/`log_sigma` are the variational parameters; the
 * ELBO is the loss. Fills `posterior` with the variational mean, variance, and
 * per-iteration provenance (no sample matrix). */
static LanaError infer_vi(LanaVM *vm, uint32_t model_fn, const LanaTensor *prior,
                          const LanaTensor *data, size_t iterations,
                          uint32_t scratch, LanaPosterior *posterior) {
    size_t param_count = tensor_element_count(prior);
    const double lr = 0.01;
    LanaTensor *mu = tensor_copy_contiguous(vm, prior);
    LanaTensor *log_sigma = tensor_new(vm, prior->ndim, prior->shape, false);
    LanaTensor *eps = tensor_new(vm, prior->ndim, prior->shape, false);
    LanaTensor *sigma = tensor_new(vm, prior->ndim, prior->shape, false);
    LanaTensor *params = tensor_new(vm, prior->ndim, prior->shape, false);
    if (mu == NULL || log_sigma == NULL || eps == NULL || sigma == NULL || params == NULL)
        return LANA_ERR_OOM;
    for (size_t k = 0u; k < param_count; ++k) tensor_set_real(log_sigma, k, log(0.1));
    LanaArray *steps = lana_vm_alloc(vm, sizeof(*steps));
    if (steps == NULL) return LANA_ERR_OOM;
    steps->count = 0u;
    steps->capacity = iterations;
    steps->items = lana_vm_alloc(vm, iterations * sizeof(*steps->items));
    if (steps->items == NULL && iterations > 0u) return LANA_ERR_OOM;

    Value mu_root = lana_value_tensor(mu);
    size_t root_base = lana_vm_root_push(vm, &mu_root);
    Value log_sigma_root = lana_value_tensor(log_sigma);
    (void)lana_vm_root_push(vm, &log_sigma_root);
    Value eps_root = lana_value_tensor(eps);
    (void)lana_vm_root_push(vm, &eps_root);
    Value sigma_root = lana_value_tensor(sigma);
    (void)lana_vm_root_push(vm, &sigma_root);
    Value params_root = lana_value_tensor(params);
    (void)lana_vm_root_push(vm, &params_root);
    Value steps_root = lana_value_array(steps);
    (void)lana_vm_root_push(vm, &steps_root);

    for (size_t i = 0u; i < iterations; ++i) {
        LanaError error = consume_sampling_budget(vm);
        if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }
        for (size_t k = 0u; k < param_count; ++k) {
            tensor_set_real(eps, k, gaussian_sample(vm));
            double s = exp(tensor_get_real(log_sigma, k));
            tensor_set_real(sigma, k, s);
            tensor_set_real(params, k, tensor_get_real(mu, k) + s * tensor_get_real(eps, k));
        }
        Value params_value = lana_value_tensor(params);
        Value pred;
        error = run_function(vm, model_fn, &params_value, scratch, &pred);
        if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }
        if (pred.type != VAL_TENSOR || pred.as.tensor->is_complex) {
            lana_vm_root_pop(vm, root_base); return LANA_ERR_TYPE;
        }
        if (!tensor_shape_equal(pred.as.tensor, data)) {
            lana_vm_root_pop(vm, root_base); return LANA_ERR_TYPE;
        }
        Value diff;
        error = tensor_elementwise(vm, pred.as.tensor, data, 1, &diff);
        if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }
        size_t diff_root = lana_vm_root_push(vm, &diff);
        Value model_value = lana_value_function(model_fn);
        Value grad_value;
        error = ad_vjp(vm, &model_value, &params_value, &diff, scratch, &grad_value);
        lana_vm_root_pop(vm, diff_root);
        if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }
        const LanaTensor *grad = grad_value.as.tensor;
        for (size_t k = 0u; k < param_count; ++k) {
            double g = tensor_get_real(grad, grad->offset + k);
            double d_mu = g + (tensor_get_real(mu, k) - tensor_get_real(prior, prior->offset + k));
            double d_log_sigma = g * tensor_get_real(sigma, k) * tensor_get_real(eps, k) +
                                 (tensor_get_real(sigma, k) * tensor_get_real(sigma, k) - 1.0);
            tensor_set_real(mu, k, tensor_get_real(mu, k) - lr * d_mu);
            tensor_set_real(log_sigma, k, tensor_get_real(log_sigma, k) - lr * d_log_sigma);
        }
        double ll;
        error = infer_log_likelihood(vm, model_fn, mu, data, scratch, &ll);
        if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }
        error = infer_record_step(vm, steps, i, mu, ll, prior, data);
        if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }
    }

    LanaTensor *mean = tensor_new(vm, prior->ndim, prior->shape, false);
    LanaTensor *variance = tensor_new(vm, prior->ndim, prior->shape, false);
    if (mean == NULL || variance == NULL) { lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM; }
    for (size_t k = 0u; k < param_count; ++k) {
        tensor_set_real(mean, k, tensor_get_real(mu, k));
        tensor_set_real(variance, k, exp(2.0 * tensor_get_real(log_sigma, k)));
    }

    lana_vm_root_pop(vm, root_base);
    posterior->mean = mean;
    posterior->variance = variance;
    posterior->samples = NULL;
    posterior->steps = steps;
    return LANA_OK;
}

/* Sequential Monte Carlo (particle filter) over a single observation. `P`
 * particles propagate with Gaussian noise, are weighted by the likelihood, and
 * are systematically resampled. Fills `posterior` with the weighted mean,
 * variance, the particle matrix, and per-step provenance. */
static LanaError infer_smc(LanaVM *vm, uint32_t model_fn, const LanaTensor *prior,
                           const LanaTensor *data, size_t particles,
                           uint32_t scratch, LanaPosterior *posterior) {
    size_t param_count = tensor_element_count(prior);
    size_t particle_shape[2] = { particles, param_count };
    LanaTensor *matrix = tensor_new(vm, 2, particle_shape, false);
    if (matrix == NULL) return LANA_ERR_OOM;
    for (size_t p = 0u; p < particles; ++p)
        for (size_t k = 0u; k < param_count; ++k)
            tensor_set_real(matrix, p * param_count + k, tensor_get_real(prior, prior->offset + k));
    double *weights = lana_vm_alloc(vm, particles * sizeof(*weights));
    if (weights == NULL) return LANA_ERR_OOM;
    LanaArray *steps = lana_vm_alloc(vm, sizeof(*steps));
    if (steps == NULL) return LANA_ERR_OOM;
    steps->count = 0u;
    steps->capacity = 1u;
    steps->items = lana_vm_alloc(vm, sizeof(*steps->items));
    if (steps->items == NULL) return LANA_ERR_OOM;

    Value matrix_root = lana_value_tensor(matrix);
    size_t root_base = lana_vm_root_push(vm, &matrix_root);
    Value steps_root = lana_value_array(steps);
    (void)lana_vm_root_push(vm, &steps_root);

    LanaError error = consume_sampling_budget(vm);
    if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }

    /* Propagate each particle with Gaussian noise. */
    for (size_t p = 0u; p < particles; ++p)
        for (size_t k = 0u; k < param_count; ++k)
            tensor_set_real(matrix, p * param_count + k, tensor_get_real(matrix, p * param_count + k) + 0.1 * gaussian_sample(vm));

    /* Weight by likelihood. */
    double weight_sum = 0.0;
    for (size_t p = 0u; p < particles; ++p) {
        LanaTensor *particle = tensor_new(vm, prior->ndim, prior->shape, false);
        if (particle == NULL) { lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM; }
        for (size_t k = 0u; k < param_count; ++k)
            tensor_set_real(particle, k, tensor_get_real(matrix, p * param_count + k));
        double ll;
        error = infer_log_likelihood(vm, model_fn, particle, data, scratch, &ll);
        if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }
        weights[p] = exp(ll);
        weight_sum += weights[p];
    }
    if (!(weight_sum > 0.0) || !isfinite(weight_sum)) {
        lana_vm_root_pop(vm, root_base); return LANA_ERR_INVALID_PARAMETERS;
    }
    for (size_t p = 0u; p < particles; ++p) weights[p] /= weight_sum;

    /* Systematic resampling. */
    LanaTensor *resampled = tensor_new(vm, 2, particle_shape, false);
    if (resampled == NULL) { lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM; }
    double u0 = uniform01(vm) / (double)particles;
    double cumulative = 0.0;
    size_t source = 0u;
    for (size_t p = 0u; p < particles; ++p) {
        double threshold = u0 + (double)p / (double)particles;
        while (cumulative < threshold && source < particles) {
            cumulative += weights[source];
            ++source;
        }
        size_t pick = source == 0u ? 0u : source - 1u;
        for (size_t k = 0u; k < param_count; ++k)
            tensor_set_real(resampled, p * param_count + k, tensor_get_real(matrix, pick * param_count + k));
    }

    /* Weighted mean and variance over the resampled particles. */
    LanaTensor *mean = tensor_new(vm, prior->ndim, prior->shape, false);
    LanaTensor *variance = tensor_new(vm, prior->ndim, prior->shape, false);
    if (mean == NULL || variance == NULL) { lana_vm_root_pop(vm, root_base); return LANA_ERR_OOM; }
    for (size_t k = 0u; k < param_count; ++k) {
        double sum = 0.0;
        for (size_t p = 0u; p < particles; ++p)
            sum += tensor_get_real(resampled, p * param_count + k);
        double m = sum / (double)particles;
        double var = 0.0;
        for (size_t p = 0u; p < particles; ++p) {
            double d = tensor_get_real(resampled, p * param_count + k) - m;
            var += d * d;
        }
        tensor_set_real(mean, k, m);
        tensor_set_real(variance, k, var / (double)particles);
    }

    /* Record a single provenance step carrying the weighted mean. */
    error = infer_record_step(vm, steps, 0u, mean, 0.0, prior, data);
    if (error != LANA_OK) { lana_vm_root_pop(vm, root_base); return error; }

    lana_vm_root_pop(vm, root_base);
    posterior->mean = mean;
    posterior->variance = variance;
    posterior->samples = resampled;
    posterior->steps = steps;
    return LANA_OK;
}

/* infer(prior, model, data, algorithm) -> VAL_POSTERIOR. */
static LanaError host_infer(LanaVM *vm, const Value *arguments, size_t argc,
                            uint32_t scratch_register, Value *out) {
    if (argc != 4u) return LANA_ERR_TYPE;
    const Value *prior = &arguments[0];
    const Value *model = &arguments[1];
    const Value *data = &arguments[2];
    const Value *algorithm_value = &arguments[3];

    if (model->type != VAL_FUNCTION) return LANA_ERR_TYPE;
    if (model->as.function >= vm->chunk->function_count) return LANA_ERR_TYPE;
    if (vm->chunk->functions[model->as.function].arity != 1u) return LANA_ERR_TYPE;
    if (prior->type != VAL_TENSOR || prior->as.tensor->is_complex) return LANA_ERR_TYPE;
    if (data->type != VAL_TENSOR || data->as.tensor->is_complex) return LANA_ERR_TYPE;
    if (algorithm_value->type != VAL_INFERENCE_ALGORITHM ||
        algorithm_value->as.inference_algorithm == NULL)
        return LANA_ERR_TYPE;
    const LanaInferenceAlgorithm *algorithm = algorithm_value->as.inference_algorithm;

    if (tensor_element_count(data->as.tensor) == 0u) return LANA_ERR_INVALID_PARAMETERS;

    if (!vm_has_named_capability(vm, "infer")) return LANA_ERR_CAPABILITY;

    LanaPosterior *posterior = lana_vm_alloc(vm, sizeof(*posterior));
    if (posterior == NULL) return LANA_ERR_OOM;
    posterior->mean = NULL;
    posterior->variance = NULL;
    posterior->samples = NULL;
    posterior->steps = NULL;
    posterior->seed = vm->root_seed;

    LanaError error;
    if (strcmp(algorithm->name, "mcmc") == 0) {
        error = infer_mcmc(vm, model->as.function, prior->as.tensor, data->as.tensor,
                           (size_t)algorithm->samples, (size_t)algorithm->burn_in,
                           scratch_register, posterior);
    } else if (strcmp(algorithm->name, "vi") == 0) {
        error = infer_vi(vm, model->as.function, prior->as.tensor, data->as.tensor,
                         (size_t)algorithm->iterations, scratch_register, posterior);
    } else if (strcmp(algorithm->name, "smc") == 0) {
        error = infer_smc(vm, model->as.function, prior->as.tensor, data->as.tensor,
                          (size_t)algorithm->samples, scratch_register, posterior);
    } else {
        return LANA_ERR_INVALID_PARAMETERS;
    }
    if (error != LANA_OK) return error;

    *out = lana_value_posterior(posterior);
    return LANA_OK;
}

LanaError lana_vm_run(LanaVM *vm) {
    LanaError verify;
    if (vm == NULL || vm->chunk == NULL) return LANA_ERR_FORMAT;
    verify = lana_chunk_verify(vm->chunk, &vm->error);
    if (verify != LANA_OK) return verify;
    while (vm->running) {
        LanaError error = vm_step(vm);
        if (error != LANA_OK) return error;
    }
    return LANA_OK;
}
