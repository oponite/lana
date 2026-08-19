#include "ss/vm.h"

#include <math.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static SSFrame *current_frame(VM *vm) { return &vm->frames[vm->frame_count - 1u]; }

static SSError clone_value(VM *destination, const Value *source, Value *out);
static SSError wait_task(VM *vm, SSTask *task, double timeout_seconds, Value *out);

static SSError vm_fail(VM *vm, SSError code, size_t ip, const SSInstruction *ins,
                       const char *message) {
    ss_error_set(&vm->error, code, ip, ins == NULL ? OP_NOP : ins->opcode,
                 ins == NULL ? 0u : ins->line, "%s", message);
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
    vm->result = ss_value_null();
    vm->next_task_id = 1u;
    vm->next_group_id = 1u;
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

static void cancel_task(SSTask *task) {
    if (task != NULL && task->child != NULL) atomic_store(&task->child->cancelled, true);
}

static void destroy_task(SSTask *task) {
    if (task == NULL) return;
    cancel_task(task);
    if (task->thread_started) (void)pthread_join(task->thread, NULL);
    if (task->child != NULL) { ss_vm_free(task->child); free(task->child); }
    (void)pthread_cond_destroy(&task->completed_condition);
    (void)pthread_mutex_destroy(&task->mutex);
    free(task);
}

void ss_vm_free(VM *vm) {
    size_t frame_index, register_index;
    SSAllocation *allocation;
    SSTask *task;
    if (vm == NULL) return;
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

static SSError clone_value(VM *destination, const Value *source, Value *out) {
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
        SSArray *array = ss_vm_alloc(destination, sizeof(*array));
        if (array == NULL) return SS_ERR_OOM;
        array->count = source->as.array->count;
        array->items = ss_vm_alloc(destination, array->count * sizeof(*array->items));
        if (array->items == NULL && array->count > 0u) return SS_ERR_OOM;
        for (index = 0; index < array->count; ++index) {
            error = clone_value(destination, &source->as.array->items[index], &array->items[index]);
            if (error != SS_OK) return error;
        }
        out->as.array = array;
    } else if (source->type == VAL_JOINT_STATE) {
        SSJointState *joint = ss_vm_alloc(destination, sizeof(*joint));
        if (joint == NULL) return SS_ERR_OOM;
        error = clone_state_value(destination, &source->as.joint->left, &joint->left);
        if (error == SS_OK) error = clone_state_value(destination, &source->as.joint->right, &joint->right);
        if (error != SS_OK) return error;
        out->as.joint = joint;
    } else if (source->type == VAL_TASK) {
        return SS_ERR_TYPE;
    }
    return SS_OK;
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

static SSError start_task(VM *parent, uint32_t function_index,
                          const Value *arguments, size_t argc, SSTask **out) {
    const SSFunction *function = &parent->chunk->functions[function_index];
    SSTask *task;
    size_t index;
    SSError error = SS_OK;
    if (argc != function->arity) return SS_ERR_TYPE;
    task = calloc(1u, sizeof(*task));
    if (task == NULL) return SS_ERR_OOM;
    task->child = malloc(sizeof(*task->child));
    if (task->child == NULL) { free(task); return SS_ERR_OOM; }
    if (pthread_mutex_init(&task->mutex, NULL) != 0 ||
        pthread_cond_init(&task->completed_condition, NULL) != 0) {
        free(task->child); free(task); return SS_ERR_TASK;
    }
    task->id = parent->next_task_id++;
    task->group_id = parent->current_group_id;
    task->result = ss_value_null();
    ss_vm_init(task->child, parent->chunk);
    task->child->ip = function->entry;
    task->child->frames[0].function = function_index;
    task->child->task_id = task->id;
    task->child->instruction_limit = parent->instruction_limit;
    task->child->memory_limit = parent->memory_limit;
    task->child->trace = parent->trace;
    ss_vm_seed(task->child, parent->rng_state ^ (task->id * UINT64_C(0x9e3779b97f4a7c15)));
    ss_vm_set_program_args(task->child, parent->program_argc, parent->program_argv);
    for (index = 0; index < argc && error == SS_OK; ++index)
        error = clone_value(task->child, &arguments[index], &task->child->frames[0].registers[index]);
    if (error != SS_OK) { destroy_task(task); return error; }
    task->next = parent->tasks;
    parent->tasks = task;
    if (pthread_create(&task->thread, NULL, run_task, task) != 0) {
        parent->tasks = task->next; destroy_task(task); return SS_ERR_TASK;
    }
    task->thread_started = true;
    *out = task;
    return SS_OK;
}

static SSError wait_task(VM *vm, SSTask *task, double timeout_seconds, Value *out) {
    int wait_result = 0;
    if (task == NULL) return SS_ERR_TASK;
    (void)pthread_mutex_lock(&task->mutex);
    if (timeout_seconds < 0.0) {
        while (!task->completed) wait_result = pthread_cond_wait(&task->completed_condition, &task->mutex);
    } else {
        struct timespec deadline;
        (void)timespec_get(&deadline, TIME_UTC);
        deadline.tv_sec += (time_t)timeout_seconds;
        deadline.tv_nsec += (long)((timeout_seconds - floor(timeout_seconds)) * 1000000000.0);
        if (deadline.tv_nsec >= 1000000000L) { ++deadline.tv_sec; deadline.tv_nsec -= 1000000000L; }
        while (!task->completed && wait_result == 0)
            wait_result = pthread_cond_timedwait(&task->completed_condition, &task->mutex, &deadline);
    }
    (void)pthread_mutex_unlock(&task->mutex);
    if (wait_result == ETIMEDOUT) return SS_ERR_TIMEOUT;
    if (wait_result != 0) return SS_ERR_TASK;
    if (task->thread_started) { (void)pthread_join(task->thread, NULL); task->thread_started = false; }
    if (task->status != SS_OK) {
        vm->error = task->error;
        return task->status;
    }
    if (!task->joined) {
        SSError error = clone_value(vm, &task->result, &task->result);
        if (error != SS_OK) return error;
        ss_vm_free(task->child); free(task->child); task->child = NULL; task->joined = true;
    }
    *out = task->result;
    return SS_OK;
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
    if (!ss_state_valid(&state.state)) return SS_ERR_INVALID_STATE;
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

static SSError transform_decay(VM *vm, SSStateValue *value, const Value *args, size_t argc) {
    double rate;
    (void)vm;
    if (argc != 1 || require_number_args(args, argc) != SS_OK) return SS_ERR_TYPE;
    rate = args[0].as.number;
    if (rate < 0.0 || rate > 1.0) return SS_ERR_TRANSFORM;
    value->state.p += rate * (0.5 - value->state.p);
    value->state.d *= 1.0 - rate;
    return ss_state_valid(&value->state) ? SS_OK : SS_ERR_INVALID_STATE;
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

static bool values_equal(const Value *left, const Value *right) {
    if (left->type != right->type) return false;
    switch (left->type) {
        case VAL_NULL: return true;
        case VAL_NUMBER: return left->as.number == right->as.number;
        case VAL_BOOL: return left->as.boolean == right->as.boolean;
        case VAL_STRING: return strcmp(left->as.string, right->as.string) == 0;
        case VAL_SAMPLE: return left->as.sample == right->as.sample;
        default: return left == right;
    }
}

static void trace_instruction(VM *vm, size_t ip, const SSInstruction *ins) {
    SSFrame *frame = current_frame(vm);
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
                frame->registers[ins->a] = frame->registers[ins->b]; break;
            case OP_STATE_NEW: {
                const Value *p = &vm->chunk->constants[ins->b];
                const Value *d = &vm->chunk->constants[ins->c];
                SSState state;
                if (p->type != VAL_NUMBER || d->type != VAL_NUMBER) error = SS_ERR_TYPE;
                else error = ss_state_make(p->as.number, d->as.number, &state);
                if (error == SS_OK) error = store_state(vm, ins->a, ss_value_state(state).as.state);
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
                if (frame->registers[ins->a].type != VAL_STATE || frame->registers[ins->b].type != VAL_STATE) { error = SS_ERR_TYPE; break; }
                joint = ss_vm_alloc(vm, sizeof(*joint));
                if (joint == NULL) { error = SS_ERR_OOM; break; }
                joint->left = frame->registers[ins->a].as.state; joint->right = frame->registers[ins->b].as.state;
                frame->registers[ins->c].type = VAL_JOINT_STATE; frame->registers[ins->c].as.joint = joint;
                break;
            }
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
            case OP_GET_FIELD: {
                const Value *source = &frame->registers[ins->a];
                if (source->type == VAL_STATE && ins->c <= 1u)
                    frame->registers[ins->b] = ss_value_number(ins->c == 0u ? source->as.state.state.p : source->as.state.state.d);
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
                if (left->type != VAL_NUMBER || right->type != VAL_NUMBER) { error = SS_ERR_TYPE; break; }
                if (ins->imm == SS_BINARY_ADD) frame->registers[ins->c] = ss_value_number(left->as.number + right->as.number);
                else if (ins->imm == SS_BINARY_SUBTRACT) frame->registers[ins->c] = ss_value_number(left->as.number - right->as.number);
                else if (ins->imm == SS_BINARY_MULTIPLY) frame->registers[ins->c] = ss_value_number(left->as.number * right->as.number);
                else if (ins->imm == SS_BINARY_DIVIDE && right->as.number != 0.0) frame->registers[ins->c] = ss_value_number(left->as.number / right->as.number);
                else error = SS_ERR_TYPE;
                break;
            }
            case OP_UNARY:
                if (frame->registers[ins->a].type == VAL_NUMBER && ins->imm == 0u) frame->registers[ins->b] = ss_value_number(-frame->registers[ins->a].as.number);
                else if (frame->registers[ins->a].type == VAL_BOOL && ins->imm == 1u) frame->registers[ins->b] = ss_value_bool(!frame->registers[ins->a].as.boolean);
                else error = SS_ERR_TYPE;
                break;
            case OP_COMPARE: {
                const Value *left = &frame->registers[ins->a], *right = &frame->registers[ins->b];
                bool result = false;
                if (ins->imm == SS_COMPARE_EQUAL || ins->imm == SS_COMPARE_NOT_EQUAL) {
                    result = values_equal(left, right); if (ins->imm == SS_COMPARE_NOT_EQUAL) result = !result;
                } else if (left->type == VAL_NUMBER && right->type == VAL_NUMBER) {
                    if (ins->imm == SS_COMPARE_LESS) result = left->as.number < right->as.number;
                    else if (ins->imm == SS_COMPARE_LESS_EQUAL) result = left->as.number <= right->as.number;
                    else if (ins->imm == SS_COMPARE_GREATER) result = left->as.number > right->as.number;
                    else if (ins->imm == SS_COMPARE_GREATER_EQUAL) result = left->as.number >= right->as.number;
                    else error = SS_ERR_TYPE;
                } else error = SS_ERR_TYPE;
                if (error == SS_OK) frame->registers[ins->c] = ss_value_bool(result);
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
                array->count = ins->c; array->items = ss_vm_alloc(vm, ins->c * sizeof(*array->items));
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
                for (index = 0; index < ins->imm; ++index) callee->registers[index] = frame->registers[ins->c + index];
                vm->ip = function->entry; break;
            }
            case OP_RETURN: {
                Value returned = frame->registers[ins->a];
                if (vm->frame_count == 1u) { vm->result = returned; vm->running = false; }
                else { size_t return_ip = frame->return_ip; uint32_t destination = frame->return_register; --vm->frame_count; current_frame(vm)->registers[destination] = returned; vm->ip = return_ip; }
                break;
            }
            case OP_PRINT: ss_value_print(&frame->registers[ins->a]); (void)printf("\n"); break;
            case OP_HALT: vm->running = false; break;
            case OP_COUNT: error = SS_ERR_OPCODE; break;
        }
        if (error != SS_OK) return vm_fail(vm, error, instruction_ip, ins, ss_error_name(error));
    }
    return SS_OK;
}
