#ifndef SS_VM_H
#define SS_VM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>

#include "ss/bytecode.h"

typedef enum {
    SS_HISTORY_NONE = 0,
    SS_HISTORY_LATEST,
    SS_HISTORY_DURATION
} SSHistoryPolicy;

typedef struct {
    SSHistoryPolicy policy;
    double amount;
    SSStateValue *versions;
    size_t count;
    size_t capacity;
} SSHistory;

typedef struct {
    Value registers[SS_MAX_REGISTERS];
    SSHistory histories[SS_MAX_REGISTERS];
    size_t return_ip;
    uint32_t return_register;
    uint32_t function;
} SSFrame;

typedef struct SSAllocation {
    void *pointer;
    size_t size;
    struct SSAllocation *next;
} SSAllocation;

typedef struct VM VM;

struct SSTask {
    uint64_t id;
    uint64_t group_id;
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t completed_condition;
    VM *child;
    SSError status;
    SSErrorInfo error;
    Value result;
    bool completed;
    bool joined;
    bool thread_started;
    struct SSTask *next;
};

typedef SSError (*SSTransformFn)(VM *vm, SSStateValue *state,
                                 const Value *args, size_t argc);
typedef SSError (*SSMeasureFn)(VM *vm, SSStateValue *state, Value *out,
                               const Value *args, size_t argc);

struct VM {
    const SSChunk *chunk;
    size_t ip;
    bool running;
    bool trace;
    uint64_t instruction_limit;
    uint64_t instruction_count;
    uint64_t opcode_counts[OP_COUNT];
    uint64_t state_transition_count;
    uint64_t allocation_count;
    size_t memory_limit;
    size_t allocated_bytes;
    uint64_t rng_state;
    uint64_t rng_increment;
    uint64_t task_id;
    uint64_t next_task_id;
    uint64_t current_group_id;
    uint64_t next_group_id;
    uint64_t group_stack[SS_MAX_CALL_FRAMES];
    size_t group_depth;
    _Atomic bool cancelled;
    int program_argc;
    const char **program_argv;
    SSTask *tasks;
    SSFrame frames[SS_MAX_CALL_FRAMES];
    size_t frame_count;
    SSAllocation *allocations;
    SSErrorInfo error;
    Value result;
};

void ss_vm_init(VM *vm, const SSChunk *chunk);
void ss_vm_seed(VM *vm, uint64_t seed);
void ss_vm_set_program_args(VM *vm, int argc, const char **argv);
void ss_vm_free(VM *vm);
SSError ss_vm_run(VM *vm);
void *ss_vm_alloc(VM *vm, size_t size);
uint32_t ss_vm_random(VM *vm);

#endif
