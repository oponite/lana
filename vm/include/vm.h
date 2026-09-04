#ifndef LANA_VM_H
#define LANA_VM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>

#include "bytecode.h"
#include "gc.h"

typedef enum {
    LANA_HISTORY_NONE = 0,
    LANA_HISTORY_LATEST,
    LANA_HISTORY_DURATION
} LanaHistoryPolicy;

typedef enum {
    LANA_INSPECT_JSON = 0,
    LANA_INSPECT_DOT
} LanaInspectFormat;

typedef struct {
    LanaHistoryPolicy policy;
    double amount;
    LanaStateValue *versions;
    size_t count;
    size_t capacity;
} LanaHistory;

typedef struct {
    Value registers[LANA_MAX_REGISTERS];
    LanaHistory histories[LANA_MAX_REGISTERS];
    size_t return_ip;
    uint32_t return_register;
    uint32_t function;
} LanaFrame;

typedef struct LanaVM LanaVM;
typedef struct LanaScheduler LanaScheduler;
typedef struct LanaPathExecution LanaPathExecution;
typedef struct LanaSharedReference LanaSharedReference;
typedef bool (*LanaDebugHook)(LanaVM *vm, size_t instruction,
                             uint32_t source_line, void *context);
typedef LanaError (*LanaEffectExecutor)(LanaVM *vm, const char *kind,
                                       const Value *payload, void *context,
                                       Value *out);

struct LanaTask {
    uint64_t id;
    uint64_t group_id;
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t completed_condition;
    LanaVM *child;
    LanaError status;
    LanaErrorInfo error;
    Value result;
    bool completed;
    bool joined;
    bool thread_started;
    bool queued;
    LanaScheduler *scheduler;
    struct LanaTask *queue_next;
    struct LanaTask *all_next;
    struct LanaTask *next;
};

struct LanaVM {
    const LanaChunk *chunk;
    size_t ip;
    bool running;
    bool trace;
    bool debug_step;
    uint32_t debug_break_line;
    LanaDebugHook debug_hook;
    void *debug_context;
    uint64_t instruction_limit;
    uint64_t instruction_count;
    uint64_t opcode_counts[OP_COUNT];
    uint64_t state_transition_count;
    uint64_t allocation_count;
    size_t memory_limit;
    size_t allocated_bytes;
    uint64_t rng_state;
    uint64_t rng_increment;
    uint64_t root_seed;
    uint64_t lineage;
    uint64_t spawn_counter;
    uint64_t task_id;
    uint64_t next_task_id;
    uint64_t current_group_id;
    uint64_t next_group_id;
    uint64_t group_stack[LANA_MAX_CALL_FRAMES];
    size_t group_depth;
    _Atomic bool cancelled;
    int program_argc;
    const char **program_argv;
    LanaTask *tasks;
    LanaScheduler *scheduler;
    bool scheduler_owner;
    size_t configured_worker_count;
    size_t configured_task_limit;
    size_t path_limit;
    size_t active_path_count;
    uint64_t next_dependency_id;
    uint64_t next_reactive_id;
    uint64_t next_effect_id;
    LanaPathExecution *path_execution;
    LanaSharedReference *shared_references;
    size_t observation_count;
    uint64_t revision;
    uint64_t derivation_sequence;
    LanaFrame frames[LANA_MAX_CALL_FRAMES];
    size_t frame_count;
    LanaGC gc;
    LanaErrorInfo error;
    Value result;
};

void lana_vm_init(LanaVM *vm, const LanaChunk *chunk);
LanaVM *lana_vm_create(void);
void lana_vm_destroy(LanaVM *vm);
void lana_vm_seed(LanaVM *vm, uint64_t seed);
void lana_vm_set_program_args(LanaVM *vm, int argc, const char **argv);
LanaError lana_vm_set_worker_count(LanaVM *vm, size_t workers);
LanaError lana_vm_set_task_limit(LanaVM *vm, size_t tasks);
void lana_vm_free(LanaVM *vm);
LanaError lana_vm_run(LanaVM *vm);
void *lana_vm_alloc(LanaVM *vm, size_t size);
LanaError lana_vm_clone_value(LanaVM *destination, const Value *source,
                              Value *out);
LanaError lana_vm_clone_live_value(LanaVM *destination, const Value *source,
                                   Value *out);
bool lana_vm_value_equal(const Value *left, const Value *right);
bool lana_vm_collect(LanaVM *vm);
size_t lana_vm_root_push(LanaVM *vm, Value *value);
void lana_vm_root_pop(LanaVM *vm, size_t previous_count);
void lana_vm_write_barrier_value(LanaVM *vm, void *owner,
                                 const Value *value);
uint32_t lana_vm_random(LanaVM *vm);
LanaError lana_vm_state_dist_dirac(LanaVM *vm, const LanaStateValue *state, LanaStateDist **out);
LanaError lana_vm_state_dist_append(LanaVM *vm, const Value *left, const Value *right,
                                LanaStateDist **out);
LanaError lana_vm_state_dist_transform(LanaVM *vm, uint32_t transform_id,
                                   LanaStateDist *child, LanaStateDist **out);
LanaError lana_vm_state_dist_attenuate(LanaVM *vm, LanaStateDist *child, double factor,
                                   LanaStateDist **out);
LanaError lana_vm_state_dist_append_relationship(LanaVM *vm, const Value *left,
                                   const Value *right, uint32_t mode, double strength,
                                   LanaStateDist **out);
LanaError lana_vm_state_dist_expected_probability(const LanaStateDist *distribution,
                                              double *out);
LanaError lana_vm_state_dist_sample(LanaVM *vm, const LanaStateDist *distribution,
                                LanaStateValue *out);
LanaError lana_vm_state_dist_inspect(const LanaStateDist *distribution,
                                     LanaInspectFormat format, char **out);
LanaError lana_vm_joint_build(LanaVM *vm, const Value *values, size_t count,
                          const char *descriptor, LanaJointState **out);
LanaError lana_vm_joint_build_finite(LanaVM *vm, const char *names,
                                 const Value *rows, const double *weights,
                                 size_t row_count, size_t variable_count,
                                 LanaJointState **out);
LanaError lana_vm_joint_project(LanaVM *vm, const LanaJointState *source,
                            const char *names, LanaJointState **out);
LanaError lana_vm_joint_rename(LanaVM *vm, const LanaJointState *source,
                           const char *old_name, const char *new_name,
                           LanaJointState **out);
LanaError lana_vm_joint_condition(LanaVM *vm, const LanaJointState *source,
                              const char *name, const Value *evidence,
                              LanaJointState **out);
LanaError lana_vm_joint_observe(LanaVM *vm, const LanaJointState *source,
                            const char *name, const Value *evidence,
                            LanaJointState **out);
LanaError lana_vm_joint_sample(LanaVM *vm, const LanaJointState *source, Value *out);
LanaError lana_vm_joint_resolve(LanaVM *vm, const LanaJointState *source, Value *out);
LanaError lana_vm_possibility_build(LanaVM *vm, const Value *values, size_t count,
                                LanaPossibility **out);
LanaError lana_vm_information_sample(LanaVM *vm, const Value *source, Value *out);
LanaError lana_vm_information_resolve(LanaVM *vm, const Value *source, Value *out);
LanaError lana_vm_reactive_root(LanaVM *vm, const Value *source,
                                LanaDerivationExactness exactness, Value *out);
LanaError lana_vm_reactive_observe(LanaVM *vm, const Value *source,
                                   const Value *evidence, Value *out);
LanaError lana_vm_claim(LanaVM *vm, const Value *source, const char *proposition,
                        LanaDerivationExactness exactness, double tolerance,
                        bool source_valid, Value *out);
LanaError lana_vm_planned_effect(LanaVM *vm, const char *kind,
                                 const Value *payload, Value *out);
LanaError lana_vm_execute_planned_effect(LanaVM *vm, const Value *plan,
                                         LanaEffectExecutor executor,
                                         void *context, Value *out);
LanaError lana_vm_provenance_root(LanaVM *vm, const Value *source, const char *label,
                              uint32_t line, bool assumption, Value *out);
LanaError lana_vm_derivation(LanaVM *vm, const Value *source, Value *out);
LanaError lana_vm_explain(LanaVM *vm, const Value *source, Value *out);
LanaEvidenceStatus lana_derivation_status(const LanaDerivation *d);
const char *lana_evidence_status_name(LanaEvidenceStatus status);

#endif
