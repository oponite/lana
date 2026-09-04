#ifndef LANA_EFFECTS_H
#define LANA_EFFECTS_H

#include <stdint.h>
#include "policy.h"
#include "vm.h"

typedef enum {
    EFFECT_STATUS_PENDING = 0,
    EFFECT_STATUS_SUCCEEDED,
    EFFECT_STATUS_FAILED,
    EFFECT_STATUS_CANCELLED
} EffectStatus;

typedef struct {
    size_t struct_size;
    uint32_t schema_version;
    uint64_t attempt_id;
    uint64_t decision_id;
    const char *effect_desc;
    const char *target;
    EffectStatus status;
    uint64_t start_time;
    uint64_t end_time;
    const char *failure_reason;
} LanaEffectAttempt;

/*
 * Effects execute only through the capability and planned-effect boundary:
 * lana_effect_execute_attempt stores a PENDING attempt, runs the planned
 * effect through the caller-supplied executor, and records the terminal
 * status. There is no direct execution from a decision ID.
 */
LanaError lana_effect_execute_planned(LanaVM *vm, const Value *plan,
                                      LanaEffectExecutor executor, void *context,
                                      Value *out);
LanaError lana_effect_store_attempt(LanaStore *store, LanaVM *vm,
                                    const LanaEffectAttempt *attempt);
LanaError lana_effect_execute_attempt(LanaStore *store, LanaVM *vm,
                                      const LanaDecision *decision, uint64_t attempt_id,
                                      const Value *plan, LanaEffectExecutor executor,
                                      void *context, Value *out,
                                      LanaEffectAttempt *out_attempt);
#endif
