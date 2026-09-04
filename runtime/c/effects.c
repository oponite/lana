#include "effects.h"
#include "store.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static bool json_safe(const char *text) {
    size_t index;
    if (text == NULL) return true;
    for (index = 0u; text[index] != '\0'; ++index) if ((unsigned char)text[index] < 0x20u || text[index] == '\\' || text[index] == '"') return false;
    return true;
}

static LanaError store_attempt(LanaStore *store, LanaVM *vm,
                               const LanaEffectAttempt *attempt, bool allow_overwrite) {
    char key[64], json[2048]; Value existing, value; LanaError error;
    if (store == NULL || vm == NULL || attempt == NULL || attempt->struct_size < sizeof(*attempt) ||
        attempt->schema_version != 1u || attempt->attempt_id == 0u || attempt->decision_id == 0u ||
        attempt->effect_desc == NULL || attempt->failure_reason == NULL || attempt->status > EFFECT_STATUS_CANCELLED)
        return LANA_ERR_SCHEMA;
    if (!json_safe(attempt->effect_desc) || !json_safe(attempt->target) || !json_safe(attempt->failure_reason))
        return LANA_ERR_SCHEMA;
    (void)snprintf(key, sizeof(key), "effect-attempt/%llu/%llu",
                   (unsigned long long)attempt->decision_id,
                   (unsigned long long)attempt->attempt_id);
    if (!allow_overwrite) {
        error = lana_store_get(store, vm, key, &existing);
        if (error == LANA_OK) return LANA_ERR_CONFLICT;
        if (error != LANA_ERR_NOT_FOUND) return error;
    }
    (void)snprintf(json, sizeof(json),
                   "{\"attempt_id\":%llu,\"decision_id\":%llu,\"effect\":\"%s\",\"target\":\"%s\",\"end_time\":%llu,\"failure_reason\":\"%s\",\"schema\":1,\"start_time\":%llu,\"status\":%u}",
                   (unsigned long long)attempt->attempt_id, (unsigned long long)attempt->decision_id,
                   attempt->effect_desc, attempt->target == NULL ? "" : attempt->target,
                   (unsigned long long)attempt->end_time, attempt->failure_reason,
                   (unsigned long long)attempt->start_time, (unsigned)attempt->status);
    if (strlen(json) >= sizeof(json) - 1u) return LANA_ERR_LIMIT;
    value = lana_value_string(json); return lana_store_put(store, key, value);
}

LanaError lana_effect_execute_planned(LanaVM *vm, const Value *plan,
                                      LanaEffectExecutor executor, void *context,
                                      Value *out) {
    if (vm == NULL || plan == NULL || executor == NULL || out == NULL)
        return LANA_ERR_INVALID_STATE;
    return lana_vm_execute_planned_effect(vm, plan, executor, context, out);
}

LanaError lana_effect_store_attempt(LanaStore *store, LanaVM *vm,
                                    const LanaEffectAttempt *attempt) {
    return store_attempt(store, vm, attempt, false);
}

LanaError lana_effect_execute_attempt(LanaStore *store, LanaVM *vm,
                                      const LanaDecision *decision, uint64_t attempt_id,
                                      const Value *plan, LanaEffectExecutor executor,
                                      void *context, Value *out,
                                      LanaEffectAttempt *out_attempt) {
    LanaEffectAttempt attempt = {0}, pending = {0}; time_t started, ended; LanaError error, execution_error;
    if (decision == NULL || decision->outcome != POLICY_OUTCOME_AUTHORIZE || decision->effect == NULL || attempt_id == 0u)
        return LANA_ERR_CAPABILITY;
    started = time(NULL); if (started < 0) return LANA_ERR_IO;
    pending.struct_size = sizeof(pending); pending.schema_version = 1u; pending.attempt_id = attempt_id;
    pending.decision_id = decision->decision_id; pending.effect_desc = decision->effect;
    pending.target = decision->target;
    pending.status = EFFECT_STATUS_PENDING; pending.start_time = (uint64_t)started; pending.failure_reason = "";
    if ((error = store_attempt(store, vm, &pending, false)) != LANA_OK ||
        (error = lana_store_commit(store, NULL)) != LANA_OK) return error;
    execution_error = lana_effect_execute_planned(vm, plan, executor, context, out);
    ended = time(NULL); if (ended < 0) return LANA_ERR_IO;
    attempt.struct_size = sizeof(attempt); attempt.schema_version = 1u; attempt.attempt_id = attempt_id;
    attempt.decision_id = decision->decision_id; attempt.effect_desc = decision->effect;
    attempt.target = decision->target;
    attempt.status = execution_error == LANA_OK ? EFFECT_STATUS_SUCCEEDED : EFFECT_STATUS_FAILED;
    attempt.start_time = (uint64_t)started; attempt.end_time = (uint64_t)ended;
    attempt.failure_reason = execution_error == LANA_OK ? "" : lana_error_name(execution_error);
    if (out_attempt != NULL) *out_attempt = attempt;
    if ((error = store_attempt(store, vm, &attempt, true)) != LANA_OK) return error;
    if ((error = lana_store_commit(store, NULL)) != LANA_OK) return error;
    return execution_error;
}
