#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>

#include "lana/effects.h"
#include "lana/policy.h"
#include "lana/store.h"
#include "lana/error.h"
#include "lana/data.h"
#include "lana/vm.h"

/* Mock executor: returns the error code in context, or a fixed value. */
static LanaError mock_executor(LanaVM *vm, const char *kind, const Value *payload,
                               void *context, Value *out) {
    (void)vm; (void)kind; (void)payload;
    if (context != NULL) {
        LanaError code = *(const LanaError *)context;
        if (code != LANA_OK) return code;
    }
    *out = lana_value_number(42.0);
    return LANA_OK;
}

static void cleanup_store(const char *path) {
    char file[256];
    (void)snprintf(file, sizeof(file), "%s/journal", path); (void)unlink(file);
    (void)snprintf(file, sizeof(file), "%s/manifest", path); (void)unlink(file);
    (void)rmdir(path);
}

/* Store a policy, evaluate it to AUTHORIZE, and store the decision. */
static void build_authorized_decision(LanaStore *store, LanaVM *vm, LanaDecision *decision) {
    LanaPolicyRule rule = {sizeof(rule), 1u, LANA_POLICY_PROBABILITY_AT_LEAST,
                           "probability", 0.8, NULL, "notify"};
    LanaPolicy policy = {sizeof(policy), 1u, "health", rule};
    LanaPolicyEvaluation evaluation = {sizeof(evaluation), 1u, 7u, "service-a", "health",
                                       3u, "evidence-1", "derivation-1", "explicit", 100u,
                                       "probability threshold", "measurements"};
    LanaMap *map; Value key = lana_value_number(0.95); Value input;
    assert(lana_map_new(vm, 1u, &map) == LANA_OK);
    assert(lana_map_set(vm, map, "probability", &key, false) == LANA_OK);
    input = lana_value_map(map);
    assert(lana_policy_store(store, vm, &policy) == LANA_OK);
    assert(lana_policy_evaluate(&policy, &input, &evaluation, decision) == LANA_OK);
    assert(decision->outcome == POLICY_OUTCOME_AUTHORIZE);
    assert(lana_policy_store_decision(store, vm, decision) == LANA_OK);
    assert(lana_store_commit(store, NULL) == LANA_OK);
}

static void test_effect_success(void) {
    printf("Testing effect attempt success...\n");
    char path[] = "/tmp/lana-effects-XXXXXX";
    LanaStoreOptions options = {sizeof(options), 1u, path, 1000u};
    LanaStore *store = NULL; LanaVM vm; LanaDecision decision = {0};
    LanaEffectAttempt attempt = {0}; Value plan, payload, out, stored, decision_after;
    char *decision_before = NULL;
    assert(mkdtemp(path) != NULL);
    assert(lana_store_open(&options, &store) == LANA_OK);
    lana_vm_init(&vm, NULL);
    build_authorized_decision(store, &vm, &decision);
    payload = lana_value_number(1.0);
    assert(lana_vm_planned_effect(&vm, "notify", &payload, &plan) == LANA_OK);
    {
        Value before;
        assert(lana_store_get(store, &vm, "decision/7", &before) == LANA_OK);
        assert(before.type == VAL_STRING);
        decision_before = strdup(before.as.string);
        assert(decision_before != NULL);
    }
    assert(lana_effect_execute_attempt(store, &vm, &decision, 1u, &plan, mock_executor,
                                       NULL, &out, &attempt) == LANA_OK);
    assert(attempt.status == EFFECT_STATUS_SUCCEEDED);
    assert(attempt.target != NULL && strcmp(attempt.target, "service-a") == 0);
    assert(out.type == VAL_NUMBER && out.as.number == 42.0);
    assert(lana_store_get(store, &vm, "effect-attempt/7/1", &stored) == LANA_OK);
    assert(stored.type == VAL_STRING);
    assert(strstr(stored.as.string, "\"status\":1") != NULL);
    assert(strstr(stored.as.string, "\"target\":\"service-a\"") != NULL);
    assert(lana_store_get(store, &vm, "decision/7", &decision_after) == LANA_OK);
    assert(decision_after.type == VAL_STRING);
    assert(strcmp(decision_before, decision_after.as.string) == 0);
    free(decision_before);
    lana_vm_free(&vm); assert(lana_store_close(store) == LANA_OK);
    cleanup_store(path);
    printf("Pass.\n");
}

static void test_effect_failure_distinction(void) {
    printf("Testing effect attempt failure distinction...\n");
    char path[] = "/tmp/lana-effects-XXXXXX";
    LanaStoreOptions options = {sizeof(options), 1u, path, 1000u};
    LanaStore *store = NULL; LanaVM vm; LanaDecision decision = {0};
    LanaEffectAttempt attempt = {0}; Value plan, payload, out;
    LanaError io_error = LANA_ERR_IO;
    LanaStoreHistory history = {0};
    assert(mkdtemp(path) != NULL);
    assert(lana_store_open(&options, &store) == LANA_OK);
    lana_vm_init(&vm, NULL);
    build_authorized_decision(store, &vm, &decision);
    payload = lana_value_number(1.0);
    assert(lana_vm_planned_effect(&vm, "notify", &payload, &plan) == LANA_OK);
    assert(lana_effect_execute_attempt(store, &vm, &decision, 2u, &plan, mock_executor,
                                       &io_error, &out, &attempt) == LANA_ERR_IO);
    assert(attempt.status == EFFECT_STATUS_FAILED);
    assert(strcmp(attempt.failure_reason, "LANA_ERR_IO") == 0);
    /* PENDING and FAILED records both present in history. */
    assert(lana_store_history(store, "effect-attempt/7/2", &history) == LANA_OK);
    assert(history.count == 2u);
    assert(history.records[0].value.type == VAL_STRING &&
           strstr(history.records[0].value.as.string, "\"status\":0") != NULL);
    assert(history.records[1].value.type == VAL_STRING &&
           strstr(history.records[1].value.as.string, "\"status\":2") != NULL);
    free(history.records);
    lana_vm_free(&vm); assert(lana_store_close(store) == LANA_OK);
    cleanup_store(path);
    printf("Pass.\n");
}

static void test_effect_resource_limit(void) {
    printf("Testing effect attempt resource limit...\n");
    char path[] = "/tmp/lana-effects-XXXXXX";
    LanaStoreOptions options = {sizeof(options), 1u, path, 1000u};
    LanaStore *store = NULL; LanaVM vm; LanaDecision decision = {0};
    LanaEffectAttempt attempt = {0}; Value plan, payload, out;
    LanaError limit_error = LANA_ERR_LIMIT;
    assert(mkdtemp(path) != NULL);
    assert(lana_store_open(&options, &store) == LANA_OK);
    lana_vm_init(&vm, NULL);
    build_authorized_decision(store, &vm, &decision);
    payload = lana_value_number(1.0);
    assert(lana_vm_planned_effect(&vm, "notify", &payload, &plan) == LANA_OK);
    assert(lana_effect_execute_attempt(store, &vm, &decision, 3u, &plan, mock_executor,
                                       &limit_error, &out, &attempt) == LANA_ERR_LIMIT);
    assert(attempt.status == EFFECT_STATUS_FAILED);
    assert(strcmp(attempt.failure_reason, "LANA_ERR_LIMIT") == 0);
    lana_vm_free(&vm); assert(lana_store_close(store) == LANA_OK);
    cleanup_store(path);
    printf("Pass.\n");
}

static void test_effect_malformed(void) {
    printf("Testing effect attempt malformed input...\n");
    char path[] = "/tmp/lana-effects-XXXXXX";
    LanaStoreOptions options = {sizeof(options), 1u, path, 1000u};
    LanaStore *store = NULL; LanaVM vm; LanaDecision decision = {0};
    LanaEffectAttempt attempt = {0}, bad = {0}; Value plan, payload, out;
    assert(mkdtemp(path) != NULL);
    assert(lana_store_open(&options, &store) == LANA_OK);
    lana_vm_init(&vm, NULL);
    build_authorized_decision(store, &vm, &decision);
    payload = lana_value_number(1.0);
    assert(lana_vm_planned_effect(&vm, "notify", &payload, &plan) == LANA_OK);
    /* NULL decision. */
    assert(lana_effect_execute_attempt(store, &vm, NULL, 4u, &plan, mock_executor,
                                       NULL, &out, &attempt) == LANA_ERR_CAPABILITY);
    /* Non-authorize decision. */
    {
        LanaDecision refuse = {0};
        refuse.struct_size = sizeof(refuse); refuse.schema_version = 1u;
        refuse.decision_id = 99u; refuse.outcome = POLICY_OUTCOME_REFUSE; refuse.effect = "notify";
        assert(lana_effect_execute_attempt(store, &vm, &refuse, 4u, &plan, mock_executor,
                                           NULL, &out, &attempt) == LANA_ERR_CAPABILITY);
    }
    /* Zero attempt id. */
    assert(lana_effect_execute_attempt(store, &vm, &decision, 0u, &plan, mock_executor,
                                       NULL, &out, &attempt) == LANA_ERR_CAPABILITY);
    /* NULL attempt to store. */
    assert(lana_effect_store_attempt(store, &vm, NULL) == LANA_ERR_SCHEMA);
    /* Small struct_size. */
    bad.struct_size = sizeof(bad) - 1u; bad.schema_version = 1u; bad.attempt_id = 1u;
    bad.decision_id = 7u; bad.effect_desc = "notify"; bad.failure_reason = ""; bad.status = EFFECT_STATUS_PENDING;
    assert(lana_effect_store_attempt(store, &vm, &bad) == LANA_ERR_SCHEMA);
    /* Wrong schema version. */
    bad.struct_size = sizeof(bad); bad.schema_version = 2u;
    assert(lana_effect_store_attempt(store, &vm, &bad) == LANA_ERR_SCHEMA);
    /* Missing effect description. */
    bad.schema_version = 1u; bad.effect_desc = NULL;
    assert(lana_effect_store_attempt(store, &vm, &bad) == LANA_ERR_SCHEMA);
    /* NULL args to planned execution. */
    assert(lana_effect_execute_planned(NULL, &plan, mock_executor, NULL, &out) == LANA_ERR_INVALID_STATE);
    lana_vm_free(&vm); assert(lana_store_close(store) == LANA_OK);
    cleanup_store(path);
    printf("Pass.\n");
}

static void test_effect_conflict(void) {
    printf("Testing effect attempt conflict...\n");
    char path[] = "/tmp/lana-effects-XXXXXX";
    LanaStoreOptions options = {sizeof(options), 1u, path, 1000u};
    LanaStore *store = NULL; LanaVM vm; LanaDecision decision = {0};
    LanaEffectAttempt attempt = {0}; Value plan, payload, out;
    assert(mkdtemp(path) != NULL);
    assert(lana_store_open(&options, &store) == LANA_OK);
    lana_vm_init(&vm, NULL);
    build_authorized_decision(store, &vm, &decision);
    payload = lana_value_number(1.0);
    assert(lana_vm_planned_effect(&vm, "notify", &payload, &plan) == LANA_OK);
    assert(lana_effect_execute_attempt(store, &vm, &decision, 5u, &plan, mock_executor,
                                       NULL, &out, &attempt) == LANA_OK);
    /* Same attempt id again must conflict. */
    assert(lana_effect_execute_attempt(store, &vm, &decision, 5u, &plan, mock_executor,
                                       NULL, &out, &attempt) == LANA_ERR_CONFLICT);
    lana_vm_free(&vm); assert(lana_store_close(store) == LANA_OK);
    cleanup_store(path);
    printf("Pass.\n");
}

int main(void) {
    test_effect_success();
    test_effect_failure_distinction();
    test_effect_resource_limit();
    test_effect_malformed();
    test_effect_conflict();
    return 0;
}
