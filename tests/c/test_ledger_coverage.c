#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>

#include "lana/ledger.h"
#include "lana/effects.h"
#include "lana/policy.h"
#include "lana/store.h"
#include "lana/error.h"
#include "lana/data.h"
#include "lana/vm.h"

static LanaError mock_executor(LanaVM *vm, const char *kind, const Value *payload,
                               void *context, Value *out) {
    (void)vm; (void)kind; (void)payload; (void)context;
    *out = lana_value_number(42.0);
    return LANA_OK;
}

static void cleanup_store(const char *path) {
    char file[256];
    (void)snprintf(file, sizeof(file), "%s/journal", path); (void)unlink(file);
    (void)snprintf(file, sizeof(file), "%s/manifest", path); (void)unlink(file);
    (void)rmdir(path);
}

static void test_coverage_success(void) {
    printf("Testing ledger coverage success...\n");
    char path[] = "/tmp/lana-ledger-cov-XXXXXX";
    LanaStoreOptions options = {sizeof(options), 1u, path, 1000u};
    LanaStore *store = NULL; LanaLedger *ledger = NULL; LanaVM vm; LanaEvent event;
    LanaEventInput input = {sizeof(input), 1u, "entity-a", "actor1", "action1", "test", 10u, 0u};
    const char *targets[] = {"entity-a", "entity-b", "entity-d"};
    LanaLedgerCoverageQuery query = {sizeof(query), 1u, NULL, NULL, NULL, 0u, 0u, targets, 3u};
    LanaLedgerCoverageEntry *entries = NULL; size_t count = 0u;
    assert(mkdtemp(path) != NULL);
    assert(lana_store_open(&options, &store) == LANA_OK);
    assert(lana_ledger_open(store, &ledger) == LANA_OK);
    lana_vm_init(&vm, NULL);
    input.entity = "entity-a"; assert(lana_ledger_append(ledger, &input, &event) == LANA_OK);
    input.entity = "entity-b"; assert(lana_ledger_append(ledger, &input, &event) == LANA_OK);
    input.entity = "entity-c"; assert(lana_ledger_append(ledger, &input, &event) == LANA_OK);
    assert(lana_ledger_query_coverage(ledger, &vm, &query, &entries, &count) == LANA_OK);
    assert(count == 3u);
    assert(strcmp(entries[0].entity, "entity-a") == 0 && entries[0].matched && entries[0].event_count == 1u);
    assert(strcmp(entries[1].entity, "entity-b") == 0 && entries[1].matched && entries[1].event_count == 1u);
    assert(strcmp(entries[2].entity, "entity-d") == 0 && !entries[2].matched && entries[2].event_count == 0u);
    lana_ledger_coverage_free(entries, count);
    lana_vm_free(&vm); assert(lana_ledger_close(ledger) == LANA_OK); assert(lana_store_close(store) == LANA_OK);
    cleanup_store(path);
    printf("Pass.\n");
}

static void test_coverage_malformed(void) {
    printf("Testing ledger coverage malformed input...\n");
    char path[] = "/tmp/lana-ledger-cov-XXXXXX";
    LanaStoreOptions options = {sizeof(options), 1u, path, 1000u};
    LanaStore *store = NULL; LanaLedger *ledger = NULL; LanaVM vm;
    const char *targets[] = {"entity-a"};
    LanaLedgerCoverageQuery query = {sizeof(query), 1u, NULL, NULL, NULL, 0u, 0u, targets, 1u};
    LanaLedgerCoverageEntry *entries = NULL; size_t count = 0u;
    assert(mkdtemp(path) != NULL);
    assert(lana_store_open(&options, &store) == LANA_OK);
    assert(lana_ledger_open(store, &ledger) == LANA_OK);
    lana_vm_init(&vm, NULL);
    assert(lana_ledger_query_coverage(NULL, &vm, &query, &entries, &count) == LANA_ERR_INVALID_STATE);
    assert(lana_ledger_query_coverage(ledger, &vm, NULL, &entries, &count) == LANA_ERR_INVALID_STATE);
    assert(lana_ledger_query_coverage(ledger, &vm, &query, NULL, &count) == LANA_ERR_INVALID_STATE);
    query.struct_size = sizeof(query) - 1u;
    assert(lana_ledger_query_coverage(ledger, &vm, &query, &entries, &count) == LANA_ERR_INVALID_STATE);
    query.struct_size = sizeof(query); query.schema_version = 2u;
    assert(lana_ledger_query_coverage(ledger, &vm, &query, &entries, &count) == LANA_ERR_INVALID_STATE);
    query.schema_version = 1u; query.target_entities = NULL;
    assert(lana_ledger_query_coverage(ledger, &vm, &query, &entries, &count) == LANA_ERR_INVALID_STATE);
    lana_vm_free(&vm); assert(lana_ledger_close(ledger) == LANA_OK); assert(lana_store_close(store) == LANA_OK);
    cleanup_store(path);
    printf("Pass.\n");
}

static void test_coverage_empty_and_large(void) {
    printf("Testing ledger coverage empty and large target set...\n");
    char path[] = "/tmp/lana-ledger-cov-XXXXXX";
    LanaStoreOptions options = {sizeof(options), 1u, path, 1000u};
    LanaStore *store = NULL; LanaLedger *ledger = NULL; LanaVM vm; LanaEvent event;
    LanaEventInput input = {sizeof(input), 1u, "entity-a", "actor1", "action1", "test", 10u, 0u};
    const char *targets[20]; LanaLedgerCoverageQuery query; LanaLedgerCoverageEntry *entries = NULL; size_t count = 0u, index;
    assert(mkdtemp(path) != NULL);
    assert(lana_store_open(&options, &store) == LANA_OK);
    assert(lana_ledger_open(store, &ledger) == LANA_OK);
    lana_vm_init(&vm, NULL);
    /* Empty ledger: coverage returns all targets unmatched, not NO_MATCHING_EVENT. */
    targets[0] = "entity-a"; targets[1] = "entity-b";
    query = (LanaLedgerCoverageQuery){sizeof(query), 1u, NULL, NULL, NULL, 0u, 0u, targets, 2u};
    assert(lana_ledger_query_coverage(ledger, &vm, &query, &entries, &count) == LANA_OK);
    assert(count == 2u && !entries[0].matched && !entries[1].matched);
    lana_ledger_coverage_free(entries, count);
    /* 100 events across 10 entities, then a 20-target coverage query. */
    for (index = 0u; index < 100u; ++index) {
        char entity[16];
        (void)snprintf(entity, sizeof(entity), "entity-%zu", index % 10u);
        input.entity = entity;
        assert(lana_ledger_append(ledger, &input, &event) == LANA_OK);
    }
    for (index = 0u; index < 20u; ++index) {
        static char names[20][16];
        (void)snprintf(names[index], sizeof(names[index]), "entity-%zu", index);
        targets[index] = names[index];
    }
    query = (LanaLedgerCoverageQuery){sizeof(query), 1u, NULL, NULL, NULL, 0u, 0u, targets, 20u};
    assert(lana_ledger_query_coverage(ledger, &vm, &query, &entries, &count) == LANA_OK);
    assert(count == 20u);
    for (index = 0u; index < 20u; ++index) {
        if (index < 10u) assert(entries[index].matched && entries[index].event_count == 10u);
        else assert(!entries[index].matched && entries[index].event_count == 0u);
    }
    lana_ledger_coverage_free(entries, count);
    lana_vm_free(&vm); assert(lana_ledger_close(ledger) == LANA_OK); assert(lana_store_close(store) == LANA_OK);
    cleanup_store(path);
    printf("Pass.\n");
}

static void test_coverage_corruption(void) {
    printf("Testing ledger coverage corruption distinction...\n");
    char path[] = "/tmp/lana-ledger-cov-XXXXXX";
    LanaStoreOptions options = {sizeof(options), 1u, path, 1000u};
    LanaStore *store = NULL; LanaLedger *ledger = NULL; LanaVM vm; LanaEvent event;
    LanaEventInput input = {sizeof(input), 1u, "entity-a", "actor1", "action1", "test", 10u, 0u};
    const char *targets[] = {"entity-a"};
    LanaLedgerCoverageQuery query = {sizeof(query), 1u, NULL, NULL, NULL, 0u, 0u, targets, 1u};
    LanaLedgerCoverageEntry *entries = NULL; size_t count = 0u;
    assert(mkdtemp(path) != NULL);
    assert(lana_store_open(&options, &store) == LANA_OK);
    assert(lana_ledger_open(store, &ledger) == LANA_OK);
    lana_vm_init(&vm, NULL);
    assert(lana_ledger_append(ledger, &input, &event) == LANA_OK);
    /* Overwrite the event record with a non-JSON string and commit. */
    assert(lana_store_put(store, "event/1", lana_value_string("not-json")) == LANA_OK);
    assert(lana_store_commit(store, NULL) == LANA_OK);
    assert(lana_ledger_query_coverage(ledger, &vm, &query, &entries, &count) == LANA_ERR_CORRUPTION);
    lana_vm_free(&vm); assert(lana_ledger_close(ledger) == LANA_OK); assert(lana_store_close(store) == LANA_OK);
    cleanup_store(path);
    printf("Pass.\n");
}

static void test_traversal(void) {
    printf("Testing evidence-to-event traversal...\n");
    char path[] = "/tmp/lana-ledger-cov-XXXXXX";
    LanaStoreOptions options = {sizeof(options), 1u, path, 1000u};
    LanaStore *store = NULL; LanaLedger *ledger = NULL; LanaVM vm; LanaEvent event;
    LanaEventInput input = {sizeof(input), 1u, "service-a", "sre", "alert", "decision 7", 10u, 0u};
    LanaPolicyRule rule = {sizeof(rule), 1u, LANA_POLICY_PROBABILITY_AT_LEAST,
                           "probability", 0.8, NULL, "notify"};
    LanaPolicy policy = {sizeof(policy), 1u, "health", rule};
    LanaPolicyEvaluation evaluation = {sizeof(evaluation), 1u, 7u, "service-a", "health",
                                       3u, "evidence-1", "derivation-1", "explicit", 100u,
                                       "probability threshold", "measurements"};
    LanaDecision decision = {0}; LanaMap *map; Value key = lana_value_number(0.95); Value input_value;
    LanaEffectAttempt attempt = {0}; Value plan, payload, out;
    LanaTraversalQuery tquery = {sizeof(tquery), 1u, "evidence-1"};
    LanaTraversalRecord *records = NULL; size_t count = 0u;
    assert(mkdtemp(path) != NULL);
    assert(lana_store_open(&options, &store) == LANA_OK);
    assert(lana_ledger_open(store, &ledger) == LANA_OK);
    lana_vm_init(&vm, NULL);
    assert(lana_map_new(&vm, 1u, &map) == LANA_OK);
    assert(lana_map_set(&vm, map, "probability", &key, false) == LANA_OK);
    input_value = lana_value_map(map);
    assert(lana_policy_store(store, &vm, &policy) == LANA_OK);
    assert(lana_policy_evaluate(&policy, &input_value, &evaluation, &decision) == LANA_OK);
    assert(decision.outcome == POLICY_OUTCOME_AUTHORIZE);
    assert(lana_policy_store_decision(store, &vm, &decision) == LANA_OK);
    assert(lana_store_commit(store, NULL) == LANA_OK);
    payload = lana_value_number(1.0);
    assert(lana_vm_planned_effect(&vm, "notify", &payload, &plan) == LANA_OK);
    assert(lana_effect_execute_attempt(store, &vm, &decision, 1u, &plan, mock_executor, NULL, &out, &attempt) == LANA_OK);
    assert(lana_ledger_append(ledger, &input, &event) == LANA_OK);
    assert(lana_ledger_traverse(ledger, &vm, &tquery, &records, &count) == LANA_OK);
    assert(count == 1u);
    assert(records[0].decision_id == 7u);
    assert(strcmp(records[0].policy_id, "health") == 0);
    assert(records[0].outcome == POLICY_OUTCOME_AUTHORIZE);
    assert(records[0].attempt_id == 1u && records[0].attempt_status == EFFECT_STATUS_SUCCEEDED);
    assert(records[0].event_id != 0u);
    lana_traversal_records_free(records, count);
    lana_vm_free(&vm); assert(lana_ledger_close(ledger) == LANA_OK); assert(lana_store_close(store) == LANA_OK);
    cleanup_store(path);
    printf("Pass.\n");
}

int main(void) {
    test_coverage_success();
    test_coverage_malformed();
    test_coverage_empty_and_large();
    test_coverage_corruption();
    test_traversal();
    return 0;
}
