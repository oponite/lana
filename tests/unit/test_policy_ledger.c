#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <unistd.h>

#include "policy.h"
#include "ledger.h"
#include "error.h"
#include "data.h"
#include "vm.h"

static void test_ledger_append_query(void) {
    printf("Testing ledger append/query...\n");
    char path[] = "/tmp/lana-ledger-XXXXXX";
    LanaStoreOptions options = {sizeof(options), 1u, path, 1000u};
    LanaStore *store = NULL; LanaLedger *ledger = NULL; LanaVM vm; LanaEvent event;
    LanaEventInput input = {sizeof(input), 1u, "entity\"1\\\n", "actor1", "action1", NULL, UINT64_MAX, UINT64_MAX - 1u};
    LanaLedgerQuery query = {sizeof(query), 1u, input.entity, NULL, NULL, 0u, 0u};
    LanaEvent *events = NULL; size_t count = 0u;
    assert(mkdtemp(path) != NULL);
    assert(lana_store_open(&options, &store) == LANA_OK);
    assert(lana_ledger_open(store, &ledger) == LANA_OK);
    assert(lana_ledger_append(ledger, &input, &event) == LANA_OK && event.event_id == 1u);
    lana_vm_init(&vm, NULL);
    assert(lana_ledger_query(ledger, &vm, &query, &events, &count) == LANA_OK && count == 1u);
    assert(strcmp(events[0].action, "action1") == 0);
    assert(strcmp(events[0].reason, "") == 0);
    assert(events[0].timestamp == UINT64_MAX);
    assert(events[0].correction_of == UINT64_MAX - 1u);
    lana_ledger_events_free(events, count); lana_vm_free(&vm);
    assert(lana_ledger_close(ledger) == LANA_OK); assert(lana_store_close(store) == LANA_OK);
    {
        char file[256];
        (void)snprintf(file, sizeof(file), "%s/journal", path); (void)unlink(file);
        (void)snprintf(file, sizeof(file), "%s/manifest", path); (void)unlink(file);
    }
    (void)rmdir(path);

    printf("Pass.\n");
}

static void test_policy_evaluation(void) {
    printf("Testing policy evaluation...\n");
    LanaPolicyRule rule = {sizeof(rule), 1u, LANA_POLICY_PROBABILITY_AT_LEAST,
                           "probability", 0.8, NULL, "notify"};
    LanaPolicy policy = {sizeof(policy), 1u, "health", rule};
    LanaPolicyEvaluation evaluation = {sizeof(evaluation), 1u, 41u, "service-a", "health",
                                       3u, "evidence-1", "derivation-1", "resolution-1", 100u,
                                       "probability threshold", "measurements"};
    LanaDecision decision = {0};
    LanaVM vm;
    LanaMap *map;
    Value key = lana_value_number(0.95);
    Value input;
    lana_vm_init(&vm, NULL);
    assert(lana_map_new(&vm, 1u, &map) == LANA_OK);
    assert(lana_map_set(&vm, map, "probability", &key, false) == LANA_OK);
    input = lana_value_map(map);
    assert(lana_policy_evaluate(&policy, &input, &evaluation, &decision) == LANA_OK);
    assert(decision.outcome == POLICY_OUTCOME_AUTHORIZE && strcmp(decision.effect, "notify") == 0);
    key = lana_value_number(0.2);
    assert(lana_map_set(&vm, map, "probability", &key, false) == LANA_OK);
    assert(lana_policy_evaluate(&policy, &input, &evaluation, &decision) == LANA_OK);
    assert(decision.outcome == POLICY_OUTCOME_REQUEST_MORE_EVIDENCE);
    {
        const double invalid[] = {NAN, INFINITY, -INFINITY, -0.1, 1.1};
        unsigned char digest[32], empty[32] = {0};
        for (size_t index = 0; index < sizeof(invalid) / sizeof(*invalid); ++index) {
            policy.rule.threshold = invalid[index];
            memset(digest, 0xa5, sizeof(digest));
            assert(lana_policy_version(&policy, digest) == LANA_ERR_INVALID_PROBABILITY);
            assert(memcmp(digest, empty, sizeof(digest)) == 0);
            assert(lana_policy_evaluate(&policy, &input, &evaluation, &decision) == LANA_ERR_INVALID_PROBABILITY);
            assert(decision.struct_size == 0u && decision.effect == NULL && decision.policy_id == NULL);
        }
        policy.rule.threshold = 0.5;
        policy.rule.kind = (LanaPolicyRuleKind)-1;
        assert(lana_policy_version(&policy, digest) == LANA_ERR_SCHEMA);
    }
    lana_vm_free(&vm);

    printf("Pass.\n");
}

int main(void) {
    test_ledger_append_query();
    test_policy_evaluation();
    return 0;
}
