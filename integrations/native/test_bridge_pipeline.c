#include "lana/bridge.h"

#include "data.h"
#include "error.h"
#include "store.h"
#include "vm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) { \
    (void)fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #condition); \
    return 1; } } while (0)

static const char *map_string(const Value *map, const char *key) {
    Value value;
    if (map == NULL || map->type != VAL_MAP) return NULL;
    if (lana_map_get(map->as.map, key, &value) != LANA_OK) return NULL;
    if (value.type != VAL_STRING) return NULL;
    return value.as.string;
}

static double map_number(const Value *map, const char *key) {
    Value value;
    if (map == NULL || map->type != VAL_MAP) return -1.0;
    if (lana_map_get(map->as.map, key, &value) != LANA_OK) return -1.0;
    if (value.type != VAL_NUMBER) return -1.0;
    return value.as.number;
}

static int rule_kind(const char *name) {
    if (strcmp(name, "probability_at_least") == 0) return LANA_POLICY_PROBABILITY_AT_LEAST;
    if (strcmp(name, "equals") == 0) return LANA_POLICY_EQUALS;
    if (strcmp(name, "order_less_than") == 0) return LANA_POLICY_ORDER_LESS_THAN;
    if (strcmp(name, "present") == 0) return LANA_POLICY_PRESENT;
    return -1;
}

int main(int argc, char **argv) {
    LanaBridgeOptions options = {0};
    LanaBridgePipelineConfig config = {0};
    LanaVM *vm;
    LanaStore *store;
    LanaStoreOptions store_options = {0};
    LanaStoreRevisionInfo revision;
    LanaStoreScanRecord *records = NULL;
    char *envelope = NULL;
    char *policy_text = NULL;
    Value policy_value = {0}, rule_value = {0}, result_value = {0};
    Value stored_decision = {0}, outcome_value = {0};
    const char *expected_outcome;
    const char *outcome;
    size_t count = 0u;
    int result;
    FILE *file;
    long size;
    size_t read;

    CHECK(argc == 7 || argc == 8);
    expected_outcome = argv[6];

    file = fopen(argv[5], "rb");
    CHECK(file != NULL);
    CHECK(fseek(file, 0, SEEK_END) == 0);
    size = ftell(file);
    CHECK(size > 0);
    CHECK(fseek(file, 0, SEEK_SET) == 0);
    policy_text = malloc((size_t)size + 1u);
    CHECK(policy_text != NULL);
    read = fread(policy_text, 1u, (size_t)size, file);
    CHECK(read == (size_t)size);
    CHECK(fclose(file) == 0);
    policy_text[size] = '\0';

    vm = lana_vm_create();
    CHECK(vm != NULL);
    CHECK(lana_json_parse(vm, policy_text, &policy_value) == LANA_OK);
    CHECK(policy_value.type == VAL_MAP);
    free(policy_text);

    config.struct_size = sizeof(config);
    config.abi_version = LANA_BRIDGE_ABI_VERSION;
    config.store_path = argv[4];
    config.policy_id = map_string(&policy_value, "policy_id");
    CHECK(config.policy_id != NULL);
    CHECK(lana_map_get(policy_value.as.map, "rule", &rule_value) == LANA_OK);
    CHECK(rule_value.type == VAL_MAP);
    config.rule.struct_size = sizeof(config.rule);
    config.rule.schema_version = 1u;
    config.rule.kind = (LanaPolicyRuleKind)rule_kind(map_string(&rule_value, "kind"));
    CHECK((int)config.rule.kind >= 0);
    config.rule.field = map_string(&rule_value, "field");
    config.rule.threshold = map_number(&rule_value, "threshold");
    config.rule.expected = map_string(&rule_value, "expected");
    config.rule.effect = map_string(&rule_value, "effect");
    CHECK(config.rule.field != NULL);
    CHECK(config.rule.effect != NULL);

    options.struct_size = sizeof(options);
    options.abi_version = LANA_BRIDGE_ABI_VERSION;

    result = lana_bridge_run_pipeline(argv[1], argv[2], argv[3], &options,
                                      &config, &envelope);
    CHECK(result == LANA_BRIDGE_OK);
    CHECK(envelope != NULL);
    CHECK(strstr(envelope, "\"ok\":true") != NULL);

    CHECK(lana_json_parse(vm, envelope, &result_value) == LANA_OK);
    CHECK(result_value.type == VAL_MAP);
    CHECK(lana_map_get(result_value.as.map, "result", &result_value) == LANA_OK);
    CHECK(result_value.type == VAL_MAP);
    outcome = map_string(&result_value, "outcome");
    CHECK(outcome != NULL);
    CHECK(map_number(&result_value, "decision_id") > 0.0);

    if (strcmp(expected_outcome, "authorize") == 0) {
        CHECK(strcmp(outcome, "authorize") == 0);
        CHECK(map_string(&result_value, "attempt_status") != NULL);
        CHECK(map_number(&result_value, "event_id") > 0.0);
    } else {
        CHECK(strcmp(outcome, "request_more_evidence") == 0);
        CHECK(map_string(&result_value, "attempt_status") == NULL);
    }

    store_options.struct_size = sizeof(store_options);
    store_options.schema_version = 1u;
    store_options.path = argv[4];
    CHECK(lana_store_open(&store_options, &store) == LANA_OK);

    CHECK(lana_store_scan(store, vm, "decision/", &records, &count) == LANA_OK);
    CHECK(count == 1u);
    CHECK(records[0].value.type == VAL_STRING);
    CHECK(lana_json_parse(vm, records[0].value.as.string, &stored_decision) == LANA_OK);
    CHECK(stored_decision.type == VAL_MAP);
    CHECK(lana_map_get(stored_decision.as.map, "outcome", &outcome_value) == LANA_OK);
    CHECK(outcome_value.type == VAL_NUMBER);
    if (strcmp(expected_outcome, "authorize") == 0)
        CHECK(outcome_value.as.number == 0.0); /* POLICY_OUTCOME_AUTHORIZE */
    else
        CHECK(outcome_value.as.number == 2.0); /* POLICY_OUTCOME_REQUEST_MORE_EVIDENCE */
    lana_store_scan_free(records, count);
    records = NULL; count = 0u;

    if (strcmp(expected_outcome, "authorize") == 0) {
        CHECK(lana_store_scan(store, vm, "effect-attempt/", &records, &count) == LANA_OK);
        CHECK(count == 1u);
        lana_store_scan_free(records, count);
        records = NULL; count = 0u;
        CHECK(lana_store_scan(store, vm, "event/", &records, &count) == LANA_OK);
        CHECK(count == 1u);
        lana_store_scan_free(records, count);
        records = NULL; count = 0u;
    } else {
        CHECK(lana_store_scan(store, vm, "effect-attempt/", &records, &count) == LANA_OK);
        CHECK(count == 0u);
        lana_store_scan_free(records, count);
        records = NULL; count = 0u;
        CHECK(lana_store_scan(store, vm, "event/", &records, &count) == LANA_OK);
        CHECK(count == 0u);
        lana_store_scan_free(records, count);
        records = NULL; count = 0u;
    }

    CHECK(lana_store_current_revision(store, &revision) == LANA_OK);
    CHECK(revision.revision_id > 0u);
    CHECK(lana_store_close(store) == LANA_OK);

    if (argc == 8) {
        FILE *out = fopen(argv[7], "wb");
        CHECK(out != NULL);
        CHECK(fwrite(envelope, 1u, strlen(envelope), out) == strlen(envelope));
        CHECK(fclose(out) == 0);
    }

    lana_bridge_free(envelope);
    lana_vm_destroy(vm);
    (void)printf("bridge pipeline test passed (outcome=%s)\n", outcome);
    return 0;
}
