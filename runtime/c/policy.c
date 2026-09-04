#include "policy.h"

#include "codec.h"
#include "data.h"
#include "sha256.h"
#include "store.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool text_ok(const char *text) { return text != NULL && text[0] != '\0'; }

static void hex_digest(const unsigned char digest[32], char output[65]) {
    static const char hex[] = "0123456789abcdef"; size_t index;
    for (index = 0u; index < 32u; ++index) { output[index * 2u] = hex[digest[index] >> 4u]; output[index * 2u + 1u] = hex[digest[index] & 15u]; }
    output[64] = '\0';
}

static bool json_safe(const char *text) {
    size_t index;
    if (text == NULL) return true;
    for (index = 0u; text[index] != '\0'; ++index) if ((unsigned char)text[index] < 0x20u || text[index] == '\\' || text[index] == '"') return false;
    return true;
}

static LanaError validate_policy(const LanaPolicy *policy) {
    if (policy == NULL || policy->struct_size < sizeof(*policy) || policy->schema_version != 1u ||
        !text_ok(policy->policy_id) || policy->rule.struct_size < sizeof(policy->rule) ||
        policy->rule.schema_version != 1u || !text_ok(policy->rule.field) || !text_ok(policy->rule.effect))
        return LANA_ERR_SCHEMA;
    if ((policy->rule.kind == LANA_POLICY_PROBABILITY_AT_LEAST ||
         policy->rule.kind == LANA_POLICY_ORDER_LESS_THAN) &&
        (policy->rule.threshold < 0.0 || policy->rule.threshold > 1.0)) return LANA_ERR_INVALID_PROBABILITY;
    if ((policy->rule.kind == LANA_POLICY_EQUALS && !text_ok(policy->rule.expected)) ||
        policy->rule.kind > LANA_POLICY_PRESENT) return LANA_ERR_SCHEMA;
    return LANA_OK;
}

LanaError lana_policy_version(const LanaPolicy *policy, unsigned char out_version[32]) {
    LanaBuffer canonical = {0}; LanaError error;
    if (out_version == NULL) return LANA_ERR_INVALID_STATE;
    error = validate_policy(policy); if (error != LANA_OK) return error;
    error = lana_codec_encode_value(&canonical, lana_value_string((char *)policy->policy_id));
    if (error == LANA_OK) error = lana_codec_encode_value(&canonical, lana_value_string((char *)policy->rule.field));
    if (error == LANA_OK) error = lana_codec_encode_value(&canonical, lana_value_number((double)policy->rule.kind));
    if (error == LANA_OK) error = lana_codec_encode_value(&canonical, lana_value_number(policy->rule.threshold));
    if (error == LANA_OK) error = lana_codec_encode_value(&canonical, lana_value_string((char *)(policy->rule.expected == NULL ? "" : policy->rule.expected)));
    if (error == LANA_OK) error = lana_codec_encode_value(&canonical, lana_value_string((char *)policy->rule.effect));
    if (error == LANA_OK) lana_sha256(canonical.data, canonical.length, out_version);
    free(canonical.data); return error;
}

LanaError lana_policy_store(LanaStore *store, LanaVM *vm, const LanaPolicy *policy) {
    unsigned char digest[32]; char version[65], key[160], json[1024]; Value existing, value; LanaError error;
    if (store == NULL || vm == NULL) return LANA_ERR_INVALID_STATE;
    error = validate_policy(policy); if (error != LANA_OK) return error;
    if ((error = lana_policy_version(policy, digest)) != LANA_OK) return error;
    if (!json_safe(policy->policy_id) || !json_safe(policy->rule.field) || !json_safe(policy->rule.expected) || !json_safe(policy->rule.effect)) return LANA_ERR_SCHEMA;
    hex_digest(digest, version);
    (void)snprintf(key, sizeof(key), "policy/%s/%s", policy->policy_id, version);
    error = lana_store_get(store, vm, key, &existing);
    if (error == LANA_OK) return LANA_ERR_CONFLICT;
    if (error != LANA_ERR_NOT_FOUND) return error;
    (void)snprintf(json, sizeof(json), "{\"effect\":\"%s\",\"expected\":\"%s\",\"field\":\"%s\",\"kind\":%u,\"policy_id\":\"%s\",\"schema\":1,\"threshold\":%.17g,\"version\":\"%s\"}",
                   policy->rule.effect, policy->rule.expected == NULL ? "" : policy->rule.expected,
                   policy->rule.field, (unsigned)policy->rule.kind, policy->policy_id, policy->rule.threshold, version);
    if (strlen(json) >= sizeof(json) - 1u) return LANA_ERR_LIMIT;
    value = lana_value_string(json); return lana_store_put(store, key, value);
}

static bool condition_matches(const LanaPolicyRule *rule, const Value *input) {
    Value value;
    if (input == NULL || input->type != VAL_MAP || lana_map_get(input->as.map, rule->field, &value) != LANA_OK) return false;
    switch (rule->kind) {
        case LANA_POLICY_PROBABILITY_AT_LEAST: return value.type == VAL_NUMBER && value.as.number >= rule->threshold && value.as.number <= 1.0;
        case LANA_POLICY_ORDER_LESS_THAN: return value.type == VAL_NUMBER && value.as.number < rule->threshold && value.as.number >= 0.0;
        case LANA_POLICY_EQUALS: return value.type == VAL_STRING && strcmp(value.as.string, rule->expected) == 0;
        case LANA_POLICY_PRESENT: return true;
        default: return false;
    }
}

LanaError lana_policy_evaluate(const LanaPolicy *policy, const Value *input,
                               const LanaPolicyEvaluation *evaluation, LanaDecision *out) {
    LanaError error; bool matches;
    if (out == NULL || evaluation == NULL || evaluation->struct_size < sizeof(*evaluation) || evaluation->schema_version != 1u ||
        !text_ok(evaluation->target) || !text_ok(evaluation->scope) || !text_ok(evaluation->reason)) return LANA_ERR_SCHEMA;
    error = validate_policy(policy); if (error != LANA_OK) return error;
    memset(out, 0, sizeof(*out)); out->struct_size = sizeof(*out); out->schema_version = 1u;
    out->decision_id = evaluation->decision_id; out->policy_id = policy->policy_id;
    out->target = evaluation->target; out->scope = evaluation->scope; out->input_revision = evaluation->input_revision;
    out->evidence_ids = evaluation->evidence_ids; out->derivation_ids = evaluation->derivation_ids;
    out->relationship_resolution = evaluation->relationship_resolution; out->evaluation_time = evaluation->evaluation_time;
    out->reason = evaluation->reason; out->requested_evidence = evaluation->requested_evidence;
    error = lana_policy_version(policy, out->policy_version); if (error != LANA_OK) return error;
    matches = condition_matches(&policy->rule, input);
    out->outcome = matches ? POLICY_OUTCOME_AUTHORIZE : POLICY_OUTCOME_REQUEST_MORE_EVIDENCE;
    out->effect = matches ? policy->rule.effect : NULL;
    return LANA_OK;
}

LanaError lana_policy_store_decision(LanaStore *store, LanaVM *vm, const LanaDecision *decision) {
    char key[64], json[4096], version[65]; Value existing, value; LanaError error;
    if (store == NULL || vm == NULL || decision == NULL || decision->struct_size < sizeof(*decision) || decision->schema_version != 1u) return LANA_ERR_SCHEMA;
    (void)snprintf(key, sizeof(key), "decision/%llu", (unsigned long long)decision->decision_id);
    error = lana_store_get(store, vm, key, &existing);
    if (error == LANA_OK) return LANA_ERR_CONFLICT;
    if (error != LANA_ERR_NOT_FOUND) return error;
    if (!text_ok(decision->policy_id) || !text_ok(decision->target) || !text_ok(decision->scope) ||
        !text_ok(decision->reason) || !json_safe(decision->policy_id) || !json_safe(decision->target) ||
        !json_safe(decision->scope) || !json_safe(decision->reason) || !json_safe(decision->evidence_ids) ||
        !json_safe(decision->derivation_ids) || !json_safe(decision->relationship_resolution) ||
        !json_safe(decision->effect) || !json_safe(decision->requested_evidence)) return LANA_ERR_SCHEMA;
    hex_digest(decision->policy_version, version);
    (void)snprintf(json, sizeof(json),
                   "{\"decision_id\":%llu,\"derivation_ids\":\"%s\",\"effect\":\"%s\",\"evaluation_time\":%llu,\"evidence_ids\":\"%s\",\"input_revision\":%llu,\"outcome\":%u,\"policy_id\":\"%s\",\"policy_version\":\"%s\",\"relationship_resolution\":\"%s\",\"requested_evidence\":\"%s\",\"reason\":\"%s\",\"scope\":\"%s\",\"target\":\"%s\"}",
                   (unsigned long long)decision->decision_id, decision->derivation_ids == NULL ? "" : decision->derivation_ids,
                   decision->effect == NULL ? "" : decision->effect, (unsigned long long)decision->evaluation_time,
                   decision->evidence_ids == NULL ? "" : decision->evidence_ids, (unsigned long long)decision->input_revision,
                   (unsigned)decision->outcome, decision->policy_id, version,
                   decision->relationship_resolution == NULL ? "" : decision->relationship_resolution,
                   decision->requested_evidence == NULL ? "" : decision->requested_evidence,
                   decision->reason, decision->scope, decision->target);
    if (strlen(json) >= sizeof(json) - 1u) return LANA_ERR_LIMIT;
    value = lana_value_string(json); return lana_store_put(store, key, value);
}

LanaError lana_policy_replay(const LanaPolicy *policy, const Value *input,
                             const LanaPolicyEvaluation *evaluation,
                             const LanaDecision *expected, LanaDecision *out) {
    LanaDecision replayed; LanaError error;
    if (expected == NULL || expected->struct_size < sizeof(*expected) || expected->schema_version != 1u) return LANA_ERR_SCHEMA;
    if ((error = lana_policy_evaluate(policy, input, evaluation, &replayed)) != LANA_OK) return error;
    if (memcmp(replayed.policy_version, expected->policy_version, sizeof(replayed.policy_version)) != 0 ||
        replayed.outcome != expected->outcome || replayed.decision_id != expected->decision_id ||
        replayed.input_revision != expected->input_revision) return LANA_ERR_INTEGRITY;
    if (out != NULL) *out = replayed;
    return LANA_OK;
}

void lana_decision_free(LanaDecision *decision) { (void)decision; }
