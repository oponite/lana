#ifndef LANA_POLICY_H
#define LANA_POLICY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "lana/error.h"
#include "lana/value.h"

typedef struct LanaStore LanaStore;
typedef struct LanaVM LanaVM;

typedef enum {
    LANA_POLICY_PROBABILITY_AT_LEAST = 0,
    LANA_POLICY_EQUALS,
    LANA_POLICY_ORDER_LESS_THAN,
    LANA_POLICY_PRESENT
} LanaPolicyRuleKind;

typedef enum {
    POLICY_OUTCOME_AUTHORIZE = 0,
    POLICY_OUTCOME_REFUSE,
    POLICY_OUTCOME_REQUEST_MORE_EVIDENCE
} PolicyOutcome;

typedef struct {
    size_t struct_size;
    uint32_t schema_version;
    LanaPolicyRuleKind kind;
    const char *field;
    double threshold;
    const char *expected;
    const char *effect;
} LanaPolicyRule;

typedef struct {
    size_t struct_size;
    uint32_t schema_version;
    const char *policy_id;
    LanaPolicyRule rule;
} LanaPolicy;

typedef struct {
    size_t struct_size;
    uint32_t schema_version;
    uint64_t decision_id;
    const char *target;
    const char *scope;
    uint64_t input_revision;
    const char *evidence_ids;
    const char *derivation_ids;
    const char *relationship_resolution;
    uint64_t evaluation_time;
    const char *reason;
    const char *requested_evidence;
} LanaPolicyEvaluation;

typedef struct {
    size_t struct_size;
    uint32_t schema_version;
    uint64_t decision_id;
    const char *policy_id;
    unsigned char policy_version[32];
    const char *target;
    const char *scope;
    uint64_t input_revision;
    const char *evidence_ids;
    const char *derivation_ids;
    const char *relationship_resolution;
    PolicyOutcome outcome;
    const char *effect;
    uint64_t evaluation_time;
    const char *reason;
    const char *requested_evidence;
} LanaDecision;

LanaError lana_policy_version(const LanaPolicy *policy, unsigned char out_version[32]);
LanaError lana_policy_store(LanaStore *store, LanaVM *vm, const LanaPolicy *policy);
LanaError lana_policy_evaluate(const LanaPolicy *policy, const Value *input,
                               const LanaPolicyEvaluation *evaluation,
                               LanaDecision *out_decision);
LanaError lana_policy_store_decision(LanaStore *store, LanaVM *vm,
                                     const LanaDecision *decision);
LanaError lana_policy_replay(const LanaPolicy *policy, const Value *input,
                             const LanaPolicyEvaluation *evaluation,
                             const LanaDecision *expected,
                             LanaDecision *out_decision);
void lana_decision_free(LanaDecision *decision);

#endif
