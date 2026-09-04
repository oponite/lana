#ifndef LANA_BRIDGE_H
#define LANA_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#include "policy.h"

#define LANA_BRIDGE_ABI_VERSION 1u

typedef enum {
    LANA_BRIDGE_OK = 0,
    LANA_BRIDGE_ERR_ARGUMENT = -1,
    LANA_BRIDGE_ERR_ABI = -2,
    LANA_BRIDGE_ERR_IO = -3,
    LANA_BRIDGE_ERR_PROTOCOL = -4,
    LANA_BRIDGE_ERR_OOM = -5
} LanaBridgeStatus;

typedef struct {
    size_t struct_size;
    uint32_t abi_version;
    uint64_t seed;
    uint64_t instruction_limit;
    size_t memory_limit_bytes;
    size_t workers;
    size_t max_tasks;
} LanaBridgeOptions;

/*
 * Pipeline configuration. The embedded rule is validated by the policy layer;
 * struct_size and schema_version must be set on both the config and the rule.
 */
typedef struct {
    size_t struct_size;
    uint32_t abi_version;
    const char *store_path;
    const char *policy_id;
    LanaPolicyRule rule;
    uint64_t seed;
    uint64_t instruction_limit;
    size_t memory_limit_bytes;
    size_t workers;
    size_t max_tasks;
} LanaBridgePipelineConfig;

/*
 * Returns zero on success, a positive LanaError on VM failure, or a negative
 * LanaBridgeStatus on bridge failure. envelope_json is always bridge-owned and
 * must be released with lana_bridge_free when non-NULL.
 */
int lana_bridge_run_labc(const char *labc_path, const char *request_path,
                         const char *response_path,
                         const LanaBridgeOptions *options,
                         char **envelope_json);

/*
 * Runs the Lana program, then executes the durable pipeline: store the policy
 * (idempotent), evaluate the decision-request against the policy, persist the
 * decision, and on AUTHORIZE execute the planned effect and append a ledger
 * event. The envelope reports decision_id, outcome, attempt_status, event_id,
 * and the store revision. REFUSE/REQUEST_MORE_EVIDENCE execute no effect and
 * append no event.
 */
int lana_bridge_run_pipeline(const char *labc_path, const char *request_path,
                             const char *response_path,
                             const LanaBridgeOptions *options,
                             const LanaBridgePipelineConfig *config,
                             char **envelope_json);

void lana_bridge_free(char *value);
const char *lana_bridge_version(void);

#endif
