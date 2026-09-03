#include "lana/bridge.h"

#include "lana/bytecode.h"
#include "lana/data.h"
#include "lana/effects.h"
#include "lana/error.h"
#include "lana/ledger.h"
#include "lana/sha256.h"
#include "lana/store.h"
#include "lana/vm.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef LANA_VERSION
#define LANA_VERSION "unknown"
#endif

#define LANA_BRIDGE_RESPONSE_LIMIT (64u * 1024u * 1024u)

static char *json_escape(const char *text) {
    static const char hex[] = "0123456789abcdef";
    size_t length = 0u, index, output_index = 0u;
    char *output;
    if (text == NULL) text = "";
    for (index = 0u; text[index] != '\0'; ++index) {
        unsigned char byte = (unsigned char)text[index];
        size_t addition = byte == '"' || byte == '\\' ? 2u : byte < 0x20u ? 6u : 1u;
        if (length > SIZE_MAX - addition) return NULL;
        length += addition;
    }
    output = malloc(length + 1u);
    if (output == NULL) return NULL;
    for (index = 0u; text[index] != '\0'; ++index) {
        unsigned char byte = (unsigned char)text[index];
        if (byte == '"' || byte == '\\') {
            output[output_index++] = '\\'; output[output_index++] = (char)byte;
        } else if (byte < 0x20u) {
            output[output_index++] = '\\'; output[output_index++] = 'u';
            output[output_index++] = '0'; output[output_index++] = '0';
            output[output_index++] = hex[byte >> 4u]; output[output_index++] = hex[byte & 15u];
        } else output[output_index++] = (char)byte;
    }
    output[output_index] = '\0'; return output;
}

static char *error_envelope(const char *phase, const char *code,
                            const char *message, int status) {
    char *escaped_phase = json_escape(phase);
    char *escaped_code = json_escape(code);
    char *escaped_message = json_escape(message);
    char *output;
    size_t needed;
    if (escaped_phase == NULL || escaped_code == NULL || escaped_message == NULL) {
        free(escaped_phase); free(escaped_code); free(escaped_message); return NULL;
    }
    needed = strlen(escaped_phase) + strlen(escaped_code) + strlen(escaped_message) + 256u;
    output = malloc(needed);
    if (output != NULL)
        (void)snprintf(output, needed,
            "{\"schema\":1,\"ok\":false,\"phase\":\"%s\","
            "\"error\":{\"code\":\"%s\",\"message\":\"%s\"},"
            "\"exit_code\":%d,\"stdout\":\"\",\"stderr\":\"\","
            "\"execution\":{\"engine\":\"native\",\"lana_version\":\"%s\"}}",
            escaped_phase, escaped_code, escaped_message, status, LANA_VERSION);
    free(escaped_phase); free(escaped_code); free(escaped_message); return output;
}

static char *success_envelope(const char *response) {
    size_t needed = strlen(response) + 256u;
    char *output = malloc(needed);
    if (output != NULL)
        (void)snprintf(output, needed,
            "{\"schema\":1,\"ok\":true,\"result\":%s,"
            "\"stdout\":\"\",\"stderr\":\"\","
            "\"execution\":{\"engine\":\"native\",\"lana_version\":\"%s\"}}",
            response, LANA_VERSION);
    return output;
}

static char *read_response(const char *path, int *status) {
    FILE *file = fopen(path, "rb");
    long length;
    char *text;
    if (file == NULL) { *status = LANA_BRIDGE_ERR_IO; return NULL; }
    if (fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) < 0 ||
        (unsigned long)length > LANA_BRIDGE_RESPONSE_LIMIT ||
        fseek(file, 0, SEEK_SET) != 0) {
        (void)fclose(file); *status = LANA_BRIDGE_ERR_PROTOCOL; return NULL;
    }
    text = malloc((size_t)length + 1u);
    if (text == NULL) { (void)fclose(file); *status = LANA_BRIDGE_ERR_OOM; return NULL; }
    if (fread(text, 1u, (size_t)length, file) != (size_t)length || fclose(file) != 0) {
        free(text); *status = LANA_BRIDGE_ERR_IO; return NULL;
    }
    text[length] = '\0'; return text;
}

static int fail(char **envelope_json, int status, const char *phase,
                const char *code, const char *message) {
    *envelope_json = error_envelope(phase, code, message, status);
    if (*envelope_json == NULL) return LANA_BRIDGE_ERR_OOM;
    return status;
}

/* Runs the LABC program with the given request/response paths and options.
 * On success returns the response text (caller frees) and sets *status to
 * LANA_BRIDGE_OK. On failure returns NULL, sets *status, and writes an error
 * envelope. */
static char *run_program(const char *labc_path, const char *request_path,
                         const char *response_path,
                         const LanaBridgeOptions *options,
                         int *status, char **envelope_json) {
    LanaChunk chunk;
    LanaVM vm;
    LanaErrorInfo error = {0};
    LanaError result;
    const char *arguments[2];
    char *response;
    Value parsed;
    int response_status = LANA_BRIDGE_OK;
    if (envelope_json == NULL) return NULL;
    *envelope_json = NULL;
    (void)remove(response_path);
    result = lana_chunk_read_file(&chunk, labc_path, &error);
    if (result != LANA_OK) {
        *status = (int)result;
        (void)fail(envelope_json, (int)result, "load", lana_error_name(result),
                   error.message[0] == '\0' ? "failed to load LABC" : error.message);
        return NULL;
    }
    lana_vm_init(&vm, &chunk);
    if (options != NULL) {
        if (options->seed != 0u) lana_vm_seed(&vm, options->seed);
        if (options->instruction_limit != 0u) vm.instruction_limit = options->instruction_limit;
        if (options->memory_limit_bytes != 0u) vm.memory_limit = options->memory_limit_bytes;
        if ((options->workers != 0u &&
             lana_vm_set_worker_count(&vm, options->workers) != LANA_OK) ||
            (options->max_tasks != 0u &&
             lana_vm_set_task_limit(&vm, options->max_tasks) != LANA_OK)) {
            lana_vm_free(&vm); lana_chunk_free(&chunk);
            *status = LANA_ERR_TASK;
            (void)fail(envelope_json, LANA_ERR_TASK, "run", "LANA_ERR_TASK",
                        "invalid scheduler options");
            return NULL;
        }
    }
    arguments[0] = request_path; arguments[1] = response_path;
    lana_vm_set_program_args(&vm, 2, arguments);
    result = lana_vm_run(&vm);
    if (result != LANA_OK) {
        char message[LANA_ERROR_MESSAGE_CAPACITY];
        (void)snprintf(message, sizeof(message), "%s",
                       vm.error.message[0] == '\0' ? lana_error_name(result) : vm.error.message);
        lana_vm_free(&vm); lana_chunk_free(&chunk);
        *status = (int)result;
        (void)fail(envelope_json, (int)result, "run", lana_error_name(result), message);
        return NULL;
    }
    response = read_response(response_path, &response_status);
    if (response == NULL) {
        lana_vm_free(&vm); lana_chunk_free(&chunk);
        *status = response_status;
        (void)fail(envelope_json, response_status, "protocol",
                    response_status == LANA_BRIDGE_ERR_OOM ? "LANA_BRIDGE_OOM" :
                    response_status == LANA_BRIDGE_ERR_IO ? "LANA_RESPONSE_MISSING" :
                    "LANA_RESPONSE_LIMIT",
                    response_status == LANA_BRIDGE_ERR_IO ?
                    "program did not write a readable response file" :
                    "response could not be loaded");
        return NULL;
    }
    result = lana_json_parse(&vm, response, &parsed);
    if (result != LANA_OK) {
        free(response); lana_vm_free(&vm); lana_chunk_free(&chunk);
        *status = LANA_BRIDGE_ERR_PROTOCOL;
        (void)fail(envelope_json, LANA_BRIDGE_ERR_PROTOCOL, "protocol",
                    "LANA_RESPONSE_INVALID", "program wrote invalid response JSON");
        return NULL;
    }
    lana_vm_free(&vm); lana_chunk_free(&chunk);
    *status = LANA_BRIDGE_OK;
    return response;
}

int lana_bridge_run_labc(const char *labc_path, const char *request_path,
                         const char *response_path,
                         const LanaBridgeOptions *options,
                         char **envelope_json) {
    char *response;
    int status;
    if (envelope_json == NULL) return LANA_BRIDGE_ERR_ARGUMENT;
    *envelope_json = NULL;
    if (labc_path == NULL || request_path == NULL || response_path == NULL)
        return fail(envelope_json, LANA_BRIDGE_ERR_ARGUMENT, "input",
                    "LANA_BRIDGE_ARGUMENT", "paths must not be null");
    if (options != NULL && (options->struct_size < sizeof(*options) ||
                            options->abi_version != LANA_BRIDGE_ABI_VERSION))
        return fail(envelope_json, LANA_BRIDGE_ERR_ABI, "compatibility",
                    "LANA_BRIDGE_ABI", "unsupported bridge options ABI");
    response = run_program(labc_path, request_path, response_path, options,
                           &status, envelope_json);
    if (response == NULL) return status;
    *envelope_json = success_envelope(response);
    free(response);
    if (*envelope_json == NULL) return LANA_BRIDGE_ERR_OOM;
    return LANA_BRIDGE_OK;
}

static void hex_digest(const unsigned char digest[32], char output[65]) {
    static const char hex[] = "0123456789abcdef"; size_t index;
    for (index = 0u; index < 32u; ++index) {
        output[index * 2u] = hex[digest[index] >> 4u];
        output[index * 2u + 1u] = hex[digest[index] & 15u];
    }
    output[64] = '\0';
}

static const char *map_string(const Value *map, const char *key) {
    Value value;
    if (map == NULL || map->type != VAL_MAP) return NULL;
    if (lana_map_get(map->as.map, key, &value) != LANA_OK) return NULL;
    if (value.type != VAL_STRING) return NULL;
    return value.as.string;
}

/* Deterministic default executor: records the effect kind and succeeds. */
static LanaError default_executor(LanaVM *vm, const char *kind,
                                  const Value *payload, void *context,
                                  Value *out) {
    size_t length;
    char *text;
    (void)payload; (void)context;
    if (vm == NULL || kind == NULL || out == NULL) return LANA_ERR_INVALID_STATE;
    length = strlen(kind) + 16u;
    text = lana_vm_alloc(vm, length);
    if (text == NULL) return LANA_ERR_OOM;
    (void)snprintf(text, length, "executed:%s", kind);
    *out = lana_value_string(text);
    return LANA_OK;
}

int lana_bridge_run_pipeline(const char *labc_path, const char *request_path,
                             const char *response_path,
                             const LanaBridgeOptions *options,
                             const LanaBridgePipelineConfig *config,
                             char **envelope_json) {
    char *response;
    int status;
    LanaVM *vm;
    Value request_value = {0}, input_value = {0}, plan_value = {0}, effect_out = {0};
    LanaStore *store = NULL;
    LanaLedger *ledger = NULL;
    LanaPolicy policy;
    LanaPolicyEvaluation evaluation;
    LanaDecision decision;
    LanaEffectAttempt attempt;
    LanaEvent event;
    LanaStoreOptions store_options;
    LanaEventInput event_input;
    LanaStoreRevisionInfo revision;
    LanaError error;
    char *json = NULL;
    const char *target, *scope, *reason, *evidence_ids, *derivation_ids;
    const char *relationship_resolution, *requested_evidence;
    uint64_t decision_id;
    unsigned char digest[32];
    char version[65], event_reason[64];
    size_t needed, index;
    if (envelope_json == NULL) return LANA_BRIDGE_ERR_ARGUMENT;
    *envelope_json = NULL;
    if (labc_path == NULL || request_path == NULL || response_path == NULL)
        return fail(envelope_json, LANA_BRIDGE_ERR_ARGUMENT, "input",
                    "LANA_BRIDGE_ARGUMENT", "paths must not be null");
    if (options != NULL && (options->struct_size < sizeof(*options) ||
                            options->abi_version != LANA_BRIDGE_ABI_VERSION))
        return fail(envelope_json, LANA_BRIDGE_ERR_ABI, "compatibility",
                    "LANA_BRIDGE_ABI", "unsupported bridge options ABI");
    if (config == NULL || config->struct_size < sizeof(*config) ||
        config->abi_version != LANA_BRIDGE_ABI_VERSION ||
        config->store_path == NULL || config->policy_id == NULL)
        return fail(envelope_json, LANA_BRIDGE_ERR_ABI, "compatibility",
                    "LANA_BRIDGE_ABI", "unsupported pipeline config ABI");

    response = run_program(labc_path, request_path, response_path, options,
                           &status, envelope_json);
    if (response == NULL) return status;

    vm = lana_vm_create();
    if (vm == NULL) { free(response); return fail(envelope_json, LANA_BRIDGE_ERR_OOM,
        "pipeline", "LANA_BRIDGE_OOM", "out of memory"); }

    error = lana_json_parse(vm, response, &request_value);
    if (error != LANA_OK || request_value.type != VAL_MAP) {
        free(response); lana_vm_destroy(vm);
        return fail(envelope_json, LANA_BRIDGE_ERR_PROTOCOL, "pipeline",
                    "LANA_REQUEST_INVALID", "program wrote invalid decision-request JSON");
    }

    /* Deterministic decision id from the request text. */
    lana_sha256((const unsigned char *)response, strlen(response), digest);
    decision_id = 0u;
    for (index = 0u; index < 8u; ++index)
        decision_id |= (uint64_t)digest[index] << (8u * index);
    if (decision_id == 0u) decision_id = 1u;
    free(response);

    target = map_string(&request_value, "target");
    scope = map_string(&request_value, "scope");
    reason = map_string(&request_value, "reason");
    if (target == NULL || scope == NULL || reason == NULL) {
        lana_vm_destroy(vm);
        return fail(envelope_json, LANA_BRIDGE_ERR_PROTOCOL, "pipeline",
                    "LANA_REQUEST_INVALID", "decision-request missing target/scope/reason");
    }
    evidence_ids = map_string(&request_value, "evidence_ids");
    derivation_ids = map_string(&request_value, "derivation_ids");
    relationship_resolution = map_string(&request_value, "relationship_resolution");
    requested_evidence = map_string(&request_value, "requested_evidence");
    if (lana_map_get(request_value.as.map, "input", &input_value) != LANA_OK ||
        input_value.type != VAL_MAP) {
        lana_vm_destroy(vm);
        return fail(envelope_json, LANA_BRIDGE_ERR_PROTOCOL, "pipeline",
                    "LANA_REQUEST_INVALID", "decision-request missing input map");
    }

    memset(&store_options, 0, sizeof(store_options));
    store_options.struct_size = sizeof(store_options);
    store_options.schema_version = 1u;
    store_options.path = config->store_path;
    error = lana_store_open(&store_options, &store);
    if (error != LANA_OK) {
        lana_vm_destroy(vm);
        return fail(envelope_json, (int)error, "pipeline", lana_error_name(error),
                    "failed to open store");
    }

    memset(&policy, 0, sizeof(policy));
    policy.struct_size = sizeof(policy);
    policy.schema_version = 1u;
    policy.policy_id = config->policy_id;
    policy.rule = config->rule;
    error = lana_policy_store(store, vm, &policy);
    if (error != LANA_OK && error != LANA_ERR_CONFLICT) {
        (void)lana_store_close(store); lana_vm_destroy(vm);
        return fail(envelope_json, (int)error, "pipeline", lana_error_name(error),
                    "failed to store policy");
    }

    error = lana_store_current_revision(store, &revision);
    if (error != LANA_OK) {
        (void)lana_store_close(store); lana_vm_destroy(vm);
        return fail(envelope_json, (int)error, "pipeline", lana_error_name(error),
                    "failed to read store revision");
    }
    memset(&evaluation, 0, sizeof(evaluation));
    evaluation.struct_size = sizeof(evaluation);
    evaluation.schema_version = 1u;
    evaluation.decision_id = decision_id;
    evaluation.target = target;
    evaluation.scope = scope;
    evaluation.input_revision = revision.revision_id;
    evaluation.evidence_ids = evidence_ids;
    evaluation.derivation_ids = derivation_ids;
    evaluation.relationship_resolution = relationship_resolution;
    evaluation.evaluation_time = (uint64_t)time(NULL);
    evaluation.reason = reason;
    evaluation.requested_evidence = requested_evidence;

    error = lana_policy_evaluate(&policy, &input_value, &evaluation, &decision);
    if (error != LANA_OK) {
        (void)lana_store_close(store); lana_vm_destroy(vm);
        return fail(envelope_json, (int)error, "pipeline", lana_error_name(error),
                    "policy evaluation failed");
    }
    error = lana_policy_store_decision(store, vm, &decision);
    if (error != LANA_OK) {
        (void)lana_store_close(store); lana_vm_destroy(vm);
        return fail(envelope_json, (int)error, "pipeline", lana_error_name(error),
                    "failed to store decision");
    }
    error = lana_store_commit(store, &revision);
    if (error != LANA_OK) {
        (void)lana_store_close(store); lana_vm_destroy(vm);
        return fail(envelope_json, (int)error, "pipeline", lana_error_name(error),
                    "failed to commit decision");
    }

    if (decision.outcome == POLICY_OUTCOME_AUTHORIZE) {
        error = lana_vm_planned_effect(vm, decision.effect, &input_value, &plan_value);
        if (error != LANA_OK) {
            (void)lana_store_close(store); lana_vm_destroy(vm);
            return fail(envelope_json, (int)error, "pipeline", lana_error_name(error),
                        "failed to build planned effect");
        }
        error = lana_effect_execute_attempt(store, vm, &decision, decision_id,
                                            &plan_value, default_executor, NULL,
                                            &effect_out, &attempt);
        if (error != LANA_OK) {
            (void)lana_store_close(store); lana_vm_destroy(vm);
            return fail(envelope_json, (int)error, "pipeline", lana_error_name(error),
                        "effect attempt failed");
        }
        error = lana_ledger_open(store, &ledger);
        if (error != LANA_OK) {
            (void)lana_store_close(store); lana_vm_destroy(vm);
            return fail(envelope_json, (int)error, "pipeline", lana_error_name(error),
                        "failed to open ledger");
        }
        memset(&event_input, 0, sizeof(event_input));
        event_input.struct_size = sizeof(event_input);
        event_input.schema_version = 1u;
        event_input.entity = decision.target;
        event_input.actor = "lana-bridge";
        event_input.action = decision.effect;
        (void)snprintf(event_reason, sizeof(event_reason), "decision %llu",
                       (unsigned long long)decision.decision_id);
        event_input.reason = event_reason;
        event_input.timestamp = (uint64_t)time(NULL);
        error = lana_ledger_append(ledger, &event_input, &event);
        (void)lana_ledger_close(ledger);
        if (error != LANA_OK) {
            (void)lana_store_close(store); lana_vm_destroy(vm);
            return fail(envelope_json, (int)error, "pipeline", lana_error_name(error),
                        "ledger append failed");
        }
    }

    error = lana_store_current_revision(store, &revision);
    if (error != LANA_OK) {
        (void)lana_store_close(store); lana_vm_destroy(vm);
        return fail(envelope_json, (int)error, "pipeline", lana_error_name(error),
                    "failed to read final revision");
    }

    {
        char *escaped_target = json_escape(decision.target);
        char *escaped_scope = json_escape(decision.scope);
        char *escaped_reason = json_escape(decision.reason);
        char *escaped_policy = json_escape(decision.policy_id);
        hex_digest(decision.policy_version, version);
        if (escaped_target == NULL || escaped_scope == NULL || escaped_reason == NULL ||
            escaped_policy == NULL) {
            free(escaped_target); free(escaped_scope); free(escaped_reason); free(escaped_policy);
            (void)lana_store_close(store); lana_vm_destroy(vm);
            return fail(envelope_json, LANA_BRIDGE_ERR_OOM, "pipeline",
                        "LANA_BRIDGE_OOM", "out of memory");
        }
        if (decision.outcome == POLICY_OUTCOME_AUTHORIZE) {
            const char *attempt_status =
                attempt.status == EFFECT_STATUS_SUCCEEDED ? "succeeded" :
                attempt.status == EFFECT_STATUS_FAILED ? "failed" :
                attempt.status == EFFECT_STATUS_CANCELLED ? "cancelled" : "pending";
            needed = strlen(escaped_target) + strlen(escaped_scope) + strlen(escaped_reason) +
                     strlen(escaped_policy) + strlen(version) + strlen(attempt_status) + 512u;
            json = malloc(needed);
            if (json != NULL)
                (void)snprintf(json, needed,
                    "{\"schema\":1,\"ok\":true,\"result\":{\"decision_id\":%llu,\"outcome\":\"authorize\","
                    "\"attempt_status\":\"%s\",\"event_id\":%llu,\"revision\":%llu,"
                    "\"policy_id\":\"%s\",\"policy_version\":\"%s\",\"target\":\"%s\",\"scope\":\"%s\",\"reason\":\"%s\"},"
                    "\"stdout\":\"\",\"stderr\":\"\",\"execution\":{\"engine\":\"native\",\"lana_version\":\"%s\"}}",
                    (unsigned long long)decision.decision_id, attempt_status,
                    (unsigned long long)event.event_id, (unsigned long long)revision.revision_id,
                    escaped_policy, version, escaped_target, escaped_scope, escaped_reason,
                    LANA_VERSION);
        } else {
            const char *outcome_name =
                decision.outcome == POLICY_OUTCOME_REFUSE ? "refuse" : "request_more_evidence";
            needed = strlen(escaped_target) + strlen(escaped_scope) + strlen(escaped_reason) +
                     strlen(escaped_policy) + strlen(version) + 512u;
            json = malloc(needed);
            if (json != NULL)
                (void)snprintf(json, needed,
                    "{\"schema\":1,\"ok\":true,\"result\":{\"decision_id\":%llu,\"outcome\":\"%s\","
                    "\"revision\":%llu,"
                    "\"policy_id\":\"%s\",\"policy_version\":\"%s\",\"target\":\"%s\",\"scope\":\"%s\",\"reason\":\"%s\"},"
                    "\"stdout\":\"\",\"stderr\":\"\",\"execution\":{\"engine\":\"native\",\"lana_version\":\"%s\"}}",
                    (unsigned long long)decision.decision_id, outcome_name,
                    (unsigned long long)revision.revision_id,
                    escaped_policy, version, escaped_target, escaped_scope, escaped_reason,
                    LANA_VERSION);
        }
        free(escaped_target); free(escaped_scope); free(escaped_reason); free(escaped_policy);
    }
    (void)lana_store_close(store);
    lana_vm_destroy(vm);
    if (json == NULL) return fail(envelope_json, LANA_BRIDGE_ERR_OOM, "pipeline",
                                  "LANA_BRIDGE_OOM", "out of memory");
    *envelope_json = json;
    return LANA_BRIDGE_OK;
}

void lana_bridge_free(char *value) { free(value); }

const char *lana_bridge_version(void) { return LANA_VERSION; }
