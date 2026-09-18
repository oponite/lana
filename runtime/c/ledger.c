#include "ledger.h"

#include "data.h"
#include "store.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct LanaLedger {
    LanaStore *store;
    FILE *journal;
};

static bool text_ok(const char *text) { return text != NULL && text[0] != '\0'; }

static LanaError map_text(const LanaMap *map, const char *key, const char **out) {
    Value value;
    if (lana_map_get(map, key, &value) != LANA_OK || value.type != VAL_STRING || !text_ok(value.as.string)) return LANA_ERR_CORRUPTION;
    *out = value.as.string; return LANA_OK;
}

static LanaError map_number(const LanaMap *map, const char *key, uint64_t *out) {
    Value value;
    if (lana_map_get(map, key, &value) != LANA_OK || value.type != VAL_NUMBER) return LANA_ERR_CORRUPTION;
    *out = (uint64_t)value.as.number; return LANA_OK;
}

LanaError lana_ledger_open(LanaStore *store, LanaLedger **out) {
    LanaLedger *ledger;
    LanaError error;
    char *path;
    if (store == NULL || out == NULL) return LANA_ERR_INVALID_STATE;
    ledger = calloc(1u, sizeof(*ledger)); if (ledger == NULL) return LANA_ERR_OOM;
    ledger->store = store;

    if ((error = lana_store_get_path(store, &path)) != LANA_OK) {
        free(ledger); return error;
    }

    char journal_path[1024];
    (void)snprintf(journal_path, sizeof(journal_path), "%s/ledger.journal", path);
    free(path);

    ledger->journal = fopen(journal_path, "a+b");
    if (ledger->journal == NULL) {
        free(ledger); return LANA_ERR_IO;
    }

    *out = ledger; return LANA_OK;
}

LanaError lana_ledger_close(LanaLedger *ledger) {
    if (ledger == NULL) return LANA_OK;
    if (ledger->journal != NULL) fclose(ledger->journal);
    free(ledger); return LANA_OK;
}

LanaError lana_ledger_append(LanaLedger *ledger, const LanaEventInput *input, LanaEvent *out_event) {
    LanaStoreRevisionInfo revision; char key[64], json[2048]; Value value; LanaError error;
    if (ledger == NULL || input == NULL || input->struct_size < sizeof(*input) || input->schema_version != 1u ||
        !text_ok(input->entity) || !text_ok(input->actor) || !text_ok(input->action)) return LANA_ERR_SCHEMA;
    if (input->reason != NULL && strchr(input->reason, '"') != NULL) return LANA_ERR_SCHEMA;
    if (lana_store_current_revision(ledger->store, &revision) != LANA_OK || revision.revision_id == UINT64_MAX) return LANA_ERR_IO;
    (void)snprintf(key, sizeof(key), "event/%llu", (unsigned long long)(revision.revision_id + 1u));
    (void)snprintf(json, sizeof(json), "{\"action\":\"%s\",\"actor\":\"%s\",\"correction_of\":%llu,\"entity\":\"%s\",\"reason\":\"%s\",\"timestamp\":%llu}",
                   input->action, input->actor, (unsigned long long)input->correction_of, input->entity,
                   input->reason == NULL ? "" : input->reason, (unsigned long long)input->timestamp);
    if (strlen(json) >= sizeof(json) - 1u) return LANA_ERR_LIMIT;

    /* Write to separate append-only journal first. */
    if (ledger->journal != NULL) {
        if (fprintf(ledger->journal, "%s\n", json) < 0 || fflush(ledger->journal) != 0) return LANA_ERR_IO;
    }

    value = lana_value_string(json);
    if ((error = lana_store_put(ledger->store, key, value)) != LANA_OK) return error;
    if ((error = lana_store_commit(ledger->store, &revision)) != LANA_OK) return error;
    if (out_event != NULL) {
        memset(out_event, 0, sizeof(*out_event)); out_event->struct_size = sizeof(*out_event); out_event->schema_version = 1u;
        out_event->event_id = revision.revision_id; out_event->revision = revision.revision_id;
        out_event->entity = input->entity; out_event->actor = input->actor; out_event->action = input->action;
        out_event->reason = input->reason == NULL ? "" : input->reason; out_event->timestamp = input->timestamp;
        out_event->correction_of = input->correction_of;
    }
    return LANA_OK;
}

/* Scan the event journal, applying filters. Returns LANA_OK with count possibly
 * zero, or a distinct corruption/I/O error. */
static LanaError scan_events(LanaLedger *ledger, LanaVM *vm,
                             const char *entity, const char *actor, const char *action,
                             uint64_t start_timestamp, uint64_t end_timestamp,
                             LanaEvent **out_events, size_t *out_count) {
    LanaStoreRevisionInfo revision; LanaEvent *events; size_t count = 0u, index;
    if (lana_store_current_revision(ledger->store, &revision) != LANA_OK) return LANA_ERR_IO;
    events = calloc((size_t)revision.revision_id, sizeof(*events));
    if (revision.revision_id != 0u && events == NULL) return LANA_ERR_OOM;
    for (index = 1u; index <= revision.revision_id; ++index) {
        char key[64]; Value raw, parsed; LanaEvent event; const char *text; LanaError get_error;
        (void)snprintf(key, sizeof(key), "event/%llu", (unsigned long long)index);
        get_error = lana_store_get(ledger->store, vm, key, &raw);
        if (get_error == LANA_ERR_NOT_FOUND) continue;
        if (get_error != LANA_OK) { free(events); return get_error; }
        if (raw.type != VAL_STRING) { free(events); return LANA_ERR_CORRUPTION; }
        if (lana_json_parse(vm, raw.as.string, &parsed) != LANA_OK || parsed.type != VAL_MAP) { free(events); return LANA_ERR_CORRUPTION; }
        memset(&event, 0, sizeof(event)); event.struct_size = sizeof(event); event.schema_version = 1u; event.event_id = index; event.revision = index;
        if (map_text(parsed.as.map, "entity", &text) != LANA_OK) { free(events); return LANA_ERR_CORRUPTION; } event.entity = text;
        if (map_text(parsed.as.map, "actor", &text) != LANA_OK) { free(events); return LANA_ERR_CORRUPTION; } event.actor = text;
        if (map_text(parsed.as.map, "action", &text) != LANA_OK) { free(events); return LANA_ERR_CORRUPTION; } event.action = text;
        if (map_text(parsed.as.map, "reason", &text) != LANA_OK) { free(events); return LANA_ERR_CORRUPTION; } event.reason = text;
        if (map_number(parsed.as.map, "timestamp", &event.timestamp) != LANA_OK) { free(events); return LANA_ERR_CORRUPTION; }
        if (entity != NULL && strcmp(entity, event.entity) != 0) continue;
        if (actor != NULL && strcmp(actor, event.actor) != 0) continue;
        if (action != NULL && strcmp(action, event.action) != 0) continue;
        if (start_timestamp != 0u && event.timestamp < start_timestamp) continue;
        if (end_timestamp != 0u && event.timestamp > end_timestamp) continue;
        events[count++] = event;
    }
    *out_events = events; *out_count = count; return LANA_OK;
}

LanaError lana_ledger_query(LanaLedger *ledger, LanaVM *vm, const LanaLedgerQuery *query,
                            LanaEvent **out_events, size_t *out_count) {
    LanaError error;
    if (ledger == NULL || vm == NULL || query == NULL || query->struct_size < sizeof(*query) || query->schema_version != 1u ||
        out_events == NULL || out_count == NULL || (query->start_timestamp > query->end_timestamp && query->end_timestamp != 0u)) return LANA_ERR_INVALID_STATE;
    error = scan_events(ledger, vm, query->entity, query->actor, query->action,
                        query->start_timestamp, query->end_timestamp, out_events, out_count);
    if (error != LANA_OK) return error;
    return *out_count == 0u ? LANA_ERR_NO_MATCHING_EVENT : LANA_OK;
}

void lana_ledger_events_free(LanaEvent *events, size_t count) { (void)count; free(events); }

LanaError lana_ledger_query_coverage(LanaLedger *ledger, LanaVM *vm,
                                     const LanaLedgerCoverageQuery *query,
                                     LanaLedgerCoverageEntry **out_entries,
                                     size_t *out_count) {
    LanaLedgerCoverageEntry *entries; size_t index;
    if (ledger == NULL || vm == NULL || query == NULL || query->struct_size < sizeof(*query) || query->schema_version != 1u ||
        out_entries == NULL || out_count == NULL || query->target_entities == NULL || query->target_count == 0u ||
        (query->start_timestamp > query->end_timestamp && query->end_timestamp != 0u)) return LANA_ERR_INVALID_STATE;
    entries = calloc(query->target_count, sizeof(*entries));
    if (entries == NULL) return LANA_ERR_OOM;
    for (index = 0u; index < query->target_count; ++index) {
        LanaEvent *events = NULL; size_t count = 0u; LanaError error;
        if (query->target_entities[index] == NULL) { free(entries); return LANA_ERR_SCHEMA; }
        error = scan_events(ledger, vm, query->target_entities[index], query->actor, query->action,
                            query->start_timestamp, query->end_timestamp, &events, &count);
        if (error != LANA_OK) { free(events); free(entries); return error; }
        entries[index].struct_size = sizeof(entries[index]); entries[index].schema_version = 1u;
        entries[index].entity = query->target_entities[index];
        entries[index].matched = count != 0u;
        entries[index].events = events; entries[index].event_count = count;
    }
    *out_entries = entries; *out_count = query->target_count; return LANA_OK;
}

void lana_ledger_coverage_free(LanaLedgerCoverageEntry *entries, size_t count) {
    size_t index;
    if (entries == NULL) return;
    for (index = 0u; index < count; ++index) free(entries[index].events);
    free(entries);
}

static int hex_value(char character) {
    if (character >= '0' && character <= '9') return character - '0';
    if (character >= 'a' && character <= 'f') return character - 'a' + 10;
    if (character >= 'A' && character <= 'F') return character - 'A' + 10;
    return -1;
}

static LanaError hex_decode(const char *text, unsigned char out[32]) {
    size_t index;
    if (text == NULL || strlen(text) != 64u) return LANA_ERR_CORRUPTION;
    for (index = 0u; index < 32u; ++index) {
        int high = hex_value(text[index * 2u]);
        int low = hex_value(text[index * 2u + 1u]);
        if (high < 0 || low < 0) return LANA_ERR_CORRUPTION;
        out[index] = (unsigned char)((high << 4) | low);
    }
    return LANA_OK;
}

/* Parse the trailing numeric id from a key of the form "prefix/{id}" or
 * "prefix/{id}/{attempt}". Returns 0 on malformed input. */
static uint64_t key_tail_id(const char *key) {
    const char *slash = strrchr(key, '/');
    if (slash == NULL || slash[1] == '\0') return 0u;
    return (uint64_t)strtoull(slash + 1, NULL, 10);
}

/* Stored records are JSON text (VAL_STRING); decode to a map. */
static LanaError scan_value_map(LanaVM *vm, const Value *value, Value *out) {
    if (value->type == VAL_MAP) { *out = *value; return LANA_OK; }
    if (value->type != VAL_STRING) return LANA_ERR_CORRUPTION;
    if (lana_json_parse(vm, value->as.string, out) != LANA_OK || out->type != VAL_MAP)
        return LANA_ERR_CORRUPTION;
    return LANA_OK;
}

LanaError lana_ledger_traverse(LanaLedger *ledger, LanaVM *vm,
                               const LanaTraversalQuery *query,
                               LanaTraversalRecord **out_records,
                               size_t *out_count) {
    LanaStoreScanRecord *decisions = NULL, *events = NULL, *attempts = NULL;
    size_t decision_count = 0u, event_count = 0u, attempt_count = 0u;
    LanaTraversalRecord *records = NULL; size_t count = 0u, capacity = 0u;
    size_t decision_index, event_index;
    LanaError error;
    if (ledger == NULL || vm == NULL || query == NULL || query->struct_size < sizeof(*query) || query->schema_version != 1u ||
        query->evidence_id == NULL || out_records == NULL || out_count == NULL) return LANA_ERR_INVALID_STATE;
    error = lana_store_scan(ledger->store, vm, "decision/", &decisions, &decision_count);
    if (error != LANA_OK) return error;
    error = lana_store_scan(ledger->store, vm, "event/", &events, &event_count);
    if (error != LANA_OK) goto done;
    for (decision_index = 0u; decision_index < decision_count; ++decision_index) {
        Value parsed; const char *text; uint64_t decision_id, outcome_value;
        LanaTraversalRecord record; char decision_text[32];
        if (scan_value_map(vm, &decisions[decision_index].value, &parsed) != LANA_OK) { error = LANA_ERR_CORRUPTION; goto done; }
        if (map_text(parsed.as.map, "evidence_ids", &text) != LANA_OK) { error = LANA_ERR_CORRUPTION; goto done; }
        if (strstr(text, query->evidence_id) == NULL) continue;
        memset(&record, 0, sizeof(record)); record.struct_size = sizeof(record); record.schema_version = 1u;
        if (map_number(parsed.as.map, "decision_id", &decision_id) != LANA_OK) { error = LANA_ERR_CORRUPTION; goto done; }
        record.decision_id = decision_id;
        if (map_text(parsed.as.map, "policy_id", &text) != LANA_OK) { error = LANA_ERR_CORRUPTION; goto done; }
        record.policy_id = strdup(text);
        if (record.policy_id == NULL) { error = LANA_ERR_OOM; goto done; }
        if (map_text(parsed.as.map, "policy_version", &text) != LANA_OK) { error = LANA_ERR_CORRUPTION; goto done; }
        if (hex_decode(text, record.policy_version) != LANA_OK) { error = LANA_ERR_CORRUPTION; goto done; }
        if (map_number(parsed.as.map, "outcome", &outcome_value) != LANA_OK) { error = LANA_ERR_CORRUPTION; goto done; }
        record.outcome = (PolicyOutcome)(unsigned)outcome_value;
        if (map_text(parsed.as.map, "effect", &text) != LANA_OK) { error = LANA_ERR_CORRUPTION; goto done; }
        record.effect = strdup(text);
        if (record.effect == NULL) { error = LANA_ERR_OOM; goto done; }
        /* Link the effect attempt for this decision (lowest attempt id). */
        {
            char attempt_prefix[64];
            (void)snprintf(attempt_prefix, sizeof(attempt_prefix), "effect-attempt/%llu/",
                           (unsigned long long)decision_id);
            error = lana_store_scan(ledger->store, vm, attempt_prefix, &attempts, &attempt_count);
            if (error != LANA_OK) goto done;
            if (attempt_count != 0u) {
                Value attempt_parsed; uint64_t status_value;
                if (scan_value_map(vm, &attempts[0].value, &attempt_parsed) != LANA_OK) { error = LANA_ERR_CORRUPTION; goto done; }
                if (map_number(attempt_parsed.as.map, "status", &status_value) != LANA_OK) { error = LANA_ERR_CORRUPTION; goto done; }
                record.attempt_id = key_tail_id(attempts[0].key);
                record.attempt_status = (EffectStatus)(unsigned)status_value;
            }
            lana_store_scan_free(attempts, attempt_count); attempts = NULL; attempt_count = 0u;
        }
        /* Link the event referencing this decision. */
        (void)snprintf(decision_text, sizeof(decision_text), "%llu", (unsigned long long)decision_id);
        for (event_index = 0u; event_index < event_count; ++event_index) {
            Value event_parsed; const char *reason;
            if (scan_value_map(vm, &events[event_index].value, &event_parsed) != LANA_OK) { error = LANA_ERR_CORRUPTION; goto done; }
            if (map_text(event_parsed.as.map, "reason", &reason) != LANA_OK) { error = LANA_ERR_CORRUPTION; goto done; }
            if (strstr(reason, decision_text) != NULL) { record.event_id = key_tail_id(events[event_index].key); break; }
        }
        if (count == capacity) {
            size_t new_capacity = capacity == 0u ? 4u : capacity * 2u;
            LanaTraversalRecord *grown = realloc(records, new_capacity * sizeof(*grown));
            if (grown == NULL) { error = LANA_ERR_OOM; goto done; }
            records = grown; capacity = new_capacity;
        }
        records[count++] = record;
    }
    if (count == 0u) error = LANA_ERR_NO_MATCHING_EVENT;
done:
    lana_store_scan_free(decisions, decision_count);
    lana_store_scan_free(events, event_count);
    lana_store_scan_free(attempts, attempt_count);
    if (error != LANA_OK) { lana_traversal_records_free(records, count); return error; }
    *out_records = records; *out_count = count; return LANA_OK;
}

void lana_traversal_records_free(LanaTraversalRecord *records, size_t count) {
    size_t index;
    if (records == NULL) return;
    for (index = 0u; index < count; ++index) {
        free((void *)records[index].policy_id);
        free((void *)records[index].effect);
    }
    free(records);
}
