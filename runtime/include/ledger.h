#ifndef LANA_LEDGER_H
#define LANA_LEDGER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "effects.h"
#include "error.h"
#include "policy.h"
#include "store.h"
#include "vm.h"

typedef struct LanaLedger LanaLedger;

typedef struct {
    size_t struct_size;
    uint32_t schema_version;
    const char *entity;
    const char *actor;
    const char *action;
    const char *reason;
    uint64_t timestamp;
    uint64_t correction_of;
} LanaEventInput;

typedef struct {
    size_t struct_size;
    uint32_t schema_version;
    uint64_t event_id;
    const char *entity;
    const char *actor;
    const char *action;
    const char *reason;
    uint64_t timestamp;
    uint64_t revision;
    uint64_t correction_of;
} LanaEvent;

typedef struct {
    size_t struct_size;
    uint32_t schema_version;
    const char *entity;
    const char *actor;
    const char *action;
    uint64_t start_timestamp;
    uint64_t end_timestamp;
} LanaLedgerQuery;

LanaError lana_ledger_open(LanaStore *store, LanaLedger **out_ledger);
LanaError lana_ledger_close(LanaLedger *ledger);
LanaError lana_ledger_append(LanaLedger *ledger, const LanaEventInput *input,
                             LanaEvent *out_event);
LanaError lana_ledger_query(LanaLedger *ledger, LanaVM *vm,
                            const LanaLedgerQuery *query, LanaEvent **out_events,
                            size_t *out_count);
void lana_ledger_events_free(LanaEvent *events, size_t count);

/* Coverage query: returns one entry per target entity, matched or not. */
typedef struct {
    size_t struct_size;
    uint32_t schema_version;
    const char *entity;              /* optional filter (usually NULL for coverage) */
    const char *actor;               /* optional filter */
    const char *action;              /* optional filter */
    uint64_t start_timestamp;
    uint64_t end_timestamp;
    const char *const *target_entities;  /* required: coverage targets */
    size_t target_count;
} LanaLedgerCoverageQuery;

typedef struct {
    size_t struct_size;
    uint32_t schema_version;
    const char *entity;              /* target entity */
    bool matched;                    /* any event matched the filters */
    LanaEvent *events;               /* matching events (caller frees) */
    size_t event_count;
} LanaLedgerCoverageEntry;

LanaError lana_ledger_query_coverage(LanaLedger *ledger, LanaVM *vm,
                                     const LanaLedgerCoverageQuery *query,
                                     LanaLedgerCoverageEntry **out_entries,
                                     size_t *out_count);
void lana_ledger_coverage_free(LanaLedgerCoverageEntry *entries, size_t count);

/* Evidence-to-event traversal: evidence -> decision -> effect attempt -> event. */
typedef struct {
    size_t struct_size;
    uint32_t schema_version;
    const char *evidence_id;
} LanaTraversalQuery;

typedef struct {
    size_t struct_size;
    uint32_t schema_version;
    uint64_t decision_id;
    const char *policy_id;
    unsigned char policy_version[32];
    PolicyOutcome outcome;
    const char *effect;
    uint64_t attempt_id;
    EffectStatus attempt_status;
    uint64_t event_id;
} LanaTraversalRecord;

LanaError lana_ledger_traverse(LanaLedger *ledger, LanaVM *vm,
                               const LanaTraversalQuery *query,
                               LanaTraversalRecord **out_records,
                               size_t *out_count);
void lana_traversal_records_free(LanaTraversalRecord *records, size_t count);

#endif
