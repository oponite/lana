#ifndef LANA_RECORD_H
#define LANA_RECORD_H

#include <stddef.h>
#include <stdint.h>

#define LANA_RECORD_ABI_VERSION 2u

typedef struct LanaRecord LanaRecord;
typedef struct LanaDecisionRecord LanaDecisionRecord;
typedef struct LanaExecution LanaExecution;
typedef struct LanaValue LanaValue;
typedef struct LanaRecordBuffer {
    size_t struct_size;
    uint8_t *data;
    size_t length;
} LanaRecordBuffer;

uint32_t lana_record_abi_version(void);
int lana_record_parse(const uint8_t *data, size_t length, LanaRecord **out);
int lana_record_json(const LanaRecord *record, LanaRecordBuffer *out);
void lana_record_free(LanaRecord *record);
void lana_record_buffer_free(LanaRecordBuffer *buffer);

int lana_value_inspect_json(const LanaValue *value, LanaRecordBuffer *out);

int lana_decision_record_create(const char *value, const char *reason, LanaDecisionRecord **out);
int lana_decision_record_json(const LanaDecisionRecord *decision, LanaRecordBuffer *out);
void lana_decision_record_free(LanaDecisionRecord *decision);

int lana_execution_create(const char *plan, const char *args_json, LanaExecution **out);
int lana_execution_json(const LanaExecution *execution, LanaRecordBuffer *out);
int lana_execute_effect(const LanaExecution *execution, LanaRecord **out);
void lana_execution_free(LanaExecution *execution);

#endif
