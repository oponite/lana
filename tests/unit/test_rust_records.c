#include "record.h"
#include "error.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    const uint8_t input[] = "{\"record_schema\":1,\"status\":\"unknown\"}";
    LanaRecord *record = NULL;
    LanaRecordBuffer buffer = {sizeof(buffer), NULL, 0};
    assert(lana_record_abi_version() == LANA_RECORD_ABI_VERSION);
    assert(lana_record_parse(input, sizeof(input) - 1, &record) == LANA_OK);
    assert(record != NULL);
    assert(lana_record_json(record, &buffer) == LANA_OK);
    assert(buffer.length == sizeof(input) - 1);
    assert(memcmp(buffer.data, input, buffer.length) == 0);
    lana_record_buffer_free(&buffer);
    lana_record_buffer_free(&buffer);
    assert(buffer.data == NULL && buffer.length == 0);
    lana_record_free(record);
    record = NULL;
    assert(lana_record_parse((const uint8_t *)"{", 1, &record) == LANA_ERR_SCHEMA);
    assert(record == NULL);
    assert(lana_record_parse(input, SIZE_MAX, &record) == LANA_ERR_LIMIT);
    assert(lana_record_json(NULL, &buffer) == LANA_ERR_TYPE);
    buffer.struct_size = 0;
    assert(lana_record_json(NULL, &buffer) == LANA_ERR_TYPE);
    puts("RUST_RECORD_CONSUMER_PASS");
    return 0;
}
