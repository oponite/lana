#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "lana/codec.h"
#include "lana/value.h"
#include "lana/error.h"

static void test_basic_values(void) {
    printf("Testing basic value encoding/decoding...\n");
    LanaBuffer buf = { .data = malloc(1024), .length = 0, .capacity = 1024 };

    Value vals[] = {
        lana_value_null(),
        lana_value_number(3.14),
        lana_value_bool(true),
        lana_value_string("hello lana")
    };

    for (int i = 0; i < 4; i++) {
        buf.length = 0;
        assert(lana_codec_encode_value(&buf, vals[i]) == LANA_OK);

        size_t offset = 0;
        Value decoded;
        assert(lana_codec_decode_value(&buf, &offset, &decoded) == LANA_OK);
        assert(decoded.type == vals[i].type);
        if (decoded.type == VAL_NUMBER) assert(decoded.as.number == vals[i].as.number);
        if (decoded.type == VAL_BOOL) assert(decoded.as.boolean == vals[i].as.boolean);
        if (decoded.type == VAL_STRING) assert(strcmp(decoded.as.string, vals[i].as.string) == 0);
    }

    free(buf.data);
    printf("Pass.\n");
}

static void test_canonical_json_and_rejection(void) {
    LanaBuffer buf = {0};
    LanaMap map = {0};
    Value first = lana_value_number(1.0);
    Value second = lana_value_bool(false);
    LanaMapEntry entries[2] = {
        {"z", &first},
        {"a", &second}
    };
    Value map_value;
    Value decoded;

    map.count = 2;
    map.capacity = 2;
    map.entries = entries;
    map_value = lana_value_map(&map);
    assert(lana_codec_encode_value(&buf, map_value) == LANA_OK);
    assert(buf.length == strlen("{\"a\":false,\"z\":1}"));
    assert(memcmp(buf.data, "{\"a\":false,\"z\":1}", buf.length) == 0);
    assert(lana_codec_decode_document(&buf, &decoded) == LANA_OK);
    free(buf.data);

    buf = (LanaBuffer){(unsigned char *)"1 true", 6u, 6u};
    assert(lana_codec_decode_document(&buf, &decoded) == LANA_ERR_PARSE);
    buf = (LanaBuffer){0};
    assert(lana_codec_encode_value(&buf, lana_value_number(NAN)) == LANA_ERR_UNSUPPORTED_VALUE);
    free(buf.data);
}

int main(void) {
    test_basic_values();
    test_canonical_json_and_rejection();
    return 0;
}
