#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "lana/state_codec.h"
#include "lana/error.h"

static void test_state_roundtrip(void) {
    printf("Testing STATE round-trip...\n");
    LanaState state = {
        .p = 0.75,
        .d_re = 0.12,
        .d_im = 0.34
    };

    void *buf = NULL;
    size_t len = 0;
    assert(lana_state_encode(state, &buf, &len) == LANA_OK);
    assert(buf != NULL);
    assert(len > 0);

    LanaState decoded;
    assert(lana_state_decode(buf, len, &decoded) == LANA_OK);
    assert(decoded.p == state.p);
    assert(decoded.d_re == state.d_re);
    assert(decoded.d_im == state.d_im);
    ((unsigned char *)buf)[4] = 2u;
    assert(lana_state_decode(buf, len, &decoded) == LANA_ERR_SCHEMA);
    assert(lana_state_decode(buf, len - 1u, &decoded) == LANA_ERR_SCHEMA);

    free(buf);
    printf("Pass.\n");
}

static void test_persistent_state_roundtrip(void) {
    LanaPersistentState state = {
        .struct_size = sizeof(LanaPersistentState), .schema_version = 1u,
        .state = {.p = 0.5, .d_re = 0.2, .d_im = 0.1},
        .metadata = {.has_timestamp = true, .timestamp = 12.0, .has_source = true,
                     .source = "sensor", .has_weight = true, .weight = 0.8},
        .has_provenance = true,
        .provenance = {.derivation_id = 42u, .input_revision = 9u,
                       .kind = "evidence", .operation = "observe",
                       .captured_inputs = "sample-1"}
    };
    LanaPersistentState decoded = {0};
    void *buffer = NULL;
    size_t length = 0u;
    assert(lana_persistent_state_encode(&state, &buffer, &length) == LANA_OK);
    assert(lana_persistent_state_decode(buffer, length, &decoded) == LANA_OK);
    assert(decoded.metadata.has_source && strcmp(decoded.metadata.source, "sensor") == 0);
    assert(decoded.has_provenance && decoded.provenance.derivation_id == 42u);
    ((unsigned char *)buffer)[0] = '[';
    assert(lana_persistent_state_decode(buffer, length, &decoded) == LANA_ERR_SCHEMA);
    lana_persistent_state_free(&decoded);
    free(buffer);
}

int main(void) {
    test_state_roundtrip();
    test_persistent_state_roundtrip();
    return 0;
}
