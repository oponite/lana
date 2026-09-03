#ifndef LANA_STATE_CODEC_H
#define LANA_STATE_CODEC_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "lana/error.h"
#include "lana/state.h"

LanaError lana_state_encode(LanaState state, void **out_buf, size_t *out_len);
LanaError lana_state_decode(const void *buf, size_t len, LanaState *out_state);

typedef struct {
    bool has_timestamp, has_source, has_weight, has_confidence;
    double timestamp, weight, confidence;
    const char *source;
} LanaPersistentStateMetadata;

typedef struct {
    uint64_t derivation_id, input_revision;
    const char *kind;
    const char *operation;
    const char *captured_inputs;
} LanaPersistentStateProvenance;

typedef struct {
    size_t struct_size;
    uint32_t schema_version;
    LanaState state;
    LanaPersistentStateMetadata metadata;
    bool has_provenance;
    LanaPersistentStateProvenance provenance;
} LanaPersistentState;

LanaError lana_persistent_state_encode(const LanaPersistentState *state,
                                       void **out_buf, size_t *out_len);
LanaError lana_persistent_state_decode(const void *buf, size_t len,
                                       LanaPersistentState *out_state);
void lana_persistent_state_free(LanaPersistentState *state);

#endif
