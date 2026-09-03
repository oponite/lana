#include "lana/state_codec.h"
#include "lana/codec.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LANA_STATE_CODEC_SCHEMA 1u
#define LANA_STATE_CODEC_LENGTH 32u

static void put_u32(unsigned char *out, uint32_t value) {
    out[0] = (unsigned char)(value >> 24u); out[1] = (unsigned char)(value >> 16u);
    out[2] = (unsigned char)(value >> 8u); out[3] = (unsigned char)value;
}

static uint32_t get_u32(const unsigned char *in) {
    return ((uint32_t)in[0] << 24u) | ((uint32_t)in[1] << 16u) |
           ((uint32_t)in[2] << 8u) | (uint32_t)in[3];
}

static void put_double(unsigned char *out, double value) {
    uint64_t bits; size_t index;
    memcpy(&bits, &value, sizeof(bits));
    for (index = 0u; index < 8u; ++index) out[7u - index] = (unsigned char)(bits >> (index * 8u));
}

static double get_double(const unsigned char *in) {
    uint64_t bits = 0u; double value; size_t index;
    for (index = 0u; index < 8u; ++index) bits = (bits << 8u) | in[index];
    memcpy(&value, &bits, sizeof(value));
    return value;
}

LanaError lana_state_encode(LanaState state, void **out_buf, size_t *out_len) {
    unsigned char *out;
    if (out_buf == NULL || out_len == NULL || !lana_state_valid(&state)) return LANA_ERR_INVALID_STATE;
    out = malloc(LANA_STATE_CODEC_LENGTH);
    if (out == NULL) return LANA_ERR_OOM;
    memcpy(out, "LST1", 4u); put_u32(out + 4u, LANA_STATE_CODEC_SCHEMA);
    put_double(out + 8u, state.p); put_double(out + 16u, state.d_re); put_double(out + 24u, state.d_im);
    *out_buf = out; *out_len = LANA_STATE_CODEC_LENGTH;
    return LANA_OK;
}

LanaError lana_state_decode(const void *buf, size_t len, LanaState *out_state) {
    const unsigned char *in = buf;
    LanaState state;
    if (buf == NULL || out_state == NULL) return LANA_ERR_INVALID_STATE;
    if (len != LANA_STATE_CODEC_LENGTH || memcmp(in, "LST1", 4u) != 0 ||
        get_u32(in + 4u) != LANA_STATE_CODEC_SCHEMA) return LANA_ERR_SCHEMA;
    state.p = get_double(in + 8u); state.d_re = get_double(in + 16u); state.d_im = get_double(in + 24u);
    if (!lana_state_valid(&state)) return LANA_ERR_SCHEMA;
    *out_state = state;
    return LANA_OK;
}

static Value map_find_value(const LanaMap *map, const char *key) {
    size_t index;
    for (index = 0u; index < map->count; ++index)
        if (strcmp(map->entries[index].key, key) == 0) return *map->entries[index].value;
    return lana_value_null();
}

static bool map_has_exact_keys(const LanaMap *map, const char *const *keys, size_t count) {
    size_t index;
    if (map == NULL || map->count != count) return false;
    for (index = 0u; index < count; ++index) {
        size_t found;
        for (found = 0u; found < map->count; ++found)
            if (strcmp(map->entries[found].key, keys[index]) == 0) break;
        if (found == map->count) return false;
    }
    return true;
}

static void decoded_free(Value value) {
    size_t index;
    if (value.type == VAL_STRING) free((void *)value.as.string);
    else if (value.type == VAL_MAP && value.as.map != NULL) {
        for (index = 0u; index < value.as.map->count; ++index) {
            free((void *)value.as.map->entries[index].key);
            decoded_free(*value.as.map->entries[index].value);
            free(value.as.map->entries[index].value);
        }
        free(value.as.map->entries); free(value.as.map);
    }
}

static LanaError parse_u64(const Value *value, uint64_t *out) {
    char *end;
    unsigned long long parsed;
    if (value->type != VAL_STRING || value->as.string == NULL) return LANA_ERR_SCHEMA;
    errno = 0; parsed = strtoull(value->as.string, &end, 10);
    if (errno != 0 || *value->as.string == '\0' || *end != '\0') return LANA_ERR_SCHEMA;
    *out = (uint64_t)parsed;
    return LANA_OK;
}

void lana_persistent_state_free(LanaPersistentState *state) {
    if (state == NULL) return;
    free((void *)state->metadata.source);
    free((void *)state->provenance.kind);
    free((void *)state->provenance.operation);
    free((void *)state->provenance.captured_inputs);
    memset(state, 0, sizeof(*state));
}

LanaError lana_persistent_state_encode(const LanaPersistentState *state,
                                       void **out_buf, size_t *out_len) {
    Value metadata_values[4], provenance_values[5], top_values[6];
    LanaMapEntry metadata_entries[4], provenance_entries[5], top_entries[6];
    LanaMap metadata = {4u, 4u, metadata_entries};
    LanaMap provenance = {5u, 5u, provenance_entries};
    LanaMap top = {6u, 6u, top_entries};
    char derivation_id[32], input_revision[32];
    if (state == NULL || out_buf == NULL || out_len == NULL ||
        state->struct_size < sizeof(*state) || state->schema_version != LANA_STATE_CODEC_SCHEMA ||
        !lana_state_valid(&state->state)) return LANA_ERR_INVALID_STATE;
    if ((state->metadata.has_source && state->metadata.source == NULL) ||
        (state->has_provenance && (state->provenance.kind == NULL || state->provenance.operation == NULL ||
                                   state->provenance.captured_inputs == NULL))) return LANA_ERR_SCHEMA;
    metadata_values[0] = state->metadata.has_confidence ? lana_value_number(state->metadata.confidence) : lana_value_null();
    metadata_values[1] = state->metadata.has_source ? lana_value_string(state->metadata.source) : lana_value_null();
    metadata_values[2] = state->metadata.has_timestamp ? lana_value_number(state->metadata.timestamp) : lana_value_null();
    metadata_values[3] = state->metadata.has_weight ? lana_value_number(state->metadata.weight) : lana_value_null();
    metadata_entries[0] = (LanaMapEntry){"confidence", &metadata_values[0]};
    metadata_entries[1] = (LanaMapEntry){"source", &metadata_values[1]};
    metadata_entries[2] = (LanaMapEntry){"timestamp", &metadata_values[2]};
    metadata_entries[3] = (LanaMapEntry){"weight", &metadata_values[3]};
    if (state->has_provenance) {
        (void)snprintf(derivation_id, sizeof(derivation_id), "%llu", (unsigned long long)state->provenance.derivation_id);
        (void)snprintf(input_revision, sizeof(input_revision), "%llu", (unsigned long long)state->provenance.input_revision);
        provenance_values[0] = lana_value_string(state->provenance.captured_inputs);
        provenance_values[1] = lana_value_string(derivation_id);
        provenance_values[2] = lana_value_string(input_revision);
        provenance_values[3] = lana_value_string(state->provenance.kind);
        provenance_values[4] = lana_value_string(state->provenance.operation);
        provenance_entries[0] = (LanaMapEntry){"captured_inputs", &provenance_values[0]};
        provenance_entries[1] = (LanaMapEntry){"derivation_id", &provenance_values[1]};
        provenance_entries[2] = (LanaMapEntry){"input_revision", &provenance_values[2]};
        provenance_entries[3] = (LanaMapEntry){"kind", &provenance_values[3]};
        provenance_entries[4] = (LanaMapEntry){"operation", &provenance_values[4]};
    }
    top_values[0] = lana_value_number(state->state.d_im); top_values[1] = lana_value_number(state->state.d_re);
    top_values[2] = lana_value_map(&metadata); top_values[3] = lana_value_number(state->state.p);
    top_values[4] = state->has_provenance ? lana_value_map(&provenance) : lana_value_null();
    top_values[5] = lana_value_number((double)LANA_STATE_CODEC_SCHEMA);
    top_entries[0] = (LanaMapEntry){"d_im", &top_values[0]}; top_entries[1] = (LanaMapEntry){"d_re", &top_values[1]};
    top_entries[2] = (LanaMapEntry){"metadata", &top_values[2]}; top_entries[3] = (LanaMapEntry){"p", &top_values[3]};
    top_entries[4] = (LanaMapEntry){"provenance", &top_values[4]}; top_entries[5] = (LanaMapEntry){"schema", &top_values[5]};
    {
        LanaBuffer buffer = {0}; LanaError error = lana_codec_encode_value(&buffer, lana_value_map(&top));
        if (error != LANA_OK) { free(buffer.data); return error; }
        *out_buf = buffer.data; *out_len = buffer.length;
    }
    return LANA_OK;
}

LanaError lana_persistent_state_decode(const void *buf, size_t len,
                                       LanaPersistentState *out_state) {
    static const char *const top_keys[] = {"schema", "p", "d_re", "d_im", "metadata", "provenance"};
    static const char *const metadata_keys[] = {"timestamp", "source", "weight", "confidence"};
    static const char *const provenance_keys[] = {"derivation_id", "input_revision", "kind", "operation", "captured_inputs"};
    LanaBuffer buffer = {(unsigned char *)buf, len, len};
    Value root = lana_value_null(), value, metadata, provenance;
    LanaPersistentState decoded = {0};
    LanaError error;
    if (buf == NULL || out_state == NULL) return LANA_ERR_INVALID_STATE;
    error = lana_codec_decode_document(&buffer, &root);
    if (error != LANA_OK || root.type != VAL_MAP || !map_has_exact_keys(root.as.map, top_keys, 6u)) goto bad;
    value = map_find_value(root.as.map, "schema");
    if (value.type != VAL_NUMBER || value.as.number != (double)LANA_STATE_CODEC_SCHEMA) goto bad;
    value = map_find_value(root.as.map, "p"); if (value.type != VAL_NUMBER) goto bad; decoded.state.p = value.as.number;
    value = map_find_value(root.as.map, "d_re"); if (value.type != VAL_NUMBER) goto bad; decoded.state.d_re = value.as.number;
    value = map_find_value(root.as.map, "d_im"); if (value.type != VAL_NUMBER) goto bad; decoded.state.d_im = value.as.number;
    if (!lana_state_valid(&decoded.state)) goto bad;
    metadata = map_find_value(root.as.map, "metadata");
    if (metadata.type != VAL_MAP || !map_has_exact_keys(metadata.as.map, metadata_keys, 4u)) goto bad;
    value = map_find_value(metadata.as.map, "timestamp"); if (value.type != VAL_NULL && value.type != VAL_NUMBER) goto bad; decoded.metadata.has_timestamp = value.type == VAL_NUMBER; decoded.metadata.timestamp = value.type == VAL_NUMBER ? value.as.number : 0.0;
    value = map_find_value(metadata.as.map, "weight"); if (value.type != VAL_NULL && value.type != VAL_NUMBER) goto bad; decoded.metadata.has_weight = value.type == VAL_NUMBER; decoded.metadata.weight = value.type == VAL_NUMBER ? value.as.number : 0.0;
    value = map_find_value(metadata.as.map, "confidence"); if (value.type != VAL_NULL && value.type != VAL_NUMBER) goto bad; decoded.metadata.has_confidence = value.type == VAL_NUMBER; decoded.metadata.confidence = value.type == VAL_NUMBER ? value.as.number : 0.0;
    value = map_find_value(metadata.as.map, "source"); if (value.type != VAL_NULL && value.type != VAL_STRING) goto bad; decoded.metadata.has_source = value.type == VAL_STRING; if (decoded.metadata.has_source && (decoded.metadata.source = strdup(value.as.string)) == NULL) { error = LANA_ERR_OOM; goto done; }
    provenance = map_find_value(root.as.map, "provenance");
    if (provenance.type != VAL_NULL) {
        if (provenance.type != VAL_MAP || !map_has_exact_keys(provenance.as.map, provenance_keys, 5u)) goto bad;
        decoded.has_provenance = true;
        value = map_find_value(provenance.as.map, "derivation_id"); if ((error = parse_u64(&value, &decoded.provenance.derivation_id)) != LANA_OK) goto done;
        value = map_find_value(provenance.as.map, "input_revision"); if ((error = parse_u64(&value, &decoded.provenance.input_revision)) != LANA_OK) goto done;
        value = map_find_value(provenance.as.map, "kind"); if (value.type != VAL_STRING) goto bad; decoded.provenance.kind = strdup(value.as.string);
        value = map_find_value(provenance.as.map, "operation"); if (value.type != VAL_STRING) goto bad; decoded.provenance.operation = strdup(value.as.string);
        value = map_find_value(provenance.as.map, "captured_inputs"); if (value.type != VAL_STRING) goto bad; decoded.provenance.captured_inputs = strdup(value.as.string);
        if (decoded.provenance.kind == NULL || decoded.provenance.operation == NULL || decoded.provenance.captured_inputs == NULL) { error = LANA_ERR_OOM; goto done; }
    }
    decoded.struct_size = sizeof(decoded); decoded.schema_version = LANA_STATE_CODEC_SCHEMA;
    *out_state = decoded; memset(&decoded, 0, sizeof(decoded)); error = LANA_OK;
    goto done;
bad:
    error = LANA_ERR_SCHEMA;
done:
    lana_persistent_state_free(&decoded); decoded_free(root);
    return error;
}
