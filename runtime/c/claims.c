#include "claims.h"
#include "sha256.h"
#include "vendor/tweetnacl.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

static LanaError append(LanaBuffer *buffer, const void *data, size_t length) {
    unsigned char *grown;
    size_t capacity, needed;
    if (buffer == NULL || (data == NULL && length != 0u)) return LANA_ERR_INVALID_STATE;
    if (length > SIZE_MAX - buffer->length) return LANA_ERR_LIMIT;
    needed = buffer->length + length;
    if (needed > buffer->capacity) {
        capacity = buffer->capacity == 0u ? 128u : buffer->capacity;
        while (capacity < needed) { if (capacity > SIZE_MAX / 2u) { capacity = needed; break; } capacity *= 2u; }
        grown = realloc(buffer->data, capacity);
        if (grown == NULL) return LANA_ERR_OOM;
        buffer->data = grown; buffer->capacity = capacity;
    }
    if (length != 0u) memcpy(buffer->data + buffer->length, data, length);
    buffer->length = needed;
    return LANA_OK;
}

static LanaError put_u32(LanaBuffer *buffer, uint32_t value) {
    unsigned char bytes[4] = {(unsigned char)(value >> 24u), (unsigned char)(value >> 16u),
                              (unsigned char)(value >> 8u), (unsigned char)value};
    return append(buffer, bytes, sizeof(bytes));
}

static LanaError put_u64(LanaBuffer *buffer, uint64_t value) {
    unsigned char bytes[8]; size_t index;
    for (index = 0u; index < sizeof(bytes); ++index) bytes[index] = (unsigned char)(value >> (56u - index * 8u));
    return append(buffer, bytes, sizeof(bytes));
}

static LanaError put_blob(LanaBuffer *buffer, const void *data, size_t length) {
    LanaError error;
    if (length > UINT32_MAX) return LANA_ERR_LIMIT;
    if ((error = put_u32(buffer, (uint32_t)length)) != LANA_OK) return error;
    return append(buffer, data, length);
}

static LanaError put_text(LanaBuffer *buffer, const char *text) {
    if (text == NULL || text[0] == '\0') return LANA_ERR_SCHEMA;
    return put_blob(buffer, text, strlen(text));
}

static bool read_u32(const unsigned char **cursor, const unsigned char *end, uint32_t *out) {
    const unsigned char *p = *cursor;
    if ((size_t)(end - p) < 4u) return false;
    *out = ((uint32_t)p[0] << 24u) | ((uint32_t)p[1] << 16u) | ((uint32_t)p[2] << 8u) | p[3];
    *cursor = p + 4u; return true;
}

static bool read_u64(const unsigned char **cursor, const unsigned char *end, uint64_t *out) {
    const unsigned char *p = *cursor; size_t index; uint64_t value = 0u;
    if ((size_t)(end - p) < 8u) return false;
    for (index = 0u; index < 8u; ++index) value = (value << 8u) | p[index];
    *out = value; *cursor = p + 8u; return true;
}

static LanaError read_blob(const unsigned char **cursor, const unsigned char *end,
                           unsigned char **out, size_t *out_length) {
    uint32_t length;
    unsigned char *copy;
    if (!read_u32(cursor, end, &length) || (size_t)(end - *cursor) < length) return LANA_ERR_CORRUPTION;
    if (length == UINT32_MAX) return LANA_ERR_LIMIT;
    copy = malloc((size_t)length + 1u);
    if (copy == NULL) return LANA_ERR_OOM;
    memcpy(copy, *cursor, length); copy[length] = '\0'; *cursor += length;
    *out = copy; *out_length = length;
    return LANA_OK;
}

static LanaError validate_parameters(LanaRelationshipKind relationship,
                                     const unsigned char *parameters, size_t length) {
    if ((relationship != LANA_RELATION_EXACT && relationship != LANA_RELATION_SAME_DEPENDENCY) ||
        parameters == NULL || length != 2u || memcmp(parameters, "{}", 2u) != 0)
        return LANA_ERR_INVALID_PARAMETERS;
    return LANA_OK;
}

LanaError lana_claim_encode_payload(const LanaRelationshipClaim *claim, LanaBuffer *out) {
    LanaError error;
    if (claim == NULL || out == NULL || out->length != 0u ||
        claim->schema_version != LANA_RELATIONSHIP_CLAIM_SCHEMA_VERSION ||
        claim->validity_start > claim->validity_end ||
        (claim->lifecycle != LANA_CLAIM_ACTIVE && claim->lifecycle != LANA_CLAIM_REVOKED))
        return LANA_ERR_SCHEMA;
    if ((error = validate_parameters(claim->relationship, claim->parameters, claim->parameters_length)) != LANA_OK)
        return error;
    if ((error = append(out, "LCP1", 4u)) != LANA_OK ||
        (error = put_u32(out, claim->schema_version)) != LANA_OK ||
        (error = put_u64(out, claim->claim_id)) != LANA_OK ||
        (error = put_u32(out, claim->version)) != LANA_OK ||
        (error = put_text(out, claim->subject)) != LANA_OK ||
        (error = put_text(out, claim->scope)) != LANA_OK ||
        (error = put_text(out, claim->issuer)) != LANA_OK ||
        (error = put_text(out, claim->issuer_key_id)) != LANA_OK ||
        (error = put_text(out, claim->authority_policy_version)) != LANA_OK ||
        (error = put_text(out, claim->origin)) != LANA_OK ||
        (error = put_u32(out, (uint32_t)claim->relationship)) != LANA_OK ||
        (error = put_blob(out, claim->parameters, claim->parameters_length)) != LANA_OK ||
        (error = put_u64(out, claim->validity_start)) != LANA_OK ||
        (error = put_u64(out, claim->validity_end)) != LANA_OK)
        return error;
    return put_u32(out, (uint32_t)claim->lifecycle);
}

LanaError lana_claim_compute_payload_digest(const LanaRelationshipClaim *claim, unsigned char out_digest[32]) {
    LanaBuffer payload = {0}; LanaError error;
    if (out_digest == NULL) return LANA_ERR_INVALID_STATE;
    error = lana_claim_encode_payload(claim, &payload);
    if (error == LANA_OK) lana_sha256(payload.data, payload.length, out_digest);
    free(payload.data); return error;
}

LanaError lana_claim_verify(const LanaRelationshipClaim *claim, const unsigned char public_key[32], bool *out_valid) {
    LanaBuffer payload = {0}; unsigned char digest[32], *signed_message, *message;
    unsigned long long message_length = 0u; LanaError error;
    if (public_key == NULL || out_valid == NULL) return LANA_ERR_INVALID_STATE;
    *out_valid = false;
    error = lana_claim_encode_payload(claim, &payload);
    if (error != LANA_OK) return error;
    lana_sha256(payload.data, payload.length, digest);
    if (memcmp(digest, claim->payload_digest, sizeof(digest)) != 0) { free(payload.data); return LANA_OK; }
    if (payload.length > SIZE_MAX - 64u) { free(payload.data); return LANA_ERR_LIMIT; }
    signed_message = malloc(payload.length + 64u); message = malloc(payload.length + 64u);
    if (signed_message == NULL || message == NULL) { free(payload.data); free(signed_message); free(message); return LANA_ERR_OOM; }
    memcpy(signed_message, claim->signature, 64u);
    memcpy(signed_message + 64u, payload.data, payload.length);
    *out_valid = crypto_sign_open(message, &message_length, signed_message, payload.length + 64u, public_key) == 0 &&
                 message_length == payload.length;
    free(payload.data); free(signed_message); free(message); return LANA_OK;
}

static LanaError now_seconds(uint64_t *out) {
    time_t value;
    if (out == NULL) return LANA_ERR_INVALID_STATE;
    value = time(NULL);
    if (value < 0) return LANA_ERR_IO;
    *out = (uint64_t)value; return LANA_OK;
}

static const LanaTrustedIssuer *find_issuer(const LanaRelationshipClaim *claim, const LanaClaimTrustConfig *trust) {
    size_t index;
    for (index = 0u; index < trust->issuer_count; ++index) {
        const LanaTrustedIssuer *issuer = &trust->issuers[index];
        if (issuer->issuer != NULL && issuer->key_id != NULL && issuer->scope != NULL &&
            strcmp(issuer->issuer, claim->issuer) == 0 && strcmp(issuer->key_id, claim->issuer_key_id) == 0 &&
            strcmp(issuer->scope, claim->scope) == 0) return issuer;
    }
    return NULL;
}

static bool trust_config_valid(const LanaClaimTrustConfig *trust) {
    return trust != NULL && trust->struct_size >= sizeof(*trust) && trust->schema_version == 1u &&
           trust->authority_policy_version != NULL && trust->authority_policy_version[0] != '\0' &&
           (trust->issuer_count == 0u || trust->issuers != NULL);
}

static LanaError validate_at(const LanaRelationshipClaim *claim, const LanaClaimTrustConfig *trust,
                             const char *subject, const char *scope, uint64_t now) {
    const LanaTrustedIssuer *issuer; bool valid; LanaError error;
    if (claim == NULL || !trust_config_valid(trust) || subject == NULL || scope == NULL ||
        claim->subject == NULL || claim->scope == NULL || claim->issuer == NULL || claim->issuer_key_id == NULL ||
        claim->authority_policy_version == NULL || claim->origin == NULL) return LANA_ERR_INVALID_STATE;
    if (claim->schema_version != LANA_RELATIONSHIP_CLAIM_SCHEMA_VERSION) return LANA_ERR_SCHEMA;
    if (strcmp(claim->subject, subject) != 0 || strcmp(claim->scope, scope) != 0) return LANA_ERR_CLAIM_MISMATCH;
    if ((error = validate_parameters(claim->relationship, claim->parameters, claim->parameters_length)) != LANA_OK) return error;
    if (claim->lifecycle == LANA_CLAIM_REVOKED) return LANA_ERR_CLAIM_REVOKED;
    if (claim->lifecycle != LANA_CLAIM_ACTIVE) return LANA_ERR_SCHEMA;
    if (now < claim->validity_start || now > claim->validity_end) return LANA_ERR_CLAIM_EXPIRED;
    if (strcmp(claim->authority_policy_version, trust->authority_policy_version) != 0) return LANA_ERR_UNAUTHORIZED_ISSUER;
    issuer = find_issuer(claim, trust);
    if (issuer == NULL) return LANA_ERR_UNAUTHORIZED_ISSUER;
    error = lana_claim_verify(claim, issuer->public_key, &valid);
    if (error != LANA_OK) return error;
    return valid ? LANA_OK : LANA_ERR_INTEGRITY;
}

LanaError lana_claim_validate(const LanaRelationshipClaim *claim, const LanaClaimTrustConfig *trust,
                              const char *subject, const char *scope, uint64_t *out_evaluated_at) {
    uint64_t now; LanaError error = now_seconds(&now);
    if (error != LANA_OK) return error;
    error = validate_at(claim, trust, subject, scope, now);
    if (error == LANA_OK && out_evaluated_at != NULL) *out_evaluated_at = now;
    return error;
}

static LanaError set_resolution(LanaRelationshipResolution *out, const LanaRelationshipRequest *request,
                                const LanaClaimTrustConfig *trust, uint64_t now,
                                LanaRelationshipResolutionMode mode, LanaRelationshipResolutionReason reason,
                                const LanaRelationshipClaim *claim, LanaRelationshipKind relationship,
                                const unsigned char *parameters, size_t parameters_length) {
    if (parameters_length > SIZE_MAX - 1u) return LANA_ERR_LIMIT;
    memset(out, 0, sizeof(*out)); out->schema_version = LANA_RELATIONSHIP_RESOLUTION_SCHEMA_VERSION;
    out->mode = mode; out->reason = reason; out->evaluated_at = now; out->has_relationship = mode != LANA_RELATIONSHIP_RESOLUTION_UNRESOLVED;
    out->relationship = relationship; out->claim_id = claim == NULL ? 0u : claim->claim_id;
    out->claim_version = claim == NULL ? 0u : claim->version;
    out->subject = malloc(strlen(request->subject) + 1u);
    out->scope = malloc(strlen(request->scope) + 1u);
    out->authority_policy_version = malloc(strlen(trust->authority_policy_version) + 1u);
    if (out->subject != NULL) (void)strcpy(out->subject, request->subject);
    if (out->scope != NULL) (void)strcpy(out->scope, request->scope);
    if (out->authority_policy_version != NULL) (void)strcpy(out->authority_policy_version, trust->authority_policy_version);
    if (parameters_length != 0u) { out->parameters = malloc(parameters_length); if (out->parameters != NULL) memcpy(out->parameters, parameters, parameters_length); }
    out->parameters_length = parameters_length;
    if (out->subject == NULL || out->scope == NULL || out->authority_policy_version == NULL ||
        (parameters_length != 0u && out->parameters == NULL)) { lana_relationship_resolution_free(out); return LANA_ERR_OOM; }
    return LANA_OK;
}

LanaError lana_relationship_resolve(const LanaRelationshipRequest *request, const LanaClaimTrustConfig *trust,
                                    LanaRelationshipResolution *out) {
    const LanaRelationshipClaim *selected = NULL; uint64_t now; size_t index; LanaError error;
    if (request == NULL || !trust_config_valid(trust) || out == NULL || request->subject == NULL || request->scope == NULL ||
        (request->claim_count != 0u && request->claims == NULL)) return LANA_ERR_INVALID_STATE;
    if ((error = now_seconds(&now)) != LANA_OK) return error;
    if (request->has_explicit_relationship) {
        if ((error = validate_parameters(request->explicit_relationship, request->explicit_parameters,
                                         request->explicit_parameters_length)) != LANA_OK) return error;
        return set_resolution(out, request, trust, now, LANA_RELATIONSHIP_RESOLUTION_EXPLICIT,
                              LANA_RELATIONSHIP_REASON_EXPLICIT, NULL, request->explicit_relationship,
                              request->explicit_parameters, request->explicit_parameters_length);
    }
    for (index = 0u; index < request->claim_count; ++index) {
        const LanaRelationshipClaim *candidate = &request->claims[index];
        error = validate_at(candidate, trust, request->subject, request->scope, now);
        if (error == LANA_ERR_CLAIM_MISMATCH || error == LANA_ERR_CLAIM_REVOKED || error == LANA_ERR_CLAIM_EXPIRED ||
            error == LANA_ERR_UNAUTHORIZED_ISSUER || error == LANA_ERR_INTEGRITY) continue;
        if (error != LANA_OK) return error;
        if (selected != NULL && selected->relationship != candidate->relationship) return LANA_ERR_CONFLICT;
        if (selected == NULL || candidate->version > selected->version ||
            (candidate->version == selected->version && candidate->claim_id < selected->claim_id)) selected = candidate;
    }
    if (selected == NULL) return set_resolution(out, request, trust, now, LANA_RELATIONSHIP_RESOLUTION_UNRESOLVED,
                                                 LANA_RELATIONSHIP_REASON_NO_TRUSTED_CLAIM, NULL,
                                                 LANA_RELATION_EXACT, NULL, 0u);
    return set_resolution(out, request, trust, now, LANA_RELATIONSHIP_RESOLUTION_TRUSTED_CLAIM,
                          LANA_RELATIONSHIP_REASON_TRUSTED_CLAIM, selected, selected->relationship,
                          selected->parameters, selected->parameters_length);
}

LanaError lana_relationship_resolution_encode(const LanaRelationshipResolution *resolution, LanaBuffer *out) {
    LanaError error;
    if (resolution == NULL || out == NULL || out->length != 0u || resolution->schema_version != 1u ||
        resolution->subject == NULL || resolution->scope == NULL || resolution->authority_policy_version == NULL ||
        (!resolution->has_relationship && resolution->parameters_length != 0u) ||
        (resolution->parameters_length != 0u && resolution->parameters == NULL)) return LANA_ERR_SCHEMA;
    if ((error = append(out, "LRR1", 4u)) != LANA_OK || (error = put_u32(out, resolution->schema_version)) != LANA_OK ||
        (error = put_u32(out, (uint32_t)resolution->mode)) != LANA_OK || (error = put_u32(out, (uint32_t)resolution->reason)) != LANA_OK ||
        (error = put_u32(out, resolution->has_relationship ? 1u : 0u)) != LANA_OK ||
        (error = put_u32(out, (uint32_t)resolution->relationship)) != LANA_OK || (error = put_u64(out, resolution->claim_id)) != LANA_OK ||
        (error = put_u32(out, resolution->claim_version)) != LANA_OK || (error = put_u64(out, resolution->evaluated_at)) != LANA_OK ||
        (error = put_text(out, resolution->subject)) != LANA_OK || (error = put_text(out, resolution->scope)) != LANA_OK ||
        (error = put_text(out, resolution->authority_policy_version)) != LANA_OK) return error;
    return put_blob(out, resolution->parameters, resolution->parameters_length);
}

void lana_relationship_resolution_free(LanaRelationshipResolution *resolution) {
    if (resolution == NULL) return;
    free(resolution->subject); free(resolution->scope); free(resolution->authority_policy_version); free(resolution->parameters);
    memset(resolution, 0, sizeof(*resolution));
}

LanaError lana_relationship_resolution_decode(const unsigned char *data, size_t length, LanaRelationshipResolution *out) {
    const unsigned char *cursor, *end; uint32_t value; size_t blob_length; LanaError error;
    if (data == NULL || out == NULL || length < 4u || memcmp(data, "LRR1", 4u) != 0) return LANA_ERR_SCHEMA;
    cursor = data; end = data + length;
    memset(out, 0, sizeof(*out)); cursor += 4u;
    if (!read_u32(&cursor, end, &out->schema_version) || out->schema_version != 1u || !read_u32(&cursor, end, &value)) goto corrupt;
    out->mode = (LanaRelationshipResolutionMode)value;
    if (out->mode > LANA_RELATIONSHIP_RESOLUTION_UNRESOLVED) goto corrupt;
    if (!read_u32(&cursor, end, &value)) goto corrupt;
    out->reason = (LanaRelationshipResolutionReason)value;
    if (out->reason > LANA_RELATIONSHIP_REASON_NO_TRUSTED_CLAIM) goto corrupt;
    if (!read_u32(&cursor, end, &value) || value > 1u) goto corrupt;
    out->has_relationship = value != 0u;
    if (!read_u32(&cursor, end, &value)) goto corrupt;
    out->relationship = (LanaRelationshipKind)value;
    if (!read_u64(&cursor, end, &out->claim_id) || !read_u32(&cursor, end, &out->claim_version) ||
        !read_u64(&cursor, end, &out->evaluated_at)) goto corrupt;
    if ((error = read_blob(&cursor, end, (unsigned char **)&out->subject, &blob_length)) != LANA_OK) goto failed;
    if ((error = read_blob(&cursor, end, (unsigned char **)&out->scope, &blob_length)) != LANA_OK) goto failed;
    if ((error = read_blob(&cursor, end, (unsigned char **)&out->authority_policy_version, &blob_length)) != LANA_OK) goto failed;
    if ((error = read_blob(&cursor, end, &out->parameters, &out->parameters_length)) != LANA_OK || cursor != end) goto failed;
    if ((out->has_relationship && validate_parameters(out->relationship, out->parameters,
                                                       out->parameters_length) != LANA_OK) ||
        (!out->has_relationship && out->parameters_length != 0u)) goto corrupt;
    return LANA_OK;
corrupt:
    error = LANA_ERR_CORRUPTION;
failed:
    lana_relationship_resolution_free(out); return error;
}
