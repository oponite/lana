#ifndef LANA_CLAIMS_H
#define LANA_CLAIMS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "codec.h"
#include "error.h"
#include "value.h"

#define LANA_RELATIONSHIP_CLAIM_SCHEMA_VERSION 1u
#define LANA_RELATIONSHIP_RESOLUTION_SCHEMA_VERSION 1u

typedef enum {
    LANA_CLAIM_ACTIVE = 0,
    LANA_CLAIM_REVOKED
} LanaClaimLifecycle;

typedef struct {
    uint64_t claim_id;
    uint32_t version;
    const char *subject;
    const char *scope;
    const char *issuer;
    const char *issuer_key_id;
    const char *authority_policy_version;
    const char *origin;
    uint32_t schema_version;
    LanaRelationshipKind relationship;
    const unsigned char *parameters;
    size_t parameters_length;
    uint64_t validity_start;
    uint64_t validity_end;
    LanaClaimLifecycle lifecycle;
    unsigned char payload_digest[32];
    unsigned char signature[64];
} LanaRelationshipClaim;

typedef struct {
    const char *issuer;
    const char *key_id;
    const char *scope;
    unsigned char public_key[32];
} LanaTrustedIssuer;

typedef struct {
    size_t struct_size;
    uint32_t schema_version;
    const char *authority_policy_version;
    const LanaTrustedIssuer *issuers;
    size_t issuer_count;
} LanaClaimTrustConfig;

typedef enum {
    LANA_RELATIONSHIP_RESOLUTION_EXPLICIT = 0,
    LANA_RELATIONSHIP_RESOLUTION_TRUSTED_CLAIM,
    LANA_RELATIONSHIP_RESOLUTION_UNRESOLVED
} LanaRelationshipResolutionMode;

typedef enum {
    LANA_RELATIONSHIP_REASON_EXPLICIT = 0,
    LANA_RELATIONSHIP_REASON_TRUSTED_CLAIM,
    LANA_RELATIONSHIP_REASON_NO_TRUSTED_CLAIM
} LanaRelationshipResolutionReason;

typedef struct {
    const char *subject;
    const char *scope;
    bool has_explicit_relationship;
    LanaRelationshipKind explicit_relationship;
    const unsigned char *explicit_parameters;
    size_t explicit_parameters_length;
    const LanaRelationshipClaim *claims;
    size_t claim_count;
} LanaRelationshipRequest;

typedef struct {
    uint32_t schema_version;
    LanaRelationshipResolutionMode mode;
    LanaRelationshipResolutionReason reason;
    bool has_relationship;
    LanaRelationshipKind relationship;
    uint64_t claim_id;
    uint32_t claim_version;
    uint64_t evaluated_at;
    char *subject;
    char *scope;
    char *authority_policy_version;
    unsigned char *parameters;
    size_t parameters_length;
} LanaRelationshipResolution;

/* Encodes exactly the bytes that an Ed25519 signer must sign. */
LanaError lana_claim_encode_payload(const LanaRelationshipClaim *claim, LanaBuffer *out_payload);
LanaError lana_claim_compute_payload_digest(const LanaRelationshipClaim *claim,
                                            unsigned char out_digest[32]);
/* Verifies payload digest and Ed25519 signature only; no trust policy is applied. */
LanaError lana_claim_verify(const LanaRelationshipClaim *claim,
                            const unsigned char public_key[32], bool *out_valid);
/* Applies the caller-owned trust configuration and the current system time. */
LanaError lana_claim_validate(const LanaRelationshipClaim *claim,
                              const LanaClaimTrustConfig *trust,
                              const char *expected_subject, const char *expected_scope,
                              uint64_t *out_evaluated_at);

LanaError lana_relationship_resolve(const LanaRelationshipRequest *request,
                                    const LanaClaimTrustConfig *trust,
                                    LanaRelationshipResolution *out_resolution);
LanaError lana_relationship_resolution_encode(const LanaRelationshipResolution *resolution,
                                              LanaBuffer *out);
LanaError lana_relationship_resolution_decode(const unsigned char *data, size_t length,
                                              LanaRelationshipResolution *out_resolution);
void lana_relationship_resolution_free(LanaRelationshipResolution *resolution);

#endif
