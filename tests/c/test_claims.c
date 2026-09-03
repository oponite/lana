#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "lana/claims.h"
#include "lana/value.h"
#include "lana/error.h"
#include "lana/sha256.h"
#include "lana/vendor/tweetnacl.h"
#include "lana/vm.h"

static void test_claim_creation(void) {
    printf("Testing claim creation...\n");
    LanaVM vm;
    lana_vm_init(&vm, NULL);

    Value val = lana_value_number(1.0);
    Value claim = lana_value_null();

    LanaError err = lana_vm_claim(&vm, &val, "proposition",
                                  LANA_EXACTNESS_EXACT, 0.0, true, &claim);

    assert(err == LANA_OK);

    lana_vm_free(&vm);
    printf("Pass.\n");
}

static const unsigned char public_key[32] = {
        0xd7,0x5a,0x98,0x01,0x82,0xb1,0x0a,0xb7,0xd5,0x4b,0xfe,0xd3,0xc9,0x64,0x07,0x3a,
        0x0e,0xe1,0x72,0xf3,0xda,0xa6,0x23,0x25,0xaf,0x02,0x1a,0x68,0xf7,0x07,0x51,0x1a};
static const unsigned char secret_key[64] = {
    0x9d,0x61,0xb1,0x9d,0xef,0xfd,0x5a,0x60,0xba,0x84,0x4a,0xf4,0x92,0xec,0x2c,0xc4,
    0x44,0x49,0xc5,0x69,0x7b,0x32,0x69,0x19,0x70,0x3b,0xac,0x03,0x1c,0xae,0x7f,0x60,
    0xd7,0x5a,0x98,0x01,0x82,0xb1,0x0a,0xb7,0xd5,0x4b,0xfe,0xd3,0xc9,0x64,0x07,0x3a,
    0x0e,0xe1,0x72,0xf3,0xda,0xa6,0x23,0x25,0xaf,0x02,0x1a,0x68,0xf7,0x07,0x51,0x1a};

static LanaRelationshipClaim signed_claim(uint64_t id, LanaRelationshipKind relationship) {
    static const unsigned char parameters[] = "{}";
    LanaRelationshipClaim claim = {0};
    LanaBuffer payload = {0};
    unsigned char *signed_payload;
    unsigned long long signed_length;

    claim.claim_id = id; claim.version = 1u; claim.subject = "sensor-a"; claim.scope = "fusion";
    claim.issuer = "test-issuer"; claim.issuer_key_id = "test-key"; claim.authority_policy_version = "policy-1";
    claim.origin = "test"; claim.schema_version = LANA_RELATIONSHIP_CLAIM_SCHEMA_VERSION;
    claim.relationship = relationship; claim.parameters = parameters; claim.parameters_length = sizeof(parameters) - 1u;
    claim.validity_end = UINT64_MAX; claim.lifecycle = LANA_CLAIM_ACTIVE;
    assert(lana_claim_compute_payload_digest(&claim, claim.payload_digest) == LANA_OK);
    assert(lana_claim_encode_payload(&claim, &payload) == LANA_OK);
    signed_payload = malloc(payload.length + 64u); assert(signed_payload != NULL);
    assert(crypto_sign(signed_payload, &signed_length, payload.data, payload.length, secret_key) == 0);
    assert(signed_length == payload.length + 64u); memcpy(claim.signature, signed_payload, sizeof(claim.signature));
    free(signed_payload); free(payload.data); return claim;
}

static LanaClaimTrustConfig trust_config(void) {
    static const LanaTrustedIssuer issuers[] = {{"test-issuer", "test-key", "fusion", {
        0xd7,0x5a,0x98,0x01,0x82,0xb1,0x0a,0xb7,0xd5,0x4b,0xfe,0xd3,0xc9,0x64,0x07,0x3a,
        0x0e,0xe1,0x72,0xf3,0xda,0xa6,0x23,0x25,0xaf,0x02,0x1a,0x68,0xf7,0x07,0x51,0x1a}}};
    LanaClaimTrustConfig trust = {sizeof(trust), 1u, "policy-1", issuers, 1u};
    return trust;
}

static void test_signed_claim_verification(void) {
    LanaRelationshipClaim claim = signed_claim(7u, LANA_RELATION_EXACT);
    LanaClaimTrustConfig trust = trust_config();
    bool valid = false;

    assert(lana_claim_verify(&claim, public_key, &valid) == LANA_OK && valid);
    assert(lana_claim_validate(&claim, &trust, "sensor-a", "fusion", NULL) == LANA_OK);
    claim.signature[0] ^= 1u;
    assert(lana_claim_verify(&claim, public_key, &valid) == LANA_OK && !valid);
    assert(lana_claim_validate(&claim, &trust, "sensor-a", "fusion", NULL) == LANA_ERR_INTEGRITY);
}

static void test_validation_failures(void) {
    LanaRelationshipClaim claim = signed_claim(8u, LANA_RELATION_EXACT);
    LanaClaimTrustConfig trust = trust_config();
    LanaClaimTrustConfig wrong_policy = trust;
    static const unsigned char bad_parameters[] = "{ }";

    assert(lana_claim_validate(&claim, &trust, "other", "fusion", NULL) == LANA_ERR_CLAIM_MISMATCH);
    claim.schema_version = 99u;
    assert(lana_claim_validate(&claim, &trust, "sensor-a", "fusion", NULL) == LANA_ERR_SCHEMA);
    claim = signed_claim(8u, LANA_RELATION_EXACT); wrong_policy.authority_policy_version = "policy-2";
    assert(lana_claim_validate(&claim, &wrong_policy, "sensor-a", "fusion", NULL) == LANA_ERR_UNAUTHORIZED_ISSUER);
    claim.lifecycle = LANA_CLAIM_REVOKED;
    assert(lana_claim_validate(&claim, &trust, "sensor-a", "fusion", NULL) == LANA_ERR_CLAIM_REVOKED);
    claim.lifecycle = LANA_CLAIM_ACTIVE; claim.validity_end = 1u;
    assert(lana_claim_validate(&claim, &trust, "sensor-a", "fusion", NULL) == LANA_ERR_CLAIM_EXPIRED);
    claim = signed_claim(8u, LANA_RELATION_EXACT); claim.parameters = bad_parameters; claim.parameters_length = sizeof(bad_parameters) - 1u;
    assert(lana_claim_validate(&claim, &trust, "sensor-a", "fusion", NULL) == LANA_ERR_INVALID_PARAMETERS);
}

static void test_resolution_and_replay(void) {
    LanaRelationshipClaim claims[2] = {signed_claim(9u, LANA_RELATION_EXACT), signed_claim(10u, LANA_RELATION_SAME_DEPENDENCY)};
    LanaClaimTrustConfig trust = trust_config();
    LanaRelationshipRequest request = {"sensor-a", "fusion", false, LANA_RELATION_EXACT, NULL, 0u, claims, 1u};
    LanaRelationshipResolution resolution = {0}, replay = {0};
    LanaBuffer encoded = {0};
    static const unsigned char parameters[] = "{}";

    assert(lana_relationship_resolve(&request, &trust, &resolution) == LANA_OK);
    assert(resolution.mode == LANA_RELATIONSHIP_RESOLUTION_TRUSTED_CLAIM && resolution.claim_id == 9u);
    assert(lana_relationship_resolution_encode(&resolution, &encoded) == LANA_OK);
    assert(lana_relationship_resolution_decode(encoded.data, encoded.length, &replay) == LANA_OK);
    assert(replay.claim_id == 9u && replay.claim_version == 1u && replay.evaluated_at == resolution.evaluated_at);
    lana_relationship_resolution_free(&resolution); lana_relationship_resolution_free(&replay); free(encoded.data);

    request.has_explicit_relationship = true; request.explicit_relationship = LANA_RELATION_SAME_DEPENDENCY;
    request.explicit_parameters = parameters; request.explicit_parameters_length = sizeof(parameters) - 1u;
    assert(lana_relationship_resolve(&request, &trust, &resolution) == LANA_OK);
    assert(resolution.mode == LANA_RELATIONSHIP_RESOLUTION_EXPLICIT); lana_relationship_resolution_free(&resolution);

    request.has_explicit_relationship = false; request.claim_count = 2u;
    assert(lana_relationship_resolve(&request, &trust, &resolution) == LANA_ERR_CONFLICT);
    request.claim_count = 0u;
    assert(lana_relationship_resolve(&request, &trust, &resolution) == LANA_OK);
    assert(resolution.mode == LANA_RELATIONSHIP_RESOLUTION_UNRESOLVED); lana_relationship_resolution_free(&resolution);
}

int main(void) {
    test_claim_creation();
    test_signed_claim_verification();
    test_validation_failures();
    test_resolution_and_replay();
    return 0;
}
