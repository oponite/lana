#include "error.h"
#include "sha256.h"
#include "vm.h"

#include <stdio.h>
#include <string.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        (void)fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static int test_structured_error_helpers(void) {
    LanaErrorInfo error = {0};
    LanaErrorInfo cause = {0};
    size_t index;

    CHECK(lana_error_kind_from_code(LANA_ERR_PARSE) == LANA_ERROR_KIND_PARSE);
    CHECK(lana_error_kind_from_code(LANA_ERR_TYPE) == LANA_ERROR_KIND_TYPE);
    CHECK(lana_error_kind_from_code(LANA_ERR_INVALID_CONDITIONING) == LANA_ERROR_KIND_RESOLUTION);
    CHECK(lana_error_kind_from_code(LANA_ERR_CANCELLED) == LANA_ERROR_KIND_CANCELLATION);
    CHECK(lana_error_kind_from_code(LANA_ERR_OOM) == LANA_ERROR_KIND_RESOURCE_LIMIT);
    CHECK(strcmp(lana_error_name(LANA_ERR_NOT_FOUND), "LANA_ERR_NOT_FOUND") == 0);
    CHECK(strcmp(lana_error_name(LANA_ERR_COMPACTED_HISTORY),
                 "LANA_ERR_COMPACTED_HISTORY") == 0);
    CHECK(strcmp(lana_error_name(LANA_ERR_SCHEMA), "LANA_ERR_SCHEMA") == 0);
    CHECK(strcmp(lana_error_name(LANA_ERR_UNSUPPORTED_VALUE),
                 "LANA_ERR_UNSUPPORTED_VALUE") == 0);
    CHECK(strcmp(lana_error_name(LANA_ERR_CORRUPTION), "LANA_ERR_CORRUPTION") == 0);

    lana_error_set(&error, LANA_ERR_UNRESOLVED_VALUE, 12u, OP_RESOLVE, 7u,
                   "resolution remained ambiguous");
    lana_error_set_source_span(&error, "tests/regression/ambiguous.lana", 7u, 5u, 7u, 18u);
    lana_error_set_operation(&error, "resolve");
    lana_error_set_resolution(&error, LANA_RESOLUTION_REASON_MULTIPLE_ALTERNATIVES, 3u);
    lana_error_set_exact_support(&error, LANA_EXACT_SUPPORT_UNAVAILABLE,
                                 "finite exact support is unavailable");
    lana_error_set_cancellation(&error, 41u, "parent task cancelled");
    lana_error_set_resource_limit(&error, LANA_RESOURCE_PATHS, 8u, 9u, "paths");

    CHECK(error.kind == LANA_ERROR_KIND_RESOLUTION);
    CHECK(strcmp(error.source.path, "tests/regression/ambiguous.lana") == 0);
    CHECK(error.source.start_line == 7u && error.source.start_column == 5u);
    CHECK(error.source.end_line == 7u && error.source.end_column == 18u);
    CHECK(strcmp(error.operation, "resolve") == 0);
    CHECK(error.has_remaining_alternatives && error.remaining_alternatives == 3u);
    CHECK(error.exact_support == LANA_EXACT_SUPPORT_UNAVAILABLE);
    CHECK(error.cancellation.present && error.cancellation.task_lineage == 41u);
    CHECK(error.resource_limit.present && error.resource_limit.resource == LANA_RESOURCE_PATHS);
    CHECK(error.resource_limit.limit == 8u && error.resource_limit.observed == 9u);

    lana_error_set(&cause, LANA_ERR_INVALID_CONDITIONING, 3u, OP_JOINT_CONDITION,
                   6u, "conditioning event has zero mass");
    lana_error_set_operation(&cause, "condition");
    for (index = 0u; index < LANA_ERROR_MAX_CAUSES; ++index)
        CHECK(lana_error_add_cause(&error, &cause));
    CHECK(!lana_error_add_cause(&error, &cause));
    CHECK(error.cause_count == LANA_ERROR_MAX_CAUSES);
    CHECK(error.cause_chain_truncated);
    CHECK(error.causes[0].code == LANA_ERR_INVALID_CONDITIONING);
    CHECK(strcmp(error.causes[0].operation, "condition") == 0);
    return 0;
}

static int test_vm_failure_context_has_no_partial_result(void) {
    LanaChunk chunk;
    LanaInstruction halt = {(uint8_t)OP_HALT, 0u, 0u, 0u, 0u, 9u};
    LanaVM vm;

    lana_chunk_init(&chunk);
    CHECK(lana_chunk_emit(&chunk, halt) == LANA_OK);

    lana_vm_init(&vm, &chunk);
    vm.result = lana_value_number(42.0);
    vm.instruction_limit = 0u;
    CHECK(lana_vm_run(&vm) == LANA_ERR_LIMIT);
    CHECK(vm.result.type == VAL_NULL);
    CHECK(vm.error.kind == LANA_ERROR_KIND_RESOURCE_LIMIT);
    CHECK(strcmp(vm.error.operation, "execute") == 0);
    CHECK(vm.error.resource_limit.present);
    CHECK(vm.error.resource_limit.resource == LANA_RESOURCE_INSTRUCTIONS);
    CHECK(vm.error.resource_limit.limit == 0u);
    lana_vm_free(&vm);

    lana_vm_init(&vm, &chunk);
    vm.result = lana_value_number(42.0);
    vm.lineage = 73u;
    atomic_store(&vm.cancelled, true);
    CHECK(lana_vm_run(&vm) == LANA_ERR_CANCELLED);
    CHECK(vm.result.type == VAL_NULL);
    CHECK(vm.error.kind == LANA_ERROR_KIND_CANCELLATION);
    CHECK(vm.error.cancellation.present && vm.error.cancellation.task_lineage == 73u);
    CHECK(vm.error.resolution_reason == LANA_RESOLUTION_REASON_CANCELLED);
    lana_vm_free(&vm);

    lana_chunk_free(&chunk);
    return 0;
}

static int test_sha256_known_answers(void) {
    static const unsigned char empty[LANA_SHA256_DIGEST_SIZE] = {0xe3u, 0xb0u, 0xc4u, 0x42u, 0x98u, 0xfcu, 0x1cu, 0x14u, 0x9au, 0xfbu, 0xf4u, 0xc8u, 0x99u, 0x6fu, 0xb9u, 0x24u, 0x27u, 0xaeu, 0x41u, 0xe4u, 0x64u, 0x9bu, 0x93u, 0x4cu, 0xa4u, 0x95u, 0x99u, 0x1bu, 0x78u, 0x52u, 0xb8u, 0x55u};
    static const unsigned char abc[LANA_SHA256_DIGEST_SIZE] = {0xbau, 0x78u, 0x16u, 0xbfu, 0x8fu, 0x01u, 0xcfu, 0xeau, 0x41u, 0x41u, 0x40u, 0xdeu, 0x5du, 0xaeu, 0x22u, 0x23u, 0xb0u, 0x03u, 0x61u, 0xa3u, 0x96u, 0x17u, 0x7au, 0x9cu, 0xb4u, 0x10u, 0xffu, 0x61u, 0xf2u, 0x00u, 0x15u, 0xadu};
    LanaSha256 context;
    unsigned char digest[LANA_SHA256_DIGEST_SIZE];
    lana_sha256(NULL, 0u, digest);
    CHECK(memcmp(digest, empty, sizeof(digest)) == 0);
    lana_sha256_init(&context);
    lana_sha256_update(&context, "a", 1u);
    lana_sha256_update(&context, "bc", 2u);
    lana_sha256_final(&context, digest);
    CHECK(memcmp(digest, abc, sizeof(digest)) == 0);
    return 0;
}

int main(void) {
    CHECK(test_structured_error_helpers() == 0);
    CHECK(test_vm_failure_context_has_no_partial_result() == 0);
    CHECK(test_sha256_known_answers() == 0);
    (void)printf("Structured error tests passed\n");
    return 0;
}
