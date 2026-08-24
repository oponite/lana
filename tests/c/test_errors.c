#include "lana/error.h"
#include "lana/vm.h"

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

    lana_error_set(&error, LANA_ERR_UNRESOLVED_VALUE, 12u, OP_RESOLVE, 7u,
                   "resolution remained ambiguous");
    lana_error_set_source_span(&error, "tests/native/ambiguous.lana", 7u, 5u, 7u, 18u);
    lana_error_set_operation(&error, "resolve");
    lana_error_set_resolution(&error, LANA_RESOLUTION_REASON_MULTIPLE_ALTERNATIVES, 3u);
    lana_error_set_exact_support(&error, LANA_EXACT_SUPPORT_UNAVAILABLE,
                                 "finite exact support is unavailable");
    lana_error_set_cancellation(&error, 41u, "parent task cancelled");
    lana_error_set_resource_limit(&error, LANA_RESOURCE_PATHS, 8u, 9u, "paths");

    CHECK(error.kind == LANA_ERROR_KIND_RESOLUTION);
    CHECK(strcmp(error.source.path, "tests/native/ambiguous.lana") == 0);
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

int main(void) {
    CHECK(test_structured_error_helpers() == 0);
    CHECK(test_vm_failure_context_has_no_partial_result() == 0);
    (void)printf("Structured error tests passed\n");
    return 0;
}
