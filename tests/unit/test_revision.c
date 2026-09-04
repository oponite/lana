#include "shared.h"
#include "state.h"
#include "vm.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) { (void)fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); return 1; } } while (0)

static LanaInstruction instruction(OpCode opcode, uint32_t a, uint32_t b,
                                 uint32_t c, uint32_t imm) {
    LanaInstruction value = {(uint8_t)opcode, a, b, c, imm, 1u};
    return value;
}

static int test_vm_revision_opcode(void) {
    LanaChunk chunk;
    LanaVM vm;
    Value source, derived, reactive_value, capability_value;
    LanaSharedInformation *shared;
    LanaCapabilityToken *admin;
    LanaErrorInfo error = {0};

    /* Value with a derivation returns the derivation revision. */
    lana_chunk_init(&chunk);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_REVISION, 1, 0, 0, 0)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    source = lana_value_number(3.0);
    CHECK(lana_vm_provenance_root(&vm, &source, "test", 1u, false, &derived) == LANA_OK);
    vm.frames[0].registers[0] = derived;
    CHECK(lana_vm_run(&vm) == LANA_OK);
    CHECK(vm.frames[0].registers[1].type == VAL_NUMBER);
    CHECK(vm.frames[0].registers[1].as.number == (double)vm.revision);
    CHECK(vm.frames[0].registers[1].derivation != NULL);
    CHECK(strcmp(vm.frames[0].registers[1].derivation->operation, "revision") == 0);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);

    /* Value with a reactive returns the reactive revision. */
    lana_chunk_init(&chunk);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_REVISION, 1, 0, 0, 0)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    source = lana_value_number(3.0);
    CHECK(lana_vm_reactive_root(&vm, &source, LANA_EXACTNESS_EXACT, &reactive_value) == LANA_OK);
    vm.frames[0].registers[0] = reactive_value;
    CHECK(lana_vm_run(&vm) == LANA_OK);
    CHECK(vm.frames[0].registers[1].type == VAL_NUMBER);
    CHECK(vm.frames[0].registers[1].as.number == (double)vm.revision);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);

    /* Shared capability returns the shared information revision. */
    lana_chunk_init(&chunk);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_REVISION, 1, 0, 0, 0)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    source = lana_value_number(3.0);
    CHECK(lana_shared_information_create(&vm, &source, &shared, &admin) == LANA_OK);
    capability_value = lana_value_shared_capability(admin);
    vm.frames[0].registers[0] = capability_value;
    CHECK(lana_vm_run(&vm) == LANA_OK);
    CHECK(vm.frames[0].registers[1].type == VAL_NUMBER);
    CHECK(vm.frames[0].registers[1].as.number == (double)lana_shared_information_revision(shared));
    lana_shared_information_release(shared);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);

    /* Value without derivation or reactive returns LANA_ERR_TYPE. */
    lana_chunk_init(&chunk);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_REVISION, 1, 0, 0, 0)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    vm.frames[0].registers[0] = lana_value_number(3.0);
    CHECK(lana_vm_run(&vm) == LANA_ERR_TYPE);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);

    /* Verifier rejects non-zero c or imm. */
    lana_chunk_init(&chunk);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_REVISION, 1, 0, 1, 0)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    CHECK(lana_chunk_verify(&chunk, &error) == LANA_ERR_FORMAT);
    lana_chunk_free(&chunk);
    return 0;
}

int main(void) {
    CHECK(test_vm_revision_opcode() == 0);
    (void)printf("REVISION tests passed\n");
    return 0;
}
