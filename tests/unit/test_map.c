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

static int test_vm_map_opcode(void) {
    LanaChunk chunk;
    LanaVM vm;
    LanaStateValue state;
    LanaStateDist *dirac;
    Value state_dist_value;
    LanaErrorInfo error = {0};

    /* MAP on a DIRAC state_dist produces a lazy transform node. */
    lana_chunk_init(&chunk);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_MAP, 1, 0, LANA_TRANSFORM_INVERT, 0)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    CHECK(lana_state_make_complex(0.5, 0.0, 0.0, &state.state) == LANA_OK);
    CHECK(lana_vm_state_dist_dirac(&vm, &state, &dirac) == LANA_OK);
    state_dist_value = lana_value_state_dist(dirac);
    vm.frames[0].registers[0] = state_dist_value;
    CHECK(lana_vm_run(&vm) == LANA_OK);
    CHECK(vm.frames[0].registers[1].type == VAL_STATE_DIST);
    CHECK(vm.frames[0].registers[1].as.state_dist->kind == LANA_DIST_TRANSFORM);
    CHECK(vm.frames[0].registers[1].as.state_dist->as.transform.transform_id == LANA_TRANSFORM_INVERT);
    CHECK(vm.frames[0].registers[1].as.state_dist->as.transform.child == dirac);
    /* Derivation record is written. */
    CHECK(vm.frames[0].registers[1].derivation != NULL);
    CHECK(vm.frames[0].registers[1].derivation->kind == LANA_DERIVATION_OPERATION);
    CHECK(strcmp(vm.frames[0].registers[1].derivation->operation, "map") == 0);
    CHECK(vm.frames[0].registers[1].derivation->exactness == LANA_EXACTNESS_EXACT);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);

    /* Non-STATE_DIST input returns LANA_ERR_TYPE. */
    lana_chunk_init(&chunk);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_MAP, 1, 0, LANA_TRANSFORM_INVERT, 0)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    vm.frames[0].registers[0] = lana_value_number(1.0);
    CHECK(lana_vm_run(&vm) == LANA_ERR_TYPE);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);

    /* Verifier rejects unknown transform ID. */
    lana_chunk_init(&chunk);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_MAP, 1, 0, 99, 0)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    CHECK(lana_chunk_verify(&chunk, &error) == LANA_ERR_TRANSFORM);
    lana_chunk_free(&chunk);

    /* Verifier rejects non-zero imm. */
    lana_chunk_init(&chunk);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_MAP, 1, 0, LANA_TRANSFORM_INVERT, 1)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    CHECK(lana_chunk_verify(&chunk, &error) == LANA_ERR_FORMAT);
    lana_chunk_free(&chunk);
    return 0;
}

int main(void) {
    CHECK(test_vm_map_opcode() == 0);
    (void)printf("MAP tests passed\n");
    return 0;
}
