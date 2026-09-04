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

static int test_vm_support_opcode(void) {
    LanaChunk chunk;
    LanaVM vm;
    LanaStateValue state_a, state_b;
    LanaStateDist *dirac, *distribution;
    Value state_dist_value, left_value, right_value;
    LanaErrorInfo error = {0};

    /* DIRAC input returns a singleton array. */
    lana_chunk_init(&chunk);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_SUPPORT, 1, 0, 0, 4)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    CHECK(lana_state_make_complex(0.5, 0.0, 0.0, &state_a.state) == LANA_OK);
    CHECK(lana_vm_state_dist_dirac(&vm, &state_a, &dirac) == LANA_OK);
    state_dist_value = lana_value_state_dist(dirac);
    vm.frames[0].registers[0] = state_dist_value;
    CHECK(lana_vm_run(&vm) == LANA_OK);
    CHECK(vm.frames[0].registers[1].type == VAL_ARRAY);
    CHECK(vm.frames[0].registers[1].as.array->count == 1);
    CHECK(vm.frames[0].registers[1].as.array->items[0].type == VAL_STATE);
    CHECK(fabs(vm.frames[0].registers[1].as.array->items[0].as.state.state.p - 0.5) < LANA_STATE_EPSILON);
    CHECK(vm.frames[0].registers[1].derivation != NULL);
    CHECK(strcmp(vm.frames[0].registers[1].derivation->operation, "support") == 0);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);

    /* APPEND of identical states (sigma == 0) collapses to a single state. */
    lana_chunk_init(&chunk);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_SUPPORT, 1, 0, 0, 4)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    CHECK(lana_state_make_complex(0.5, 0.0, 0.0, &state_a.state) == LANA_OK);
    CHECK(lana_state_make_complex(0.5, 0.0, 0.0, &state_b.state) == LANA_OK);
    left_value = lana_value_state(state_a.state);
    right_value = lana_value_state(state_b.state);
    CHECK(lana_vm_state_dist_append(&vm, &left_value, &right_value, &distribution) == LANA_OK);
    state_dist_value = lana_value_state_dist(distribution);
    vm.frames[0].registers[0] = state_dist_value;
    CHECK(lana_vm_run(&vm) == LANA_OK);
    CHECK(vm.frames[0].registers[1].type == VAL_ARRAY);
    CHECK(vm.frames[0].registers[1].as.array->count == 1);
    CHECK(fabs(vm.frames[0].registers[1].as.array->items[0].as.state.state.p - 0.75) < LANA_STATE_EPSILON);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);

    /* APPEND with sigma > 0 returns LANA_ERR_UNSUPPORTED_OPERATION. */
    lana_chunk_init(&chunk);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_SUPPORT, 1, 0, 0, 4)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    CHECK(lana_state_make_complex(0.4, 0.2, 0.0, &state_a.state) == LANA_OK);
    CHECK(lana_state_make_complex(0.8, -0.1, 0.0, &state_b.state) == LANA_OK);
    left_value = lana_value_state(state_a.state);
    right_value = lana_value_state(state_b.state);
    CHECK(lana_vm_state_dist_append(&vm, &left_value, &right_value, &distribution) == LANA_OK);
    state_dist_value = lana_value_state_dist(distribution);
    vm.frames[0].registers[0] = state_dist_value;
    CHECK(lana_vm_run(&vm) == LANA_ERR_UNSUPPORTED_OPERATION);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);

    /* Non-STATE_DIST input returns LANA_ERR_TYPE. */
    lana_chunk_init(&chunk);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_SUPPORT, 1, 0, 0, 4)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    vm.frames[0].registers[0] = lana_value_number(1.0);
    CHECK(lana_vm_run(&vm) == LANA_ERR_TYPE);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);

    /* Verifier rejects zero limit. */
    lana_chunk_init(&chunk);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_SUPPORT, 1, 0, 0, 0)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    CHECK(lana_chunk_verify(&chunk, &error) == LANA_ERR_FORMAT);
    lana_chunk_free(&chunk);

    /* Verifier rejects non-zero c. */
    lana_chunk_init(&chunk);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_SUPPORT, 1, 0, 1, 4)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    CHECK(lana_chunk_verify(&chunk, &error) == LANA_ERR_FORMAT);
    lana_chunk_free(&chunk);
    return 0;
}

int main(void) {
    CHECK(test_vm_support_opcode() == 0);
    (void)printf("SUPPORT tests passed\n");
    return 0;
}
