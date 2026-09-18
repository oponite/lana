#include "state.h"
#include "vm.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) { (void)fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); return 1; } } while (0)

static int test_state_attenuate_math(void) {
    LanaState source, out;
    CHECK(lana_state_make_complex(0.5, 0.6, 0.0, &source) == LANA_OK);
    CHECK(lana_state_attenuate(&source, 0.5, &out) == LANA_OK);
    CHECK(out.p == 0.5);
    CHECK(fabs(out.d_re - 0.3) < LANA_STATE_EPSILON);
    CHECK(fabs(out.d_im) < LANA_STATE_EPSILON);
    CHECK(lana_state_attenuate(&source, 1.5, &out) == LANA_ERR_INVALID_PARAMETERS);
    CHECK(lana_state_attenuate(&source, NAN, &out) == LANA_ERR_INVALID_PARAMETERS);
    CHECK(lana_state_attenuate(&source, INFINITY, &out) == LANA_ERR_INVALID_PARAMETERS);
    return 0;
}

static int test_state_trace_distance_math(void) {
    LanaState left, right;
    double distance;
    CHECK(lana_state_make_complex(0.5, 0.6, 0.0, &left) == LANA_OK);
    CHECK(lana_state_make_complex(0.5, 0.4, 0.0, &right) == LANA_OK);
    CHECK(lana_state_trace_distance(&left, &right, &distance) == LANA_OK);
    CHECK(fabs(distance - 0.1) < LANA_STATE_EPSILON);
    return 0;
}

static int test_state_append_relationship_math(void) {
    LanaState left, right;
    double p, m_re, m_im, sigma;
    CHECK(lana_state_make_complex(0.4, 0.2, 0.0, &left) == LANA_OK);
    CHECK(lana_state_make_complex(0.8, -0.1, 0.0, &right) == LANA_OK);

    /* independent: p_C = 0.4 + 0.8 - 0.32 = 0.88. */
    CHECK(lana_state_append_relationship_parameters(&left, &right, LANA_APPEND_INDEPENDENT,
                                                    0.0, &p, &m_re, &m_im, &sigma) == LANA_OK);
    CHECK(fabs(p - 0.88) < LANA_STATE_EPSILON);

    /* redundant r=0.8: q = 0.2*0.32 + 0.8*0.4 = 0.384, p_C = 0.816. */
    CHECK(lana_state_append_relationship_parameters(&left, &right, LANA_APPEND_REDUNDANT,
                                                    0.8, &p, &m_re, &m_im, &sigma) == LANA_OK);
    CHECK(fabs(p - 0.816) < LANA_STATE_EPSILON);

    /* complementary k=0.3: q = 0.7*0.32 + 0.3*0.2 = 0.284, p_C = 0.916. */
    CHECK(lana_state_append_relationship_parameters(&left, &right, LANA_APPEND_COMPLEMENTARY,
                                                    0.3, &p, &m_re, &m_im, &sigma) == LANA_OK);
    CHECK(fabs(p - 0.916) < LANA_STATE_EPSILON);

    /* full_redundancy requires p_a == p_b. */
    CHECK(lana_state_append_relationship_parameters(&left, &right, LANA_APPEND_FULL_REDUNDANCY,
                                                    0.0, &p, &m_re, &m_im, &sigma) == LANA_ERR_INVALID_PARAMETERS);

    /* redundant strength must be in [0,1). */
    CHECK(lana_state_append_relationship_parameters(&left, &right, LANA_APPEND_REDUNDANT,
                                                    1.0, &p, &m_re, &m_im, &sigma) == LANA_ERR_INVALID_PARAMETERS);
    return 0;
}

static LanaInstruction instruction(OpCode opcode, uint32_t a, uint32_t b,
                                   uint32_t c, uint32_t imm) {
    LanaInstruction value = {(uint8_t)opcode, a, b, c, imm, 1u};
    return value;
}

static int add_number(LanaChunk *chunk, double number, uint32_t *index) {
    return lana_chunk_add_constant(chunk, lana_value_number(number), index) == LANA_OK ? 0 : 1;
}

static int test_vm_attenuate_opcode(void) {
    LanaChunk chunk;
    LanaVM vm;
    uint32_t p_a, d_a, factor, zero;
    lana_chunk_init(&chunk);
    CHECK(add_number(&chunk, 0.5, &p_a) == 0);
    CHECK(add_number(&chunk, 0.6, &d_a) == 0);
    CHECK(add_number(&chunk, 0.5, &factor) == 0);
    CHECK(add_number(&chunk, 0.0, &zero) == 0);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_STATE_NEW, 0, p_a, d_a, zero)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_LOAD_CONST, 3, 0, 0, factor)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_ATTENUATE, 2, 0, 3, 0)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    CHECK(lana_vm_run(&vm) == LANA_OK);
    CHECK(vm.frames[0].registers[2].type == VAL_STATE);
    CHECK(vm.frames[0].registers[2].as.state.state.p == 0.5);
    CHECK(fabs(vm.frames[0].registers[2].as.state.state.d_re - 0.3) < LANA_STATE_EPSILON);
    CHECK(vm.frames[0].registers[2].derivation != NULL);
    CHECK(strcmp(vm.frames[0].registers[2].derivation->operation, "attenuate") == 0);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);
    return 0;
}

static int test_vm_trace_distance_opcode(void) {
    LanaChunk chunk;
    LanaVM vm;
    uint32_t p_a, d_a, p_b, d_b, zero;
    lana_chunk_init(&chunk);
    CHECK(add_number(&chunk, 0.5, &p_a) == 0);
    CHECK(add_number(&chunk, 0.6, &d_a) == 0);
    CHECK(add_number(&chunk, 0.5, &p_b) == 0);
    CHECK(add_number(&chunk, 0.4, &d_b) == 0);
    CHECK(add_number(&chunk, 0.0, &zero) == 0);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_STATE_NEW, 0, p_a, d_a, zero)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_STATE_NEW, 1, p_b, d_b, zero)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_TRACE_DISTANCE, 2, 0, 1, 0)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    CHECK(lana_vm_run(&vm) == LANA_OK);
    CHECK(vm.frames[0].registers[2].type == VAL_NUMBER);
    CHECK(fabs(vm.frames[0].registers[2].as.number - 0.1) < LANA_STATE_EPSILON);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);
    return 0;
}

static int test_vm_append_relationship_opcodes(void) {
    LanaChunk chunk;
    LanaVM vm;
    uint32_t p_a, d_a, p_b, d_b, strength, zero;
    double expected;
    lana_chunk_init(&chunk);
    CHECK(add_number(&chunk, 0.4, &p_a) == 0);
    CHECK(add_number(&chunk, 0.2, &d_a) == 0);
    CHECK(add_number(&chunk, 0.8, &p_b) == 0);
    CHECK(add_number(&chunk, -0.1, &d_b) == 0);
    CHECK(add_number(&chunk, 0.8, &strength) == 0);
    CHECK(add_number(&chunk, 0.0, &zero) == 0);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_STATE_NEW, 0, p_a, d_a, zero)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_STATE_NEW, 1, p_b, d_b, zero)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_LOAD_CONST, 3, 0, 0, strength)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_APPEND_REDUNDANT, 0, 1, 2, 3)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    CHECK(lana_vm_run(&vm) == LANA_OK);
    CHECK(vm.frames[0].registers[2].type == VAL_STATE_DIST);
    CHECK(lana_vm_state_dist_expected_probability(vm.frames[0].registers[2].as.state_dist,
                                                  &expected) == LANA_OK);
    CHECK(fabs(expected - 0.816) < LANA_STATE_EPSILON);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);

    /* complementary k=0.3: p_C = 0.916. */
    lana_chunk_init(&chunk);
    CHECK(add_number(&chunk, 0.4, &p_a) == 0);
    CHECK(add_number(&chunk, 0.2, &d_a) == 0);
    CHECK(add_number(&chunk, 0.8, &p_b) == 0);
    CHECK(add_number(&chunk, -0.1, &d_b) == 0);
    CHECK(add_number(&chunk, 0.3, &strength) == 0);
    CHECK(add_number(&chunk, 0.0, &zero) == 0);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_STATE_NEW, 0, p_a, d_a, zero)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_STATE_NEW, 1, p_b, d_b, zero)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_LOAD_CONST, 3, 0, 0, strength)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_APPEND_COMPLEMENTARY, 0, 1, 2, 3)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    CHECK(lana_vm_run(&vm) == LANA_OK);
    CHECK(vm.frames[0].registers[2].type == VAL_STATE_DIST);
    CHECK(lana_vm_state_dist_expected_probability(vm.frames[0].registers[2].as.state_dist,
                                                  &expected) == LANA_OK);
    CHECK(fabs(expected - 0.916) < LANA_STATE_EPSILON);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);

    /* full_redundancy with p_a == p_b: p_C = 0.5. */
    lana_chunk_init(&chunk);
    CHECK(add_number(&chunk, 0.5, &p_a) == 0);
    CHECK(add_number(&chunk, 0.0, &d_a) == 0);
    CHECK(add_number(&chunk, 0.5, &p_b) == 0);
    CHECK(add_number(&chunk, 0.0, &d_b) == 0);
    CHECK(add_number(&chunk, 0.0, &zero) == 0);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_STATE_NEW, 0, p_a, d_a, zero)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_STATE_NEW, 1, p_b, d_b, zero)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_APPEND_FULL_REDUNDANCY, 0, 1, 2, 0)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    CHECK(lana_vm_run(&vm) == LANA_OK);
    CHECK(vm.frames[0].registers[2].type == VAL_STATE_DIST);
    CHECK(lana_vm_state_dist_expected_probability(vm.frames[0].registers[2].as.state_dist,
                                                  &expected) == LANA_OK);
    CHECK(fabs(expected - 0.5) < LANA_STATE_EPSILON);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);
    return 0;
}

int main(void) {
    CHECK(test_state_attenuate_math() == 0);
    CHECK(test_state_trace_distance_math() == 0);
    CHECK(test_state_append_relationship_math() == 0);
    CHECK(test_vm_attenuate_opcode() == 0);
    CHECK(test_vm_trace_distance_opcode() == 0);
    CHECK(test_vm_append_relationship_opcodes() == 0);
    (void)printf("OPERATIONS2 tests passed\n");
    return 0;
}
