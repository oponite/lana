#include "ss/assembler.h"
#include "ss/asm_probe.h"
#include "ss/vm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(OP_RETURN == 30, "Bytecode v1 OP_RETURN value changed");
_Static_assert(OP_PRINT == 31, "Bytecode v1 OP_PRINT value changed");
_Static_assert(OP_HALT == 32, "Bytecode v1 OP_HALT value changed");
_Static_assert(OP_STATE_NEW_V3 == 42, "Bytecode v3 OP_STATE_NEW_V3 value changed");
_Static_assert(OP_STATE_BUILD_V3 == 43, "Bytecode v3 OP_STATE_BUILD_V3 value changed");
_Static_assert(OP_TRANSFORM_V3 == 44, "Bytecode v3 OP_TRANSFORM_V3 value changed");
_Static_assert(OP_MEASURE_V3 == 45, "Bytecode v3 OP_MEASURE_V3 value changed");
_Static_assert(OP_APPEND == 46, "Bytecode v3 OP_APPEND value changed");
_Static_assert(OP_SAMPLE_STATE_DIST == 47, "Bytecode v3 OP_SAMPLE_STATE_DIST value changed");
_Static_assert(OP_MEASURE_BASIS_V4 == 48, "Bytecode v4 OP_MEASURE_BASIS_V4 value changed");
_Static_assert(OP_ESTIMATE_MEASURE_PROBABILITY_V4 == 49, "Bytecode v4 probability opcode value changed");
_Static_assert(OP_ESTIMATE_MEASURE_DISTRIBUTION_V4 == 50, "Bytecode v4 distribution opcode value changed");
_Static_assert(OP_JOINT_BUILD_V5 == 51, "Bytecode v5 joint opcode value changed");
_Static_assert(OP_RESOLVE_V5 == 55, "Bytecode v5 resolve opcode value changed");
_Static_assert(OP_JOINT_BUILD_FINITE_V5 == 56, "Bytecode v5 finite-law opcode changed");
_Static_assert(OP_JOINT_RENAME_V5 == 57, "Bytecode v5 rename opcode changed");
_Static_assert(OP_POSSIBILITY_BUILD_V5 == 58, "Bytecode v5 possibility opcode changed");
_Static_assert(OP_PATH_SPLIT_V5 == 59, "Bytecode v5 path split opcode changed");
_Static_assert(OP_PATH_JOIN_V5 == 60, "Bytecode v5 path join opcode changed");
_Static_assert(OP_OBSERVE_V5 == 61, "Bytecode v5 observe opcode changed");
_Static_assert(OP_INFO_SAMPLE_V5 == 62, "Bytecode v5 information sample opcode changed");
_Static_assert(OP_COUNT == 63, "Bytecode v5 opcode count changed");

#define CHECK(condition) do { if (!(condition)) { (void)fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); return 1; } } while (0)

static SSInstruction instruction(OpCode opcode, uint32_t a, uint32_t b,
                                 uint32_t c, uint32_t imm) {
    SSInstruction value = {(uint8_t)opcode, a, b, c, imm, 1u};
    return value;
}

static int add_number(SSChunk *chunk, double number, uint32_t *index);

static int test_state_validation_and_apply(void) {
    SSState source, target;
    CHECK(ss_state_make(0.9, 0.65, &source) == SS_OK);
    CHECK(ss_state_make(0.5, 0.3, &target) == SS_OK);
    CHECK(ss_apply(&source, &target) == SS_OK);
    CHECK(fabs(target.p - 0.76) < SS_STATE_EPSILON);
    CHECK(fabs(target.d - 0.3) < SS_STATE_EPSILON);
    CHECK(ss_state_make(0.9, -0.8, &source) == SS_OK);
    CHECK(ss_state_make(0.5, 0.3, &target) == SS_OK);
    CHECK(ss_apply(&source, &target) == SS_OK);
    CHECK(fabs(target.p - 0.18) < SS_STATE_EPSILON);
    CHECK(fabs(target.d - 0.3) < SS_STATE_EPSILON);
    CHECK(ss_state_make(1.1, 0.0, &target) == SS_ERR_INVALID_PROBABILITY);
    CHECK(ss_state_make(0.5, 1.0, &target) == SS_ERR_INVALID_DEPENDENCY);
    CHECK(ss_state_make(0.5, -1.0, &target) == SS_ERR_INVALID_DEPENDENCY);
    return 0;
}

static int test_v3_state_canonicalization_and_transforms(void) {
    SSState state, transformed;
    double c_re, c_im, expectation;
    CHECK(ss_state_make_complex(0.5, 0.6, 0.8, &state) == SS_OK);
    CHECK(ss_state_disposition_squared(&state) == 1.0);
    ss_state_reconstruct_c(&state, &c_re, &c_im);
    CHECK(fabs(c_re - 0.3) < SS_STATE_EPSILON);
    CHECK(fabs(c_im - 0.4) < SS_STATE_EPSILON);
    CHECK(ss_state_make_complex(-SS_STATE_EPSILON / 2.0, 0.5, 0.5, &state) == SS_OK);
    CHECK(state.p == 0.0 && state.d_re == 0.0 && state.d_im == 0.0);
    CHECK(!signbit(state.p) && !signbit(state.d_re) && !signbit(state.d_im));
    CHECK(ss_state_make_complex(0.5, 1.0 + SS_STATE_EPSILON / 2.0, 0.0, &state) == SS_OK);
    CHECK(state.d_re == 1.0);
    CHECK(ss_state_make_complex(0.5, 1.0 + 2.0 * SS_STATE_EPSILON, 0.0, &state) == SS_ERR_INVALID_STATE);
    CHECK(ss_state_make_complex(0.25, 0.3, 0.4, &state) == SS_OK);
    CHECK(ss_transform_v3_apply(SS_TRANSFORM_V3_INVERT, &state, &transformed) == SS_OK);
    CHECK(transformed.p == 0.75 && transformed.d_re == 0.3 && transformed.d_im == -0.4);
    CHECK(ss_transform_v3_apply(SS_TRANSFORM_V3_INVERT, &transformed, &transformed) == SS_OK);
    CHECK(transformed.p == state.p && transformed.d_re == state.d_re && transformed.d_im == state.d_im);
    CHECK(ss_transform_v3_apply(SS_TRANSFORM_V3_NEUTRALIZE, &state, &transformed) == SS_OK);
    CHECK(transformed.p == state.p && transformed.d_re == 0.0 && transformed.d_im == 0.0);
    CHECK(ss_transform_v3_expected_probability(SS_TRANSFORM_V3_INVERT, 0.25, &expectation) == SS_OK);
    CHECK(expectation == 0.75);
    return 0;
}

static int test_v3_distribution_runtime(void) {
    SSChunk chunk;
    VM vm;
    uint32_t p_left, d_left_re, d_left_im, p_right, d_right_re, d_right_im;
    ss_chunk_init(&chunk);
    CHECK(add_number(&chunk, 0.5, &p_left) == 0);
    CHECK(add_number(&chunk, 0.3, &d_left_re) == 0);
    CHECK(add_number(&chunk, 0.4, &d_left_im) == 0);
    CHECK(add_number(&chunk, 0.5, &p_right) == 0);
    CHECK(add_number(&chunk, -0.3, &d_right_re) == 0);
    CHECK(add_number(&chunk, -0.4, &d_right_im) == 0);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_STATE_NEW_V3, 0, p_left, d_left_re, d_left_im)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_STATE_NEW_V3, 1, p_right, d_right_re, d_right_im)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_APPEND, 0, 1, 2, 0)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_MEASURE_V3, 2, 3, SS_MEASURE_V3_PROBABILITY, 0)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_TRANSFORM_V3, 4, 2, SS_TRANSFORM_V3_INVERT, 0)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_MEASURE_V3, 4, 5, SS_MEASURE_V3_PROBABILITY, 0)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_SAMPLE_STATE_DIST, 2, 6, 0, 0)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == SS_OK);
    ss_vm_init(&vm, &chunk);
    ss_vm_seed(&vm, 7u);
    CHECK(ss_vm_run(&vm) == SS_OK);
    CHECK(vm.frames[0].registers[2].type == VAL_STATE_DIST);
    CHECK(vm.frames[0].registers[3].as.number == 0.75);
    CHECK(vm.frames[0].registers[5].as.number == 0.25);
    CHECK(vm.frames[0].registers[6].type == VAL_STATE);
    CHECK(vm.frames[0].registers[6].as.state.state.p == 0.75);
    CHECK(ss_state_valid(&vm.frames[0].registers[6].as.state.state));
    ss_vm_free(&vm);
    ss_chunk_free(&chunk);
    return 0;
}

static int test_v4_basis_measurement_and_estimation(void) {
    SSChunk chunk;
    VM vm, repeat;
    uint32_t half, one, zero, negative_one;
    double estimate;
    ss_chunk_init(&chunk);
    CHECK(add_number(&chunk, 0.5, &half) == SS_OK);
    CHECK(add_number(&chunk, 1.0, &one) == SS_OK);
    CHECK(add_number(&chunk, 0.0, &zero) == SS_OK);
    CHECK(add_number(&chunk, -1.0, &negative_one) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_STATE_NEW_V3, 0, half, one, zero)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_STATE_NEW_V3, 1, half, negative_one, zero)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_STATE_NEW_V3, 2, half, zero, one)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_STATE_NEW_V3, 3, half, zero, negative_one)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_MEASURE_BASIS_V4, 0, 4, SS_MEASURE_BASIS_X,
                                            SS_MEASURE_V3_PROBABILITY)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_MEASURE_BASIS_V4, 1, 5, SS_MEASURE_BASIS_X,
                                            SS_MEASURE_V3_DISTRIBUTION)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_MEASURE_BASIS_V4, 2, 6, SS_MEASURE_BASIS_Y,
                                            SS_MEASURE_V3_PROBABILITY)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_MEASURE_BASIS_V4, 3, 7, SS_MEASURE_BASIS_Y,
                                            SS_MEASURE_V3_SAMPLE)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_APPEND, 0, 0, 8, 0)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_ESTIMATE_MEASURE_PROBABILITY_V4, 8, 9,
                                            SS_MEASURE_BASIS_X, 25)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_ESTIMATE_MEASURE_DISTRIBUTION_V4, 8, 10,
                                            SS_MEASURE_BASIS_Y, 25)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_MEASURE_BASIS_V4, 8, 11, SS_MEASURE_BASIS_X,
                                            SS_MEASURE_V3_SAMPLE)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_APPEND, 0, 1, 12, 0)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_ESTIMATE_MEASURE_PROBABILITY_V4, 12, 13,
                                            SS_MEASURE_BASIS_X, 64)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_ESTIMATE_MEASURE_DISTRIBUTION_V4, 12, 14,
                                            SS_MEASURE_BASIS_X, 64)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == SS_OK);
    ss_vm_init(&vm, &chunk);
    ss_vm_seed(&vm, 123u);
    CHECK(ss_vm_run(&vm) == SS_OK);
    CHECK(vm.frames[0].registers[4].type == VAL_NUMBER);
    CHECK(vm.frames[0].registers[4].as.number == 1.0);
    CHECK(vm.frames[0].registers[5].type == VAL_DISTRIBUTION);
    CHECK(vm.frames[0].registers[5].as.distribution.p0 == 1.0);
    CHECK(vm.frames[0].registers[5].as.distribution.p1 == 0.0);
    CHECK(vm.frames[0].registers[6].as.number == 0.0);
    CHECK(vm.frames[0].registers[7].type == VAL_SAMPLE);
    CHECK(fabs(vm.frames[0].registers[9].as.number -
               (0.5 + sqrt(0.75 * 0.25))) < SS_STATE_EPSILON);
    CHECK(vm.frames[0].registers[10].as.distribution.p0 == 0.5);
    CHECK(vm.frames[0].registers[10].as.distribution.p1 == 0.5);
    CHECK(vm.frames[0].registers[11].type == VAL_SAMPLE);
    CHECK(vm.frames[0].registers[13].type == VAL_NUMBER);
    CHECK(vm.frames[0].registers[13].as.number >= 0.0 &&
          vm.frames[0].registers[13].as.number <= 1.0);
    CHECK(vm.frames[0].registers[14].type == VAL_DISTRIBUTION);
    CHECK(vm.frames[0].registers[14].as.distribution.p0 >= 0.0 &&
          vm.frames[0].registers[14].as.distribution.p1 >= 0.0);
    CHECK(fabs(vm.frames[0].registers[14].as.distribution.p0 +
               vm.frames[0].registers[14].as.distribution.p1 - 1.0) < SS_STATE_EPSILON);
    CHECK(vm.frames[0].registers[0].as.state.state.p == 0.5);
    CHECK(vm.frames[0].registers[0].as.state.state.d_re == 1.0);
    CHECK(vm.frames[0].registers[0].as.state.state.d_im == 0.0);
    estimate = vm.frames[0].registers[9].as.number;

    ss_vm_init(&repeat, &chunk);
    ss_vm_seed(&repeat, 123u);
    CHECK(ss_vm_run(&repeat) == SS_OK);
    CHECK(repeat.frames[0].registers[9].as.number == estimate);
    CHECK(repeat.frames[0].registers[13].as.number == vm.frames[0].registers[13].as.number);
    ss_vm_free(&repeat);
    ss_vm_free(&vm);
    ss_chunk_free(&chunk);
    return 0;
}

static int test_v4_exact_distribution_rejection_and_limits(void) {
    SSChunk chunk;
    VM vm;
    uint32_t half, one, zero;
    SSErrorInfo error = {0};
    ss_chunk_init(&chunk);
    CHECK(add_number(&chunk, 0.5, &half) == SS_OK);
    CHECK(add_number(&chunk, 1.0, &one) == SS_OK);
    CHECK(add_number(&chunk, 0.0, &zero) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_STATE_NEW_V3, 0, half, one, zero)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_APPEND, 0, 0, 1, 0)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_MEASURE_BASIS_V4, 1, 2, SS_MEASURE_BASIS_X,
                                            SS_MEASURE_V3_PROBABILITY)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == SS_OK);
    ss_vm_init(&vm, &chunk);
    CHECK(ss_vm_run(&vm) == SS_ERR_UNSUPPORTED_EXACT_MEASUREMENT);
    ss_vm_free(&vm);
    ss_chunk_free(&chunk);

    ss_chunk_init(&chunk);
    CHECK(add_number(&chunk, 0.5, &half) == SS_OK);
    CHECK(add_number(&chunk, 1.0, &one) == SS_OK);
    CHECK(add_number(&chunk, 0.0, &zero) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_STATE_NEW_V3, 0, half, one, zero)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_APPEND, 0, 0, 1, 0)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_ESTIMATE_MEASURE_PROBABILITY_V4, 1, 2,
                                            SS_MEASURE_BASIS_X, 10)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == SS_OK);
    ss_vm_init(&vm, &chunk);
    vm.instruction_limit = 3u;
    CHECK(ss_vm_run(&vm) == SS_ERR_BUDGET_EXHAUSTED);
    CHECK(vm.frames[0].registers[2].type == VAL_NULL);
    ss_vm_free(&vm);
    ss_vm_init(&vm, &chunk);
    atomic_store(&vm.cancelled, true);
    CHECK(ss_vm_run(&vm) == SS_ERR_CANCELLED);
    CHECK(vm.frames[0].registers[2].type == VAL_NULL);
    ss_vm_free(&vm);
    ss_chunk_free(&chunk);

    ss_chunk_init(&chunk);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 99, 0)) == SS_OK);
    chunk.code[0] = instruction(OP_MEASURE_BASIS_V4, 0, 1, 99, SS_MEASURE_V3_PROBABILITY);
    CHECK(ss_chunk_verify(&chunk, &error) == SS_ERR_MEASURE);
    chunk.code[0] = instruction(OP_ESTIMATE_MEASURE_PROBABILITY_V4, 0, 1,
                                 SS_MEASURE_BASIS_X, 0);
    CHECK(ss_chunk_verify(&chunk, &error) == SS_ERR_FORMAT);
    chunk.version = 3u;
    chunk.code[0] = instruction(OP_MEASURE_BASIS_V4, 0, 1, SS_MEASURE_BASIS_X,
                                 SS_MEASURE_V3_PROBABILITY);
    CHECK(ss_chunk_verify(&chunk, &error) == SS_ERR_OPCODE);
    ss_chunk_free(&chunk);
    return 0;
}

static int test_v3_distribution_sharing_metadata_and_budget(void) {
    SSChunk empty;
    VM vm;
    SSState state_a, state_b;
    SSStateDist *dirac, *shared_append, *outer, *transformed;
    SSStateValue sampled;
    Value state_value_a, state_value_b, shared_value;
    double expectation;
    ss_chunk_init(&empty);
    ss_vm_init(&vm, &empty);
    CHECK(ss_state_make_complex(0.2, 0.5, 0.0, &state_a) == SS_OK);
    CHECK(ss_state_make_complex(0.3, -0.5, 0.0, &state_b) == SS_OK);
    state_value_a = ss_value_state(state_a);
    state_value_a.as.state.indexes.has_source = true;
    state_value_a.as.state.indexes.source = "captured";
    state_value_b = ss_value_state(state_b);
    CHECK(ss_vm_state_dist_dirac(&vm, &state_value_a.as.state, &dirac) == SS_OK);
    CHECK(ss_vm_state_dist_transform(&vm, SS_TRANSFORM_V3_NEUTRALIZE, dirac, &transformed) == SS_OK);
    CHECK(ss_vm_state_dist_sample(&vm, transformed, &sampled) == SS_OK);
    CHECK(sampled.indexes.has_source && strcmp(sampled.indexes.source, "captured") == 0);
    CHECK(sampled.state.p == 0.2 && sampled.state.d_re == 0.0 && sampled.state.d_im == 0.0);
    CHECK(ss_vm_state_dist_append(&vm, &state_value_a, &state_value_b, &shared_append) == SS_OK);
    shared_value = ss_value_state_dist(shared_append);
    CHECK(ss_vm_state_dist_append(&vm, &shared_value, &shared_value, &outer) == SS_OK);
    CHECK(outer->as.append.left == shared_append && outer->as.append.right == shared_append);
    CHECK(ss_vm_state_dist_expected_probability(outer, &expectation) == SS_OK);
    CHECK(fabs(expectation - (1.0 - 0.56 * 0.56)) < SS_STATE_EPSILON);
    vm.instruction_count = vm.instruction_limit;
    CHECK(ss_vm_state_dist_sample(&vm, shared_append, &sampled) == SS_ERR_BUDGET_EXHAUSTED);
    ss_vm_free(&vm);
    ss_chunk_free(&empty);
    return 0;
}

static int test_v3_state_and_distribution_equality(void) {
    SSChunk chunk;
    VM vm;
    uint32_t p, d_re, d_im;
    ss_chunk_init(&chunk);
    CHECK(add_number(&chunk, 0.5, &p) == 0);
    CHECK(add_number(&chunk, 0.25, &d_re) == 0);
    CHECK(add_number(&chunk, -0.25, &d_im) == 0);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_STATE_NEW_V3, 0, p, d_re, d_im)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_STATE_NEW_V3, 1, p, d_re, d_im)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_COMPARE, 0, 1, 2, SS_COMPARE_EQUAL)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == SS_OK);
    ss_vm_init(&vm, &chunk);
    CHECK(ss_vm_run(&vm) == SS_OK);
    CHECK(vm.frames[0].registers[2].type == VAL_BOOL && vm.frames[0].registers[2].as.boolean);
    ss_vm_free(&vm);
    ss_chunk_free(&chunk);

    ss_chunk_init(&chunk);
    CHECK(add_number(&chunk, 0.5, &p) == 0);
    CHECK(add_number(&chunk, 0.0, &d_re) == 0);
    CHECK(add_number(&chunk, 0.0, &d_im) == 0);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_STATE_NEW_V3, 0, p, d_re, d_im)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_APPEND, 0, 0, 1, 0)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_COMPARE, 1, 1, 2, SS_COMPARE_EQUAL)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == SS_OK);
    ss_vm_init(&vm, &chunk);
    CHECK(ss_vm_run(&vm) == SS_ERR_UNSUPPORTED_OPERATION);
    ss_vm_free(&vm);
    ss_chunk_free(&chunk);
    return 0;
}

static int add_number(SSChunk *chunk, double number, uint32_t *index) {
    return ss_chunk_add_constant(chunk, ss_value_number(number), index) == SS_OK ? 0 : 1;
}

static int test_vm_native_state_program(void) {
    SSChunk chunk; VM vm; uint32_t c05, c03, c09, c065;
    ss_chunk_init(&chunk);
    CHECK(add_number(&chunk, 0.5, &c05) == 0);
    CHECK(add_number(&chunk, 0.3, &c03) == 0);
    CHECK(add_number(&chunk, 0.9, &c09) == 0);
    CHECK(add_number(&chunk, 0.65, &c065) == 0);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_STATE_NEW, 0, c05, c03, SS_NO_OPERAND)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_STATE_NEW, 1, c09, c065, SS_NO_OPERAND)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_APPLY, 1, 0, 0, 0)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_MEASURE, 0, 2, SS_MEASURE_DISTRIBUTION, 0)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == SS_OK);
    ss_vm_init(&vm, &chunk);
    CHECK(ss_vm_run(&vm) == SS_OK);
    CHECK(vm.frames[0].registers[0].type == VAL_STATE);
    CHECK(fabs(vm.frames[0].registers[0].as.state.state.p - 0.76) < SS_STATE_EPSILON);
    CHECK(vm.frames[0].registers[2].type == VAL_DISTRIBUTION);
    CHECK(fabs(vm.frames[0].registers[2].as.distribution.p1 - 0.76) < SS_STATE_EPSILON);
    ss_vm_free(&vm); ss_chunk_free(&chunk);
    return 0;
}

static int test_composition(void) {
    SSState left, right, merged, left_out, right_out;
    CHECK(ss_state_make(0.2, 0.4, &left) == SS_OK);
    CHECK(ss_state_make(0.8, 0.6, &right) == SS_OK);
    CHECK(ss_compose_merge(&left, &right, &merged) == SS_OK);
    CHECK(fabs(merged.p - 0.5) < SS_STATE_EPSILON);
    CHECK(fabs(merged.d - 0.5) < SS_STATE_EPSILON);
    CHECK(ss_compose_update(&left, &right, &left_out, &right_out) == SS_OK);
    CHECK(fabs(left_out.p - 0.56) < SS_STATE_EPSILON);
    CHECK(fabs(right_out.p - 0.56) < SS_STATE_EPSILON);
    return 0;
}

static int test_multi_state_mean_apply(void) {
    SSChunk chunk; VM vm;
    uint32_t p09, d04, p07, d06, p02, d01;
    ss_chunk_init(&chunk);
    CHECK(add_number(&chunk, 0.9, &p09) == 0);
    CHECK(add_number(&chunk, 0.4, &d04) == 0);
    CHECK(add_number(&chunk, 0.7, &p07) == 0);
    CHECK(add_number(&chunk, 0.6, &d06) == 0);
    CHECK(add_number(&chunk, 0.2, &p02) == 0);
    CHECK(add_number(&chunk, 0.1, &d01) == 0);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_STATE_NEW, 0, p09, d04, 0)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_STATE_NEW, 1, p07, d06, 0)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_STATE_NEW, 2, p02, d01, 0)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_APPLY_MANY, 0, 2, 2, SS_AGG_MEAN)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == SS_OK);
    ss_vm_init(&vm, &chunk);
    CHECK(ss_vm_run(&vm) == SS_OK);
    CHECK(fabs(vm.frames[0].registers[2].as.state.state.p - 0.5) < SS_STATE_EPSILON);
    CHECK(fabs(vm.frames[0].registers[2].as.state.state.d - 0.1) < SS_STATE_EPSILON);
    ss_vm_free(&vm); ss_chunk_free(&chunk);
    return 0;
}

static int test_transform_instruction(void) {
    SSChunk chunk; VM vm; uint32_t p08, d05, rate;
    ss_chunk_init(&chunk);
    CHECK(add_number(&chunk, 0.8, &p08) == 0);
    CHECK(add_number(&chunk, 0.5, &d05) == 0);
    CHECK(add_number(&chunk, 0.5, &rate) == 0);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_STATE_NEW, 0, p08, d05, 0)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_LOAD_CONST, 1, 0, 0, rate)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_TRANSFORM, 0, SS_TRANSFORM_DECAY, 1, 1)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == SS_OK);
    ss_vm_init(&vm, &chunk);
    CHECK(ss_vm_run(&vm) == SS_OK);
    CHECK(fabs(vm.frames[0].registers[0].as.state.state.p - 0.65) < SS_STATE_EPSILON);
    CHECK(fabs(vm.frames[0].registers[0].as.state.state.d - 0.25) < SS_STATE_EPSILON);
    ss_vm_free(&vm); ss_chunk_free(&chunk);
    return 0;
}

static int test_measurement_indexes_and_history(void) {
    SSChunk chunk; VM vm;
    uint32_t p04, d02, amount, timestamp, p09, d05;
    ss_chunk_init(&chunk);
    CHECK(add_number(&chunk, 0.4, &p04) == 0);
    CHECK(add_number(&chunk, 0.2, &d02) == 0);
    CHECK(add_number(&chunk, 2.0, &amount) == 0);
    CHECK(add_number(&chunk, 10.0, &timestamp) == 0);
    CHECK(add_number(&chunk, 0.9, &p09) == 0);
    CHECK(add_number(&chunk, 0.5, &d05) == 0);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_STATE_NEW, 0, p04, d02, SS_NO_OPERAND)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_LOAD_CONST, 1, 0, 0, amount)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_HISTORY_CONFIG, 0, 1, SS_HISTORY_LATEST, 0)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_LOAD_CONST, 8, 0, 0, timestamp)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_SET_INDEX, 0, 0, 8, 0)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_GET_INDEX, 0, 9, 0, 0)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_STATE_NEW, 2, p09, d05, SS_NO_OPERAND)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_APPLY, 2, 0, 0, 0)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_PREVIOUS, 0, 3, 0, 0)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_CHANGE, 0, 4, 0, 0)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_MEASURE, 0, 5, SS_MEASURE_PROBABILITY, 0)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_MEASURE, 0, 6, SS_MEASURE_DISTRIBUTION, 0)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_MEASURE, 0, 7, SS_MEASURE_COLLAPSE, 0)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == SS_OK);
    ss_vm_init(&vm, &chunk); ss_vm_seed(&vm, 7u);
    CHECK(ss_vm_run(&vm) == SS_OK);
    CHECK(vm.frames[0].registers[3].type == VAL_STATE);
    CHECK(fabs(vm.frames[0].registers[3].as.state.state.p - 0.4) < SS_STATE_EPSILON);
    CHECK(fabs(vm.frames[0].registers[4].as.number - 0.25) < SS_STATE_EPSILON);
    CHECK(fabs(vm.frames[0].registers[5].as.number - 0.65) < SS_STATE_EPSILON);
    CHECK(fabs(vm.frames[0].registers[6].as.distribution.p0 - 0.35) < SS_STATE_EPSILON);
    CHECK(fabs(vm.frames[0].registers[9].as.number - 10.0) < SS_STATE_EPSILON);
    CHECK(vm.frames[0].registers[7].type == VAL_SAMPLE);
    CHECK(vm.frames[0].registers[0].as.state.state.p == (double)vm.frames[0].registers[7].as.sample);
    CHECK(vm.frames[0].registers[0].as.state.state.d == 0.0);
    ss_vm_free(&vm); ss_chunk_free(&chunk);
    return 0;
}

static int test_verifier_rejects_bad_opcode_and_jump(void) {
    SSChunk chunk; SSErrorInfo error = {0}; uint32_t zero;
    ss_chunk_init(&chunk);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == SS_OK);
    chunk.code[0].opcode = 255u;
    CHECK(ss_chunk_verify(&chunk, &error) == SS_ERR_OPCODE);
    chunk.code[0] = instruction(OP_JUMP, 0, 0, 0, 99u);
    CHECK(ss_chunk_verify(&chunk, &error) == SS_ERR_JUMP);
    chunk.code[0] = instruction(OP_HALT, 0, 0, 0, 0);
    chunk.version = 6u;
    CHECK(ss_chunk_verify(&chunk, &error) == SS_ERR_FORMAT);
    chunk.version = 2u;
    CHECK(add_number(&chunk, 0.0, &zero) == 0);
    chunk.code[0] = instruction(OP_STATE_NEW_V3, 0, zero, zero, zero);
    CHECK(ss_chunk_verify(&chunk, &error) == SS_ERR_OPCODE);
    ss_chunk_free(&chunk);
    return 0;
}

static int test_assembler_serialization_and_loader(void) {
    SSChunk assembled, loaded; SSErrorInfo error = {0}; VM vm;
    const char *binary = "runtime-test.ssb";
    CHECK(ss_assemble_file(SS_SOURCE_DIR "/examples/compatibility/belief_v1.ssa", &assembled, &error) == SS_OK);
    CHECK(ss_chunk_write_file(&assembled, binary, &error) == SS_OK);
    CHECK(ss_chunk_read_file(&loaded, binary, &error) == SS_OK);
    CHECK(loaded.code_count == assembled.code_count);
    ss_vm_init(&vm, &loaded); CHECK(ss_vm_run(&vm) == SS_OK);
    CHECK(fabs(vm.frames[0].registers[0].as.state.state.p - 0.76) < SS_STATE_EPSILON);
    CHECK(vm.frames[0].registers[2].type == VAL_NUMBER);
    CHECK(fabs(vm.frames[0].registers[2].as.number - 0.76) < SS_STATE_EPSILON);
    ss_vm_free(&vm); ss_chunk_free(&loaded);
    assembled.version = 1u;
    CHECK(ss_chunk_write_file(&assembled, binary, &error) == SS_OK);
    CHECK(ss_chunk_read_file(&loaded, binary, &error) == SS_OK);
    CHECK(loaded.version == 1u);
    ss_vm_init(&vm, &loaded); CHECK(ss_vm_run(&vm) == SS_OK);
    CHECK(fabs(vm.frames[0].registers[0].as.state.state.p - 0.76) < SS_STATE_EPSILON);
    ss_vm_free(&vm); ss_chunk_free(&loaded); ss_chunk_free(&assembled);
    CHECK(remove(binary) == 0);
    return 0;
}

static int test_v5_named_joint_information(void) {
    SSChunk chunk;
    VM vm;
    Value rows[9];
    double weights[3] = {0.2, 0.3, 0.5};
    SSJointState *joint = NULL, *projected = NULL, *conditioned = NULL;
    SSJointState *renamed = NULL;
    Value evidence = ss_value_bool(true), sampled, resolved;
    size_t index;
    ss_chunk_init(&chunk);
    chunk.version = 5u;
    ss_vm_init(&vm, &chunk);
    rows[0] = ss_value_bool(false); rows[1] = ss_value_number(0.0); rows[2] = ss_value_number(10.0);
    rows[3] = ss_value_bool(true);  rows[4] = ss_value_number(1.0); rows[5] = ss_value_number(11.0);
    rows[6] = ss_value_bool(false); rows[7] = ss_value_number(2.0); rows[8] = ss_value_number(12.0);
    CHECK(ss_vm_joint_build_finite(&vm, "z;x;y", rows, weights, 3u, 3u, &joint) == SS_OK);
    CHECK(joint->count == 3u && joint->row_count == 3u);
    CHECK(joint->kind == SS_JOINT_FINITE_LAW);
    CHECK(strcmp(joint->names[0], "x") == 0);
    CHECK(strcmp(joint->names[1], "y") == 0);
    CHECK(strcmp(joint->names[2], "z") == 0);
    CHECK(ss_vm_joint_project(&vm, joint, "y;x", &projected) == SS_OK);
    CHECK(projected->row_count == 3u && projected->count == 2u);
    CHECK(ss_vm_joint_condition(&vm, joint, "z", &evidence, &conditioned) == SS_OK);
    CHECK(conditioned->row_count == 1u);
    CHECK(vm.observation_count == 0u);
    CHECK(ss_vm_joint_observe(&vm, joint, "z", &evidence, &conditioned) == SS_OK);
    CHECK(vm.observation_count == 1u);
    CHECK(ss_vm_joint_resolve(&vm, conditioned, &resolved) == SS_OK);
    CHECK(resolved.type == VAL_ARRAY && resolved.as.array->count == 3u);
    CHECK(resolved.as.array->items[0].as.number == 1.0);
    CHECK(resolved.as.array->items[1].as.number == 11.0);
    for (index = 0; index < 100u; ++index) {
        CHECK(ss_vm_joint_sample(&vm, joint, &sampled) == SS_OK);
        CHECK(sampled.type == VAL_ARRAY && sampled.as.array->count == 3u);
        CHECK(sampled.as.array->items[1].as.number ==
              sampled.as.array->items[0].as.number + 10.0);
    }
    CHECK(ss_vm_joint_resolve(&vm, joint, &resolved) == SS_ERR_UNRESOLVED_VALUE);
    CHECK(ss_vm_joint_rename(&vm, joint, "x", "subject", &renamed) == SS_OK);
    CHECK(strcmp(renamed->names[0], "subject") == 0);
    CHECK(ss_vm_joint_rename(&vm, joint, "x", "y", &renamed) ==
          SS_ERR_INVALID_DEPENDENCY);
    CHECK(ss_vm_joint_project(&vm, joint, "x;x", &projected) ==
          SS_ERR_INVALID_DEPENDENCY);
    evidence = ss_value_number(99.0);
    CHECK(ss_vm_joint_condition(&vm, joint, "x", &evidence, &conditioned) ==
          SS_ERR_INVALID_CONDITIONING);
    CHECK(ss_vm_joint_build(&vm, rows, 2u, "independent:x;x", &conditioned) ==
          SS_ERR_INVALID_DEPENDENCY);
    CHECK(ss_vm_joint_build(&vm, rows, 2u, "correlated:x;y", &conditioned) ==
          SS_ERR_UNSUPPORTED_OPERATION);
    weights[0] = -0.2;
    CHECK(ss_vm_joint_build_finite(&vm, "z;x;y", rows, weights, 3u, 3u,
                                   &conditioned) == SS_ERR_INVALID_DISTRIBUTION);
    ss_vm_free(&vm);
    ss_chunk_free(&chunk);
    return 0;
}

static int test_v5_guarded_path_limits(void) {
    SSChunk chunk;
    VM vm;
    uint32_t yes, no, ten, twenty;
    ss_chunk_init(&chunk); chunk.version = 5u;
    CHECK(ss_chunk_add_constant(&chunk, ss_value_bool(true), &yes) == SS_OK);
    CHECK(ss_chunk_add_constant(&chunk, ss_value_bool(false), &no) == SS_OK);
    CHECK(ss_chunk_add_constant(&chunk, ss_value_number(10.0), &ten) == SS_OK);
    CHECK(ss_chunk_add_constant(&chunk, ss_value_number(20.0), &twenty) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_LOAD_CONST, 0, 0, 0, yes)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_LOAD_CONST, 1, 0, 0, no)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_ARRAY_NEW, 2, 0, 2, 0)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_POSSIBILITY_BUILD_V5, 2, 3, 0, 0)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_PATH_SPLIT_V5, 3, 0, 0, 7)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_LOAD_CONST, 4, 0, 0, ten)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_JUMP, 0, 0, 0, 8)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_LOAD_CONST, 4, 0, 0, twenty)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_PATH_JOIN_V5, 0, 0, 0, 0)) == SS_OK);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == SS_OK);
    CHECK(ss_chunk_verify(&chunk, NULL) == SS_OK);
    ss_vm_init(&vm, &chunk); vm.path_limit = 1u;
    CHECK(ss_vm_run(&vm) == SS_ERR_PATH_LIMIT);
    CHECK(vm.frames[0].registers[4].type == VAL_NULL);
    ss_vm_free(&vm);
    ss_vm_init(&vm, &chunk);
    CHECK(ss_vm_run(&vm) == SS_OK);
    CHECK(vm.frames[0].registers[4].type == VAL_PATH_SET);
    CHECK(vm.frames[0].registers[4].as.paths->count == 2u);
    CHECK(vm.frames[0].registers[4].as.paths->alternatives[0].guard);
    CHECK(!vm.frames[0].registers[4].as.paths->alternatives[1].guard);
    ss_vm_free(&vm); ss_chunk_free(&chunk);
    return 0;
}

static int test_assembly_probe(void) {
    static const double cases[][3] = {
        {0.9, 0.65, 0.5}, {0.0, 0.999, 1.0}, {1.0, -0.999, 0.0},
        {0.25, 0.0, 0.75}, {1.0, 0.999, 0.0}
    };
    size_t index;
    for (index = 0; index < sizeof(cases) / sizeof(cases[0]); ++index) {
        SSState source, target;
        double assembly;
        CHECK(ss_state_make(cases[index][0], cases[index][1], &source) == SS_OK);
        CHECK(ss_state_make(cases[index][2], 0.0, &target) == SS_OK);
        CHECK(ss_apply(&source, &target) == SS_OK);
        assembly = ss_asm_apply_probe(cases[index][0], cases[index][1], cases[index][2]);
        CHECK(fabs(assembly - target.p) < SS_STATE_EPSILON);
    }
    return 0;
}

int main(void) {
    CHECK(test_state_validation_and_apply() == 0);
    CHECK(test_v3_state_canonicalization_and_transforms() == 0);
    CHECK(test_v3_distribution_runtime() == 0);
    CHECK(test_v4_basis_measurement_and_estimation() == 0);
    CHECK(test_v4_exact_distribution_rejection_and_limits() == 0);
    CHECK(test_v3_distribution_sharing_metadata_and_budget() == 0);
    CHECK(test_v3_state_and_distribution_equality() == 0);
    CHECK(test_vm_native_state_program() == 0);
    CHECK(test_composition() == 0);
    CHECK(test_multi_state_mean_apply() == 0);
    CHECK(test_transform_instruction() == 0);
    CHECK(test_measurement_indexes_and_history() == 0);
    CHECK(test_verifier_rejects_bad_opcode_and_jump() == 0);
    CHECK(test_assembler_serialization_and_loader() == 0);
    CHECK(test_v5_named_joint_information() == 0);
    CHECK(test_v5_guarded_path_limits() == 0);
    CHECK(test_assembly_probe() == 0);
    (void)printf("C runtime tests passed\n");
    return 0;
}
