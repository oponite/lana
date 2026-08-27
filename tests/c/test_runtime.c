#include "lana/assembler.h"
#include "lana/data.h"
#include "lana/vm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(OP_STATE_NEW == 3, "LABC v1 STATE_NEW value changed");
_Static_assert(OP_MEASURE == 6, "LABC v1 MEASURE value changed");
_Static_assert(OP_RETURN == 29, "LABC v1 RETURN value changed");
_Static_assert(OP_HALT == 31, "LABC v1 HALT value changed");
_Static_assert(OP_HOST_CALL == 39, "LABC v1 HOST_CALL value changed");
_Static_assert(OP_JOINT_BUILD == 40, "LABC v1 JOINT_BUILD value changed");
_Static_assert(OP_EVIDENCE == 52, "LABC v1 EVIDENCE value changed");
_Static_assert(OP_EXPLAIN == 55, "LABC v1 EXPLAIN value changed");
_Static_assert(OP_COUNT == 56, "LABC v1 opcode count changed");

#define CHECK(condition) do { if (!(condition)) { (void)fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); return 1; } } while (0)

static LanaInstruction instruction(OpCode opcode, uint32_t a, uint32_t b,
                                 uint32_t c, uint32_t imm) {
    LanaInstruction value = {(uint8_t)opcode, a, b, c, imm, 1u};
    return value;
}

static int add_number(LanaChunk *chunk, double number, uint32_t *index);

static int test_state_construction(void) {
    LanaState state;
    CHECK(lana_state_make(0.9, 0.65, &state) == LANA_OK);
    CHECK(state.p == 0.9 && state.d_re == 0.65 && state.d_im == 0.0);
    CHECK(lana_state_make(0.5, 1.0, &state) == LANA_OK);
    CHECK(lana_state_make(0.5, -1.0, &state) == LANA_OK);
    CHECK(lana_state_make(1.1, 0.0, &state) == LANA_ERR_INVALID_STATE);
    CHECK(lana_state_make(0.5, 1.1, &state) == LANA_ERR_INVALID_STATE);
    return 0;
}

static int test_state_canonicalization_and_transforms(void) {
    LanaState state, transformed;
    double c_re, c_im, expectation;
    CHECK(lana_state_make_complex(0.5, 0.6, 0.8, &state) == LANA_OK);
    CHECK(lana_state_disposition_squared(&state) == 1.0);
    lana_state_reconstruct_c(&state, &c_re, &c_im);
    CHECK(fabs(c_re - 0.3) < LANA_STATE_EPSILON);
    CHECK(fabs(c_im - 0.4) < LANA_STATE_EPSILON);
    CHECK(lana_state_make_complex(-LANA_STATE_EPSILON / 2.0, 0.5, 0.5, &state) == LANA_OK);
    CHECK(state.p == 0.0 && state.d_re == 0.0 && state.d_im == 0.0);
    CHECK(!signbit(state.p) && !signbit(state.d_re) && !signbit(state.d_im));
    CHECK(lana_state_make_complex(0.5, 1.0 + LANA_STATE_EPSILON / 2.0, 0.0, &state) == LANA_OK);
    CHECK(state.d_re == 1.0);
    CHECK(lana_state_make_complex(0.5, 1.0 + 2.0 * LANA_STATE_EPSILON, 0.0, &state) == LANA_ERR_INVALID_STATE);
    CHECK(lana_state_make_complex(0.25, 0.3, 0.4, &state) == LANA_OK);
    CHECK(lana_transform_apply(LANA_TRANSFORM_INVERT, &state, &transformed) == LANA_OK);
    CHECK(transformed.p == 0.75 && transformed.d_re == 0.3 && transformed.d_im == -0.4);
    CHECK(lana_transform_apply(LANA_TRANSFORM_INVERT, &transformed, &transformed) == LANA_OK);
    CHECK(transformed.p == state.p && transformed.d_re == state.d_re && transformed.d_im == state.d_im);
    CHECK(lana_transform_apply(LANA_TRANSFORM_NEUTRALIZE, &state, &transformed) == LANA_OK);
    CHECK(transformed.p == state.p && transformed.d_re == 0.0 && transformed.d_im == 0.0);
    CHECK(lana_transform_expected_probability(LANA_TRANSFORM_INVERT, 0.25, &expectation) == LANA_OK);
    CHECK(expectation == 0.75);
    return 0;
}

static int test_distribution_runtime(void) {
    LanaChunk chunk;
    LanaVM vm;
    uint32_t p_left, d_left_re, d_left_im, p_right, d_right_re, d_right_im;
    lana_chunk_init(&chunk);
    CHECK(add_number(&chunk, 0.5, &p_left) == 0);
    CHECK(add_number(&chunk, 0.3, &d_left_re) == 0);
    CHECK(add_number(&chunk, 0.4, &d_left_im) == 0);
    CHECK(add_number(&chunk, 0.5, &p_right) == 0);
    CHECK(add_number(&chunk, -0.3, &d_right_re) == 0);
    CHECK(add_number(&chunk, -0.4, &d_right_im) == 0);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_STATE_NEW, 0, p_left, d_left_re, d_left_im)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_STATE_NEW, 1, p_right, d_right_re, d_right_im)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_APPEND, 0, 1, 2, 0)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_MEASURE, 2, 3, LANA_MEASURE_PROBABILITY, 0)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_TRANSFORM, 4, 2, LANA_TRANSFORM_INVERT, 0)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_MEASURE, 4, 5, LANA_MEASURE_PROBABILITY, 0)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_SAMPLE_STATE_DIST, 2, 6, 0, 0)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    lana_vm_seed(&vm, 7u);
    CHECK(lana_vm_run(&vm) == LANA_OK);
    CHECK(vm.frames[0].registers[2].type == VAL_STATE_DIST);
    CHECK(vm.frames[0].registers[3].as.number == 0.75);
    CHECK(vm.frames[0].registers[5].as.number == 0.25);
    CHECK(vm.frames[0].registers[6].type == VAL_STATE);
    CHECK(vm.frames[0].registers[6].as.state.state.p == 0.75);
    CHECK(lana_state_valid(&vm.frames[0].registers[6].as.state.state));
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);
    return 0;
}

static int test_basis_measurement_and_estimation(void) {
    LanaChunk chunk;
    LanaVM vm, repeat;
    uint32_t half, one, zero, negative_one;
    double estimate;
    lana_chunk_init(&chunk);
    CHECK(add_number(&chunk, 0.5, &half) == LANA_OK);
    CHECK(add_number(&chunk, 1.0, &one) == LANA_OK);
    CHECK(add_number(&chunk, 0.0, &zero) == LANA_OK);
    CHECK(add_number(&chunk, -1.0, &negative_one) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_STATE_NEW, 0, half, one, zero)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_STATE_NEW, 1, half, negative_one, zero)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_STATE_NEW, 2, half, zero, one)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_STATE_NEW, 3, half, zero, negative_one)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_MEASURE_BASIS, 0, 4, LANA_MEASURE_BASIS_X,
                                            LANA_MEASURE_PROBABILITY)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_MEASURE_BASIS, 1, 5, LANA_MEASURE_BASIS_X,
                                            LANA_MEASURE_DISTRIBUTION)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_MEASURE_BASIS, 2, 6, LANA_MEASURE_BASIS_Y,
                                            LANA_MEASURE_PROBABILITY)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_MEASURE_BASIS, 3, 7, LANA_MEASURE_BASIS_Y,
                                            LANA_MEASURE_SAMPLE)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_APPEND, 0, 0, 8, 0)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_ESTIMATE_MEASURE_PROBABILITY, 8, 9,
                                            LANA_MEASURE_BASIS_X, 25)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_ESTIMATE_MEASURE_DISTRIBUTION, 8, 10,
                                            LANA_MEASURE_BASIS_Y, 25)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_MEASURE_BASIS, 8, 11, LANA_MEASURE_BASIS_X,
                                            LANA_MEASURE_SAMPLE)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_APPEND, 0, 1, 12, 0)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_ESTIMATE_MEASURE_PROBABILITY, 12, 13,
                                            LANA_MEASURE_BASIS_X, 64)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_ESTIMATE_MEASURE_DISTRIBUTION, 12, 14,
                                            LANA_MEASURE_BASIS_X, 64)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    lana_vm_seed(&vm, 123u);
    CHECK(lana_vm_run(&vm) == LANA_OK);
    CHECK(vm.frames[0].registers[4].type == VAL_NUMBER);
    CHECK(vm.frames[0].registers[4].as.number == 1.0);
    CHECK(vm.frames[0].registers[5].type == VAL_DISTRIBUTION);
    CHECK(vm.frames[0].registers[5].as.distribution.p0 == 1.0);
    CHECK(vm.frames[0].registers[5].as.distribution.p1 == 0.0);
    CHECK(vm.frames[0].registers[6].as.number == 0.0);
    CHECK(vm.frames[0].registers[7].type == VAL_SAMPLE);
    CHECK(fabs(vm.frames[0].registers[9].as.number -
               (0.5 + sqrt(0.75 * 0.25))) < LANA_STATE_EPSILON);
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
               vm.frames[0].registers[14].as.distribution.p1 - 1.0) < LANA_STATE_EPSILON);
    CHECK(vm.frames[0].registers[0].as.state.state.p == 0.5);
    CHECK(vm.frames[0].registers[0].as.state.state.d_re == 1.0);
    CHECK(vm.frames[0].registers[0].as.state.state.d_im == 0.0);
    estimate = vm.frames[0].registers[9].as.number;

    lana_vm_init(&repeat, &chunk);
    lana_vm_seed(&repeat, 123u);
    CHECK(lana_vm_run(&repeat) == LANA_OK);
    CHECK(repeat.frames[0].registers[9].as.number == estimate);
    CHECK(repeat.frames[0].registers[13].as.number == vm.frames[0].registers[13].as.number);
    lana_vm_free(&repeat);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);
    return 0;
}

static int test_exact_distribution_rejection_and_limits(void) {
    LanaChunk chunk;
    LanaVM vm;
    uint32_t half, one, zero;
    LanaErrorInfo error = {0};
    lana_chunk_init(&chunk);
    CHECK(add_number(&chunk, 0.5, &half) == LANA_OK);
    CHECK(add_number(&chunk, 1.0, &one) == LANA_OK);
    CHECK(add_number(&chunk, 0.0, &zero) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_STATE_NEW, 0, half, one, zero)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_APPEND, 0, 0, 1, 0)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_MEASURE_BASIS, 1, 2, LANA_MEASURE_BASIS_X,
                                            LANA_MEASURE_PROBABILITY)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    CHECK(lana_vm_run(&vm) == LANA_ERR_UNSUPPORTED_EXACT_MEASUREMENT);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);

    lana_chunk_init(&chunk);
    CHECK(add_number(&chunk, 0.5, &half) == LANA_OK);
    CHECK(add_number(&chunk, 1.0, &one) == LANA_OK);
    CHECK(add_number(&chunk, 0.0, &zero) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_STATE_NEW, 0, half, one, zero)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_APPEND, 0, 0, 1, 0)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_ESTIMATE_MEASURE_PROBABILITY, 1, 2,
                                            LANA_MEASURE_BASIS_X, 10)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    vm.instruction_limit = 3u;
    CHECK(lana_vm_run(&vm) == LANA_ERR_BUDGET_EXHAUSTED);
    CHECK(vm.frames[0].registers[2].type == VAL_NULL);
    lana_vm_free(&vm);
    lana_vm_init(&vm, &chunk);
    atomic_store(&vm.cancelled, true);
    CHECK(lana_vm_run(&vm) == LANA_ERR_CANCELLED);
    CHECK(vm.frames[0].registers[2].type == VAL_NULL);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);

    lana_chunk_init(&chunk);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 99, 0)) == LANA_OK);
    chunk.code[0] = instruction(OP_MEASURE_BASIS, 0, 1, 99, LANA_MEASURE_PROBABILITY);
    CHECK(lana_chunk_verify(&chunk, &error) == LANA_ERR_MEASURE);
    chunk.code[0] = instruction(OP_ESTIMATE_MEASURE_PROBABILITY, 0, 1,
                                 LANA_MEASURE_BASIS_X, 0);
    CHECK(lana_chunk_verify(&chunk, &error) == LANA_ERR_FORMAT);
    chunk.version = 2u;
    chunk.code[0] = instruction(OP_MEASURE_BASIS, 0, 1, LANA_MEASURE_BASIS_X,
                                 LANA_MEASURE_PROBABILITY);
    CHECK(lana_chunk_verify(&chunk, &error) == LANA_ERR_FORMAT);
    lana_chunk_free(&chunk);
    return 0;
}

static int test_distribution_sharing_metadata_and_budget(void) {
    LanaChunk empty;
    LanaVM vm;
    LanaState state_a, state_b;
    LanaStateDist *dirac, *shared_append, *outer, *transformed;
    LanaStateValue sampled;
    Value state_value_a, state_value_b, shared_value;
    double expectation;
    lana_chunk_init(&empty);
    lana_vm_init(&vm, &empty);
    CHECK(lana_state_make_complex(0.2, 0.5, 0.0, &state_a) == LANA_OK);
    CHECK(lana_state_make_complex(0.3, -0.5, 0.0, &state_b) == LANA_OK);
    state_value_a = lana_value_state(state_a);
    state_value_a.as.state.indexes.has_source = true;
    state_value_a.as.state.indexes.source = "captured";
    state_value_b = lana_value_state(state_b);
    CHECK(lana_vm_state_dist_dirac(&vm, &state_value_a.as.state, &dirac) == LANA_OK);
    CHECK(lana_vm_state_dist_transform(&vm, LANA_TRANSFORM_NEUTRALIZE, dirac, &transformed) == LANA_OK);
    CHECK(lana_vm_state_dist_sample(&vm, transformed, &sampled) == LANA_OK);
    CHECK(sampled.indexes.has_source && strcmp(sampled.indexes.source, "captured") == 0);
    CHECK(sampled.state.p == 0.2 && sampled.state.d_re == 0.0 && sampled.state.d_im == 0.0);
    CHECK(lana_vm_state_dist_append(&vm, &state_value_a, &state_value_b, &shared_append) == LANA_OK);
    shared_value = lana_value_state_dist(shared_append);
    CHECK(lana_vm_state_dist_append(&vm, &shared_value, &shared_value, &outer) == LANA_OK);
    CHECK(outer->as.append.left == shared_append && outer->as.append.right == shared_append);
    CHECK(lana_vm_state_dist_expected_probability(outer, &expectation) == LANA_OK);
    CHECK(fabs(expectation - (1.0 - 0.56 * 0.56)) < LANA_STATE_EPSILON);
    vm.instruction_count = vm.instruction_limit;
    CHECK(lana_vm_state_dist_sample(&vm, shared_append, &sampled) == LANA_ERR_BUDGET_EXHAUSTED);
    lana_vm_free(&vm);
    lana_chunk_free(&empty);
    return 0;
}

static int test_state_and_distribution_equality(void) {
    LanaChunk chunk;
    LanaVM vm;
    uint32_t p, d_re, d_im;
    lana_chunk_init(&chunk);
    CHECK(add_number(&chunk, 0.5, &p) == 0);
    CHECK(add_number(&chunk, 0.25, &d_re) == 0);
    CHECK(add_number(&chunk, -0.25, &d_im) == 0);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_STATE_NEW, 0, p, d_re, d_im)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_STATE_NEW, 1, p, d_re, d_im)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_COMPARE, 0, 1, 2, LANA_COMPARE_EQUAL)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    CHECK(lana_vm_run(&vm) == LANA_OK);
    CHECK(vm.frames[0].registers[2].type == VAL_BOOL && vm.frames[0].registers[2].as.boolean);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);

    lana_chunk_init(&chunk);
    CHECK(add_number(&chunk, 0.5, &p) == 0);
    CHECK(add_number(&chunk, 0.0, &d_re) == 0);
    CHECK(add_number(&chunk, 0.0, &d_im) == 0);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_STATE_NEW, 0, p, d_re, d_im)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_APPEND, 0, 0, 1, 0)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_COMPARE, 1, 1, 2, LANA_COMPARE_EQUAL)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    CHECK(lana_vm_run(&vm) == LANA_ERR_UNSUPPORTED_OPERATION);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);
    return 0;
}

static int add_number(LanaChunk *chunk, double number, uint32_t *index) {
    return lana_chunk_add_constant(chunk, lana_value_number(number), index) == LANA_OK ? 0 : 1;
}

static int test_verifier_rejects_bad_opcode_and_jump(void) {
    LanaChunk chunk; LanaErrorInfo error = {0};
    lana_chunk_init(&chunk);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    chunk.code[0].opcode = 255u;
    CHECK(lana_chunk_verify(&chunk, &error) == LANA_ERR_OPCODE);
    chunk.code[0] = instruction(OP_JUMP, 0, 0, 0, 99u);
    CHECK(lana_chunk_verify(&chunk, &error) == LANA_ERR_JUMP);
    chunk.code[0] = instruction(OP_HALT, 0, 0, 0, 0);
    chunk.version = LABC_VERSION + 1u;
    CHECK(lana_chunk_verify(&chunk, &error) == LANA_ERR_FORMAT);
    chunk.version = 2u;
    CHECK(lana_chunk_verify(&chunk, &error) == LANA_ERR_FORMAT);
    lana_chunk_free(&chunk);
    return 0;
}

static int test_labc_host_calls_are_contiguous(void) {
    CHECK(LANA_HOST_MAP_NEW == LANA_HOST_ASSERT + 1);
    CHECK(LANA_HOST_PATH_RESOLVE == LANA_HOST_ARGS + 28);
    CHECK(LANA_HOST_SAMPLE_RECORD == LANA_HOST_ARGS + 29);
    CHECK(LANA_HOST_DIRECTORY_LIST == LANA_HOST_INFORMATION_INSPECT + 1);
    CHECK(LANA_HOST_WRITE_TEXT_ATOMIC == LANA_HOST_DIRECTORY_LIST + 3);
    return 0;
}

static int test_assembler_serialization_and_loader(void) {
    LanaChunk assembled, loaded, minimal; LanaErrorInfo error = {0}; LanaVM vm;
    FILE *file;
    const char *binary = "runtime-test.labc";
    CHECK(lana_assemble_file(LANA_SOURCE_DIR "/examples/belief.lasm", &assembled, &error) == LANA_OK);
    CHECK(lana_chunk_write_file(&assembled, binary, &error) == LANA_OK);
    CHECK(lana_chunk_read_file(&loaded, binary, &error) == LANA_OK);
    CHECK(loaded.code_count == assembled.code_count);
    lana_vm_init(&vm, &loaded); CHECK(lana_vm_run(&vm) == LANA_OK);
    CHECK(vm.frames[0].registers[3].type == VAL_STATE_DIST);
    CHECK(vm.frames[0].registers[4].type == VAL_NUMBER);
    CHECK(vm.frames[0].registers[4].as.number >= 0.0);
    CHECK(vm.frames[0].registers[4].as.number <= 1.0);
    lana_vm_free(&vm); lana_chunk_free(&loaded);
    assembled.version = 1u;
    CHECK(lana_chunk_write_file(&assembled, binary, &error) == LANA_OK);
    CHECK(lana_chunk_read_file(&loaded, binary, &error) == LANA_OK);
    CHECK(loaded.version == 1u);
    lana_vm_init(&vm, &loaded); CHECK(lana_vm_run(&vm) == LANA_OK);
    CHECK(vm.frames[0].registers[4].type == VAL_NUMBER);
    lana_vm_free(&vm); lana_chunk_free(&loaded);
    file = fopen(binary, "ab");
    CHECK(file != NULL);
    CHECK(fwrite("JUNK", 1u, 4u, file) == 4u);
    CHECK(fclose(file) == 0);
    memset(&error, 0, sizeof(error));
    CHECK(lana_chunk_read_file(&loaded, binary, &error) == LANA_ERR_FORMAT);
    CHECK(lana_chunk_write_file(&assembled, binary, &error) == LANA_OK);
    file = fopen(binary, "r+b");
    CHECK(file != NULL);
    CHECK(fwrite("JUNK", 1u, 4u, file) == 4u);
    CHECK(fclose(file) == 0);
    memset(&error, 0, sizeof(error));
    CHECK(lana_chunk_read_file(&loaded, binary, &error) == LANA_ERR_INCOMPATIBLE_FORMAT);
    CHECK(strcmp(lana_error_name(error.code), "LANA_ERR_INCOMPATIBLE_FORMAT") == 0);
    lana_chunk_init(&minimal);
    CHECK(lana_chunk_emit(&minimal, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    CHECK(lana_chunk_write_file(&minimal, binary, &error) == LANA_OK);
    lana_chunk_free(&minimal);
    file = fopen(binary, "r+b");
    CHECK(file != NULL);
    CHECK(fseek(file, 24L, SEEK_SET) == 0);
    CHECK(fputc(255, file) != EOF);
    CHECK(fclose(file) == 0);
    memset(&error, 0, sizeof(error));
    CHECK(lana_chunk_read_file(&loaded, binary, &error) == LANA_ERR_OPCODE);
    CHECK(loaded.code == NULL && loaded.code_count == 0u);
    lana_chunk_free(&assembled);
    CHECK(remove(binary) == 0);
    return 0;
}

static int test_assembler_indexed_fixups(void) {
    LanaChunk chunk;
    LanaErrorInfo error = {0};
    const char *assembly = "runtime-indexed-fixups.lasm";
    FILE *file = fopen(assembly, "w");
    size_t index;
    CHECK(file != NULL);
    CHECK(fprintf(file, ".version 1\nJUMP main\n") > 0);
    for (index = 0u; index < 256u; ++index) {
        CHECK(fprintf(file, ".function fn_%zu 0 1\n", index) > 0);
        CHECK(fprintf(file, "LOAD_CONST R0 null\nRETURN R0\n") > 0);
    }
    CHECK(fprintf(file, "main:\n") > 0);
    for (index = 0u; index < 1024u; ++index) {
        CHECK(fprintf(file, "JUMP label_%zu\nlabel_%zu:\nNOP\n",
                      index, index) > 0);
    }
    for (index = 0u; index < 256u; ++index)
        CHECK(fprintf(file, "CALL fn_%zu R0 0 R0\n", index) > 0);
    CHECK(fprintf(file, "HALT\n") > 0);
    CHECK(fclose(file) == 0);
    CHECK(lana_assemble_file(assembly, &chunk, &error) == LANA_OK);
    CHECK(chunk.function_count == 256u);
    lana_chunk_free(&chunk);

    file = fopen(assembly, "w");
    CHECK(file != NULL);
    CHECK(fprintf(file, ".version 1\nJUMP missing\nHALT\n") > 0);
    CHECK(fclose(file) == 0);
    memset(&error, 0, sizeof(error));
    CHECK(lana_assemble_file(assembly, &chunk, &error) == LANA_ERR_JUMP);

    file = fopen(assembly, "w");
    CHECK(file != NULL);
    CHECK(fprintf(file, ".version 1\nCALL missing R0 0 R0\nHALT\n") > 0);
    CHECK(fclose(file) == 0);
    memset(&error, 0, sizeof(error));
    CHECK(lana_assemble_file(assembly, &chunk, &error) == LANA_ERR_FORMAT);
    CHECK(remove(assembly) == 0);
    return 0;
}

static int test_named_joint_information(void) {
    LanaChunk chunk;
    LanaVM vm;
    Value rows[9];
    double weights[3] = {0.2, 0.3, 0.5};
    LanaJointState *joint = NULL, *projected = NULL, *conditioned = NULL;
    LanaJointState *renamed = NULL;
    Value evidence = lana_value_bool(true), sampled, resolved;
    size_t index;
    lana_chunk_init(&chunk);
    chunk.version = LABC_VERSION;
    lana_vm_init(&vm, &chunk);
    rows[0] = lana_value_bool(false); rows[1] = lana_value_number(0.0); rows[2] = lana_value_number(10.0);
    rows[3] = lana_value_bool(true);  rows[4] = lana_value_number(1.0); rows[5] = lana_value_number(11.0);
    rows[6] = lana_value_bool(false); rows[7] = lana_value_number(2.0); rows[8] = lana_value_number(12.0);
    CHECK(lana_vm_joint_build_finite(&vm, "z;x;y", rows, weights, 3u, 3u, &joint) == LANA_OK);
    CHECK(joint->count == 3u && joint->row_count == 3u);
    CHECK(joint->kind == LANA_JOINT_FINITE_LAW);
    CHECK(strcmp(joint->names[0], "x") == 0);
    CHECK(strcmp(joint->names[1], "y") == 0);
    CHECK(strcmp(joint->names[2], "z") == 0);
    CHECK(lana_vm_joint_project(&vm, joint, "y;x", &projected) == LANA_OK);
    CHECK(projected->row_count == 3u && projected->count == 2u);
    CHECK(lana_vm_joint_condition(&vm, joint, "z", &evidence, &conditioned) == LANA_OK);
    CHECK(conditioned->row_count == 1u);
    CHECK(vm.observation_count == 0u);
    CHECK(vm.revision == 0u);
    CHECK(lana_vm_joint_observe(&vm, joint, "z", &evidence, &conditioned) == LANA_OK);
    CHECK(vm.observation_count == 1u);
    CHECK(vm.revision == 1u);
    CHECK(lana_vm_joint_resolve(&vm, conditioned, &resolved) == LANA_OK);
    CHECK(resolved.type == VAL_ARRAY && resolved.as.array->count == 3u);
    CHECK(resolved.as.array->items[0].as.number == 1.0);
    CHECK(resolved.as.array->items[1].as.number == 11.0);
    for (index = 0; index < 100u; ++index) {
        CHECK(lana_vm_joint_sample(&vm, joint, &sampled) == LANA_OK);
        CHECK(sampled.type == VAL_ARRAY && sampled.as.array->count == 3u);
        CHECK(sampled.as.array->items[1].as.number ==
              sampled.as.array->items[0].as.number + 10.0);
    }
    CHECK(vm.revision == 1u);
    CHECK(lana_vm_joint_resolve(&vm, joint, &resolved) == LANA_ERR_UNRESOLVED_VALUE);
    CHECK(lana_vm_joint_rename(&vm, joint, "x", "subject", &renamed) == LANA_OK);
    CHECK(strcmp(renamed->names[0], "subject") == 0);
    CHECK(lana_vm_joint_rename(&vm, joint, "x", "y", &renamed) ==
          LANA_ERR_INVALID_DEPENDENCY);
    CHECK(lana_vm_joint_project(&vm, joint, "x;x", &projected) ==
          LANA_ERR_INVALID_DEPENDENCY);
    evidence = lana_value_number(99.0);
    CHECK(lana_vm_joint_condition(&vm, joint, "x", &evidence, &conditioned) ==
          LANA_ERR_INVALID_CONDITIONING);
    CHECK(lana_vm_joint_observe(&vm, joint, "x", &evidence, &conditioned) ==
          LANA_ERR_INVALID_CONDITIONING);
    CHECK(vm.observation_count == 1u);
    CHECK(vm.revision == 1u);
    CHECK(lana_vm_joint_build(&vm, rows, 2u, "independent:x;x", &conditioned) ==
          LANA_ERR_INVALID_DEPENDENCY);
    CHECK(lana_vm_joint_build(&vm, rows, 2u, "correlated:x;y", &conditioned) ==
          LANA_ERR_UNSUPPORTED_OPERATION);
    weights[0] = -0.2;
    CHECK(lana_vm_joint_build_finite(&vm, "z;x;y", rows, weights, 3u, 3u,
                                   &conditioned) == LANA_ERR_INVALID_DISTRIBUTION);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);
    return 0;
}

static int test_guarded_path_limits(void) {
    LanaChunk chunk;
    LanaVM vm;
    uint32_t yes, no, ten, twenty;
    lana_chunk_init(&chunk); chunk.version = LABC_VERSION;
    CHECK(lana_chunk_add_constant(&chunk, lana_value_bool(true), &yes) == LANA_OK);
    CHECK(lana_chunk_add_constant(&chunk, lana_value_bool(false), &no) == LANA_OK);
    CHECK(lana_chunk_add_constant(&chunk, lana_value_number(10.0), &ten) == LANA_OK);
    CHECK(lana_chunk_add_constant(&chunk, lana_value_number(20.0), &twenty) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_LOAD_CONST, 0, 0, 0, yes)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_LOAD_CONST, 1, 0, 0, no)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_ARRAY_NEW, 2, 0, 2, 0)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_POSSIBILITY_BUILD, 2, 3, 0, 0)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_PATH_SPLIT, 3, 0, 0, 7)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_LOAD_CONST, 4, 0, 0, ten)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_JUMP, 0, 0, 0, 8)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_LOAD_CONST, 4, 0, 0, twenty)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_PATH_JOIN, 0, 0, 0, 0)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    CHECK(lana_chunk_verify(&chunk, NULL) == LANA_OK);
    lana_vm_init(&vm, &chunk); vm.path_limit = 1u;
    CHECK(lana_vm_run(&vm) == LANA_ERR_PATH_LIMIT);
    CHECK(vm.frames[0].registers[4].type == VAL_NULL);
    lana_vm_free(&vm);
    lana_vm_init(&vm, &chunk);
    CHECK(lana_vm_run(&vm) == LANA_OK);
    CHECK(vm.frames[0].registers[4].type == VAL_PATH_SET);
    CHECK(vm.frames[0].registers[4].as.paths->count == 2u);
    CHECK(vm.frames[0].registers[4].as.paths->alternatives[0].guard);
    CHECK(!vm.frames[0].registers[4].as.paths->alternatives[1].guard);
    lana_vm_free(&vm); lana_chunk_free(&chunk);
    return 0;
}

static int test_provenance_and_explanation(void) {
    LanaChunk chunk;
    LanaErrorInfo verify_error = {0};
    LanaVM vm;
    Value source = lana_value_number(0.75), rooted, record, explained, repeated;
    Value field;
    uint32_t number, label, quarter;
    lana_chunk_init(&chunk);
    chunk.version = LABC_VERSION;
    lana_vm_init(&vm, &chunk);
    CHECK(lana_vm_provenance_root(&vm, &source, "sensor-a", 12u, false,
                                &rooted) == LANA_OK);
    CHECK(rooted.as.number == source.as.number);
    CHECK(rooted.derivation != NULL);
    CHECK(rooted.derivation->kind == LANA_DERIVATION_EVIDENCE);
    CHECK(rooted.derivation->local_sequence == 1u);
    CHECK(rooted.derivation->revision == 0u);
    CHECK(strcmp(rooted.derivation->label, "sensor-a") == 0);
    CHECK(lana_vm_derivation(&vm, &rooted, &record) == LANA_OK);
    CHECK(record.type == VAL_MAP);
    CHECK(lana_map_get(record.as.map, "operation", &field) == LANA_OK);
    CHECK(field.type == VAL_STRING && strcmp(field.as.string, "evidence") == 0);
    CHECK(lana_map_get(record.as.map, "exactness", &field) == LANA_OK);
    CHECK(strcmp(field.as.string, "exact") == 0);
    CHECK(lana_map_get(record.as.map, "source", &field) == LANA_OK);
    CHECK(field.type == VAL_MAP);
    CHECK(lana_map_get(field.as.map, "function", &field) == LANA_OK);
    CHECK(field.type == VAL_STRING && strcmp(field.as.string, "<main>") == 0);
    CHECK(lana_vm_explain(&vm, &rooted, &explained) == LANA_OK);
    CHECK(lana_vm_explain(&vm, &rooted, &repeated) == LANA_OK);
    CHECK(explained.type == VAL_STRING);
    CHECK(strcmp(explained.as.string, repeated.as.string) == 0);
    CHECK(strstr(explained.as.string, "id=[0,1]") != NULL);
    CHECK(strstr(explained.as.string, "label=sensor-a") != NULL);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);

    lana_chunk_init(&chunk);
    chunk.version = LABC_VERSION;
    CHECK(lana_chunk_add_constant(&chunk, lana_value_number(0.75), &number) == LANA_OK);
    CHECK(lana_chunk_add_constant(&chunk, lana_value_string("sensor-a"), &label) == LANA_OK);
    CHECK(lana_chunk_add_constant(&chunk, lana_value_number(0.25), &quarter) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_LOAD_CONST, 0, 0, 0, number)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_EVIDENCE, 0, 1, label, 0)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_LOAD_CONST, 2, 0, 0, quarter)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_BINARY, 1, 2, 3, LANA_BINARY_ADD)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_DERIVATION, 3, 4, 0, 0)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    if (lana_chunk_verify(&chunk, &verify_error) != LANA_OK) {
        (void)fprintf(stderr, "provenance bytecode verification failed: %s\n",
                      verify_error.message);
        return 1;
    }
    lana_vm_init(&vm, &chunk);
    CHECK(lana_vm_run(&vm) == LANA_OK);
    record = vm.frames[0].registers[4];
    CHECK(record.type == VAL_MAP);
    CHECK(lana_map_get(record.as.map, "inputs", &field) == LANA_OK);
    CHECK(field.type == VAL_ARRAY && field.as.array->count == 1u);
    CHECK(field.as.array->items[0].type == VAL_ARRAY);
    CHECK(field.as.array->items[0].as.array->count == 2u);
    CHECK(field.as.array->items[0].as.array->items[0].type == VAL_NUMBER);
    CHECK(field.as.array->items[0].as.array->items[1].type == VAL_NUMBER);
    CHECK(field.as.array->items[0].as.array->items[1].as.number == 1.0);
    CHECK(lana_map_get(record.as.map, "source", &field) == LANA_OK);
    CHECK(lana_map_get(field.as.map, "function", &field) == LANA_OK);
    CHECK(strcmp(field.as.string, "<main>") == 0);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);
    return 0;
}

static int test_vm_gc_roots_cycles_and_cancellation(void) {
    LanaChunk chunk;
    LanaVM vm;
    LanaArray *array;
    Value native_root;
    Value provenance;
    Value source;
    char *text;
    size_t root_mark;
    size_t live_bytes;

    lana_chunk_init(&chunk);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0u, 0u, 0u, 0u)) == LANA_OK);
    lana_vm_init(&vm, &chunk);

    text = lana_vm_alloc(&vm, 6u);
    CHECK(text != NULL);
    memcpy(text, "alive", 6u);
    vm.frames[0].registers[0] = lana_value_string(text);
    live_bytes = vm.allocated_bytes;
    CHECK(lana_vm_collect(&vm));
    CHECK(vm.allocated_bytes == live_bytes);
    CHECK(strcmp(vm.frames[0].registers[0].as.string, "alive") == 0);
    vm.frames[0].registers[0] = lana_value_null();
    CHECK(lana_vm_collect(&vm));
    CHECK(vm.gc.last_reclaimed_objects == 1u);
    CHECK(vm.allocated_bytes == 0u);

    array = lana_vm_alloc(&vm, sizeof(*array));
    CHECK(array != NULL);
    array->count = array->capacity = 1u;
    array->items = lana_vm_alloc(&vm, sizeof(*array->items));
    CHECK(array->items != NULL);
    array->items[0] = lana_value_array(array);
    vm.frames[0].registers[0] = lana_value_array(array);
    live_bytes = vm.allocated_bytes;
    CHECK(lana_vm_collect(&vm));
    CHECK(vm.allocated_bytes == live_bytes);
    CHECK(vm.frames[0].registers[0].as.array->items[0].as.array == array);
    vm.frames[0].registers[0] = lana_value_null();
    CHECK(lana_vm_collect(&vm));
    CHECK(vm.gc.last_reclaimed_objects == 2u);

    text = lana_vm_alloc(&vm, 7u);
    CHECK(text != NULL);
    memcpy(text, "native", 7u);
    native_root = lana_value_string(text);
    root_mark = lana_vm_root_push(&vm, &native_root);
    CHECK(root_mark != SIZE_MAX);
    CHECK(lana_vm_collect(&vm));
    CHECK(strcmp(native_root.as.string, "native") == 0);
    lana_vm_root_pop(&vm, root_mark);
    CHECK(lana_vm_collect(&vm));
    CHECK(vm.gc.last_reclaimed_objects == 1u);

    source = lana_value_number(0.75);
    CHECK(lana_vm_provenance_root(&vm, &source, "obsolete", 1u, false,
                                  &provenance) == LANA_OK);
    vm.frames[0].registers[0] = provenance;
    CHECK(lana_vm_collect(&vm));
    CHECK(strcmp(vm.frames[0].registers[0].derivation->label, "obsolete") == 0);
    vm.frames[0].registers[0] = lana_value_null();
    CHECK(lana_vm_collect(&vm));
    CHECK(vm.gc.last_reclaimed_objects >= 6u);
    CHECK(vm.allocated_bytes == 0u);

    vm.memory_limit = 64u;
    CHECK(lana_vm_alloc(&vm, 32u) != NULL);
    atomic_store(&vm.cancelled, true);
    CHECK(lana_vm_run(&vm) == LANA_ERR_CANCELLED);
    CHECK(vm.gc.last_reclaimed_objects == 1u);
    CHECK(vm.allocated_bytes == vm.gc.allocated_bytes);

    lana_vm_free(&vm);

    lana_vm_init(&vm, &chunk);
    vm.memory_limit = 64u;
    text = lana_vm_alloc(&vm, 32u);
    CHECK(text != NULL);
    memset(text, 'x', 32u);
    vm.frames[0].registers[0] = lana_value_string(text);
    vm.memory_limit = 16u;
    CHECK(lana_vm_run(&vm) == LANA_ERR_OOM);
    CHECK(vm.gc.last_reclaimed_objects == 0u);
    CHECK(vm.frames[0].registers[0].as.string[0] == 'x');
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);
    return 0;
}

static int test_vm_gc_task_transfer_roots(void) {
    LanaChunk chunk;
    LanaVM vm;
    LanaTask *task;
    uint32_t function;
    uint32_t label;
    uint32_t number;

    lana_chunk_init(&chunk);
    chunk.version = LABC_VERSION;
    CHECK(lana_chunk_add_constant(&chunk, lana_value_number(0.5), &number) == LANA_OK);
    CHECK(lana_chunk_add_constant(&chunk, lana_value_string("worker-value"), &label) == LANA_OK);
    CHECK(lana_chunk_add_function(&chunk, "worker", 4u, 2u, 1u, &function) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_LOAD_CONST, 0u, 0u, 0u, number)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_FORK, 1u, function, 0u, 1u)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_JOIN, 1u, 2u, 0u, 0u)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0u, 0u, 0u, 0u)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_EVIDENCE, 0u, 1u, label, 0u)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_RETURN, 1u, 0u, 0u, 0u)) == LANA_OK);

    lana_vm_init(&vm, &chunk);
    CHECK(lana_vm_set_worker_count(&vm, 1u) == LANA_OK);
    CHECK(lana_vm_run(&vm) == LANA_OK);
    CHECK(vm.frames[0].registers[2].type == VAL_NUMBER);
    CHECK(vm.frames[0].registers[2].derivation != NULL);
    CHECK(strcmp(vm.frames[0].registers[2].derivation->operation, "task_join") == 0);
    CHECK(vm.frames[0].registers[2].derivation->input_count == 1u);
    CHECK(strcmp(vm.frames[0].registers[2].derivation->inputs[0]->label,
                 "worker-value") == 0);
    CHECK(lana_vm_collect(&vm));
    CHECK(strcmp(vm.frames[0].registers[2].derivation->inputs[0]->label,
                 "worker-value") == 0);
    vm.frames[0].registers[2] = lana_value_null();
    CHECK(lana_vm_collect(&vm));
    task = vm.tasks;
    CHECK(task != NULL && task->joined);
    CHECK(task->result.derivation != NULL);
    CHECK(strcmp(task->result.derivation->label, "worker-value") == 0);

    lana_vm_free(&vm);
    lana_chunk_free(&chunk);
    return 0;
}

static int m6_effect_executions;

static LanaError count_m6_effect(LanaVM *vm, const char *kind,
                                 const Value *payload, void *context,
                                 Value *out) {
    (void)vm;
    (void)kind;
    (void)context;
    ++m6_effect_executions;
    *out = *payload;
    return LANA_OK;
}

static int test_m6_reactive_observation_claim_and_effect_receipts(void) {
    LanaChunk chunk;
    LanaVM vm;
    LanaPossibility *possibility;
    Value alternatives[] = {lana_value_number(1.0), lana_value_number(2.0)};
    Value root;
    Value refined;
    Value invalid = lana_value_number(3.0);
    Value evidence = lana_value_number(2.0);
    Value claim;
    Value plan;
    Value result;
    size_t before_failed_observation;

    lana_chunk_init(&chunk);
    chunk.version = LABC_VERSION;
    lana_vm_init(&vm, &chunk);

    CHECK(lana_vm_possibility_build(&vm, alternatives, 2u, &possibility) == LANA_OK);
    refined = lana_value_possibility(possibility);
    CHECK(lana_vm_reactive_root(&vm, &refined, LANA_EXACTNESS_EXACT, &root) == LANA_OK);
    before_failed_observation = vm.revision;
    CHECK(lana_vm_reactive_observe(&vm, &root, &invalid, &refined) ==
          LANA_ERR_INVALID_CONDITIONING);
    CHECK(vm.revision == before_failed_observation);
    CHECK(root.reactive->current->type == VAL_POSSIBILITY);
    CHECK(root.reactive->current->as.possibility->count == 2u);

    CHECK(lana_vm_reactive_observe(&vm, &root, &evidence, &refined) == LANA_OK);
    CHECK(vm.revision == before_failed_observation + 1u);
    CHECK(root.reactive->current->type == VAL_NUMBER);
    CHECK(root.reactive->current->as.number == 2.0);
    CHECK(root.reactive->history_count == 1u);
    CHECK(root.reactive->history[0].revision == before_failed_observation);
    CHECK(root.reactive->history[0].value->type == VAL_POSSIBILITY);

    CHECK(lana_vm_claim(&vm, &root, "observed source", LANA_EXACTNESS_EXACT,
                        0.0, true, &claim) == LANA_OK);
    CHECK(claim.claim != NULL);
    CHECK(strcmp(claim.claim->proposition, "observed source") == 0);
    CHECK(claim.claim->exactness == LANA_EXACTNESS_EXACT);
    CHECK(claim.claim->tolerance == 0.0);
    CHECK(claim.claim->source_valid);
    CHECK(claim.claim->value->reactive == root.reactive);

    CHECK(lana_vm_planned_effect(&vm, "io", &evidence, &plan) == LANA_OK);
    m6_effect_executions = 0;
    CHECK(lana_vm_execute_planned_effect(&vm, &plan, count_m6_effect, NULL,
                                         &result) == LANA_OK);
    CHECK(result.type == VAL_NUMBER && result.as.number == 2.0);
    CHECK(lana_vm_execute_planned_effect(&vm, &plan, count_m6_effect, NULL,
                                         &result) == LANA_OK);
    CHECK(m6_effect_executions == 1);
    CHECK(plan.planned_effect->execution_count == 1u);

    vm.frames[0].registers[0] = root;
    vm.frames[0].registers[1] = claim;
    vm.frames[0].registers[2] = plan;
    CHECK(lana_vm_collect(&vm));
    CHECK(root.reactive->current->as.number == 2.0);
    vm.frames[0].registers[0] = lana_value_null();
    vm.frames[0].registers[1] = lana_value_null();
    vm.frames[0].registers[2] = lana_value_null();
    CHECK(lana_vm_collect(&vm));
    CHECK(vm.gc.last_reclaimed_objects > 0u);

    lana_vm_free(&vm);
    lana_chunk_free(&chunk);
    return 0;
}

int main(void) {
    CHECK(test_state_construction() == 0);
    CHECK(test_state_canonicalization_and_transforms() == 0);
    CHECK(test_distribution_runtime() == 0);
    CHECK(test_basis_measurement_and_estimation() == 0);
    CHECK(test_exact_distribution_rejection_and_limits() == 0);
    CHECK(test_distribution_sharing_metadata_and_budget() == 0);
    CHECK(test_state_and_distribution_equality() == 0);
    CHECK(test_verifier_rejects_bad_opcode_and_jump() == 0);
    CHECK(test_labc_host_calls_are_contiguous() == 0);
    CHECK(test_assembler_serialization_and_loader() == 0);
    CHECK(test_assembler_indexed_fixups() == 0);
    CHECK(test_named_joint_information() == 0);
    CHECK(test_guarded_path_limits() == 0);
    CHECK(test_provenance_and_explanation() == 0);
    CHECK(test_vm_gc_roots_cycles_and_cancellation() == 0);
    CHECK(test_vm_gc_task_transfer_roots() == 0);
    CHECK(test_m6_reactive_observation_claim_and_effect_receipts() == 0);
    (void)printf("C runtime tests passed\n");
    return 0;
}
