#include "ss/assembler.h"
#include "ss/asm_probe.h"
#include "ss/vm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) { (void)fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); return 1; } } while (0)

static SSInstruction instruction(OpCode opcode, uint32_t a, uint32_t b,
                                 uint32_t c, uint32_t imm) {
    SSInstruction value = {(uint8_t)opcode, a, b, c, imm, 1u};
    return value;
}

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
    SSChunk chunk; SSErrorInfo error = {0};
    ss_chunk_init(&chunk);
    CHECK(ss_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == SS_OK);
    chunk.code[0].opcode = 255u;
    CHECK(ss_chunk_verify(&chunk, &error) == SS_ERR_OPCODE);
    chunk.code[0] = instruction(OP_JUMP, 0, 0, 0, 99u);
    CHECK(ss_chunk_verify(&chunk, &error) == SS_ERR_JUMP);
    ss_chunk_free(&chunk);
    return 0;
}

static int test_assembler_serialization_and_loader(void) {
    SSChunk assembled, loaded; SSErrorInfo error = {0}; VM vm;
    const char *binary = "runtime-test.ssb";
    CHECK(ss_assemble_file(SS_SOURCE_DIR "/examples/belief.ssa", &assembled, &error) == SS_OK);
    CHECK(ss_chunk_write_file(&assembled, binary, &error) == SS_OK);
    CHECK(ss_chunk_read_file(&loaded, binary, &error) == SS_OK);
    CHECK(loaded.code_count == assembled.code_count);
    ss_vm_init(&vm, &loaded); CHECK(ss_vm_run(&vm) == SS_OK);
    CHECK(fabs(vm.frames[0].registers[0].as.state.state.p - 0.76) < SS_STATE_EPSILON);
    CHECK(vm.frames[0].registers[2].type == VAL_NUMBER);
    CHECK(fabs(vm.frames[0].registers[2].as.number - 0.76) < SS_STATE_EPSILON);
    ss_vm_free(&vm); ss_chunk_free(&loaded); ss_chunk_free(&assembled);
    CHECK(remove(binary) == 0);
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
    CHECK(test_vm_native_state_program() == 0);
    CHECK(test_composition() == 0);
    CHECK(test_multi_state_mean_apply() == 0);
    CHECK(test_transform_instruction() == 0);
    CHECK(test_measurement_indexes_and_history() == 0);
    CHECK(test_verifier_rejects_bad_opcode_and_jump() == 0);
    CHECK(test_assembler_serialization_and_loader() == 0);
    CHECK(test_assembly_probe() == 0);
    (void)printf("C runtime tests passed\n");
    return 0;
}
