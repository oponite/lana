#include "state.h"
#include "vm.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) { (void)fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); return 1; } } while (0)

static int states_equal(const LanaState *left, const LanaState *right) {
    return fabs(left->p - right->p) < LANA_STATE_EPSILON &&
           fabs(left->d_re - right->d_re) < LANA_STATE_EPSILON &&
           fabs(left->d_im - right->d_im) < LANA_STATE_EPSILON;
}

static int test_state_mix_math(void) {
    LanaState a, b, mixed, expected;
    CHECK(lana_state_make_complex(0.4, 0.2, 0.0, &a) == LANA_OK);
    CHECK(lana_state_make_complex(0.8, -0.1, 0.0, &b) == LANA_OK);

    /* Weight symmetry: mix(a,b,w) == mix(b,a,1-w). */
    CHECK(lana_state_mix(&a, &b, 0.5, &mixed) == LANA_OK);
    CHECK(lana_state_mix(&b, &a, 0.5, &expected) == LANA_OK);
    CHECK(states_equal(&mixed, &expected));

    /* Idempotence: mix(a,a,w) == a. */
    CHECK(lana_state_mix(&a, &a, 0.3, &mixed) == LANA_OK);
    CHECK(states_equal(&mixed, &a));

    /* Boundary weights: w=0 returns b; w=1 returns a. */
    CHECK(lana_state_mix(&a, &b, 0.0, &mixed) == LANA_OK);
    CHECK(states_equal(&mixed, &b));
    CHECK(lana_state_mix(&a, &b, 1.0, &mixed) == LANA_OK);
    CHECK(states_equal(&mixed, &a));

    /* Invalid weights. */
    CHECK(lana_state_mix(&a, &b, -0.1, &mixed) == LANA_ERR_INVALID_PARAMETERS);
    CHECK(lana_state_mix(&a, &b, 1.1, &mixed) == LANA_ERR_INVALID_PARAMETERS);
    CHECK(lana_state_mix(&a, &b, NAN, &mixed) == LANA_ERR_INVALID_PARAMETERS);
    CHECK(lana_state_mix(&a, &b, INFINITY, &mixed) == LANA_ERR_INVALID_PARAMETERS);
    CHECK(lana_state_mix(&a, &b, -INFINITY, &mixed) == LANA_ERR_INVALID_PARAMETERS);

    /* Boundary probabilities force d_C = 0. */
    CHECK(lana_state_make_complex(1.0, 0.0, 0.0, &a) == LANA_OK);
    CHECK(lana_state_make_complex(0.0, 0.0, 0.0, &b) == LANA_OK);
    CHECK(lana_state_mix(&a, &b, 0.5, &mixed) == LANA_OK);
    CHECK(mixed.p == 0.5 && mixed.d_re == 0.0 && mixed.d_im == 0.0);
    CHECK(lana_state_mix(&a, &b, 1.0, &mixed) == LANA_OK);
    CHECK(mixed.p == 1.0 && mixed.d_re == 0.0 && mixed.d_im == 0.0);
    CHECK(lana_state_mix(&a, &b, 0.0, &mixed) == LANA_OK);
    CHECK(mixed.p == 0.0 && mixed.d_re == 0.0 && mixed.d_im == 0.0);

    /* Invalid state inputs. */
    CHECK(lana_state_mix(NULL, &b, 0.5, &mixed) == LANA_ERR_INVALID_STATE);
    CHECK(lana_state_mix(&a, NULL, 0.5, &mixed) == LANA_ERR_INVALID_STATE);
    CHECK(lana_state_mix(&a, &b, 0.5, NULL) == LANA_ERR_INVALID_STATE);
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

static int test_vm_mix_opcode(void) {
    LanaChunk chunk;
    LanaVM vm;
    uint32_t p_a, d_a, p_b, d_b, w, zero;
    LanaStateValue state_a, state_b;
    LanaStateDist *dirac;
    Value state_dist_value;
    LanaErrorInfo error = {0};

    lana_chunk_init(&chunk);
    CHECK(add_number(&chunk, 0.4, &p_a) == 0);
    CHECK(add_number(&chunk, 0.2, &d_a) == 0);
    CHECK(add_number(&chunk, 0.8, &p_b) == 0);
    CHECK(add_number(&chunk, -0.1, &d_b) == 0);
    CHECK(add_number(&chunk, 0.5, &w) == 0);
    CHECK(add_number(&chunk, 0.0, &zero) == 0);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_STATE_NEW, 0, p_a, d_a, zero)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_STATE_NEW, 1, p_b, d_b, zero)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_LOAD_CONST, 3, 0, 0, w)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_MIX, 2, 0, 1, 3)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    CHECK(lana_vm_run(&vm) == LANA_OK);
    CHECK(vm.frames[0].registers[2].type == VAL_STATE);
    CHECK(fabs(vm.frames[0].registers[2].as.state.state.p - 0.6) < LANA_STATE_EPSILON);
    CHECK(fabs(vm.frames[0].registers[2].as.state.state.d_re - 0.05918) < 1e-4);
    CHECK(fabs(vm.frames[0].registers[2].as.state.state.d_im) < LANA_STATE_EPSILON);
    /* Derivation record is written. */
    CHECK(vm.frames[0].registers[2].derivation != NULL);
    CHECK(vm.frames[0].registers[2].derivation->kind == LANA_DERIVATION_OPERATION);
    CHECK(strcmp(vm.frames[0].registers[2].derivation->operation, "mix") == 0);
    CHECK(vm.frames[0].registers[2].derivation->exactness == LANA_EXACTNESS_EXACT);
    CHECK(vm.frames[0].registers[2].derivation->details != NULL);
    CHECK(strstr(vm.frames[0].registers[2].derivation->details, "w=") != NULL);
    /* Result metadata is empty without a declared policy. */
    CHECK(!vm.frames[0].registers[2].as.state.indexes.has_timestamp);
    CHECK(!vm.frames[0].registers[2].as.state.indexes.has_source);
    CHECK(!vm.frames[0].registers[2].as.state.indexes.has_weight);
    CHECK(!vm.frames[0].registers[2].as.state.indexes.has_confidence);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);

    /* STATE_DIST input returns LANA_ERR_UNSUPPORTED_OPERATION. */
    lana_chunk_init(&chunk);
    CHECK(add_number(&chunk, 0.5, &w) == 0);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_MIX, 2, 0, 1, 3)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    CHECK(lana_state_make_complex(0.5, 0.0, 0.0, &state_a.state) == LANA_OK);
    CHECK(lana_vm_state_dist_dirac(&vm, &state_a, &dirac) == LANA_OK);
    state_dist_value = lana_value_state_dist(dirac);
    vm.frames[0].registers[0] = state_dist_value;
    vm.frames[0].registers[1] = lana_value_state(state_a.state);
    vm.frames[0].registers[3] = lana_value_number(0.5);
    CHECK(lana_vm_run(&vm) == LANA_ERR_UNSUPPORTED_OPERATION);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);

    /* Non-state input returns LANA_ERR_TYPE. */
    lana_chunk_init(&chunk);
    CHECK(add_number(&chunk, 0.5, &w) == 0);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_MIX, 2, 0, 1, 3)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    vm.frames[0].registers[0] = lana_value_number(1.0);
    vm.frames[0].registers[1] = lana_value_number(2.0);
    vm.frames[0].registers[3] = lana_value_number(0.5);
    CHECK(lana_vm_run(&vm) == LANA_ERR_TYPE);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);

    /* Non-number weight returns LANA_ERR_TYPE. */
    lana_chunk_init(&chunk);
    CHECK(add_number(&chunk, 0.5, &w) == 0);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_MIX, 2, 0, 1, 3)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    CHECK(lana_state_make_complex(0.5, 0.0, 0.0, &state_a.state) == LANA_OK);
    CHECK(lana_state_make_complex(0.5, 0.0, 0.0, &state_b.state) == LANA_OK);
    vm.frames[0].registers[0] = lana_value_state(state_a.state);
    vm.frames[0].registers[1] = lana_value_state(state_b.state);
    vm.frames[0].registers[3] = lana_value_string("half");
    CHECK(lana_vm_run(&vm) == LANA_ERR_TYPE);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);

    /* Invalid weight returns LANA_ERR_INVALID_PARAMETERS. */
    lana_chunk_init(&chunk);
    CHECK(add_number(&chunk, 1.5, &w) == 0);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_MIX, 2, 0, 1, 3)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    CHECK(lana_state_make_complex(0.5, 0.0, 0.0, &state_a.state) == LANA_OK);
    CHECK(lana_state_make_complex(0.5, 0.0, 0.0, &state_b.state) == LANA_OK);
    vm.frames[0].registers[0] = lana_value_state(state_a.state);
    vm.frames[0].registers[1] = lana_value_state(state_b.state);
    vm.frames[0].registers[3] = lana_value_number(1.5);
    CHECK(lana_vm_run(&vm) == LANA_ERR_INVALID_PARAMETERS);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);

    /* Verifier accepts MIX with four register operands. */
    lana_chunk_init(&chunk);
    CHECK(add_number(&chunk, 0.5, &w) == 0);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_MIX, 2, 0, 1, 3)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    CHECK(lana_chunk_verify(&chunk, &error) == LANA_OK);
    lana_chunk_free(&chunk);
    return 0;
}

static int test_vm_mix_evidence_combination(void) {
    LanaChunk chunk;
    LanaVM vm;
    uint32_t w;
    LanaStateValue state_a, state_b;
    Value evidence_value, assumption_value;
    LanaEvidenceStatus status;

    /* Status classifier: a NULL derivation (bare literal) is exact. */
    CHECK(lana_derivation_status(NULL) == LANA_EVIDENCE_EXACT);
    CHECK(strcmp(lana_evidence_status_name(LANA_EVIDENCE_MODELED), "modeled") == 0);
    CHECK(strcmp(lana_evidence_status_name(LANA_EVIDENCE_OBSERVED), "observed") == 0);

    lana_chunk_init(&chunk);
    CHECK(add_number(&chunk, 0.5, &w) == 0);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_MIX, 2, 0, 1, 3)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    lana_vm_init(&vm, &chunk);

    CHECK(lana_state_make_complex(0.4, 0.0, 0.0, &state_a.state) == LANA_OK);
    CHECK(lana_state_make_complex(0.8, 0.0, 0.0, &state_b.state) == LANA_OK);

    /* exact (evidence) ⊕ modeled (assumption) = modeled. */
    CHECK(lana_vm_provenance_root(&vm, &(Value){.type = VAL_STATE, .as.state = state_a},
                                  "obs", 1u, false, &evidence_value) == LANA_OK);
    CHECK(lana_vm_provenance_root(&vm, &(Value){.type = VAL_STATE, .as.state = state_b},
                                  "model", 1u, true, &assumption_value) == LANA_OK);
    CHECK(lana_derivation_status(evidence_value.derivation) == LANA_EVIDENCE_EXACT);
    CHECK(lana_derivation_status(assumption_value.derivation) == LANA_EVIDENCE_MODELED);

    vm.frames[0].registers[0] = evidence_value;
    vm.frames[0].registers[1] = assumption_value;
    vm.frames[0].registers[3] = lana_value_number(0.5);
    CHECK(lana_vm_run(&vm) == LANA_OK);
    CHECK(vm.frames[0].registers[2].derivation != NULL);
    status = lana_derivation_status(vm.frames[0].registers[2].derivation);
    CHECK(status == LANA_EVIDENCE_MODELED);

    lana_vm_free(&vm);
    lana_chunk_free(&chunk);
    return 0;
}

int main(void) {
    CHECK(test_state_mix_math() == 0);
    CHECK(test_vm_mix_opcode() == 0);
    CHECK(test_vm_mix_evidence_combination() == 0);
    (void)printf("MIX tests passed\n");
    return 0;
}
