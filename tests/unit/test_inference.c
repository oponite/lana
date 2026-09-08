#include "tensor.h"
#include "vm.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/* LIP-009: Bayesian inference as a first-class training mode. Exercises the
 * mcmc, vi, smc, and infer host calls directly against the C11 VM, mirroring
 * the differential fixture tests/conformance/differential/hostcalls/lip009_infer.lasm
 * and the regression test tests/regression/lip009_infer_pass.lana. */

/* Run a single host call with `argc` arguments packed in registers 0..argc-1
 * and return the error code. The VM is freed before returning, so any Value
 * pointers it produced are not valid after this call. */
static LanaError run_host(uint32_t host, Value *args, uint32_t argc) {
    LanaChunk chunk;
    LanaVM vm;
    lana_chunk_init(&chunk);
    LanaInstruction invoke = {OP_HOST_CALL, 8u, host, 0u, argc, 1u};
    LanaInstruction halt = {OP_HALT, 0u, 0u, 0u, 0u, 2u};
    assert(lana_chunk_emit(&chunk, invoke) == LANA_OK);
    assert(lana_chunk_emit(&chunk, halt) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    for (uint32_t i = 0u; i < argc; ++i) vm.frames[0].registers[i] = args[i];
    LanaError error = lana_vm_run(&vm);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);
    return error;
}

/* Run mcmc/vi/smc and assert the resulting algorithm's fields before the VM
 * (and its GC heap) is torn down. */
static void check_algorithm(uint32_t host, Value *args, uint32_t argc,
                            const char *name, const char *family,
                            double samples, double burn_in, double iterations) {
    LanaChunk chunk;
    LanaVM vm;
    lana_chunk_init(&chunk);
    LanaInstruction invoke = {OP_HOST_CALL, 8u, host, 0u, argc, 1u};
    LanaInstruction halt = {OP_HALT, 0u, 0u, 0u, 0u, 2u};
    assert(lana_chunk_emit(&chunk, invoke) == LANA_OK);
    assert(lana_chunk_emit(&chunk, halt) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    for (uint32_t i = 0u; i < argc; ++i) vm.frames[0].registers[i] = args[i];
    assert(lana_vm_run(&vm) == LANA_OK);
    Value out = vm.frames[0].registers[8];
    assert(out.type == VAL_INFERENCE_ALGORITHM);
    assert(strcmp(out.as.inference_algorithm->name, name) == 0);
    if (family == NULL) {
        assert(out.as.inference_algorithm->family == NULL);
    } else {
        assert(strcmp(out.as.inference_algorithm->family, family) == 0);
    }
    assert(out.as.inference_algorithm->samples == samples);
    assert(out.as.inference_algorithm->burn_in == burn_in);
    assert(out.as.inference_algorithm->iterations == iterations);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);
}

static void test_mcmc(void) {
    /* Defaults: samples=10000, burn_in=1000. */
    check_algorithm(LANA_HOST_MCMC, NULL, 0u, "mcmc", NULL, 10000.0, 1000.0, 0.0);

    /* Explicit: mcmc(100, 10). */
    Value args[2] = {lana_value_number(100.0), lana_value_number(10.0)};
    check_algorithm(LANA_HOST_MCMC, args, 2u, "mcmc", NULL, 100.0, 10.0, 0.0);

    /* Non-positive samples is rejected. */
    args[0] = lana_value_number(0.0);
    args[1] = lana_value_number(0.0);
    assert(run_host(LANA_HOST_MCMC, args, 2u) == LANA_ERR_INVALID_PARAMETERS);

    /* burn_in >= samples is rejected. */
    args[0] = lana_value_number(10.0);
    args[1] = lana_value_number(10.0);
    assert(run_host(LANA_HOST_MCMC, args, 2u) == LANA_ERR_INVALID_PARAMETERS);

    /* Non-finite samples is rejected. */
    args[0] = lana_value_number(INFINITY);
    args[1] = lana_value_number(0.0);
    assert(run_host(LANA_HOST_MCMC, args, 2u) == LANA_ERR_INVALID_PARAMETERS);
}

static void test_vi(void) {
    /* Defaults: family="gaussian", iterations=1000. */
    check_algorithm(LANA_HOST_VI, NULL, 0u, "vi", "gaussian", 0.0, 0.0, 1000.0);

    /* Explicit: vi("mean_field", 50). */
    Value args[2] = {lana_value_string("mean_field"), lana_value_number(50.0)};
    check_algorithm(LANA_HOST_VI, args, 2u, "vi", "mean_field", 0.0, 0.0, 50.0);

    /* Unknown family is rejected. */
    args[0] = lana_value_string("bogus");
    args[1] = lana_value_number(10.0);
    assert(run_host(LANA_HOST_VI, args, 2u) == LANA_ERR_INVALID_PARAMETERS);

    /* Non-positive iterations is rejected. */
    args[0] = lana_value_string("gaussian");
    args[1] = lana_value_number(0.0);
    assert(run_host(LANA_HOST_VI, args, 2u) == LANA_ERR_INVALID_PARAMETERS);
}

static void test_smc(void) {
    /* Defaults: particles=1000. */
    check_algorithm(LANA_HOST_SMC, NULL, 0u, "smc", NULL, 1000.0, 0.0, 0.0);

    /* Explicit: smc(100). */
    Value args[1] = {lana_value_number(100.0)};
    check_algorithm(LANA_HOST_SMC, args, 1u, "smc", NULL, 100.0, 0.0, 0.0);

    /* Non-positive particles is rejected. */
    args[0] = lana_value_number(0.0);
    assert(run_host(LANA_HOST_SMC, args, 1u) == LANA_ERR_INVALID_PARAMETERS);
}

/* Build a chunk whose main function invokes `infer` with the four arguments
 * packed in registers 0..3, plus a dummy model function (arity 1). The model
 * body is never reached by the type/capability checks under test. */
static void build_infer_chunk(LanaChunk *chunk) {
    lana_chunk_init(chunk);
    LanaInstruction invoke = {OP_HOST_CALL, 8u, LANA_HOST_INFER, 0u, 4u, 1u};
    LanaInstruction halt = {OP_HALT, 0u, 0u, 0u, 0u, 2u};
    LanaInstruction model_return = {OP_RETURN, 0u, 0u, 0u, 0u, 3u};
    assert(lana_chunk_emit(chunk, invoke) == LANA_OK);
    assert(lana_chunk_emit(chunk, halt) == LANA_OK);
    size_t model_entry = chunk->code_count;
    assert(lana_chunk_emit(chunk, model_return) == LANA_OK);
    uint32_t main_index, model_index;
    assert(lana_chunk_add_function(chunk, "main", 0u, 8u, 0u, &main_index) == LANA_OK);
    assert(lana_chunk_add_function(chunk, "model", (uint32_t)model_entry, 2u, 1u,
                                   &model_index) == LANA_OK);
    assert(main_index == 0u && model_index == 1u);
}

/* Run `infer` with the four arguments in a fresh VM and return the error code.
 * The algorithm argument must remain alive in its own VM for the duration. */
static LanaError run_infer(Value *args) {
    LanaChunk chunk;
    LanaVM vm;
    build_infer_chunk(&chunk);
    lana_vm_init(&vm, &chunk);
    for (uint32_t i = 0u; i < 4u; ++i) vm.frames[0].registers[i] = args[i];
    LanaError error = lana_vm_run(&vm);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);
    return error;
}

static void test_infer_errors(void) {
    LanaChunk alg_chunk;
    LanaVM alg_vm;
    Value algorithm;
    static size_t shape[1] = {1}, strides[1] = {1};
    /* LIP-027: tensor data is a byte buffer; store the f64 bit pattern of 1.0. */
    static union { double d; uint8_t b[8]; } prior_data = { .d = 1.0 };
    LanaTensor prior = {1, shape, strides, false, LANA_TENSOR_F64, (uint8_t *)prior_data.b, 0, NULL, false, LANA_TENSOR_CPU, NULL, NULL};

    /* Create a live algorithm value in its own VM. Its GC heap must stay alive
     * until the infer calls below have consumed it, so alg_vm is freed last. */
    lana_chunk_init(&alg_chunk);
    LanaInstruction alg_invoke = {OP_HOST_CALL, 8u, LANA_HOST_MCMC, 0u, 0u, 1u};
    LanaInstruction alg_halt = {OP_HALT, 0u, 0u, 0u, 0u, 2u};
    assert(lana_chunk_emit(&alg_chunk, alg_invoke) == LANA_OK);
    assert(lana_chunk_emit(&alg_chunk, alg_halt) == LANA_OK);
    lana_vm_init(&alg_vm, &alg_chunk);
    assert(lana_vm_run(&alg_vm) == LANA_OK);
    algorithm = alg_vm.frames[0].registers[8];

    Value args[4];
    args[0] = lana_value_tensor(&prior); /* prior */
    args[1] = lana_value_function(1u);   /* model */
    args[2] = lana_value_tensor(&prior); /* data */
    args[3] = algorithm;                 /* algorithm */

    /* Without an `infer` capability, execution is denied before access. */
    assert(run_infer(args) == LANA_ERR_CAPABILITY);

    /* A non-function model is a type error. */
    args[1] = lana_value_number(0.0);
    assert(run_infer(args) == LANA_ERR_TYPE);

    /* A non-tensor prior is a type error. */
    args[1] = lana_value_function(1u);
    args[0] = lana_value_number(0.0);
    assert(run_infer(args) == LANA_ERR_TYPE);

    /* A non-tensor data is a type error. */
    args[0] = lana_value_tensor(&prior);
    args[2] = lana_value_number(0.0);
    assert(run_infer(args) == LANA_ERR_TYPE);

    /* A non-algorithm fourth argument is a type error. */
    args[2] = lana_value_tensor(&prior);
    args[3] = lana_value_number(0.0);
    assert(run_infer(args) == LANA_ERR_TYPE);

    lana_vm_free(&alg_vm);
    lana_chunk_free(&alg_chunk);
}

int main(void) {
    test_mcmc();
    test_vi();
    test_smc();
    test_infer_errors();
    (void)printf("inference host calls ok\n");
    return 0;
}
