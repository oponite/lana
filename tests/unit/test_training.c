#include "tensor.h"
#include "vm.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/* LIP-006: auditable, replayable training primitive. Exercises the sgd, adam,
 * and train host calls directly against the C11 VM, mirroring the differential
 * fixture tests/conformance/differential/hostcalls/lip006_train.lasm and the
 * regression test tests/regression/lip006_train_pass.lana. */

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

/* Run sgd/adam and assert the resulting optimizer's fields before the VM (and
 * its GC heap) is torn down. */
static void check_optimizer(uint32_t host, Value *args, uint32_t argc,
                            const char *name, double learning_rate,
                            double momentum, double beta1, double beta2,
                            double epsilon) {
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
    assert(out.type == VAL_OPTIMIZER);
    assert(strcmp(out.as.optimizer->name, name) == 0);
    assert(out.as.optimizer->learning_rate == learning_rate);
    assert(out.as.optimizer->momentum == momentum);
    assert(out.as.optimizer->beta1 == beta1);
    assert(out.as.optimizer->beta2 == beta2);
    assert(out.as.optimizer->epsilon == epsilon);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);
}

static void test_sgd(void) {
    /* Defaults: learning_rate=0.01, momentum=0.9. */
    check_optimizer(LANA_HOST_SGD, NULL, 0u, "sgd", 0.01, 0.9, 0.0, 0.0, 0.0);

    /* Explicit: sgd(0.1, 0.0). */
    Value args[2] = {lana_value_number(0.1), lana_value_number(0.0)};
    check_optimizer(LANA_HOST_SGD, args, 2u, "sgd", 0.1, 0.0, 0.0, 0.0, 0.0);

    /* Non-positive learning rate is rejected. */
    args[0] = lana_value_number(0.0);
    args[1] = lana_value_number(0.0);
    assert(run_host(LANA_HOST_SGD, args, 2u) == LANA_ERR_INVALID_PARAMETERS);

    /* Momentum must be in [0, 1). */
    args[0] = lana_value_number(0.1);
    args[1] = lana_value_number(1.0);
    assert(run_host(LANA_HOST_SGD, args, 2u) == LANA_ERR_INVALID_PARAMETERS);

    /* Non-finite learning rate is rejected. */
    args[0] = lana_value_number(INFINITY);
    args[1] = lana_value_number(0.0);
    assert(run_host(LANA_HOST_SGD, args, 2u) == LANA_ERR_INVALID_PARAMETERS);
}

static void test_adam(void) {
    /* Defaults: lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8. */
    check_optimizer(LANA_HOST_ADAM, NULL, 0u, "adam", 0.001, 0.0, 0.9, 0.999, 1e-8);

    /* Explicit: adam(0.001, 0.9, 0.999, 1e-8). */
    Value args[4] = {lana_value_number(0.001), lana_value_number(0.9),
                     lana_value_number(0.999), lana_value_number(1e-8)};
    check_optimizer(LANA_HOST_ADAM, args, 4u, "adam", 0.001, 0.0, 0.9, 0.999, 1e-8);

    /* beta1 must be in [0, 1). */
    args[1] = lana_value_number(1.0);
    assert(run_host(LANA_HOST_ADAM, args, 4u) == LANA_ERR_INVALID_PARAMETERS);

    /* epsilon must be positive. */
    args[1] = lana_value_number(0.9);
    args[3] = lana_value_number(0.0);
    assert(run_host(LANA_HOST_ADAM, args, 4u) == LANA_ERR_INVALID_PARAMETERS);
}

/* Build a chunk whose main function invokes `train` with the seven arguments
 * packed in registers 0..6, plus dummy model/loss functions (arity 2). The
 * model/loss bodies are never reached by the type/capability checks under test. */
static void build_train_chunk(LanaChunk *chunk) {
    lana_chunk_init(chunk);
    LanaInstruction invoke = {OP_HOST_CALL, 7u, LANA_HOST_TRAIN, 0u, 7u, 1u};
    LanaInstruction halt = {OP_HALT, 0u, 0u, 0u, 0u, 2u};
    LanaInstruction model_return = {OP_RETURN, 0u, 0u, 0u, 0u, 3u};
    LanaInstruction loss_return = {OP_RETURN, 0u, 0u, 0u, 0u, 4u};
    assert(lana_chunk_emit(chunk, invoke) == LANA_OK);
    assert(lana_chunk_emit(chunk, halt) == LANA_OK);
    size_t model_entry = chunk->code_count;
    assert(lana_chunk_emit(chunk, model_return) == LANA_OK);
    size_t loss_entry = chunk->code_count;
    assert(lana_chunk_emit(chunk, loss_return) == LANA_OK);
    uint32_t main_index, model_index, loss_index;
    assert(lana_chunk_add_function(chunk, "main", 0u, 8u, 0u, &main_index) == LANA_OK);
    assert(lana_chunk_add_function(chunk, "model", (uint32_t)model_entry, 4u, 2u,
                                   &model_index) == LANA_OK);
    assert(lana_chunk_add_function(chunk, "loss", (uint32_t)loss_entry, 4u, 2u,
                                   &loss_index) == LANA_OK);
    assert(main_index == 0u && model_index == 1u && loss_index == 2u);
}

/* Run `train` with the seven arguments in a fresh VM and return the error code.
 * The optimizer argument must remain alive in its own VM for the duration. */
static LanaError run_train(Value *args) {
    LanaChunk chunk;
    LanaVM vm;
    build_train_chunk(&chunk);
    lana_vm_init(&vm, &chunk);
    for (uint32_t i = 0u; i < 7u; ++i) vm.frames[0].registers[i] = args[i];
    LanaError error = lana_vm_run(&vm);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);
    return error;
}

static void test_train_errors(void) {
    LanaChunk opt_chunk;
    LanaVM opt_vm;
    Value optimizer;
    static size_t shape[1] = {1}, strides[1] = {1};
    static double data[1] = {0.0};
    LanaTensor init = {1, shape, strides, false, LANA_TENSOR_F64, (uint8_t *)data, 0, NULL, false};

    /* Create a live optimizer value in its own VM. Its GC heap must stay alive
     * until the train calls below have consumed it, so opt_vm is freed last. */
    lana_chunk_init(&opt_chunk);
    LanaInstruction opt_invoke = {OP_HOST_CALL, 8u, LANA_HOST_SGD, 0u, 0u, 1u};
    LanaInstruction opt_halt = {OP_HALT, 0u, 0u, 0u, 0u, 2u};
    assert(lana_chunk_emit(&opt_chunk, opt_invoke) == LANA_OK);
    assert(lana_chunk_emit(&opt_chunk, opt_halt) == LANA_OK);
    lana_vm_init(&opt_vm, &opt_chunk);
    assert(lana_vm_run(&opt_vm) == LANA_OK);
    optimizer = opt_vm.frames[0].registers[8];

    Value args[7];
    args[0] = lana_value_function(1u); /* model */
    args[1] = lana_value_number(0.0);  /* data (dummy; checked after capability) */
    args[2] = lana_value_function(2u); /* loss */
    args[3] = optimizer;                /* optimizer */
    args[4] = lana_value_tensor(&init);/* initial_params */
    args[5] = lana_value_number(1.0);  /* epochs */
    args[6] = lana_value_number(1.0);  /* batch_size */

    /* Without a `train` capability, execution is denied before access. */
    assert(run_train(args) == LANA_ERR_CAPABILITY);

    /* A non-function model is a type error. */
    args[0] = lana_value_number(0.0);
    assert(run_train(args) == LANA_ERR_TYPE);

    lana_vm_free(&opt_vm);
    lana_chunk_free(&opt_chunk);
}

/* LIP-010: incremental `update` host call. Exercises the validation order
 * (type, steps, capability) without running a full training pass, mirroring
 * the differential fixture tests/conformance/differential/hostcalls/lip010_update.lasm. */
static void test_update_errors(void) {
    LanaTrainingResult result;
    LanaOptimizer optimizer;
    static size_t shape[1] = {1}, strides[1] = {1};
    static double data[1] = {0.0};
    LanaTensor init = {1, shape, strides, false, LANA_TENSOR_F64, (uint8_t *)data, 0, NULL, false};

    /* A minimal training result: only the fields `host_update` inspects before
     * the steps/capability checks are populated. */
    memset(&result, 0, sizeof(result));
    result.params = &init;
    result.model_function = 1u;
    result.loss_function = 2u;
    optimizer.name = "sgd";
    optimizer.learning_rate = 0.1;
    optimizer.momentum = 0.0;
    optimizer.beta1 = 0.0;
    optimizer.beta2 = 0.0;
    optimizer.epsilon = 0.0;
    result.optimizer = &optimizer;

    Value args[3];
    args[0] = lana_value_training_result(&result);
    args[1] = lana_value_number(0.0); /* new_data (dummy) */
    args[2] = lana_value_number(0.0); /* steps = 0 (not a positive integer) */

    /* `steps` must be a positive integer. */
    assert(run_host(LANA_HOST_UPDATE, args, 3u) == LANA_ERR_INVALID_PARAMETERS);

    /* `update` on a non-training-result is a type error. */
    args[0] = lana_value_number(0.0);
    assert(run_host(LANA_HOST_UPDATE, args, 3u) == LANA_ERR_TYPE);

    /* Without a `train` capability, execution is denied before data access. */
    args[0] = lana_value_training_result(&result);
    args[2] = lana_value_number(1.0); /* steps = 1 (valid) */
    assert(run_host(LANA_HOST_UPDATE, args, 3u) == LANA_ERR_CAPABILITY);
}

/* LIP-014: `resume` host call. Exercises the validation order (argc, type,
 * index, capability) without running a full continuation, mirroring the
 * differential fixture tests/conformance/differential/hostcalls/lip014_resume.lasm. */
static LanaError run_resume(Value *args) {
    LanaChunk chunk;
    LanaVM vm;
    lana_chunk_init(&chunk);
    LanaInstruction invoke = {OP_HOST_CALL, 8u, LANA_HOST_RESUME, 0u, 2u, 1u};
    LanaInstruction halt = {OP_HALT, 0u, 0u, 0u, 0u, 2u};
    LanaInstruction model_return = {OP_RETURN, 0u, 0u, 0u, 0u, 3u};
    LanaInstruction loss_return = {OP_RETURN, 0u, 0u, 0u, 0u, 4u};
    assert(lana_chunk_emit(&chunk, invoke) == LANA_OK);
    assert(lana_chunk_emit(&chunk, halt) == LANA_OK);
    size_t model_entry = chunk.code_count;
    assert(lana_chunk_emit(&chunk, model_return) == LANA_OK);
    size_t loss_entry = chunk.code_count;
    assert(lana_chunk_emit(&chunk, loss_return) == LANA_OK);
    uint32_t main_index, model_index, loss_index;
    assert(lana_chunk_add_function(&chunk, "main", 0u, 8u, 0u, &main_index) == LANA_OK);
    assert(lana_chunk_add_function(&chunk, "model", (uint32_t)model_entry, 4u, 2u,
                                   &model_index) == LANA_OK);
    assert(lana_chunk_add_function(&chunk, "loss", (uint32_t)loss_entry, 4u, 2u,
                                   &loss_index) == LANA_OK);
    assert(main_index == 0u && model_index == 1u && loss_index == 2u);
    lana_vm_init(&vm, &chunk);
    for (uint32_t i = 0u; i < 2u; ++i) vm.frames[0].registers[i] = args[i];
    LanaError error = lana_vm_run(&vm);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);
    return error;
}

static void test_resume_errors(void) {
    LanaTrainingResult result;
    LanaOptimizer optimizer;
    LanaArray steps;
    static size_t shape[1] = {1}, strides[1] = {1};
    static double data[1] = {0.0};
    LanaTensor init = {1, shape, strides, false, LANA_TENSOR_F64, (uint8_t *)data, 0, NULL, false};

    /* A minimal training result: only the fields `host_resume` inspects before
     * the capability check are populated. `steps` is empty (count 0) because the
     * step-index check runs after the capability check. */
    memset(&result, 0, sizeof(result));
    memset(&steps, 0, sizeof(steps));
    result.params = &init;
    result.steps = &steps;
    result.model_function = 1u;
    result.loss_function = 2u;
    optimizer.name = "sgd";
    optimizer.learning_rate = 0.1;
    optimizer.momentum = 0.0;
    optimizer.beta1 = 0.0;
    optimizer.beta2 = 0.0;
    optimizer.epsilon = 0.0;
    result.optimizer = &optimizer;

    Value args[2];
    args[0] = lana_value_training_result(&result);
    args[1] = lana_value_number(0.0);

    /* `resume` takes exactly two arguments. */
    assert(run_host(LANA_HOST_RESUME, args, 1u) == LANA_ERR_TYPE);

    /* `resume` on a non-training-result is a type error. */
    args[0] = lana_value_number(0.0);
    assert(run_host(LANA_HOST_RESUME, args, 2u) == LANA_ERR_TYPE);

    /* The step index must be a number. */
    args[0] = lana_value_training_result(&result);
    args[1] = lana_value_string("x");
    assert(run_host(LANA_HOST_RESUME, args, 2u) == LANA_ERR_INVALID_PARAMETERS);

    /* The step index must be a nonnegative integer. */
    args[1] = lana_value_number(1.5);
    assert(run_host(LANA_HOST_RESUME, args, 2u) == LANA_ERR_INVALID_PARAMETERS);
    args[1] = lana_value_number(-1.0);
    assert(run_host(LANA_HOST_RESUME, args, 2u) == LANA_ERR_INVALID_PARAMETERS);

    /* Without a `train` capability, execution is denied before data access. */
    args[1] = lana_value_number(0.0);
    assert(run_resume(args) == LANA_ERR_CAPABILITY);
}

int main(void) {
    test_sgd();
    test_adam();
    test_train_errors();
    test_update_errors();
    test_resume_errors();
    (void)printf("training host calls ok\n");
    return 0;
}
