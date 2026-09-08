#include "tensor.h"
#include "vm.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/* LIP-005: linear algebra on STATEs. Exercises the density_operator, povm,
 * channel, observable, tensor_product, partial_trace, measure_with, apply_to,
 * expect, mix, trace_distance, is_separable, and to_state host calls directly
 * against the C11 VM, mirroring the differential fixtures in
 * tests/conformance/differential/hostcalls/lip005_*.lasm. */

/* Compare a result tensor's logical elements against a row-major expected
 * buffer, walking the logical layout (views share their base buffer). */
static int tensor_elements_match(const LanaTensor *actual, const LanaTensor *expected) {
    size_t total = 1u, mult = expected->is_complex ? 2u : 1u;
    if (actual->ndim != expected->ndim || actual->is_complex != expected->is_complex)
        return 0;
    for (size_t i = 0u; i < actual->ndim; ++i) {
        if (actual->shape[i] != expected->shape[i]) return 0;
        total *= actual->shape[i];
    }
    for (size_t lin = 0u; lin < total; ++lin) {
        size_t rem = lin, index = actual->offset;
        for (size_t d = actual->ndim; d-- > 0u;) {
            index += (rem % actual->shape[d]) * actual->strides[d];
            rem /= actual->shape[d];
        }
        for (size_t c = 0u; c < mult; ++c)
            if (actual->data[index * mult + c] != expected->data[lin * mult + c])
                return 0;
    }
    return 1;
}

/* Invoke a host call with `argc` arguments and assert the error code and, when
 * `result` is non-NULL, that the result is a tensor (or a thin LIP-005 wrapper
 * over one) whose elements match. */
static void call_checked(uint32_t host, Value *args, uint32_t argc, LanaError expected,
                         const LanaTensor *result) {
    LanaChunk chunk;
    LanaVM vm;
    lana_chunk_init(&chunk);
    LanaInstruction invoke = {OP_HOST_CALL, 8u, host, 0u, argc, 1u};
    LanaInstruction halt = {OP_HALT, 0u, 0u, 0u, 0u, 2u};
    assert(lana_chunk_emit(&chunk, invoke) == LANA_OK);
    assert(lana_chunk_emit(&chunk, halt) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    for (uint32_t i = 0u; i < argc; ++i) vm.frames[0].registers[i] = args[i];
    LanaError actual = lana_vm_run(&vm);
    if (actual != expected)
        (void)fprintf(stderr, "host %u: expected %d, got %d\n", host, expected, actual);
    assert(actual == expected);
    if (expected != LANA_OK) assert(vm.frames[0].registers[8].type == VAL_NULL);
    if (result != NULL) {
        Value value = vm.frames[0].registers[8];
        assert(value.type == VAL_TENSOR || value.type == VAL_NQUBIT_STATE ||
               value.type == VAL_POVM || value.type == VAL_CHANNEL ||
               value.type == VAL_OBSERVABLE);
        assert(tensor_elements_match(value.as.tensor, result));
    }
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);
}

/* Like `call_checked`, but asserts a successful scalar `number` result. */
static void call_number(uint32_t host, Value *args, uint32_t argc, double expected) {
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
    Value value = vm.frames[0].registers[8];
    assert(value.type == VAL_NUMBER && value.as.number == expected);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);
}

static void call(uint32_t host, Value *args, uint32_t argc, LanaError expected) {
    call_checked(host, args, argc, expected, NULL);
}

/* A 2x2 complex tensor laid out as interleaved [re, im] row-major data. The
 * shape/strides are static so the returned struct's pointers stay valid after
 * the function returns (all 2x2 tensors share the same layout). */
static LanaTensor complex2(double *data) {
    static size_t shape[2] = {2, 2}, strides[2] = {2, 1};
    LanaTensor t = {2, shape, strides, true, LANA_TENSOR_COMPLEX, (uint8_t *)data, 0, NULL, false, LANA_TENSOR_CPU, NULL, NULL};
    return t;
}

static void density_operator(void) {
    /* From an N=1 STATE: rho = [[p, c], [c*, 1-p]]. */
    LanaState state;
    assert(lana_state_make_complex(0.4, 0.2, 0.0, &state) == LANA_OK);
    Value args[] = {lana_value_state(state)};
    double c = 0.2 * sqrt(0.4 * 0.6);
    double expected_data[] = {0.4, 0, c, 0, c, 0, 0.6, 0};
    LanaTensor expected = complex2(expected_data);
    call_checked(LANA_HOST_DENSITY_OPERATOR, args, 1, LANA_OK, &expected);

    /* From a tensor: the maximally mixed state I/2. */
    double mixed_data[] = {0.5, 0, 0, 0, 0, 0, 0.5, 0};
    LanaTensor mixed = complex2(mixed_data);
    args[0] = lana_value_tensor(&mixed);
    call_checked(LANA_HOST_DENSITY_OPERATOR, args, 1, LANA_OK, &mixed);

    /* Non-Hermitian is rejected. */
    double non_hermitian_data[] = {1, 0, 1, 0, 0, 0, 1, 0};
    LanaTensor non_hermitian = complex2(non_hermitian_data);
    args[0] = lana_value_tensor(&non_hermitian);
    call(LANA_HOST_DENSITY_OPERATOR, args, 1, LANA_ERR_INVALID_STATE);

    /* Non-unit trace is rejected. */
    double non_trace_data[] = {1, 0, 0, 0, 0, 0, 1, 0};
    LanaTensor non_trace = complex2(non_trace_data);
    args[0] = lana_value_tensor(&non_trace);
    call(LANA_HOST_DENSITY_OPERATOR, args, 1, LANA_ERR_INVALID_STATE);

    /* A non-tensor, non-state argument is a type error. */
    args[0] = lana_value_number(1.0);
    call(LANA_HOST_DENSITY_OPERATOR, args, 1, LANA_ERR_TYPE);
}

static void povm_channel_observable(void) {
    /* povm: computational basis {|0><0|, |1><1|}. */
    double e0_data[] = {1, 0, 0, 0, 0, 0, 0, 0};
    double e1_data[] = {0, 0, 0, 0, 0, 0, 1, 0};
    LanaTensor e0 = complex2(e0_data), e1 = complex2(e1_data);
    Value items[2] = {lana_value_tensor(&e0), lana_value_tensor(&e1)};
    LanaArray arr = {2u, 2u, items};
    Value args[] = {lana_value_array(&arr)};
    size_t shape[3] = {2, 2, 2}, strides[3] = {4, 2, 1};
    double stack_data[] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0};
    LanaTensor expected = {3, shape, strides, true, LANA_TENSOR_COMPLEX, (uint8_t *)stack_data, 0, NULL, false, LANA_TENSOR_CPU, NULL, NULL};
    call_checked(LANA_HOST_POVM, args, 1, LANA_OK, &expected);

    /* povm with Σ E_i != I is rejected. */
    items[1] = lana_value_tensor(&e0);
    call(LANA_HOST_POVM, args, 1, LANA_ERR_INVALID_PARAMETERS);
    items[1] = lana_value_tensor(&e1);

    /* channel: identity channel with a single Kraus operator I. */
    double i_data[] = {1, 0, 0, 0, 0, 0, 1, 0};
    LanaTensor identity = complex2(i_data);
    Value k_items[1] = {lana_value_tensor(&identity)};
    LanaArray k_arr = {1u, 1u, k_items};
    args[0] = lana_value_array(&k_arr);
    size_t k_shape[3] = {1, 2, 2}, k_strides[3] = {4, 2, 1};
    double k_stack[] = {1, 0, 0, 0, 0, 0, 1, 0};
    LanaTensor k_expected = {3, k_shape, k_strides, true, LANA_TENSOR_COMPLEX, (uint8_t *)k_stack, 0, NULL, false, LANA_TENSOR_CPU, NULL, NULL};
    call_checked(LANA_HOST_CHANNEL, args, 1, LANA_OK, &k_expected);

    /* observable: Pauli Z = [[1, 0], [0, -1]]. */
    double z_data[] = {1, 0, 0, 0, 0, 0, -1, 0};
    LanaTensor z = complex2(z_data);
    args[0] = lana_value_tensor(&z);
    LanaTensor z_expected = complex2(z_data);
    call_checked(LANA_HOST_OBSERVABLE, args, 1, LANA_OK, &z_expected);

    /* Non-Hermitian observable is rejected. */
    double non_hermitian_data[] = {1, 0, 1, 0, 0, 0, 1, 0};
    LanaTensor non_hermitian = complex2(non_hermitian_data);
    args[0] = lana_value_tensor(&non_hermitian);
    call(LANA_HOST_OBSERVABLE, args, 1, LANA_ERR_INVALID_PARAMETERS);
}

static void operations(void) {
    /* rho = I/2 and sigma = |0><0|. */
    double rho_data[] = {0.5, 0, 0, 0, 0, 0, 0.5, 0};
    double sigma_data[] = {1, 0, 0, 0, 0, 0, 0, 0};
    LanaTensor rho = complex2(rho_data), sigma = complex2(sigma_data);
    Value rho_v = lana_value_nqubit_state(&rho);
    Value sigma_v = lana_value_nqubit_state(&sigma);

    /* tensor_product(rho, rho) = I/4 (4x4 diagonal). */
    Value args[3] = {rho_v, rho_v, {0}};
    size_t shape4[2] = {4, 4}, strides4[2] = {4, 1};
    double tp_data[] = {0.25, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0.25, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0.25, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0.25, 0};
    LanaTensor tp_expected = {2, shape4, strides4, true, LANA_TENSOR_COMPLEX, (uint8_t *)tp_data, 0, NULL, false, LANA_TENSOR_CPU, NULL, NULL};
    call_checked(LANA_HOST_TENSOR_PRODUCT, args, 2, LANA_OK, &tp_expected);

    /* partial_trace(rho ⊗ rho, 1) = rho. */
    args[0] = lana_value_nqubit_state(&tp_expected);
    args[1] = lana_value_number(1);
    LanaTensor rho_expected = complex2(rho_data);
    call_checked(LANA_HOST_PARTIAL_TRACE, args, 2, LANA_OK, &rho_expected);

    /* partial_trace with subsystem >= N is rejected. */
    args[0] = rho_v;
    call(LANA_HOST_PARTIAL_TRACE, args, 2, LANA_ERR_INVALID_PARAMETERS);

    /* measure_with(rho, povm) = [0.5, 0.5]. */
    size_t p_shape[3] = {2, 2, 2}, p_strides[3] = {4, 2, 1};
    double p_stack[] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0};
    LanaTensor povm = {3, p_shape, p_strides, true, LANA_TENSOR_COMPLEX, (uint8_t *)p_stack, 0, NULL, false, LANA_TENSOR_CPU, NULL, NULL};
    args[0] = rho_v;
    args[1] = lana_value_povm(&povm);
    LanaChunk chunk;
    LanaVM vm;
    lana_chunk_init(&chunk);
    LanaInstruction invoke = {OP_HOST_CALL, 8u, LANA_HOST_MEASURE_WITH, 0u, 2u, 1u};
    LanaInstruction halt = {OP_HALT, 0u, 0u, 0u, 0u, 2u};
    assert(lana_chunk_emit(&chunk, invoke) == LANA_OK);
    assert(lana_chunk_emit(&chunk, halt) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    vm.frames[0].registers[0] = args[0];
    vm.frames[0].registers[1] = args[1];
    assert(lana_vm_run(&vm) == LANA_OK);
    Value dist = vm.frames[0].registers[8];
    assert(dist.type == VAL_ARRAY && dist.as.array->count == 2u);
    assert(dist.as.array->items[0].as.number == 0.5);
    assert(dist.as.array->items[1].as.number == 0.5);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);

    /* apply_to(identity channel, rho) = rho. */
    size_t k_shape[3] = {1, 2, 2}, k_strides[3] = {4, 2, 1};
    double k_stack[] = {1, 0, 0, 0, 0, 0, 1, 0};
    LanaTensor chan = {3, k_shape, k_strides, true, LANA_TENSOR_COMPLEX, (uint8_t *)k_stack, 0, NULL, false, LANA_TENSOR_CPU, NULL, NULL};
    args[0] = lana_value_channel(&chan);
    args[1] = rho_v;
    call_checked(LANA_HOST_APPLY_TO, args, 2, LANA_OK, &rho_expected);

    /* expect(rho, Z) = 0. */
    double z_data[] = {1, 0, 0, 0, 0, 0, -1, 0};
    LanaTensor z = complex2(z_data);
    args[0] = rho_v;
    args[1] = lana_value_observable(&z);
    call_number(LANA_HOST_EXPECT, args, 2, 0.0);

    /* mix(rho, sigma, 0.5) = [[0.75, 0], [0, 0.25]]. */
    args[0] = rho_v;
    args[1] = sigma_v;
    args[2] = lana_value_number(0.5);
    double mix_data[] = {0.75, 0, 0, 0, 0, 0, 0.25, 0};
    LanaTensor mix_expected = complex2(mix_data);
    call_checked(LANA_HOST_MIX, args, 3, LANA_OK, &mix_expected);

    /* trace_distance(rho, sigma) = 0.5. */
    args[0] = rho_v;
    args[1] = sigma_v;
    call_number(LANA_HOST_TRACE_DISTANCE, args, 2, 0.5);

    /* is_separable(rho ⊗ rho, 1) = "separable" (product state). */
    args[0] = lana_value_nqubit_state(&tp_expected);
    args[1] = lana_value_number(1);
    lana_chunk_init(&chunk);
    invoke = (LanaInstruction){OP_HOST_CALL, 8u, LANA_HOST_IS_SEPARABLE, 0u, 2u, 1u};
    assert(lana_chunk_emit(&chunk, invoke) == LANA_OK);
    assert(lana_chunk_emit(&chunk, halt) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    vm.frames[0].registers[0] = args[0];
    vm.frames[0].registers[1] = args[1];
    assert(lana_vm_run(&vm) == LANA_OK);
    Value sep = vm.frames[0].registers[8];
    assert(sep.type == VAL_STRING && strcmp(sep.as.string, "separable") == 0);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);

    /* to_state(rho) recovers (p, d_re, d_im). */
    args[0] = rho_v;
    lana_chunk_init(&chunk);
    invoke = (LanaInstruction){OP_HOST_CALL, 8u, LANA_HOST_TO_STATE, 0u, 1u, 1u};
    assert(lana_chunk_emit(&chunk, invoke) == LANA_OK);
    assert(lana_chunk_emit(&chunk, halt) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    vm.frames[0].registers[0] = args[0];
    assert(lana_vm_run(&vm) == LANA_OK);
    Value st = vm.frames[0].registers[8];
    assert(st.type == VAL_STATE && st.as.state.state.p == 0.5);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);
}

int main(void) {
    density_operator();
    povm_channel_observable();
    operations();
    return 0;
}
