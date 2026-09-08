#include "tensor.h"
#include "vm.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/* LIP-007: differentiable STATE tensors. Exercises the state_tensor, append,
 * measure, and transform host calls directly against the C11 VM, mirroring the
 * differential fixtures in tests/conformance/differential/hostcalls/lip007_*. */

/* A 2x2 complex tensor laid out as interleaved [re, im] row-major data. */
static LanaTensor complex2(double *data) {
    static size_t shape[2] = {2, 2}, strides[2] = {2, 1};
    LanaTensor t = {2, shape, strides, true, LANA_TENSOR_COMPLEX, (uint8_t *)data, 0, NULL, false, LANA_TENSOR_CPU, NULL, NULL};
    return t;
}

/* A single-element STATE tensor (shape [1, 2, 2]) over interleaved complex
 * data. */
static LanaTensor state1(double *data) {
    static size_t shape[3] = {1, 2, 2}, strides[3] = {4, 2, 1};
    LanaTensor t = {3, shape, strides, true, LANA_TENSOR_COMPLEX, (uint8_t *)data, 0, NULL, true, LANA_TENSOR_CPU, NULL, NULL};
    return t;
}

/* Invoke a host call with `argc` arguments and assert the error code. */
static void call(uint32_t host, Value *args, uint32_t argc, LanaError expected) {
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
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);
}

/* Invoke a host call and assert a successful tensor result whose logical
 * elements match `expected` (including the `is_state` flag). */
static void call_tensor(uint32_t host, Value *args, uint32_t argc,
                        const LanaTensor *expected) {
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
    assert(value.type == VAL_TENSOR);
    const LanaTensor *actual = value.as.tensor;
    assert(actual->ndim == expected->ndim);
    assert(actual->is_complex == expected->is_complex);
    assert(actual->is_state == expected->is_state);
    size_t total = 1u, mult = expected->is_complex ? 2u : 1u;
    for (size_t i = 0u; i < actual->ndim; ++i) {
        assert(actual->shape[i] == expected->shape[i]);
        total *= actual->shape[i];
    }
    for (size_t lin = 0u; lin < total; ++lin) {
        size_t rem = lin, index = actual->offset;
        for (size_t d = actual->ndim; d-- > 0u;) {
            index += (rem % actual->shape[d]) * actual->strides[d];
            rem /= actual->shape[d];
        }
        for (size_t c = 0u; c < mult; ++c)
            assert(actual->data[index * mult + c] == expected->data[lin * mult + c]);
    }
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);
}

/* Build a nested array literal for state_tensor: a batch of 2x2 density
 * matrices given as real entries (imaginary parts are zero). */
static Value state_literal(double m00, double m01, double m10, double m11) {
    static Value row0_items[2], row1_items[2], state_items[2], outer_items[1];
    static LanaArray row0, row1, state, outer;
    row0_items[0] = lana_value_number(m00);
    row0_items[1] = lana_value_number(m01);
    row0.count = row0.capacity = 2u;
    row0.items = row0_items;
    row1_items[0] = lana_value_number(m10);
    row1_items[1] = lana_value_number(m11);
    row1.count = row1.capacity = 2u;
    row1.items = row1_items;
    state_items[0] = lana_value_array(&row0);
    state_items[1] = lana_value_array(&row1);
    state.count = state.capacity = 2u;
    state.items = state_items;
    outer_items[0] = lana_value_array(&state);
    outer.count = outer.capacity = 1u;
    outer.items = outer_items;
    return lana_value_array(&outer);
}

static void state_tensor_construction(void) {
    /* Maximally mixed state I/2 = [[0.5, 0], [0, 0.5]]. */
    Value args[1] = {state_literal(0.5, 0.0, 0.0, 0.5)};
    double mixed_data[] = {0.5, 0, 0, 0, 0, 0, 0.5, 0};
    LanaTensor mixed = state1(mixed_data);
    call_tensor(LANA_HOST_STATE_TENSOR, args, 1, &mixed);

    /* Pure state |0><0| = [[1, 0], [0, 0]]. */
    args[0] = state_literal(1.0, 0.0, 0.0, 0.0);
    double pure_data[] = {1, 0, 0, 0, 0, 0, 0, 0};
    LanaTensor pure = state1(pure_data);
    call_tensor(LANA_HOST_STATE_TENSOR, args, 1, &pure);

    /* Non-Hermitian is rejected. */
    args[0] = state_literal(0.5, 0.3, 0.0, 0.5);
    call(LANA_HOST_STATE_TENSOR, args, 1, LANA_ERR_INVALID_STATE);

    /* Non-unit trace is rejected. */
    args[0] = state_literal(1.0, 0.0, 0.0, 1.0);
    call(LANA_HOST_STATE_TENSOR, args, 1, LANA_ERR_INVALID_STATE);

    /* Non-PSD (negative eigenvalue) is rejected. */
    args[0] = state_literal(1.5, 0.0, 0.0, -0.5);
    call(LANA_HOST_STATE_TENSOR, args, 1, LANA_ERR_INVALID_STATE);

    /* A non-array argument is a type error. */
    args[0] = lana_value_number(1.0);
    call(LANA_HOST_STATE_TENSOR, args, 1, LANA_ERR_TYPE);
}

static void append_measure_transform(void) {
    /* s0 = I/2, s1 = |0><0|. */
    double s0_data[] = {0.5, 0, 0, 0, 0, 0, 0.5, 0};
    double s1_data[] = {1, 0, 0, 0, 0, 0, 0, 0};
    LanaTensor s0 = state1(s0_data), s1 = state1(s1_data);
    Value s0_v = lana_value_tensor(&s0), s1_v = lana_value_tensor(&s1);

    /* append(s0, s0): p_C = 0.5 + 0.5 - 0.25 = 0.75. */
    Value args[2] = {s0_v, s0_v};
    double app_data[] = {0.75, 0, 0, 0, 0, 0, 0.25, 0};
    LanaTensor app = state1(app_data);
    call_tensor(LANA_HOST_APPEND, args, 2, &app);

    /* append(s0, s1): p_C = 0.5 + 1 - 0.5 = 1.0. */
    args[1] = s1_v;
    double app2_data[] = {1, 0, 0, 0, 0, 0, 0, 0};
    LanaTensor app2 = state1(app2_data);
    call_tensor(LANA_HOST_APPEND, args, 2, &app2);

    /* append on a non-state tensor is a type error. */
    double plain_data[] = {0.5, 0, 0, 0, 0, 0, 0.5, 0};
    LanaTensor plain = complex2(plain_data);
    args[0] = lana_value_tensor(&plain);
    args[1] = s0_v;
    call(LANA_HOST_APPEND, args, 2, LANA_ERR_TYPE);

    /* Computational POVM {|0><0|, |1><1|}: stack shape [2, 2, 2]. */
    size_t p_shape[3] = {2, 2, 2}, p_strides[3] = {4, 2, 1};
    double p_stack[] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0};
    LanaTensor povm = {3, p_shape, p_strides, true, LANA_TENSOR_COMPLEX, (uint8_t *)p_stack, 0, NULL, false, LANA_TENSOR_CPU, NULL, NULL};
    Value povm_v = lana_value_povm(&povm);

    /* measure(s0, povm) = [0.5, 0.5]. */
    args[0] = s0_v;
    args[1] = povm_v;
    size_t q_shape[2] = {1, 2}, q_strides[2] = {2, 1};
    double q_data[] = {0.5, 0.5};
    LanaTensor q_expected = {2, q_shape, q_strides, false, LANA_TENSOR_F64, (uint8_t *)q_data, 0, NULL, false, LANA_TENSOR_CPU, NULL, NULL};
    call_tensor(LANA_HOST_MEASURE, args, 2, &q_expected);

    /* measure(s1, povm) = [1.0, 0.0]. */
    args[0] = s1_v;
    double q1_data[] = {1.0, 0.0};
    LanaTensor q1_expected = {2, q_shape, q_strides, false, LANA_TENSOR_F64, (uint8_t *)q1_data, 0, NULL, false, LANA_TENSOR_CPU, NULL, NULL};
    call_tensor(LANA_HOST_MEASURE, args, 2, &q1_expected);

    /* measure on a non-state tensor is a type error. */
    args[0] = lana_value_tensor(&plain);
    call(LANA_HOST_MEASURE, args, 2, LANA_ERR_TYPE);

    /* Identity channel (single Kraus operator I): stack shape [1, 2, 2]. */
    size_t k_shape[3] = {1, 2, 2}, k_strides[3] = {4, 2, 1};
    double k_stack[] = {1, 0, 0, 0, 0, 0, 1, 0};
    LanaTensor chan = {3, k_shape, k_strides, true, LANA_TENSOR_COMPLEX, (uint8_t *)k_stack, 0, NULL, false, LANA_TENSOR_CPU, NULL, NULL};
    Value chan_v = lana_value_channel(&chan);

    /* transform(s0, identity) = s0. */
    args[0] = s0_v;
    args[1] = chan_v;
    call_tensor(LANA_HOST_TRANSFORM, args, 2, &s0);

    /* transform on a non-state tensor is a type error. */
    args[0] = lana_value_tensor(&plain);
    call(LANA_HOST_TRANSFORM, args, 2, LANA_ERR_TYPE);
}

int main(void) {
    state_tensor_construction();
    append_measure_transform();
    (void)puts("STATE_TENSOR_PASS");
    return 0;
}
