#include "tensor.h"
#include "vm.h"
#include "data.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/* Compare a result tensor's logical elements against a row-major expected
 * buffer. A view shares its base buffer and carries its own offset/strides,
 * so walk the logical layout instead of reading data[0..] linearly. */
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

/* Like `tensor_elements_match`, but compares each element within a relative
 * tolerance (for the approximate float32 GPU path). */
static int tensor_elements_match_tol(const LanaTensor *actual, const LanaTensor *expected,
                                     double tol) {
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
        for (size_t c = 0u; c < mult; ++c) {
            double a = actual->data[index * mult + c];
            double e = expected->data[lin * mult + c];
            if (fabs(a - e) > tol * (1.0 + fabs(e))) return 0;
        }
    }
    return 1;
}

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
        assert(value.type == VAL_TENSOR);
        assert(tensor_elements_match(value.as.tensor, result));
    }
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);
}

/* Like `call`, but asserts a successful scalar `number` result. */
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

/* LIP-027: construct a tensor via `constructor` with the given args (the last
 * of which may be a dtype string), then call dtype(t) on the result and assert
 * it equals `expected`. Runs both host calls in one VM so the tensor survives. */
static void call_dtype(uint32_t constructor, Value *args, uint32_t argc,
                       const char *expected) {
    LanaChunk chunk;
    LanaVM vm;
    lana_chunk_init(&chunk);
    LanaInstruction ctor = {OP_HOST_CALL, 8u, constructor, 0u, argc, 1u};
    LanaInstruction dtype = {OP_HOST_CALL, 9u, LANA_HOST_TENSOR_DTYPE, 8u, 1u, 1u};
    LanaInstruction halt = {OP_HALT, 0u, 0u, 0u, 0u, 2u};
    assert(lana_chunk_emit(&chunk, ctor) == LANA_OK);
    assert(lana_chunk_emit(&chunk, dtype) == LANA_OK);
    assert(lana_chunk_emit(&chunk, halt) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    for (uint32_t i = 0u; i < argc; ++i) vm.frames[0].registers[i] = args[i];
    assert(lana_vm_run(&vm) == LANA_OK);
    Value value = vm.frames[0].registers[9];
    assert(value.type == VAL_STRING);
    assert(strcmp(value.as.string, expected) == 0);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);
}

static void axis_reductions(void) {
    size_t shape[] = {2, 3}, strides[] = {3, 1}, out_size = 3;
    double data[] = {1, 2, 3, 4, 5, 6};
    LanaTensor t = {2, shape, strides, false, LANA_TENSOR_F64, data, 0, NULL, false};
    Value args[] = {lana_value_tensor(&t), lana_value_number(0)};
    double sums[] = {5, 7, 9}, means[] = {2.5, 3.5, 4.5};
    LanaTensor expected = {1, &out_size, NULL, false, LANA_TENSOR_F64, sums, 0, NULL, false};
    call_checked(LANA_HOST_TENSOR_SUM, args, 2, LANA_OK, &expected);
    expected.data = means;
    call_checked(LANA_HOST_TENSOR_MEAN, args, 2, LANA_OK, &expected);
    args[1] = lana_value_number(-1);
    double maxima[] = {3, 6}, minima[] = {1, 4};
    out_size = 2; expected.data = maxima;
    call_checked(LANA_HOST_TENSOR_MAX, args, 2, LANA_OK, &expected);
    expected.data = minima;
    call_checked(LANA_HOST_TENSOR_MIN, args, 2, LANA_OK, &expected);
    const double bad_axes[] = {NAN, INFINITY, -INFINITY, 0.5, -3, 2, 0x1p64};
    for (size_t i = 0; i < sizeof(bad_axes) / sizeof(bad_axes[0]); ++i) {
        args[1] = lana_value_number(bad_axes[i]);
        call(LANA_HOST_TENSOR_SUM, args, 2, LANA_ERR_INVALID_PARAMETERS);
    }
    args[1] = lana_value_bool(false);
    call(LANA_HOST_TENSOR_SUM, args, 2, LANA_ERR_TYPE);
    args[1] = lana_value_number(0);
    shape[0] = 0; out_size = 3;
    double zeros[] = {0, 0, 0}; expected.data = zeros;
    call_checked(LANA_HOST_TENSOR_SUM, args, 2, LANA_OK, &expected);
    for (uint32_t op = LANA_HOST_TENSOR_MEAN; op <= LANA_HOST_TENSOR_MIN; ++op)
        call(op, args, 2, LANA_ERR_INVALID_PARAMETERS);
    args[1] = lana_value_number(1); out_size = 0;
    call_checked(LANA_HOST_TENSOR_MEAN, args, 2, LANA_OK, &expected);
    t.ndim = 0; args[1] = lana_value_number(0);
    call(LANA_HOST_TENSOR_SUM, args, 2, LANA_ERR_INVALID_PARAMETERS);
    t.ndim = 1; shape[0] = 2; strides[0] = 1; t.is_complex = true;
    double complex_sum[] = {4, 6};
    expected.ndim = 0; expected.is_complex = true; expected.data = complex_sum;
    call_checked(LANA_HOST_TENSOR_SUM, args, 2, LANA_OK, &expected);
    call(LANA_HOST_TENSOR_MAX, args, 2, LANA_ERR_TYPE);
    data[1] = INFINITY;
    call(LANA_HOST_TENSOR_SUM, args, 2, LANA_ERR_INVALID_PARAMETERS);
}

/* LIP-004 indexing and slicing through the shared index_get / index_set host
 * calls: integer positions (negative wrap, Key out of range, Type on
 * non-integers), clamped slices, views that share the source buffer, views
 * of views, strided traversal in arithmetic and reduction, and the complex
 * scalar form. */
static void indexing(void) {
    /* t = [[1, 2, 3], [4, 5, 6]] */
    size_t shape[] = {2, 3}, strides[] = {3, 1};
    double data[] = {1, 2, 3, 4, 5, 6};
    LanaTensor t = {2, shape, strides, false, LANA_TENSOR_F64, data, 0, NULL, false};
    Value args[3] = {lana_value_tensor(&t), {0}, {0}};

    /* One integer position selects a row: a rank-1 view sharing the buffer. */
    size_t row = 3, one = 1;
    double row_456[] = {4, 5, 6};
    LanaTensor expected_row = {1, &row, &one, false, LANA_TENSOR_F64, row_456, 0, NULL, false};
    args[1] = lana_value_number(1);
    call_checked(LANA_HOST_INDEX_GET, args, 2, LANA_OK, &expected_row);
    args[1] = lana_value_number(-1); /* negative counts from the end */
    call_checked(LANA_HOST_INDEX_GET, args, 2, LANA_OK, &expected_row);

    /* Out-of-range integer positions are Key errors (after negative wrap). */
    const double out_of_range[] = {2, -3, 0x1p64};
    for (size_t i = 0; i < sizeof(out_of_range) / sizeof(out_of_range[0]); ++i) {
        args[1] = lana_value_number(out_of_range[i]);
        call(LANA_HOST_INDEX_GET, args, 2, LANA_ERR_KEY);
    }

    /* Non-number, non-integer, and non-finite positions are Type errors. */
    const double bad_positions[] = {NAN, INFINITY, -INFINITY, 0.5, -0.5};
    for (size_t i = 0; i < sizeof(bad_positions) / sizeof(bad_positions[0]); ++i) {
        args[1] = lana_value_number(bad_positions[i]);
        call(LANA_HOST_INDEX_GET, args, 2, LANA_ERR_TYPE);
    }
    args[1] = lana_value_bool(true);
    call(LANA_HOST_INDEX_GET, args, 2, LANA_ERR_TYPE);

    /* Position lists: numbers are integer positions, two-element arrays are
     * [start, end] slices. Fewer positions than the rank keep trailing axes
     * whole; integer axes drop out, slice axes stay. */
    Value pair_items[3], positions[3];
    LanaArray pair = {2u, 2u, pair_items}, spec = {2u, 2u, positions};
    pair_items[0] = lana_value_number(0);
    pair_items[1] = lana_value_number(2);
    positions[0] = lana_value_array(&pair);
    positions[1] = lana_value_number(1);
    args[1] = lana_value_array(&spec);
    size_t two = 2, col_stride = 3;
    double col_25[] = {2, 5};
    LanaTensor expected_col = {1, &two, &col_stride, false, LANA_TENSOR_F64, col_25, 0, NULL, false};
    call_checked(LANA_HOST_INDEX_GET, args, 2, LANA_OK, &expected_col); /* t[0:2, 1] */

    /* A full set of integer positions selects one element as a number. */
    positions[0] = lana_value_number(1);
    positions[1] = lana_value_number(2);
    call_number(LANA_HOST_INDEX_GET, args, 2, 6.0);
    positions[0] = lana_value_number(-1);
    positions[1] = lana_value_number(-3); /* wraps: -1 -> 1, -3 -> 0 */
    call_number(LANA_HOST_INDEX_GET, args, 2, 4.0);

    /* More positions than the rank is InvalidParameters. */
    positions[0] = positions[1] = positions[2] = lana_value_number(0);
    spec.count = 3u;
    call(LANA_HOST_INDEX_GET, args, 2, LANA_ERR_INVALID_PARAMETERS);
    spec.count = 2u;

    /* Slice bounds clamp and never range-error. */
    pair_items[0] = lana_value_number(-10); /* clamps to 0 */
    pair_items[1] = lana_value_number(10);  /* clamps to 2 */
    positions[0] = lana_value_array(&pair);
    spec.count = 1u; /* axis 0 sliced, axis 1 kept whole */
    LanaTensor expected_whole = {2, shape, strides, false, LANA_TENSOR_F64, data, 0, NULL, false};
    call_checked(LANA_HOST_INDEX_GET, args, 2, LANA_OK, &expected_whole);
    pair_items[0] = lana_value_number(5); /* clamps past the end: empty axis */
    size_t empty_shape[] = {0, 3};
    LanaTensor expected_empty = {2, empty_shape, strides, false, LANA_TENSOR_F64, data, 0, NULL, false};
    call_checked(LANA_HOST_INDEX_GET, args, 2, LANA_OK, &expected_empty);
    pair_items[0] = lana_value_number(1);
    pair_items[1] = lana_value_number(0); /* start > end: empty axis */
    call_checked(LANA_HOST_INDEX_GET, args, 2, LANA_OK, &expected_empty);

    /* Malformed specs are Type errors: a non-number position, a pair holding
     * a non-number bound, and a pair that is not two elements long. */
    positions[0] = lana_value_bool(true);
    call(LANA_HOST_INDEX_GET, args, 2, LANA_ERR_TYPE);
    pair_items[0] = lana_value_number(0);
    pair_items[1] = lana_value_bool(true);
    positions[0] = lana_value_array(&pair);
    call(LANA_HOST_INDEX_GET, args, 2, LANA_ERR_TYPE);
    pair_items[1] = lana_value_number(0);
    pair_items[2] = lana_value_number(0);
    pair.count = 3u;
    call(LANA_HOST_INDEX_GET, args, 2, LANA_ERR_TYPE);
    pair.count = 2u;

    /* index_set never writes tensor elements: views are read-only. */
    args[2] = lana_value_number(0);
    call(LANA_HOST_INDEX_SET, args, 3, LANA_ERR_TYPE);

    /* A position on a rank-zero tensor has no axis to select. */
    LanaTensor scalar = {0, NULL, NULL, false, LANA_TENSOR_F64, data, 0, NULL, false};
    args[0] = lana_value_tensor(&scalar);
    args[1] = lana_value_number(0);
    call(LANA_HOST_INDEX_GET, args, 2, LANA_ERR_INVALID_PARAMETERS);

    /* Views chain: selecting a column view again still reads through offset
     * and strides, and the base chain keeps the shared buffer reachable. */
    size_t col_shape = 2, col_strides = 3;
    LanaTensor column = {1, &col_shape, &col_strides, false, LANA_TENSOR_F64, data, 1, &t, false};
    args[0] = lana_value_tensor(&column);
    args[1] = lana_value_number(1);
    call_number(LANA_HOST_INDEX_GET, args, 2, 5.0);
    args[1] = lana_value_number(-1);
    call_number(LANA_HOST_INDEX_GET, args, 2, 5.0);
    args[1] = lana_value_number(2);
    call(LANA_HOST_INDEX_GET, args, 2, LANA_ERR_KEY);

    /* Strided traversal: arithmetic and full reduction over the column view
     * follow its strides rather than assuming a contiguous fiber. */
    args[1] = lana_value_tensor(&column);
    double doubled[] = {4, 10};
    LanaTensor expected_doubled = {1, &two, &one, false, LANA_TENSOR_F64, doubled, 0, NULL, false};
    call_checked(LANA_HOST_TENSOR_ADD, args, 2, LANA_OK, &expected_doubled);
    call_number(LANA_HOST_TENSOR_SUM, args, 1, 7.0);

    /* A complex element selection yields the rank-zero complex form. */
    size_t c_size = 1, c_stride = 1;
    double cdata[] = {3, 4};
    LanaTensor ctensor = {1, &c_size, &c_stride, true, LANA_TENSOR_COMPLEX, cdata, 0, NULL, false};
    args[0] = lana_value_tensor(&ctensor);
    args[1] = lana_value_number(0);
    LanaTensor expected_c = {0, NULL, NULL, true, LANA_TENSOR_COMPLEX, cdata, 0, NULL, false};
    call_checked(LANA_HOST_INDEX_GET, args, 2, LANA_OK, &expected_c);
}

/* LIP-004 section 5: matmul through the shared native backend. Covers the
 * NumPy shape semantics (dot, both vector-matrix promotions, 2-d, batched
 * with broadcast), complex zgemm, direct-strided views (lda = row stride),
 * packed non-contiguous cores, and the empty contraction. */
static void matmul(void) {
    Value args[2];

    /* 1d x 1d is a dot product: a rank-zero tensor. */
    size_t v3 = 3, one = 1;
    double a1[] = {1, 2, 3}, b1[] = {4, 5, 6};
    LanaTensor ta = {1, &v3, &one, false, LANA_TENSOR_F64, a1, 0, NULL, false};
    LanaTensor tb = {1, &v3, &one, false, LANA_TENSOR_F64, b1, 0, NULL, false};
    double dot[] = {32};
    LanaTensor expected_dot = {0, NULL, NULL, false, LANA_TENSOR_F64, dot, 0, NULL, false};
    args[0] = lana_value_tensor(&ta); args[1] = lana_value_tensor(&tb);
    call_checked(LANA_HOST_TENSOR_MATMUL, args, 2, LANA_OK, &expected_dot);

    /* 1d x 2d promotes the vector to a row: [1,2,3] . [[1,2],[3,4],[5,6]]. */
    size_t s23[] = {3, 2}, st32[] = {2, 1};
    double bm[] = {1, 2, 3, 4, 5, 6};
    LanaTensor tbm = {2, s23, st32, false, LANA_TENSOR_F64, bm, 0, NULL, false};
    size_t two = 2;
    double vm[] = {22, 28};
    LanaTensor expected_vm = {1, &two, &one, false, LANA_TENSOR_F64, vm, 0, NULL, false};
    args[1] = lana_value_tensor(&tbm);
    call_checked(LANA_HOST_TENSOR_MATMUL, args, 2, LANA_OK, &expected_vm);

    /* 2d x 1d promotes the vector to a column: [[1,2,3],[4,5,6]] . [1,2,3]. */
    size_t s23b[] = {2, 3}, st23b[] = {3, 1};
    double am[] = {1, 2, 3, 4, 5, 6};
    LanaTensor tam = {2, s23b, st23b, false, LANA_TENSOR_F64, am, 0, NULL, false};
    double mv[] = {14, 32};
    LanaTensor expected_mv = {1, &two, &one, false, LANA_TENSOR_F64, mv, 0, NULL, false};
    args[0] = lana_value_tensor(&tam); args[1] = lana_value_tensor(&ta);
    call_checked(LANA_HOST_TENSOR_MATMUL, args, 2, LANA_OK, &expected_mv);

    /* 2d x 2d: [[1,2],[3,4]] . [[5,6],[7,8]]. */
    size_t s22[] = {2, 2}, st22[] = {2, 1};
    double a2[] = {1, 2, 3, 4}, b2[] = {5, 6, 7, 8};
    LanaTensor ta2 = {2, s22, st22, false, LANA_TENSOR_F64, a2, 0, NULL, false};
    LanaTensor tb2 = {2, s22, st22, false, LANA_TENSOR_F64, b2, 0, NULL, false};
    double mm[] = {19, 22, 43, 50};
    LanaTensor expected_mm = {2, s22, st22, false, LANA_TENSOR_F64, mm, 0, NULL, false};
    args[0] = lana_value_tensor(&ta2); args[1] = lana_value_tensor(&tb2);
    call_checked(LANA_HOST_TENSOR_MATMUL, args, 2, LANA_OK, &expected_mm);

    /* Batched with broadcast: a[2,2,2] . b[1,2,2] -> [2,2,2]. */
    size_t s222[] = {2, 2, 2}, st222[] = {4, 2, 1};
    double ab[] = {1, 2, 3, 4, 5, 6, 7, 8};
    LanaTensor tab = {3, s222, st222, false, LANA_TENSOR_F64, ab, 0, NULL, false};
    size_t s122[] = {1, 2, 2}, st122[] = {4, 2, 1};
    double bb[] = {1, 0, 0, 1}; /* identity, broadcast across the batch */
    LanaTensor tbb = {3, s122, st122, false, LANA_TENSOR_F64, bb, 0, NULL, false};
    double batched[] = {1, 2, 3, 4, 5, 6, 7, 8};
    LanaTensor expected_batched = {3, s222, st222, false, LANA_TENSOR_F64, batched, 0, NULL, false};
    args[0] = lana_value_tensor(&tab); args[1] = lana_value_tensor(&tbb);
    call_checked(LANA_HOST_TENSOR_MATMUL, args, 2, LANA_OK, &expected_batched);

    /* Complex zgemm: (iI) . (iI) = -I. */
    double ca[] = {0, 1, 0, 0, 0, 0, 0, 1};
    double cb[] = {0, 1, 0, 0, 0, 0, 0, 1};
    LanaTensor tca = {2, s22, st22, true, LANA_TENSOR_COMPLEX, ca, 0, NULL, false};
    LanaTensor tcb = {2, s22, st22, true, LANA_TENSOR_COMPLEX, cb, 0, NULL, false};
    double cm[] = {-1, 0, 0, 0, 0, 0, -1, 0};
    LanaTensor expected_cm = {2, s22, st22, true, LANA_TENSOR_COMPLEX, cm, 0, NULL, false};
    args[0] = lana_value_tensor(&tca); args[1] = lana_value_tensor(&tcb);
    call_checked(LANA_HOST_TENSOR_MATMUL, args, 2, LANA_OK, &expected_cm);

    /* Direct-strided view: columns 0:3 of a 2x4 base keep column stride 1 but
     * row stride 4, so lda = 4 > K = 3 and BLAS reads the true row stride. */
    size_t s24[] = {2, 4}, st24[] = {4, 1};
    double base24[] = {1, 2, 3, 4, 5, 6, 7, 8};
    LanaTensor tbase24 = {2, s24, st24, false, LANA_TENSOR_F64, base24, 0, NULL, false};
    size_t s23v[] = {2, 3}, st23v[] = {4, 1};
    LanaTensor view23 = {2, s23v, st23v, false, LANA_TENSOR_F64, base24, 0, &tbase24, false};
    double b32[] = {1, 0, 0, 1, 1, 1};
    LanaTensor tb32 = {2, s23, st32, false, LANA_TENSOR_F64, b32, 0, NULL, false};
    double strided[] = {4, 5, 12, 13};
    LanaTensor expected_strided = {2, s22, st22, false, LANA_TENSOR_F64, strided, 0, NULL, false};
    args[0] = lana_value_tensor(&view23); args[1] = lana_value_tensor(&tb32);
    call_checked(LANA_HOST_TENSOR_MATMUL, args, 2, LANA_OK, &expected_strided);

    /* Packed non-contiguous core: columns 0:4:2 of a 2x4 base have column
     * stride 2, so the operand is gathered into scratch before the call. */
    size_t s22v[] = {2, 2}, st22v[] = {4, 2};
    LanaTensor view22 = {2, s22v, st22v, false, LANA_TENSOR_F64, base24, 0, &tbase24, false};
    double b22p[] = {1, 1, 0, 1};
    LanaTensor tb22p = {2, s22, st22, false, LANA_TENSOR_F64, b22p, 0, NULL, false};
    double packed[] = {1, 4, 5, 12};
    LanaTensor expected_packed = {2, s22, st22, false, LANA_TENSOR_F64, packed, 0, NULL, false};
    args[0] = lana_value_tensor(&view22); args[1] = lana_value_tensor(&tb22p);
    call_checked(LANA_HOST_TENSOR_MATMUL, args, 2, LANA_OK, &expected_packed);

    /* Empty contraction (K = 0): [2,0] . [0,3] -> [2,3] of zeros. */
    size_t s20[] = {2, 0}, st20[] = {0, 1};
    double empty_a[] = {0};
    LanaTensor tea = {2, s20, st20, false, LANA_TENSOR_F64, empty_a, 0, NULL, false};
    size_t s03[] = {0, 3}, st03[] = {3, 1};
    double empty_b[] = {0};
    LanaTensor teb = {2, s03, st03, false, LANA_TENSOR_F64, empty_b, 0, NULL, false};
    size_t s23o[] = {2, 3}, st23o[] = {3, 1};
    double empty_out[] = {0, 0, 0, 0, 0, 0};
    LanaTensor expected_empty = {2, s23o, st23o, false, LANA_TENSOR_F64, empty_out, 0, NULL, false};
    args[0] = lana_value_tensor(&tea); args[1] = lana_value_tensor(&teb);
    call_checked(LANA_HOST_TENSOR_MATMUL, args, 2, LANA_OK, &expected_empty);

    /* Error cases: mismatched complex, rank-zero operand, incompatible inner
     * dims, and incompatible batch dims. */
    args[0] = lana_value_tensor(&tca); args[1] = lana_value_tensor(&ta2);
    call(LANA_HOST_TENSOR_MATMUL, args, 2, LANA_ERR_TYPE);
    LanaTensor scalar = {0, NULL, NULL, false, LANA_TENSOR_F64, a1, 0, NULL, false};
    args[0] = lana_value_tensor(&scalar); args[1] = lana_value_tensor(&ta2);
    call(LANA_HOST_TENSOR_MATMUL, args, 2, LANA_ERR_INVALID_PARAMETERS);
    size_t s33[] = {3, 3}, st33[] = {3, 1};
    double a3[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    LanaTensor ta3 = {2, s33, st33, false, LANA_TENSOR_F64, a3, 0, NULL, false};
    args[0] = lana_value_tensor(&ta3); args[1] = lana_value_tensor(&ta2);
    call(LANA_HOST_TENSOR_MATMUL, args, 2, LANA_ERR_INVALID_PARAMETERS);
    size_t s322[] = {3, 2, 2}, st322[] = {4, 2, 1};
    double abad[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    LanaTensor tabad = {3, s322, st322, false, LANA_TENSOR_F64, abad, 0, NULL, false};
    size_t s222b[] = {2, 2, 2}, st222b[] = {4, 2, 1};
    double bbad[] = {1, 0, 0, 1, 1, 0, 0, 1};
    LanaTensor tbbad = {3, s222b, st222b, false, LANA_TENSOR_F64, bbad, 0, NULL, false};
    args[0] = lana_value_tensor(&tabad); args[1] = lana_value_tensor(&tbbad);
    call(LANA_HOST_TENSOR_MATMUL, args, 2, LANA_ERR_INVALID_PARAMETERS);
}

/* LIP-004 section 5: explicit GPU matmul. Runs a small real matmul on the
 * actual Metal device, asserts the result is within float32 tolerance of the
 * CPU result, and checks the APPROXIMATE derivation records the backend and
 * precision. Skips (does not fail) when no Metal device is present. */
static void gpu_matmul(void) {
    size_t s22[] = {2, 2}, st22[] = {2, 1};
    double a2[] = {1, 2, 3, 4}, b2[] = {5, 6, 7, 8};
    LanaTensor ta2 = {2, s22, st22, false, LANA_TENSOR_F64, a2, 0, NULL, false};
    LanaTensor tb2 = {2, s22, st22, false, LANA_TENSOR_F64, b2, 0, NULL, false};
    Value args[3];
    args[0] = lana_value_tensor(&ta2);
    args[1] = lana_value_tensor(&tb2);
    args[2] = lana_value_string("float32");

    LanaChunk chunk;
    LanaVM vm;
    lana_chunk_init(&chunk);
    LanaInstruction invoke = {OP_HOST_CALL, 8u, LANA_HOST_GPU_MATMUL, 0u, 3u, 1u};
    LanaInstruction halt = {OP_HALT, 0u, 0u, 0u, 0u, 2u};
    assert(lana_chunk_emit(&chunk, invoke) == LANA_OK);
    assert(lana_chunk_emit(&chunk, halt) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    for (uint32_t i = 0u; i < 3u; ++i) vm.frames[0].registers[i] = args[i];
    LanaError actual = lana_vm_run(&vm);
    if (actual == LANA_ERR_UNSUPPORTED_OPERATION) {
        /* No Metal device on this host; the GPU path is optional. */
        lana_vm_free(&vm);
        lana_chunk_free(&chunk);
        return;
    }
    assert(actual == LANA_OK);
    Value value = vm.frames[0].registers[8];
    assert(value.type == VAL_TENSOR);
    double expected[] = {19, 22, 43, 50};
    LanaTensor texpected = {2, s22, st22, false, LANA_TENSOR_F64, expected, 0, NULL, false};
    assert(tensor_elements_match_tol(value.as.tensor, &texpected, 1e-5));
    assert(value.derivation != NULL);
    assert(value.derivation->kind == LANA_DERIVATION_APPROXIMATION);
    assert(value.derivation->exactness == LANA_EXACTNESS_APPROXIMATE);
    assert(strcmp(value.derivation->operation, "gpu_matmul") == 0);
    assert(strcmp(value.derivation->details, "backend=metal precision=float32") == 0);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);

    /* Error cases: unknown precision and complex operand. */
    args[2] = lana_value_string("float64");
    call(LANA_HOST_GPU_MATMUL, args, 3, LANA_ERR_TYPE);
    args[2] = lana_value_string("float32");
    double ca[] = {0, 1, 0, 0, 0, 0, 0, 1};
    LanaTensor tca = {2, s22, st22, true, LANA_TENSOR_COMPLEX, ca, 0, NULL, false};
    args[0] = lana_value_tensor(&tca);
    call(LANA_HOST_GPU_MATMUL, args, 3, LANA_ERR_TYPE);
}

/* LIP-027: mixed-precision tensor dtypes. Construction with each dtype,
 * dtype(t) reporting, default f64, f16/bf16 rounding of literals, and the
 * error cases (unknown dtype, dtype: on a non-tensor constructor, dtype(t) on
 * a non-tensor). */
static void dtype_construction(void) {
    Value args[3];
    size_t one = 1, stride = 1;

    /* A 1-element array [0.1] for the tensor constructor. */
    Value items[1];
    items[0] = lana_value_number(0.1);
    LanaArray arr = {1, 1, items};
    Value arrv = lana_value_array(&arr);

    /* Default dtype is f64. */
    args[0] = arrv;
    call_dtype(LANA_HOST_TENSOR, args, 1, "f64");

    /* Explicit dtype: on tensor. */
    args[1] = lana_value_string("f32");
    call_dtype(LANA_HOST_TENSOR, args, 2, "f32");
    args[1] = lana_value_string("f16");
    call_dtype(LANA_HOST_TENSOR, args, 2, "f16");
    args[1] = lana_value_string("bf16");
    call_dtype(LANA_HOST_TENSOR, args, 2, "bf16");

    /* dtype: on zeros/ones/eye. */
    Value dims[1];
    dims[0] = lana_value_number(2.0);
    LanaArray shape = {1, 1, dims};
    args[0] = lana_value_array(&shape);
    args[1] = lana_value_string("f32");
    call_dtype(LANA_HOST_TENSOR_ZEROS, args, 2, "f32");
    call_dtype(LANA_HOST_TENSOR_ONES, args, 2, "f32");
    args[0] = lana_value_number(2.0);
    call_dtype(LANA_HOST_TENSOR_EYE, args, 2, "f32");

    /* f16 rounding of 0.1 -> 0.0999755859375 (round-to-nearest-even). */
    args[0] = arrv;
    args[1] = lana_value_string("f16");
    double r16[] = {0.0999755859375};
    LanaTensor expected16 = {1, &one, &stride, false, LANA_TENSOR_F16, r16, 0, NULL, false};
    call_checked(LANA_HOST_TENSOR, args, 2, LANA_OK, &expected16);

    /* bf16 rounding of 3.14159 -> 3.140625 (mantissa rounded to 7 bits). */
    Value pi_items[1];
    pi_items[0] = lana_value_number(3.14159);
    LanaArray pi_arr = {1, 1, pi_items};
    args[0] = lana_value_array(&pi_arr);
    args[1] = lana_value_string("bf16");
    double rbf[] = {3.140625};
    LanaTensor expected_bf = {1, &one, &stride, false, LANA_TENSOR_BF16, rbf, 0, NULL, false};
    call_checked(LANA_HOST_TENSOR, args, 2, LANA_OK, &expected_bf);

    /* Unknown dtype -> LANA_ERR_INVALID_PARAMETERS on every constructor. */
    args[0] = arrv;
    args[1] = lana_value_string("int8");
    call(LANA_HOST_TENSOR, args, 2, LANA_ERR_INVALID_PARAMETERS);
    args[0] = lana_value_array(&shape);
    call(LANA_HOST_TENSOR_ZEROS, args, 2, LANA_ERR_INVALID_PARAMETERS);
    call(LANA_HOST_TENSOR_ONES, args, 2, LANA_ERR_INVALID_PARAMETERS);
    args[0] = lana_value_number(2.0);
    call(LANA_HOST_TENSOR_EYE, args, 2, LANA_ERR_INVALID_PARAMETERS);

    /* dtype: on a non-tensor constructor (tensor_complex takes no dtype) ->
     * LANA_ERR_TYPE. */
    args[0] = arrv; args[1] = arrv; args[2] = lana_value_string("f32");
    call(LANA_HOST_TENSOR_COMPLEX, args, 3, LANA_ERR_TYPE);

    /* dtype(t) on a non-tensor -> LANA_ERR_TYPE. */
    args[0] = lana_value_number(42.0);
    call(LANA_HOST_TENSOR_DTYPE, args, 1, LANA_ERR_TYPE);
}

/* ===== LIP-008 uncertainty-carrying tensor tests ===== */

/* Build an uncertain tensor map { prediction, uncertainty } with `vm`. */
static Value uncertain_tensor(LanaVM *vm, LanaTensor *pred, LanaTensor *var) {
    LanaMap *map;
    assert(lana_map_new(vm, 2u, &map) == LANA_OK);
    Value pred_value = lana_value_tensor(pred);
    Value var_value = lana_value_tensor(var);
    assert(lana_map_set(vm, map, "prediction", &pred_value, false) == LANA_OK);
    assert(lana_map_set(vm, map, "uncertainty", &var_value, false) == LANA_OK);
    return lana_value_map(map);
}

/* Run a binary host call over uncertain operands and check the resulting
 * { prediction, uncertainty } map. A NULL var means the operand is certain. */
static void call_uncertain_binary(uint32_t host, LanaTensor *a_pred, LanaTensor *a_var,
                                  LanaTensor *b_pred, LanaTensor *b_var,
                                  const LanaTensor *expected_pred, const LanaTensor *expected_var) {
    LanaChunk chunk;
    LanaVM vm;
    lana_chunk_init(&chunk);
    LanaInstruction invoke = {OP_HOST_CALL, 8u, host, 0u, 2u, 1u};
    LanaInstruction halt = {OP_HALT, 0u, 0u, 0u, 0u, 2u};
    assert(lana_chunk_emit(&chunk, invoke) == LANA_OK);
    assert(lana_chunk_emit(&chunk, halt) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    vm.frames[0].registers[0] = a_var ? uncertain_tensor(&vm, a_pred, a_var)
                                      : lana_value_tensor(a_pred);
    vm.frames[0].registers[1] = b_var ? uncertain_tensor(&vm, b_pred, b_var)
                                      : lana_value_tensor(b_pred);
    assert(lana_vm_run(&vm) == LANA_OK);
    Value value = vm.frames[0].registers[8];
    assert(value.type == VAL_MAP);
    Value pred, var;
    assert(lana_map_get(value.as.map, "prediction", &pred) == LANA_OK);
    assert(lana_map_get(value.as.map, "uncertainty", &var) == LANA_OK);
    assert(pred.type == VAL_TENSOR && var.type == VAL_TENSOR);
    assert(tensor_elements_match(pred.as.tensor, expected_pred));
    assert(tensor_elements_match(var.as.tensor, expected_var));
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);
}

/* Like `call_uncertain_binary`, but asserts a specific error code. */
static void call_uncertain_binary_error(uint32_t host, LanaTensor *a_pred, LanaTensor *a_var,
                                        LanaTensor *b_pred, LanaTensor *b_var, LanaError expected) {
    LanaChunk chunk;
    LanaVM vm;
    lana_chunk_init(&chunk);
    LanaInstruction invoke = {OP_HOST_CALL, 8u, host, 0u, 2u, 1u};
    LanaInstruction halt = {OP_HALT, 0u, 0u, 0u, 0u, 2u};
    assert(lana_chunk_emit(&chunk, invoke) == LANA_OK);
    assert(lana_chunk_emit(&chunk, halt) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    vm.frames[0].registers[0] = a_var ? uncertain_tensor(&vm, a_pred, a_var)
                                      : lana_value_tensor(a_pred);
    vm.frames[0].registers[1] = b_var ? uncertain_tensor(&vm, b_pred, b_var)
                                      : lana_value_tensor(b_pred);
    assert(lana_vm_run(&vm) == expected);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);
}

/* Run a reduce host call over an uncertain operand and check the resulting
 * { prediction, uncertainty } map. `axis` NULL means a full reduction. */
static void call_uncertain_reduce(uint32_t host, LanaTensor *pred, LanaTensor *var,
                                  const Value *axis, const LanaTensor *expected_pred,
                                  const LanaTensor *expected_var) {
    LanaChunk chunk;
    LanaVM vm;
    lana_chunk_init(&chunk);
    uint32_t argc = axis ? 2u : 1u;
    LanaInstruction invoke = {OP_HOST_CALL, 8u, host, 0u, argc, 1u};
    LanaInstruction halt = {OP_HALT, 0u, 0u, 0u, 0u, 2u};
    assert(lana_chunk_emit(&chunk, invoke) == LANA_OK);
    assert(lana_chunk_emit(&chunk, halt) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    vm.frames[0].registers[0] = uncertain_tensor(&vm, pred, var);
    if (axis) vm.frames[0].registers[1] = *axis;
    assert(lana_vm_run(&vm) == LANA_OK);
    Value value = vm.frames[0].registers[8];
    assert(value.type == VAL_MAP);
    Value p, v;
    assert(lana_map_get(value.as.map, "prediction", &p) == LANA_OK);
    assert(lana_map_get(value.as.map, "uncertainty", &v) == LANA_OK);
    assert(p.type == VAL_TENSOR && v.type == VAL_TENSOR);
    assert(tensor_elements_match(p.as.tensor, expected_pred));
    assert(tensor_elements_match(v.as.tensor, expected_var));
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);
}

/* Like `call_uncertain_reduce`, but asserts a specific error code. */
static void call_uncertain_reduce_error(uint32_t host, LanaTensor *pred, LanaTensor *var,
                                        const Value *axis, LanaError expected) {
    LanaChunk chunk;
    LanaVM vm;
    lana_chunk_init(&chunk);
    uint32_t argc = axis ? 2u : 1u;
    LanaInstruction invoke = {OP_HOST_CALL, 8u, host, 0u, argc, 1u};
    LanaInstruction halt = {OP_HALT, 0u, 0u, 0u, 0u, 2u};
    assert(lana_chunk_emit(&chunk, invoke) == LANA_OK);
    assert(lana_chunk_emit(&chunk, halt) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    vm.frames[0].registers[0] = uncertain_tensor(&vm, pred, var);
    if (axis) vm.frames[0].registers[1] = *axis;
    assert(lana_vm_run(&vm) == expected);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);
}

/* Run a host call with a two-entry map argument and check the error. */
static void call_map2_error(uint32_t host, const char *key1, Value value1,
                            const char *key2, Value value2, LanaError expected) {
    LanaChunk chunk;
    LanaVM vm;
    lana_chunk_init(&chunk);
    LanaInstruction invoke = {OP_HOST_CALL, 8u, host, 0u, 1u, 1u};
    LanaInstruction halt = {OP_HALT, 0u, 0u, 0u, 0u, 2u};
    assert(lana_chunk_emit(&chunk, invoke) == LANA_OK);
    assert(lana_chunk_emit(&chunk, halt) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    LanaMap *map;
    assert(lana_map_new(&vm, 2u, &map) == LANA_OK);
    assert(lana_map_set(&vm, map, key1, &value1, false) == LANA_OK);
    assert(lana_map_set(&vm, map, key2, &value2, false) == LANA_OK);
    vm.frames[0].registers[0] = lana_value_map(map);
    assert(lana_vm_run(&vm) == expected);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);
}

static void uncertainty_propagation(void) {
    size_t n2 = 2, one = 1;

    /* Element-wise add/sub: var = var_a + var_b. */
    double a_pred[] = {1, 2}, a_var[] = {0.5, 0.25};
    double b_pred[] = {3, 4}, b_var[] = {0.125, 0.0625};
    LanaTensor ta = {1, &n2, &one, false, LANA_TENSOR_F64, a_pred, 0, NULL, false};
    LanaTensor va = {1, &n2, &one, false, LANA_TENSOR_F64, a_var, 0, NULL, false};
    LanaTensor tb = {1, &n2, &one, false, LANA_TENSOR_F64, b_pred, 0, NULL, false};
    LanaTensor vb = {1, &n2, &one, false, LANA_TENSOR_F64, b_var, 0, NULL, false};
    double add_pred[] = {4, 6}, add_var[] = {0.625, 0.3125};
    LanaTensor e_add_pred = {1, &n2, &one, false, LANA_TENSOR_F64, add_pred, 0, NULL, false};
    LanaTensor e_add_var = {1, &n2, &one, false, LANA_TENSOR_F64, add_var, 0, NULL, false};
    call_uncertain_binary(LANA_HOST_TENSOR_ADD, &ta, &va, &tb, &vb, &e_add_pred, &e_add_var);
    double sub_pred[] = {-2, -2};
    LanaTensor e_sub_pred = {1, &n2, &one, false, LANA_TENSOR_F64, sub_pred, 0, NULL, false};
    call_uncertain_binary(LANA_HOST_TENSOR_SUB, &ta, &va, &tb, &vb, &e_sub_pred, &e_add_var);

    /* Mixed certain/uncertain: the certain operand contributes zero variance. */
    double mixed_var[] = {0.125, 0.0625};
    LanaTensor e_mixed_var = {1, &n2, &one, false, LANA_TENSOR_F64, mixed_var, 0, NULL, false};
    call_uncertain_binary(LANA_HOST_TENSOR_ADD, &ta, NULL, &tb, &vb, &e_add_pred, &e_mixed_var);

    /* Mul: var = var_a*b^2 + var_b*a^2. */
    double mul_pred[] = {3, 8}, mul_var[] = {4.625, 4.25};
    LanaTensor e_mul_pred = {1, &n2, &one, false, LANA_TENSOR_F64, mul_pred, 0, NULL, false};
    LanaTensor e_mul_var = {1, &n2, &one, false, LANA_TENSOR_F64, mul_var, 0, NULL, false};
    call_uncertain_binary(LANA_HOST_TENSOR_MUL, &ta, &va, &tb, &vb, &e_mul_pred, &e_mul_var);

    /* Div: var = var_a/b^2 + var_b*a^2/b^4. */
    double d_pred[] = {1, 1}, d_var[] = {1, 1};
    double e_pred[] = {2, 2}, e_var[] = {1, 1};
    LanaTensor td = {1, &n2, &one, false, LANA_TENSOR_F64, d_pred, 0, NULL, false};
    LanaTensor vd = {1, &n2, &one, false, LANA_TENSOR_F64, d_var, 0, NULL, false};
    LanaTensor te = {1, &n2, &one, false, LANA_TENSOR_F64, e_pred, 0, NULL, false};
    LanaTensor ve = {1, &n2, &one, false, LANA_TENSOR_F64, e_var, 0, NULL, false};
    double div_pred[] = {0.5, 0.5}, div_var[] = {0.3125, 0.3125};
    LanaTensor e_div_pred = {1, &n2, &one, false, LANA_TENSOR_F64, div_pred, 0, NULL, false};
    LanaTensor e_div_var = {1, &n2, &one, false, LANA_TENSOR_F64, div_var, 0, NULL, false};
    call_uncertain_binary(LANA_HOST_TENSOR_DIV, &td, &vd, &te, &ve, &e_div_pred, &e_div_var);

    /* Matmul (dot): var = matmul(var_a, b^2) + matmul(a^2, var_b). */
    double dot_pred[] = {11}, dot_var[] = {8.875};
    LanaTensor e_dot_pred = {0, NULL, NULL, false, LANA_TENSOR_F64, dot_pred, 0, NULL, false};
    LanaTensor e_dot_var = {0, NULL, NULL, false, LANA_TENSOR_F64, dot_var, 0, NULL, false};
    call_uncertain_binary(LANA_HOST_TENSOR_MATMUL, &ta, &va, &tb, &vb, &e_dot_pred, &e_dot_var);

    /* Full sum/mean: var = sum(var) and sum(var)/n^2. */
    size_t n4 = 4;
    double s_pred[] = {1, 2, 3, 4}, s_var[] = {0.5, 0.25, 0.125, 0.0625};
    LanaTensor ts = {1, &n4, &one, false, LANA_TENSOR_F64, s_pred, 0, NULL, false};
    LanaTensor vs = {1, &n4, &one, false, LANA_TENSOR_F64, s_var, 0, NULL, false};
    double sum_pred[] = {10}, sum_var[] = {0.9375};
    LanaTensor e_sum_pred = {0, NULL, NULL, false, LANA_TENSOR_F64, sum_pred, 0, NULL, false};
    LanaTensor e_sum_var = {0, NULL, NULL, false, LANA_TENSOR_F64, sum_var, 0, NULL, false};
    call_uncertain_reduce(LANA_HOST_TENSOR_SUM, &ts, &vs, NULL, &e_sum_pred, &e_sum_var);
    double mean_pred[] = {2.5}, mean_var[] = {0.05859375};
    LanaTensor e_mean_pred = {0, NULL, NULL, false, LANA_TENSOR_F64, mean_pred, 0, NULL, false};
    LanaTensor e_mean_var = {0, NULL, NULL, false, LANA_TENSOR_F64, mean_var, 0, NULL, false};
    call_uncertain_reduce(LANA_HOST_TENSOR_MEAN, &ts, &vs, NULL, &e_mean_pred, &e_mean_var);

    /* Axis sum/mean over a 2x2 tensor. */
    size_t s22[] = {2, 2}, st22[] = {2, 1};
    double m_pred[] = {1, 2, 3, 4}, m_var[] = {0.5, 0.25, 0.125, 0.0625};
    LanaTensor tm = {2, s22, st22, false, LANA_TENSOR_F64, m_pred, 0, NULL, false};
    LanaTensor vm2 = {2, s22, st22, false, LANA_TENSOR_F64, m_var, 0, NULL, false};
    Value axis0 = lana_value_number(0);
    double asum_pred[] = {4, 6}, asum_var[] = {0.625, 0.3125};
    LanaTensor e_asum_pred = {1, &n2, &one, false, LANA_TENSOR_F64, asum_pred, 0, NULL, false};
    LanaTensor e_asum_var = {1, &n2, &one, false, LANA_TENSOR_F64, asum_var, 0, NULL, false};
    call_uncertain_reduce(LANA_HOST_TENSOR_SUM, &tm, &vm2, &axis0, &e_asum_pred, &e_asum_var);
    double amean_pred[] = {2, 3}, amean_var[] = {0.15625, 0.078125};
    LanaTensor e_amean_pred = {1, &n2, &one, false, LANA_TENSOR_F64, amean_pred, 0, NULL, false};
    LanaTensor e_amean_var = {1, &n2, &one, false, LANA_TENSOR_F64, amean_var, 0, NULL, false};
    call_uncertain_reduce(LANA_HOST_TENSOR_MEAN, &tm, &vm2, &axis0, &e_amean_pred, &e_amean_var);

    /* max/min reject uncertain operands. */
    call_uncertain_reduce_error(LANA_HOST_TENSOR_MAX, &ts, &vs, NULL, LANA_ERR_TYPE);
    call_uncertain_reduce_error(LANA_HOST_TENSOR_MIN, &ts, &vs, NULL, LANA_ERR_TYPE);

    /* Non-finite uncertainty is InvalidParameters. */
    double inf_var[] = {0.5, INFINITY};
    LanaTensor vinf = {1, &n2, &one, false, LANA_TENSOR_F64, inf_var, 0, NULL, false};
    call_uncertain_binary_error(LANA_HOST_TENSOR_ADD, &ta, &vinf, &tb, &vb,
                                LANA_ERR_INVALID_PARAMETERS);

    /* Complex uncertain operands are Type errors. */
    double c_pred[] = {1, 0, 2, 0}, c_var[] = {0.5, 0, 0.25, 0};
    LanaTensor tc = {1, &n2, &one, true, LANA_TENSOR_COMPLEX, c_pred, 0, NULL, false};
    LanaTensor vc = {1, &n2, &one, true, LANA_TENSOR_COMPLEX, c_var, 0, NULL, false};
    call_uncertain_binary_error(LANA_HOST_TENSOR_ADD, &tc, &vc, &tb, &vb, LANA_ERR_TYPE);

    /* Malformed maps are Type errors: wrong entry count, wrong keys, and
     * non-tensor values. */
    LanaChunk chunk;
    LanaVM vm;
    lana_chunk_init(&chunk);
    LanaInstruction invoke = {OP_HOST_CALL, 8u, LANA_HOST_TENSOR_ADD, 0u, 1u, 1u};
    LanaInstruction halt = {OP_HALT, 0u, 0u, 0u, 0u, 2u};
    assert(lana_chunk_emit(&chunk, invoke) == LANA_OK);
    assert(lana_chunk_emit(&chunk, halt) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    LanaMap *map1;
    assert(lana_map_new(&vm, 1u, &map1) == LANA_OK);
    Value pv = lana_value_tensor(&ta);
    assert(lana_map_set(&vm, map1, "prediction", &pv, false) == LANA_OK);
    vm.frames[0].registers[0] = lana_value_map(map1);
    assert(lana_vm_run(&vm) == LANA_ERR_TYPE);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);

    call_map2_error(LANA_HOST_TENSOR_ADD, "prediction", lana_value_tensor(&ta),
                    "variance", lana_value_tensor(&va), LANA_ERR_TYPE);
    call_map2_error(LANA_HOST_TENSOR_ADD, "prediction", lana_value_tensor(&ta),
                    "uncertainty", lana_value_number(1.0), LANA_ERR_TYPE);
}

int main(void) {
    axis_reductions();
    indexing();
    matmul();
    gpu_matmul();
    dtype_construction();
    uncertainty_propagation();
    Value dims[33];
    LanaArray shape = {0};
    shape.items = dims;
    shape.count = shape.capacity = 1u;
    Value args[2] = {{.type = VAL_ARRAY, .as.array = &shape}, {0}};
    const double bad[] = {NAN, INFINITY, -INFINITY, -1.0, 0.5, 0x1p64};
    for (size_t i = 0u; i < sizeof(bad) / sizeof(bad[0]); ++i) {
        dims[0] = lana_value_number(bad[i]);
        call(LANA_HOST_TENSOR_ZEROS, args, 1u, LANA_ERR_INVALID_PARAMETERS);
        call(LANA_HOST_TENSOR_ONES, args, 1u, LANA_ERR_INVALID_PARAMETERS);
        call(LANA_HOST_TENSOR_EYE, dims, 1u, LANA_ERR_INVALID_PARAMETERS);
    }
    for (size_t i = 0u; i < 33u; ++i) dims[i] = lana_value_number(1.0);
    shape.count = 32u;
    call(LANA_HOST_TENSOR_ZEROS, args, 1u, LANA_OK);
    shape.count = 33u;
    call(LANA_HOST_TENSOR_ZEROS, args, 1u, LANA_ERR_INVALID_PARAMETERS);
    shape.count = 1u;
    dims[0] = lana_value_number(0x1p61);
    call(LANA_HOST_TENSOR_ZEROS, args, 1u, LANA_ERR_OOM);
    args[1] = lana_value_bool(true);
    call(LANA_HOST_TENSOR_ALLOC, args, 2u, LANA_ERR_OOM);
    dims[0] = args[0]; /* cyclic constructor input */
    call(LANA_HOST_TENSOR, args, 1u, LANA_ERR_INVALID_PARAMETERS);

    double data[4] = {1.0, INFINITY, 2.0, 0.0};
    size_t size = 2u, stride = 1u;
    LanaTensor tensor = {1u, &size, &stride, true, LANA_TENSOR_COMPLEX, data, 0, NULL, false};
    args[0] = lana_value_tensor(&tensor);
    call(LANA_HOST_TENSOR_SUM, args, 1u, LANA_ERR_INVALID_PARAMETERS);
    data[0] = data[2] = DBL_MAX;
    data[1] = 0.0;
    call(LANA_HOST_TENSOR_SUM, args, 1u, LANA_ERR_INVALID_PARAMETERS);
    tensor.is_complex = false;
    data[1] = DBL_MAX;
    call(LANA_HOST_TENSOR_SUM, args, 1u, LANA_ERR_INVALID_PARAMETERS);
    size = 0u;
    call(LANA_HOST_TENSOR_SUM, args, 1u, LANA_OK);
    call(LANA_HOST_TENSOR_MEAN, args, 1u, LANA_ERR_INVALID_PARAMETERS);
    call(LANA_HOST_TENSOR_MIN, args, 1u, LANA_ERR_INVALID_PARAMETERS);
    call(LANA_HOST_TENSOR_MAX, args, 1u, LANA_ERR_INVALID_PARAMETERS);
    (void)puts("TENSOR_BOUNDARIES_PASS");
    return 0;
}
