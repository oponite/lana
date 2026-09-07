#ifndef LANA_TENSOR_H
#define LANA_TENSOR_H

#include <stddef.h>
#include <stdbool.h>

#define LANA_TENSOR_MAX_RANK 32u

/* LIP-027: tensor numeric dtype. `is_complex` is derived from this: a
 * COMPLEX tensor has `is_complex == true`, every other dtype has it false.
 * The default is F64, so existing programs are unchanged. */
typedef enum LanaTensorDtype {
    LANA_TENSOR_F64 = 0,
    LANA_TENSOR_F32 = 1,
    LANA_TENSOR_F16 = 2,
    LANA_TENSOR_BF16 = 3,
    LANA_TENSOR_COMPLEX = 4
} LanaTensorDtype;

/* LIP‑004: First‑class tensor.
 *   ndim   – number of dimensions (0 for scalar).
 *   shape  – array of dimension lengths.
 *   strides – row‑major strides (computed from shape).
 *   is_complex – true for complex tensors (each element stores real & imag).
 *   data   – contiguous buffer of double values (real interleaved with imag if complex).
 *   offset – index of the first element (in elements; 0 for a base tensor).
 *   base   – the source tensor when this tensor is a view, NULL for a base
 *            tensor. The GC marks the whole base chain so a shared buffer
 *            outlives every view of it.
 *
 * The base buffer size is prod(shape) * (is_complex ? 2 : 1). A view shares
 * that buffer; its shape/strides/offset select a subset, and strides may be
 * non‑contiguous. Element access always goes through offset and strides.
 */

typedef struct LanaTensor {
    size_t ndim;
    size_t *shape;      // length ndim, owned by the tensor (malloc/free)
    size_t *strides;    // length ndim, row‑major stride for each dimension
    bool is_complex;
    LanaTensorDtype dtype; // LIP-027: numeric dtype (F64 default)
    double *data;       // length prod(shape) * (is_complex ? 2 : 1)
    size_t offset;      // first element, in elements (0 for a base tensor)
    struct LanaTensor *base; // source tensor for a view, NULL for a base tensor
    bool is_state;      // LIP-007: a STATE tensor (each element is a density matrix)
} LanaTensor;

/* Helper to compute row‑major strides from shape. Caller must allocate
 * space for `strides` (size ndim). Returns 0 on success, non‑zero on overflow. */
int lana_tensor_compute_strides(const size_t *shape, size_t ndim, size_t *strides);

#endif // LANA_TENSOR_H
