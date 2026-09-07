#ifndef LANA_TENSOR_H
#define LANA_TENSOR_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

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
 *   dtype  – LIP-027 numeric dtype (F64 default); determines element width.
 *   data   – contiguous byte buffer of `prod(shape) * element_width` bytes.
 *            For a complex tensor each element is 16 bytes (real, imag).
 *            Element access goes through the tensor_get/set helpers in vm.c,
 *            which convert between the storage dtype and double.
 *   offset – index of the first element (in elements; 0 for a base tensor).
 *   base   – the source tensor when this tensor is a view, NULL for a base
 *            tensor. The GC marks the whole base chain so a shared buffer
 *            outlives every view of it.
 *
 * The base buffer size is prod(shape) * element_width, where element_width is
 * 8 for f64, 4 for f32, 2 for f16/bf16, and 16 for complex. A view shares that
 * buffer; its shape/strides/offset select a subset, and strides may be
 * non‑contiguous. Element access always goes through offset and strides.
 */

typedef struct LanaTensor {
    size_t ndim;
    size_t *shape;      // length ndim, owned by the tensor (malloc/free)
    size_t *strides;    // length ndim, row‑major stride for each dimension
    bool is_complex;
    LanaTensorDtype dtype; // LIP-027: numeric dtype (F64 default)
    uint8_t *data;      // length prod(shape) * element_width bytes
    size_t offset;      // first element, in elements (0 for a base tensor)
    struct LanaTensor *base; // source tensor for a view, NULL for a base tensor
    bool is_state;      // LIP-007: a STATE tensor (each element is a density matrix)
} LanaTensor;

/* Helper to compute row‑major strides from shape. Caller must allocate
 * space for `strides` (size ndim). Returns 0 on success, non‑zero on overflow. */
int lana_tensor_compute_strides(const size_t *shape, size_t ndim, size_t *strides);

/* LIP-027: element accessors. `i` is the absolute element index (already
 * including `offset`). `tensor_get_real` returns the real part of element `i`
 * converted to double; `tensor_get_imag` returns the imaginary part (0.0 for a
 * real dtype). Defined in vm.c. */
double tensor_get_real(const LanaTensor *t, size_t i);
double tensor_get_imag(const LanaTensor *t, size_t i);

#endif // LANA_TENSOR_H
