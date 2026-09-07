#include "tensor.h"
#include <stdlib.h>
#include <limits.h>
#include <unistd.h>

int lana_tensor_compute_strides(const size_t *shape, size_t ndim, size_t *strides) {
    if (ndim == 0) {
        return 0; // scalar has no strides
    }
    size_t stride = 1;
    // Compute from last dimension backwards (row‑major)
    for (ssize_t i = (ssize_t)ndim - 1; i >= 0; --i) {
        strides[i] = stride;
        // Guard against overflow: stride * shape[i] must not exceed SIZE_MAX.
        if (shape[i] != 0 && stride > SIZE_MAX / shape[i]) {
            return -1; // overflow
        }
        stride *= shape[i];
    }
    return 0;
}
