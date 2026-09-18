// LIP-004 §5: explicit GPU matmul. A single deterministic float32 sgemm
// compute kernel, one thread per output element, fixed reduction order. Both
// the C11 VM (vm/c/metal.m) and the Rust VM (vm/rust/lana-vm/src/metal.rs)
// compile this same source, so a given device produces byte-identical results
// from both VMs. Binary64 inputs are downcast to float32 by the caller before
// dispatch and upcast after — never silently.
#include <metal_stdlib>
using namespace metal;

kernel void sgemm(const device float* A [[buffer(0)]],
                  const device float* B [[buffer(1)]],
                  device float* C [[buffer(2)]],
                  constant uint& M [[buffer(3)]],
                  constant uint& K [[buffer(4)]],
                  constant uint& N [[buffer(5)]],
                  uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.x;
    uint col = gid.y;
    if (row >= M || col >= N) {
        return;
    }
    float acc = 0.0f;
    for (uint p = 0; p < K; ++p) {
        acc += A[row * K + p] * B[p * N + col];
    }
    C[row * N + col] = acc;
}
