// Explicit GPU matmul (LIP-004 section 5). A single float32 sgemm on the
// system Metal device. The shader source below must stay byte-identical to
// vm/metal/matmul.metal (the Rust VM embeds that file via include_str!), so
// both VMs run the same kernel and produce identical results on a device.
#import <Metal/Metal.h>
#include <stdint.h>
#include <string.h>
#include "metal.h"

static const char *kSgemmSource =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "\n"
    "kernel void sgemm(const device float* A [[buffer(0)]],\n"
    "                  const device float* B [[buffer(1)]],\n"
    "                  device float* C [[buffer(2)]],\n"
    "                  constant uint& M [[buffer(3)]],\n"
    "                  constant uint& K [[buffer(4)]],\n"
    "                  constant uint& N [[buffer(5)]],\n"
    "                  uint2 gid [[thread_position_in_grid]]) {\n"
    "    uint row = gid.x;\n"
    "    uint col = gid.y;\n"
    "    if (row >= M || col >= N) { return; }\n"
    "    float acc = 0.0f;\n"
    "    for (uint p = 0; p < K; ++p) {\n"
    "        acc += A[row * K + p] * B[p * N + col];\n"
    "    }\n"
    "    C[row * N + col] = acc;\n"
    "}\n";

/* The device and compiled pipeline are cached across calls (the VM is
 * single-threaded, so no locking is needed). The command queue and buffers
 * are per-call. */
static id<MTLDevice> sDevice = nil;
static id<MTLComputePipelineState> sPipeline = nil;

static bool metal_pipeline_ready(void) {
    if (sPipeline != nil) {
        return true;
    }
    sDevice = MTLCreateSystemDefaultDevice();
    if (sDevice == nil) {
        return false;
    }
    NSError *error = nil;
    NSString *source = [NSString stringWithUTF8String:kSgemmSource];
    id<MTLLibrary> library = [sDevice newLibraryWithSource:source
                                                  options:nil
                                                    error:&error];
    if (library == nil) {
        return false;
    }
    id<MTLFunction> function = [library newFunctionWithName:@"sgemm"];
    if (function == nil) {
        return false;
    }
    sPipeline = [sDevice newComputePipelineStateWithFunction:function
                                                      error:&error];
    return sPipeline != nil;
}

bool lana_metal_sgemm(size_t m, size_t k, size_t n,
                      const float *a, const float *b, float *c) {
    if (m == 0u || n == 0u) {
        return true; /* nothing to write */
    }
    if (m > UINT32_MAX || k > UINT32_MAX || n > UINT32_MAX) {
        return false;
    }
    if (!metal_pipeline_ready()) {
        return false;
    }
    id<MTLCommandQueue> queue = [sDevice newCommandQueue];
    if (queue == nil) {
        return false;
    }
    size_t a_bytes = m * k * sizeof(float);
    size_t b_bytes = k * n * sizeof(float);
    size_t c_bytes = m * n * sizeof(float);
    id<MTLBuffer> bufA = [sDevice newBufferWithBytes:a
                                             length:a_bytes
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [sDevice newBufferWithBytes:b
                                             length:b_bytes
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufC = [sDevice newBufferWithLength:c_bytes
                                             options:MTLResourceStorageModeShared];
    uint32_t dims[3] = {(uint32_t)m, (uint32_t)k, (uint32_t)n};
    id<MTLBuffer> bufDims = [sDevice newBufferWithBytes:dims
                                                 length:sizeof(dims)
                                                options:MTLResourceStorageModeShared];
    if (bufA == nil || bufB == nil || bufC == nil || bufDims == nil) {
        return false;
    }

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:sPipeline];
    [encoder setBuffer:bufA offset:0 atIndex:0];
    [encoder setBuffer:bufB offset:0 atIndex:1];
    [encoder setBuffer:bufC offset:0 atIndex:2];
    [encoder setBuffer:bufDims offset:0 atIndex:3];
    [encoder setBuffer:bufDims offset:4 atIndex:4];
    [encoder setBuffer:bufDims offset:8 atIndex:5];

    MTLSize threadsPerThreadgroup = MTLSizeMake(16, 16, 1);
    MTLSize threadgroups = MTLSizeMake((m + 15u) / 16u, (n + 15u) / 16u, 1);
    [encoder dispatchThreadgroups:threadgroups
            threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    if (commandBuffer.status == MTLCommandBufferStatusError) {
        return false;
    }
    memcpy(c, bufC.contents, c_bytes);
    return true;
}
