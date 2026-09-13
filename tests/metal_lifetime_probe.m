// Standalone platform diagnostic: no Lana code or Rust runtime is linked.
// clang -fsanitize=address -framework Foundation -framework Metal \
//   tests/metal_lifetime_probe.m -o /tmp/lana-metal-probe
// ASAN_OPTIONS=detect_leaks=1 /tmp/lana-metal-probe
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdlib.h>
#include <string.h>

static void run_deliberate_leak(void) {
    static void *allocation;
    for (int i = 0; i < 16; ++i) {
        allocation = malloc(12345);
        if (!allocation) abort();
        ((unsigned char *)allocation)[0] = 1;
    }
    allocation = NULL;
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--clean-control") == 0) return 0;
    if (argc == 2 && strcmp(argv[1], "--leak-control") == 0) {
        run_deliberate_leak();
        return 0;
    }
    if (argc != 1) return 2;
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return 77;
        for (int i = 0; i < 64; ++i) {
            @autoreleasepool {
                id<MTLCommandQueue> queue = [device newCommandQueue];
                id<MTLCommandBuffer> command = [queue commandBuffer];
                [command commit];
                [command waitUntilCompleted];
                [queue release];
            }
        }
        [device release];
    }
    return 0;
}
