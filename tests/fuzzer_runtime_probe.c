// Runtime qualification controls. Each failure mode must still be detected.
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    (void)data;
    (void)size;
    const char *mode = getenv("LANA_FUZZ_PROBE");
    if (!mode) return 0;
    if (strcmp(mode, "crash") == 0) abort();
    if (strcmp(mode, "timeout") == 0) sleep(30);
    if (strcmp(mode, "leak") == 0) {
        volatile unsigned char *lost = malloc(12345);
        if (!lost) abort();
        lost[0] = 1;
    }
    if (strcmp(mode, "rss") == 0) {
        size_t bytes = 256 * 1024 * 1024;
        volatile unsigned char *held = malloc(bytes);
        if (!held) abort();
        for (size_t i = 0; i < bytes; ++i) held[i] = 1;
        sleep(5);
        free((void *)held);
    }
    return 0;
}
