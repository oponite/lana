#define _POSIX_C_SOURCE 200809L

#include "lana/bytecode.h"

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    char path[] = "/tmp/lana-labc-fuzz-XXXXXX";
    LanaChunk chunk;
    LanaErrorInfo error = {0};
    size_t written = 0u;
    int fd = mkstemp(path);
    if (fd < 0) return 0;
    while (written < size) {
        ssize_t count = write(fd, data + written, size - written);
        if (count < 0) {
            if (errno == EINTR) continue;
            (void)close(fd);
            (void)unlink(path);
            return 0;
        }
        written += (size_t)count;
    }
    if (close(fd) != 0) {
        (void)unlink(path);
        return 0;
    }
    if (lana_chunk_read_file(&chunk, path, &error) == LANA_OK) {
        if (lana_chunk_verify(&chunk, &error) != LANA_OK) abort();
        lana_chunk_free(&chunk);
    }
    (void)unlink(path);
    return 0;
}
