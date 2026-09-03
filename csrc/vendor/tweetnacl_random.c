#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

void randombytes(unsigned char *buffer, unsigned long long length) {
#ifdef __APPLE__
    if (length > SIZE_MAX) abort();
    arc4random_buf(buffer, (size_t)length);
#else
    int file = open("/dev/urandom", O_RDONLY);
    size_t offset = 0u;
    if (file < 0) abort();
    while (offset < length) {
        ssize_t read_count = read(file, buffer + offset, (size_t)(length - offset));
        if (read_count <= 0) { (void)close(file); abort(); }
        offset += (size_t)read_count;
    }
    (void)close(file);
#endif
}
