/* External driver: compile against each pinned release, never into Lana. */
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <libproc.h>
#include <sys/resource.h>
#include <os/signpost.h>
#ifdef LANA_OLD_HEADERS
#include "lana/vm.h"
#else
#include "vm.h"
#endif

static uint64_t cycles(void) {
    struct rusage_info_v4 usage = {0};
    if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *)&usage) != 0) return 0;
    return usage.ri_cycles;
}

static uint64_t nanos(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) exit(1);
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec;
}

static unsigned long integer(const char *text) {
    char *end;
    errno = 0;
    unsigned long value = strtoul(text, &end, 10);
    if (errno || end == text || *end || *text == '-' || value > 1000000) exit(2);
    return value;
}

int main(int argc, char **argv) {
    if (argc != 6) { fprintf(stderr, "usage: driver bytecode expected batches seed sample\n"); return 2; }
    unsigned long batches = integer(argv[3]), seed = integer(argv[4]), sample = integer(argv[5]);
    if (!batches || sample > 1) return 2;
    double expected[256][3], last[256][3], sum_re = 0, sum_im = 0;
    int types[256];
    FILE *input = fopen(argv[2], "r");
    if (!input) return 2;
    for (size_t i = 0; i < 256; i++) {
        if (fscanf(input, "%lf %lf %lf %d", &expected[i][0], &expected[i][1], &expected[i][2], &types[i]) != 4) return 2;
    }
    fclose(input);
    LanaChunk chunk;
    LanaErrorInfo error = {0};
    if (lana_chunk_read_file(&chunk, argv[1], &error) != LANA_OK) return 1;
    LanaVM *vm = malloc(sizeof(*vm));
    if (!vm) return 1;
    uint64_t elapsed = 0, count = 0, allocations = 0, retained_bytes = 0;
    bool have_cycles = true;
    os_log_t log = os_log_create("org.lana.version-metrics", OS_LOG_CATEGORY_POINTS_OF_INTEREST);
    for (unsigned long batch = 0; batch < batches; batch++) {
        os_signpost_interval_begin(log, 1, "Lana execution", "%s", "");
        uint64_t c0 = cycles(), t0 = nanos();
        lana_vm_init(vm, &chunk);
        lana_vm_seed(vm, seed + batch);
        lana_vm_set_memory_limit(vm, 256u * 1024u * 1024u);
        vm->instruction_limit = UINT64_C(50000000);
        LanaError status = lana_vm_run(vm);
        uint64_t t1 = nanos(), c1 = cycles();
        os_signpost_interval_end(log, 1, "Lana execution", "%s", "");
        elapsed += t1 - t0;
        if (!c0 || c1 <= c0) have_cycles = false; else count += c1 - c0;
        allocations += vm->allocation_count;
        retained_bytes += vm->allocated_bytes;
        bool valid = status == LANA_OK && vm->result.type == VAL_ARRAY && vm->result.as.array->count == 256;
        for (size_t i = 0; valid && i < 256; i++) {
            Value *value = &vm->result.as.array->items[i];
            double p = 0, re = 0, im = 0;
            if (value->type == VAL_NUMBER) p = value->as.number;
            else if (value->type == VAL_STATE) {
                p = value->as.state.state.p;
                re = value->as.state.state.d_re;
                im = value->as.state.state.d_im;
            } else if (value->type == VAL_STATE_DIST && !sample) {
                /* Force only for validation, outside both measured intervals. */
                valid = lana_vm_state_dist_expected_probability(value->as.state_dist, &p) == LANA_OK;
            } else valid = false;
            valid = valid && (int)value->type == types[i] && isfinite(p) && isfinite(re) && isfinite(im) &&
                fabs(p - expected[i][0]) <= 1e-12 &&
                (sample ? (value->type == VAL_STATE && p >= 0 && p <= 1 && re*re + im*im <= 1 + 1e-12 &&
                           ((p != 0 && p != 1) || (re == 0 && im == 0))) :
                          (fabs(re - expected[i][1]) <= 1e-12 && fabs(im - expected[i][2]) <= 1e-12));
            last[i][0] = p; last[i][1] = re; last[i][2] = im;
            sum_re += re; sum_im += im;
            if (!valid) fprintf(stderr, "row %zu: type %d expected %d; values %.17g %.17g %.17g expected %.17g %.17g %.17g\n",
                                i, value->type, types[i], p, re, im, expected[i][0], expected[i][1], expected[i][2]);
        }
        os_signpost_interval_begin(log, 2, "Lana cleanup", "%s", "");
        c0 = cycles(); t0 = nanos();
        lana_vm_free(vm);
        t1 = nanos(); c1 = cycles();
        os_signpost_interval_end(log, 2, "Lana cleanup", "%s", "");
        elapsed += t1 - t0;
        if (!c0 || c1 <= c0) have_cycles = false; else count += c1 - c0;
        if (!valid) {
            fprintf(stderr, "incorrect/incomplete batch %lu: runtime status %d\n", batch, status);
            free(vm); lana_chunk_free(&chunk); return 1;
        }
    }
    printf("{\"units\":%lu,\"elapsed_ns\":%llu,\"cycles\":", batches * 256,
           (unsigned long long)elapsed);
    if (have_cycles) printf("%llu", (unsigned long long)count); else printf("null");
    printf(",\"allocations\":%llu,\"retained_bytes\":%llu,\"mean_re\":%.17g,\"mean_im\":%.17g,\"last\":[",
           (unsigned long long)allocations, (unsigned long long)retained_bytes,
           sum_re/(batches*256), sum_im/(batches*256));
    for (size_t i = 0; i < 256; i++) printf("%s[%.17g,%.17g,%.17g]", i ? "," : "", last[i][0], last[i][1], last[i][2]);
    printf("]}\n");
    free(vm); lana_chunk_free(&chunk);
    return 0;
}
