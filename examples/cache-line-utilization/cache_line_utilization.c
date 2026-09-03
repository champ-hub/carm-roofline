#include <errno.h>
#include <limits.h>
#include <math.h>
#include <papi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CACHE_LINE_BYTES 64U
#define FLOATS_PER_LINE (CACHE_LINE_BYTES / sizeof(float))

static void usage(const char *program)
{
    fprintf(stderr, "Usage: %s CACHE_SIZE_KIB TARGET_CLU_PERCENT [PASSES]\n", program);
}

static unsigned reverse_bits(unsigned value, unsigned bits)
{
    unsigned reversed = 0;
    for (unsigned bit = 0; bit < bits; ++bit) {
        reversed = (reversed << 1U) | (value & 1U);
        value >>= 1U;
    }
    return reversed;
}

static int parse_double(const char *text, double *value)
{
    char *end = NULL;
    errno = 0;
    *value = strtod(text, &end);
    return errno == 0 && end != text && *end == '\0' && isfinite(*value);
}

static int parse_size(const char *text, size_t *value)
{
    char *end = NULL;
    unsigned long long parsed;
    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed > SIZE_MAX) {
        return 0;
    }
    *value = (size_t) parsed;
    return 1;
}

int main(int argc, char **argv)
{
    size_t cache_kib, passes = 100, line_count = 1, work_bytes, total_operations, base_operations, remainder;
    double target;
    float *data;
    volatile float *kernel_data;
    double checksum = 0.0;
    int papi_result;
    unsigned bits = 0;

    if (argc < 3 || argc > 4 || !parse_size(argv[1], &cache_kib) || !parse_double(argv[2], &target) ||
        (argc == 4 && !parse_size(argv[3], &passes))) {
        usage(argv[0]);
        return 1;
    }
    if (cache_kib == 0 || target < 12.5 || passes == 0) {
        fprintf(stderr,
                "Error: cache size must be nonzero, target must be at least 12.5, and passes must be positive.\n");
        return 1;
    }
    if (cache_kib > SIZE_MAX / 4096U) {
        fprintf(stderr, "Error: cache size causes allocation overflow.\n");
        return 1;
    }
    work_bytes = cache_kib * 4096U;
    while (line_count < work_bytes / CACHE_LINE_BYTES) {
        if (line_count > SIZE_MAX / 2U) {
            fprintf(stderr, "Error: cache size causes allocation overflow.\n");
            return 1;
        }
        line_count <<= 1U;
        ++bits;
    }
    if (line_count > SIZE_MAX / CACHE_LINE_BYTES || target > (double) SIZE_MAX * 12.5 / (double) line_count) {
        fprintf(stderr, "Error: operation count causes overflow.\n");
        return 1;
    }
    work_bytes = line_count * CACHE_LINE_BYTES;
    total_operations = (size_t) ((double) line_count * target / 12.5 + 0.5);
    base_operations = total_operations / line_count;
    remainder = total_operations % line_count;
    if (posix_memalign((void **) &data, CACHE_LINE_BYTES, work_bytes) != 0) {
        fprintf(stderr, "Error: allocation failed.\n");
        return 1;
    }
    for (size_t i = 0; i < work_bytes / sizeof(*data); ++i) {
        data[i] = (float) i * 0.001f;
    }
    papi_result = PAPI_library_init(PAPI_VER_CURRENT);
    if (papi_result != PAPI_VER_CURRENT) {
        fprintf(stderr, "Error: PAPI initialization failed: %s\n", PAPI_strerror(papi_result));
        free(data);
        return 1;
    }
    papi_result = PAPI_hl_region_begin("cache_line_utilization");
    if (papi_result != PAPI_OK) {
        fprintf(stderr, "Error: PAPI region begin failed: %s\n", PAPI_strerror(papi_result));
        free(data);
        return 1;
    }
    kernel_data = data;
    for (size_t pass = 0; pass < passes; ++pass) {
        for (size_t logical = 0; logical < line_count; ++logical) {
            size_t line = reverse_bits((unsigned) logical, bits);
            size_t operations = base_operations + (logical < remainder);
            for (size_t op = 0; op < operations; ++op) {
                volatile float *element = &kernel_data[line * FLOATS_PER_LINE + op % FLOATS_PER_LINE];
                float value = *element;
                *element = value * 1.0001f;
            }
        }
    }
    papi_result = PAPI_hl_region_end("cache_line_utilization");
    if (papi_result != PAPI_OK) {
        fprintf(stderr, "Error: PAPI region end failed: %s\n", PAPI_strerror(papi_result));
        free(data);
        return 1;
    }
    for (size_t i = 0; i < work_bytes / sizeof(*data); ++i) {
        checksum += data[i];
    }
    printf("cache size: %zu KiB\nworking set: %zu KiB\ntarget: %.1f%%\npasses: %zu\noperations: %zu\nchecksum: %.9g\n",
           cache_kib,
           work_bytes / 1024U,
           target,
           passes,
           total_operations,
           checksum);
    free(data);
    return 0;
}
