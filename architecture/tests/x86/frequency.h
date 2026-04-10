// x86-specific frequency measurement implementation

#ifndef X86_FREQUENCY_H
#define X86_FREQUENCY_H

#include <stdint.h>
#include <string.h>
#include <time.h>

static inline void serialize(void)
{
    asm volatile("lfence;" : : :);
}

static inline uint64_t read_tsc_start(void)
{
    uint64_t d, a;
    // clang-format off
    asm __volatile__(
        "lfence;"
        "rdtsc;"
        "movq %%rdx, %0;"
        "movq %%rax, %1;"
        : "=r"(d), "=r"(a)
        :
        : "%rax", "%rdx"
    );
    // clang-format on
    return ((uint64_t) d << 32 | a);
}

static inline uint64_t read_tsc_end(void)
{
    uint64_t d, a;
    // clang-format off
    asm __volatile__(
        "rdtscp;"
        "movq %%rdx, %0;"
        "movq %%rax, %1;"
        "lfence;"
        : "=r"(d), "=r"(a)
        :
        : "%rax", "%rdx"
    );
    // clang-format on
    return ((uint64_t) d << 32 | a);
}

static volatile uint64_t tsc_start_val;
static volatile uint64_t tsc_end_val;

static inline void measure_clock_ticks_start(void)
{
    tsc_start_val = read_tsc_start();
}

static inline void measure_clock_ticks_end(void)
{
    tsc_end_val = read_tsc_end();
}

// x86 supports nominal frequency via TSC
static inline int has_nominal_frequency(void)
{
    return 1;
}

// Calculate both real and nominal frequencies
static inline void calculate_frequencies(
        uint64_t iterations,
        struct timespec *t_start,
        struct timespec *t_end,
        float *freq_real_out,
        float *freq_nominal_out)
{
    // Calculate elapsed time in milliseconds
    uint64_t time_ms = 1000 * (t_end->tv_sec - t_start->tv_sec) + ((t_end->tv_nsec - t_start->tv_nsec) / 1000000);

    // Real frequency from wall-clock time (minimizes floating point error)
    // freq_ghz = iterations / (time_ms * 1e6)
    *freq_real_out = (float) iterations / ((float) time_ms * 1e6f);

    // Nominal frequency from TSC
    uint64_t tsc_cycles = tsc_end_val - tsc_start_val;
    // freq_ghz = tsc_cycles / (time_ms * 1e6)
    *freq_nominal_out = (float) tsc_cycles / ((float) time_ms * 1e6f);
}

// x86 benchmark function: inline assembly clock test
static void bench_function(uint64_t iterations)
{
    // clang-format off
    asm volatile(
        "movq $1, %%r8\n\t"
        "movq $20, %%r9\n\t"
        "xorq %%rbx, %%rbx\n\t"
        "clktest_loop:\n\t"
        "addq %%r8, %%rbx\n\t"
        "addq %%r8, %%rbx\n\t"
        "addq %%r8, %%rbx\n\t"
        "addq %%r8, %%rbx\n\t"
        "addq %%r8, %%rbx\n\t"
        "addq %%r8, %%rbx\n\t"
        "addq %%r8, %%rbx\n\t"
        "addq %%r8, %%rbx\n\t"
        "addq %%r8, %%rbx\n\t"
        "addq %%r8, %%rbx\n\t"
        "addq %%r8, %%rbx\n\t"
        "addq %%r8, %%rbx\n\t"
        "addq %%r8, %%rbx\n\t"
        "addq %%r8, %%rbx\n\t"
        "addq %%r8, %%rbx\n\t"
        "addq %%r8, %%rbx\n\t"
        "addq %%r8, %%rbx\n\t"
        "addq %%r8, %%rbx\n\t"
        "addq %%r8, %%rbx\n\t"
        "addq %%r8, %%rbx\n\t"
        "subq %%r9, %0\n\t"
        "jnz clktest_loop\n\t"
        : "+r"(iterations)
        :
        : "%rbx", "%r8", "%r9", "cc", "memory"
    );
    // clang-format on
}

#endif
