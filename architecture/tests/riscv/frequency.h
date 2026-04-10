// RISC-V-specific frequency measurement implementation

#ifndef RISCV_FREQUENCY_H
#define RISCV_FREQUENCY_H

#include <stdint.h>
#include <time.h>

static inline void serialize(void)
{
    // RISC-V: fence for serialization
    asm volatile("fence" ::: "memory");
}

static inline void measure_clock_ticks_start(void)
{
    // RISC-V uses clock_gettime, no additional preparation needed
}

static inline void measure_clock_ticks_end(void)
{
    // RISC-V uses clock_gettime, no additional preparation needed
}

// RISC-V does not support nominal frequency
static inline int has_nominal_frequency(void)
{
    return 0;
}

// Calculate real frequency only (no TSC on RISC-V)
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

    // No nominal frequency on RISC-V
    *freq_nominal_out = 0.0f;
}

// RISC-V benchmark function: inline assembly clock test
static void bench_function(uint64_t iterations)
{
    register uint64_t acc = 0;
    register uint64_t inc = 1;
    register uint64_t dec = 20;

    // clang-format off
    asm volatile(
        "1:\n\t"
        "add %0, %0, %3\n\t"
        "add %0, %0, %3\n\t"
        "add %0, %0, %3\n\t"
        "add %0, %0, %3\n\t"
        "add %0, %0, %3\n\t"
        "add %0, %0, %3\n\t"
        "add %0, %0, %3\n\t"
        "add %0, %0, %3\n\t"
        "add %0, %0, %3\n\t"
        "add %0, %0, %3\n\t"
        "add %0, %0, %3\n\t"
        "add %0, %0, %3\n\t"
        "add %0, %0, %3\n\t"
        "add %0, %0, %3\n\t"
        "add %0, %0, %3\n\t"
        "add %0, %0, %3\n\t"
        "add %0, %0, %3\n\t"
        "add %0, %0, %3\n\t"
        "add %0, %0, %3\n\t"
        "add %0, %0, %3\n\t"
        "sub %1, %1, %2\n\t"
        "bnez %1, 1b\n\t"
        : "+r"(acc), "+r"(iterations)
        : "r"(dec), "r"(inc)
        : "memory"
    );
    // clang-format on
}

#endif
