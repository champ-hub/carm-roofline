#ifndef BENCHMARK_SUITE_H
#define BENCHMARK_SUITE_H

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define WRAPPER_NAME(ubench_name) wrapper_##ubench_name

#ifndef NUM_RUNS
#define NUM_RUNS 1024
#endif

#ifndef EXPECTED_TIME_NS
#define EXPECTED_TIME_NS 100000000 /* 100ms in nanoseconds */
#endif

#if defined(__riscv)
#define CARM_ARCH_RISCV 1
#elif defined(__aarch64__) || defined(__arm__)
#define CARM_ARCH_ARM 1
#elif defined(__x86_64__) || defined(__i386__)
#define CARM_ARCH_X86 1
#else
#define CARM_ARCH_UNKNOWN 1
#endif

/* IMPORTANT NOTE
If an ISA is set to not use wall-clock time here, it MUST return a nominal frequency in the architeture module tests.
These two must match: if a nominal frequency is detected, it is fed to the this benchmark, which must use TSC. If no
nominal frequency is detected, this benchmark must use wall-clock time. If there is a mismatch, the benchmark will
produce incorrect results.
*/
#if defined(CARM_ARCH_ARM) || defined(CARM_ARCH_RISCV)
#define CARM_BENCH_USE_WALLTIME 1
#else
#define CARM_BENCH_USE_WALLTIME 0
#endif

/**
 * Thread data structure used by wrapper functions
 */
typedef struct {
    int tid;
    void *read_ptr;  /* Per-thread slice of the read (load) buffer */
    void *write_ptr; /* Per-thread slice of the write (store) buffer */
    float freq;
    /* Timing results per thread (nanoseconds) */
    uint64_t *elapsed_ns; /* Elapsed time in nanoseconds */
} thread_wrapper_data_t;

/**
 * Per-benchmark metadata structure
 *
 * Contains all configuration needed to run a benchmark independently.
 * Populated at header generation time (compile-time) from Python context.
 */
typedef struct {
    const char *name;                /* Benchmark identifier */
    float frequency_ghz;             /* ISA-specific frequency in GHz */
    const char *cache_level;         /* "L1", "L2", "L3", "DRAM", or NULL for arithmetic */
    int num_threads;                 /* Optimal thread count for this benchmark */
    int *thread_affinity;            /* Specific CPU IDs to run the benchmark on */
    uint64_t read_array_size_bytes;  /* Read (load) buffer size per thread (0 for arithmetic) */
    uint64_t write_array_size_bytes; /* Write (store) buffer size per thread (0 for arithmetic or read-only) */
} benchmark_metadata_t;

/* Global synchronization primitives - declared here, defined in test_bench.c */
extern pthread_barrier_t g_barrier;
extern pthread_mutex_t g_mutex;
extern uint64_t g_max_reps;

/* ============================================================================
 * Architecture-Specific Helper Functions
 * ========================================================================== */

#ifndef CARM_BENCH_MIN_CAL_TIME_NS
#define CARM_BENCH_MIN_CAL_TIME_NS 50000000ULL /* 50ms */
#endif

#ifndef CARM_BENCH_MIN_RELIABLE_TIME_NS
#define CARM_BENCH_MIN_RELIABLE_TIME_NS 1000000ULL /* 1ms - minimum for proportional scaling */
#endif

#ifndef CARM_BENCH_START_REPS
#define CARM_BENCH_START_REPS 10
#endif

void *aligned_malloc(size_t align, size_t size)
{
    // align must be a power of 2
    void *raw = malloc(size + align - 1 + sizeof(void *));
    if (!raw)
        return NULL;

    // Find aligned address after space for the original pointer
    void **aligned = (void **) (((uintptr_t) raw + sizeof(void *) + align - 1) & ~(align - 1));
    aligned[-1] = raw; // store original pointer just before aligned block
    return aligned;
}

void aligned_free(void *ptr)
{
    if (ptr)
        free(((void **) ptr)[-1]);
}

/* Architecture-aware serialization barrier. */
static inline void serialize(void)
{
#if defined(CARM_ARCH_ARM)
    __asm__ __volatile__("dsb sy" ::: "memory");
#elif defined(CARM_ARCH_RISCV)
    __asm__ __volatile__("fence" ::: "memory");
#else
    __asm__ __volatile__("lfence" ::: "memory");
#endif
}

#if !CARM_BENCH_USE_WALLTIME
/* x86 TSC reading functions */
static inline uint64_t read_tsc_start(void)
{
    uint64_t d, a;
    // clang-format off
    __asm__ __volatile__(
        "lfence\n\t"
        "rdtsc\n\t"
        "movq %%rdx, %0\n\t"
        "movq %%rax, %1\n\t"
        : "=r"(d), "=r"(a)
        :
        : "%rax", "%rdx"
    );
    // clang-format on
    return ((uint64_t) d << 32) | a;
}

static inline uint64_t read_tsc_end(void)
{
    uint64_t d, a;
    // clang-format off
    __asm__ __volatile__(
        "rdtscp\n\t"
        "movq %%rdx, %0\n\t"
        "movq %%rax, %1\n\t"
        "lfence\n\t"
        : "=r"(d), "=r"(a)
        :
        : "%rax", "%rdx"
    );
    // clang-format on
    return ((uint64_t) d << 32) | a;
}
#endif

static inline uint64_t carm_bench_time_start(void)
{
#if CARM_BENCH_USE_WALLTIME
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ULL + (uint64_t) ts.tv_nsec;
#else
    return read_tsc_start();
#endif
}

static inline uint64_t carm_bench_time_end(void)
{
#if CARM_BENCH_USE_WALLTIME
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ULL + (uint64_t) ts.tv_nsec;
#else
    return read_tsc_end();
#endif
}

/* Calculate elapsed time in nanoseconds, converting TSC cycles if needed. */
static inline uint64_t carm_bench_elapsed_units(uint64_t start, uint64_t end, float freq_ghz)
{
    uint64_t elapsed;
#if CARM_BENCH_USE_WALLTIME
    (void) freq_ghz;
    elapsed = end - start; /* Already in nanoseconds */
#else
    /* Convert TSC cycles to nanoseconds: ns = cycles / (freq_GHz) */
    uint64_t cycles = end - start;
    elapsed = (uint64_t) ceil((double) cycles / (double) freq_ghz);
#endif
    return elapsed;
}

/* Check if elapsed nanoseconds is sufficient for calibration. */
static inline int carm_bench_elapsed_sufficient(uint64_t elapsed_ns)
{
    return elapsed_ns > CARM_BENCH_MIN_CAL_TIME_NS;
}

/* Calculate repetitions needed for main measurement (nanoseconds). */
static inline uint64_t carm_bench_calculate_reps(uint64_t elapsed_ns, uint64_t reps)
{
    return (uint64_t) ceil((double) EXPECTED_TIME_NS * (double) reps / (double) elapsed_ns);
}

/* Calculate repetitions needed to reach calibration target time (nanoseconds). */
static inline uint64_t carm_bench_calculate_cal_reps(uint64_t elapsed_ns, uint64_t reps)
{
    /* Target 120% of minimum time to ensure we overshoot and avoid oscillation.
     * Without this margin, we can get stuck scaling by 1.00x repeatedly. */
    return (uint64_t) ceil((double) CARM_BENCH_MIN_CAL_TIME_NS * 1.2 * (double) reps / (double) elapsed_ns);
}

static inline void sleep0(void)
{
    sched_yield();
}

/* Synchronization helper: serialize, barrier, yield, serialize */
static inline void barrier_sync(void)
{
    serialize();
    pthread_barrier_wait(&g_barrier);
    sleep0();
    serialize();
}

/**
 * Inline-Only Benchmark Execution
 *
 * Benchmarks are executed via an inline wrapper function that accepts
 * an inline benchmark function as a parameter. The compiler inlines
 * the benchmark directly into the wrapper without indirection overhead,
 * achieving the same result as a macro while maintaining type safety
 * and better code organization.
 *
 * Parity checklist (implement incrementally):
 * - Threading + affinity mapping (including --interleaved mapping)
 * - Priority boost/restore (PRIO_MIN)
 * - Barriers and serialization (x86 lfence, ARM dsb sy)
 * - Timing source selection (x86 TSC vs CLOCK_MONOTONIC)
 * - Repetition calibration to target runtime (implemented below)
 * - Working-set allocation sized by NUM_REP/OPS/LD/ST (+ VLEN/VLMUL)
 * - Per-run aggregation (min start/max end vs max wall-time)
 * - Output format parity (median, num_rep_max, freq_real, freq_nominal)
 */

/* Benchmark function type: both FP and memory benchmarks use this signature.
 * Arithmetic benchmarks ignore both pointer parameters; memory benchmarks use them.
 * read_ptr is the load buffer; write_ptr is the store buffer. */
typedef void (*benchmark_fn_t)(void *read_ptr, void *write_ptr, uint32_t num_reps);

/* Verbosity levels (same as output_utils.py):
 *   0 - QUIET:   no output
 *   1 - ERROR:   errors/warnings only
 *   2 - RESULT:  test results
 *   3 - CONFIG:  configuration/details
 *   4 - DEBUG:   debug output (calibration steps, etc.)
 */
#ifndef VERBOSITY
#define VERBOSITY 2
#endif

#if VERBOSITY >= 4
#define CARM_BENCH_DEBUG_PRINT(...) fprintf(stderr, "[debug] " __VA_ARGS__)
#else
#define CARM_BENCH_DEBUG_PRINT(...) \
    do {                            \
    } while (0)
#endif

#endif /* BENCHMARK_SUITE_H */
