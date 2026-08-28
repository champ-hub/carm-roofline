#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "test_bench.h"

#include <assert.h>
#include <microbenchmarks.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>

#define CARM_BENCH_DATA_ALIGNMENT 4096UL

static int align_up_size(size_t size, size_t align, size_t *aligned_size)
{
    if (align == 0 || (align & (align - 1)) != 0)
        return -1;

    if (size > SIZE_MAX - (align - 1))
        return -1;

    *aligned_size = (size + (align - 1)) & ~(align - 1);
    return 0;
}

static int multiply_size(size_t a, size_t b, size_t *result)
{
    if (a != 0 && b > SIZE_MAX / a)
        return -1;

    *result = a * b;
    return 0;
}

/**
 * test_bench.c: Wrapper-based benchmark measurement and main entry point
 *
 * This module combines:
 * 1. measure_benchmark() function - orchestrates threading, affinity, and result aggregation
 * 2. Main entry point - argument parsing and benchmark dispatch
 *
 * Design: Each benchmark has a wrapper function (in wrapper.inl) that contains the
 * full measurement logic (calibration, synchronization, timing). The wrapper directly
 * calls the inline benchmark function, eliminating function pointer indirection.
 */

static void set_process_priority_high(void)
{
    setpriority(PRIO_PROCESS, 0, PRIO_MIN);
}

static void set_process_priority_normal(void)
{
    setpriority(PRIO_PROCESS, 0, 0);
}

// Compare function for qsort
static int compare_uint64(const void *a, const void *b)
{
    uint64_t val_a = *(const uint64_t *) a;
    uint64_t val_b = *(const uint64_t *) b;
    if (val_a < val_b)
        return -1;
    if (val_a > val_b)
        return 1;
    return 0;
}

// Median calculation
static uint64_t median(int n, uint64_t x[])
{
    uint64_t *x_aux = (uint64_t *) malloc(n * sizeof(uint64_t));
    if (!x_aux) {
        fprintf(stderr, "ERROR: Failed to allocate memory for median calculation\n");
        return 0;
    }

    for (int i = 0; i < n; i++) {
        x_aux[i] = x[i];
    }

    // sort
    qsort(x_aux, n, sizeof(uint64_t), (int (*)(const void *, const void *)) compare_uint64);

    uint64_t val = x_aux[n / 2];
    // average the middle two elements if n is even
    if (n % 2 == 0) {
        val = (val + x_aux[n / 2 - 1]) / 2;
    }

    free(x_aux);
    return val;
}

// Wrapper function type: thread entry point matching pthread signature
typedef void *(*wrapper_fn_t)(void *arg);

// Global synchronization primitives - definitions (declared in test_bench.h)
pthread_barrier_t g_barrier;
pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;

// Global calibration state - definition (declared in test_bench.h)
uint64_t g_max_reps = 0;

/**
 * measure_benchmark: Execute a single benchmark with multi-threaded measurement
 *
 * This function performs:
 * 1. Thread setup with CPU affinity binding
 * 2. Dispatch threads to wrapper functions containing measurement logic
 * 3. Wait for all threads to complete (calibration + NUM_RUNS iterations)
 * 4. Result aggregation and median calculation across threads
 * 5. CSV output (median_cycles/ms, num_reps, freq_real, freq_nominal)
 *
 * Each thread executes a wrapper function (defined in wrapper.inl) that:
 * - Performs repetition calibration (multiply by 10 until >100ms, then compute optimal reps)
 * - Synchronizes with other threads via barriers
 * - Runs NUM_RUNS timed iterations
 * - Directly calls the inline benchmark function (zero function pointer overhead)
 */
static inline __attribute__((always_inline)) void
measure_benchmark(const benchmark_metadata_t *metadata, wrapper_fn_t wrapper_fn, void *combined_data)
{
    /* Extract configuration from metadata */
    int num_threads = metadata->num_threads;
    float freq = metadata->frequency_ghz;
    const char *test_id = metadata->name;

    pthread_t *threads = (pthread_t *) malloc(num_threads * sizeof(pthread_t));
    thread_wrapper_data_t *tdata = (thread_wrapper_data_t *) malloc(num_threads * sizeof(thread_wrapper_data_t));
    if (!threads || !tdata) {
        fprintf(stderr, "ERROR: Failed to allocate thread data\n");
        return;
    }
    pthread_attr_t attr;
    cpu_set_t cpus;
    size_t aligned_read = 0;
    size_t aligned_write = 0;

    if (metadata->read_array_size_bytes > 0 &&
        align_up_size((size_t) metadata->read_array_size_bytes, CARM_BENCH_DATA_ALIGNMENT, &aligned_read) != 0) {
        fprintf(stderr,
                "ERROR: Failed to align per-thread read data size (%llu bytes, alignment %lu)\n",
                (unsigned long long) metadata->read_array_size_bytes,
                (unsigned long) CARM_BENCH_DATA_ALIGNMENT);
        return;
    }

    if (metadata->write_array_size_bytes > 0 &&
        align_up_size((size_t) metadata->write_array_size_bytes, CARM_BENCH_DATA_ALIGNMENT, &aligned_write) != 0) {
        fprintf(stderr,
                "ERROR: Failed to align per-thread write data size (%llu bytes, alignment %lu)\n",
                (unsigned long long) metadata->write_array_size_bytes,
                (unsigned long) CARM_BENCH_DATA_ALIGNMENT);
        return;
    }

    size_t per_thread_stride = aligned_read + aligned_write;

    /* Initialize global state */
    g_max_reps = 0;
    pthread_barrier_init(&g_barrier, NULL, num_threads);

    /* Allocate timing arrays for each thread */
    for (int i = 0; i < num_threads; i++) {
        tdata[i].tid = i;
        size_t thread_offset = (size_t) i * per_thread_stride;
        tdata[i].read_ptr = (void *) ((char *) combined_data + thread_offset);
        tdata[i].write_ptr = (void *) ((char *) combined_data + thread_offset + aligned_read);
        tdata[i].freq = freq;
        tdata[i].read_size = aligned_read;
        tdata[i].write_size = aligned_write;

#if VERBOSITY >= 4
        if (tdata[i].read_ptr != NULL) {
            fprintf(stderr,
                    "[debug] %s thread %d read_ptr=%p per_thread_stride=%zu read_size=%llu align_mod=%zu\n",
                    test_id,
                    i,
                    tdata[i].read_ptr,
                    per_thread_stride,
                    (unsigned long long) metadata->read_array_size_bytes,
                    ((size_t) (uintptr_t) tdata[i].read_ptr) % CARM_BENCH_DATA_ALIGNMENT);
        }
        if (tdata[i].write_ptr != NULL) {
            fprintf(stderr,
                    "[debug] %s thread %d write_ptr=%p per_thread_stride=%zu write_size=%llu align_mod=%zu\n",
                    test_id,
                    i,
                    tdata[i].write_ptr,
                    per_thread_stride,
                    (unsigned long long) metadata->write_array_size_bytes,
                    ((size_t) (uintptr_t) tdata[i].write_ptr) % CARM_BENCH_DATA_ALIGNMENT);
        }
#endif

        tdata[i].elapsed_ns = (uint64_t *) malloc(NUM_RUNS * sizeof(uint64_t));
    }

    pthread_attr_init(&attr);

    /* Launch threads with CPU affinity */
    for (int i = 0; i < num_threads; i++) {
        CPU_ZERO(&cpus);

        if (metadata->thread_affinity != NULL) {
            // Pin this thread to its individual CPU
            CPU_SET(metadata->thread_affinity[i], &cpus);

            int affinity_rc = pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);
            if (affinity_rc != 0) {
                fprintf(stderr, "ERROR: pthread_attr_setaffinity_np failed with code %d\n", affinity_rc);
                exit(-1);
            }
        }

        int rc = pthread_create(&threads[i], &attr, wrapper_fn, &tdata[i]);
        if (rc) {
            fprintf(stderr, "ERROR: pthread_create failed with code %d\n", rc);
            exit(-1);
        }
    }

    // Wait for all threads to complete
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_barrier_destroy(&g_barrier);

    // Aggregate results across threads - find max elapsed time across threads per run.
    uint64_t *max_elapsed_ns = (uint64_t *) calloc(NUM_RUNS, sizeof(uint64_t));
    for (int i = 0; i < NUM_RUNS; i++) {
        max_elapsed_ns[i] = tdata[0].elapsed_ns[i];
        for (int j = 1; j < num_threads; j++) {
            if (tdata[j].elapsed_ns[i] > max_elapsed_ns[i]) {
                max_elapsed_ns[i] = tdata[j].elapsed_ns[i];
            }
        }
    }

    // Debug: print all max_elapsed_ns values as comma-separated ms
#if VERBOSITY >= 4
    fprintf(stderr, "[debug] %s times (ms): ", test_id);
    for (int i = 0; i < NUM_RUNS; i++) {
        double ms = (double) max_elapsed_ns[i] / 1000000.0;
        fprintf(stderr, "%s%.3f", i > 0 ? ", " : "", ms);
    }
    fprintf(stderr, "\n");
#endif

    uint64_t median_ns = median(NUM_RUNS, max_elapsed_ns);
    double median_ms = (double) median_ns / 1000000.0;
    printf("%s, %f, %llu\n", test_id, median_ms, (unsigned long long) g_max_reps);
#if VERBOSITY >= 3
    fflush(stdout);
#endif

    free(max_elapsed_ns);

    /* Free per-thread timing arrays */
    for (int i = 0; i < num_threads; i++) {
        free(tdata[i].elapsed_ns);
    }

    free(threads);
    free(tdata);
}

/* ============================================================================
 * Macro-Expanded Benchmark Dispatch
 * ========================================================================== */

static inline void run_all_benchmarks(void *combined_data, size_t combined_total)
{
#define X(wrapper_fn, metadata_ptr)                                              \
    do {                                                                         \
        CARM_BENCH_DEBUG_PRINT("Running benchmark: %s\n", (metadata_ptr)->name); \
        madvise(combined_data, combined_total, MADV_DONTNEED);                   \
        measure_benchmark((metadata_ptr), (wrapper_fn), combined_data);          \
    } while (0);

    MICROBENCHMARK_LIST

#undef X
}

static inline void update_max_combined_stride(size_t *max_total_size, const benchmark_metadata_t *metadata)
{
    size_t aligned_read = 0;
    size_t aligned_write = 0;
    size_t per_thread_stride = 0;
    size_t required_total_size = 0;

    if (align_up_size((size_t) metadata->read_array_size_bytes, CARM_BENCH_DATA_ALIGNMENT, &aligned_read) != 0) {
        fprintf(stderr,
                "ERROR: Read buffer size overflow for benchmark %s (read_array_size_bytes=%llu)\n",
                metadata->name,
                (unsigned long long) metadata->read_array_size_bytes);
        exit(1);
    }

    if (align_up_size((size_t) metadata->write_array_size_bytes, CARM_BENCH_DATA_ALIGNMENT, &aligned_write) != 0) {
        fprintf(stderr,
                "ERROR: Write buffer size overflow for benchmark %s (write_array_size_bytes=%llu)\n",
                metadata->name,
                (unsigned long long) metadata->write_array_size_bytes);
        exit(1);
    }

    if (aligned_read > SIZE_MAX - aligned_write) {
        fprintf(stderr, "ERROR: Combined stride overflow for benchmark %s\n", metadata->name);
        exit(1);
    }
    per_thread_stride = aligned_read + aligned_write;

    if (multiply_size(per_thread_stride, (size_t) metadata->num_threads, &required_total_size) != 0) {
        fprintf(stderr,
                "ERROR: Total buffer size overflow for benchmark %s (stride=%zu, num_threads=%d)\n",
                metadata->name,
                per_thread_stride,
                metadata->num_threads);
        exit(1);
    }

    if (required_total_size > *max_total_size)
        *max_total_size = required_total_size;
}

int main(int argc, char *argv[])
{
    /* Compute maximum combined (read+write) per-thread stride across all benchmarks.
     * For each benchmark, each thread gets a contiguous [read | write] region of size
     * aligned(read_array_size_bytes) + aligned(write_array_size_bytes), so the read
     * and write arrays are physically adjacent, eliminating cache-set aliasing. */
    size_t max_combined_total = 0;
#define X(wrapper_fn, metadata_ptr) update_max_combined_stride(&max_combined_total, (metadata_ptr));
    MICROBENCHMARK_LIST
#undef X

    void *combined_buffer = NULL;

    if (max_combined_total > 0) {
        combined_buffer = aligned_malloc(CARM_BENCH_DATA_ALIGNMENT, max_combined_total);
        if (combined_buffer == NULL) {
            fprintf(stderr, "ERROR: failed to allocate combined read/write buffer\n");
            exit(1);
        }
#if VERBOSITY >= 4
        fprintf(stderr,
                "[debug] test_bench.c: Combined buffer: %lu bytes at %p (alignment %zu)\n",
                (unsigned long) max_combined_total,
                combined_buffer,
                ((uintptr_t) combined_buffer) % CARM_BENCH_DATA_ALIGNMENT);
#endif
    }

    /* Elevate process priority for accurate benchmarking */
    set_process_priority_high();

    /* Run all available tests with parsed arguments */
    run_all_benchmarks(combined_buffer, max_combined_total);

    /* Restore normal process priority */
    set_process_priority_normal();

    aligned_free(combined_buffer);

    return 0;
}
