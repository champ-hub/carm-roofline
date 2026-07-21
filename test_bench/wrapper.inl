// This file is to be included repeatedly in microbenchmarks.h (generated code)
// It wraps each microbenchmark with full measurement logic for thread execution

#include "test_bench.h"

#ifndef UBENCH_NAME
#error "UBENCH_NAME must be defined before including wrapper.inl"
#endif

// Helper to force macro expansion before token pasting
#define _WRAPPER_EXPAND(name) WRAPPER_NAME(name)

// Generated wrapper function containing measurement logic
// This function is called by each thread and directly calls the inline benchmark
// Returns NULL to satisfy pthread_create signature (void *(*)(void *))
static void *_WRAPPER_EXPAND(UBENCH_NAME)(void *arg)
{
    thread_wrapper_data_t *tdata = (thread_wrapper_data_t *) arg;
    void *read_ptr = tdata->read_ptr;
    void *write_ptr = tdata->write_ptr;

    /* Force COW: write one byte per page so the kernel allocates real physical pages (not the shared zero page). This
     * ensures NUMA-correct first-touch placement and prevents load-only benchmarks from measuring L1 bandwidth
     * regardless of working set size. */
    for (size_t off = 0; off < tdata->read_size; off += 4096)
        ((volatile char *) read_ptr)[off] = 0;
    for (size_t off = 0; off < tdata->write_size; off += 4096)
        ((volatile char *) write_ptr)[off] = 0;
    barrier_sync();

    uint64_t reps = CARM_BENCH_START_REPS;
    int sufficient_time = 0;
    volatile uint64_t t_start = 0;
    volatile uint64_t t_end = 0;
    uint64_t elapsed_ns = 0;
    uint64_t prev_elapsed_ns = 0;

    // Calibration loop: find suitable repetition count to reach target time
    // Scales the number of repetitions until elapsed time exceeds minimum calibration time
    // NOTE: All debug output is deferred until after timing to avoid I/O interference
    while (!sufficient_time) {
        serialize();

        t_start = carm_bench_time_start();
        UBENCH_NAME(read_ptr, write_ptr, reps);
        t_end = carm_bench_time_end();
        prev_elapsed_ns = elapsed_ns;
        elapsed_ns = carm_bench_elapsed_units(t_start, t_end, tdata->freq);

        /* Detect anomalous measurements: if elapsed time decreased despite same/more reps,
         * something interfered (context switch, interrupt, etc). Retry measurement. */
        if (prev_elapsed_ns > 0 && elapsed_ns < prev_elapsed_ns * 0.8) {
            continue; /* Retry with same reps */
        }

        if (carm_bench_elapsed_sufficient(elapsed_ns)) {
            sufficient_time = 1;
        } else {
            /* Scale reps proportionally to reach target time in next iteration.
             * Guard: If elapsed time is too small, ratio becomes unreliable and can
             * cause reps to explode. Use conservative 10x scaling in this case. */
            if (elapsed_ns < CARM_BENCH_MIN_RELIABLE_TIME_NS) {
                reps *= 10;
            } else {
                reps = carm_bench_calculate_cal_reps(elapsed_ns, reps);
            }
        }
    }

    pthread_barrier_wait(&g_barrier);

    uint64_t number_rep =
            (uint64_t) ceil((double) reps * (double) EXPECTED_TIME_NS / ((double) CARM_BENCH_MIN_CAL_TIME_NS * 1.2));

    /* Update global maximum reps across all threads */
    pthread_mutex_lock(&g_mutex);
    if (g_max_reps < number_rep) {
        g_max_reps = number_rep;
    }
    pthread_mutex_unlock(&g_mutex);

    pthread_barrier_wait(&g_barrier);

    /* All threads use the same (maximum) repetition count */
    uint64_t num_reps = g_max_reps;

    barrier_sync();

    /* Measurement loop: run NUM_RUNS iterations */
    for (int i = 0; i < NUM_RUNS; i++) {
        barrier_sync();
        t_start = carm_bench_time_start();
        UBENCH_NAME(read_ptr, write_ptr, num_reps);
        t_end = carm_bench_time_end();
        elapsed_ns = carm_bench_elapsed_units(t_start, t_end, tdata->freq);

        tdata->elapsed_ns[i] = elapsed_ns;

        // barrier_sync();
    }

    serialize();

    pthread_exit(NULL);
}

#undef _WRAPPER_EXPAND
