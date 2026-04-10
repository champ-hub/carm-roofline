// Generic frequency detection test
// Includes ISA-specific implementation via frequency.h

#include "frequency.h"

#include <pthread.h>
#include <sched.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

pthread_barrier_t bar;
pthread_mutex_t freq_mutex;

float freq_real_max = 0.0;
float freq_nominal_max = 0.0;

static void sleep0(void)
{
    sched_yield();
}

struct thread_args {
    int tid;
};

void *benchmark_thread(void *args)
{
    (void) args; // Unused
    uint64_t iterations = 1e9;
    struct timespec t_start, t_end;

    // Run 10 iterations and take the maximum frequency
    for (int i = 0; i < 10; i++) {
        serialize();
        pthread_barrier_wait(&bar);
        sleep0();
        serialize();

        clock_gettime(CLOCK_MONOTONIC, &t_start);
        measure_clock_ticks_start();
        bench_function(iterations);
        measure_clock_ticks_end();
        clock_gettime(CLOCK_MONOTONIC, &t_end);

        // Calculate elapsed time in milliseconds
        uint64_t test_time_diff_ms =
                1000 * (t_end.tv_sec - t_start.tv_sec) + ((t_end.tv_nsec - t_start.tv_nsec) / 1000000);

        // Avoid division by zero
        if (test_time_diff_ms == 0) {
            serialize();
            pthread_barrier_wait(&bar);
            sleep0();
            continue;
        }

        // Calculate frequencies (ISA-specific)
        float freq_real = 0.0;
        float freq_nominal = 0.0;
        calculate_frequencies(iterations, &t_start, &t_end, &freq_real, &freq_nominal);

        // Update global maximums under lock
        pthread_mutex_lock(&freq_mutex);
        if (freq_real > freq_real_max) {
            freq_real_max = freq_real;
        }
        if (freq_nominal > freq_nominal_max) {
            freq_nominal_max = freq_nominal;
        }
        pthread_mutex_unlock(&freq_mutex);

        serialize();
        pthread_barrier_wait(&bar);
        sleep0();
        serialize();
    }

    pthread_exit(NULL);
}

static void parse_arguments(int argc, char *argv[], int *num_threads)
{
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            *num_threads = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [--threads N]\n", argv[0]);
            printf("  --threads N   Number of threads to use (default: 1)\n");
            exit(0);
        }
    }
}

int main(int argc, char *argv[])
{
    int num_threads = 1;
    parse_arguments(argc, argv, &num_threads);

    pthread_t threads[num_threads];
    struct thread_args t_args[num_threads];

    pthread_barrier_init(&bar, NULL, num_threads);
    pthread_mutex_init(&freq_mutex, NULL);

    // Spawn threads
    for (int i = 0; i < num_threads; i++) {
        t_args[i].tid = i;
        pthread_create(&threads[i], NULL, benchmark_thread, &t_args[i]);
    }

    // Join threads
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_barrier_destroy(&bar);
    pthread_mutex_destroy(&freq_mutex);

    /* IMPORTANT NOTE
    If an ISA returns a nominal frequency, the test_bench must use TSC for timing. If an ISA does not return a nominal
    frequency, the test_bench must use wall-clock time. The two must match to produce accurate results.
    */
    // Output JSON with detected frequency/frequencies
    printf("{\n");
    printf("  \"frequency_hz\": %lu", (uint64_t) (freq_real_max * 1e9));
    if (has_nominal_frequency()) {
        printf(",\n  \"frequency_nominal_hz\": %lu", (uint64_t) (freq_nominal_max * 1e9));
    }
    printf("\n}\n");

    return 0;
}
