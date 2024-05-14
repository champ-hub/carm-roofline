#define NUM_COUNTERS (NUM_FIXED_COUNTERS+NUM_GP_COUNTERS+UNC_ENABLE_FIXED_CTR+NUM_UNC_CBO_COUNTERS+NUM_UNC_ARB_COUNTERS+NUM_UNC_IMC_COUNTERS)

#define ROUND_UP(A,B)  ((A+B-1)/B*B)

typedef cpu_set_t ProcMaskType;

int long long num_rep_max;

pthread_attr_t attr;
pthread_barrier_t bar;
pthread_mutex_t mutexsum = PTHREAD_MUTEX_INITIALIZER;

int long long ** __attribute__((aligned(64))) cycles_s, ** __attribute__((aligned(64))) cycles_e;
int long long ** __attribute__((aligned(64))) cycles_overhead_s, ** __attribute__((aligned(64))) cycles_overhead_e;
int long long *** __attribute__((aligned(64))) counters_val_s, *** __attribute__((aligned(64))) counters_val_e;
int long long *** __attribute__((aligned(64))) counters_overhead_s, *** __attribute__((aligned(64))) counters_overhead_e;
int long long *** __attribute__((aligned(64))) batch_val_s, *** __attribute__((aligned(64))) batch_val_e;
int long long *** __attribute__((aligned(64))) batch_overhead_s, *** __attribute__((aligned(64))) batch_overhead_e;

pid_t get_thread_id();
int bind_thread_to_core(int core_id);
inline __attribute__((always_inline))  void sleep0();
void set_process_priority_high();
void set_process_priority_normal();
inline __attribute__((always_inline)) void clear_cache_deep();
int long long median(int n, int long long x[]);
inline __attribute__((always_inline)) void serialize();
inline __attribute__((always_inline)) long long read_tsc_start();
inline __attribute__((always_inline)) long long read_tsc_end();
void * benchmark_test(void *threadid);
void run_test();
