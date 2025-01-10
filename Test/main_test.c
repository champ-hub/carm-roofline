#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif


#include <linux/unistd.h>
#include <unistd.h> 
#include <sys/resource.h>
#include <sys/time.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <sched.h>
#include <stdint.h>
#include <pthread.h>
#include <math.h>
#include <stdbool.h>

#include "test_params.h" 
#include "../Bench/calc_param.c"
#include "../Bench/config_test.h"

#if defined(ARM) || defined(RISCV) || defined(RISCVVECTOR)
	#include <stdlib.h>
#else
	#include <mm_malloc.h>
#endif


#define NUM_RUNS 1024
#define EXPECTED_TIME 100000000  //in ns

//ARM SECTION
#if defined(ARM)
	extern uint64_t clktestarm(uint64_t iterations) __attribute(());
//x86 SECTION
#elif !defined(ARM) && !defined(RISCV) && !defined(RISCVVECTOR)
	extern uint64_t clktestx86(uint64_t iterations) __attribute((sysv_abi));
//RISCV SECTION
#elif defined(RISCV) || defined(RISCVVECTOR)
	extern uint64_t clktestriscv(uint64_t iterations) __attribute(());
#endif

int long long num_reps_2;

void sleep0(){
	sched_yield();
}

void set_process_priority_high(){
	setpriority(PRIO_PROCESS, 0, PRIO_MIN);
}

void set_process_priority_normal(){
	setpriority(PRIO_PROCESS, 0, 0);
}


pid_t get_thread_id()
{
	return syscall(__NR_gettid);
}

int long long median(int n, int long long x[]){
	int long long temp;
	int i, j;
	int long long val;

	int long long * x_aux = (int long long *)malloc(n*sizeof(int long long));

	for(i = 0; i < n; i++){
		x_aux[i] = x[i];
	}

	for(i=0; i<n-1; i++){
		for(j=i+1; j<n; j++){
			if(x_aux[j] < x_aux[i]){
				temp = x_aux[i];
				x_aux[i] = x_aux[j];
				x_aux[j] = temp;
			}
		}
	}
	if(n%2==0){
		val = ((x_aux[n/2] + x_aux[n/2 - 1]) / 2);
	}else{
		val = x_aux[n/2];
	}
	free(x_aux);
	return val;
}
//ARM SECTION
#if defined(ARM)
	static inline void serialize() {
    	asm volatile ("dsb sy" ::: "memory");
	}
//x86 SECTION
#elif !defined(ARM) && !defined(RISCV) && !defined(RISCVVECTOR)
	static  inline void serialize(){
		asm volatile ( "lfence;" : : : );
	}

	static  inline long long read_tsc_start(){
	uint64_t d;
	uint64_t a;
	asm __volatile__ (
		"lfence;"
		"rdtsc;"
		"movq %%rdx, %0;"
		"movq %%rax, %1;"
		: "=r" (d), "=r" (a)
		:
		: "%rax", "%rdx"
	);

	return ((long long)d << 32 | a);
	}

	static inline long long read_tsc_end(){
		uint64_t d;
		uint64_t a;
		asm __volatile__ (
			"rdtscp;"
			"movq %%rdx, %0;"
			"movq %%rax, %1;"
			"lfence;"
			: "=r" (d), "=r" (a)
			:
			: "%rax", "%rdx"
		);

		return ((long long)d << 32 | a);
	}
#endif

pthread_attr_t attr;
pthread_barrier_t bar;
pthread_barrier_t bar2;
pthread_mutex_t mutexsum;

int long long num_rep_max;

int tsc_cycle_counting = 1;
int measure_freq = 1;


int long long ** cycles_s, ** cycles_e;

uint64_t ** time_test_total;

float freq_real = 0;
float freq_nominal = 0;

struct pthread_args{
    int tid;
    float freq;
};

void * benchmark_test(void *t_args){

	int i;
	int long long num_reps_t;
	int long long expected_time = EXPECTED_TIME;
	uint64_t time_diff_ms = 0;

    struct pthread_args *args = t_args;

	struct timespec t_start, t_end;

	long tid = (long)args->tid;
    freq_real = args->freq;
	uint64_t iterationsHigh = 8e9;
	float clockSpeedGhzmax = 0;

	int sufficient_time = 0;
	//ARM SECTION
	#if defined(ARM)

	//x86 / RISCV SECTION
	#elif !defined(ARM) && !defined(RISCV) && !defined(RISCVVECTOR)
		float nominalClockSpeed = 0;
		volatile long long tsc_s;
		volatile long long tsc_e;
	#endif

	int long long test_reps = 1;

	#if defined (MEM) || defined (DIV) || defined(MIXED)
		//ARM / RISCV SECTION
		#if defined(ARM) || defined(RISCV)
			double size_kb = (double)(NUM_REP*OPS*(NUM_LD+NUM_ST)*sizeof(PRECISION)/1024.0);
			int size_kb_rounded_up = (int)ceil(size_kb);
			PRECISION * test_var = (PRECISION*)malloc(size_kb_rounded_up*1024);
			for(i=0; i< NUM_REP*OPS*(NUM_LD+NUM_ST); i++){
				test_var[i] = 1;
			}
		#elif defined(RISCVVECTOR)
			double size_kb = (double)(NUM_REP*OPS*(NUM_LD+NUM_ST)*sizeof(PRECISION)*VLEN*VLMUL/1024.0);
			int size_kb_rounded_up = (int)ceil(size_kb);
			PRECISION * test_var = (PRECISION*)malloc(size_kb_rounded_up*1024);
			for(i=0; i< NUM_REP*OPS*(NUM_LD+NUM_ST)*VLEN*VLMUL; i++){
				test_var[i] = 1;
			}
		//x86 SECTION
		#else
			double size_kb = (double)(NUM_REP*OPS*(NUM_LD+NUM_ST)*sizeof(PRECISION)/1024.0);
			int size_kb_rounded_up = (int)ceil(size_kb);
			PRECISION * test_var = (PRECISION*)_mm_malloc(size_kb_rounded_up*1024, ALIGN);
			for(i=0; i< NUM_REP*OPS*(NUM_LD+NUM_ST); i++){
				test_var[i] = 1;
			}
		#endif
	//fprintf(stderr, "Size: %d Kb ", size_kb_rounded_up);
	#endif
	pthread_barrier_wait(&bar);

	//CLOCK SPEED MEASURING
	#if defined(ARM) //ARM SECTION
		if (measure_freq == 1){
			for (int i=0; i<10; i++){
				serialize();
				pthread_barrier_wait(&bar);
				sleep0();
				serialize();

				clock_gettime(CLOCK_MONOTONIC, &t_start);
				clktestarm(iterationsHigh);
				clock_gettime(CLOCK_MONOTONIC, &t_end);

				uint64_t test_time_diff_ms = 1000 * (t_end.tv_sec - t_start.tv_sec) + ((t_end.tv_nsec - t_start.tv_nsec) / 1000000);
				float latency = 1e6 * (float)test_time_diff_ms / (float)iterationsHigh;
				float clockspeedGhzaux = 1 / latency;

				pthread_mutex_lock (&mutexsum);
				if(clockSpeedGhzmax < clockspeedGhzaux){
					clockSpeedGhzmax = clockspeedGhzaux;
				}
				pthread_mutex_unlock (&mutexsum);

				freq_real = clockSpeedGhzmax;

				serialize();
				pthread_barrier_wait(&bar);
				sleep0();
				serialize();
			}
		}
	#elif defined(RISCV) || defined(RISCVVECTOR) //RISCV SECTION
		if (measure_freq == 1){
			for (int i=0; i<10; i++){

				pthread_barrier_wait(&bar);
				sleep0();

				clock_gettime(CLOCK_MONOTONIC, &t_start);
				clktestriscv(iterationsHigh);
				clock_gettime(CLOCK_MONOTONIC, &t_end);
				
				uint64_t test_time_diff_ms = 1000 * (t_end.tv_sec - t_start.tv_sec) + ((t_end.tv_nsec - t_start.tv_nsec) / 1000000);
				float latency = 1e6 * (float)test_time_diff_ms / (float)iterationsHigh;
				float clockspeedGhzaux = 1 / latency;

				pthread_mutex_lock (&mutexsum);
				if(clockSpeedGhzmax < clockspeedGhzaux){
					clockSpeedGhzmax = clockspeedGhzaux;
				}
				pthread_mutex_unlock (&mutexsum);

				freq_real = clockSpeedGhzmax;

				pthread_barrier_wait(&bar);
				sleep0();
			}
		}
	#elif !defined(ARM) && !defined(RISCV) && !defined(RISCVVECTOR) //x86 SECTION
		if (measure_freq == 1){
			for (int i=0; i<10; i++){
				serialize();
				pthread_barrier_wait(&bar);
				sleep0();
				serialize();

				clock_gettime(CLOCK_MONOTONIC, &t_start);
				tsc_s = read_tsc_start();
				clktestx86(iterationsHigh);
				tsc_e = read_tsc_end();
				clock_gettime(CLOCK_MONOTONIC, &t_end);

				uint64_t test_time_diff_ms = 1000 * (t_end.tv_sec - t_start.tv_sec) + ((t_end.tv_nsec - t_start.tv_nsec) / 1000000);
				float latency = 1e6 * (float)test_time_diff_ms / (float)iterationsHigh;
				float clockspeedGhzaux = 1 / latency;
				float nominalClockSpeedaux = (float) ((tsc_e-tsc_s)/((float) (test_time_diff_ms * 1000000)));

				pthread_mutex_lock (&mutexsum);
				if(clockSpeedGhzmax < clockspeedGhzaux){
					clockSpeedGhzmax = clockspeedGhzaux;
				}
				if(nominalClockSpeed < nominalClockSpeedaux){
					nominalClockSpeed = nominalClockSpeedaux;
				}
				pthread_mutex_unlock (&mutexsum);

				freq_real = clockSpeedGhzmax;
				freq_nominal = nominalClockSpeed;

				serialize();
				pthread_barrier_wait(&bar);
				sleep0();
				serialize();
			}
		}
	#endif //End of Clock Speed Measuring
	pthread_barrier_wait(&bar);
	
	#if !defined(RISCV) && !defined(RISCVVECTOR)
		serialize();
	#endif

	//Calculate Number Iterations necessary to obtain minimum time
	while (sufficient_time == 0){
		test_reps *= 10;
		#if !defined(RISCV) && !defined(RISCVVECTOR)
			serialize();
		#endif

		#if defined(ARM) || defined(RISCV) || defined(RISCVVECTOR)//ARM / RISCV SECTION
			clock_gettime(CLOCK_MONOTONIC, &t_start);
		#else//x86 SECTION
			tsc_s = read_tsc_start();
		#endif
		
		//CALCULATE NUMBER ITERATIONS FOR TEST CODE TO EXECUTE THE EXPECTED TIME
		#if defined (MEM) || defined (DIV) || defined(MIXED)
			#if defined (VAR)
				test_function(test_var,test_reps, num_reps_2);
			#else
				test_function(test_var,test_reps);
			#endif
		#else
			test_function(test_reps);
		#endif
		
		#if defined(ARM) || defined(RISCV) || defined(RISCVVECTOR)//ARM / RISCV SECTION
			clock_gettime(CLOCK_MONOTONIC, &t_end);
		#else //x86 SECTION
			tsc_e = read_tsc_end();
		#endif

		//ARM / RISCV SECTION
		#if defined(ARM) || defined(RISCV) || defined(RISCVVECTOR)
			time_diff_ms = 1000 * (t_end.tv_sec - t_start.tv_sec) + ((t_end.tv_nsec - t_start.tv_nsec) / 1000000);
			if (time_diff_ms > 100){
				sufficient_time = 1;
			}
		
		//x86 SECTION
		#else
			if ((tsc_e-tsc_s) > 100000){
				sufficient_time = 1;
			}
		#endif
	}

	if(FP_INST >= 131072){
		expected_time = 100000000;  
	}
	pthread_barrier_wait(&bar);
	
	#if defined(ARM) || defined(RISCV) || defined(RISCVVECTOR) //ARM / RISCV SECTION
		int long long number_rep_aux = (int long long) ceil( (double) expected_time*freq_real*test_reps/(((long long)time_diff_ms)*freq_real*1000000));
	#else//x86 SECTION
		int long long number_rep_aux = (int long long) ceil( (double) expected_time*freq_real*test_reps/(tsc_e-tsc_s));
	#endif

	pthread_mutex_lock (&mutexsum);
	if(num_rep_max < number_rep_aux){
		num_rep_max = number_rep_aux;
			}
	pthread_mutex_unlock (&mutexsum);

	pthread_barrier_wait(&bar);

	num_reps_t = num_rep_max;

	#if !defined(RISCV) && !defined(RISCVVECTOR)
		serialize();
	#endif
	pthread_barrier_wait(&bar);
	sleep0();
	#if !defined(RISCV) && !defined(RISCVVECTOR)
		serialize();
	#endif

	//MICROBENCHMARK LOOP
	for(i=0;i<NUM_RUNS;i++){
		
		#if !defined(RISCV) && !defined(RISCVVECTOR)
			serialize();
		#endif
		pthread_barrier_wait(&bar);
		sleep0();
		#if !defined(RISCV) && !defined(RISCVVECTOR)
			serialize();
		#endif

		#if defined(ARM) || defined(RISCV) || defined(RISCVVECTOR)//ARM / RISCV SECTION
			clock_gettime(CLOCK_MONOTONIC, &t_start);
		#else //x86 SECTION
			tsc_s = read_tsc_start();
		#endif

		#if defined (MEM) || defined (DIV) || defined(MIXED)
			#if defined (VAR)
				test_function(test_var, num_reps_t, num_reps_2);
			#else
				test_function(test_var, num_reps_t);
			#endif
		#else
			test_function(num_reps_t);
		#endif
		
		#if defined(ARM) || defined(RISCV) || defined(RISCVVECTOR) //ARM / RISCV SECTION
			clock_gettime(CLOCK_MONOTONIC, &t_end);
		#else//x86 SECTION
        	tsc_e = read_tsc_end();
		#endif
		
		#if !defined(RISCV) && !defined(RISCVVECTOR)
			serialize();
		#endif
		
		#if defined(ARM) || defined(RISCV) || defined(RISCVVECTOR) //ARM / RISCV SECTION
			time_test_total[tid][i] = (long long) (1000 * (t_end.tv_sec - t_start.tv_sec) + ((t_end.tv_nsec - t_start.tv_nsec) / 1000000));
		#else //x86 SECTION
			cycles_s[tid][i] = tsc_s;
			cycles_e[tid][i] = tsc_e;
		#endif

		#if !defined(RISCV) && !defined(RISCVVECTOR)
			serialize();
		#endif
		pthread_barrier_wait(&bar);
        sleep0();

	}

	#if !defined(RISCV) && !defined(RISCVVECTOR)
		serialize();
	#endif
	
	#if defined (MEM) || defined (DIV) || defined(MIXED)
		#if defined(ARM) || defined(RISCV) || defined(RISCVVECTOR)//ARM / RISCV SECTION
			free(test_var);
		#else //x86 SECTION
			_mm_free(test_var);
		#endif
	#endif
	
	pthread_exit(NULL);
}


void input_parser(int n_args, char*args[], int* num_threads, bool* interleaved, float* freq){
    
    int i;
    
    for(i = 0; i < n_args; i++){
        if(strcmp(args[i], "-threads") == 0){
            (*num_threads) = atoi(args[i+1]);
        } 
        if(strcmp(args[i], "-freq") == 0){
            (*freq) = atof(args[i+1]);
        }
		if(strcmp(args[i], "-measure_freq") == 0){
			if (strcmp(args[i+1], "0") == 0){
				measure_freq = 1;
			} else if (strcmp(args[i+1], "1") == 0)
			{
				measure_freq = 0;
			}
		}
        if(strcmp(args[i], "--interleaved") == 0){
            (*interleaved) = 1;
        }
        if(strcmp(args[i], "-h") == 0 || strcmp(args[i], "--help") == 0){
            printf("Usage: ./test -threads <num_threads> -freq <nominal_freq> -measure_freq [0, 1]  [--interleaved]\n");
            printf("Default Values:\n");
            printf("num_threads = 1\n");
            printf("nominal_freq = 1.0\n");
			printf("measure_freq = 0\n");
            printf("Use --interleaved for systems with several NUMA domains where the cores domain is interleaved (core 0 - node 0; core 1 - node 1; core 2 - node 0 ...)\n");
        }   
    }
}


int main(int argc, char*argv[]){

    int num_threads = 1;
    bool interleaved = 0;
    freq_real = 1.0;
	
    input_parser(argc, argv, &num_threads, &interleaved, &freq_real);

	freq_nominal = freq_real;

	#if defined (VAR)
		int num_aux;
		int VLEN_aux = 1;
		num_reps_2 = mem_math (NUM_REP, NUM_LD, NUM_ST, &num_aux, ALIGN, VLEN_aux);
	#endif

    int i, j;
	num_rep_max = 0;


	int rc;
	pthread_t threads[num_threads];
	void * status;
	cpu_set_t cpus;

    struct pthread_args *t_args = malloc(num_threads*sizeof(struct pthread_args));
	
	//ARM SECTION
	#if defined(ARM) || defined(RISCV) || defined(RISCVVECTOR)
		time_test_total = (uint64_t **)malloc(num_threads*sizeof(uint64_t *));
		for(i = 0; i < num_threads; i++){
			time_test_total[i] = (uint64_t *)malloc(NUM_RUNS*sizeof(uint64_t));
		}
	//x86 / RISCV SECTION
	#else
		cycles_s = (int long long **)malloc(num_threads*sizeof(int long long *));
		cycles_e = (int long long **)malloc(num_threads*sizeof(int long long *));
		for(i = 0; i < num_threads; i++){
			cycles_s[i] = (int long long *)malloc(NUM_RUNS*sizeof(int long long));
			cycles_e[i] = (int long long *)malloc(NUM_RUNS*sizeof(int long long));
		}
	#endif

	set_process_priority_high();

	pthread_barrier_init(&bar, NULL, num_threads);
	pthread_barrier_init(&bar2, NULL, num_threads);
	
    if(interleaved){
		//In case of a NUMA environment that requires manual assignment:
		//You can use this structure to help define custom thread binding for NUMA environments
		/*int numa_node_cpus_manual_assign[64] = {
			0,   // Thread 0 goes to core 0
			8,   // Thread 1 goes to core 8
			32,  // Thread 2 goes to core 32
			40,  // Thread 3 goes to core 40
			16,  // Thread 4 goes to core 16
			24,  // Thread 5 goes to core 24
			48,  // Thread 6 goes to core 48
			56,  // Thread 7 goes to core 56
			1, 9, 33, 41, 17, 25, 49, 57,
			2, 10, 34, 42, 18, 26, 50, 58,
			3, 11, 35, 43, 19, 27, 51, 59,
			4, 12, 36, 44, 20, 28, 52, 60,
			5, 13, 37, 45, 21, 29, 53, 61,
			6, 14, 38, 46, 22, 30, 54, 62,
			7, 15, 39, 47, 23, 31, 55, 63,
		};
		for(i = 0; i < num_threads; i++){
			t_args[i].tid = i;
			t_args[i].freq = freq_real;
			CPU_ZERO(&cpus);
			int core = numa_node_cpus_manual_assign[i];
			//fprintf(stderr, "Threads %d goes to core %d\n", i, core);
			// Set the affinity to the CPU core calculated for the current thread
			CPU_SET(core, &cpus);
			pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);
			rc = pthread_create(&threads[i], &attr, benchmark_test,(void *) &t_args[i]);
			if (rc){
				printf("ERROR; return code from pthread_create() is %d\n", rc);
				exit(-1);
			}
			
		}*/

		//In case of a NUMA environment organized in blocks uncomment this code section
		//Example 1: Node 1 -> 0-31 | Node 2 -> 32-63 | Node 3 -> 64-95 | Node 4 -> 96-127
		//int max_threads = 128
		//int num_nodes =2;
		//Example 2: Node 1 -> 0-127 | Node 2 -> 128-255
		/*int max_threads = 256;
		int num_nodes = 2;
		for(i = 0; i < num_threads; i++){
				t_args[i].tid = i;
				t_args[i].freq = freq_real;

				int node = i % num_nodes;

				// Calculate the CPU number within the NUMA node's CPU range
				int cpu = i / num_nodes;

				// Calculate the absolute CPU number based on the NUMA node
				int core = node * (max_threads / num_nodes) + cpu;

				// Set the affinity to the CPU core calculated for the current thread

				CPU_ZERO(&cpus);
				//fprintf(stderr, "Threads %d goes to core %d\n", i, core);
				// Set the affinity to the CPU core calculated for the current thread
				CPU_SET(core, &cpus);
				pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);
				rc = pthread_create(&threads[i], &attr, benchmark_test,(void *) &t_args[i]);
				if (rc){
						printf("ERROR; return code from pthread_create() is %d\n", rc);
						exit(-1);
				}

		}*/
		//In case of a NUMA environment organized with interleaved cores on 2 nodes (Default)
        for(i = 0; i < num_threads; i++){
            t_args[i].tid = i;
            t_args[i].freq = freq_real;
            CPU_ZERO(&cpus);
            if(i < (int)num_threads/2){
				CPU_SET(i*2, &cpus);
			}
            if(i >= (int)num_threads/2){
				CPU_SET((i-((int)num_threads/2))*2+1, &cpus);
			}
			//fprintf(stderr, "Threads %d goes to core\n", i);
            pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);
            rc = pthread_create(&threads[i], &attr, benchmark_test,(void *) &t_args[i]);
            if (rc){
                printf("ERROR; return code from pthread_create() is %d\n", rc);
                exit(-1);
            }
        }
    }else{
        for(i = 0; i < num_threads; i++){
            t_args[i].tid = i;
            t_args[i].freq = freq_real;
            CPU_ZERO(&cpus);
            CPU_SET(i, &cpus);
            pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);
            rc = pthread_create(&threads[i], &attr, benchmark_test,(void *) &t_args[i]);
            if (rc){
                printf("ERROR; return code from pthread_create() is %d\n", rc);
                exit(-1);
            }
        }
    }

	
	for(i = 0; i < num_threads; i++){
		rc = pthread_join(threads[i], &status);
		if (rc){
			printf("ERROR; return code from pthread_join() is %d\n", rc);
			exit(-1);
		}
	}
	
	pthread_barrier_destroy(&bar);
	pthread_barrier_destroy(&bar2);
	pthread_mutex_destroy(&mutexsum);

	set_process_priority_normal();

	//PARSE RESULTS

	//ARM / RISCV SECTION
	#if defined(ARM) || defined(RISCV) || defined(RISCVVECTOR)
		uint64_t * max_time = calloc(NUM_RUNS, sizeof(uint64_t));

		for(i=0;i<NUM_RUNS;i++){
				max_time[i] = time_test_total[0][i];
			for(j=1;j<num_threads;j++){
				if(time_test_total[j][i] > max_time[i]) {
					max_time[i] = time_test_total[j][i];
					}

			}
		}

		printf("%f, %lld, %f, %f",(double) median(NUM_RUNS,(int long long *)max_time), num_rep_max, freq_real, freq_nominal);

		for(i=0;i<num_threads;i++){
		free(time_test_total[i]);
		}
		free(max_time);
		free(time_test_total);

	//x86 SECTION
	#else
		int long long * min_cycles_start = calloc(NUM_RUNS,sizeof(int long long));
		int long long * max_cycles_end = calloc(NUM_RUNS,sizeof(int long long));

		for(i=0;i<NUM_RUNS;i++){
			min_cycles_start[i] = cycles_s[0][i];
			max_cycles_end[i] = cycles_e[0][i];
			for(j=1;j<num_threads;j++){
				if(cycles_s[j][i] < min_cycles_start[i]) min_cycles_start[i] = cycles_s[j][i];
				if(cycles_e[j][i] > max_cycles_end[i]) max_cycles_end[i] = cycles_e[j][i];
			}
			max_cycles_end[i] = max_cycles_end[i] - min_cycles_start[i];
		}

		printf("%f, %lld, %f, %f",(float)median(NUM_RUNS,max_cycles_end), num_rep_max, freq_real, freq_nominal);

		//FREE ALL VARIABLES
		for(i=0;i<num_threads;i++){
			free(cycles_s[i]);
			free(cycles_e[i]);
		}

		free(min_cycles_start);
		free(max_cycles_end);

		free(cycles_s);
		free(cycles_e);
	#endif

	

	return 0;
}
