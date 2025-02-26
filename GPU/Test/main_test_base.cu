#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include <iostream>

using namespace std;

#define NUM_REPS 1024

// DEFINE KERNEL PARAMETERS

// DEFINE PRECISION

// DEFINE DEVICE

__global__ void benchmark(PRECISION *d_X, int iterations);

int main() {
	// Allocate memory in GPU
	cudaSetDevice(DEVICE);

	int iterations = 10;
	float milliseconds = 0;

	PRECISION *d_X;
	cudaMalloc((void **)&d_X, NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(PRECISION));

	// Timers
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Determine the minimum number of iterations
	while (milliseconds < 150.f) {
		iterations *= 2;
		cudaEventRecord(start);
		benchmark<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_X, iterations);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&milliseconds, start, stop);
	}

	// Perform testing
	for (int i = 0; i < NUM_REPS; i++) {
		cudaEventRecord(start);
		benchmark<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_X, iterations);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&milliseconds, start, stop);
		double flops = 2. * 4 * iterations * 128 * THREADS_PER_BLOCK * NUM_BLOCKS / 1e9;
		float perf = flops * 1e3 / milliseconds;

		cout << perf << " GFLOPS" << endl;
	}

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(d_X);
	return 0;
}