#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdlib.h>

#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

// DEFINE NUM_REPS

// DEFINE KERNEL PARAMETERS

#define A_SIZE M *K *(THREADS_PER_BLOCK / 32) * NUM_BLOCKS
#define B_SIZE K *N *(THREADS_PER_BLOCK / 32) * NUM_BLOCKS
#define C_SIZE M *N *(THREADS_PER_BLOCK / 32) * NUM_BLOCKS

// DEFINE PRECISION

// DEFINE DEVICE

__global__ void benchmark(PRECISION_A *d_A, PRECISION_B *d_B, PRECISION_C *d_C, int iterations) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	// DEFINE INITIALIZATION

	for (int i = 0; i < iterations; i++) {
#pragma unroll
		for (int j = 0; j < 128; j++) {
			// DEFINE LOOP
		}
	}
	d_C[id] = fragsC[0];
}

int main() {
	// Allocate memory in GPU
	cudaSetDevice(DEVICE);

	int iterations = 1;
	float milliseconds = 0;
	vector<float> time_series;

	PRECISION_A *d_A;
	cudaMalloc((void **)&d_A, A_SIZE * sizeof(PRECISION_A));
	PRECISION_B *d_B;
	cudaMalloc((void **)&d_B, B_SIZE * sizeof(PRECISION_B));
	PRECISION_C *d_C;
	cudaMalloc((void **)&d_C, C_SIZE * sizeof(PRECISION_C));

	// Timers
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Determine the minimum number of iterations
	while (milliseconds < 150.f) {
		iterations *= 2;
		cudaEventRecord(start);

		benchmark<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, iterations);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&milliseconds, start, stop);
	}

	// Perform testing
	for (int i = 0; i < NUM_REPS; i++) {
		cudaEventRecord(start);

		benchmark<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, iterations);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&milliseconds, start, stop);
		time_series.push_back(milliseconds);
	}

	// calculate median of execution time
	sort(time_series.begin(), time_series.end());
	float median = 0;

	if (time_series.size() % 2 == 0) {
		median =
			(time_series[time_series.size() / 2] + time_series[time_series.size() / 2 - 1]) / 2;
	} else {
		median = time_series[time_series.size() / 2];
	}

	double flops = 2. * M * N * K * iterations * 128 * (THREADS_PER_BLOCK / 32.) * NUM_BLOCKS / 1e9;
	float perf = flops * 1e3 / median;

	cout << perf << "GFLOPS/s" << endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return 0;
}
