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

// DEFINE PRECISION

// DEFINE DEVICE

// DEFINE TEST

__global__ void benchmark(PRECISION *d_X, int iterations) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	// DEFINE INITIALIZATION

	for (int i = 0; i < iterations; i++) {
#pragma unroll
		for (int j = 0; j < 128; j++) {
			// DEFINE LOOP
		}
	}
	d_X[id] = d;
}

int main() {
	// Allocate memory in GPU
	cudaSetDevice(DEVICE);

	int iterations = 1;
	float milliseconds = 0;
	vector<float> time_series;

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

	double flops = MULTIPLIER * 4. * iterations * 128 * THREADS_PER_BLOCK * NUM_BLOCKS / 1e9;
	float perf = flops * 1e3 / median;

	cout << perf << " GFLOPS/s" << endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(d_X);

	return 0;
}
