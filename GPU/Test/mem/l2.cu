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

#define THREADS_PER_BLOCK 1024

// DEFINE NUM_REPS

// DEFINE PRECISION

// DEFINE DEVICE

__global__ void benchmark(PRECISION *__restrict__ d_X, PRECISION *__restrict__ d_Y, int iterations,
						  const uint64_t csize) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = 0; i < iterations; i++) {
		for (int j = id; j < csize; j += gridDim.x * blockDim.x)
			if (i & 1)
				d_X[j] = d_Y[j];
			else
				d_Y[j] = d_X[j];
	}
}

int main() {
	cudaSetDevice(DEVICE);

	cudaDeviceProp deviceProps;
	cudaGetDeviceProperties(&deviceProps, DEVICE);
	uint sm = deviceProps.multiProcessorCount;

	int iterations = 1;
	float milliseconds = 0;
	vector<float> time_series;

	uint64_t csize = 0;
	while (2 * csize * sizeof(PRECISION) < deviceProps.l2CacheSize / 2) csize += 2 * sm * 1024;

	PRECISION *d_X;
	// create a larger than necessary vector just in case
	cudaMalloc((void **)&d_X, 1024 * THREADS_PER_BLOCK * sizeof(PRECISION));

	// Timers
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Determine the minimum number of iterations
	while (milliseconds < 150.f) {
		iterations *= 2;
		cudaEventRecord(start);

		benchmark<<<2 * sm, THREADS_PER_BLOCK>>>(d_X, d_X + csize, iterations, csize);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&milliseconds, start, stop);
	}

	// Perform testing
	for (int i = 0; i < NUM_REPS; i++) {
		cudaEventRecord(start);

		benchmark<<<2 * sm, THREADS_PER_BLOCK>>>(d_X, d_X + csize, iterations, csize);
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

	double bytes = sizeof(PRECISION) * 2. * iterations * csize / 1e9;
	float bandwidth = bytes * 1e3 / median;

	cout << bandwidth << " GB/s" << endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(d_X);

	return 0;
}
