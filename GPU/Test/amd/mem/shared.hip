#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <iostream>
#include <vector>

// DEFINE NUM_REPS

// DEFINE KERNEL PARAMETERS

// DEFINE PRECISION

// DEFINE DEVICE

using namespace std;

__global__ void benchmark(PRECISION *dummy, int numIter) {
	const uint32_t tid = threadIdx.x;
	__shared__ PRECISION shmem[64];

	if (tid == 0) {
#pragma unroll
		for (int i = 0; i < 63; i++) shmem[i] = i + 1;

		shmem[63] = 0;
	}

	__syncthreads();

	PRECISION index = tid;
	for (uint32_t iter = 0; iter < numIter; iter++)
#pragma unroll
		for (int j = 0; j < 128; j++) index = shmem[index];

	dummy[tid] = index;
}

int main() {
	hipSetDevice(DEVICE);

	int iterations = 1;
	float milliseconds = 0;
	vector<float> time_series;

	PRECISION *d_X;
	hipMalloc((void **)&d_X, NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(PRECISION));

	// Timers
	hipEvent_t start, stop;
	hipEventCreate(&start);
	hipEventCreate(&stop);

	// Determine the minimum number of iterations
	while (milliseconds < 150.f) {
		iterations *= 2;
		hipEventRecord(start);

		benchmark<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_X, iterations);
		hipEventRecord(stop);
		hipEventSynchronize(stop);

		hipEventElapsedTime(&milliseconds, start, stop);
	}

	// Perform testing
	for (int i = 0; i < NUM_REPS; i++) {
		hipEventRecord(start);

		benchmark<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_X, iterations);
		hipEventRecord(stop);
		hipEventSynchronize(stop);

		hipEventElapsedTime(&milliseconds, start, stop);
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

	double bytes = sizeof(PRECISION) * iterations * 128 * THREADS_PER_BLOCK * NUM_BLOCKS / 1e9;
	float bandwidth = bytes * 1e3 / median;

	cout << bandwidth << " GB/s" << endl;

	hipEventDestroy(start);
	hipEventDestroy(stop);

	hipFree(d_X);

	return 0;
}