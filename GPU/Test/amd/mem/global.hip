#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <iostream>
#include <vector>

// DEFINE NUM_REPS

// DEFINE KERNEL PARAMETERS

// DEFINE PRECISION

// DEFINE DEVICE

#define STRIDE 32768 * 8L

using namespace std;

__global__ void benchmark(PRECISION *d_X, PRECISION *d_Y, int iterations) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = 0; i < iterations; i++) {
#pragma unroll
		for (int j = 0; j < 64; j++) {
			d_Y[id + j * STRIDE] = d_X[id + j * STRIDE];
		}
	}
}

int main() {
	hipSetDevice(DEVICE);

	hipDeviceProp_t deviceProps;
	hipGetDeviceProperties(&deviceProps, DEVICE);

	int iterations = 1;
	float milliseconds = 0;
	vector<float> time_series;

	PRECISION *d_X;
	hipMalloc((void **)&d_X, (NUM_BLOCKS * THREADS_PER_BLOCK + 64 * STRIDE) * sizeof(PRECISION));
	PRECISION *d_Y;
	hipMalloc((void **)&d_Y, (NUM_BLOCKS * THREADS_PER_BLOCK + 64 * STRIDE) * sizeof(PRECISION));

	// Timers
	hipEvent_t start, stop;
	hipEventCreate(&start);
	hipEventCreate(&stop);

	// Determine the minimum number of iterations
	while (milliseconds < 150.f) {
		iterations *= 2;
		hipEventRecord(start);

		benchmark<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_X, d_Y, iterations);
		hipEventRecord(stop);
		hipEventSynchronize(stop);

		hipEventElapsedTime(&milliseconds, start, stop);
	}

	// Perform testing
	for (int i = 0; i < NUM_REPS; i++) {
		hipEventRecord(start);

		benchmark<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_X, d_Y, iterations);
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

	double bytes = sizeof(PRECISION) * 2. * iterations * 128 * THREADS_PER_BLOCK * NUM_BLOCKS / 1e9;
	float bandwidth = bytes * 1e3 / median;

	cout << bandwidth << " GB/s" << endl;

	hipEventDestroy(start);
	hipEventDestroy(stop);

	hipFree(d_X);

	return 0;
}
