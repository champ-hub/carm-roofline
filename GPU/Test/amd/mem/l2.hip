#include <hip/hip_runtime.h>

#include <iostream>
#include <vector>

#define PRECISION float
#define THREADS_PER_BLOCK 1024
#define NUM_BLOCKS 32768

#define NUM_REPS 64

#define DEVICE 0

using namespace std;

__global__ void benchmark(const PRECISION *d_X, PRECISION *dummy, int iterations, int cacheSize) {
	int id = threadIdx.x;

	PRECISION sink;
	sink = 0;

	for (int i = 0; i < iterations; ++i)
#pragma unroll
		for (int j = 0; j < cacheSize; j += THREADS_PER_BLOCK) sink += d_X[id + j];

	dummy[id] = sink;
}

int main() {
	hipSetDevice(DEVICE);

	hipDeviceProp_t deviceProps;
	hipGetDeviceProperties(&deviceProps, DEVICE);

	int iterations = 1;
	float milliseconds = 0;
	vector<float> time_series;

	uint64_t csize = deviceProps.l2CacheSize;

	PRECISION *d_X;
	hipMalloc((void **)&d_X, 32 * deviceProps.l2CacheSize);

	// Timers
	hipEvent_t start, stop;
	hipEventCreate(&start);
	hipEventCreate(&stop);

	// Determine the minimum number of iterations
	while (milliseconds < 150.f) {
		iterations *= 2;
		hipEventRecord(start);

		benchmark<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_X, d_X + csize, iterations,
													 csize / sizeof(PRECISION));
		hipEventRecord(stop);
		hipEventSynchronize(stop);

		hipEventElapsedTime(&milliseconds, start, stop);
	}

	// Perform testing
	for (int i = 0; i < NUM_REPS; i++) {
		hipEventRecord(start);

		benchmark<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_X, d_X + csize, iterations,
													 csize / sizeof(PRECISION));
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

	double bytes = NUM_BLOCKS * iterations * csize / 1e9;
	float bandwidth = bytes * 1e3 / median;

	cout << bandwidth << " GB/s" << endl;

	hipEventDestroy(start);
	hipEventDestroy(stop);

	hipFree(d_X);

	return 0;

	return 0;
}