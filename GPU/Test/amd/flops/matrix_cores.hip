#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <iostream>
#include <vector>

// DEFINE NUM_REPS

// DEFINE KERNEL PARAMETERS

// DEFINE PRECISION

// DEFINE DEVICE

using namespace std;

__global__ void benchmark(int iter, PRECISION *dummy)
{

	// DEFINE INITIALIZATION

	resultType result = {0};

	for(int i = 0; i< iter; ++i) {
#pragma unroll
		for (int j = 0; j < 128; ++j) {
			// DEFINE LOOP
		}
	}

		dummy[0] = result[0];
}

int main() {
	hipSetDevice(DEVICE);

	int iterations = 1;
	float milliseconds = 0;
	vector<float> time_series;

	PRECISION *d_X;
	hipMalloc((void **)&d_X, 64 * sizeof(PRECISION));

	// Timers
	hipEvent_t start, stop;
	hipEventCreate(&start);
	hipEventCreate(&stop);

	// Determine the minimum number of iterations
	while (milliseconds < 150.f) {
		iterations *= 2;
		hipEventRecord(start);

		benchmark<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(iterations, d_X);
		hipEventRecord(stop);
		hipEventSynchronize(stop);

		hipEventElapsedTime(&milliseconds, start, stop);
	}

	// Perform testing
	for (int i = 0; i < NUM_REPS; i++) {
		hipEventRecord(start);

		benchmark<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(iterations, d_X);
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

	double flops = (uint64_t)2 * M * N * K * 128 * iterations * (THREADS_PER_BLOCK / 32 / 2) * NUM_BLOCKS / 1e9;
	float perf = flops * 1e3 / median;

	cout << perf << " GFLOPS/s" << endl;

	hipEventDestroy(start);
	hipEventDestroy(stop);

	hipFree(d_X);

	return 0;

    return 0;
}