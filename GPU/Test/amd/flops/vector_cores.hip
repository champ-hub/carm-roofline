#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <iostream>
#include <vector>

// DEFINE NUM_REPS

// DEFINE KERNEL PARAMETERS

// DEFINE PRECISION

// DEFINE DEVICE

// DEFINE TEST

#define HIP_CHECK(expression)                  \
{                                              \
    const hipError_t status = expression;      \
    if(status != hipSuccess){                  \
        std::cerr << "HIP error "              \
                  << status << ": "            \
                  << hipGetErrorString(status) \
                  << " at " << __FILE__ << ":" \
                  << __LINE__ << std::endl;    \
    }                                          \
}

using namespace std;

__global__ void flops_benchmark(PRECISION *buf, uint32_t nSize, int iterations)
{
    const uint32_t gid = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    const uint32_t nThreads  = gridDim.x * blockDim.x;
    const uint32_t nEntriesPerThread = (uint32_t) nSize / nThreads;
    const uint32_t maxOffset = nEntriesPerThread * nThreads;

    PRECISION *ptr;
    const PRECISION y = (PRECISION) 1.0;

    ptr = &buf[gid];
    PRECISION x = (PRECISION) 2.0;

    for(uint32_t offset=0; offset < maxOffset; offset += nThreads)
    {
        for(int j=0; j<iterations; j++)
        {
#pragma unroll
            for (int i = 0; i < 128; i++)
                x = ptr[offset] * x + y;
        }
    }

    ptr[0] = -x;

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

		flops_benchmark<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_X, NUM_BLOCKS*THREADS_PER_BLOCK, iterations);
		hipEventRecord(stop);
		hipEventSynchronize(stop);

		hipEventElapsedTime(&milliseconds, start, stop);
	}

	// Perform testing
	for (int i = 0; i < NUM_REPS; i++) {
		hipEventRecord(start);

		flops_benchmark<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_X, NUM_BLOCKS*THREADS_PER_BLOCK, iterations);
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

	double flops = MULTIPLIER * iterations * 128. * THREADS_PER_BLOCK * NUM_BLOCKS / 1e9;
	float perf = flops * 1e3 / median;

	cout << perf << " GFLOPS/s" << endl;

	hipEventDestroy(start);
	hipEventDestroy(stop);

	hipFree(d_X);

	return 0;

    return 0;
}