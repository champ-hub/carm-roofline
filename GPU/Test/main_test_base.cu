#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include <iostream>

using namespace std;

// DEFINE KERNEL PARAMETERS

// DEFINE PRECISION

// DEFINE DEVICE

__global__ void benchmark(float *d_X);

int main() {
	// Allocate memory in GPU
	cudaSetDevice(DEVICE);
	PRECISION *h_X = (PRECISION *)malloc(NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(PRECISION));
	PRECISION *d_X;
	cudaMalloc((void **)&d_X, NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(PRECISION));

	// Timers
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	benchmark<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_X);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaMemcpy(h_X, d_X, NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(PRECISION),
			   cudaMemcpyDeviceToHost);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	double flops = 2. * 4 * ITERATIONS * THREADS_PER_BLOCK * NUM_BLOCKS / 1e9;
	float perf = flops * 1e3 / milliseconds;

	cout << perf << " GFLOPS" << endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(d_X);
	return 0;
}