#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <iostream>

using namespace::std;

//DEFINE KERNEL PARAMETERS

//DEFINE PRECISION

//DEFINE DEVICE

__global__ void benchmark(float *d_X);

int main() {
	// Allocate memory in GPU
	cudaSetDevice(DEVICE);
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

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	
	cout << milliseconds << endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(d_X);
	return 0;
}