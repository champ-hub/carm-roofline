#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// DEFINE PRECISION

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