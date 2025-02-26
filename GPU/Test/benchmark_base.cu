#include <cuda.h>
#include <cuda_runtime.h>

// DEFINE ITERATIONS

// DEFINE PRECISION

__global__ void benchmark(PRECISION *d_X) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	// DEFINE INITIALIZATION
	PRECISION a = 1.f;
	PRECISION b = 2.f;
	PRECISION c = 3.f;
	PRECISION d = 4.f;

	// DEFINE LOOP
	for (int i = 0; i < ITERATIONS; i++) {
		a = a * a + b;
		b = b * b + c;
		c = c * c + d;
		d = d * d + a;
	}

	d_X[id] = d;
}