# PAPI-instrumented BLAS kernels

A self-contained C example for profiling several basic linear-algebra kernels. It marks each kernel with a named PAPI High-Level region and verifies its result against a reference calculation before exiting. The vector kernels process $2^{20}$ elements so the measured work is substantial relative to region instrumentation.

The six regions are `blas_daxpy`, `blas_dot`, `blas_scal`, `blas_gemv`, `blas_gemm_naive`, and `blas_gemm_blocked_transposed`. Both GEMM kernels compute the same `C = A * B` result for 192-by-192 matrices. Before either timed region, the program computes a reference result and pretransposes `B`; the blocked kernel uses that transposed copy for its inner product.

## Build and run

Requires a C compiler and PAPI development headers/library.

```sh
make
./blas_papi
```

The program prints a checksum/value for each verified region. A nonzero exit status indicates PAPI setup, region instrumentation, or result verification failed. The printed results and volatile aggregate sink ensure calculated values are observable.

## Profile

Run from the repository root with PAPI and merge measurements by region name:

```sh
carm profile --backend papi --aggregation region_merged --data-type f64 -- examples/blas-papi/blas_papi
```

`region_merged` produces one profile point for each named kernel region.
