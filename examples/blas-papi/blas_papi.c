#include <math.h>
#include <papi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 192
#define BLOCK 24
#define VECTOR_N (1 << 20)

static double A[N * N], B[N * N], transposed_B[N * N], C[N * N], reference[N * N];
static double x[VECTOR_N], y[VECTOR_N], z[VECTOR_N], expected[VECTOR_N];
static volatile double result_sink;

static int begin_region(const char *name)
{
    int status = PAPI_hl_region_begin(name);
    if (status != PAPI_OK) {
        fprintf(stderr, "PAPI region begin '%s' failed: %s\n", name, PAPI_strerror(status));
    }
    return status;
}

static int end_region(const char *name)
{
    int status = PAPI_hl_region_end(name);
    if (status != PAPI_OK) {
        fprintf(stderr, "PAPI region end '%s' failed: %s\n", name, PAPI_strerror(status));
    }
    return status;
}

static void initialize(void)
{
    for (int i = 0; i < VECTOR_N; ++i) {
        x[i] = (double) (i % 13 - 6) / 7.0;
        y[i] = (double) (i % 9 - 4) / 5.0;
    }
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            A[i * N + j] = (double) ((i + 2 * j) % 17 - 8) / 11.0;
            B[i * N + j] = (double) ((3 * i + j) % 19 - 9) / 13.0;
        }
}

static int check_vector(const char *name, const double *actual, const double *expected_values, int length)
{
    double sum = 0.0;
    for (int i = 0; i < length; ++i) {
        if (fabs(actual[i] - expected_values[i]) > 1e-10 * (1.0 + fabs(expected_values[i]))) {
            fprintf(stderr, "%s verification failed at %d: %.12g != %.12g\n", name, i, actual[i], expected_values[i]);
            return 0;
        }
        sum += actual[i];
    }
    result_sink += sum;
    printf("%-24s checksum %.12g\n", name, sum);
    return 1;
}

static int check_matrix(const char *name)
{
    double sum = 0.0;
    for (int i = 0; i < N * N; ++i) {
        if (fabs(C[i] - reference[i]) > 1e-10 * (1.0 + fabs(reference[i]))) {
            fprintf(stderr, "%s verification failed at %d: %.12g != %.12g\n", name, i, C[i], reference[i]);
            return 0;
        }
        sum += C[i];
    }
    result_sink += sum;
    printf("%-24s checksum %.12g\n", name, sum);
    return 1;
}

int main(void)
{
    int papi_status = PAPI_library_init(PAPI_VER_CURRENT);
    if (papi_status != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI initialization failed: %s\n", PAPI_strerror(papi_status));
        return EXIT_FAILURE;
    }
    initialize();

    for (int i = 0; i < VECTOR_N; ++i)
        expected[i] = y[i] + 2.0 * x[i];
    if (begin_region("blas_daxpy") != PAPI_OK)
        return EXIT_FAILURE;
    for (int i = 0; i < VECTOR_N; ++i)
        z[i] = y[i] + 2.0 * x[i];
    if (end_region("blas_daxpy") != PAPI_OK || !check_vector("blas_daxpy", z, expected, VECTOR_N))
        return EXIT_FAILURE;

    double dot_expected = 0.0;
    for (int i = 0; i < VECTOR_N; ++i)
        dot_expected += x[i] * y[i];
    double dot = 0.0;
    if (begin_region("blas_dot") != PAPI_OK)
        return EXIT_FAILURE;
    for (int i = 0; i < VECTOR_N; ++i)
        dot += x[i] * y[i];
    if (end_region("blas_dot") != PAPI_OK || fabs(dot - dot_expected) > 1e-10 * (1.0 + fabs(dot_expected))) {
        fprintf(stderr, "blas_dot verification failed\n");
        return EXIT_FAILURE;
    }
    result_sink += dot;
    printf("%-24s value %.12g\n", "blas_dot", dot);

    for (int i = 0; i < VECTOR_N; ++i)
        expected[i] = 3.0 * x[i];
    if (begin_region("blas_scal") != PAPI_OK)
        return EXIT_FAILURE;
    for (int i = 0; i < VECTOR_N; ++i)
        z[i] = 3.0 * x[i];
    if (end_region("blas_scal") != PAPI_OK || !check_vector("blas_scal", z, expected, VECTOR_N))
        return EXIT_FAILURE;

    for (int i = 0; i < N; ++i) {
        double sum = 0.0;
        for (int j = 0; j < N; ++j)
            sum += A[i * N + j] * x[j];
        reference[i] = sum;
    }
    if (begin_region("blas_gemv") != PAPI_OK)
        return EXIT_FAILURE;
    for (int i = 0; i < N; ++i) {
        double sum = 0.0;
        for (int j = 0; j < N; ++j)
            sum += A[i * N + j] * x[j];
        z[i] = sum;
    }
    if (end_region("blas_gemv") != PAPI_OK || !check_vector("blas_gemv", z, reference, N))
        return EXIT_FAILURE;

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            transposed_B[j * N + i] = B[i * N + j];
            double sum = 0.0;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * N + j];
            reference[i * N + j] = sum;
        }
    if (begin_region("blas_gemm_naive") != PAPI_OK)
        return EXIT_FAILURE;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    if (end_region("blas_gemm_naive") != PAPI_OK || !check_matrix("blas_gemm_naive"))
        return EXIT_FAILURE;

    if (begin_region("blas_gemm_blocked_transposed") != PAPI_OK)
        return EXIT_FAILURE;
    for (int i = 0; i < N * N; ++i)
        C[i] = 0.0;
    for (int ii = 0; ii < N; ii += BLOCK)
        for (int jj = 0; jj < N; jj += BLOCK)
            for (int kk = 0; kk < N; kk += BLOCK)
                for (int i = ii; i < ii + BLOCK; ++i)
                    for (int j = jj; j < jj + BLOCK; ++j)
                        for (int k = kk; k < kk + BLOCK; ++k)
                            C[i * N + j] += A[i * N + k] * transposed_B[j * N + k];
    if (end_region("blas_gemm_blocked_transposed") != PAPI_OK || !check_matrix("blas_gemm_blocked_transposed"))
        return EXIT_FAILURE;

    printf("result_sink %.12g\n", result_sink);
    return EXIT_SUCCESS;
}
