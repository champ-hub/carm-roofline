// Run in isolation to verify that the target executes the selected RVV dialect.
#include <stdint.h>
#include <stdio.h>

int main(void)
{
    uint64_t input[8] = {
            UINT64_C(0x123456789abcdef0),
            UINT64_C(0x0fedcba987654321),
            UINT64_C(0xa5a5a5a5a5a5a5a5),
            UINT64_C(0x5a5a5a5a5a5a5a5a),
            UINT64_C(0x1111222233334444),
            UINT64_C(0xaaaabbbbccccdddd),
            UINT64_C(0x0123456789abcdef),
            UINT64_C(0xfedcba9876543210),
    };
    uint64_t output[8] = {0};
    unsigned long vl;

    // clang-format off
    __asm__ __volatile__(
#if defined(RISCV_RVV_0_7_1)
        "li t0, 8\n\t"
        "vsetvli %[vl], t0, e64, m1\n\t"
        "vle.v v0, (%[input])\n\t"
        "vse.v v0, (%[output])\n\t"
#elif defined(RISCV_RVV)
        "vsetivli %[vl], 8, e64, m1\n\t"
        "vle64.v v0, (%[input])\n\t"
        "vse64.v v0, (%[output])\n\t"
#else
#error "Select an RVV dialect"
#endif
        : [vl] "=&r"(vl)
        : [input] "r"(input), [output] "r"(output)
        : "t0", "v0", "memory");
    // clang-format on

    if (vl == 0 || vl > 8)
        return 1;
    for (unsigned long i = 0; i < vl; ++i)
        if (output[i] != input[i])
            return 2;
    for (unsigned long i = vl; i < 8; ++i)
        if (output[i] != 0)
            return 3;
#if defined(RISCV_RVV_0_7_1)
    puts("{\"rvv_version\":\"RISCV_RVV_0_7_1\"}");
#else
    puts("{\"rvv_version\":\"RISCV_RVV\"}");
#endif
    return 0;
}
