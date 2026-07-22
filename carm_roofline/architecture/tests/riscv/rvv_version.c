// Probe for detecting RISC-V Vector Extension version (0.7.1 vs 1.0).
// Meant to be compiled with each flag to test which version is supported by the compiler (and hence
// the machine).

#include <stdint.h>

int main(void)
{
    uint64_t buf[8] = {0};

    // clang-format off
    __asm__ __volatile__(
        "vsetivli x0, 8, e64, m1\n\t"
#if defined(RISCV_RVV_0_7_1)
        // RVV 0.7.1 unit-stride load (type-less):
        "vle.v v0, (%[p])\n\t"
#elif defined(RISCV_RVV)
        // RVV 1.0 unit-stride load (typed by element width):
        "vle64.v v0, (%[p])\n\t"
#endif
        :
        : [p] "r"(buf)
        : "v0", "memory");
    // clang-format on

    return 0;
}
