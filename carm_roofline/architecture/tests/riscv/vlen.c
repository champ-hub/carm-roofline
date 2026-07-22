// Detects the RISC-V Vector Extension VLEN in bytes
#include <stdint.h>
#include <stdio.h>

int detect_rvv_vlen_bytes(void)
{
    // Use vsetvli with e64,m1 to read VL (elements), then convert to bytes
    register unsigned long vl asm("t0") = 8192; // large enough request
    __asm volatile("vsetvli %[vl], %[vl], e64, m1\n\t" : [vl] "+r"(vl) : : "v0", "t1", "t2");
    // 'vl' now holds element count for 64-bit elems; bytes = vl * 8
    return (int) (vl * 8UL);
}

int main(void)
{
    int vlen_bytes = detect_rvv_vlen_bytes();
    printf("{\"vector_length\": %d}\n", vlen_bytes);

    return 0;
}
