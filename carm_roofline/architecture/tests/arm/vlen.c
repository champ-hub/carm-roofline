// Detects the Arm SVE vector length in bytes
#include <stdint.h>
#include <stdio.h>

static unsigned long detect_sve_vector_bytes(void)
{
    unsigned long bytes = 0;
    __asm__("cntb %0" : "=r"(bytes));
    return bytes;
}

int main(void)
{
    unsigned long vlen_bytes = detect_sve_vector_bytes();
    printf("{\"vector_length\": %lu}\n", vlen_bytes);

    return 0;
}
