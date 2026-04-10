// Detects x86 ISA features, printing JSON directly.

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>
#endif

int main(void)
{
    printf("{\n");

    // Detect vendor and print it
    char vendor[32];
    memset(vendor, 0, sizeof(vendor));

    if (__builtin_cpu_is("intel")) {
        strncpy(vendor, "GenuineIntel", sizeof(vendor) - 1);
    } else if (__builtin_cpu_is("amd")) {
        strncpy(vendor, "AuthenticAMD", sizeof(vendor) - 1);
    } else {
        strncpy(vendor, "UnknownVendor", sizeof(vendor) - 1);
    }
    printf("  \"vendor\": \"%s\",\n", vendor);

    // Detect and print ISAs
    printf("  \"isa\": [");
    printf("\"x86\""); // Base x86 ISA always present

    if (__builtin_cpu_supports("avx512f")) {
        printf(", \"x86_avx512\"");
    }
    if (__builtin_cpu_supports("avx2")) {
        printf(", \"x86_avx2\"");
    }
    if (__builtin_cpu_supports("sse")) {
        printf(", \"x86_sse\"");
    }
    printf("]\n");

    printf("}\n");

    return 0;
}
