// Detects x86 cache sizes, printing JSON directly.

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

    // Detect vendor
    char vendor[32];
    memset(vendor, 0, sizeof(vendor));

    if (__builtin_cpu_is("intel")) {
        strncpy(vendor, "GenuineIntel", sizeof(vendor) - 1);
    } else if (__builtin_cpu_is("amd")) {
        strncpy(vendor, "AuthenticAMD", sizeof(vendor) - 1);
    } else {
        strncpy(vendor, "UnknownVendor", sizeof(vendor) - 1);
    }

    // Detect and print caches
    printf("  \"caches_kib\": [");
    int first_cache = 1;

    if (strcmp(vendor, "GenuineIntel") == 0) {
        for (uint32_t i = 1; i < 8; i++) {
            uint32_t eax, ebx, ecx, edx;
            eax = 4;
            ecx = i;
            __asm__("cpuid" : "+a"(eax), "=b"(ebx), "+c"(ecx), "=d"(edx));

            int cache_type = eax & 0x1F;
            if (cache_type == 0)
                break;

            unsigned int cache_sets = ecx + 1;
            unsigned int cache_coherency_line_size = (ebx & 0xFFF) + 1;
            unsigned int cache_physical_line_partitions = ((ebx >>= 12) & 0x3FF) + 1;
            unsigned int cache_ways_of_associativity = ((ebx >>= 10) & 0x3FF) + 1;
            int cache_total_size = cache_ways_of_associativity * cache_physical_line_partitions *
                                   cache_coherency_line_size * cache_sets;
            int cache_kib = cache_total_size >> 10;

            printf("%s%d", first_cache ? "" : ", ", cache_kib);
            first_cache = 0;
        }
    } else if (strcmp(vendor, "AuthenticAMD") == 0) {
        uint32_t eax, ebx, ecx, edx;

        eax = 0x80000005;
        __asm__("cpuid" : "+a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx));
        int l1_kib = (ecx >> 24) & 0xFF;
        if (l1_kib > 0) {
            printf("%d", l1_kib);
            first_cache = 0;
        }

        eax = 0x80000006;
        __asm__("cpuid" : "+a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx));
        int l2_kib = (ecx >> 16) & 0xFFFF;
        int l3_kib = ((edx >> 18) & 0x3FFF) * 512;

        if (l2_kib > 0) {
            printf("%s%d", first_cache ? "" : ", ", l2_kib);
            first_cache = 0;
        }
        if (l3_kib > 0) {
            printf("%s%d", first_cache ? "" : ", ", l3_kib);
        }
    }

    printf("]\n");
    printf("}\n");

    return 0;
}
