// Detects Arm ISA features
#include <stdint.h>
#include <stdio.h>

#define NEON_BIT 1 << 0
#define SVE_BIT 1 << 1

static int cpu_features(void)
{
    int features = 0;
    uint64_t id_aa64pfr0_el1;
    __asm__("mrs %0, ID_AA64PFR0_EL1" : "=r"(id_aa64pfr0_el1));

    const uint8_t ADV_SIMD_MASK = 0xF;
    const uint8_t advsimd_field = (id_aa64pfr0_el1 >> 20) & ADV_SIMD_MASK;
    if (advsimd_field != ADV_SIMD_MASK) {
        features |= NEON_BIT;
    }
    if (id_aa64pfr0_el1 >> 32 & 0xF) {
        features |= SVE_BIT;
    }
    return features;
}

int main(void)
{
    printf("{\"isa\": [\"arm\"");

    int features = cpu_features();
    if (features & NEON_BIT) {
        printf(", \"arm_neon\"");
    }
    if (features & SVE_BIT) {
        printf(", \"arm_sve\"");
    }

    printf("]}\n");

    return 0;
}
