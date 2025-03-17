#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#if defined(__x86_64__) || defined(_M_X64)
    #include <cpuid.h>
#endif
#include <string.h>
#include <stdbool.h>

#define MAX_FREQ_LEN 20

int get_num_cpus() {
    FILE *fp = fopen("/proc/cpuinfo", "r");
    if (fp == NULL) {
        perror("Failed to open /proc/cpuinfo");
        exit(1);
    }

    int num_cpus = 0;
    char line[100];
    while (fgets(line, sizeof(line), fp) != NULL) {
        if (strstr(line, "processor") != NULL) {
            num_cpus++;
        }
    }

    fclose(fp);
    return num_cpus;
}

void set_cpu_frequency(int new_freq) {
    int num_cpus = get_num_cpus();
    char freq_path[MAX_FREQ_LEN];
    int checker = 0;
    int previous_max_freq = 0;

    for (int cpu = 0; cpu < num_cpus; cpu++) {
        char filename[100];
        snprintf(filename, sizeof(filename), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_max_freq", cpu);

        FILE *fp = fopen(filename, "r+");

        if (fscanf(fp, "%d", &previous_max_freq) != 1) {
        perror("Failed to read integer from file");
        fclose(fp);
        exit(1);
        }

        fprintf(stderr, "Previous Max Frequency for CPU %d: %d\n", cpu, previous_max_freq);

        if (fp == NULL) {
            perror("Failed to open scaling_max_freq file");
            exit(1);
        }

         // Set your new max frequency value here
        fprintf(fp, "%d", new_freq);
        fseek(fp, 0, SEEK_SET);
        fclose(fp);

        fp = fopen(filename, "r");
        if (fp == NULL) {
            perror("Failed to reopen scaling_max_freq file");
            exit(1);
        }

        if (fscanf(fp, "%d", &checker) != 1) {
            perror("Failed to read integer from file");
            fclose(fp);
            exit(1);
        }

        fclose(fp);

        fprintf(stderr, "New Max Frequency for CPU %d: %d\n\n", cpu, checker);
    }
}
#if defined(__aarch64__) || defined(_M_ARM64)
bool neon_supported(void) {
    uint64_t id_aa64pfr0_el1;
    // Read the ID_AA64PFR0_EL1 register into id_aa64pfr0_el1.
    __asm__("mrs %0, ID_AA64PFR0_EL1" : "=r" (id_aa64pfr0_el1));

    // The Advanced SIMD (NEON) feature field is located in bits [23:20].
    // A value of 0xF (binary 1111) in this field means NEON is not implemented.
    const uint8_t ADV_SIMD_MASK = 0xF;
    const uint8_t advsimd_field = (id_aa64pfr0_el1 >> 20) & ADV_SIMD_MASK;

    // NEON is supported if the field is not equal to 0xF.
    return (advsimd_field != ADV_SIMD_MASK);
}
#endif

int main(int argc, char *argv[]) {

     if (argc != 3) {
        fprintf(stderr, "Usage: %s <new_max_freq>\n", argv[1]);
        fprintf(stderr, "Usage: %s <set_freq>\n", argv[2]);
        return 1;
    }

    int new_max_freq = atoi(argv[1]);
    int set_freq = atoi(argv[2]);

    if (new_max_freq > 0 && set_freq >0){
        set_cpu_frequency(new_max_freq);
    }
    #if defined(__x86_64__) || defined(_M_X64)
        //Automatic ISA detection
        char isaSSE[4] = "";
        char isaAVX2[5] = "";
        char isaAVX512[7] = "";
        int sseSupported = 0, avxSupported = 0;

        avxSupported = __builtin_cpu_supports("avx");
        sseSupported = __builtin_cpu_supports("sse");
        
        uint32_t cpuidEax, cpuidEbx, cpuidEcx, cpuidEdx;
        __cpuid_count(7, 0, cpuidEax, cpuidEbx, cpuidEcx, cpuidEdx);
        if (cpuidEbx & (1UL << 16)) {
            strcpy(isaAVX512, "avx512");
        }
        if (avxSupported)
        {
            strcpy(isaAVX2, "avx2");
        }
        if (sseSupported)
        {
            strcpy(isaSSE, "sse");
        }

        printf("%s\n", isaAVX512);
        printf("%s\n", isaAVX2);
        printf("%s\n", isaSSE);

        //Automatic CPU vendor Detection (Intel or AMD)
        char vendor[] = "UnknownVendor";

        if(__builtin_cpu_is("intel")){
            strcpy(vendor, "GenuineIntel");
        }else if (__builtin_cpu_is("amd")){
            strcpy(vendor, "AuthenticAMD");
        }

        printf("%s\n", vendor);


        //Automatic cache size detection Intel
        if (strcmp(vendor, "GenuineIntel") == 0){
            int i;
            for (i = 1; i < 5; i++) {

                // Variables to hold the contents of the 4 i386 legacy registers
                uint32_t eax, ebx, ecx, edx; 

                eax = 4; // get cache info
                ecx = i; // cache id

                __asm__ (
                    "cpuid" // call i386 cpuid instruction
                    : "+a" (eax) // contains the cpuid command code, 4 for cache query
                    , "=b" (ebx)
                    , "+c" (ecx) // contains the cache id
                    , "=d" (edx)
                ); // generates output in 4 registers eax, ebx, ecx and edx 

                // See the page 3-191 of the manual.
                int cache_type = eax & 0x1F; 

                if (cache_type == 0) // end of valid cache identifiers
                    break;

                char * cache_type_string;
                switch (cache_type) {
                    case 1: cache_type_string = "Data Cache"; break;
                    case 2: cache_type_string = "Instruction Cache"; break;
                    case 3: cache_type_string = "Unified Cache"; break;
                    default: cache_type_string = "Unknown Type Cache"; break;
                }

                int cache_level = (eax >>= 5) & 0x7;

                // See the page 3-192 of the manual.
                // ebx contains 3 integers of 10, 10 and 12 bits respectively
                unsigned int cache_sets = ecx + 1;
                unsigned int cache_coherency_line_size = (ebx & 0xFFF) + 1;
                unsigned int cache_physical_line_partitions = ((ebx >>= 12) & 0x3FF) + 1;
                unsigned int cache_ways_of_associativity = ((ebx >>= 10) & 0x3FF) + 1;

                // Total cache size is the product
                int cache_total_size = cache_ways_of_associativity * cache_physical_line_partitions * cache_coherency_line_size * cache_sets;

                printf("%d\n", cache_total_size >> 10);
            }
            //Automatic cache size detection AMD
        }else if (strcmp(vendor, "AuthenticAMD") == 0)
        {
            uint32_t eax, ebx, ecx, edx;
            // L1
                eax = 0x80000005; //the specific code of the cpuid instruction for L1

                __asm__ (
                    "cpuid"
                    : "+a" (eax)
                    , "=b" (ebx)
                    , "=c" (ecx)
                    , "=d" (edx)
                );

                uint32_t
                    dataCache_size = (ecx >> 24) & 0xFF,
                    dataCache_associativity = (ecx >> 16) & 0xFF,
                    dataCache_linesPerTag = (ecx >> 8) & 0xFF,
                    dataCache_lineSize = ecx & 0xFF;

                printf("%d\n", dataCache_size);
            

            // L2
                eax = 0x80000006; // the specific code of the cpuid instruction for L1

                __asm__ (
                    "cpuid"
                    : "+a" (eax)
                    , "=b" (ebx)
                    , "=c" (ecx)
                    , "=d" (edx)
                );

                uint32_t
                    L2_size = (ecx >> 16) & 0xFFFF,
                    L2_associativity = (ecx >> 12) & 0xF,
                    L2_linesPerTag = (ecx >> 8) & 0xF,
                    L2_lineSize = ecx & 0xFF;


                uint32_t
                    L3_size = (edx >> 18) & 0x3FFF,
                    L3_associativity = (edx >> 12) & 0xF,
                    L3_linesPerTag = (edx >> 8) & 0xF,
                    L3_lineSize = (edx >> 0) & 0xFF;

                printf("%d\n", L2_size);
                printf("%d\n", L3_size*512);
            
        }
        else{
            printf("0\n");
            printf("0\n");
            printf("0\n");
        }
    #elif defined(__aarch64__) || defined(_M_ARM64)
        bool arm_sve = neon_supported();
        if (arm_sve){
            printf("neon\n");
        }else{
            printf("armscalar\n");
        }
    #endif
    
    return 0;
}