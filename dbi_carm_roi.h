#ifndef CARM_ROI_H
#define CARM_ROI_H

#include <time.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

#ifndef __SSC_MARK
#if defined(__x86_64__) || defined(_M_X64)
    #define __SSC_MARK(tag) __asm__ __volatile__("movl %0, %%ebx; .byte 0x64, 0x67, 0x90 "::"i"(tag) : "%ebx")
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define __SSC_MARK(tag) __asm__ __volatile__("mov x9, %0; .inst 0x9b03e03f"::"i"(tag) : "%x9")
#endif
#endif

struct timespec t_start, t_end;
double timing_duration = 0.0;
bool register_atexit = false;

void print_results()
{
        char outfile[256];
        sprintf(outfile,"timing_results_%d.txt",getpid());
        FILE *timing_results_file = fopen(outfile, "w");
        if (timing_results_file != NULL) {
                fprintf(timing_results_file, "Time Taken:\t%0.9lf seconds\n", timing_duration);
                fclose(timing_results_file);
        } else {
                perror("Failed to open timing_results.txt \n");
        }
}

void CARM_roi_begin() {
        if (!register_atexit) {
                atexit(print_results);
                register_atexit = true;
        }
        clock_gettime(CLOCK_MONOTONIC, &t_start);
        __SSC_MARK(0xFACE);
}

void CARM_roi_end() {
        __SSC_MARK(0xDEAD);
        clock_gettime(CLOCK_MONOTONIC, &t_end);
        timing_duration += ((t_end.tv_sec + (double) t_end.tv_nsec / 1000000000) - (t_start.tv_sec + (double) t_start.tv_nsec / 1000000000));
}

#endif

