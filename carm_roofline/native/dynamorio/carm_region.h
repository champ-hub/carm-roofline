#ifndef CARM_REGION_H
#define CARM_REGION_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Link carm_region_timing.c into the application; timing is enabled only when
 * CARM_REGION_TIMING_OUTPUT names an output file. */
#if defined(__GNUC__)
#define CARM_REGION_NOINLINE __attribute__((noinline, visibility("default")))
#else
#define CARM_REGION_NOINLINE
#endif

CARM_REGION_NOINLINE void carm_region_begin(const char *name);
CARM_REGION_NOINLINE void carm_region_end(const char *name);
CARM_REGION_NOINLINE void carm_region_set_process_identity(uint64_t process_index, uint64_t process_id);
CARM_REGION_NOINLINE void carm_region_set_thread_identity(uint64_t thread_index, uint64_t thread_id);

#ifdef __cplusplus
}
#endif
#endif
