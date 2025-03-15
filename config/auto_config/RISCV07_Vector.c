#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

int main(int argc, char* argv[])
{
    int vec_length = 0;
    uint64_t dummy = 0;

    asm volatile
    (
        "li         t0, 8192\n\t"
        "vsetvli    t0, t0, e64, m1\n\t"
        "sw         t0, %[vl]\n\t"
        "vle.v v0,(%[dummy_ptr])\n\t"

        :
        :   [vl] "m" (vec_length), [dummy_ptr] "r" (&dummy)
        :   "t0", "t1", "t2"
    );

    printf("%d", vec_length);

    return 0;
}
