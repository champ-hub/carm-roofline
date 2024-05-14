#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <signal.h>

void illegal_opcode_handler(int signum) {
    printf("Illegal opcode occurred. Handling the error gracefully...\n");
    exit(1);
}

int main(int argc, char* argv[])
{
    int vec_length = 0;
    int elem_length = 0;

    signal(SIGILL, illegal_opcode_handler);

    if (argc > 1 && strcmp(argv[1], "dp") == 0) {
        elem_length = 8;
        asm volatile
        (
            "li         t0, 8192\n\t"
            "vsetvli    t0, t0, e64, m1\n\t"
            "sw         t0, %[vl]\n\t"

            :
            :   [vl] "m" (vec_length)
            :   "t0", "t1", "t2"
        );
    } else if (argc > 1 && strcmp(argv[1], "sp") == 0) {
        elem_length = 4;
        asm volatile
        (
            "li         t0, 8192\n\t"
            "vsetvli    t0, t0, e32, m1\n\t"
            "sw         t0, %[vl]\n\t"

            :
            :   [vl] "m" (vec_length)
            :   "t0", "t1", "t2"
        );
    }
    printf("%d", vec_length);

    return 0;
}
