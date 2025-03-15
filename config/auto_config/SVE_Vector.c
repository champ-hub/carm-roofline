#include <stdio.h>

unsigned long detect_sve_max_vector_length() {
    unsigned long max_vl;
    asm volatile (
        "cntb %[result] \n\t" //Count the number of bytes in one vector register
        : [result] "=r" (max_vl)
        :
        :
    );
    return max_vl;
}

int main() {
    unsigned long max_vl = detect_sve_max_vector_length();
    printf("%d", max_vl/8);
    return 0;
}

