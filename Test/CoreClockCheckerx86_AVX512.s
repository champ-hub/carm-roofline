.global clktestx86

/*
  %rdi = arg0 = iteration count
*/
clktestx86:
  push %rbx
  push %r8
  push %r9
  mov $1, %r8
  mov $20, %r9
  xor %rbx, %rbx
  vpxord %zmm0, %zmm0, %zmm0
  vpxord %zmm1, %zmm1, %zmm1
  vpbroadcastd %r8d, %zmm0
clktest_loop:
  vpaddq %zmm0, %zmm1, %zmm1
  vpaddq %zmm0, %zmm1, %zmm1
  vpaddq %zmm0, %zmm1, %zmm1
  vpaddq %zmm0, %zmm1, %zmm1
  vpaddq %zmm0, %zmm1, %zmm1
  vpaddq %zmm0, %zmm1, %zmm1
  vpaddq %zmm0, %zmm1, %zmm1
  vpaddq %zmm0, %zmm1, %zmm1
  vpaddq %zmm0, %zmm1, %zmm1
  vpaddq %zmm0, %zmm1, %zmm1
  vpaddq %zmm0, %zmm1, %zmm1
  vpaddq %zmm0, %zmm1, %zmm1
  vpaddq %zmm0, %zmm1, %zmm1
  vpaddq %zmm0, %zmm1, %zmm1
  vpaddq %zmm0, %zmm1, %zmm1
  vpaddq %zmm0, %zmm1, %zmm1
  vpaddq %zmm0, %zmm1, %zmm1
  vpaddq %zmm0, %zmm1, %zmm1
  vpaddq %zmm0, %zmm1, %zmm1
  vpaddq %zmm0, %zmm1, %zmm1
  sub %r9, %rdi
  jnz clktest_loop
  pop %r9
  pop %r8
  pop %rbx
  ret
.section .note.GNU-stack,"",%progbits
