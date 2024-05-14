.global clktestarm

/*
  x0 = arg0 = iteration count
*/
clktestarm:
  stp x29, x30, [sp, #-16]!
  mov x8, 1
  mov x9, 20
  eor x29, x29, x29
clktest_loop:
  add x29, x29, x8
  add x29, x29, x8
  add x29, x29, x8
  add x29, x29, x8
  add x29, x29, x8
  add x29, x29, x8
  add x29, x29, x8
  add x29, x29, x8
  add x29, x29, x8
  add x29, x29, x8
  add x29, x29, x8
  add x29, x29, x8
  add x29, x29, x8
  add x29, x29, x8
  add x29, x29, x8
  add x29, x29, x8
  add x29, x29, x8
  add x29, x29, x8
  add x29, x29, x8
  add x29, x29, x8
  sub x0, x0, x9
  cbnz x0, clktest_loop
  ldp x29, x30, [sp], 16
  ret
  