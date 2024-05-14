.global clktestriscv

/*
  a0 = arg0 = iteration count
*/
clktestriscv:
  addi sp, sp, -32
  sd ra, 0(sp)
  sd t0, 8(sp)
  sd t1, 16(sp)
  sd t2, 24(sp)
  li t0, 0
  li t1, 1
  li t2, 20
clktest_loop:
  add t0, t0, t1
  add t0, t0, t1
  add t0, t0, t1
  add t0, t0, t1
  add t0, t0, t1
  add t0, t0, t1
  add t0, t0, t1
  add t0, t0, t1
  add t0, t0, t1
  add t0, t0, t1
  add t0, t0, t1
  add t0, t0, t1
  add t0, t0, t1
  add t0, t0, t1
  add t0, t0, t1
  add t0, t0, t1
  add t0, t0, t1
  add t0, t0, t1
  add t0, t0, t1
  add t0, t0, t1
  sub a0, a0, t2
  bnez a0, clktest_loop
  ld ra, 0(sp)
  ld t0, 8(sp)
  ld t1, 16(sp)
  ld t2, 24(sp)
  addi sp, sp, 32
  ret

