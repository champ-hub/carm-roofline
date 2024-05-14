COMPILER = gcc
FLAGS = -g -Wall -Wextra -O2

ifeq ($(isa),sse)
	FLAGS += -DSSE
else ifeq ($(isa),avx)
	FLAGS += -DAVX
else ifeq ($(isa),avx2)
	FLAGS += -DAVX2
else ifeq ($(isa),avx512)
	FLAGS += -DAVX512
else ifeq ($(isa),neon)
	FLAGS += -DNEON
else ifeq ($(isa),armscalar)
	FLAGS += -DASCALAR
else ifeq ($(isa),riscvscalar)
	FLAGS += -DRISCVSCALAR
else ifeq ($(isa),riscvvector)
	FLAGS += -DRISCVVECTOR
else
	FLAGS += -DSCALAR
endif

Bench:
	$(COMPILER) $(FLAGS) Bench.c select_op.c create_bench.c write_asm.c calc_param.c -o Bench -lm -Wno-unused-variable -Wno-unused-parameter

clean:
	rm -f Bench
