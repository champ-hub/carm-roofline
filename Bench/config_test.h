#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "functions.h"
#include <string.h>

//Loop Body Size
#define BASE_LOOP_SIZE 256
#define INST_LOOP_SIZE 256

/*-------------------------------------------------------x86 ISA------------------------------------------------------------------*/
#if defined (AVX512)
	#define ISA "avx512"
	#define NUM_REGISTER 32
	#define MEM_SP_REGISTER "zmm"
	#define MEM_DP_REGISTER "zmm"
	#define FP_SP_REGISTER "zmm"
	#define FP_DP_REGISTER "zmm"
	#define MEM_SP_REGISTER "zmm"
	#define MEM_DP_REGISTER "zmm"
	#define FP_SP_REGISTER "zmm"
	#define FP_DP_REGISTER "zmm"
	#define DP_ALIGN 64
	#define SP_ALIGN 64
	#define COBLERED "\"%%zmm0\",\"%%zmm1\",\"%%zmm2\",\"%%zmm3\",\"%%zmm4\",\"%%zmm5\",\"%%zmm6\",\"%%zmm7\",\"%%zmm8\",\"%%zmm9\",\"%%zmm10\",\"%%zmm11\",\"%%zmm12\",\"%%zmm13\",\"%%zmm14\",\"%%zmm15\",\"memory\""
	#define DP_OPS 8
	#define DP_DIV "vdivpd"
	#define DP_ADD "vaddpd"
	#define DP_MUL "vmulpd"
	#define DP_FMA "vfmadd132pd" 
	#define DP_LOAD "vmovapd"
	#define DP_STORE "vmovapd"
	#define DP_LOAD "vmovapd"
	#define DP_STORE "vmovapd"

	#define SP_OPS 16
	#define SP_DIV "vdivps"
	#define SP_ADD "vaddps"
	#define SP_MUL "vmulps"
	#define SP_FMA "vfmadd132ps"
	#define SP_LOAD "vmovaps"
	#define SP_STORE "vmovaps"
	#define SP_LOAD "vmovaps"
	#define SP_STORE "vmovaps"
#elif defined (AVX)
	#define ISA "avx"
	#define NUM_REGISTER 32
	#define MEM_SP_REGISTER "ymm"
	#define MEM_DP_REGISTER "ymm"
	#define FP_SP_REGISTER "ymm"
	#define FP_DP_REGISTER "ymm"
	#define MEM_SP_REGISTER "ymm"
	#define MEM_DP_REGISTER "ymm"
	#define FP_SP_REGISTER "ymm"
	#define FP_DP_REGISTER "ymm"
	#define DP_ALIGN 32
	#define SP_ALIGN 32
	#define COBLERED "\"%%ymm0\",\"%%ymm1\",\"%%ymm2\",\"%%ymm3\",\"%%ymm4\",\"%%ymm5\",\"%%ymm6\",\"%%ymm7\",\"%%ymm8\",\"%%ymm9\",\"%%ymm10\",\"%%ymm11\",\"%%ymm12\",\"%%ymm13\",\"%%ymm14\",\"%%ymm15\",\"memory\""
	#define DP_OPS 4
	#define DP_DIV "vdivpd"
	#define DP_ADD "vaddpd"
	#define DP_MUL "vmulpd"
	#define DP_FMA "vfmadd132pd" 
	#define DP_LOAD "vmovapd"
	#define DP_STORE "vmovapd"
	#define DP_LOAD "vmovapd"
	#define DP_STORE "vmovapd"

	#define SP_OPS 8
	#define SP_DIV "vdivps"
	#define SP_ADD "vaddps"
	#define SP_MUL "vmulps"
	#define SP_FMA "vfmadd132ps"
	#define SP_LOAD "vmovaps"
	#define SP_STORE "vmovaps"
	#define SP_LOAD "vmovaps"
	#define SP_STORE "vmovaps"
#elif defined (AVX2)
	#define ISA "avx2"
	#define NUM_REGISTER 16
	#define MEM_SP_REGISTER "ymm"
	#define MEM_DP_REGISTER "ymm"
	#define FP_SP_REGISTER "ymm"
	#define FP_DP_REGISTER "ymm"
	#define MEM_SP_REGISTER "ymm"
	#define MEM_DP_REGISTER "ymm"
	#define FP_SP_REGISTER "ymm"
	#define FP_DP_REGISTER "ymm"
	#define DP_ALIGN 32
	#define SP_ALIGN 32
	#define COBLERED "\"%%ymm0\",\"%%ymm1\",\"%%ymm2\",\"%%ymm3\",\"%%ymm4\",\"%%ymm5\",\"%%ymm6\",\"%%ymm7\",\"%%ymm8\",\"%%ymm9\",\"%%ymm10\",\"%%ymm11\",\"%%ymm12\",\"%%ymm13\",\"%%ymm14\",\"%%ymm15\",\"memory\""
	#define DP_OPS 4
	#define DP_DIV "vdivpd"
	#define DP_ADD "vaddpd"
	#define DP_MUL "vmulpd"
	#define DP_FMA "vfmadd132pd" 
	#define DP_LOAD "vmovapd"
	#define DP_STORE "vmovapd"
	#define DP_LOAD "vmovapd"
	#define DP_STORE "vmovapd"

	#define SP_OPS 8
	#define SP_DIV "vdivps"
	#define SP_ADD "vaddps"
	#define SP_MUL "vmulps"
	#define SP_FMA "vfmadd132ps"
	#define SP_LOAD "vmovaps"
	#define SP_STORE "vmovaps"
	#define SP_LOAD "vmovaps"
	#define SP_STORE "vmovaps"
#elif defined(SSE)
	#define ISA "sse"
	#define NUM_REGISTER 16
	#define MEM_SP_REGISTER "xmm"
	#define MEM_DP_REGISTER "xmm"
	#define FP_SP_REGISTER "xmm"
	#define FP_DP_REGISTER "xmm"
	#define MEM_SP_REGISTER "xmm"
	#define MEM_DP_REGISTER "xmm"
	#define FP_SP_REGISTER "xmm"
	#define FP_DP_REGISTER "xmm"
	#define DP_ALIGN 16
	#define SP_ALIGN 16
	#define COBLERED "\"%%xmm0\",\"%%xmm1\",\"%%xmm2\",\"%%xmm3\",\"%%xmm4\",\"%%xmm5\",\"%%xmm6\",\"%%xmm7\",\"%%xmm8\",\"%%xmm9\",\"%%xmm10\",\"%%xmm11\",\"%%xmm12\",\"%%xmm13\",\"%%xmm14\",\"%%xmm15\",\"memory\""
	#define DP_OPS 2
	#define DP_DIV "divpd"
	#define DP_ADD "addpd"
	#define DP_MUL "mulpd"
	#define DP_FMA "vfmadd132pd"
	#define DP_LOAD "movapd"
	#define DP_STORE "movapd"
	#define DP_LOAD "movapd"
	#define DP_STORE "movapd"

	#define SP_OPS 4
	#define SP_DIV "divps"
	#define SP_ADD "addps"
	#define SP_MUL "mulps"
	#define SP_FMA "vfmadd132ps"
	#define SP_LOAD "movaps"
	#define SP_STORE "movaps"
#elif defined(NEON)
	#define ISA "neon"
	#define NUM_REGISTER 32
	#define MEM_SP_REGISTER "d"
	#define MEM_DP_REGISTER "q"
	#define FP_SP_REGISTER ".4s"
	#define FP_DP_REGISTER ".2d"
	#define DP_ALIGN 16
	#define SP_ALIGN 16
	#define COBLERED "\"w0\", \"w1\", \"v0\", \"v1\", \"v2\", \"v3\", \"v4\", \"v5\", \"v6\", \"v7\", \"v8\", \"v9\", \"v10\", \"v11\", \"v12\", \"v13\", \"v14\", \"v15\", \"v16\", \"v17\", \"v18\", \"v19\", \"v20\", \"v21\", \"v22\", \"v23\", \"v24\", \"v25\", \"v26\", \"v27\", \"v28\", \"v29\", \"v30\", \"v31\", \"memory\""
	#define DP_OPS 2
	#define DP_DIV "fdiv"
	#define DP_ADD "fadd"
	#define DP_MUL "fmul"
	#define DP_FMA "fmla"
	#define DP_LOAD "ldr"
	#define DP_STORE "str"

	#define SP_OPS 4
	#define SP_DIV "fdiv"
	#define SP_ADD "fadd"
	#define SP_MUL "fmul"
	#define SP_FMA "fmla"
	#define SP_LOAD "ldr"
	#define SP_STORE "str"
#elif defined(ASCALAR)
	#define ISA "armscalar"
	#define NUM_REGISTER 32
	#define MEM_SP_REGISTER "s"
	#define MEM_DP_REGISTER "d"
	#define FP_SP_REGISTER "s"
	#define FP_DP_REGISTER "d"
	#define DP_ALIGN 8
	#define SP_ALIGN 4
	#define COBLERED "\"w0\", \"w1\", \"d0\", \"d1\", \"d2\", \"d3\", \"d4\", \"d5\", \"d6\", \"d7\", \"d8\", \"d9\", \"d10\", \"d11\", \"d12\", \"d13\", \"d14\", \"d15\", \"d16\", \"d17\", \"d18\", \"d19\", \"d20\", \"d21\", \"d22\", \"d23\", \"d24\", \"d25\", \"d26\", \"d27\", \"d28\", \"d29\", \"d30\", \"d31\", \"memory\""
	#define DP_OPS 1
	#define DP_DIV "fdiv"
	#define DP_ADD "fadd"
	#define DP_MUL "fmul"
	#define DP_FMA "fmadd"
	#define DP_LOAD "ldr"
	#define DP_STORE "str"

	#define SP_OPS 1
	#define SP_DIV "fdiv"
	#define SP_ADD "fadd"
	#define SP_MUL "fmul"
	#define SP_FMA "fmadd"
	#define SP_LOAD "ldr"
	#define SP_STORE "str"
#elif defined(RISCVSCALAR)
	#define ISA "riscvscalar"
	#define NUM_REGISTER 32
	#define MEM_SP_REGISTER "f"
	#define MEM_DP_REGISTER "f"
	#define FP_SP_REGISTER "f"
	#define FP_DP_REGISTER "f"
	#define DP_ALIGN 8
	#define SP_ALIGN 4
	#define COBLERED "\"f0\",\"f1\",\"f2\",\"f3\",\"f4\",\"f5\",\"f6\",\"f7\",\"f8\",\"f9\",\"f10\",\"f11\",\"f12\",\"f13\",\"f14\",\"f15\",\"f16\",\"f17\",\"f18\",\"f19\",\"f20\",\"f21\",\"f22\",\"f23\",\"f24\",\"f25\",\"f26\",\"f27\",\"f28\",\"f29\",\"f30\",\"f31\",\"memory\""
	#define DP_OPS 1
	#define DP_DIV "fdiv.d"
	#define DP_ADD "fadd.d"
	#define DP_MUL "fmul.d"
	#define DP_FMA "fmadd.d"
	#define DP_LOAD "fld"
	#define DP_STORE "fsd"

	#define SP_OPS 1
	#define SP_DIV "fdiv.s"
	#define SP_ADD "fadd.s"
	#define SP_MUL "fmul.s"
	#define SP_FMA "fmadd.s"
	#define SP_LOAD "flw"
	#define SP_STORE "fsw"
#elif defined(RVV07)
	#define ISA "rvv0.7"
	#define NUM_REGISTER 32
	#define MEM_SP_REGISTER "v"
	#define MEM_DP_REGISTER "v"
	#define FP_SP_REGISTER "v"
	#define FP_DP_REGISTER "v"
	#define DP_ALIGN 8
	#define SP_ALIGN 4
	#define COBLERED "\"v0\",\"v1\",\"v2\",\"v3\",\"v4\",\"v5\",\"v6\",\"v7\",\"v8\",\"v9\",\"v10\",\"v11\",\"v12\",\"v13\",\"v14\",\"v15\",\"v16\",\"v17\",\"v18\",\"v19\",\"v20\",\"v21\",\"v22\",\"v23\",\"v24\",\"v25\",\"v26\",\"v27\",\"v28\",\"v29\",\"v30\",\"v31\",\"memory\""
	#define DP_OPS 1
	#define DP_DIV "vfdiv.vv"
	#define DP_ADD "vfadd.vv"
	#define DP_MUL "vfmul.vv"
	#define DP_FMA "vfmadd.vv"
	#define DP_LOAD "vle.v"
	#define DP_STORE "vse.v"

	#define SP_OPS 1
	#define SP_DIV "vfdiv.vv"
	#define SP_ADD "vfadd.vv"
	#define SP_MUL "vfmul.vv"
	#define SP_FMA "vfmadd.vv"
	#define SP_LOAD "vle.v"
	#define SP_STORE "vse.v"
#elif defined(RVV1)
	#define ISA "rvv1.0"
	#define NUM_REGISTER 32
	#define MEM_SP_REGISTER "v"
	#define MEM_DP_REGISTER "v"
	#define FP_SP_REGISTER "v"
	#define FP_DP_REGISTER "v"
	#define DP_ALIGN 8
	#define SP_ALIGN 4
	#define COBLERED "\"v0\",\"v1\",\"v2\",\"v3\",\"v4\",\"v5\",\"v6\",\"v7\",\"v8\",\"v9\",\"v10\",\"v11\",\"v12\",\"v13\",\"v14\",\"v15\",\"v16\",\"v17\",\"v18\",\"v19\",\"v20\",\"v21\",\"v22\",\"v23\",\"v24\",\"v25\",\"v26\",\"v27\",\"v28\",\"v29\",\"v30\",\"v31\",\"memory\""
	#define DP_OPS 1
	#define DP_DIV "vfdiv.vv"
	#define DP_ADD "vfadd.vv"
	#define DP_MUL "vfmul.vv"
	#define DP_FMA "vfmadd.vv"
	#define DP_LOAD "vle64.v"
	#define DP_STORE "vse64.v"

	#define SP_OPS 1
	#define SP_DIV "vfdiv.vv"
	#define SP_ADD "vfadd.vv"
	#define SP_MUL "vfmul.vv"
	#define SP_FMA "vfmadd.vv"
	#define SP_LOAD "vle32.v"
	#define SP_STORE "vse32.v"
#else
	#define ISA "scalar"
	#define NUM_REGISTER 16
	#define MEM_SP_REGISTER "xmm"
	#define MEM_DP_REGISTER "xmm"
	#define FP_SP_REGISTER "xmm"
	#define FP_DP_REGISTER "xmm"
	#define MEM_SP_REGISTER "xmm"
	#define MEM_DP_REGISTER "xmm"
	#define FP_SP_REGISTER "xmm"
	#define FP_DP_REGISTER "xmm"
	#define DP_ALIGN 8
	#define SP_ALIGN 4
	#define COBLERED "\"%%xmm0\",\"%%xmm1\",\"%%xmm2\",\"%%xmm3\",\"%%xmm4\",\"%%xmm5\",\"%%xmm6\",\"%%xmm7\",\"%%xmm8\",\"%%xmm9\",\"%%xmm10\",\"%%xmm11\",\"%%xmm12\",\"%%xmm13\",\"%%xmm14\",\"%%xmm15\",\"memory\""

	#define DP_OPS 1
	#define DP_DIV "vdivsd"
	#define DP_ADD "vaddsd"
	#define DP_MUL "vmulsd"
	#define DP_FMA "vfmadd132sd"
	#define DP_LOAD "movsd"
	#define DP_STORE "movsd"
	#define DP_LOAD "movsd"
	#define DP_STORE "movsd"
	
	#define SP_OPS 1
	#define SP_DIV "vdivss"
	#define SP_ADD "vaddss"
	#define SP_MUL "vmulss"
	#define SP_FMA "vfmadd132ss"
	#define SP_LOAD "movss"
	#define SP_STORE "movss"
	#define SP_LOAD "movss"
	#define SP_STORE "movss"
#endif

