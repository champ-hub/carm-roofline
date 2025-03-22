#include "config_test.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//																					WRITE FP TEST
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void write_asm_fp (int long long fp, char * op, int flops, char * registr, char * assembly_op_flops_1, char * assembly_op_flops_2, char * precision, int Vlen, int LMUL, int num_runs){
	
	int i, j;
	FILE * file,*file_header;
	int long long iter;
	j = 0;
		
	char path[8192];
	snprintf(path, sizeof(path), "%s/../Test/test_params.h", PROJECT_DIR);

	file_header =  fopen(path, "w");
	file = file_header;
	
	fprintf(file_header,"#define NUM_RUNS %d\n", num_runs);
	//Specific test data
	//ARM SECTION
	#if defined(ASCALAR) || defined(NEON)
		fprintf(file_header,"#define ARM 1\n");
	//ARM SVE SECTION
	#elif defined(SVE)
		fprintf(file_header,"#define SVE 1\n");
		fprintf(file_header,"#define VLEN %d\n", Vlen);
		fprintf(file_header,"#define VLMUL %d\n", LMUL);
	//RISCV SCALAR SECTION
	#elif defined(RISCVSCALAR)
		fprintf(file_header,"#define RISCV 1\n");
	//RISCV RVV SECTION
	#elif defined(RVV07) || defined(RVV1)
		fprintf(file_header,"#define RISCVVECTOR 1\n");
		fprintf(file_header,"#define VLEN %d\n", Vlen);
		fprintf(file_header,"#define VLMUL %d\n", LMUL);
	#endif

	if(strcmp(op,"div") == 0){
		fprintf(file_header,"#define DIV 1\n");
		fprintf(file_header,"#define NUM_LD 1\n");
		fprintf(file_header,"#define NUM_ST 0\n");
		fprintf(file_header,"#define OPS %d\n",flops);
		fprintf(file_header,"#define NUM_REP 1\n");
		if(strcmp(precision, "dp") == 0){
			fprintf(file_header,"#define PRECISION double\n");
			fprintf(file_header,"#define ALIGN %d\n", (int) DP_ALIGN);
		}else{
			fprintf(file_header,"#define PRECISION float\n");
			fprintf(file_header,"#define ALIGN %d\n", (int) SP_ALIGN);
		}
	}

	fprintf(file_header,"#define FP_INST %lld\n",fp);
	
	iter = flops_math(fp); //Calculate necessary iterations
	
	//Creating Test Function
	if(strcmp(op,"div") == 0){
		fprintf(file,"static inline __attribute__((always_inline)) void test_function(PRECISION * test_var, int long long num_rep_max){\n");
	}else{
		fprintf(file,"static inline __attribute__((always_inline)) void test_function(int long long num_rep_max){\n");
	}
	
	fprintf(file,"\t__asm__ __volatile__ (\n");
	//RISCV RVV SECTION
	#if defined(RVV07) || defined(RVV1)
		if(strcmp(precision, "dp") == 0){
			int Real_Vlen = ((Vlen * 64 * LMUL) / 64);
			fprintf(file,"\t\t\"li t4, %d\\n\\t\"\n", Real_Vlen);
			fprintf(file,"\t\t\"vsetvli t0, t4, e64, m%d\\n\\t\"\n", LMUL);
		}
		else{
			int Real_Vlen = ((Vlen * 32 * LMUL) / 32);
			fprintf(file,"\t\t\"li t4, %d\\n\\t\"\n", Real_Vlen);
			fprintf(file,"\t\t\"vsetvli t0, t4, e32, m%d\\n\\t\"\n", LMUL);
		}
	#endif
	//ARM SVE SECTION
	#if defined(SVE)
	if(strcmp(precision, "dp") == 0){
		fprintf(file,"\t\t\"ptrue p0.d\\n\\t\"\n");
	}
	else{
		fprintf(file,"\t\t\"ptrue p0.s\\n\\t\"\n");
	}
	#endif
	
	
	//x86 SECTION
	#if defined(SCALAR) || defined(SSE) || defined(AVX2) || defined(AVX512)
		fprintf(file,"\t\t\"movq %%0, %%%%r8\\n\\t\\t\"\n");
		if(strcmp(op,"div") == 0){
			fprintf(file,"\t\t\"movq %%1, %%%%rax\\n\\t\\t\"\n");
			if(strcmp(precision, "dp") == 0){
				fprintf(file,"\t\t\"%s (%%%%rax), %%%%%s0\\n\\t\\t\"\n", DP_LOAD, registr);	
			}else{
				fprintf(file,"\t\t\"%s (%%%%rax), %%%%%s0\\n\\t\\t\"\n", SP_LOAD, registr);	
			}
		}
		fprintf(file,"\t\t\"Loop2_%%=:\\n\\t\\t\"\n");
	//ARM SECTION
	#elif defined(ASCALAR) || defined(NEON) || defined(SVE)
		fprintf(file,"\t\t\"mov w0, %%w0\\n\\t\"\n");
		fprintf(file,"\t\t\"Loop2_%%=:\\n\\t\"\n");
	//RISCV SECTION
	#elif defined(RISCVSCALAR) || defined(RVV07) || defined(RVV1)
		fprintf(file,"\t\t\"ld t0, %%0\\n\\t\"\n");
		fprintf(file,"\t\t\"Loop2_%%=:\\n\\t\"\n");
	#endif

	if(iter > 1){
		//x86 SECTION
		#if defined(SCALAR) || defined(SSE) || defined(AVX2) || defined(AVX512)
			fprintf(file,"\t\t\"movl $%lld, %%%%edi\\n\\t\\t\"\n",iter);
			fprintf(file,"\t\t\"Loop1_%%=:\\n\\t\\t\"\n");
		//ARM SECTION
		#elif defined(ASCALAR) || defined(NEON) || defined(SVE)
			fprintf(file,"\t\t\"ldr w1, =%lld\\n\\t\"\n",iter);
			fprintf(file,"\t\t\"Loop1_%%=:\\n\\t\"\n");
		//RISCV SECTION
		#elif defined(RISCVSCALAR) || defined(RVV07) || defined(RVV1)
			fprintf(file,"\t\t\"li t1, %lld\\n\\t\"\n", iter);
			fprintf(file,"\t\t\"Loop1_%%=:\\n\\t\"\n");
		#endif

		for(i = 0; i < BASE_LOOP_SIZE; i+=1){
			/*if(i % NUM_REGISTER == 0){
				j = 0;
			}*/
			if(j  >= NUM_REGISTER){
				j = 0;
			}
			//x86 AVX or SCALAR SECTION
			#if defined(SCALAR) || defined(AVX2) || defined(AVX512)
				if(strcmp(op,"div") == 0){
					fprintf(file,"\t\t\"%s %%%%%s0, %%%%%s%d, %%%%%s%d\\n\\t\\t\"\n", assembly_op_flops_1, registr, registr, j, registr, j);
				}else if(strcmp(op,"mad") == 0){
					if(j  >= NUM_REGISTER){
						j = 0;
					}
					fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d, %%%%%s%d\\n\\t\\t\"\n", assembly_op_flops_1, registr, j, registr, j, registr, j);
					j++;
					if(j  >= NUM_REGISTER){
						j = 0;
					}
					fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d, %%%%%s%d\\n\\t\\t\"\n", assembly_op_flops_2, registr, j, registr, j, registr, j);
				}else{	
					fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d, %%%%%s%d\\n\\t\\t\"\n", assembly_op_flops_1, registr, j, registr, j, registr, j);
				}
			//x86 SSE SECTION
			#elif defined(SSE)
				if(strcmp(op,"div") == 0){
					fprintf(file,"\t\t\"%s %%%%%s0, %%%%%s%d;\"\n", assembly_op_flops_1, registr, registr, j);
				}else if(strcmp(op,"mad") == 0){
					if(j  >= NUM_REGISTER){
						j = 0;
					}
					fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d;\"\n", assembly_op_flops_1, registr, j, registr, j);
					j++;
					if(j  >= NUM_REGISTER){
						j = 0;
					}
					fprintf(file,"\t\t\"%s %%%%%st%d, %%%%%s%d;\"\n", assembly_op_flops_2, registr, j, registr, j);
				}else if(strcmp(op,"fma") == 0){
					fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d, %%%%%s%d\\n\\t\\t\"\n", assembly_op_flops_1, registr, j, registr, j, registr, j);
				}else{
					fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d;\"\n", assembly_op_flops_1, registr, j, registr, j);
				}
			//ARM SCALAR SECTION
			#elif defined(ASCALAR)
				if(strcmp(op,"fma") == 0){
					fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr, j, registr, j, registr, j, registr, j);
				}
				else{
					fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr, j, registr, j, registr, j);
				}
			//ARM NEON SECTION
			#elif defined(NEON)
				fprintf(file,"\t\t\"%s V%d%s, V%d%s, V%d%s\\n\\t\"\n", assembly_op_flops_1, j, registr, j, registr, j, registr);
			//ARM SVE SECTION
			#elif defined(SVE)
				fprintf(file,"\t\t\"%s z%d%s, p0/m, z%d%s, z%d%s\\n\\t\"\n", assembly_op_flops_1, j, registr, j, registr, j, registr);
			//RISCV SCALAR SECTION
			#elif defined(RISCVSCALAR)
				if(strcmp(op,"fma") == 0){
					fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr, j, registr, j, registr, j, registr, j);
				}
				else{
					fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr, j, registr, j, registr, j);
				}
			//RISCV RVV SECTION
			#elif defined(RVV07) || defined(RVV1)
				fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr, j, registr, j, registr, j);
				j+=(LMUL-1);
			#endif
			j++;
			fp -= iter;
		}
		//x86 SECTION
		#if defined(SCALAR) || defined(SSE) || defined(AVX2) || defined(AVX512)
			fprintf(file,"\t\t\"subl $1, %%%%edi\\n\\t\\t\"\n");
			fprintf(file,"\t\t\"jnz Loop1_%%=\\n\\t\\t\"\n");
		//ARM SECTION
		#elif defined(ASCALAR) || defined(NEON) || defined(SVE)
			fprintf(file,"\t\t\"sub w1, w1, 1\\n\\t\"\n");
			fprintf(file,"\t\t\"cbnz w1, Loop1_%%=\\n\\t\"\n");
		//RISCV SECTION
		#elif defined(RISCVSCALAR) || defined(RVV07) || defined(RVV1)
			fprintf(file,"\t\t\"addi t1, t1, -1\\n\\t\"\n");
			fprintf(file,"\t\t\"bgtz t1, Loop1_%%=\\n\\t\"\n");
		#endif
	}
	

	
	for(i = 0; i < fp; i+=1){
		/*if(i % 16 == 0){
			j = 0;
		}*/
		if(j  >= NUM_REGISTER){
			j = 0;
		}
		//x86 AVX or SCALAR SECTION
		#if defined(SCALAR) || defined(AVX2) || defined(AVX512)
			if(strcmp(op,"div") == 0){
				fprintf(file,"\t\t\"%s %%%%%s0, %%%%%s%d, %%%%%s%d\\n\\t\\t\"\n", assembly_op_flops_1, registr, registr, j, registr, j);
			}else if(strcmp(op,"mad") == 0){
				if(j  >= NUM_REGISTER){
					j = 0;
				}
				fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d, %%%%%s%d\\n\\t\\t\"\n", assembly_op_flops_1, registr, j, registr, j, registr, j);
				j++;
				if(j  >= NUM_REGISTER){
					j = 0;
				}
				fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d, %%%%%s%d\\n\\t\\t\"\n", assembly_op_flops_2, registr, j, registr, j, registr, j);
			}else{	
				fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d, %%%%%s%d\\n\\t\\t\"\n", assembly_op_flops_1, registr, j, registr, j, registr, j);
			}	
		//x86 SSE SECTION
		#elif defined(SSE)
			if(strcmp(op,"div") == 0){
				fprintf(file,"\t\t\"%s %%%%%s0, %%%%%s%d;\"\n", assembly_op_flops_1, registr, registr, j);
			}else if(strcmp(op,"mad") == 0){
				if(j  >= NUM_REGISTER){
					j = 0;
				}
				fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d;\"\n", assembly_op_flops_1, registr, j, registr, j);
				j++;
				if(j  >= NUM_REGISTER){
					j = 0;
				}
				fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d;\"\n", assembly_op_flops_2, registr, j, registr, j);
			}else if(strcmp(op,"FMA") == 0){
				fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d, %%%%%s%d\\n\\t\\t\"\n", assembly_op_flops_1, registr, j, registr, j, registr, j);	
			}else{
				fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d;\"\n", assembly_op_flops_1, registr, j, registr, j);
			}
		//ARM SCALAR SECTION
		#elif defined(ASCALAR)
			if(strcmp(op,"fma") == 0){
				fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr, j, registr, j, registr, j, registr, j);
			}
			else{
				fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr, j, registr, j, registr, j);
			}
		//ARM NEON SECTION
		#elif defined(NEON)
			fprintf(file,"\t\t\"%s V%d%s, V%d%s, V%d%s\\n\\t\"\n", assembly_op_flops_1, j, registr, j, registr, j, registr);
		//ARM SVE SECTION
		#elif defined(SVE)
			fprintf(file,"\t\t\"%s z%d%s, p0/m, z%d%s, z%d%s\\n\\t\"\n", assembly_op_flops_1, j, registr, j, registr, j, registr);
		//RISCV SCALAR SECTION
		#elif defined(RISCVSCALAR)
			if(strcmp(op,"fma") == 0){
				fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr, j, registr, j, registr, j, registr, j);
			}
			else{
				fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr, j, registr, j, registr, j);
			}
		//RISCV RVV SECTION
		#elif defined(RVV07) || defined(RVV1)
			fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr, j, registr, j, registr, j);
			j+=(LMUL-1);
		#endif
		j++;
	}
	
	
	//x86 SECTION
	#if defined(SCALAR) || defined(SSE) || defined(AVX2) || defined(AVX512)
		fprintf(file,"\t\t\"sub $1, %%%%r8\\n\\t\\t\"\n");
		fprintf(file,"\t\t\"jnz Loop2_%%=\\n\\t\\t\"\n");
	//ARM SECTION
	#elif defined(ASCALAR) || defined(NEON) || defined(SVE)
		fprintf(file,"\t\t\"sub w0, w0, 1\\n\\t\"\n");
		fprintf(file,"\t\t\"cbnz w0, Loop2_%%=\\n\\t\"\n");
	//RISCV SECTION
	#elif defined(RISCVSCALAR) || defined(RVV07) || defined(RVV1)
		fprintf(file,"\t\t\"addi t0, t0, -1\\n\\t\"\n");
		fprintf(file,"\t\t\"bgtz t0, Loop2_%%=\\n\\t\"\n");
	#endif
	
	//End Test Function
	//x86 SECTION
	#if defined(SCALAR) || defined(SSE) || defined(AVX2) || defined(AVX512)
	if(strcmp(op,"div") == 0){
		fprintf(file,"\t\t:\n\t\t:\"r\"(num_rep_max),\"r\" (test_var)\n\t\t:\"%%rax\",\"%%rdi\","COBLERED"\n\t);\n");
	}else{
		fprintf(file,"\t\t:\n\t\t:\"r\"(num_rep_max)\n\t\t:\"%%rax\",\"%%rdi\","COBLERED"\n\t);\n");
	}
	//ARM SECTION
	#elif defined(ASCALAR) || defined(NEON) || defined(SVE)
		fprintf(file,"\t\t:\n\t\t:\"r\"(num_rep_max)\n\t\t:"COBLERED"\n\t);\n");
	//RISCV SECTION
	#elif defined(RISCVSCALAR) || defined(RVV07) || defined(RVV1)
			fprintf(file,"\t\t:\n\t\t:\"m\"(num_rep_max)\n\t\t:"COBLERED"\n\t);\n");
	#endif
	
	fprintf(file,"}\n\n");

	fclose(file_header);
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//																					WRITE MEM TEST
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void write_asm_mem (int long long num_rep, int align, int ops, int num_ld, int num_st, char * registr, char * assembly_op, char * assembly_op_2, char * precision, int Vlen, int LMUL, int num_runs){
	
	int offset = 0;
	int aux = num_rep;
	int i, j = 0, k;
	FILE * file, * file_header;
	int num_aux = 0;
	int long long iter;
	
	char path[8192];
	snprintf(path, sizeof(path), "%s/../Test/test_params.h", PROJECT_DIR);

	file_header =  fopen(path, "w");
	file = file_header;

	iter = mem_math (num_rep, num_ld, num_st, &num_aux, align); //Calculate number of iterations

	fprintf(file_header,"#define NUM_RUNS %d\n", num_runs);
	//ARM SECTION
	#if defined(ASCALAR) || defined(NEON)
		fprintf(file_header,"#define ARM 1\n");
	//ARM SVE SECTION
	#elif defined(SVE)
		fprintf(file_header,"#define SVE 1\n");
		fprintf(file_header,"#define VLEN %d\n", Vlen);
		fprintf(file_header,"#define VLMUL %d\n", LMUL);
	//RISCV SECTION
	#elif defined(RISCVSCALAR)
		fprintf(file_header,"#define RISCV 1\n");
	//RISCV RVV SECTION
	#elif defined(RVV07) || defined(RVV1)
		fprintf(file_header,"#define RISCVVECTOR 1\n");
		fprintf(file_header,"#define VLEN %d\n", Vlen);
		fprintf(file_header,"#define VLMUL %d\n", LMUL);
	#endif

	fprintf(file_header,"#define MEM 1\n");	
	fprintf(file_header,"#define NUM_LD %d\n",num_ld);
	fprintf(file_header,"#define NUM_ST %d\n",num_st);
	fprintf(file_header,"#define OPS %d\n",ops);
	fprintf(file_header,"#define NUM_REP %lld\n",num_rep);
	if(strcmp(precision, "dp") == 0){
			fprintf(file_header,"#define PRECISION double\n");
	}else{
			fprintf(file_header,"#define PRECISION float\n");
	}
	fprintf(file_header,"#define ALIGN %d\n\n", align);
	fprintf(file_header,"#define FP_INST 1\n\n");
	
	//x86 SECTION
	#if defined(SCALAR) || defined(SSE) || defined(AVX2) || defined(AVX512)
	//Create Test Function
	fprintf(file,"static inline __attribute__((always_inline)) void test_function(PRECISION * test_var, int long long num_reps_t){\n");	
	fprintf(file,"\t__asm__ __volatile__ (\n");
	fprintf(file,"\t\t\"movq %%0, %%%%r8\\n\\t\\t\"\n");
	fprintf(file,"\t\t\"Loop2_%%=:\\n\\t\\t\"\n");
	fprintf(file,"\t\t\"movq %%1, %%%%rax\\n\\t\\t\"\n");
	//ARM SECTION
	#elif defined(ASCALAR) || defined(NEON) || defined(SVE)
		fprintf(file,"static inline __attribute__((always_inline)) void test_function(PRECISION * test_var, int long long num_reps_t){\n");
		fprintf(file,"\t__asm__ __volatile__ (\n");
		//ARM SVE SECTION
		#if defined(SVE)
			if(strcmp(precision, "dp") == 0){
				int bump = Vlen * 8 * 8;
				fprintf(file,"\t\t\"mov x2, %d\\n\\t\"\n", bump);
				fprintf(file,"\t\t\"ptrue p0.d\\n\\t\"\n");
			}
			else{
				int bump = Vlen * 4 * 8;
				fprintf(file,"\t\t\"mov x2, %d\\n\\t\"\n", bump);
				fprintf(file,"\t\t\"ptrue p0.s\\n\\t\"\n");
			}
		#endif
		fprintf(file,"\t\t\"mov w0, %%w0\\n\\t\"\n");
		fprintf(file,"\t\t\"Loop2_%%=:\\n\\t\"\n");
		fprintf(file,"\t\t\"mov x3, %%1\\n\\t\\t\"\n");
	//RISCV SECTION
	#elif defined(RISCVSCALAR) || defined(RVV07) || defined(RVV1)
		fprintf(file,"static inline __attribute__((always_inline)) void test_function(PRECISION * test_var, int long long num_reps_t){\n");
		fprintf(file,"\t__asm__ __volatile__ (\n");
		//RISCV RVV SECTION
		#if defined(RVV07) || defined(RVV1)
			if(strcmp(precision, "dp") == 0){
				int Real_Vlen = ((Vlen * 64 * LMUL) / 64);
				int bump = Real_Vlen * 8;
				fprintf(file,"\t\t\"li t3, %d\\n\\t\"\n", bump);
				fprintf(file,"\t\t\"li t4, %d\\n\\t\"\n", Real_Vlen);
				fprintf(file,"\t\t\"vsetvli t0, t4, e64, m%d\\n\\t\"\n", LMUL);
			}
			else{
				int Real_Vlen = ((Vlen * 32 * LMUL) / 32);
				int bump = Real_Vlen * 4;
				fprintf(file,"\t\t\"li t3, %d\\n\\t\"\n", bump);
				fprintf(file,"\t\t\"li t4, %d\\n\\t\"\n", Real_Vlen);
				fprintf(file,"\t\t\"vsetvli t0, t4, e32, m%d\\n\\t\"\n", LMUL);
			}
		#endif
		fprintf(file,"\t\t\"ld t0, %%0\\n\\t\"\n");
		fprintf(file,"\t\t\"Loop2_%%=:\\n\\t\"\n");
		fprintf(file,"\t\t\"ld t2, %%1\\n\\t\"\n");
	#endif

	if(iter > 1){
		//x86 SECTION
		#if defined(SCALAR) || defined(SSE) || defined(AVX2) || defined(AVX512)
			fprintf(file,"\t\t\"movq $%lld, %%%%rdi\\n\\t\\t\"\n",iter);
			fprintf(file,"\t\t\"Loop1_%%=:\\n\\t\\t\"\n");
		//ARM SECTION
		#elif defined(ASCALAR) || defined(NEON) || defined(SVE)
			fprintf(file,"\t\t\"ldr w1, =%lld\\n\\t\"\n",iter);
			fprintf(file,"\t\t\"Loop1_%%=:\\n\\t\"\n");
		//RISCV SECTION
		#elif defined(RISCVSCALAR) || defined(RVV07) || defined(RVV1)
			fprintf(file,"\t\t\"li t1, %lld\\n\\t\"\n", iter);
			fprintf(file,"\t\t\"Loop1_%%=:\\n\\t\"\n");
		#endif
		
		for(i = 0; i < num_aux; i++){
				for(k = 0;k < num_ld;k++){
					if(j  >= NUM_REGISTER){
						j = 0;
					}
					//x86 SECTION
					#if defined(SCALAR) || defined(SSE) || defined(AVX2) || defined(AVX512)
						fprintf(file,"\t\t\"%s %d(%%%%rax), %%%%%s%d\\n\\t\\t\"\n", assembly_op, offset, registr,j);
					//ARM SECTION
					#elif defined(ASCALAR) || defined(NEON)
						fprintf(file,"\t\t\"%s %s%d, [x3, #%d]\\n\\t\\t\"\n", assembly_op, registr, j, offset);
					//ARM SVE SECTION
					#elif defined(SVE)
						if (offset > 7){
							offset = 0;
							fprintf(file,"\t\t\"add x3, x3, x2\\n\\t\\t\"\n");
						}
						fprintf(file,"\t\t\"%s z%d%s, p0/z, [x3, #%d, mul vl]\\n\\t\\t\"\n", assembly_op, j, registr, offset);
					//RISCV SCALAR SECTION
					#elif defined(RISCVSCALAR)
						fprintf(file,"\t\t\"%s %s%d, %d(t2)\\n\\t\\t\"\n", assembly_op, registr, j, offset);
					//RISCV RVV SECTION
					#elif defined(RVV07) || defined(RVV1)
						fprintf(file,"\t\t\"%s %s%d, (t2)\\n\\t\\t\"\n", assembly_op, registr, j);
						fprintf(file,"\t\t\"add t2, t2, t3\\n\\t\\t\"\n");
						j+=(LMUL-1);
					#endif
					j++;
					#if !defined(SVE)
						offset += align;
					//ARM SVE SECTION
					#else
						offset++;
					#endif
				}
				for(k = 0;k < num_st;k++){
					if(j  >= NUM_REGISTER){
						j = 0;
					}
					//x86 SECTION
					#if defined(SCALAR) || defined(SSE) || defined(AVX2) || defined(AVX512)
						fprintf(file,"\t\t\"%s %%%%%s%d, %d(%%%%rax)\\n\\t\\t\"\n", assembly_op_2, registr, j, offset);
					//ARM SECTION
					#elif defined(ASCALAR) || defined(NEON)
						fprintf(file,"\t\t\"%s %s%d, [x3, #%d]\\n\\t\\t\"\n", assembly_op_2, registr, j, offset);
					#elif defined(SVE)
						if (offset > 7){
							offset = 0;
							fprintf(file,"\t\t\"add x3, x3, x2\\n\\t\\t\"\n");
						}
						fprintf(file,"\t\t\"%s z%d%s, p0, [x3, #%d, mul vl]\\n\\t\\t\"\n", assembly_op_2, j, registr, offset);
					//RISCV SCALAR SECTION
					#elif defined(RISCVSCALAR)
						fprintf(file,"\t\t\"%s %s%d, %d(t2)\\n\\t\\t\"\n", assembly_op_2, registr, j, offset);
					//RISCV RVV SECTION
					#elif defined(RVV07) || defined(RVV1)
						fprintf(file,"\t\t\"%s %s%d, (t2)\\n\\t\\t\"\n", assembly_op_2, registr, j);
						fprintf(file,"\t\t\"add t2, t2, t3\\n\\t\\t\"\n");
						j+=(LMUL-1);
					#endif
					j++;
					#if !defined(SVE)
						offset += align;
					//ARM SVE SECTION
					#else
						offset++;
					#endif
				}
				aux -= iter;
		}	
		//x86 SECTION
		#if defined(SCALAR) || defined(SSE) || defined(AVX2) || defined(AVX512)
			fprintf(file,"\t\t\"addq $%d, %%%%rax\\n\\t\\t\"\n",offset);
			fprintf(file,"\t\t\"subq $1, %%%%rdi\\n\\t\\t\"\n");
			fprintf(file,"\t\t\"jnz Loop1_%%=\\n\\t\\t\"\n");
		//ARM SECTION
		#elif defined(ASCALAR) || defined(NEON) || defined(SVE)
			//ARM SVE SECTION
			#if defined(SVE)
				if(strcmp(precision, "dp") == 0){
					fprintf(file,"\t\t\"add x3, x3,  #%d\\n\\t\"\n",Vlen*offset*8);
				}else{
					fprintf(file,"\t\t\"add x3, x3,  #%d\\n\\t\"\n",Vlen*offset*4);
				}
			#else
				fprintf(file,"\t\t\"add x3, x3, #%d\\n\\t\"\n",offset);
			#endif
			fprintf(file,"\t\t\"sub w1, w1, 1\\n\\t\"\n");
			fprintf(file,"\t\t\"cbnz w1, Loop1_%%=\\n\\t\"\n");
		//RISCV SECTION
		#elif defined(RISCVSCALAR) || defined(RVV07) || defined(RVV1)
			//RISCV SCALAR SECTION
			#if defined(RISCVSCALAR)
				fprintf(file,"\t\t\"addi t2, t2, %d\\n\\t\"\n",offset);
			#endif
			fprintf(file,"\t\t\"addi t1, t1, -1\\n\\t\"\n");
			fprintf(file,"\t\t\"bgtz t1, Loop1_%%=\\n\\t\"\n");
		#endif
	}
	
	num_rep = aux;
	offset = 0;

	for(i = 0; i < num_rep; i++){
		for(k = 0;k < num_ld;k++){
			if(j  >= NUM_REGISTER){
				j = 0;
			}
			//x86 SECTION
			#if defined(SCALAR) || defined(SSE) || defined(AVX2) || defined(AVX512)
				fprintf(file,"\t\t\"%s %d(%%%%rax), %%%%%s%d\\n\\t\\t\"\n", assembly_op, offset, registr,j);
			//ARM SECTION
			#elif defined(ASCALAR) || defined(NEON)
				fprintf(file,"\t\t\"%s %s%d, [x3, #%d]\\n\\t\\t\"\n", assembly_op, registr, j, offset);
			//ARM SVE SECTION
			#elif defined(SVE)
				if (offset > 7){
					offset = 0;
					fprintf(file,"\t\t\"add x3, x3, x2\\n\\t\\t\"\n");
				}
				fprintf(file,"\t\t\"%s z%d%s, p0/z, [x3, #%d, mul vl]\\n\\t\\t\"\n", assembly_op, j, registr, offset);
			//RISCV SCALAR SECTION
			#elif defined(RISCVSCALAR)
				fprintf(file,"\t\t\"%s %s%d, %d(t2)\\n\\t\\t\"\n", assembly_op, registr, j, offset);
			//RISCV RVV SECTION
			#elif defined(RVV07) || defined(RVV1)
				fprintf(file,"\t\t\"%s %s%d, (t2)\\n\\t\\t\"\n", assembly_op, registr, j);
				fprintf(file,"\t\t\"add t2, t2, t3\\n\\t\\t\"\n");
				j+=(LMUL-1);
			#endif
			j++;
			#if !defined(SVE)
				offset += align;
			//ARM SVE SECTION
			#else
				offset++;
			#endif
		}
		for(k = 0;k < num_st;k++){
			if(j  >= NUM_REGISTER){
				j = 0;
			}
			//x86 SECTION
			#if defined(SCALAR) || defined(SSE) || defined(AVX2) || defined(AVX512)
				fprintf(file,"\t\t\"%s %%%%%s%d, %d(%%%%rax)\\n\\t\\t\"\n", assembly_op_2, registr, j, offset);
			//ARM SCALAR SECTION
			#elif defined(ASCALAR) || defined(NEON)
				fprintf(file,"\t\t\"%s %s%d, [x3, #%d]\\n\\t\\t\"\n", assembly_op_2, registr, j, offset);
			//ARM SVE SECTION
			#elif defined(SVE)
				if (offset > 7){
					offset = 0;
					fprintf(file,"\t\t\"add x3, x3, x2\\n\\t\\t\"\n");
				}
				fprintf(file,"\t\t\"%s z%d%s, p0, [x3, #%d, mul vl]\\n\\t\\t\"\n", assembly_op_2, j, registr, offset);
			//RISCV SCALAR SECTION
			#elif defined(RISCVSCALAR)
				fprintf(file,"\t\t\"%s %s%d, %d(t2)\\n\\t\\t\"\n", assembly_op_2, registr, j, offset);
			//RISCV RVV SECTION
			#elif defined(RVV07) || defined(RVV1)
				fprintf(file,"\t\t\"%s %s%d, (t2)\\n\\t\\t\"\n", assembly_op_2, registr, j);
				fprintf(file,"\t\t\"add t2, t2, t3\\n\\t\\t\"\n");
				j+=(LMUL-1);
			#endif
			j++;
			#if !defined(SVE)
				offset += align;
			//ARM SVE SECTION
			#else
				offset++;
			#endif
		}
	}

	//X86 SECTION
	#if defined(SCALAR) || defined(SSE) || defined(AVX2) || defined(AVX512)
		fprintf(file,"\t\t\"subq $1, %%%%r8\\n\\t\\t\"\n");
		fprintf(file,"\t\t\"jnz Loop2_%%=\\n\\t\\t\"\n");
	//ARM SECTION
	#elif defined(ASCALAR) || defined(NEON) || defined(SVE)
		fprintf(file,"\t\t\"sub w0, w0, 1\\n\\t\"\n");
		fprintf(file,"\t\t\"cbnz w0, Loop2_%%=\\n\\t\"\n");
	//RISCV SECTION
	#elif defined(RISCVSCALAR) || defined(RVV07) || defined(RVV1)
		fprintf(file,"\t\t\"addi t0, t0, -1\\n\\t\"\n");
		fprintf(file,"\t\t\"bgtz t0, Loop2_%%=\\n\\t\"\n");
	#endif
	
	
	//End Test Function
	//x86 SECTION
	#if defined(SCALAR) || defined(SSE) || defined(AVX2) || defined(AVX512)
		fprintf(file,"\t\t:\n\t\t:\"r\"(num_reps_t),\"r\" (test_var)\n\t\t:\"%%rax\",\"%%rdi\",\"%%r8\","COBLERED"\n\t);\n");
	//ARM SECTION
	#elif defined(ASCALAR) || defined(NEON) || defined(SVE)
		fprintf(file,"\t\t:\n\t\t:\"r\"(num_reps_t),\"r\" (test_var)\n\t\t:\"x3\","COBLERED"\n\t);\n");
	//RISCV SECTION
	#elif defined(RISCVSCALAR) || defined(RVV07) || defined(RVV1)
		fprintf(file,"\t\t:\n\t\t:\"m\"(num_reps_t),\"m\" (test_var)\n\t\t:\"t0\",\"t1\",\"t2\",\"t3\",\"t4\","COBLERED"\n\t);\n");
	#endif
	fprintf(file,"}\n\n");
	
	fclose(file_header);
}	


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//																					WRITE MIXED TEST
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void write_asm_mixed (int long long num_rep, int align, char * op, int ops, int num_ld, int num_st, int num_fp, char * registr, char * registr_flops, char * assembly_op, char * assembly_op_2, char * assembly_op_flops_1, char * assembly_op_flops_2, char * precision, int Vlen, int LMUL, int num_runs){
	
	int offset = 0;
	int aux = num_rep;
	int i, j = 0, k;
	FILE * file, * file_header;
	int num_aux;
	int long long iter;
	
	char path[8192];
	snprintf(path, sizeof(path), "%s/../Test/test_params.h", PROJECT_DIR);

	file_header =  fopen(path, "w");
	file = file_header;

	iter = mem_math (num_rep, num_ld, num_st, &num_aux, align); //Calculate number of iterations

	int half_point = (num_fp + 1) / 2;

	fprintf(file_header,"#define NUM_RUNS %d\n", num_runs);
	//ARM SECTION
	#if defined(ASCALAR) || defined(NEON)
		fprintf(file_header,"#define ARM 1\n");
	//ARM SVE SECTION
	#elif defined(SVE)
		fprintf(file_header,"#define SVE 1\n");
		fprintf(file_header,"#define VLEN %d\n", Vlen);
		fprintf(file_header,"#define VLMUL %d\n", LMUL);
	//RISCV SCALAR SECTION
	#elif defined(RISCVSCALAR)
		fprintf(file_header,"#define RISCV 1\n");
	//RISCV RVV SECTION
	#elif defined(RVV07) || defined(RVV1)
		fprintf(file_header,"#define RISCVVECTOR 1\n");
		fprintf(file_header,"#define VLEN %d\n", Vlen);
		fprintf(file_header,"#define VLMUL %d\n", LMUL);
	#endif

	fprintf(file_header,"#define MIXED 1\n");	
	fprintf(file_header,"#define NUM_LD %d\n",num_ld);
	fprintf(file_header,"#define NUM_ST %d\n",num_st);
	fprintf(file_header,"#define NUM_FP %d\n",num_fp);
	fprintf(file_header,"#define OPS %d\n",ops);
	fprintf(file_header,"#define NUM_REP %lld\n",num_rep);

	if(strcmp(precision, "dp") == 0){
			fprintf(file_header,"#define PRECISION double\n");
	}else{
			fprintf(file_header,"#define PRECISION float\n");
	}
	fprintf(file_header,"#define ALIGN %d\n\n", align);
	fprintf(file_header,"#define FP_INST 1\n\n");
	
	//x86 SECTION
	#if defined(SCALAR) || defined(SSE) || defined(AVX2) || defined(AVX512)
	//Create Test Function
	fprintf(file,"static inline __attribute__((always_inline)) void test_function(PRECISION * test_var, int long long num_reps_t){\n");
	
	fprintf(file,"\t__asm__ __volatile__ (\n");
	
	fprintf(file,"\t\t\"movq %%0, %%%%r8\\n\\t\\t\"\n");
	fprintf(file,"\t\t\"Loop2_%%=:\\n\\t\\t\"\n");
	fprintf(file,"\t\t\"movq %%1, %%%%rax\\n\\t\\t\"\n");
	//ARM SECTION
	#elif defined(ASCALAR) || defined(NEON) || defined(SVE)
		fprintf(file,"static inline __attribute__((always_inline)) void test_function(PRECISION * test_var, int long long num_reps_t){\n");
		fprintf(file,"\t__asm__ __volatile__ (\n");
		//ARM SVE SECTION
		#if defined(SVE)
			int bump = Vlen * 8 * 8;
			fprintf(file,"\t\t\"mov x2, %d\\n\\t\"\n", bump);
			if(strcmp(precision, "dp") == 0){
				fprintf(file,"\t\t\"ptrue p0.d\\n\\t\"\n");
			}
			else{
				fprintf(file,"\t\t\"ptrue p0.s\\n\\t\"\n");
			}
		#endif

		fprintf(file,"\t\t\"mov w0, %%w0\\n\\t\"\n");
		fprintf(file,"\t\t\"Loop2_%%=:\\n\\t\"\n");
		fprintf(file,"\t\t\"mov x3, %%1\\n\\t\\t\"\n");
	//RISCV SECTION
	#elif defined(RISCVSCALAR) || defined(RVV07) || defined(RVV1)
		fprintf(file,"static inline __attribute__((always_inline)) void test_function(PRECISION * test_var, int long long num_reps_t){\n");
		fprintf(file,"\t__asm__ __volatile__ (\n");
		//RISCV RVV SECTION
		#if defined(RVV07) || defined(RVV1)
			if(strcmp(precision, "dp") == 0){
				int Real_Vlen = ((Vlen * 64 * LMUL) / 64);
				int bump = Real_Vlen * 8;
				fprintf(file,"\t\t\"li t3, %d\\n\\t\"\n", bump);
				fprintf(file,"\t\t\"li t4, %d\\n\\t\"\n", Real_Vlen);
				fprintf(file,"\t\t\"vsetvli t0, t4, e64, m%d\\n\\t\"\n", LMUL);
			}
			else{
				int Real_Vlen = ((Vlen * 32 * LMUL) / 32);
				int bump = Real_Vlen * 4;
				fprintf(file,"\t\t\"li t3, %d\\n\\t\"\n", bump);
				fprintf(file,"\t\t\"li t4, %d\\n\\t\"\n", Real_Vlen);
				fprintf(file,"\t\t\"vsetvli t0, t4, e32, m%d\\n\\t\"\n", LMUL);
			}
		#endif
		fprintf(file,"\t\t\"ld t0, %%0\\n\\t\"\n");
		fprintf(file,"\t\t\"Loop2_%%=:\\n\\t\"\n");
		fprintf(file,"\t\t\"ld t2, %%1\\n\\t\"\n");
	#endif

	if(iter > 1){
		//x86 SECTION
		#if defined(SCALAR) || defined(SSE) || defined(AVX2) || defined(AVX512)
			fprintf(file,"\t\t\"movq $%lld, %%%%rdi\\n\\t\\t\"\n",iter);
			fprintf(file,"\t\t\"Loop1_%%=:\\n\\t\\t\"\n");
		//ARM SECTION
		#elif defined(ASCALAR) || defined(NEON) || defined(SVE)
			fprintf(file,"\t\t\"mov w1, %lld\\n\\t\"\n",iter);
			fprintf(file,"\t\t\"Loop1_%%=:\\n\\t\"\n");
		//RISCV SECTION
		#elif defined(RISCVSCALAR) || defined(RVV07) || defined(RVV1)
			fprintf(file,"\t\t\"li t1, %lld\\n\\t\"\n", iter);
			fprintf(file,"\t\t\"Loop1_%%=:\\n\\t\"\n");
		#endif
		
		for(i = 0; i < num_aux; i+=1){
			//for (k = 0; k < half_point; k++){
			for (k = 0; k < num_fp+num_ld+num_st; k++){
				if(j  >= NUM_REGISTER){
					j = 0;
				}
				if (k < half_point){
				//x86 AVX or SCALAR SECTION
				#if defined(SCALAR) || defined(AVX2) || defined(AVX512)
					if(strcmp(op,"div") == 0){
						fprintf(file,"\t\t\"%s %%%%%s0, %%%%%s%d, %%%%%s%d\\n\\t\\t\"\n", assembly_op_flops_1, registr_flops, registr_flops, j, registr_flops, j);
					}else if(strcmp(op,"mad") == 0){
						if(j  >= NUM_REGISTER){
							j = 0;
						}
						fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d, %%%%%s%d\\n\\t\\t\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j, registr_flops, j);
						j++;
						if(j  >= NUM_REGISTER){
							j = 0;
						}
						fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d, %%%%%s%d\\n\\t\\t\"\n", assembly_op_flops_2, registr_flops, j, registr_flops, j, registr_flops, j);
					}else{	
						fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d, %%%%%s%d\\n\\t\\t\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j, registr_flops, j);
					}
				//x86 SSE SECTION
				#elif defined(SSE)
					if(strcmp(op,"div") == 0){
						fprintf(file,"\t\t\"%s %%%%%s0, %%%%%s%d;\"\n", assembly_op_flops_1, registr_flops, registr_flops, j);
					}else if(strcmp(op,"mad") == 0){
						if(j  >= NUM_REGISTER){
							j = 0;
						}
						fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d;\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j);
						j++;
						if(j  >= NUM_REGISTER){
							j = 0;
						}
						fprintf(file,"\t\t\"%s %%%%%st%d, %%%%%s%d;\"\n", assembly_op_flops_2, registr_flops, j, registr_flops, j);
					}else if(strcmp(op,"fma") == 0){
						fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d, %%%%%s%d\\n\\t\\t\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j, registr_flops, j);
					}else{
						fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d;\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j);
					}
				//ARM SCALAR SECTION
				#elif defined(ASCALAR)
					if(strcmp(op,"fma") == 0){
						fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j, registr_flops, j, registr_flops, j);
					}
					else{
						fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j, registr_flops, j);
					}
				//ARM NEON SECTION
				#elif defined(NEON)
					fprintf(file,"\t\t\"%s V%d%s, V%d%s, V%d%s\\n\\t\"\n", assembly_op_flops_1, j, registr_flops, j, registr_flops, j, registr_flops);
				//ARM SVE SECTION
				#elif defined(SVE)
					fprintf(file,"\t\t\"%s z%d%s, p0/m, z%d%s, z%d%s\\n\\t\"\n", assembly_op_flops_1, j, registr, j, registr, j, registr);
				//RISCV SCALAR SECTION
				#elif defined(RISCVSCALAR)
					if(strcmp(op,"fma") == 0){
						fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j, registr_flops, j, registr_flops, j);
					}
					else{
						fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j, registr_flops, j);
					}
				//RISCV RVV SECTION
				#elif defined(RVV07) || defined(RVV1)
					fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr, j, registr, j, registr, j);
					j+=(LMUL-1);
				#endif
				j++;
			}

			//for(k = 0;k < num_ld;k++){
				if(j  >= NUM_REGISTER){
					j = 0;
				}
				if (k < num_ld){
				//x86 SECTION
				#if defined(SCALAR) || defined(SSE) || defined(AVX2) || defined(AVX512)
					fprintf(file,"\t\t\"%s %d(%%%%rax), %%%%%s%d\\n\\t\\t\"\n", assembly_op, offset, registr,j);
				//ARM SECTION
				#elif defined(ASCALAR) || defined(NEON)
					fprintf(file,"\t\t\"%s %s%d, [x3, #%d]\\n\\t\\t\"\n", assembly_op, registr, j, offset);
				//ARM SVE SECTION
				#elif defined(SVE)
					if (offset > 7){
						offset = 0;
						fprintf(file,"\t\t\"add x3, x3, x2\\n\\t\\t\"\n");
					}
					fprintf(file,"\t\t\"%s z%d%s, p0/z, [x3, #%d, mul vl]\\n\\t\\t\"\n", assembly_op, j, registr, offset);
				//RISCV SCALAR SECTION
				#elif defined(RISCVSCALAR)
					fprintf(file,"\t\t\"%s %s%d, %d(t2)\\n\\t\\t\"\n", assembly_op, registr, j, offset);
				//RISCV RVV SECTION
				#elif defined(RVV07) || defined(RVV1)
						fprintf(file,"\t\t\"%s %s%d, (t2)\\n\\t\\t\"\n", assembly_op, registr, j);
						fprintf(file,"\t\t\"add t2, t2, t3\\n\\t\\t\"\n");
						j+=(LMUL-1);
				#endif
				j++;
				#if !defined(SVE)
					offset += align;
				//ARM SVE SECTION
				#else
					offset++;
				#endif
			}
			
			//for (k = half_point; k < num_fp; k++){
				if(j  >= NUM_REGISTER){
					j = 0;
				}
				if (k < (num_fp - half_point)){
				//x86 AVX or SCALAR SECTION
				#if defined(SCALAR) || defined(AVX2) || defined(AVX512)
					if(strcmp(op,"div") == 0){
						fprintf(file,"\t\t\"%s %%%%%s0, %%%%%s%d, %%%%%s%d\\n\\t\\t\"\n", assembly_op_flops_1, registr_flops, registr_flops, j, registr_flops, j);
					}else if(strcmp(op,"mad") == 0){
						if(j  >= NUM_REGISTER){
							j = 0;
						}
						fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d, %%%%%s%d\\n\\t\\t\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j, registr_flops, j);
						j++;
						if(j  >= NUM_REGISTER){
							j = 0;
						}
						fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d, %%%%%s%d\\n\\t\\t\"\n", assembly_op_flops_2, registr_flops, j, registr_flops, j, registr_flops, j);
					}else{	
						fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d, %%%%%s%d\\n\\t\\t\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j, registr_flops, j);
					}
				//x86 SSE SECTION
				#elif defined(SSE)
					if(strcmp(op,"div") == 0){
						fprintf(file,"\t\t\"%s %%%%%s0, %%%%%s%d;\"\n", assembly_op_flops_1, registr_flops, registr_flops, j);
					}else if(strcmp(op,"mad") == 0){
						if(j  >= NUM_REGISTER){
							j = 0;
						}
						fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d;\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j);
						j++;
						if(j  >= NUM_REGISTER){
							j = 0;
						}
						fprintf(file,"\t\t\"%s %%%%%st%d, %%%%%s%d;\"\n", assembly_op_flops_2, registr_flops, j, registr_flops, j);
					}else if(strcmp(op,"fma") == 0){
						fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d, %%%%%s%d\\n\\t\\t\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j, registr_flops, j);
					}else{
						fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d;\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j);
					}
				//ARM SCALAR SECTION
				#elif defined(ASCALAR)
					if(strcmp(op,"fma") == 0){
						fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j, registr_flops, j, registr_flops, j);
					}
					else{
						fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j, registr_flops, j);
					}
				//ARM NEON SECTION
				#elif defined(NEON)
					fprintf(file,"\t\t\"%s V%d%s, V%d%s, V%d%s\\n\\t\"\n", assembly_op_flops_1, j, registr_flops, j, registr_flops, j, registr_flops);
				//ARM SVE SECTION
				#elif defined(SVE)
					fprintf(file,"\t\t\"%s z%d%s, p0/m, z%d%s, z%d%s\\n\\t\"\n", assembly_op_flops_1, j, registr, j, registr, j, registr);
				//RISCV SCALAR SECTION
				#elif defined(RISCVSCALAR)
					if(strcmp(op,"fma") == 0){
						fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j, registr_flops, j, registr_flops, j);
					}
					else{
						fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j, registr_flops, j);
					}
				//RISCV RVV SECTION
				#elif defined(RVV07) || defined(RVV1)
					fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr, j, registr, j, registr, j);
					j+=(LMUL-1);
				#endif
				j++;
			}
			//for(k = 0;k < num_st;k++){
				if(j  >= NUM_REGISTER){
					j = 0;
				}
				if (k < num_st){
				//x86 SECTION
				#if defined(SCALAR) || defined(SSE) || defined(AVX2) || defined(AVX512)
					fprintf(file,"\t\t\"%s %%%%%s%d, %d(%%%%rax)\\n\\t\\t\"\n", assembly_op_2, registr, j, offset);
				//ARM SECTION
				#elif defined(ASCALAR) || defined(NEON)
					fprintf(file,"\t\t\"%s %s%d, [x3, #%d]\\n\\t\\t\"\n", assembly_op_2, registr, j, offset);
				//ARM SVE SECTION
				#elif defined(SVE)
					if (offset > 7){
						offset = 0;
						fprintf(file,"\t\t\"add x3, x3, x2\\n\\t\\t\"\n");
					}
	
					fprintf(file,"\t\t\"%s z%d%s, p0, [x3, #%d, mul vl]\\n\\t\\t\"\n", assembly_op_2, j, registr, offset);
				//RISCV SCALAR SECTION
				#elif defined(RISCVSCALAR)
					fprintf(file,"\t\t\"%s %s%d, %d(t2)\\n\\t\\t\"\n", assembly_op_2, registr, j, offset);
				//RISCV RVV SECTION
				#elif defined(RVV07) || defined(RVV1)
						fprintf(file,"\t\t\"%s %s%d, (t2)\\n\\t\\t\"\n", assembly_op_2, registr, j);
						fprintf(file,"\t\t\"add t2, t2, t3\\n\\t\\t\"\n");
						j+=(LMUL-1);
				#endif
				j++;
				#if !defined(SVE)
					offset += align;
				//ARM SVE SECTION
				#else
					offset++;
				#endif
			}
			}
			aux -= iter;
		}	
		//x86 SECTION
		#if defined(SCALAR) || defined(SSE) || defined(AVX2) || defined(AVX512)
			fprintf(file,"\t\t\"addq $%d, %%%%rax\\n\\t\\t\"\n",offset);
			fprintf(file,"\t\t\"subq $1, %%%%rdi\\n\\t\\t\"\n");
			fprintf(file,"\t\t\"jnz Loop1_%%=\\n\\t\\t\"\n");
		//ARM SECTION
		#elif defined(ASCALAR) || defined(NEON) || defined(SVE)
			//ARM SVE SECTION
			#if defined(SVE)
				fprintf(file,"\t\t\"add x3, x3,  #%d\\n\\t\"\n",Vlen*offset*8);
			#else
				fprintf(file,"\t\t\"add x3, x3, #%d\\n\\t\"\n",offset);
			#endif
			fprintf(file,"\t\t\"sub w1, w1, 1\\n\\t\"\n");
			fprintf(file,"\t\t\"cbnz w1, Loop1_%%=\\n\\t\"\n");
		//RISCV SECTION
		#elif defined(RISCVSCALAR) || defined(RVV07) || defined(RVV1)
			//RISCV SCALAR SECTION
			#if defined(RISCVSCALAR)
			fprintf(file,"\t\t\"addi t2, t2, %d\\n\\t\"\n",offset);
			#endif
			fprintf(file,"\t\t\"addi t1, t1, -1\\n\\t\"\n");
			fprintf(file,"\t\t\"bgtz t1, Loop1_%%=\\n\\t\"\n");
		#endif
	}
	
	num_rep = aux;
	offset = 0;
	
	for(i = 0; i < num_rep; i+=1){
		//for (k = 0; k < half_point; k++){
		for (k = 0; k < num_fp+num_ld+num_st; k++){
			if (k < half_point){
				if(j  >= NUM_REGISTER){
					j = 0;
				}
				//x86 AVX or SCALAR SECTION
				#if defined(SCALAR) || defined(AVX2) || defined(AVX512)
					if(strcmp(op,"div") == 0){
						fprintf(file,"\t\t\"%s %%%%%s0, %%%%%s%d, %%%%%s%d\\n\\t\\t\"\n", assembly_op_flops_1, registr_flops, registr_flops, j, registr_flops, j);
					}else if(strcmp(op,"mad") == 0){
						if(j  >= NUM_REGISTER){
							j = 0;
						}
						fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d, %%%%%s%d\\n\\t\\t\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j, registr_flops, j);
						j++;
						if(j  >= NUM_REGISTER){
							j = 0;
						}
						fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d, %%%%%s%d\\n\\t\\t\"\n", assembly_op_flops_2, registr_flops, j, registr_flops, j, registr_flops, j);
					}else{	
						fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d, %%%%%s%d\\n\\t\\t\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j, registr_flops, j);
					}
				//x86 SSE SECTION
				#elif defined(SSE)
					if(strcmp(op,"div") == 0){
						fprintf(file,"\t\t\"%s %%%%%s0, %%%%%s%d;\"\n", assembly_op_flops_1, registr_flops, registr_flops, j);
					}else if(strcmp(op,"mad") == 0){
						if(j  >= NUM_REGISTER){
							j = 0;
						}
						fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d;\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j);
						j++;
						if(j  >= NUM_REGISTER){
							j = 0;
						}
						fprintf(file,"\t\t\"%s %%%%%st%d, %%%%%s%d;\"\n", assembly_op_flops_2, registr_flops, j, registr_flops, j);
					}else if(strcmp(op,"fma") == 0){
						fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d, %%%%%s%d\\n\\t\\t\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j, registr_flops, j);
					}else{
						fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d;\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j);
					}
				//ARM SCALAR SECTION
				#elif defined(ASCALAR)
					if(strcmp(op,"fma") == 0){
						fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j, registr_flops, j, registr_flops, j);
					}
					else{
						fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j, registr_flops, j);
					}
				//ARM NEON SECTION
				#elif defined(NEON)
					fprintf(file,"\t\t\"%s V%d%s, V%d%s, V%d%s\\n\\t\"\n", assembly_op_flops_1, j, registr_flops, j, registr_flops, j, registr_flops);
				//ARM SVE SECTION
				#elif defined(SVE)
					fprintf(file,"\t\t\"%s z%d%s, p0/m, z%d%s, z%d%s\\n\\t\"\n", assembly_op_flops_1, j, registr, j, registr, j, registr);
				//RISCV SCALAR SECTION
				#elif defined(RISCVSCALAR)
					if(strcmp(op,"fma") == 0){
						fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j, registr_flops, j, registr_flops, j);
					}
					else{
						fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j, registr_flops, j);
					}
				//RISCV RVV SECTION
				#elif defined(RVV07) || defined(RVV1)
					fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr, j, registr, j, registr, j);
					j+=(LMUL-1);
				#endif
				j++;
			}
		//for(k = 0;k < num_ld;k++){
			if(j  >= NUM_REGISTER){
				j = 0;
			}
			if (k < num_ld){
			//x86 SECTION
			#if defined(SCALAR) || defined(SSE) || defined(AVX2) || defined(AVX512)
				fprintf(file,"\t\t\"%s %d(%%%%rax), %%%%%s%d\\n\\t\\t\"\n", assembly_op, offset, registr,j);
			//ARM SECTION
			#elif defined(ASCALAR) || defined(NEON)
				fprintf(file,"\t\t\"%s %s%d, [x3, #%d]\\n\\t\\t\"\n", assembly_op, registr, j, offset);
			//ARM SVE SECTION
			#elif defined(SVE)
				if (offset > 7){
					offset = 0;
					fprintf(file,"\t\t\"add x3, x3, x2\\n\\t\\t\"\n");
				}
				fprintf(file,"\t\t\"%s z%d%s, p0/z, [x3, #%d, mul vl]\\n\\t\\t\"\n", assembly_op, j, registr, offset);
			//RISCV SCALAR SECTION
			#elif defined(RISCVSCALAR)
				fprintf(file,"\t\t\"%s %s%d, %d(t2)\\n\\t\\t\"\n", assembly_op, registr, j, offset);
			//RISCV RVV SECTION
			#elif defined(RVV07) || defined(RVV1)
				fprintf(file,"\t\t\"%s %s%d, (t2)\\n\\t\\t\"\n", assembly_op, registr, j);
				fprintf(file,"\t\t\"add t2, t2, t3\\n\\t\\t\"\n");
				j+=(LMUL-1);
			#endif
			j++;
			#if !defined(SVE)
				offset += align;
			#else
				offset++;
			#endif
			
		}
		//for (k = half_point; k < num_fp; k++){
			if(j  >= NUM_REGISTER){
				j = 0;
			}
			if (k < (num_fp - half_point)){
			//x86 AVX or SCALAR SECTION
			#if defined(SCALAR) || defined(AVX2) || defined(AVX512)
				if(strcmp(op,"div") == 0){
					fprintf(file,"\t\t\"%s %%%%%s0, %%%%%s%d, %%%%%s%d\\n\\t\\t\"\n", assembly_op_flops_1, registr_flops, registr_flops, j, registr_flops, j);
				}else if(strcmp(op,"mad") == 0){
					if(j  >= NUM_REGISTER){
						j = 0;
					}
					fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d, %%%%%s%d\\n\\t\\t\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j, registr_flops, j);
					j++;
					if(j  >= NUM_REGISTER){
						j = 0;
					}
					fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d, %%%%%s%d\\n\\t\\t\"\n", assembly_op_flops_2, registr_flops, j, registr_flops, j, registr_flops, j);
				}else{	
					fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d, %%%%%s%d\\n\\t\\t\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j, registr_flops, j);
				}
			//x86 SSE SECTION
			#elif defined(SSE)
				if(strcmp(op,"div") == 0){
					fprintf(file,"\t\t\"%s %%%%%s0, %%%%%s%d;\"\n", assembly_op_flops_1, registr_flops, registr_flops, j);
				}else if(strcmp(op,"mad") == 0){
					if(j  >= NUM_REGISTER){
						j = 0;
					}
					fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d;\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j);
					j++;
					if(j  >= NUM_REGISTER){
						j = 0;
					}
					fprintf(file,"\t\t\"%s %%%%%st%d, %%%%%s%d;\"\n", assembly_op_flops_2, registr_flops, j, registr_flops, j);
				}else if(strcmp(op,"fma") == 0){
					fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d, %%%%%s%d\\n\\t\\t\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j, registr_flops, j);
				}else{
					fprintf(file,"\t\t\"%s %%%%%s%d, %%%%%s%d;\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j);
				}
			//ARM SCALAR SECTION
			#elif defined(ASCALAR)
				if(strcmp(op,"fma") == 0){
					fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j, registr_flops, j, registr_flops, j);
				}
				else{
					fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j, registr_flops, j);
				}
			//ARM NEON SECTION
			#elif defined(NEON)
				fprintf(file,"\t\t\"%s V%d%s, V%d%s, V%d%s\\n\\t\"\n", assembly_op_flops_1, j, registr_flops, j, registr_flops, j, registr_flops);
			//ARM SVE SECTION
			#elif defined(SVE)
				fprintf(file,"\t\t\"%s z%d%s, p0/m, z%d%s, z%d%s\\n\\t\"\n", assembly_op_flops_1, j, registr, j, registr, j, registr);
			//RISCV SCALAR SECTION
			#elif defined(RISCVSCALAR)
				if(strcmp(op,"fma") == 0){
					fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j, registr_flops, j, registr_flops, j);
				}
				else{
					fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j, registr_flops, j);
				}
			//RISCV RVV SECTION
			#elif defined(RVV07) || defined(RVV1)
				fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr, j, registr, j, registr, j);
				j+=(LMUL-1);
			#endif
			j++;
		}
		//for(k = 0;k < num_st;k++){
			if(j  >= NUM_REGISTER){
				j = 0;
			}
			if (k < num_st){
			//x86 SECTION
			#if defined(SCALAR) || defined(SSE) || defined(AVX2) || defined(AVX512)
				fprintf(file,"\t\t\"%s %%%%%s%d, %d(%%%%rax)\\n\\t\\t\"\n", assembly_op_2, registr, j, offset);
			//ARM SECTION
			#elif defined(ASCALAR) || defined(NEON)
				fprintf(file,"\t\t\"%s %s%d, [x3, #%d]\\n\\t\\t\"\n", assembly_op_2, registr, j, offset);
			//ARM SVE SECTION
			#elif defined(SVE)
				if (offset > 7){
					offset = 0;
					fprintf(file,"\t\t\"add x3, x3, x2\\n\\t\\t\"\n");
				}
			//RISCV SCALAR SECTION
			#elif defined(RISCVSCALAR)
				fprintf(file,"\t\t\"%s %s%d, %d(t2)\\n\\t\\t\"\n", assembly_op_2, registr, j, offset);
			//RISCV RVV SECTION
			#elif defined(RVV07) || defined(RVV1)
				fprintf(file,"\t\t\"%s %s%d, (t2)\\n\\t\\t\"\n", assembly_op_2, registr, j);
				fprintf(file,"\t\t\"add t2, t2, t3\\n\\t\\t\"\n");
				j+=(LMUL-1);
			#endif
			j++;
			#if !defined(SVE)
				offset += align;
			//ARM SVE SECTION
			#else
				offset++;
			#endif
		}
	}
	}
	
	//x86 SECTION
	#if defined(SCALAR) || defined(SSE) || defined(AVX2) || defined(AVX512)
		fprintf(file,"\t\t\"subq $1, %%%%r8\\n\\t\\t\"\n");
		fprintf(file,"\t\t\"jnz Loop2_%%=\\n\\t\\t\"\n");
	//ARM SECTION
	#elif defined(ASCALAR) || defined(NEON) || defined(SVE)
		fprintf(file,"\t\t\"sub w0, w0, 1\\n\\t\"\n");
		fprintf(file,"\t\t\"cbnz w0, Loop2_%%=\\n\\t\"\n");
	//RISCV SECTION
	#elif defined(RISCVSCALAR) || defined(RVV07) || defined(RVV1)
		fprintf(file,"\t\t\"addi t0, t0, -1\\n\\t\"\n");
		fprintf(file,"\t\t\"bgtz t0, Loop2_%%=\\n\\t\"\n");
	#endif
	
	
	//End Test Function
	//x86 SECTION
	#if defined(SCALAR) || defined(SSE) || defined(AVX2) || defined(AVX512)
		fprintf(file,"\t\t:\n\t\t:\"r\"(num_reps_t),\"r\" (test_var)\n\t\t:\"%%rax\",\"%%rdi\",\"%%r8\","COBLERED"\n\t);\n");
	//ARM SECTION
	#elif defined(ASCALAR) || defined(NEON) || defined(SVE)
		fprintf(file,"\t\t:\n\t\t:\"r\"(num_reps_t),\"r\" (test_var)\n\t\t:\"x3\","COBLERED"\n\t);\n");
	//RISCV SECTION
	#elif defined(RISCVSCALAR) || defined(RVV07) || defined(RVV1)
		fprintf(file,"\t\t:\n\t\t:\"m\"(num_reps_t),\"m\" (test_var)\n\t\t:"COBLERED"\n\t);\n");
	#endif
	fprintf(file,"}\n\n");
	
	fclose(file_header);
}	
