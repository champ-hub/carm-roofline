#include "config_test.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//																					WRITE FP TEST
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void write_asm_fp (int long long fp, char * op, int flops, char * registr, char * assembly_op_flops_1, char * assembly_op_flops_2, char * precision, int Vlen, int LMUL){
	
	int i, j;
	FILE * file,*file_header;
	int long long iter;
	j = 0;
		
	file_header =  fopen("Test/test_params.h", "w");
	file = file_header;
	
	//Specific test data
	#if defined(ASCALAR) || defined(NEON)
		fprintf(file_header,"#define ARM 1\n");
	#elif defined(RISCVSCALAR)
		fprintf(file_header,"#define RISCV 1\n");
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
	
	iter = flops_math(fp, LMUL); //Calculate necessary iterations

	fprintf(stderr, "FP: %lld | iter: %lld | extra: %lld\n", fp, iter, fp%iter);
	
	//Creating Test Function
	if(strcmp(op,"div") == 0){
		fprintf(file,"static inline __attribute__((always_inline)) void test_function(PRECISION * test_var, int long long num_rep_max){\n");
	}else{
		fprintf(file,"static inline __attribute__((always_inline)) void test_function(int long long num_rep_max){\n");
	}
	
	fprintf(file,"\t__asm__ __volatile__ (\n");
	//RISCV VECTOR SECTION
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
	
	
	//x86 SECTION
	#if !defined(ASCALAR) && !defined(NEON) && !defined(RISCVSCALAR) && !defined(RVV07) && !defined(RVV1)
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
	#elif defined(ASCALAR) || defined(NEON)
		fprintf(file,"\t\t\"mov w0, %%w0\\n\\t\"\n");
		fprintf(file,"\t\t\"Loop2_%%=:\\n\\t\"\n");
	//RISCV SECTION
	#elif defined(RISCVSCALAR) || defined(RVV07) || defined(RVV1)
		fprintf(file,"\t\t\"ld t0, %%0\\n\\t\"\n");
		fprintf(file,"\t\t\"Loop2_%%=:\\n\\t\"\n");
	#endif

	if(iter > 1){
		//x86 SECTION
		#if !defined(ASCALAR) && !defined(NEON) && !defined(RISCVSCALAR) && !defined(RVV07) && !defined(RVV1)
			fprintf(file,"\t\t\"movl $%lld, %%%%edi\\n\\t\\t\"\n",iter);
			fprintf(file,"\t\t\"Loop1_%%=:\\n\\t\\t\"\n");
		//ARM SECTION
		#elif defined(ASCALAR) || defined(NEON)
			fprintf(file,"\t\t\"mov w1, %lld\\n\\t\"\n",iter);
			fprintf(file,"\t\t\"Loop1_%%=:\\n\\t\"\n");
		//RISCV SECTION
		#elif defined(RISCVSCALAR) || defined(RVV07) || defined(RVV1)
			fprintf(file,"\t\t\"li t1, %lld\\n\\t\"\n", iter);
			fprintf(file,"\t\t\"Loop1_%%=:\\n\\t\"\n");
		#endif

		for(i = 0; i < BASE_LOOP_SIZE; i+=LMUL){
			if(i % NUM_REGISTER == 0){
				j = 0;
			}
			//x86 AVX or SCALAR SECTION
			#if defined(AVX) || defined(AVX512) || defined(AVX2) || (!defined(SSE) && !defined(NEON) && !defined(ASCALAR) && !defined(RISCVSCALAR) && !defined(RVV07) && !defined(RVV1))
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
			#elif !defined(ASCALAR) && !defined(NEON) && !defined(RISCVSCALAR) && !defined(RVV07) && !defined(RVV1)
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
			//RISCV SECTION
			#elif defined(RISCVSCALAR)
				if(strcmp(op,"fma") == 0){
					fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr, j, registr, j, registr, j, registr, j);
				}
				else{
					fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr, j, registr, j, registr, j);
				}
			#elif defined(RVV07) || defined(RVV1)
				fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr, j, registr, j, registr, j);
				j+=(LMUL-1);
			#endif
			j++;
			fp -= iter*LMUL;
		}
		//x86 SECTION
		#if !defined(ASCALAR) && !defined(NEON) && !defined(RISCVSCALAR) && !defined(RVV07) && !defined(RVV1)
			fprintf(file,"\t\t\"subl $1, %%%%edi\\n\\t\\t\"\n");
			fprintf(file,"\t\t\"jnz Loop1_%%=\\n\\t\\t\"\n");
		//ARM SECTION
		#elif defined(ASCALAR) || defined(NEON)
			fprintf(file,"\t\t\"sub w1, w1, 1\\n\\t\"\n");
			fprintf(file,"\t\t\"cbnz w1, Loop1_%%=\\n\\t\"\n");
		//RISCV SECTION
		#elif defined(RISCVSCALAR) || defined(RVV07) || defined(RVV1)
			fprintf(file,"\t\t\"addi t1, t1, -1\\n\\t\"\n");
			fprintf(file,"\t\t\"bgtz t1, Loop1_%%=\\n\\t\"\n");
		#endif
	}
	

	
	for(i = 0; i < fp; i+=LMUL){
		if(i % 16 == 0){
			j = 0;
		}
		//x86 AVX or SCALAR SECTION
		#if defined (AVX512) || defined (AVX) || defined (AVX2) || (!defined(NEON) && !defined(ASCALAR) && !defined(RISCVSCALAR) && !defined(RVV07) && !defined(RVV1))
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
		#elif !defined(ASCALAR) && !defined(NEON) && !defined(RISCVSCALAR) && !defined(RVV07) && !defined(RVV1)
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
		#elif defined(RISCVSCALAR)
			if(strcmp(op,"fma") == 0){
				fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr, j, registr, j, registr, j, registr, j);
			}
			else{
				fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr, j, registr, j, registr, j);
			}
		#elif defined(RVV07) || defined(RVV1)
			fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr, j, registr, j, registr, j);
			j+=(LMUL-1);
		#endif
		j++;
	}
	
	
	//x86 SECTION
	#if !defined(ASCALAR) && !defined(NEON) && !defined(RISCVSCALAR) && !defined(RVV07) && !defined(RVV1)
		fprintf(file,"\t\t\"sub $1, %%%%r8\\n\\t\\t\"\n");
		fprintf(file,"\t\t\"jnz Loop2_%%=\\n\\t\\t\"\n");
	//ARM SECTION
	#elif defined(ASCALAR) || defined(NEON)
		fprintf(file,"\t\t\"sub w0, w0, 1\\n\\t\"\n");
		fprintf(file,"\t\t\"cbnz w0, Loop2_%%=\\n\\t\"\n");
	//RISCV SECTION
	#elif defined(RISCVSCALAR) || defined(RVV07) || defined(RVV1)
		fprintf(file,"\t\t\"addi t0, t0, -1\\n\\t\"\n");
		fprintf(file,"\t\t\"bgtz t0, Loop2_%%=\\n\\t\"\n");
	#endif
	
	//End Test Function
	//x86 SECTION
	#if !defined(ASCALAR) && !defined(NEON) && !defined(RISCVSCALAR) && !defined(RVV07) && !defined(RVV1)
	if(strcmp(op,"div") == 0){
		fprintf(file,"\t\t:\n\t\t:\"r\"(num_rep_max),\"r\" (test_var)\n\t\t:\"%%rax\",\"%%rdi\","COBLERED"\n\t);\n");
	}else{
		fprintf(file,"\t\t:\n\t\t:\"r\"(num_rep_max)\n\t\t:\"%%rax\",\"%%rdi\","COBLERED"\n\t);\n");
	}
	//ARM SECTION
	#elif defined(ASCALAR) || defined(NEON)
		fprintf(file,"\t\t:\n\t\t:\"r\"(num_rep_max)\n\t\t:"COBLERED"\n\t);\n");
	#elif defined(RISCVSCALAR) || defined(RVV07) || defined(RVV1)
			fprintf(file,"\t\t:\n\t\t:\"m\"(num_rep_max)\n\t\t:"COBLERED"\n\t);\n");
	#endif
	
	fprintf(file,"}\n\n");
	
	//fclose(file);
	fclose(file_header);
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//																					WRITE MEM TEST
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void write_asm_mem (int long long num_rep, int align, int ops, int num_ld, int num_st, char * registr, char * assembly_op, char * assembly_op_2, char * precision, int Vlen, int LMUL){
	
	int offset = 0;
	int aux = num_rep;
	int i, j = 0, k;
	FILE * file, * file_header;
	int num_aux;
	int long long iter;
	
	file_header =  fopen("Test/test_params.h", "w");
	file = file_header;

	fprintf(stderr,"\nVLEN: %d | LMUL: %d\n", Vlen, LMUL);

	iter = mem_math (num_rep, num_ld, num_st, &num_aux, align, Vlen, LMUL); //Calculate number of iterations
	int extra_iter = (num_rep-iter*num_aux);

	if (extra_iter < LMUL){
		extra_iter = LMUL - extra_iter;
	}else{
		extra_iter = (num_rep-iter*num_aux)%(LMUL);
	}

	fprintf(stderr, "\n MEM Iterations: %lld | NUM AUX: %d, | NUM REP: %lld | REAL NUM REP: %lld | Expected missing: %lld | Expected extra: %d\n", iter, num_aux, num_rep, iter*num_aux, num_rep-iter*num_aux, extra_iter);
	
	//ARM SECTION
	#if defined(ASCALAR) || defined(NEON)
		fprintf(file_header,"#define ARM 1\n");
		if (iter > 4096){
			fprintf(file_header,"#define VAR 1\n");
		}
	//RISCV SECTION
	#elif defined(RISCVSCALAR)
		fprintf(file_header,"#define RISCV 1\n");
	#elif defined(RVV07) || defined(RVV1)
		fprintf(file_header,"#define RISCVVECTOR 1\n");
		fprintf(file_header,"#define VLEN %d\n", Vlen);
		fprintf(file_header,"#define VLMUL %d\n", LMUL);
	#endif
	fprintf(file_header,"#define MEM 1\n");	
	fprintf(file_header,"#define NUM_LD %d\n",num_ld);
	fprintf(file_header,"#define NUM_ST %d\n",num_st);
	fprintf(file_header,"#define OPS %d\n",ops);
	fprintf(file_header,"#define NUM_REP %lld\n",num_rep+extra_iter);
	if(strcmp(precision, "dp") == 0){
			fprintf(file_header,"#define PRECISION double\n");
	}else{
			fprintf(file_header,"#define PRECISION float\n");
	}
	fprintf(file_header,"#define ALIGN %d\n\n", align);
	fprintf(file_header,"#define FP_INST 1\n\n");
	
	//x86 SECTION
	#if !defined(ASCALAR) && !defined(NEON) && !defined(RISCVSCALAR) && !defined(RVV07) && !defined(RVV1)
	//Create Test Function
	fprintf(file,"static inline __attribute__((always_inline)) void test_function(PRECISION * test_var, int long long num_reps_t){\n");	
	fprintf(file,"\t__asm__ __volatile__ (\n");
	fprintf(file,"\t\t\"movq %%0, %%%%r8\\n\\t\\t\"\n");
	fprintf(file,"\t\t\"Loop2_%%=:\\n\\t\\t\"\n");
	fprintf(file,"\t\t\"movq %%1, %%%%rax\\n\\t\\t\"\n");
	//ARM SECTION
	#elif defined(ASCALAR) || defined(NEON)
		if (iter > 4096){ //Iterations trick for ARM, to avoid using immediate bigger than 4096
			fprintf(file,"static inline __attribute__((always_inline)) void test_function(PRECISION * test_var, int long long num_reps_t, int long long num_reps_2){\n");
		}else{
			fprintf(file,"static inline __attribute__((always_inline)) void test_function(PRECISION * test_var, int long long num_reps_t){\n");
		}
		fprintf(file,"\t__asm__ __volatile__ (\n");

		fprintf(file,"\t\t\"mov w0, %%w0\\n\\t\"\n");
		fprintf(file,"\t\t\"Loop2_%%=:\\n\\t\"\n");
		fprintf(file,"\t\t\"mov x3, %%1\\n\\t\\t\"\n");
	//RISCV SECTION
	#elif defined(RISCVSCALAR) || defined(RVV07) || defined(RVV1)
		fprintf(file,"static inline __attribute__((always_inline)) void test_function(PRECISION * test_var, int long long num_reps_t){\n");
		fprintf(file,"\t__asm__ __volatile__ (\n");
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
		#if !defined(ASCALAR) && !defined(NEON) && !defined(RISCVSCALAR) && !defined(RVV07) && !defined(RVV1)
			fprintf(file,"\t\t\"movq $%lld, %%%%rdi\\n\\t\\t\"\n",iter);
			fprintf(file,"\t\t\"Loop1_%%=:\\n\\t\\t\"\n");
		//ARM SECTION
		#elif defined(ASCALAR) || defined(NEON)
			if (iter > 4096){ //Iterations trick for ARM, to avoid using immediate bigger than 4096
				fprintf(file,"\t\t\"mov w1, %%w2\\n\\t\"\n");
			}else{
				fprintf(file,"\t\t\"mov w1, %lld\\n\\t\"\n",iter);
			}
			
			fprintf(file,"\t\t\"Loop1_%%=:\\n\\t\"\n");
		//RISCV SECTION
		#elif defined(RISCVSCALAR) || defined(RVV07) || defined(RVV1)
			fprintf(file,"\t\t\"li t1, %lld\\n\\t\"\n", iter);
			fprintf(file,"\t\t\"Loop1_%%=:\\n\\t\"\n");
		#endif
		
		for(i = 0; i < num_aux; i+=LMUL){
				for(k = 0;k < num_ld;k++){
					if(j  >= NUM_REGISTER){
						j = 0;
					}
					//x86 SECTION
					#if !defined(ASCALAR) && !defined(NEON) && !defined(RISCVSCALAR) && !defined(RVV07) && !defined(RVV1)
						fprintf(file,"\t\t\"%s %d(%%%%rax), %%%%%s%d\\n\\t\\t\"\n", assembly_op, offset, registr,j);
					//ARM SECTION
					#elif defined(ASCALAR) || defined(NEON)
						fprintf(file,"\t\t\"%s %s%d, [x3, #%d]\\n\\t\\t\"\n", assembly_op, registr, j, offset);
					//RISCV SECTION
					#elif defined(RISCVSCALAR)
						fprintf(file,"\t\t\"%s %s%d, %d(t2)\\n\\t\\t\"\n", assembly_op, registr, j, offset);
					#elif defined(RVV07) || defined(RVV1)
						fprintf(file,"\t\t\"%s %s%d, (t2)\\n\\t\\t\"\n", assembly_op, registr, j);
						fprintf(file,"\t\t\"add t2, t2, t3\\n\\t\\t\"\n");
						j+=(LMUL-1);
					#endif
					j++;
					offset += align;
				}
				for(k = 0;k < num_st;k++){
					if(j  >= NUM_REGISTER){
						j = 0;
					}
					//x86 SECTION
					#if !defined(ASCALAR) && !defined(NEON) && !defined(RISCVSCALAR) && !defined(RVV07) && !defined(RVV1)
						fprintf(file,"\t\t\"%s %%%%%s%d, %d(%%%%rax)\\n\\t\\t\"\n", assembly_op_2, registr, j, offset);
					//ARM SECTION
					#elif defined(ASCALAR) || defined(NEON)
						fprintf(file,"\t\t\"%s %s%d, [x3, #%d]\\n\\t\\t\"\n", assembly_op_2, registr, j, offset);
					//RISCV SECTION
					#elif defined(RISCVSCALAR)
						fprintf(file,"\t\t\"%s %s%d, %d(t2)\\n\\t\\t\"\n", assembly_op_2, registr, j, offset);
					#elif defined(RVV07) || defined(RVV1)
						fprintf(file,"\t\t\"%s %s%d, (t2)\\n\\t\\t\"\n", assembly_op_2, registr, j);
						fprintf(file,"\t\t\"add t2, t2, t3\\n\\t\\t\"\n");
						j+=(LMUL-1);
					#endif
					j++;
					offset += align;
				}
				aux -= iter*LMUL;
		}	
		//x86 SECTION
		#if !defined(ASCALAR) && !defined(NEON) && !defined(RISCVSCALAR) && !defined(RVV07) && !defined(RVV1)
			fprintf(file,"\t\t\"addq $%d, %%%%rax\\n\\t\\t\"\n",offset);
			fprintf(file,"\t\t\"subq $1, %%%%rdi\\n\\t\\t\"\n");
			fprintf(file,"\t\t\"jnz Loop1_%%=\\n\\t\\t\"\n");
		//ARM SECTION
		#elif defined(ASCALAR) || defined(NEON)
			fprintf(file,"\t\t\"add x3, x3, #%d\\n\\t\"\n",offset);
			fprintf(file,"\t\t\"sub w1, w1, 1\\n\\t\"\n");
			fprintf(file,"\t\t\"cbnz w1, Loop1_%%=\\n\\t\"\n");
		//RISCV SECTION
		#elif defined(RISCVSCALAR) || defined(RVV07) || defined(RVV1)
			#if defined(RISCVSCALAR)
			fprintf(file,"\t\t\"addi t2, t2, %d\\n\\t\"\n",offset);
			#endif
			fprintf(file,"\t\t\"addi t1, t1, -1\\n\\t\"\n");
			fprintf(file,"\t\t\"bgtz t1, Loop1_%%=\\n\\t\"\n");
		#endif
	}
	
	num_rep = aux;
	offset = 0;
	fprintf(stderr,"\nMissing NUM REP: %lld\n", num_rep);
	
	for(i = 0; i < num_rep; i+=LMUL){
		for(k = 0;k < num_ld;k++){
			if(j  >= NUM_REGISTER){
				j = 0;
			}
			//x86 SECTION
			#if !defined(ASCALAR) && !defined(NEON) && !defined(RISCVSCALAR) && !defined(RVV07) && !defined(RVV1)
				fprintf(file,"\t\t\"%s %d(%%%%rax), %%%%%s%d\\n\\t\\t\"\n", assembly_op, offset, registr,j);
			//ARM SECTION
			#elif defined(ASCALAR) || defined(NEON)
				fprintf(file,"\t\t\"%s %s%d, [x3, #%d]\\n\\t\\t\"\n", assembly_op, registr, j, offset);
			//RISCV SECTION
			#elif defined(RISCVSCALAR)
				fprintf(file,"\t\t\"%s %s%d, %d(t2)\\n\\t\\t\"\n", assembly_op, registr, j, offset);
			#elif defined(RVV07) || defined(RVV1)
				fprintf(file,"\t\t\"%s %s%d, (t2)\\n\\t\\t\"\n", assembly_op, registr, j);
				fprintf(file,"\t\t\"add t2, t2, t3\\n\\t\\t\"\n");
				j+=(LMUL-1);
			#endif
			j++;
			offset += align;
			
		}
		for(k = 0;k < num_st;k++){
			if(j  >= NUM_REGISTER){
				j = 0;
			}
			//x86 SECTION
			#if !defined(ASCALAR) && !defined(NEON) && !defined(RISCVSCALAR) && !defined(RVV07) && !defined(RVV1)
				fprintf(file,"\t\t\"%s %%%%%s%d, %d(%%%%rax)\\n\\t\\t\"\n", assembly_op_2, registr, j, offset);
			//ARM SECTION
			#elif defined(ASCALAR) || defined(NEON)
				fprintf(file,"\t\t\"%s %s%d, [x3, #%d]\\n\\t\\t\"\n", assembly_op_2, registr, j, offset);
			//RISCV SECTION
			#elif defined(RISCVSCALAR)
				fprintf(file,"\t\t\"%s %s%d, %d(t2)\\n\\t\\t\"\n", assembly_op_2, registr, j, offset);
			#elif defined(RVV07) || defined(RVV1)
				fprintf(file,"\t\t\"%s %s%d, (t2)\\n\\t\\t\"\n", assembly_op_2, registr, j);
				fprintf(file,"\t\t\"add t2, t2, t3\\n\\t\\t\"\n");
				j+=(LMUL-1);
			#endif
			j++;
			offset += align;
		}
	}

	fprintf(stderr,"\nEXTRA INSTRUCTIONS: %lld\n", i-num_rep);

	#if !defined(ASCALAR) && !defined(NEON) && !defined(RISCVSCALAR) && !defined(RVV07) && !defined(RVV1)
		fprintf(file,"\t\t\"subq $1, %%%%r8\\n\\t\\t\"\n");
		fprintf(file,"\t\t\"jnz Loop2_%%=\\n\\t\\t\"\n");
	//ARM SECTION
	#elif defined(ASCALAR) || defined(NEON)
		fprintf(file,"\t\t\"sub w0, w0, 1\\n\\t\"\n");
		fprintf(file,"\t\t\"cbnz w0, Loop2_%%=\\n\\t\"\n");
	//RISCV SECTION
	#elif defined(RISCVSCALAR) || defined(RVV07) || defined(RVV1)
		fprintf(file,"\t\t\"addi t0, t0, -1\\n\\t\"\n");
		fprintf(file,"\t\t\"bgtz t0, Loop2_%%=\\n\\t\"\n");
	#endif
	
	
	//End Test Function
	//x86 SECTION
	#if !defined(ASCALAR) && !defined(NEON) && !defined(RISCVSCALAR) && !defined(RVV07) && !defined(RVV1)
		fprintf(file,"\t\t:\n\t\t:\"r\"(num_reps_t),\"r\" (test_var)\n\t\t:\"%%rax\",\"%%rdi\",\"%%r8\","COBLERED"\n\t);\n");
	//ARM SECTION
	#elif defined(ASCALAR) || defined(NEON)
		if (iter > 4096){ //Iterations trick for ARM, to avoid using immediate bigger than 4096
			fprintf(file,"\t\t:\n\t\t:\"r\"(num_reps_t),\"r\" (test_var), \"r\" (num_reps_2)\n\t\t:\"x3\","COBLERED"\n\t);\n");
		}else{
			fprintf(file,"\t\t:\n\t\t:\"r\"(num_reps_t),\"r\" (test_var)\n\t\t:\"x3\","COBLERED"\n\t);\n");
		}
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

void write_asm_mixed (int long long num_rep, int align, char * op, int ops, int num_ld, int num_st, int num_fp, char * registr, char * registr_flops, char * assembly_op, char * assembly_op_2, char * assembly_op_flops_1, char * assembly_op_flops_2, char * precision, int Vlen, int LMUL){
	
	int offset = 0;
	int aux = num_rep;
	int i, j = 0, k;
	FILE * file, * file_header;
	int num_aux;
	int long long iter;
	
	file_header =  fopen("Test/test_params.h", "w");
	file = file_header;

	fprintf(stderr,"\nVLEN: %d | LMUL: %d\n", Vlen, LMUL);

	iter = mem_math (num_rep, num_ld, num_st, &num_aux, align, Vlen, LMUL); //Calculate number of iterations
	int extra_iter = (num_rep-iter*num_aux);

	if (extra_iter < LMUL){
		extra_iter = LMUL - extra_iter;
	}else{
		extra_iter = (num_rep-iter*num_aux)%(LMUL);
	}
	
	fprintf(stderr, "\n MEM Iterations: %lld | NUM AUX: %d, | NUM REP: %lld | REAL NUM REP: %lld | Expected missing: %lld | Expected extra: %d\n", iter, num_aux, num_rep, iter*num_aux, num_rep-iter*num_aux, extra_iter);


	//ARM SECTION
	#if defined(ASCALAR) || defined(NEON)
		fprintf(file_header,"#define ARM 1\n");
		if (iter > 4096){
			fprintf(file_header,"#define VAR 1\n");
		}
	//RISCV SECTION
	#elif defined(RISCVSCALAR)
		fprintf(file_header,"#define RISCV 1\n");
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
	fprintf(file_header,"#define NUM_REP %lld\n",num_rep+extra_iter);
	if(strcmp(precision, "dp") == 0){
			fprintf(file_header,"#define PRECISION double\n");
	}else{
			fprintf(file_header,"#define PRECISION float\n");
	}
	fprintf(file_header,"#define ALIGN %d\n\n", align);
	fprintf(file_header,"#define FP_INST 1\n\n");
	
	//x86 SECTION
	#if !defined(ASCALAR) && !defined(NEON) && !defined(RISCVSCALAR) && !defined(RVV07) && !defined(RVV1)
	//Create Test Function
	fprintf(file,"static inline __attribute__((always_inline)) void test_function(PRECISION * test_var, int long long num_reps_t){\n");
	
	fprintf(file,"\t__asm__ __volatile__ (\n");
	
	fprintf(file,"\t\t\"movq %%0, %%%%r8\\n\\t\\t\"\n");
	fprintf(file,"\t\t\"Loop2_%%=:\\n\\t\\t\"\n");
	fprintf(file,"\t\t\"movq %%1, %%%%rax\\n\\t\\t\"\n");
	//ARM SECTION
	#elif defined(ASCALAR) || defined(NEON)
		if (iter > 4096){ //Iterations trick for ARM, to avoid using immediate bigger than 4096
			fprintf(file,"static inline __attribute__((always_inline)) void test_function(PRECISION * test_var, int long long num_reps_t, int long long num_reps_2){\n");
		}else{
			fprintf(file,"static inline __attribute__((always_inline)) void test_function(PRECISION * test_var, int long long num_reps_t){\n");
		}
		fprintf(file,"\t__asm__ __volatile__ (\n");

		fprintf(file,"\t\t\"mov w0, %%w0\\n\\t\"\n");
		fprintf(file,"\t\t\"Loop2_%%=:\\n\\t\"\n");
		fprintf(file,"\t\t\"mov x3, %%1\\n\\t\\t\"\n");
	//RISCV SECTION
	#elif defined(RISCVSCALAR) || defined(RVV07) || defined(RVV1)
		fprintf(file,"static inline __attribute__((always_inline)) void test_function(PRECISION * test_var, int long long num_reps_t){\n");
		fprintf(file,"\t__asm__ __volatile__ (\n");
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
		#if !defined(ASCALAR) && !defined(NEON) && !defined(RISCVSCALAR) && !defined(RVV07) && !defined(RVV1)
			fprintf(file,"\t\t\"movq $%lld, %%%%rdi\\n\\t\\t\"\n",iter);
			fprintf(file,"\t\t\"Loop1_%%=:\\n\\t\\t\"\n");
		//ARM SECTION
		#elif defined(ASCALAR) || defined(NEON)
			if (iter > 4096){ //Iterations trick for ARM, to avoid using immediate bigger than 4096
				fprintf(file,"\t\t\"mov w1, %%w2\\n\\t\"\n");
			}else{
				fprintf(file,"\t\t\"mov w1, %lld\\n\\t\"\n",iter);
			}
			
			fprintf(file,"\t\t\"Loop1_%%=:\\n\\t\"\n");
		//RISCV SECTION
		#elif defined(RISCVSCALAR) || defined(RVV07) || defined(RVV1)
			fprintf(file,"\t\t\"li t1, %lld\\n\\t\"\n", iter);
			fprintf(file,"\t\t\"Loop1_%%=:\\n\\t\"\n");
		#endif
		
		for(i = 0; i < num_aux; i+=LMUL){
			for(k = 0;k < num_ld;k++){
				if(j  >= NUM_REGISTER){
					j = 0;
				}
				//x86 SECTION
				#if !defined(ASCALAR) && !defined(NEON) && !defined(RISCVSCALAR) && !defined(RVV07) && !defined(RVV1)
					fprintf(file,"\t\t\"%s %d(%%%%rax), %%%%%s%d\\n\\t\\t\"\n", assembly_op, offset, registr,j);
				//ARM SECTION
				#elif defined(ASCALAR) || defined(NEON)
					fprintf(file,"\t\t\"%s %s%d, [x3, #%d]\\n\\t\\t\"\n", assembly_op, registr, j, offset);
				//RISCV SECTION
				#elif defined(RISCVSCALAR)
					fprintf(file,"\t\t\"%s %s%d, %d(t2)\\n\\t\\t\"\n", assembly_op, registr, j, offset);
				#elif defined(RVV07) || defined(RVV1)
						fprintf(file,"\t\t\"%s %s%d, (t2)\\n\\t\\t\"\n", assembly_op, registr, j);
						fprintf(file,"\t\t\"add t2, t2, t3\\n\\t\\t\"\n");
						j+=(LMUL-1);
				#endif
				j++;
				offset += align;
			}
			
			for (k = 0; k < num_fp; k++){
				if(j  >= NUM_REGISTER){
					j = 0;
				}
				//x86 AVX or SCALAR SECTION
				#if defined(AVX) || defined(AVX512) || defined(AVX2) || (!defined(SSE) && !defined(NEON) && !defined(ASCALAR) && !defined(RISCVSCALAR) && !defined(RVV07) && !defined(RVV1))
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
				#elif !defined(ASCALAR) && !defined(NEON) && !defined(RISCVSCALAR) && !defined(RVV07) && !defined(RVV1)
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
				//RISCV SECTION
				#elif defined(RISCVSCALAR)
					if(strcmp(op,"fma") == 0){
						fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j, registr_flops, j, registr_flops, j);
					}
					else{
						fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j, registr_flops, j);
					}
				#elif defined(RVV07) || defined(RVV1)
					fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr, j, registr, j, registr, j);
					j+=(LMUL-1);
				#endif
				j++;
			}
			for(k = 0;k < num_st;k++){
				if(j  >= NUM_REGISTER){
					j = 0;
				}
				//x86 SECTION
				#if !defined(ASCALAR) && !defined(NEON) && !defined(RISCVSCALAR) && !defined(RVV07) && !defined(RVV1)
					fprintf(file,"\t\t\"%s %%%%%s%d, %d(%%%%rax)\\n\\t\\t\"\n", assembly_op_2, registr, j, offset);
				//ARM SECTION
				#elif defined(ASCALAR) || defined(NEON)
					fprintf(file,"\t\t\"%s %s%d, [x3, #%d]\\n\\t\\t\"\n", assembly_op_2, registr, j, offset);
				//RISCV SECTION
				#elif defined(RISCVSCALAR)
					fprintf(file,"\t\t\"%s %s%d, %d(t2)\\n\\t\\t\"\n", assembly_op_2, registr, j, offset);
				#elif defined(RVV07) || defined(RVV1)
						fprintf(file,"\t\t\"%s %s%d, (t2)\\n\\t\\t\"\n", assembly_op_2, registr, j);
						fprintf(file,"\t\t\"add t2, t2, t3\\n\\t\\t\"\n");
						j+=(LMUL-1);
				#endif
				j++;
				offset += align;
			}
			aux -= iter*LMUL;
		}	
		//x86 SECTION
		#if !defined(ASCALAR) && !defined(NEON) && !defined(RISCVSCALAR) && !defined(RVV07) && !defined(RVV1)
			fprintf(file,"\t\t\"addq $%d, %%%%rax\\n\\t\\t\"\n",offset);
			fprintf(file,"\t\t\"subq $1, %%%%rdi\\n\\t\\t\"\n");
			fprintf(file,"\t\t\"jnz Loop1_%%=\\n\\t\\t\"\n");
		//ARM SECTION
		#elif defined(ASCALAR) || defined(NEON)
			fprintf(file,"\t\t\"add x3, x3, #%d\\n\\t\"\n",offset);
			fprintf(file,"\t\t\"sub w1, w1, 1\\n\\t\"\n");
			fprintf(file,"\t\t\"cbnz w1, Loop1_%%=\\n\\t\"\n");
		//RISCV SECTION
		#elif defined(RISCVSCALAR) || defined(RVV07) || defined(RVV1)
			#if defined(RISCVSCALAR)
			fprintf(file,"\t\t\"addi t2, t2, %d\\n\\t\"\n",offset);
			#endif
			fprintf(file,"\t\t\"addi t1, t1, -1\\n\\t\"\n");
			fprintf(file,"\t\t\"bgtz t1, Loop1_%%=\\n\\t\"\n");
		#endif
	}
	
	num_rep = aux;
	offset = 0;
	fprintf(stderr,"\nMissing NUM REP: %lld\n", num_rep);
	
	for(i = 0; i < num_rep; i+=LMUL){
		for(k = 0;k < num_ld;k++){
			if(j  >= NUM_REGISTER){
				j = 0;
			}
			//x86 SECTION
			#if !defined(ASCALAR) && !defined(NEON) && !defined(RISCVSCALAR) && !defined(RVV07) && !defined(RVV1)
				fprintf(file,"\t\t\"%s %d(%%%%rax), %%%%%s%d\\n\\t\\t\"\n", assembly_op, offset, registr,j);
			//ARM SECTION
			#elif defined(ASCALAR) || defined(NEON)
				fprintf(file,"\t\t\"%s %s%d, [x3, #%d]\\n\\t\\t\"\n", assembly_op, registr, j, offset);
			//RISCV SECTION
			#elif defined(RISCVSCALAR)
				fprintf(file,"\t\t\"%s %s%d, %d(t2)\\n\\t\\t\"\n", assembly_op, registr, j, offset);
			#elif defined(RVV07) || defined(RVV1)
				fprintf(file,"\t\t\"%s %s%d, (t2)\\n\\t\\t\"\n", assembly_op, registr, j);
				fprintf(file,"\t\t\"add t2, t2, t3\\n\\t\\t\"\n");
				j+=(LMUL-1);
			#endif
			j++;
			offset += align;
			
		}
		for (k = 0; k < num_fp; k++){
			if(j  >= NUM_REGISTER){
				j = 0;
			}
			//x86 AVX or SCALAR SECTION
			#if defined(AVX) || defined(AVX512) || defined(AVX2) || (!defined(SSE) && !defined(NEON) && !defined(ASCALAR) && !defined(RISCVSCALAR) && !defined(RVV07) && !defined(RVV1))
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
			#elif !defined(ASCALAR) && !defined(NEON) && !defined(RISCVSCALAR) && !defined(RVV07) && !defined(RVV1)
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
			//RISCV SECTION
			#elif defined(RISCVSCALAR)
				if(strcmp(op,"fma") == 0){
					fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j, registr_flops, j, registr_flops, j);
				}
				else{
					fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr_flops, j, registr_flops, j, registr_flops, j);
				}
			#elif defined(RVV07) || defined(RVV1)
				fprintf(file,"\t\t\"%s %s%d, %s%d, %s%d\\n\\t\"\n", assembly_op_flops_1, registr, j, registr, j, registr, j);
				j+=(LMUL-1);
			#endif
			j++;
		}
		for(k = 0;k < num_st;k++){
			if(j  >= NUM_REGISTER){
				j = 0;
			}
			//x86 SECTION
			#if !defined(ASCALAR) && !defined(NEON) && !defined(RISCVSCALAR) && !defined(RVV07) && !defined(RVV1)
				fprintf(file,"\t\t\"%s %%%%%s%d, %d(%%%%rax)\\n\\t\\t\"\n", assembly_op_2, registr, j, offset);
			//ARM SECTION
			#elif defined(ASCALAR) || defined(NEON)
				fprintf(file,"\t\t\"%s %s%d, [x3, #%d]\\n\\t\\t\"\n", assembly_op_2, registr, j, offset);
			//RISCV SECTION
			#elif defined(RISCVSCALAR)
				fprintf(file,"\t\t\"%s %s%d, %d(t2)\\n\\t\\t\"\n", assembly_op_2, registr, j, offset);
			#elif defined(RVV07) || defined(RVV1)
				fprintf(file,"\t\t\"%s %s%d, (t2)\\n\\t\\t\"\n", assembly_op_2, registr, j);
				fprintf(file,"\t\t\"add t2, t2, t3\\n\\t\\t\"\n");
				j+=(LMUL-1);
			#endif
			j++;
			offset += align;
		}
	}
	fprintf(stderr,"\nEXTRA INSTRUCTIONS: %lld\n", i-num_rep);
	#if !defined(ASCALAR) && !defined(NEON) && !defined(RISCVSCALAR) && !defined(RVV07) && !defined(RVV1)
		fprintf(file,"\t\t\"subq $1, %%%%r8\\n\\t\\t\"\n");
		fprintf(file,"\t\t\"jnz Loop2_%%=\\n\\t\\t\"\n");
	//ARM SECTION
	#elif defined(ASCALAR) || defined(NEON)
		fprintf(file,"\t\t\"sub w0, w0, 1\\n\\t\"\n");
		fprintf(file,"\t\t\"cbnz w0, Loop2_%%=\\n\\t\"\n");
	//RISCV SECTION
	#elif defined(RISCVSCALAR) || defined(RVV07) || defined(RVV1)
		fprintf(file,"\t\t\"addi t0, t0, -1\\n\\t\"\n");
		fprintf(file,"\t\t\"bgtz t0, Loop2_%%=\\n\\t\"\n");
	#endif
	
	
	//End Test Function
	//x86 SECTION
	#if !defined(ASCALAR) && !defined(NEON) && !defined(RISCVSCALAR) && !defined(RVV07) && !defined(RVV1)
		fprintf(file,"\t\t:\n\t\t:\"r\"(num_reps_t),\"r\" (test_var)\n\t\t:\"%%rax\",\"%%rdi\",\"%%r8\","COBLERED"\n\t);\n");
	//ARM SECTION
	#elif defined(ASCALAR) || defined(NEON)
		if (iter > 4096){ //Iterations trick for ARM, to avoid using immediate bigger than 4096
			fprintf(file,"\t\t:\n\t\t:\"r\"(num_reps_t),\"r\" (test_var), \"r\" (num_reps_2)\n\t\t:\"x3\","COBLERED"\n\t);\n");
		}else{
			fprintf(file,"\t\t:\n\t\t:\"r\"(num_reps_t),\"r\" (test_var)\n\t\t:\"x3\","COBLERED"\n\t);\n");
		}
	//RISCV SECTION
	#elif defined(RISCVSCALAR) || defined(RVV07) || defined(RVV1)
		fprintf(file,"\t\t:\n\t\t:\"m\"(num_reps_t),\"m\" (test_var)\n\t\t:"COBLERED"\n\t);\n");
	#endif
	fprintf(file,"}\n\n");
	
	fclose(file_header);
}	
