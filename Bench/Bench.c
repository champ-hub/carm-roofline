#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "functions.h"

int main (int argc, char*argv[]){
	
	int i;
	char * operation = NULL, * precision = NULL, * test = NULL;
	int long long num_rep_fp = -1, num_rep = -1;
	int num_ld = -1, num_st = -1, num_fp = -1, Vlen = -1, LMUL = -1, verbose = -1, num_runs = -1;
	
	for(i = 0; i < argc; i++){
		if(strcmp(argv[i], "-test") == 0){        			//Select the test type 
			size_t len = strlen(argv[i+1]);
			test = (char*) malloc(len+1);
			strcpy(test, argv[i+1]);
		}else if(strcmp(argv[i],"-num_LD") == 0){			//Number of LD INST per iteration
			num_ld = atoi(argv[i+1]);
		}else if(strcmp(argv[i],"-num_ST")== 0){			//Number of ST INST per iteration
			num_st = atoi(argv[i+1]);
		}else if(strcmp(argv[i],"-num_FP")== 0){			//Number of FP INST per iteration
			num_fp = atoi(argv[i+1]);
		}else if(strcmp(argv[i], "-op") == 0){        		//Operation type
			size_t len = strlen(argv[i+1]);
			operation = (char*) malloc(len+1);
			strcpy(operation, argv[i+1]);
		}else if(strcmp(argv[i], "-fp") == 0){ 				//Number of FP INST per iteration
			num_rep_fp = atol(argv[i+1]);
		}else if(strcmp(argv[i], "-precision")==0){			//Select data precision
			size_t len = strlen(argv[i+1]);
			precision = (char*)  malloc(len+1);
			strcpy(precision, argv[i+1]);
		}else if(strcmp(argv[i], "-num_rep")==0){
			num_rep = atol(argv[i+1]);
		}else if(strcmp(argv[i], "-Vlen")==0){
			Vlen = atol(argv[i+1]);
		}else if(strcmp(argv[i], "-LMUL")==0){
			LMUL = atol(argv[i+1]);
		}else if(strcmp(argv[i], "-verbose")==0){
			verbose = atol(argv[i+1]);
		}else if(strcmp(argv[i], "-num_runs")==0){
			num_runs = atol(argv[i+1]);
		}else if(strcmp(argv[i], "-h") ==0){				//Select to see help
			printf("Usage for FLOPS: ./Bench -test FLOPS [options]\nOptions:\n\t-op - Select the operation (add | mul | div | fma | mad)\n\t-fp:Number of FP INST\n\t-mode:SCALAR or SIMD INST (scalar | simd)\n\t-precision:Select FP precision (dp | fp)\n");
			printf("Usage for MEM: ./Bench -test MEM [options]\nOptions:\n\t-num_LD:Number of LD INST per loop iteration (>= 1 if num_ST = 0) \n\t-num_ST:Number of ST INST per loop iteration (>= 1 if num_LD = 0)\n\t-num_rep: Number of repetions of the LD/ST INST ( >=1 and useful to maintain a constant LD/ST ratio )\n\t-mode:SCALAR or SIMD INST (scalar | simd)\n\t-precision:Select FP precision (dp | fp)\n");
			printf("Usage for CARM: ./Bench -test CARM [options]\nOptions:\n\t-op - Select the operation (add | mul | div | fma | mad)\n\t-ratio_mem_fp:Ratio between memory INST and FP INST (>=1)\n\t-fp:Number of FP INST per loop iteration (>=1)\n\t-num_LD:Number of LD INST per loop iteration (> 0 (>=0 if num_ST > 0)) \n\t-num_ST:Number of ST INST per loop iteration (> 0 (=>0 if num_LD > 0))\n\t-num_rep: Number of repetions of the LD/ST INST ( >=1 and useful to maintain a constant LD/ST ratio )\n\t-mode:SCALAR or SIMD INST (scalar | simd)\n\t-precision:Select FP precision (dp | fp)\n");
			printf("Usage for MIXED: ./Bench -test MIXED [options]\nOptions:\n\t-num_LD:Number of LD INST per loop iteration (>= 1 if num_ST = 0) \n\t-num_ST:Number of ST INST per loop iteration (>= 1 if num_LD = 0)\n\t-num_rep: Number of repetions of the LD/ST INST ( >=1 and useful to maintain a constant LD/ST ratio )\n\t-mode:SCALAR or SIMD INST (scalar | simd)\n\t-precision:Select FP precision (dp | fp)\n");
			exit(1);
		}
	}

	if(test != 0){
		if(strcmp(test,"FLOPS") == 0){
			if(operation == NULL ||(strcmp(operation,"add") != 0 && strcmp(operation,"mul") != 0 && strcmp(operation,"div") != 0 && strcmp(operation,"fma") != 0 && strcmp(operation,"mad") != 0)|| num_rep_fp < 1 || precision == NULL || (strcmp(precision,"dp") != 0 && strcmp(precision,"sp") != 0)){
				printf("Usage for FLOPS: ./Bench -test FLOPS [options]\nOptions:\n\t-op - Select the operation (add | mul | div | fma | mad)\n\t-fp:Number of FP INST\n\t-precision:Select FP precision (dp | fp)\n");
				exit(2);
			}else{
				if (verbose > 3){
					printf("Benchmark flops\n");
				}
				create_benchmark_flops(operation, precision, num_rep_fp, Vlen, LMUL, verbose, num_runs);
			}
		}else if(strcmp(test,"MEM") == 0){
			if(num_ld < 0 || num_st < 0 || (num_ld == 0 && num_st == 0) || (num_rep < 0) || precision == NULL || (strcmp(precision,"dp") != 0 && strcmp(precision,"sp") != 0)){
				printf("Usage for MEM: ./Bench -test MEM [options]\nOptions:\n\t-num_LD:Number of LD INST per loop iteration (>= 1 if num_ST = 0) \n\t-num_ST:Number of ST INST per loop iteration (>= 1 if num_LD = 0)\n\t-num_rep: Number of repetions of the LD/ST INST ( >=1 and useful to maintain a constant LD/ST ratio )\n\t-precision:Select FP precision (dp | fp)\n");
				exit(3);
			}else{
				if (verbose > 3){
					printf("Benchmark mem\n");
				}
				create_benchmark_mem(num_rep, num_ld, num_st, precision, Vlen, LMUL, verbose, num_runs);
			}
		}else if(strcmp(test,"MIXED") == 0){
			if(num_ld < 0 || num_st < 0 || (num_ld == 0 && num_st == 0) || (num_rep < 0) || precision == NULL || (strcmp(precision,"dp") != 0 && strcmp(precision,"sp") != 0) || operation == NULL || (strcmp(operation,"add") != 0 && strcmp(operation,"mul") != 0 && strcmp(operation,"div") != 0 && strcmp(operation,"fma") != 0 && strcmp(operation,"mad") != 0)|| num_fp < 1 || precision == NULL || (strcmp(precision,"dp") != 0 && strcmp(precision,"sp") != 0)){
				printf("Usage for MIXED: ./Bench -test MIXED [options]\nOptions:\n\t-num_LD:Number of LD INST per loop iteration (>= 1 if num_ST = 0) \n\t-num_ST:Number of ST INST per loop iteration (>= 1 if num_LD = 0)\n\t-num_rep: Number of repetions of the LD/ST INST ( >=1 and useful to maintain a constant LD/ST ratio )\n\t-precision:Select FP precision (dp | fp)\n");
				exit(3);
			}else{
				if (verbose > 3){
					printf("Benchmark mixed\n");
				}
				create_benchmark_mixed(operation, num_rep, num_ld, num_st, num_fp, precision, Vlen, LMUL, verbose, num_runs);
			}
		}else{
			printf("Error, select a valid test. Select -h for help\n");
			exit(3);
		}

	}else{
		printf("Error, select a valid test. Select -h for help\n");
		exit(4);	
	}
	
	free(operation);
	free(precision);
	
	return 0;
}
