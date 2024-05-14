#include "config_test.h"

void select_ISA_flops(int * flop, char ** assembly_op, char * operation, char * precision){
	size_t len;
	if(strcmp(precision, "dp") == 0){
		*flop = DP_OPS;
		if(strcmp(operation, "div") == 0){
			len = strlen(DP_DIV);
			*assembly_op = (char *) malloc(len+1);
			strcpy(*assembly_op, DP_DIV);
		}	
		if(strcmp(operation, "add") == 0){
			len = strlen(DP_ADD);
			*assembly_op = (char *) malloc(len+1);
			strcpy(*assembly_op, DP_ADD);
		}		
		if(strcmp(operation, "mul") == 0){ 
			len = strlen(DP_MUL);
			*assembly_op = (char *) malloc(len+1);
			strcpy(*assembly_op, DP_MUL);  
		}
		if(strcmp(operation, "fma") == 0){
			len = strlen(DP_FMA);
			*assembly_op = (char *) malloc(len+1);
			strcpy(*assembly_op, DP_FMA);  
		}
	}else{
		*flop = SP_OPS;		
		if(strcmp(operation, "div") == 0){
			len = strlen(SP_DIV);
			*assembly_op = (char *) malloc(len+1);
			strcpy(*assembly_op,SP_DIV);
		}
		if(strcmp(operation, "add") == 0){
			size_t len = strlen(SP_ADD);
			*assembly_op = (char *) malloc(len+1);
			strcpy(*assembly_op,SP_ADD);
		}
		if(strcmp(operation, "mul") == 0){
			len = strlen(SP_MUL);
			*assembly_op = (char *) malloc(len+1);
			strcpy(*assembly_op,SP_MUL);
		}
		if(strcmp(operation, "fma") == 0){
			len = strlen(SP_FMA);
			*assembly_op = (char *) malloc(len+1);
			strcpy(*assembly_op,SP_FMA);	
		}

	}
}

void select_ISA_flops_register(char ** registr, char * precision){
	size_t len;
	if(strcmp(precision, "dp") == 0){
			len = strlen(FP_DP_REGISTER);
			*registr = (char *) malloc(len+1);
			strcpy(*registr, FP_DP_REGISTER);
		}
	else{
		len = strlen(FP_SP_REGISTER);
		*registr = (char *) malloc(len+1);
		strcpy(*registr, FP_SP_REGISTER);
	}
}

void select_ISA_mem(int * align, int * ops, char ** assembly_op, char * operation, char * precision){
	size_t len;
	if(strcmp(precision, "dp") == 0){
		*ops = DP_OPS;
		*align = DP_ALIGN;
		if(strcmp(operation, "load") == 0){
			len = strlen(DP_LOAD);
			*assembly_op = (char *) malloc(len+1);
			strcpy(*assembly_op,DP_LOAD);
			}
		if(strcmp(operation, "store") == 0){
			len = strlen(DP_STORE);
			*assembly_op = (char *) malloc(len+1);
			strcpy(*assembly_op,DP_STORE);
			}
	}else{
		*ops = SP_OPS;
		*align = SP_ALIGN;
		if(strcmp(operation, "load") == 0){
			len = strlen(SP_LOAD);
			*assembly_op = (char *) malloc(len+1);
			strcpy(*assembly_op,SP_LOAD);
			}
		if(strcmp(operation, "store") == 0){
			len = strlen(SP_STORE);
			*assembly_op = (char *) malloc(len+1);
			strcpy(*assembly_op,SP_STORE);
			}	
	}                                
}

void select_ISA_mem_register(char **registr, char * precision){
	size_t len;
	if(strcmp(precision, "dp") == 0){
			len = strlen(MEM_DP_REGISTER);
			*registr = (char *) malloc(len+1);
			strcpy(*registr, MEM_DP_REGISTER);
		}
	else{
		len = strlen(MEM_SP_REGISTER);
		*registr = (char *) malloc(len+1);
		strcpy(*registr, MEM_SP_REGISTER);
	}                               
}