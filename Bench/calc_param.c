#include "config_test.h"


int long long flops_math(int long long fp){
	int long long iter;
	
	iter = 1;
	if(fp > BASE_LOOP_SIZE){
		iter = (int long long) floor((float)fp/BASE_LOOP_SIZE);
	}
	
	return iter;
}

int long long mem_math (int long long num_rep, int num_ld, int num_st, int * num_aux, int align, int Vlen){
	int long long iter;
	iter = 1;
	(*num_aux) = 1;
	#if defined(ASCALAR) || defined(NEON)
		if(num_rep*(num_ld+num_st) > BASE_LOOP_SIZE){
			while((*num_aux)*(num_ld+num_st) < BASE_LOOP_SIZE){
				//To avoid going over the 4096 limit on AARCH64 Assembly
				if (((*num_aux)*(num_ld+num_st)*align) < (4096 - (num_ld+num_st)*align)){
					(*num_aux) ++;
				}else{
					break;
				}
			}
			iter = (int long long) floor((float)num_rep/(*num_aux));
		}
	#elif defined(RISCVSCALAR)
		if(num_rep*(num_ld+num_st) > BASE_LOOP_SIZE){
			while((*num_aux)*(num_ld+num_st) < BASE_LOOP_SIZE){
				//To avoid going over the 2048 limit on RISCV Assembly
				if (((*num_aux)*(num_ld+num_st)*align) < (2048 - (num_ld+num_st)*align)){
					(*num_aux) ++;
				}else{
					break;
				}
			}
			iter = (int long long) floor((float)num_rep/(*num_aux));
		}
	#elif defined(RISCVVECTOR)
		if(num_rep*(num_ld+num_st) > BASE_LOOP_SIZE){
		while((*num_aux)*(num_ld+num_st) < BASE_LOOP_SIZE){
				(*num_aux) ++;
		}
		iter = (int long long) floor((float)num_rep/(*num_aux));
	}
	#else
		if(num_rep*(num_ld+num_st) > BASE_LOOP_SIZE){
			while((*num_aux)*(num_ld+num_st) < BASE_LOOP_SIZE){
				(*num_aux) ++;
			}
			iter = (int long long) floor((float)num_rep/(*num_aux));
		}
	#endif
	if(iter == 0){
		iter = 1;
	}
	return iter;
}






