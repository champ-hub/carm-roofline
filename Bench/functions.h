//Operands Selection
void select_ISA_flops(int * flop, char ** assembly_op, char * operation, char * precision);
void select_ISA_flops_register(char ** registr, char * precision);
void select_ISA_mem(int * align, int * ops, char ** assembly_op, char * operation, char * precision);
void select_ISA_mem_register(char **registr, char * precision);

//Create Benchmarks
void create_benchmark_flops(char * op, char * precision, int long long fp, int Vlen, int LMUL, int verbose, int num_runs);
void create_benchmark_mem(int long long num_rep, int num_ld, int num_st, char * precision, int Vlen, int LMUL, int verbose, int num_runs);
void create_benchmark_mixed(char * op, int long long num_rep, int num_ld, int num_st, int num_fp, char * precision, int Vlen, int LMUL, int verbose, int num_runs);

//Write Assembly Codes
void write_asm_fp (int long long fp, char * op, int flops, char * registr, char * assembly_op_flops_1, char * assembly_op_flops_2, char * precision, int Vlen, int LMUL, int num_runs);
void write_asm_mem (int long long num_rep, int align, int ops, int num_ld, int num_st, char * registr, char * assembly_op, char * assembly_op_2, char * precision, int Vlen, int LMUL, int num_runs);
void write_asm_mixed (int long long num_rep, int align, char * op, int ops, int num_ld, int num_st, int num_fp, char * registr, char * registr_flops, char * assembly_op, char * assembly_op_2, char * assembly_op_flops_1, char * assembly_op_flops_2, char * precision, int Vlen, int LMUL, int num_runs);

//Params calculation
int long long flops_math(int long long fp);
int long long mem_math (int long long num_rep, int num_ld, int num_st, int * num_aux, int align);
