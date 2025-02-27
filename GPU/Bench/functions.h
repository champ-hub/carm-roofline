#include <string>
using namespace std;
void create_benchmark_flops(int device, int compute_capability, string target, string operation,
							string precision, int threads_per_block, int num_blocks);

void create_benchmark_mem(int device, int compute_capability, string target, string precision,
						  int threads_per_block, int num_blocks);