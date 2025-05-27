#include <string>
using namespace std;
void create_benchmark_flops(int device, string arch, string compute_capability, string operation, string precision,
							int threads_per_block, int num_blocks);

void create_benchmark_tensor(int device, string compute_capability, string precision,
							 int threads_per_block, int num_blocks);

void create_benchmark_mem(int device, string arch, string compute_capability, string target, string precision,
						  int threads_per_block, int num_blocks);