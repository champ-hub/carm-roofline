#include <string>
using namespace std;
void create_benchmark_flops(int device, string target, string operation, string precision,
							int threads_per_block, int num_blocks, int iterations);