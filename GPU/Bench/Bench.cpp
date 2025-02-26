#include <getopt.h>
#include <stdlib.h>

#include <cstdlib>
#include <iostream>
#include <string>

#include "functions.h"

#define no_argument 0
#define required_argument 1
#define optional_argument 2

#define DEVICE 0  // TODO: FIX

using namespace std;

int main(int argc, char* argv[]) {
	const struct option longopts[] = {{"test", required_argument, 0, 't'},
									  {"compute", required_argument, 0, 'c'},
									  {"target", required_argument, 0, 'a'},
									  {"precision", required_argument, 0, 'p'},
									  {"operation", required_argument, 0, 'o'},
									  {"help", no_argument, 0, 'h'},
									  {0, 0, 0, 0}};

	int o;

	string test, target, precision, operation;
	int compute_capability = 0;

	while ((o = getopt_long(argc, argv, "t:c:a:p:o:h", longopts, NULL)) != -1) switch (o) {
			case 't':
				test = optarg;
				break;
			case 'c':
				compute_capability = atoi(optarg);
				break;
			case 'a':
				target = optarg;
				break;
			case 'p':
				precision = optarg;
				break;
			case 'o':
				operation = optarg;	 // fma, mul, add, div for cuda core operations
				break;
			case 'h':
				// TODO: IMPLEMENT
				// fprintf(stdout,
				//         "Usage: %s [OPTION]...\n\n\t-m \t M dimension [int] "
				//         "[default=1024]\n\t-n \t N "
				//         "dimension [int] [default=1024]\n\t-k \t K dimension [int] "
				//         "[default=1024]\n\t-a \t All "
				//         "dimensions [int]\n\t-c \t Disable Tensor Cores\n\n",
				//         argv[0]);
				exit(EXIT_SUCCESS);
			default:
				// TODO:IMPLEMENT
				// fprintf(stderr,
				//         "Usage: %s [OPTION]...\n\n\t-m \t M dimension [int] "
				//         "[default=1024]\n\t-n \t N "
				//         "dimension [int] [default=1024]\n\t-k \t K dimension [int] "
				//         "[default=1024]\n\t-a \t All "
				//         "dimensions [int]\n\t-c \t Disable Tensor Cores\n\n",
				//         argv[0]);
				exit(EXIT_FAILURE);
		}

	if (compute_capability == 0) {
		cerr << "ERROR: Compute Capability not set. Unable to compile benchmarks." << endl;
		return 3;
	}

	if (test == "FLOPS") {
		// TODO
		create_benchmark_flops(DEVICE, compute_capability, target, operation, precision, 1024,
							   32768);
	} else if (test == "MEM") {
		// TODO
	} else if (test == "MIXED") {
		// TODO
	} else {
		cerr << "ERROR: Test not found. Please select a valid test." << endl;
		return 2;
	};

	// cout << "Test:" << test << endl;
	// cout << "Precision:" << precision << endl;
	// cout << "Target:" << target << endl;
	return 0;
}