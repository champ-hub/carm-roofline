#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

using namespace std;

void create_benchmark_flops(int device, string target, string operation, string precision,
							int threads_per_block, int num_blocks, int iterations) {
	if (!filesystem::is_directory("../bin")) {
		filesystem::create_directory("../bin");
	}

	if (target == "cuda") {
		// CUDA cores
		string text;

		ifstream input("../Test/main_test_base.cu");
		ofstream output("../bin/main_test.cu");

		while (getline(input, text)) {
			output << text << endl;
			if (text == "//DEFINE KERNEL PARAMETERS") {
				output << "#define THREADS_PER_BLOCK " << threads_per_block << endl;
				output << "#define NUM_BLOCKS " << num_blocks << endl;
				output << "#define ITERATIONS " << iterations << endl;
			} else if (text == "//DEFINE PRECISION") {
				string aux;
				if (precision == "sp")
					aux = "float";
				else if (precision == "dp")
					aux = "double";
				else if (precision == "int")
					aux = "int";
				else if (precision == "hp")
					aux = "__half";
				else if (precision == "bf16")
					exit(5);

				output << "#define PRECISION " << aux << endl;
			} else if (text == "//DEFINE DEVICE") {
				output << "#define DEVICE " << device << endl;
			}
		}

		input.close();
		output.close();

		input.open("../Test/benchmark_base.cu");
		output.open("../bin/benchmark.cu");

		while (getline(input, text)) {
			output << text << endl;
			if (text == "//DEFINE ITERATIONS") {
				output << "#define ITERATIONS " << iterations << endl;
			} else if (text == "//DEFINE PRECISION") {
				string aux;
				if (precision == "sp")
					aux = "float";
				else if (precision == "dp")
					aux = "double";
				else if (precision == "int")
					aux = "int";
				else if (precision == "hp")
					aux = "__half";
				else if (precision == "bf16")
					exit(5);

				output << "#define PRECISION " << aux << endl;
			}  // else if (text == "//DEFINE DEVICE") {
			   // 	output << "#define DEVICE " << device << endl;
			   // }
		}

		input.close();
		output.close();

		int check = system("make -f ../Test/Makefile");
		if (check != 0) {
			cerr << "ERROR: It was not possible to generate the benchmark." << endl;
			exit(6);
		}

	} else {
		// Tensor
	}
}