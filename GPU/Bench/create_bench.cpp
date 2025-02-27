#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

using namespace std;

void create_benchmark_flops(int device, int compute_capability, string target, string operation,
							string precision, int threads_per_block, int num_blocks) {
	if (!filesystem::is_directory("GPU/bin")) {
		if (!filesystem::create_directory("GPU/bin")) {
			cerr << "ERROR: Wasn't able to create bin directory" << endl;
			exit(6);
		}
	}

	if (target == "cuda") {
		// CUDA cores
		string text;

		ifstream input("GPU/Test/main_test_base.cu");
		ofstream output("GPU/bin/main_test.cu");

		while (getline(input, text)) {
			output << text << endl;
			if (text == "// DEFINE KERNEL PARAMETERS") {
				output << "#define THREADS_PER_BLOCK " << threads_per_block << endl;
				output << "#define NUM_BLOCKS " << num_blocks << endl;
			} else if (text == "// DEFINE PRECISION") {
				string aux;
				if (precision == "sp")
					aux = "float";
				else if (precision == "dp")
					aux = "double";
				else if (precision == "int")
					aux = "int";
				else if (precision == "hp")
					aux = "half";
				else if (precision == "bf16")
					aux = "nv_bfloat16";

				output << "#define PRECISION " << aux << endl;
			} else if (text == "// DEFINE DEVICE") {
				output << "#define DEVICE " << device << endl;
			} else if (text == "// DEFINE TEST") {
				output << "#define FLOPS 1" << endl;
			}
		}

		input.close();
		output.close();

		input.open("GPU/Test/benchmark_base.cu");
		output.open("GPU/bin/benchmark.cu");

		while (getline(input, text)) {
			output << text << endl;
			if (text == "// DEFINE PRECISION") {
				string aux;
				if (precision == "sp")
					aux = "float";
				else if (precision == "dp")
					aux = "double";
				else if (precision == "int")
					aux = "int";
				else if (precision == "hp")
					aux = "half";
				else if (precision == "bf16")
					aux = "nv_bfloat16";

				output << "#define PRECISION " << aux << endl;
			} else if (text == "\t// DEFINE INITIALIZATION") {
				if (precision == "sp")
					output << "\tPRECISION a = 1.f;\n\tPRECISION b = 2.f;\n\tPRECISION c = "
							  "3.f;\n\tPRECISION d = 4.f;"
						   << endl;
				else if (precision == "dp")
					output << "\tPRECISION a = 1.;\n\tPRECISION b = 2.;\n\tPRECISION c = "
							  "3.;\n\tPRECISION d = 4.;"
						   << endl;
				else if (precision == "int")
					output << "\tPRECISION a = 1;\n\tPRECISION b = 2;\n\tPRECISION c = "
							  "3;\n\tPRECISION d = 4;"
						   << endl;
				else if (precision == "hp")
					output << "\tPRECISION a = __float2half(1.f);\n\tPRECISION b = "
							  "__float2half(2.f);\n\tPRECISION c = __float2half(3.f);\n\tPRECISION "
							  "d = __float2half(4.f);"
						   << endl;
				else if (precision == "bf16")
					output << "\tPRECISION a = __float2bfloat16(1.f);\n\tPRECISION b = "
							  "__float2bfloat16(2.f);\n\tPRECISION c = "
							  "__float2bfloat16(3.f);\n\tPRECISION "
							  "d = __float2bfloat16(4.f);"
						   << endl;
			} else if (text.find("// DEFINE LOOP") != string::npos) {
				if (precision == "hp" || precision == "bf16") {
					output << "\t\ta = __hfma(a, a, b);\n\t\tb = __hfma(b, b, c);\n\t\tc = "
							  "__hfma(c, c, d);\n\t\td = __hfma(d, d, a);"
						   << endl;
				} else {
					output << "\t\ta = a * a + b;\n\t\tb = b * b + c;\n\t\tc = c * c + d;\n\t\td = "
							  "d * d + a;"
						   << endl;
				}
			}
		}

		input.close();
		output.close();
		char buffer[100];
		cout << endl;
		sprintf(buffer, "make compute_capability=%d -f GPU/Test/Makefile", compute_capability);
		int check = system(buffer);
		if (check != 0) {
			cerr << "ERROR: It was not possible to generate the benchmark." << endl;
			exit(7);
		}

	} else {
		// Tensor
	}
}

void create_benchmark_mem(int device, int compute_capability, string target, string precision,
						  int threads_per_block, int num_blocks) {
	if (!filesystem::is_directory("GPU/bin")) {
		if (!filesystem::create_directory("GPU/bin")) {
			cerr << "ERROR: Wasn't able to create bin directory" << endl;
			exit(11);
		}
	}

	if (target == "shared") {
		// Shared Memory
		string text;

		ifstream input("GPU/Test/main_test_base.cu");
		ofstream output("GPU/bin/main_test.cu");

		while (getline(input, text)) {
			output << text << endl;
			if (text == "// DEFINE KERNEL PARAMETERS") {
				output << "#define THREADS_PER_BLOCK " << threads_per_block << endl;
				output << "#define NUM_BLOCKS " << num_blocks << endl;
			} else if (text == "// DEFINE PRECISION") {
				string aux;
				if (precision == "sp")
					aux = "float";
				else if (precision == "dp")
					aux = "double";
				else if (precision == "int")
					aux = "int";
				else if (precision == "hp")
					aux = "half";
				else if (precision == "bf16")
					aux = "nv_bfloat16";

				output << "#define PRECISION " << aux << endl;
			} else if (text == "// DEFINE DEVICE") {
				output << "#define DEVICE " << device << endl;
			} else if (text == "// DEFINE TEST") {
				output << "#define FLOPS 0" << endl;
			}
		}

		input.close();
		output.close();

		input.open("GPU/Test/benchmark_base.cu");
		output.open("GPU/bin/benchmark.cu");

		while (getline(input, text)) {
			output << text << endl;
			if (text == "// DEFINE PRECISION") {
				string aux;
				if (precision == "sp")
					aux = "float";
				else if (precision == "dp")
					aux = "double";
				else if (precision == "int")
					aux = "int";
				else if (precision == "hp")
					aux = "half";
				else if (precision == "bf16")
					aux = "nv_bfloat16";

				output << "#define PRECISION " << aux << endl;
				output << "#define THREADS_PER_BLOCK " << threads_per_block << endl;
			} else if (text == "\t// DEFINE INITIALIZATION") {
				output << "\t__shared__ PRECISION s[THREADS_PER_BLOCK];" << endl;
				output << "\tPRECISION d;" << endl;
			} else if (text.find("// DEFINE LOOP") != string::npos) {
				output << "\t\td = s[threadIdx.x];\n\t\ts[THREADS_PER_BLOCK - threadIdx.x -1] = d;"
					   << endl;
			}
		}

		input.close();
		output.close();
		char buffer[100];
		cout << endl;
		sprintf(buffer, "make compute_capability=%d -f GPU/Test/Makefile", compute_capability);
		int check = system(buffer);
		if (check != 0) {
			cerr << "ERROR: It was not possible to generate the benchmark." << endl;
			exit(12);
		}

	} else if (target == "global") {
		// Global Memory
	}
}