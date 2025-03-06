#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

using namespace std;

void create_benchmark_flops(int device, int compute_capability, string operation, string precision,
							int threads_per_block, int num_blocks) {
	if (!filesystem::is_directory("GPU/bin")) {
		if (!filesystem::create_directory("GPU/bin")) {
			cerr << "ERROR: Wasn't able to create bin directory" << endl;
			exit(6);
		}
	}
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
}

void create_benchmark_tensor(int device, int compute_capability, string operation, string precision,
							 int threads_per_block, int num_blocks) {
	if (!filesystem::is_directory("GPU/bin")) {
		if (!filesystem::create_directory("GPU/bin")) {
			cerr << "ERROR: Wasn't able to create bin directory" << endl;
			exit(14);
		}
	}
	// Tensor
	string text;

	ifstream input("GPU/Test/main_test_base.cu");
	ofstream output("GPU/bin/main_test.cu");

	while (getline(input, text)) {
		output << text << endl;
		if (text == "// DEFINE KERNEL PARAMETERS") {
			output << "#define THREADS_PER_BLOCK " << threads_per_block << endl;
			output << "#define NUM_BLOCKS " << num_blocks << endl;

			if (precision == "fp16_16" || precision == "fp16_32" || precision == "bf16") {
				output << "#define M 16\n#define N 8\n#define K 16" << endl;
			}
			output << "#define A_SIZE M *K *(THREADS_PER_BLOCK / 32) * NUM_BLOCKS" << endl;
			output << "#define B_SIZE K *N *(THREADS_PER_BLOCK / 32) * NUM_BLOCKS" << endl;
			output << "#define C_SIZE M *N *(THREADS_PER_BLOCK / 32) * NUM_BLOCKS" << endl;
		} else if (text == "// DEFINE PRECISION") {
			if (precision == "fp16_16") {
				output << "#define PRECISION_A"
					   << " half" << endl;
				output << "#define PRECISION_B"
					   << " half" << endl;
				output << "#define PRECISION_C"
					   << " half" << endl;
			} else if (precision == "fp16_32") {
				output << "#define PRECISION_A"
					   << " half" << endl;
				output << "#define PRECISION_B"
					   << " half" << endl;
				output << "#define PRECISION_C"
					   << " float" << endl;
			} else if (precision == "bf16") {
				output << "#define PRECISION_A"
					   << " nv_bfloat16" << endl;
				output << "#define PRECISION_B"
					   << " nv_bfloat16" << endl;
				output << "#define PRECISION_C"
					   << " float" << endl;
			}
		} else if (text == "// DEFINE DEVICE") {
			output << "#define DEVICE " << device << endl;
		} else if (text == "// DEFINE TEST") {
			output << "#define FLOPS 2" << endl;
		} else if (text == "// DEFINE FUNCTION") {
			output << "__global__ void benchmark(PRECISION_A *d_A, PRECISION_B *d_B, "
					  "PRECISION_C *d_C, int iterations);"
				   << endl;
			getline(input, text);
		} else if (text == "\t// DEFINE VECTORS") {
			output << "PRECISION_A *d_A;\ncudaMalloc((void **)&d_A, A_SIZE * sizeof(PRECISION_A));"
				   << endl;
			output << "PRECISION_B *d_B;\ncudaMalloc((void **)&d_B, B_SIZE * sizeof(PRECISION_B));"
				   << endl;
			output << "PRECISION_C *d_C;\ncudaMalloc((void **)&d_C, C_SIZE * sizeof(PRECISION_C));"
				   << endl;
			getline(input, text);
			getline(input, text);
		} else if (text.find("// DEFINE CALL") != string::npos) {
			output << "benchmark<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, iterations);"
				   << endl;
			getline(input, text);
		}
	}

	input.close();
	output.close();

	input.open("GPU/Test/benchmark_base.cu");
	output.open("GPU/bin/benchmark.cu");

	while (getline(input, text)) {
		output << text << endl;
		if (text == "// DEFINE PRECISION") {
			if (precision == "fp16_16") {
				output << "#define PRECISION_A"
					   << " half" << endl;
				output << "#define PRECISION_B"
					   << " half" << endl;
				output << "#define PRECISION_C"
					   << " half" << endl;
			} else if (precision == "fp16_32") {
				output << "#define PRECISION_A"
					   << " half" << endl;
				output << "#define PRECISION_B"
					   << " half" << endl;
				output << "#define PRECISION_C"
					   << " float" << endl;
			} else if (precision == "bf16") {
				output << "#define PRECISION_A"
					   << " nv_bfloat16" << endl;
				output << "#define PRECISION_B"
					   << " nv_bfloat16" << endl;
				output << "#define PRECISION_C"
					   << " float" << endl;
			}
		} else if (text == "\t// DEFINE INITIALIZATION") {
			if (precision == "fp16_16") {
				output << "half fragsA[8];\nhalf fragsB[4];\nhalf fragsC[4];" << endl;
				output << "fragsA[0] = d_A[id];\nfragsB[0] = d_B[id];\nfragsC[0] = d_C[id];"
					   << endl;
				output << "uint32_t const *A = reinterpret_cast<uint32_t const *>(&fragsA[0]);"
					   << endl;
				output << "uint32_t const *B = reinterpret_cast<uint32_t const *>(&fragsB[0]);"
					   << endl;
				output << "uint32_t *C = reinterpret_cast<uint32_t *>(&fragsC[0]);" << endl;
			} else if (precision == "dp")
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
			if (precision == "fp16_16") {
				output << "asm volatile(\"mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
						  "{%0,%1}, {%2,%3,%4,%5}, {%2,%3}, {%0,%1};\\n\""
					   << endl;
				output << ": \"+r\"(C[0]), \"+r\"(C[1]) : \"r\"(A[0]), \"r\"(A[1]), "
						  "\"r\"(A[2]), \"r\"(A[3]), \"r\"(B[0]), \"r\"(B[1]));"
					   << endl;
			} else {
				output << "\t\ta = a * a + b;\n\t\tb = b * b + c;\n\t\tc = c * c + d;\n\t\td = "
						  "d * d + a;"
					   << endl;
			}
		} else if (text == "// DEFINE FUNCTION") {
			output << "__global__ void benchmark(PRECISION_A *d_A, PRECISION_B *d_B, "
					  "PRECISION_C *d_C, int iterations) {"
				   << endl;
			getline(input, text);
		} else if (text == "\t// DEFINE CLOSURE") {
			output << "d_C[id] = fragsC[0];" << endl;
			getline(input, text);
		}
	}
	char buffer[100];
	cout << endl;
	sprintf(buffer, "make compute_capability=%d -f GPU/Test/Makefile", compute_capability);
	int check = system(buffer);
	if (check != 0) {
		cerr << "ERROR: It was not possible to generate the benchmark." << endl;
		exit(15);
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
		string text;

		ifstream input("GPU/Test/main_test_base.cu");
		ofstream output("GPU/bin/main_test.cu");

		while (getline(input, text)) {
			output << text << endl;
			if (text == "// DEFINE KERNEL PARAMETERS") {
				output << "#define THREADS_PER_BLOCK " << threads_per_block << endl;
				output << "#define NUM_BLOCKS " << num_blocks << endl;
				output << "#define STRIDE 32768 * 8L" << endl;
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
			} else if (text == "// DEFINE FUNCTION") {
				output << "__global__ void benchmark(PRECISION *d_X, PRECISION *d_Y, int "
						  "iterations);"
					   << endl;
				getline(input, text);
			} else if (text == "\t// DEFINE VECTORS") {
				output << "PRECISION *d_X;\ncudaMalloc((void **)&d_X, (NUM_BLOCKS * "
						  "THREADS_PER_BLOCK + 128 * STRIDE)* sizeof(PRECISION));"
					   << endl;
				output << "PRECISION *d_Y;\ncudaMalloc((void **)&d_Y, (NUM_BLOCKS * "
						  "THREADS_PER_BLOCK + 128 * STRIDE)* sizeof(PRECISION));"
					   << endl;
				getline(input, text);
				getline(input, text);
			} else if (text.find("// DEFINE CALL") != string::npos) {
				output << "benchmark<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_X, d_Y, iterations);"
					   << endl;
				getline(input, text);
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
				output << "#define STRIDE 32768 * 8L" << endl;
			} else if (text == "\t// DEFINE INITIALIZATION") {
				output << "\tPRECISION d;" << endl;
			} else if (text.find("// DEFINE LOOP") != string::npos) {
				output << "d = d_X[id + j * STRIDE];\nd_Y[id + j * STRIDE] = d;" << endl;
			} else if (text == "// DEFINE FUNCTION") {
				output << "__global__ void benchmark(PRECISION *d_X, PRECISION *d_Y, int "
						  "iterations) {"
					   << endl;
				getline(input, text);
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
			exit(13);
		}
	}
}