#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

using namespace std;

const uint Num_Reps = 64;

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

	ifstream input("GPU/Test/flops/cuda_cores.cu");
	ofstream output("GPU/bin/test.cu");

	while (getline(input, text)) {
		output << text << endl;
		if (text == "// DEFINE KERNEL PARAMETERS") {
			output << "#define THREADS_PER_BLOCK " << threads_per_block << endl;
			output << "#define NUM_BLOCKS " << num_blocks << endl;

		} else if (text == "// DEFINE NUM_REPS") {
			output << "#define NUM_REPS " << Num_Reps << endl;

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
			if (operation == "fma") {
				output << "#define MULTIPLIER 2" << endl;
			} else if (operation == "add" || operation == "mul") {
				output << "#define MULTIPLIER 1" << endl;
			}
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
				if (operation == "fma") {
					output << "\t\ta = __hfma(a, a, b);\n\t\tb = __hfma(b, b, c);\n\t\tc = "
							  "__hfma(c, c, d);\n\t\td = __hfma(d, d, a);"
						   << endl;
				} else if (operation == "add") {
					output << "\t\ta = __hadd(a, b);\n\t\tb = __hadd(b, c);\n\t\tc = "
							  "__hadd(c, d);\n\t\td = __hadd(d, a);"
						   << endl;
				} else if (operation == "mul") {
					output << "\t\ta = __hmul(a, b);\n\t\tb = __hmul(b, c);\n\t\tc = "
							  "__hmul(c, d);\n\t\td = __hmul(d, a);"
						   << endl;
				}
			} else {
				if (operation == "fma") {
					output << "\t\ta = a * a + b;\n\t\tb = b * b + c;\n\t\tc = c * c + d;\n\t\td = "
							  "d * d + a;"
						   << endl;
				} else if (operation == "add") {
					output << "\t\ta = a + b;\n\t\tb = b + c;\n\t\tc = c + d;\n\t\td = "
							  "d + a;"
						   << endl;
				} else if (operation == "mul") {
					output << "\t\ta = a * b;\n\t\tb = b * c;\n\t\tc = c * d;\n\t\td = "
							  "d * a;"
						   << endl;
				}
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

void create_benchmark_tensor(int device, int compute_capability, string precision,
							 int threads_per_block, int num_blocks) {
	if (!filesystem::is_directory("GPU/bin")) {
		if (!filesystem::create_directory("GPU/bin")) {
			cerr << "ERROR: Wasn't able to create bin directory" << endl;
			exit(14);
		}
	}
	// Tensor
	string text;

	ifstream input("GPU/Test/flops/tensor_cores.cu");
	ofstream output("GPU/bin/test.cu");

	while (getline(input, text)) {
		output << text << endl;
		if (text == "// DEFINE KERNEL PARAMETERS") {
			output << "#define THREADS_PER_BLOCK " << threads_per_block << endl;
			output << "#define NUM_BLOCKS " << num_blocks << endl;

			if (precision == "fp16_16" || precision == "fp16_32" || precision == "bf16") {
				output << "#define M 16\n#define N 8\n#define K 16" << endl;
			} else if (precision == "tf32") {
				output << "#define M 16\n#define N 8\n#define K 8" << endl;
			} else if (precision == "int8") {
				output << "#define M 16\n#define N 8\n#define K 32" << endl;
			} else if (precision == "int4") {
				output << "#define M 16\n#define N 8\n#define K 64" << endl;
			} else if (precision == "int1") {
				output << "#define M 16L\n#define N 8L\n#define K 128L" << endl;
			}

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
			} else if (precision == "tf32") {
				output << "#define PRECISION_A"
					   << " float" << endl;
				output << "#define PRECISION_B"
					   << " float" << endl;
				output << "#define PRECISION_C"
					   << " float" << endl;
			} else if (precision == "int8" || precision == "int4" || precision == "int1") {
				output << "#define PRECISION_A"
					   << " char" << endl;
				output << "#define PRECISION_B"
					   << " char" << endl;
				output << "#define PRECISION_C"
					   << " int" << endl;
			}
		} else if (text == "// DEFINE DEVICE") {
			output << "#define DEVICE " << device << endl;
		} else if (text == "// DEFINE NUM_REPS") {
			output << "#define NUM_REPS " << Num_Reps << endl;

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

			} else if (precision == "fp16_32") {
				output << "half fragsA[8];\nhalf fragsB[4];\nfloat fragsC[4];" << endl;
				output << "fragsA[0] = d_A[id];\nfragsB[0] = d_B[id];\nfragsC[0] = d_C[id];"
					   << endl;
				output << "uint32_t const *A = reinterpret_cast<uint32_t const *>(&fragsA[0]);"
					   << endl;
				output << "uint32_t const *B = reinterpret_cast<uint32_t const *>(&fragsB[0]);"
					   << endl;
				output << "float *C = reinterpret_cast<float *>(&fragsC[0]);" << endl;

			} else if (precision == "bf16") {
				output << "nv_bfloat16 fragsA[8];\nnv_bfloat16 fragsB[4];\nfloat fragsC[4];"
					   << endl;
				output << "fragsA[0] = d_A[id];\nfragsB[0] = d_B[id];\nfragsC[0] = d_C[id];"
					   << endl;
				output << "uint32_t const *A = reinterpret_cast<uint32_t const *>(&fragsA[0]);"
					   << endl;
				output << "uint32_t const *B = reinterpret_cast<uint32_t const *>(&fragsB[0]);"
					   << endl;
				output << "float *C = reinterpret_cast<float *>(&fragsC[0]);" << endl;

			} else if (precision == "tf32") {
				output << "float fragsA[4];\nfloat fragsB[2];\nfloat fragsC[4];" << endl;
				output << "fragsA[0] = d_A[id];\nfragsB[0] = d_B[id];\nfragsC[0] = d_C[id];"
					   << endl;
				output << "uint32_t const *A = reinterpret_cast<uint32_t const *>(&fragsA[0]);"
					   << endl;
				output << "uint32_t const *B = reinterpret_cast<uint32_t const *>(&fragsB[0]);"
					   << endl;
				output << "float *C = reinterpret_cast<float *>(&fragsC[0]);" << endl;

			} else if (precision == "int8" || precision == "int4") {
				output << "char fragsA[16];\nchar fragsB[8];\nint fragsC[4];" << endl;
				output << "fragsA[0] = d_A[id];\nfragsB[0] = d_B[id];\nfragsC[0] = d_C[id];"
					   << endl;
				output << "uint32_t const *A = reinterpret_cast<uint32_t const *>(&fragsA[0]);"
					   << endl;
				output << "uint32_t const *B = reinterpret_cast<uint32_t const *>(&fragsB[0]);"
					   << endl;
				output << "int *C = reinterpret_cast<int *>(&fragsC[0]);" << endl;
			} else if (precision == "int1") {
				output << "char fragsA[8];\nchar fragsB[4];\nint fragsC[4];" << endl;
				output << "fragsA[0] = d_A[id];\nfragsB[0] = d_B[id];\nfragsC[0] = d_C[id];"
					   << endl;
				output << "uint32_t const *A = reinterpret_cast<uint32_t const *>(&fragsA[0]);"
					   << endl;
				output << "uint32_t const *B = reinterpret_cast<uint32_t const *>(&fragsB[0]);"
					   << endl;
				output << "int *C = reinterpret_cast<int *>(&fragsC[0]);" << endl;
			}

		} else if (text.find("// DEFINE LOOP") != string::npos) {
			if (precision == "fp16_16") {
				output << "asm volatile(\"mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
						  "{%0,%1}, {%2,%3,%4,%5}, {%2,%3}, {%0,%1};\\n\""
					   << endl;
				output << ": \"+r\"(C[0]), \"+r\"(C[1]) : \"r\"(A[0]), \"r\"(A[1]), "
						  "\"r\"(A[2]), \"r\"(A[3]), \"r\"(B[0]), \"r\"(B[1]));"
					   << endl;

			} else if (precision == "fp16_32") {
				output << "asm volatile(\"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
						  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\\n\""
					   << endl;
				output << ": \"+f\"(C[0]), \"+f\"(C[1]), \"+f\"(C[2]), \"+f\"(C[3]) : \"r\"(A[0]), "
						  "\"r\"(A[1]), \"r\"(A[2]), \"r\"(A[3]), \"r\"(B[0]), \"r\"(B[1]));"
					   << endl;

			} else if (precision == "bf16") {
				output << "asm volatile(\"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
						  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\\n\""
					   << endl;
				output << ": \"+f\"(C[0]), \"+f\"(C[1]), \"+f\"(C[2]), \"+f\"(C[3]) : \"r\"(A[0]), "
						  "\"r\"(A[1]), \"r\"(A[2]), \"r\"(A[3]), \"r\"(B[0]), \"r\"(B[1]));"
					   << endl;

			} else if (precision == "tf32") {
				output << "asm volatile(\"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
						  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\\n\""
					   << endl;
				output << ": \"+f\"(C[0]), \"+f\"(C[1]), \"+f\"(C[2]), \"+f\"(C[3]) : \"r\"(A[0]), "
						  "\"r\"(A[1]), \"r\"(A[2]), \"r\"(A[3]), \"r\"(B[0]), \"r\"(B[1]));"
					   << endl;

			} else if (precision == "int8") {
				output << "asm volatile(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
						  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\\n\""
					   << endl;
				output << ": \"+r\"(C[0]), \"+r\"(C[1]), \"+r\"(C[2]), \"+r\"(C[3]) : \"r\"(A[0]), "
						  "\"r\"(A[1]), \"r\"(A[2]), \"r\"(A[3]), \"r\"(B[0]), \"r\"(B[1]));"
					   << endl;

			} else if (precision == "int4") {
				output << "asm volatile(\"mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
						  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\\n\""
					   << endl;
				output << ": \"+r\"(C[0]), \"+r\"(C[1]), \"+r\"(C[2]), \"+r\"(C[3]) : \"r\"(A[0]), "
						  "\"r\"(A[1]), \"r\"(A[2]), \"r\"(A[3]), \"r\"(B[0]), \"r\"(B[1]));"
					   << endl;

			} else if (precision == "int1") {
				output
					<< "asm volatile(\"mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.xor.popc "
					   "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\\n\""
					<< endl;
				output << ": \"+r\"(C[0]), \"+r\"(C[1]), \"+r\"(C[2]), \"+r\"(C[3]) : \"r\"(A[0]), "
						  "\"r\"(A[1]), \"r\"(B[0]));"
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

		ifstream input("GPU/Test/mem/shared.cu");
		ofstream output("GPU/bin/test.cu");

		while (getline(input, text)) {
			output << text << endl;
			if (text == "// DEFINE KERNEL PARAMETERS") {
				output << "#define THREADS_PER_BLOCK " << threads_per_block << endl;
				output << "#define NUM_BLOCKS " << num_blocks << endl;

			} else if (text == "// DEFINE PRECISION") {
				string aux;
				if (precision == "sp" || precision == "tf32")
					aux = "float";
				else if (precision == "dp")
					aux = "double";
				else if (precision == "int" || precision == "int8" || precision == "int4" ||
						 precision == "int1")
					aux = "int";
				else if (precision == "hp" || precision == "fp16_16" || precision == "fp16_32")
					aux = "half";
				else if (precision == "bf16")
					aux = "nv_bfloat16";

				output << "#define PRECISION " << aux << endl;

			} else if (text == "// DEFINE DEVICE") {
				output << "#define DEVICE " << device << endl;
			} else if (text == "// DEFINE NUM_REPS") {
				output << "#define NUM_REPS " << Num_Reps << endl;
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

		ifstream input("GPU/Test/mem/global.cu");
		ofstream output("GPU/bin/test.cu");

		while (getline(input, text)) {
			output << text << endl;
			if (text == "// DEFINE KERNEL PARAMETERS") {
				output << "#define THREADS_PER_BLOCK " << threads_per_block << endl;
				output << "#define NUM_BLOCKS " << num_blocks << endl;

			} else if (text == "// DEFINE PRECISION") {
				string aux;
				if (precision == "sp" || precision == "tf32")
					aux = "float";
				else if (precision == "dp")
					aux = "double";
				else if (precision == "int" || precision == "int8" || precision == "int4" ||
						 precision == "int1")
					aux = "int";
				else if (precision == "hp" || precision == "fp16_16" || precision == "fp16_32")
					aux = "half";
				else if (precision == "bf16")
					aux = "nv_bfloat16";

				output << "#define PRECISION " << aux << endl;

			} else if (text == "// DEFINE DEVICE") {
				output << "#define DEVICE " << device << endl;

			} else if (text == "// DEFINE NUM_REPS") {
				output << "#define NUM_REPS " << Num_Reps << endl;
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
	} else if (target == "L2") {
		// L2 Cache
		string text;

		ifstream input("GPU/Test/mem/l2.cu");
		ofstream output("GPU/bin/test.cu");

		// This test executes with a predefined number of threads and blocks for accurate
		// testing

		while (getline(input, text)) {
			output << text << endl;
			if (text == "// DEFINE PRECISION") {
				string aux;
				if (precision == "sp" || precision == "tf32")
					aux = "float";
				else if (precision == "dp")
					aux = "double";
				else if (precision == "int" || precision == "int8" || precision == "int4" ||
						 precision == "int1")
					aux = "int";
				else if (precision == "hp" || precision == "fp16_16" || precision == "fp16_32")
					aux = "half";
				else if (precision == "bf16")
					aux = "nv_bfloat16";

				output << "#define PRECISION " << aux << endl;

			} else if (text == "// DEFINE DEVICE") {
				output << "#define DEVICE " << device << endl;

			} else if (text == "// DEFINE NUM_REPS") {
				output << "#define NUM_REPS " << Num_Reps << endl;
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
			exit(22);
		}
	}
}