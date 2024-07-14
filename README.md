# The CARM Tool

<p>
  <a href="https://doi.org/10.1109/L-CA.2013.6" alt="Publication">
    <img src="https://img.shields.io/badge/DOI-10.1109/L--CA.2013.6-blue.svg"/></a>
    
</p>

<p>
  <a href="https://doi.org/10.1016/j.future.2020.01.044" alt="Publication">
    <img src="https://img.shields.io/badge/DOI-10.1016/j.future.2020.01.044-blue.svg"/></a>
    
</p>

This tool performs the micro-benchmarking necessary to constuct the Cache-Aware Roofline Model (CARM) for floating-point operations on Intel, AMD, AARCH64, and RISCV64 CPUs. It supports different instruction set extensions (AVX512, AVX, SSE, Scalar, Neon, RVV0.7), different data precisions (double- and single-precision), different floating point instructions (fused multiply and add, addition, multiplication and division). The micro-benchmarks can be performed for any number of threads. The tool provides as output a vizualization of CARM, as well as the measurements obtained for the different memory levels and selected FP instruction. The tool is also capable of the micro-benchmarking necessary to construct a memory bandwidth graph for various problem sizes, and perform mixed tests that stress the FP units and memory system at the same time.

The tool can also perform application analysis using either performance counters (via PAPI) or dynamic binary instrumentation (via DynamoRIO or Intel SDE), to view the output of these results in a CARM graph the GUI is required.

For better results visualization, ResultsGUI.py can be ran to generate a web browser based user interface for result visualization, results from other machines can be imported for visualization on any machine via the GUI, by moving the necessary result csv files to the results folder of the machine running the GUI.

## Requirements
- gcc (>= 4.9 for AVX512 tests and only tested with gcc 9.3)
- python (only tested with python 3.8.8)
    - matplolib (only tested with 3.3.4)
    - numpy
    - dash (GUI only)
    - dash-bootstrap-components (GUI only)
    - pandas (GUI only)
    - plotly (GUI only)
    - diskcache (GUI only)
- DynamoRIO (only tested with 10.93.19916) - for DBI application analysis (x86 and AARCH64)
    (Might require an edit to line 132 in the CustomClient Makefile to match the installed version of DynamoRIO)
- PAPI (only tested with 7.0.1) - for PMU application analysis
- Optional:
    - Intel SDE (only tested with 9.33.0)  - for DBI application analysis (x86)

## How to use (CLI)

The first step is to create an optional configuration file for the system to test under the **config** folder. This configuration file is optional in x86 systems since the tool is able to automatically scan the cache sizes present, however this detection can sometimes be wrong (you can check what cache sizes have been detected by using -v 3), so a configuration file is still advised. You can also skip the configuration file by using the arguments: 
-l1 <l1_size (per core)> -l2 <l2_size (per core)> -l3 <l3_size (total)> and --name <name>.

This configuration file can include four fields:
- identifier of the system
- L1 size per core (in KiB)
- L2 size per core (in KiB)
- Total L3 size (in KiB)

An example configuration file looks like:
```
name=venus
l1_cache=32
l2_cache=1024
l3_cache=25344
```

After the optional creation of the configuration file, the tool can executed as:

```
python run.py <path_config_file> --name <name> --test <test> --inst <fp_inst> --num_ops <num_ops> --isa <[isa]> --precision <[data_precision]> --ld_st_ratio <ld_st_ratio> --fp_ld_st_ratio <fp_ld_st_ratio> --dram_bytes <dram_bytes> is the size of the array used for the DRAM benchmark in KiB; --threads <[num_threads]> --freq <frequency> --l1_size <l1_size> --l2_size <l2_size> --l3_size <l3_size> --threads_per_l1 <threads_per_l1> --threads_per_l2 <threads_per_l2> --vector_length <vector_length> --verbose [0, 1, 2, 3] [--only_ld] [--only_st] [--no_freq_measure] [--set_freq] [--interleaved] [--dram_auto] [--plot]
```

where
 - <path_config_file> is the path for configuration file of the system. This should be your first argument.
 - --name <name> is the name for machine running the benchmarks (Default: unnamed)
 - --test <test> is the test to be performed (roofline, MEM, FP, L1, L2, L3, DRAM, mixedL1, mixedL2, mixedL3, mixedDRAM);
 - --inst <fp_inst> is the floating point instruction to be used (add, mul, div), fma performance is also measured by default;
 - --num_ops <num_ops> is the number of FP operations used for the FP benchmark;
 - --isa <isa> is the instruction set extension, multiple options can be seletcted by spacing them (avx512, avx2, sse, scalar, neon, armscalar, riscvvector, riscvscalar, auto);
 - --precision <data_precision> is the precision of the data, multiple options can be seletcted by spacing them (dp, sp);
 - --ld_st_ratio <ld_st_ratio> is the number of loads per store involed in the memory benchmarks;
 - --fp_ld_st_ratio <fp_ld_st_ratio> is the FP to Load/Store ratio involved in the mixed benchmarks;
 - --dram_bytes <dram_bytes> is the size of the array used for the DRAM benchmark in KiB;
 - --threads <num_threads> is the number of threads used for the test, multiple options can be selected by spacing them
 - --freq <frequency> expected CPU frequency if not auto-measuring (in GHz)
 - --l1_size <l1_size> is the L1 size per core of the machine being benchmarked
 - --l2_size <l2_size> is the L2 size per core of the machine being benchmarked
 - --l3_size <l3_size> is the total L3 size of the machine being benchmarked
 - --threads_per_l1 <threads_per_l1> are the expected number of threads that will share the same L1 cache (Default: 2)
 - --threads_per_l2 <threads_per_l2> are the expected number of threads that will share the same L2 cache (Default: 2)
 - --vector_length <vector_length> is the desired vector length in elements to be used (for riscvvector only, tool will use the max by default)
 - --verbose [0, 1, 2, 3] is the level of output information to be displayed during execution (0 -> No Output 1 -> Only ISA Errors and Test Details, 2 -> Intermediate Test Results, 3 -> Configuration Values Selected/Detected)
 - [--only_ld] indicates that the memory benchmarks will just contain loads (<ld_st_ratio> is ignored);
 - [--only_st] indicates that the memory benchmarks will just contain stores (<ld_st_ratio> is ignored);
 - [--no_freq_measure] disables the automatic frequency measuring (CPU frequency should be provided in config file or via --freq argument)
 - [--set_freq] will set the cpu frequency to the specified one (sudo is required, x86 only, might not work)
 - [--interleaved] indicates if the cores belong to interleaved numa domains (e.g. core 0 -> node 0, core 1 -> node 1, core 2 -> node 0, etc). Used for thread binding;
 - [--dram_auto] automatically adjust the DRAM test size according to number of threads to ensure individual thread data only fits in DRAM (Default: 0)
 - [--plot] enables the plotting of CARM/MEM results as an SVG image, allowing for result visualization without using the GUI (Default: 0)


A simple run can be executed with the command

```
python run.py <path_config_file>
```

which by default runs the micro-benchmarks necessary to obtain CARM data, for all available ISAs and double-precision variables. The FP instruction used is the ADD and FMA (32768 operations) and the memory benchmarks contain 2 loads per each store, with the DRAM test using an array with size 512MiB and 1 thread.


For additional information regarding the input arguments, run the command:

```
python run.py -h
```

To profile an application using **Performance Counters**, PMU_AI_Calculator.py should be executed with the following arguments:

 - <PAPI_path> Path to the PAPI directory.
 - <executable_path> Path to the executable to analyze.
 - <additional_args> Arguments for the executable that will be analyzed.
 - --name <name> Name for the machine running the executable (Default: unnamed);
 - --app_name <app_name> Name for the executable (if empty, executable name will be used);
 - --isa <isa> Main ISA used by the executable, if not sure leave blank (optional only for naming facilitation);

Note that this requires the PAPI_LST_INS, PAPI_SP_OPS, and PAPI_DP_OPS events to be available on your system.

To profile an application using **Dynamic Binary Instrumentation**, DBI_AI_Calculator.py should be executed with the following arguments:

 - <DBI_path> Path to the DynamoRIO directory, or Intel SDE directory if --sde is used.
 - <executable_path> Path to the executable to analyze.
 - [--roi] Measure only Region of Interest, or not. (Must be previously marked in the source code using the provided markers);
 - [--sde] Measure using Intel SDE, instead of DynamoRIO (x86 only);
 - --name <name> Name for the machine running the executable (Default: unnamed);
 - --app_name <app_name> Name for the executable (if empty, executable name will be used);
 - --isa <isa> Main ISA used by the executable, if not sure leave blank (optional only for naming facilitation);
 - --threads <threads> Number of threads used by the application (optional only for naming facilitation);
 - --precision <data_precision> Data Precision used by the application (optional only for naming facilitation);
 - <additional_args> Arguments for the executable that will be analyzed. (This should be your last argument)

Note that both the PMU analysis and the DBI with ROI analysis require the previous injection of the source code with Region of Interest specific code, to facilitate this proccess you can run CodeInject.py with the following arguments:

 - <source_path> Path to the source code file. (.c ou .cpp only)
 - <papi_path> Path to your local PAPI installation (only needed if using automatic compilation)
 - [--PMU] Inject code for PMU (PAPI) ROI instrumentation, instead of the DBI default.
 - [--new_file] Inject code in a new source file instead of altering the provided one. (Default: 0)
 - [--compile] Enable automatic compilation of the injected source code. (Default: 0)
 - <comp_args> Compiler arguments for automatic compilation. (This should be your last argument)

Note that the automatic compilation is intended for simple single source file applications and can be prone to errors, in such case manual compilation is advised, which in case of PMU analysis requires linking the PAPI library during compilation, this can usually be done following one of these methods:

```
Method 1:
gcc -<Compiler flags> -I/Path/To/Papi/src <source_file.c> -o <executable_file> /Path/To/Papi/src/libpapi.a

Method 2:
gcc -<Compiler flags> -I/${PAPI_DIR}/include -L/${PAPI_DIR}/lib  <source_file.c> -o <executable_file> -lpapi
```

Finaly the automatic code injection requires the presence of the Region of Interest flags in the source code file, these are:
```
//CARM ROI START
//CARM ROI END
```
In case you want to skip the automatic code injection, you can manually inject the necessary ROI instrumentation code in place of the flags.
For DBI analysis:
```
//Dependencies
#include <stdlib.h>
#include time.h

//CARM ROI START
struct timespec t_start_roi, t_end_roi;
clock_gettime(CLOCK_MONOTONIC, &t_start_roi);
__SSC_MARK(0xFACE);

//CARM ROI END
__SSC_MARK(0xDEAD);
clock_gettime(CLOCK_MONOTONIC, &t_end_roi);
double timing_duration = ((t_end_roi.tv_sec + ((double) t_end_roi.tv_nsec / 1000000000)) - (t_start_roi.tv_sec + ((double) t_start_roi.tv_nsec / 1000000000)));
FILE *timing_results_file = fopen("timing_results.txt", "w");
fprintf(timing_results_file, "Time Taken:\t%0.9lf seconds\\n", timing_duration);
fclose(timing_results_file);
```
For PMU analysis:
```
//Dependencies
#include <stdlib.h>
#include <papi.h>

//CARM ROI START
int retval = PAPI_hl_region_begin("roi");
    if ( retval != PAPI_OK ){
        printf("HL begin error\n");
        exit(0);
    }

//CARM ROI END
retval = PAPI_hl_region_end("roi");
    if ( retval != PAPI_OK ){
        printf("HL end error\n");
        exit(0);
    }
```

The profiling results are automatically stored in a csv assocaited with the provided machine name, these results can then be viewed using the GUI, make sure to match the machine name used in the profiling with the machine name used in the CARM benchmarks execution.

## How to use (GUI)

The tool can also be used via the GUI, by running **ResultsGui.py**, and then opening the provided link in the browser, the CARM benchmarks can be executed by opening the sidebar and entering your desired configuration values and clicking the "Run CARM Benchmarks" button, the "Stop Benchmark/Analysis" button can be used to stop execution at any time. After benchmark execution is finished, refreshing the page should be suficient to view the new results in the GUI. The tool output during benchmarking will be visible in the terminal where the ResultsGui.py script was launched from. Note that only the roofline test type is available in the GUI.

From the GUI you can also execute other functions of the tool, like the application profiling using either DBI or PMUs, this can be done by clicking the "Run Application Analysis" button, then selecting what kind of analysis method is desired (DBI, DBI with ROI, PMU with ROI), and providing the file path to the target application executable along with any arguments that it may take. Note that for Region of Interest analysis, the source code must be previously injected with instrumentation code, specific to the DBI method or the PMU method.

In order to inject the ROI instrumentation code, you can click the "Run Application ROI Code Injection" button, then selecting what kind of analysis method you will later use to profile the application, and then by providing the path to your source code file (.c or .cpp only), the necessary Region of Interest source code will be injected (this code can be injected in a copy of the source file if desired). Note that you must have the Region of Interest flags present in your code, these are:

```
//CARM ROI START
//CARM ROI END
```

After injecting the source code, you must compile it, in the case of the PMU option, the PAPI library must be linked during compilation to allow correct compilation, usually done in one of these 2 ways:

```
Method 1:
gcc -<Compiler flags> -I/Path/To/Papi/src <source_file.c> -o <executable_file> /Path/To/Papi/src/libpapi.a

Method 2:
gcc -<Compiler flags> -I/${PAPI_DIR}/include -L/${PAPI_DIR}/lib  <source_file.c> -o <executable_file> -lpapi
```


## In papers and reports, please refer to this tool as follows

A. Ilic, F. Pratas and L. Sousa, "Cache-aware Roofline model: Upgrading the loft," in IEEE Computer Architecture Letters, vol. 13, no. 1, pp. 21-24, 21 Jan.-June 2014, doi: 10.1109/L-CA.2013.6.

Diogo Marques, Aleksandar Ilic, Zakhar A. Matveev, and Leonel Sousa. "Application-driven cache-aware roofline model." Future Generation Computer Systems 107 (2020): 257-273.