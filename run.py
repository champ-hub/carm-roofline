import argparse
import os
import subprocess
import csv
plot_numpy = 1
try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    plot_numpy = None
import datetime
import platform
import sys

import utils as ut

rvv_SVE_compiler_path = "gcc"

script_dir = os.path.dirname(os.path.abspath(__file__))
bench_ex = os.path.join(script_dir, 'Bench', 'Bench')
test_ex = os.path.join(script_dir, 'bin', 'test')

#Mapping between ISA and memory transfer size
mem_inst_size = {"avx512": {"sp": 64, "dp": 64}, "avx": {"sp": 32, "dp": 32}, "avx2": {"sp": 32, "dp": 32}, "sse": {"sp": 16, "dp": 16}, "scalar": {"sp": 4, "dp": 8}, "neon": {"sp": 16, "dp": 16}, "armscalar": {"sp": 4, "dp": 8},  "sve": {"sp": 4, "dp": 8}, "riscvscalar": {"sp": 4, "dp": 8}, "rvv0.7": {"sp": 4, "dp": 8}, "rvv1.0": {"sp": 4, "dp": 8}}
ops_fp = {"avx512": {"sp": 16, "dp": 8}, "avx": {"sp": 8, "dp": 4}, "avx2": {"sp": 8, "dp": 4}, "sse": {"sp": 4, "dp": 2}, "scalar": {"sp": 1, "dp": 1}, "neon":{"sp": 4, "dp": 2}, "armscalar": {"sp": 1, "dp": 1}, "sve": {"sp": 1, "dp": 1}, "riscvscalar": {"sp": 1, "dp": 1}, "rvv0.7": {"sp": 1, "dp": 1}, "rvv1.0": {"sp": 1, "dp": 1}}
test_sizes = [2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 600, 768, 1024, 1536, 2048, 3072, 4096, 5120, 6144, 8192, 10240, 12288, 16384, 24567, 32768, 65536, 98304, 131072, 262144, 393216, 524288]

x86_ISAs = ["avx512", "avx2", "sse", "scalar"]
other_ISAs = ["sve", "neon", "armscalar", "rvv1.0", "rvv0.7", "riscvscalar"]
vector_agnostic_ISA = ["sve", "rvv0.7", "rvv1.0"]

def check_hardware(isa_set, freq, set_freq, verbose, precision, l1_size, l2_size, l3_size, VLEN, LMUL):
    CPU_Type = platform.machine()
    config_dir = os.path.join(script_dir, 'config', 'auto_config')

    if CPU_Type == "x86_64":
        
        auto_args = autoconf(freq*1000000, set_freq)
        #If user defines no ISA in arguments, uses all of the best ones
        if (isa_set[0] == "auto"):
            #If avx512 is supported
            if auto_args[0] == "avx512":
                isa_set[0] = auto_args[0]
                #We can then use 32 registers for avx2
                if auto_args[1] == "avx2":
                    isa_set.append("avx2")
                    #If sse is supported
                    if auto_args[2] == "sse":
                        isa_set.append("sse")
                #If we have avx512 but dont have avx2 (unlikely)
                else:
                    if auto_args[2] == "sse":
                        isa_set.append("sse")
            #If only avx2 is supported we only use 16 registers
            elif auto_args[1] == "avx2":
                isa_set[0] = "avx2"
                #If sse is supported
                if auto_args[2] == "sse":
                    isa_set.append("sse")
            #If only sse is supported
            elif auto_args[2] == "sse":
                isa_set[0] = "sse"
            isa_set.append("scalar")
        else:
            supported_isas = []

            for item in isa_set:
                if item in other_ISAs:
                    if (verbose > 0):
                        print("WARNING: Selected ISA " + item + " was detected and removed since it is not supported by " + CPU_Type + " architectures.")
                else:
                    supported_isas.append(item)
            isa_set = supported_isas
            supported_isas = []

            for isa in isa_set:
                #If user defines ISA, check support on the machine
                if(isa == "avx512" and auto_args[0] != "avx512"):
                    if (verbose > 0):
                        print("WARNING: AVX512 not supported on this machine.")
                    continue
                if((isa == "avx2" or isa == "avx") and auto_args[0] == "avx512" and auto_args[1] == "avx2"):
                    #To use more registers with avx2
                    isa = "avx2"
                if ((isa == "avx2" or isa == "avx") and auto_args[1] != "avx2"):
                    if (verbose > 0):
                        print("WARNING: AVX2 not supported on this machine.")
                    continue
                if (isa == "sse" and auto_args[2] != "sse"):
                    if (verbose > 0):
                        print("WARNING: SSE not supported on this machine.")
                    continue
                supported_isas.append(isa)
            #If no ISA specified is valid, default to Scalar
            if not supported_isas:
                supported_isas = ["scalar"]

            isa_set = supported_isas
        if (verbose > 2):
            print("-----------------CPU INFORMATION-----------------")
            print("CPU Vendor:", auto_args[3])
            print("Vector Instruction ISAs Supported:", auto_args[0], auto_args[1], auto_args[2])
            print("L1 cache size:", auto_args[4], "KB" + (f" (Warning: User specified: {l1_size} KB)" if l1_size > 0 and l1_size != auto_args[4] else ""))
            print("L2 cache size:", auto_args[5], "KB" + (f" (Warning: User specified: {l2_size} KB)" if l2_size > 0 and l2_size != auto_args[5] else ""))
            print("L3 cache size:", auto_args[6], "KB" + (f" (Warning: User specified: {l3_size} KB)" if l3_size > 0 and l3_size != auto_args[6] else ""))
        #uses cache sizes from probing if not present in config file or arguments
        if (l1_size == 0):
            l1_size = auto_args[4]
        if (l2_size == 0):
            l2_size = auto_args[5]
        if (l3_size == 0):
            l3_size = auto_args[6]

        if VLEN != 1:
            if (verbose > 0):
                print("WARNING: --vector_length (-vl) argument is only used for RVV benchmarks.")
            VLEN = 1
        if LMUL != 1:
            if (verbose > 0):
                print("WARNING: --vector_lmul (-vlmul) argument is only used for RVV benchmarks.")
            LMUL = 1
        
        return isa_set, int(l1_size), int(l2_size), int(l3_size), int(VLEN), int(LMUL)

    elif CPU_Type == "aarch64":
        auto_args = autoconf(freq*1000000, set_freq)
        #if we have an ARM CPU
        supported_isas = []
        for item in isa_set:
            if item in ["sse", "avx2", "avx512", "rvv1.0", "rvv0.7", "riscvscalar"]:
                if (verbose > 0):
                    print("WARNING: Selected ISA " + item + " was detected and removed since it is not supported by " + CPU_Type + " architectures.")
            else:
                supported_isas.append(item)
        if LMUL != 1:
            if (verbose > 0):
                print("WARNING: --vector_lmul (-vlmul) argument is only used for RVV benchmarks.")
            LMUL = 1
        if (verbose > 2):
            print("-----------------CPU INFORMATION-----------------")
            print("CPU ISA: ARM")
        isa_set = supported_isas
        if len(isa_set) == 0:
            isa_set.append("auto")
        #If in auto mode
        if (isa_set[0] == "auto"):
            #If Neon is supported we move on to SVE check
            if (auto_args[0]=="neon"):
                isa_set.append("neon")
                isa_set.append("armscalar")
            #If Neon is not supported we switch to scalar
            else:
                isa_set[0] = "armscalar"
                if (verbose > 2):
                    print("No Vector Instruction ISAs Supported")
                if VLEN != 1:
                    if (verbose > 0):
                        print("WARNING: --vector_length (-vl) argument is only used for SVE benchmarks.")
                VLEN = 1
                return isa_set, int(l1_size), int(l2_size), int(l3_size), int(VLEN), int(LMUL)
        #If user specified an ISA
        else:
            #If there is no NEON support, then there is also no SVE support, switch to scalar
            if (auto_args[0]!="neon"):
                if (verbose > 0):
                    if ("neon" in isa_set):
                        print("WARNING: NEON not supported on this machine.")
                    if ("sve" in isa_set):
                        print("WARNING: SVE not supported on this machined")
                isa_set.clear()
                isa_set.append("armscalar")
                if (verbose > 2):
                    print("No Vector Instruction ISAs Supported")
                if VLEN != 1:
                    if (verbose > 0):
                        print("WARNING: --vector_length (-vl) argument is only used for SVE benchmarks.")
                VLEN = 1
                return isa_set, int(l1_size), int(l2_size), int(l3_size), int(VLEN), int(LMUL)

        isa_set = ["armscalar" if x == "scalar" else x for x in isa_set]

        #Now we check for SVE
        sve_source = os.path.join(config_dir, 'SVE_Vector.c')
        sve_ex = os.path.join(config_dir, 'SVE_Vector')
        try:
            subprocess.run([rvv_SVE_compiler_path, "-o", sve_ex, sve_source, "-march=armv8-a+sve"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print("SVE Compiler Error Output:", e.stderr.decode("utf-8"))
        if not os.path.exists(sve_ex):
            if (verbose > 2):
                print("Vector Instruction ISAs Supported: NEON")
            if "sve" in isa_set:
                if (verbose>0):
                    print("WARNING: Compilation of SVE vector length detection program failed please check your compiler support.\n"
                "Compiler used: ", rvv_SVE_compiler_path)
                isa_set.remove("sve")
            if VLEN != 1:
                if (verbose > 0):
                    print("WARNING: --vector_length (-vl) argument is only used for SVE benchmarks.")
            VLEN = 1
            if "auto" in isa_set:
                isa_set.remove("auto")
            return isa_set, int(l1_size), int(l2_size), int(l3_size), int(VLEN), int(LMUL)
        else:
            try:
                result = subprocess.run([sve_ex], stdout=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                print("SVE Execution Error Output:", e.stderr.decode("utf-8"))

            VLEN_Check = int(result.stdout.decode('utf-8'))
            os.remove(sve_ex)
            if (verbose > 2):
                print("Vector Instruction ISAs Supported: NEON SVE")
            
            if VLEN < VLEN_Check:
                if VLEN == 1:
                    VLEN = VLEN_Check
            else:
                if verbose > 0:
                    print("WARNING: Requested SVE Vector Length Not Supported")
            VLEN = VLEN_Check
            if verbose > 0:
                print("WARNING: Custom SVE Vector Length Not Supported")
            if (verbose > 2):
                if "dp" in precision:
                    print("Maximum vector size allows " + str(VLEN_Check) + " double precision elements.")
                if "sp" in precision:
                    print("Maximum vector size allows " + str(VLEN_Check*2) + " single precision elements.")
            if isa_set[0] == "auto":
                isa_set.clear()
                isa_set.append("sve")
                isa_set.append("neon")
                isa_set.append("armscalar")
                
            return isa_set, int(l1_size), int(l2_size), int(l3_size), int(VLEN), int(LMUL)
        
    elif CPU_Type == "riscv64":
        #if we have a RISCV CPU
        supported_isas = []
        if (verbose > 2):
            print("-----------------CPU INFORMATION-----------------")
            print("CPU ISA: RISC-V")

        for item in isa_set:
            if item in ["sve", "neon", "armscalar", "sse", "avx2", "avx512"]:
                if (verbose > 0):
                    print("WARNING: Selected ISA " + item + " was detected and removed since it is not supported by " + CPU_Type + " architectures.")
            else:
                supported_isas.append(item)
        isa_set = supported_isas
        if len(isa_set) == 0:
            isa_set.append("auto")
        
        isa_set = ["riscvscalar" if x == "scalar" else x for x in isa_set]
        if (isa_set[0] == "auto" or "rvv0.7" in isa_set or "rvv1.0" in isa_set):
            rvv07_source = os.path.join(config_dir, 'RISCV07_Vector.c')
            rvv10_source = os.path.join(config_dir, 'RISCV10_Vector.c')
            rvv07_ex = os.path.join(config_dir, 'RISCV07_Vector')
            rvv10_ex = os.path.join(config_dir, 'RISCV10_Vector')
            rvv10_compilation = True
            rvv07_compilation = True
            rvv10_execution = True
            rvv07_execution = True

            #Check if RVV1.0 compilation works
            try:
                subprocess.run([rvv_SVE_compiler_path, "-o", rvv10_ex, rvv10_source, "-march=rv64gcv"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError as e:
                print("RVV1.0 Compiler Error Output:", e.stderr.decode("utf-8"))
            if not os.path.exists(rvv10_ex):
                rvv10_compilation = False
                rvv10_execution = False
                if "rvv1.0" in isa_set:
                    if (verbose>0):
                        print("WARNING: Compilation of RVV vector length detection program failed for RVV1.0, please check your compiler support.\n"
                    "Compiler used: ", rvv_SVE_compiler_path)
                    isa_set.clear()
                    isa_set.append("riscvscalar")
                    VLEN = LMUL = 1
                    return isa_set, int(l1_size), int(l2_size), int(l3_size), int(VLEN), int(LMUL)
 
            #Check if RVV0.7 compilation works
            try:
                if "gcc" in rvv_SVE_compiler_path:
                    subprocess.run([rvv_SVE_compiler_path, "-o", rvv07_ex, rvv07_source, "-march=rv64gcv0p7"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                elif "clang" in rvv_SVE_compiler_path:
                    #For Clang -menable-experimental-extensions is sometimes needed
                    subprocess.run([rvv_SVE_compiler_path, "-o", rvv07_ex, rvv07_source, "-march=rv64gcv0p7", "-menable-experimental-extensions"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError as e:
                print("RVV0.7 Compiler Error Output:", e.stderr.decode("utf-8"))
            if not os.path.exists(rvv07_ex):
                rvv07_compilation = False
                rvv07_execution = False
                if "rvv0.7" in isa_set:
                    if (verbose>0):
                        print("WARNING: Compilation of RVV vector length detection program failed for RVV0.7, please check your compiler support.\n"
                    "Compiler used: ", rvv_SVE_compiler_path)
                    isa_set.clear()
                    isa_set.append("riscvscalar")
                    VLEN = LMUL = 1
                    return isa_set, int(l1_size), int(l2_size), int(l3_size), int(VLEN), int(LMUL)
            #If both RVV1.0 and RVV0.7 compilation fail, go for scalar
            if not rvv10_compilation and not rvv07_compilation:
                isa_set.clear()
                isa_set.append("riscvscalar")
                VLEN = LMUL = 1
                if (verbose>0):
                    print("WARNING: Compilation of RVV vector length detection program failed for both RVV0.7 and RVV1.0, please check your compiler support.\n"
                    "Compiler used: ", rvv_SVE_compiler_path)
                return isa_set, int(l1_size), int(l2_size), int(l3_size), int(VLEN), int(LMUL)
            
            #Check if execution with RVV1.0 works
            if ("rvv1.0" in isa_set or "auto" in isa_set) and rvv10_compilation:
                try:
                    result = subprocess.run([rvv10_ex], stdout=subprocess.PIPE)
                except subprocess.CalledProcessError as e:
                    print("RVV1.0 Execution Error Output:", e.stderr.decode("utf-8"))
                    rvv10_execution = False
                    if "auto" not in isa_set:
                        if (verbose>0):
                            print("WARNING: Execution of RVV vector length program failed for RVV1.0, please check your CPU RVV support.")
                        isa_set.clear()
                        isa_set.append("riscvscalar")
                        VLEN = LMUL = 1
                        return isa_set, int(l1_size), int(l2_size), int(l3_size), int(VLEN), int(LMUL)
                if (verbose > 2):
                    print("Vector Instruction ISAs Supported: RVV1.0")
            #Check if execution with RVV0.7 works
            if ("rvv0.7" in isa_set or "auto" in isa_set) and rvv07_compilation:
                try:
                    result = subprocess.run([rvv07_ex], stdout=subprocess.PIPE)
                except subprocess.CalledProcessError as e:
                    print("RVV0.7 Compiler Error Output:", e.stderr.decode("utf-8"))
                    rvv07_execution = False
                    if "auto" not in isa_set:
                        if (verbose>0):
                            print("WARNING: Execution of RVV vector length program failed for RVV0.7, please check your CPU RVV support.")
                        isa_set.clear()
                        isa_set.append("riscvscalar")
                        VLEN = LMUL = 1
                        return isa_set, int(l1_size), int(l2_size), int(l3_size), int(VLEN), int(LMUL)
                if (verbose > 2):
                    print("Vector Instruction ISAs Supported: RVV0.7")
            #If both RVV1.0 and RVV0.7 executions fail, go for scalar
            if not rvv10_execution and not rvv07_execution:
                isa_set.clear()
                isa_set.append("riscvscalar")
                VLEN = LMUL = 1
                if (verbose>0):
                    print("WARNING: Execution of RVV vector length detection program failed for both RVV0.7 and RVV1.0, please check CPU RVV support.")
                return isa_set, int(l1_size), int(l2_size), int(l3_size), int(VLEN), int(LMUL)

            VLEN_Check = int(result.stdout.decode('utf-8'))
            if rvv10_compilation:
                os.remove(rvv10_ex)
            if rvv07_compilation:
                os.remove(rvv07_ex)
            if VLEN < VLEN_Check:
                if VLEN == 1:
                    VLEN = VLEN_Check
            else:
                if verbose >0:
                    print("WARNING: Requested RVV Vector Length Not Supported")
                VLEN = VLEN_Check
            if (verbose > 2):
                if "dp" in precision:
                    print("Maximum vector size allows " + str(VLEN_Check) + " double precision elements.")
                if "sp" in precision:
                    print("Maximum vector size allows " + str(VLEN_Check*2) + " single precision elements.")
            if isa_set[0] == "auto":
                isa_set.clear()
                if rvv10_execution:
                    isa_set.append("rvv1.0")
                if rvv07_execution:
                    isa_set.append("rvv0.7")
                isa_set.append("riscvscalar")
                
            return isa_set, int(l1_size), int(l2_size), int(l3_size), int(VLEN), int(LMUL)
        else:
            if VLEN != 1:
                if (verbose > 0):
                    print("WARNING: --vector_length (-vl) argument is only used for RVV benchmarks.")
                VLEN = 1
            if LMUL != 1:
                if (verbose > 0):
                    print("WARNING: --vector_lmul (-vlmul) argument is only used for RVV benchmarks.")
                LMUL = 1
            isa_set.clear()
            isa_set.append("riscvscalar")
            return isa_set, int(l1_size), int(l2_size), int(l3_size), int(VLEN), int(LMUL)
    else:
        print("ERROR: Unsupported architecture " + CPU_Type + " detected. Exiting Program.")
        sys.exit(1)
    
#Call system probing and frequency setting code (x86 Only)
def autoconf(new_max_freq, set_freq):

    config_dir = os.path.join(script_dir, 'config', 'auto_config')
    c_file = os.path.join(config_dir, 'autoconfig.c')
    output_bin = os.path.join(config_dir, 'autoconfig')

    subprocess.run(["gcc", "-o", output_bin, c_file])

    if (set_freq == 0):
        new_max_freq = 0
        
    result = subprocess.run([output_bin, str(new_max_freq), str(set_freq)], stdout=subprocess.PIPE)
    arguments = result.stdout.decode('utf-8').split('\n')

    os.remove(output_bin)

    return arguments

#Read system configuration file
def read_config(config_file):
    f = open(config_file, "r")
    name = ''
    l1_size = "0"
    l2_size = "0"
    l3_size = "0"

    for line in f:
        l = line.split('=')
        if(l[0] == 'name'):
            name = l[1].rstrip()
        if(l[0] == 'l1_cache'):
            l1_size = l[1].rstrip()

        if(l[0] == 'l2_cache'):
            l2_size = l[1].rstrip()
        
        if(l[0] == 'l3_cache'):
            l3_size = l[1].rstrip()

    return name, int(l1_size), int(l2_size), int(l3_size)

def plot_roofline(name, data, date, isa, precision, threads, num_ld, num_st, inst, interleaved):
    if plot_numpy == None:
        print("No Matplotlib and/or Numpy found, in order to draw CARM graphs make sure to install them.")
        return
    fig, ax = plt.subplots(figsize=(7.1875*1.5,3.75*1.5))
    plt.xlim(0.015625, 256)
    if inst == "fma":
        plt.ylim(0.25, ut.round_power_of_2(int(data['FP'])))
    else:
        plt.ylim(0.25, ut.round_power_of_2(int(data['FP_FMA'])))
    ai = np.linspace(0.00390625, 256, num=200000)

    #Ploting Lines
    if inst == "fma":
        plt.plot(ai, ut.carm_eq(ai, data['L1'], data['FP']), 'k', lw = 3, label='L1')
        plt.plot(ai, ut.carm_eq(ai, data['L2'], data['FP']), 'grey', lw = 3, label='L2')
        plt.plot(ai, ut.carm_eq(ai, data['L3'], data['FP']), 'k', linestyle='dashed', lw = 3, label='L3')
        plt.plot(ai, ut.carm_eq(ai, data['DRAM'], data['FP']), 'k', linestyle='dotted', lw = 3, label='DRAM')
    else:
        plt.plot(ai, ut.carm_eq(ai, data['L1'], data['FP_FMA']), 'k', lw = 3, label='L1')
        plt.plot(ai, ut.carm_eq(ai, data['L2'], data['FP_FMA']), 'grey', lw = 3, label='L2')
        plt.plot(ai, ut.carm_eq(ai, data['L3'], data['FP_FMA']), 'k', linestyle='dashed', lw = 3, label='L3')
        plt.plot(ai, ut.carm_eq(ai, data['DRAM'], data['FP_FMA']), 'k', linestyle='dotted', lw = 3, label='DRAM')
        plt.plot(ai, ut.carm_eq(ai, data['L1'], data['FP']), 'k', linestyle='dashdot', lw = 3, label=inst)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if(interleaved):
        plt.title(name + ' CARM: ' + str(isa) + " " + str(precision) + " " + str(threads) + " Threads " + str(num_ld) + " Load " + str(num_st) + " Store " + inst + " Interleaved", fontsize=18)
    else:
        plt.title(name + ' CARM: ' + str(isa) + " " + str(precision) + " " + str(threads) + " Threads " + str(num_ld) + " Load " + str(num_st) + " Store " + inst, fontsize=18)
    plt.ylabel('Performance [GFLOP/s]', fontsize=18)
    plt.xlabel('Arithmetic Intensity [flops/bytes]', fontsize=18)
    plt.setp(ax.get_xticklabels(), fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)
    plt.yscale('log', base=2)
    plt.xscale('log', base=2)
    plt.legend(fontsize=18, loc='lower right')
    new_rc_params = {'text.usetex': False,"svg.fonttype": 'none'}
    plt.rcParams.update(new_rc_params)
    plt.tight_layout()
    if(interleaved):
        plt.savefig('carm_results/Roofline/' + name + '_roofline_' + str(date) + '_' + isa + "_" + str(precision) + "_" + str(threads) + "_Threads_" + str(num_ld) + "Load_" + str(num_st) + "Store_" + inst + "_Interleaved"'.svg')
    else:
        plt.savefig('carm_results/Roofline/' + name + '_roofline_' + str(date) + '_' + isa + "_" + str(precision) + "_" + str(threads) + "_Threads_" + str(num_ld) + "Load_" + str(num_st) + "Store_" + inst +'.svg')
    
def update_csv(name, test_type, data, data_cycles, date, isa, precision, threads, num_ld, num_st, inst, interleaved, l1_size, l2_size, l3_size, dram_bytes, VLEN, LMUL, out_path):

    csv_path = f"{out_path}/roofline/{name}_{test_type.replace(' ', '_')}.csv"
    if (isa in ["rvv0.7", "rvv1.0"]):
        isa = str(isa) + "_vl" + str(VLEN) + "_lmul" + str(LMUL)
    elif isa in ["sve"]:
        isa = str(isa) + "_vl" + str(VLEN)
    results = [
        date,
        isa,
        precision,
        threads,
        num_ld,
        num_st,
        "Yes" if interleaved else "No",
        dram_bytes,
        inst,
    ]
    if l1_size > 0:
        results.append(ut.custom_round(data["L1"]))
        results.append(ut.custom_round(data_cycles["L1"]))
    else:
        results.append(0)
        results.append(0)
    if l2_size > 0:
        results.append(ut.custom_round(data["L2"]))
        results.append(ut.custom_round(data_cycles["L2"]))
    else:
        results.append(0)
        results.append(0)
    if l3_size > 0:
        results.append(ut.custom_round(data["L3"]))
        results.append(ut.custom_round(data_cycles["L3"]))
    else:
        results.append(0)
        results.append(0)

    results.append(ut.custom_round(data["DRAM"]))
    results.append(ut.custom_round(data_cycles["DRAM"]))
    results.append(ut.custom_round(data["FP"]))
    results.append(ut.custom_round(data_cycles["FP"]))


    if not inst == "fma":
        results.append(ut.custom_round(data["FP_FMA"]))
        results.append(ut.custom_round(data_cycles["FP_FMA"]))
    else:
        results.append(0)
        results.append(0)

    secondary_headers = ['Name:', name, 'L1 Size:', l1_size, 'L2 Size:', l2_size, 'L3 Size:', l3_size, '', 'L1', 'L1', 'L2', 'L2', 'L3', 'L3', 'DRAM', 'DRAM', 'FP', 'FP', 'FP FMA', 'FP_FMA']
    primary_headers = ['Date', 'ISA', 'Precision', 'Threads', 'Loads', 'Stores', 'Interleaved', 'DRAM Bytes', 'FP Inst.', 'GB/s', 'I/Cycle', 'GB/s', 'I/Cycle', 'GB/s', 'I/Cycle', 'GB/s', 'I/Cycle', 'Gflop/s', 'I/Cycle', 'Gflop/s', 'I/Cycle']

    #Check if the file exists
    if os.path.exists(csv_path):
        #If exists, append without header
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(results)
    else:
        #If not, write with header and include secondary headers
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(secondary_headers)
            writer.writerow(primary_headers)
            writer.writerow(results)

def print_start(test_type, threads_set, isa_set, precision_set, VLEN, LMUL):
    print("-------------------------------------------------")
    print(f"Running {test_type[0].upper()}{test_type[1:]} Benchmarks for the Following Thread Counts: {threads_set}")
    print("On the Following ISA extensions: ", isa_set)
    print("Using the Following Precisions:", precision_set)
    if "rvv0.7" in isa_set or "rvv1.0" in isa_set or "sve" in isa_set:
        if "dp" in precision_set:
            print("Vector with " + str(VLEN) + " double precision elements will be used for vector benchmark.")
        if "sp" in precision_set:
            print("Vector with " + str(VLEN*2) + " single precision elements will be used for vector benchmark.")
    if "rvv0.7" in isa_set or "rvv1.0" in isa_set:
        print(f"Register grouping (-vlmul) of {LMUL} will be used for RVV benchmark.")
    print("-------------------------------------------------")

def print_mid(verbose, test_type, threads_set, isa_set, precision_set, threads, isa, precision, VLEN, LMUL):
    if verbose == 4:
        print("-------------------------------------------------")
        print(f"Running {test_type[0].upper()}{test_type[1:]} Benchmarks for the Following Thread Counts: {threads_set}")
        print("On the Following ISA extensions: ", isa_set)
        print("Using the Following Precisions:", precision_set)

    if verbose > 0:
        print(f"Now Testing {test_type[0].upper()}{test_type[1:]} with: {threads} {'Thread' if threads == 1 else 'Threads'} using {isa} with {precision}")
        if isa in vector_agnostic_ISA:
            if precision == "dp":
                print("Vector with " + str(VLEN) + " double precision elements will be used.")
            if precision == "sp":
                print("Vector with " + str(VLEN) + " single precision elements will be used.")
        if isa in ["rvv0.7", "rvv1.0"]:
            print(f"Register grouping (-vlmul) of {LMUL} will be used.")

def print_results(isa, test_type, test_data, data_cycles, num_reps, test_size, inner_loop_reps, freq_real, cycles, freq_nominal, time_ms, threads, num_ld, num_st, inst, FP_factor, precision, VLEN, interleaved, LMUL, verbose):
    if interleaved:
        inter = "Yes"
    else:
        inter = "No"

    print(f"------------------{test_type} RESULTS---------------------")
    if test_type not in ["FP", "FP_FMA"]:
        print("ISA:", isa,  "| Number of Threads:", threads, "| Allocated Size:", test_size, "Kb | Precision:", precision, "| Interleaved:", inter, "| Number of Loads:", num_ld, "| Number of Stores:", num_st, "| Memory Instruction Size:", mem_inst_size[isa][precision]*VLEN)
    else:
        print("ISA:", isa,  "| Number of Threads:", threads, "| Instruction:", inst, "| Precision:", precision, "| Interleaved:", inter, "| FP Operations per Instruction:", (FP_factor*ops_fp[isa][precision])*VLEN)
    if isa in x86_ISAs:
        print("Best Average Cycles:", int(cycles), "| Best Average Time (in ms):", ut.custom_round(time_ms))
    else:
        print("Best Average Time (in ms):", ut.custom_round(time_ms))
    
    print("Instructions per Cycle:", ut.custom_round(data_cycles))

    if test_type not in ["FP", "FP_FMA"]:
        print("Bytes per Cycle:", ut.custom_round(data_cycles*mem_inst_size[isa][precision]*VLEN*LMUL))
        print("Bandwidth (GB/s):", ut.custom_round(test_data))
    else:
        print("Flops per Cycle:", ut.custom_round(data_cycles*ops_fp[isa][precision]*FP_factor*VLEN*LMUL))
        print("GFLOP/s:", ut.custom_round(test_data))
    if isa in x86_ISAs:
        print("Max Recorded Frequency (GHz):", ut.custom_round(freq_real), "| Nominal Frequency (GHz):", ut.custom_round(freq_nominal), "| Actual Frequency to Nominal Frequency Ratio:", ut.custom_round(float(freq_real/freq_nominal)))
    else:
        print("Max Recorded Frequency (GHz):", ut.custom_round(freq_real))
    if isa in ["rvv0.7", "rvv1.0"]:
        print("Vector Length:", VLEN, "Elements | Vector LMUL:", LMUL)
    elif isa in ["sve"]:
        print("Vector Length:", VLEN)
    if verbose == 4:
        print("Results Debug -> Total Inner Loop Reps:", int(inner_loop_reps), "| Number of reps:", num_reps)

#Run Roofline tests
def run_roofline(name, freq, l1_size, l2_size, l3_size, inst, isa_set, precision_set, num_ld, num_st, threads_set, interleaved, num_ops, l3_bytes, dram_bytes, dram_auto, test_type, verbose, set_freq, no_freq_measure, VLEN, tl1, tl2, plot, LMUL, out_path, num_runs):
    
    num_reps = {}
    test_size = {}
    data = {}
    data_cycles = {}
    FP_factor = 1
    time_ms = 0
    cycles = 0
    freq_nominal = freq
    freq_real = freq

    #CPU Type Verification (x86 / ARM / RISCV)
    isa_set, l1_size, l2_size, l3_size, VLEN, LMUL = check_hardware(isa_set, freq, set_freq, verbose, precision_set, l1_size, l2_size, l3_size, VLEN, LMUL)
    VLEN_aux = VLEN
    LMUL_aux = LMUL
    dram_bytes_aux = dram_bytes
    no_freq_measure_aux = no_freq_measure

    if inst == "fma":
        FP_factor = 2

    if 0 < verbose < 4:
        print_start(test_type, threads_set, isa_set, precision_set, VLEN, LMUL)

    for threads in threads_set:
        for isa in isa_set:
            for precision in precision_set:
                dram_bytes = dram_bytes_aux
                if inst == "fma":
                    FP_factor = 2
                else:
                    FP_factor = 1
                no_freq_measure = no_freq_measure_aux
                if VLEN_aux > 1 and precision == "dp":
                    VLEN = VLEN_aux
                    LMUL = LMUL_aux
                if VLEN_aux > 1 and precision == "sp":
                    VLEN = VLEN_aux * 2
                    LMUL = LMUL_aux
                if isa not in vector_agnostic_ISA:
                    VLEN = 1
                    LMUL = 1
                if isa in ["sve"]:
                    LMUL = 1

                print_mid(verbose, test_type, threads_set, isa_set, precision_set, threads, isa, precision, VLEN, LMUL)

                #Calculate number of repetitions for each test
                #L1 Reps
                if (l1_size > 0) and test_type in ["L1", "roofline"]:
                    num_reps["L1"] = int(int(l1_size)*1024/(tl1*2*mem_inst_size[isa][precision]*(num_ld+num_st)*VLEN*LMUL))
                    test_size["L1"] = (int(l1_size))/(tl1*2)
                elif test_type in ["L1", "roofline"]:
                    print("ERROR: No L1 Size Found, you can use the -l1 <l1_size> argument, or a configuration file to specify it.")
                    return
                #L2 Reps
                if (l2_size > 0) and test_type in ["L2", "roofline"]:
                    num_reps["L2"] = int(int(1024*int(l2_size)/tl2)/(mem_inst_size[isa][precision]*(num_ld+num_st)*VLEN*LMUL))
                    test_size["L2"] = int(int(l2_size)/tl2)
                elif test_type  == "roofline"  and verbose > 0:
                    print("WARNING: No L2 Size Found, you can use the -l2 <l2_size> argument, or a configuration file to specify it.")
                elif test_type  == "L2":
                    print("ERROR: No L2 Size Found, you can use the -l2 <l2_size> argument, or a configuration file to specify it.")
                    return
                #L3 Reps
                if l3_size == 0 and test_type == "L3":
                    print("ERROR: No L3 Size Found, you can use the -l3 <l3_size> argument, or a configuration file to specify it.")
                    return
                if l3_bytes > 0 and test_type in ["L3", "roofline"]:
                    num_reps["L3"] = l3_bytes*1024/(threads*mem_inst_size[isa][precision]*(num_ld+num_st)*VLEN*LMUL)
                    test_size["L3"] = int(l3_bytes/(threads))
                elif l3_size > 0 and (int(l2_size)*threads + (int(l3_size) - int(l2_size)*threads)/2)/threads > l2_size and test_type in ["L3", "roofline"]:
                    num_reps["L3"] = int(1024*(int(l2_size)*threads + (int(l3_size) - int(l2_size)*threads)/2)/(threads*mem_inst_size[isa][precision]*(num_ld+num_st)*VLEN*LMUL))
                    test_size["L3"] = int((int(l2_size)*threads + (int(l3_size) - int(l2_size)*threads)/2)/threads)
                elif test_type in ["L3", "roofline"]:
                    num_reps["L3"] = int(1024*(int(l2_size*1.2))/(mem_inst_size[isa][precision]*(num_ld+num_st)*VLEN*LMUL))
                    test_size["L3"] = int(l2_size*1.2)
                    if verbose > 0 and l3_size > 0:
                        print("WARNING: L3 size insuficient to give each thread a memory slice significantly larger than L2 without exceeding L3 size.\n"
                        "For a more detailed memory analysis run '--test MEM' to identify the most appropriate size for the L3 test.\n"
                        "Then use the argument '--l3_kbytes <test_size>' to enforce a custom test size")
                if l3_size == 0 and test_type == "roofline" and verbose > 0:
                    print("WARNING: No L3 Size Found, you can use the -l3 <l3_size> argument, or a configuration file to specify it.")
                if l3_size > 0 and test_type in ["L3", "roofline"] and verbose > 3:
                    print(f"Total L3 Test Size: {test_size['L3']*threads}Kb | L3 Size: {l3_size}kb | L3 Test Size per Thread: {test_size['L3']}Kb | L2 Size: {l2_size}Kb")
                #DRAM Reps
                if (dram_auto and l3_size > 0 and dram_bytes/(threads) < (l3_size*2) and test_type in ["DRAM", "roofline"]):
                    num_reps["DRAM"] = int((int(l3_size)*2)*1024/(mem_inst_size[isa][precision]*(num_ld+num_st)*VLEN*LMUL))
                    test_size["DRAM"] = int((int(l3_size)*2))
                    dram_bytes = int(l3_size)*2*threads
                elif test_type in ["DRAM", "roofline"]:
                    if (dram_bytes > 0):
                        num_reps["DRAM"] = int(dram_bytes*1024/(threads*mem_inst_size[isa][precision]*(num_ld+num_st)*VLEN*LMUL))
                        test_size["DRAM"] = int(dram_bytes/(threads))
                        if int(test_size["DRAM"]) <= int(l3_size) and verbose > 0:
                            print("WARNING: DRAM test size per thread is not sufficient to guarantee best results, to guarantee best results consider changing the default test size.")
                            print("By using --dram_bytes", int(l3_size)*2*int(threads), "(", ut.custom_round(float((int(l3_size)*2*int(threads))/1048576)), "Gb) the minimum test size necessary for", threads, "threads is achieved, using the --dram_auto flag will automatically apply this adjustement.")
                if verbose > 3 and test_type in ["DRAM", "roofline"]:
                    print("DRAM Test Size per Thread:", test_size["DRAM"], "Kb | Total DRAM Test Size:", ut.custom_round(float((test_size["DRAM"]*threads)/1048576)), "Gb")

                num_reps["FP"] = int(num_ops/(FP_factor*ops_fp[isa][precision]*LMUL*VLEN))
                if inst != "fma":
                    num_reps["FP_FMA"] = int(num_ops/(2*ops_fp[isa][precision]*LMUL*VLEN))

                if verbose == 4:
                    print("-------------------------------------------------")
                    print("Debug output:")
                    make_verb_flag = ""
                else:
                    make_verb_flag = "-s"
        
                os.system(f"make {make_verb_flag} -C {script_dir} clean && make {make_verb_flag} -C {script_dir} isa={isa}")

                if test_type in ["L1", "roofline"] and l1_size > 0:
                    #Run L1 Test

                    os.system(str(bench_ex) + " -test MEM -num_LD " + str(num_ld) + " -num_ST " + str(num_st) + " -precision " + precision + " -num_rep " + str(num_reps["L1"]) + " -Vlen " + str(VLEN) + " -LMUL " + str(LMUL) + " -verbose " + str(verbose) + " -num_runs " + str(num_runs))
                    
                    if(interleaved):
                        result = subprocess.run([test_ex, "-threads", str(threads), "-freq", str(freq), "-measure_freq", str(no_freq_measure), "--interleaved"], stdout=subprocess.PIPE)
                    else:
                        result = subprocess.run([test_ex, "-threads", str(threads), "-freq", str(freq), "-measure_freq", str(no_freq_measure)], stdout=subprocess.PIPE)
                
                    out = result.stdout.decode('utf-8').split(',')
                    
                    inner_loop_reps = float(out[1])
                    freq_real = float(out[2])
                    if isa in x86_ISAs:
                        cycles = float(out[0])
                        freq_nominal = float(out[3])
                        data['L1'] = float((threads*num_reps["L1"]*(num_ld+num_st)*mem_inst_size[isa][precision]*freq_real*inner_loop_reps)/(cycles*float(freq_real/freq_nominal)))
                        data_cycles['L1'] = float((threads*num_reps["L1"]*(num_ld+num_st)*inner_loop_reps)/(cycles*float(freq_real/freq_nominal)))
                        time_ms = float (cycles / (freq_nominal * 1e6))
                    else:
                        time_ms = float(out[0])
                        data['L1'] = (float((threads*num_reps["L1"]*(num_ld+num_st)*mem_inst_size[isa][precision]*VLEN*LMUL*inner_loop_reps)/(1000000000))/((time_ms/1000)))
                        data_cycles['L1'] = float((threads*num_reps["L1"]*(num_ld+num_st)*inner_loop_reps)/((time_ms/1000)*freq_real*1000000000))
                    
                    if (verbose > 1):
                        print_results(isa, "L1", data["L1"], data_cycles["L1"], num_reps["L1"], test_size["L1"], inner_loop_reps, freq_real, cycles, freq_nominal, time_ms, threads, num_ld, num_st, inst, FP_factor, precision, VLEN, interleaved, LMUL, verbose)
                        if test_type != 'roofline' and verbose < 4:
                            print("-------------------------------------------------")
                if test_type in ["L2", "roofline"] and l2_size > 0:
                    #Run L2 Test
                    if verbose == 4 and test_type == 'roofline':
                        print("-------------------------------------------------")
                        print("Debug output:")
                    os.system(str(bench_ex) + " -test MEM -num_LD " + str(num_ld) + " -num_ST " + str(num_st) + " -precision " + precision + " -num_rep " + str(num_reps["L2"]) + " -Vlen " + str(VLEN) + " -LMUL " + str(LMUL) + " -verbose " + str(verbose) + " -num_runs " + str(num_runs))
                    
                    if test_type == 'roofline' and l1_size > 0:
                        no_freq_measure = 1
                    
                    if(interleaved):
                        result = subprocess.run([test_ex, "-threads", str(threads), "-freq", str(freq), "-measure_freq", str(no_freq_measure), "--interleaved"], stdout=subprocess.PIPE)
                    else:
                        result = subprocess.run([test_ex, "-threads", str(threads), "-freq", str(freq), "-measure_freq", str(no_freq_measure)], stdout=subprocess.PIPE)
                    
                    out = result.stdout.decode('utf-8').split(',')
                    inner_loop_reps = float(out[1])
                    if no_freq_measure == 0:
                        freq_real = float(out[2])
                    if isa in x86_ISAs:
                        cycles = float(out[0])
                        if no_freq_measure == 0:
                            freq_nominal = float(out[3])
                        data['L2'] = float((threads*num_reps["L2"]*(num_ld+num_st)*mem_inst_size[isa][precision]*freq_real*inner_loop_reps)/(cycles*float(freq_real/freq_nominal)))
                        data_cycles['L2'] = float((threads*num_reps["L2"]*(num_ld+num_st)*inner_loop_reps)/(cycles*float(freq_real/freq_nominal)))
                        time_ms = float (cycles / (freq_nominal * 1e6))
                    else:
                        time_ms = float(out[0])
                        data['L2'] = (float((threads*num_reps["L2"]*(num_ld+num_st)*mem_inst_size[isa][precision]*VLEN*LMUL*inner_loop_reps)/(1000000000))/((time_ms/1000)))
                        data_cycles['L2'] = float((threads*num_reps["L2"]*(num_ld+num_st)*inner_loop_reps)/((time_ms/1000)*freq_real*1000000000))
                    if (verbose > 1):
                        print_results(isa, "L2", data["L2"], data_cycles["L2"], num_reps["L2"], test_size["L2"], inner_loop_reps, freq_real, cycles, freq_nominal, time_ms, threads, num_ld, num_st, inst, FP_factor, precision, VLEN, interleaved, LMUL, verbose)
                        if test_type != 'roofline' and verbose < 4:
                            print("-------------------------------------------------")
                if (test_type in ["L3", "roofline"] and int(l3_size) > 0):
                    #Run L3 Test 
                    if verbose == 4 and test_type == 'roofline':
                        print("-------------------------------------------------")
                        print("Debug output:")
                    os.system(str(bench_ex) + " -test MEM -num_LD " + str(num_ld) + " -num_ST " + str(num_st) + " -precision " + precision + " -num_rep " + str(num_reps["L3"]) + " -Vlen " + str(VLEN) + " -LMUL " + str(LMUL) + " -verbose " + str(verbose) + " -num_runs " + str(num_runs))
                    
                    if test_type == 'roofline' and (l1_size > 0 or l2_size > 0):
                        no_freq_measure = 1

                    if(interleaved):
                        result = subprocess.run([test_ex, "-threads", str(threads), "-freq", str(freq), "-measure_freq", str(no_freq_measure), "--interleaved"], stdout=subprocess.PIPE)
                    else:
                        result = subprocess.run([test_ex, "-threads", str(threads), "-freq", str(freq), "-measure_freq", str(no_freq_measure)], stdout=subprocess.PIPE)
                
                    out = result.stdout.decode('utf-8').split(',')
                    inner_loop_reps = float(out[1])
                    if no_freq_measure == 0:
                        freq_real = float(out[2])
                    if isa in x86_ISAs:
                        cycles = float(out[0])
                        if no_freq_measure == 0:
                            freq_nominal = float(out[3])
                        data['L3'] = float((threads*num_reps["L3"]*(num_ld+num_st)*mem_inst_size[isa][precision]*freq_real*inner_loop_reps)/(cycles*float(freq_real/freq_nominal)))
                        data_cycles['L3'] = float((threads*num_reps["L3"]*(num_ld+num_st)*inner_loop_reps)/(cycles*float(freq_real/freq_nominal)))
                        time_ms = float (cycles / (freq_nominal * 1e6))
                    else:
                        time_ms = float(out[0])
                        data['L3'] = (float((threads*num_reps["L3"]*(num_ld+num_st)*mem_inst_size[isa][precision]*VLEN*LMUL*inner_loop_reps)/(1000000000))/((time_ms/1000)))
                        data_cycles['L3'] = float((threads*num_reps["L3"]*(num_ld+num_st)*inner_loop_reps)/((time_ms/1000)*freq_real*1000000000))
                    if (verbose > 1):
                        print_results(isa, "L3", data["L3"], data_cycles["L3"], num_reps["L3"], test_size["L3"], inner_loop_reps, freq_real, cycles, freq_nominal, time_ms, threads, num_ld, num_st, inst, FP_factor, precision, VLEN, interleaved, LMUL, verbose)
                        if test_type != 'roofline' and verbose < 4:
                            print("-------------------------------------------------")
                if (test_type in ["DRAM", "roofline"]):
                    #Run DRAM Test
                    if verbose == 4 and test_type == 'roofline':
                        print("-------------------------------------------------")
                        print("Debug output:")
                    os.system(str(bench_ex) + " -test MEM -num_LD " + str(num_ld) + " -num_ST " + str(num_st) + " -precision " + precision + " -num_rep " + str(num_reps["DRAM"]) + " -Vlen " + str(VLEN) + " -LMUL " + str(LMUL) + " -verbose " + str(verbose) + " -num_runs " + str(num_runs))
                    
                    if test_type == 'roofline' and (l1_size > 0 or l2_size > 0 or l3_size > 0):
                        no_freq_measure = 1

                    if(interleaved):
                        result = subprocess.run([test_ex, "-threads", str(threads), "-freq", str(freq), "-measure_freq", str(no_freq_measure), "--interleaved"], stdout=subprocess.PIPE)
                    else:
                        result = subprocess.run([test_ex, "-threads", str(threads), "-freq", str(freq), "-measure_freq", str(no_freq_measure)], stdout=subprocess.PIPE)
                    
                    out = result.stdout.decode('utf-8').split(',')
                    inner_loop_reps = float(out[1])
                    if no_freq_measure == 0:
                        freq_real = float(out[2])
                    if isa in x86_ISAs:
                        cycles = float(out[0])
                        if no_freq_measure == 0:
                            freq_nominal = float(out[3])
                        data['DRAM'] = float((threads*num_reps["DRAM"]*(num_ld+num_st)*mem_inst_size[isa][precision]*freq_real*inner_loop_reps)/(cycles*float(freq_real/freq_nominal)))
                        data_cycles['DRAM'] = float((threads*num_reps["DRAM"]*(num_ld+num_st)*inner_loop_reps)/(cycles*float(freq_real/freq_nominal)))
                        time_ms = float (cycles / (freq_nominal * 1e6))
                    else:
                        time_ms = float(out[0])
                        data['DRAM'] = (float((threads*num_reps["DRAM"]*(num_ld+num_st)*mem_inst_size[isa][precision]*VLEN*LMUL*inner_loop_reps)/(1000000000))/((time_ms/1000)))
                        data_cycles['DRAM'] = float((threads*num_reps["DRAM"]*(num_ld+num_st)*inner_loop_reps)/((time_ms/1000)*freq_real*1000000000))
                    if (verbose > 1):
                        print_results(isa, "DRAM", data["DRAM"], data_cycles["DRAM"], num_reps["DRAM"], test_size["DRAM"], inner_loop_reps, freq_real, cycles, freq_nominal, time_ms, threads, num_ld, num_st, inst, FP_factor, precision, VLEN, interleaved, LMUL, verbose)
                        if test_type != 'roofline' and verbose < 4:
                            print("-------------------------------------------------")
                if test_type in ["FP", "roofline"]:
                    #Run FP Test
                    if verbose == 4 and test_type == 'roofline':
                        print("-------------------------------------------------")
                        print("Debug output:")
                    os.system(str(bench_ex) + " -test FLOPS -op " + inst + " -precision " + precision + " -fp " + str(num_reps["FP"]) + " -Vlen " + str(VLEN) + " -LMUL " + str(LMUL) + " -verbose " + str(verbose) + " -num_runs " + str(num_runs))
                    
                    if test_type == 'roofline' and (l1_size > 0 or l2_size > 0 or l3_size > 0 or dram_bytes > 0):
                        no_freq_measure = 1

                    if(interleaved):
                        result = subprocess.run([test_ex, "-threads", str(threads), "-freq", str(freq), "-measure_freq", str(no_freq_measure), "--interleaved"], stdout=subprocess.PIPE)
                    else:
                        result = subprocess.run([test_ex, "-threads", str(threads), "-freq", str(freq), "-measure_freq", str(no_freq_measure)], stdout=subprocess.PIPE)
                    
                    out = result.stdout.decode('utf-8').split(',')
                    inner_loop_reps = float(out[1])
                    if no_freq_measure == 0:
                        freq_real = float(out[2])
                    if isa in x86_ISAs:
                        cycles = float(out[0])
                        if no_freq_measure == 0:
                            freq_nominal = float(out[3])
                        data['FP'] = float(threads*num_reps["FP"]*FP_factor*ops_fp[isa][precision]*freq_real*inner_loop_reps)/(cycles*float(freq_real/freq_nominal))
                        data_cycles['FP'] = float((threads*num_reps["FP"]*inner_loop_reps)/(cycles*float(freq_real/freq_nominal)))
                        time_ms = float (cycles / (freq_nominal * 1e6))
                        
                    else:
                        time_ms = float(out[0])
                        data['FP'] = float((threads*num_reps["FP"]*FP_factor*ops_fp[isa][precision]*inner_loop_reps*VLEN*LMUL)/(1000000000))/((time_ms/1000))
                        data_cycles['FP'] = float((threads*num_reps["FP"]*inner_loop_reps)/((time_ms/1000)*freq_real*1000000000))
                    if (verbose > 1):
                        print_results(isa, "FP", data["FP"], data_cycles["FP"], num_reps["FP"], 0, inner_loop_reps, freq_real, cycles, freq_nominal, time_ms, threads, num_ld, num_st, inst, FP_factor, precision, VLEN, interleaved, LMUL, verbose)
                        if test_type != 'roofline' and verbose < 4:
                            print("-------------------------------------------------")
                
                if (test_type == 'roofline' and inst != 'fma'):
                    #Run FP FMA Test

                    inst_fma = 'fma'
                    FP_factor = 2
                    if verbose == 4:
                        print("-------------------------------------------------")
                        print("Debug output:")
                    os.system(str(bench_ex) + " -test FLOPS -op " + inst_fma + " -precision " + precision + " -fp " + str(num_reps["FP_FMA"]) + " -Vlen " + str(VLEN) + " -LMUL " + str(LMUL) + " -verbose " + str(verbose) + " -num_runs " + str(num_runs))
                    
                    if test_type == 'roofline' and (l1_size > 0 or l2_size > 0 or l3_size > 0 or dram_bytes > 0):
                        no_freq_measure = 1

                    if(interleaved):
                        result = subprocess.run([test_ex, "-threads", str(threads), "-freq", str(freq), "-measure_freq", str(no_freq_measure), "--interleaved"], stdout=subprocess.PIPE)
                    else:
                        result = subprocess.run([test_ex, "-threads", str(threads), "-freq", str(freq), "-measure_freq", str(no_freq_measure)], stdout=subprocess.PIPE)
                    
                    out = result.stdout.decode('utf-8').split(',')

                    inner_loop_reps = float(out[1])
                    if no_freq_measure == 0:
                        freq_real = float(out[2])
                    if isa in x86_ISAs:
                        cycles = float(out[0])
                        if no_freq_measure == 0:
                            freq_nominal = float(out[3])
                        data['FP_FMA'] = float(threads*num_reps["FP_FMA"]*FP_factor*ops_fp[isa][precision]*freq_real*inner_loop_reps)/(cycles*float(freq_real/freq_nominal))
                        data_cycles['FP_FMA'] = float((threads*num_reps["FP_FMA"]*inner_loop_reps)/(cycles*float(freq_real/freq_nominal)))
                        time_ms = float (cycles / (freq_nominal * 1e6))
                    else:
                        time_ms = float(out[0])
                        data['FP_FMA'] = float((threads*num_reps["FP_FMA"]*FP_factor*ops_fp[isa][precision]*inner_loop_reps*VLEN*LMUL)/(1000000000))/((time_ms/1000))
                        data_cycles['FP_FMA'] = float((threads*num_reps["FP_FMA"]*inner_loop_reps)/((time_ms/1000)*freq_real*1000000000))
                    if (verbose > 1):
                        print_results(isa, "FP_FMA", data["FP_FMA"], data_cycles["FP_FMA"], num_reps["FP_FMA"], 0, inner_loop_reps, freq_real, cycles, freq_nominal, time_ms, threads, num_ld, num_st, inst_fma, FP_factor, precision, VLEN, interleaved, LMUL, verbose)
                        if verbose < 4:
                            print("-------------------------------------------------")
                #Save Results if a full Roofline test is done
                if (test_type == 'roofline'):
                    if (out_path == './carm_results'):
                        if(os.path.isdir('carm_results') == False):
                            os.mkdir('carm_results')
                        if(os.path.isdir('carm_results/roofline') == False):
                            os.mkdir('carm_results/roofline')
                    else:
                        if(os.path.isdir(out_path)):
                            if(os.path.isdir(out_path + "/roofline") == False):
                                os.mkdir(out_path + "/roofline")
                        else:
                            print("ERROR: Provided output path does not exist")
                    ct = datetime.datetime.now()
                    if plot:
                        date = ct.strftime('%Y-%m-%d_%H-%M-%S')
                        plot_roofline(name, data, date, isa, precision, threads, num_ld, num_st, inst, interleaved, out_path)
                    
                    date = ct.strftime('%Y-%m-%d %H:%M:%S')
                    update_csv(name, "roofline", data, data_cycles, date, isa, precision, threads, num_ld, num_st, inst, interleaved, l1_size, l2_size, l3_size, dram_bytes, VLEN, LMUL, out_path)

#Run Memory Bandwidth tests
def run_memory(name, freq, set_freq, l1_size, l2_size, l3_size, isa_set, precision_set, num_ld, num_st, threads_set, interleaved, verbose, no_freq_measure, VLEN, tl1, plot, LMUL, num_runs):
    if interleaved:
        inter = "Yes"
    else:
        inter = "No"

    isa_set, l1_size, l2_size, l3_size, VLEN, LMUL = check_hardware(isa_set, freq, set_freq, verbose, precision_set, l1_size, l2_size, l3_size, VLEN, LMUL)
    
    VLEN_aux = VLEN
    LMUL_aux = LMUL
    freq_nominal = freq
    freq_real = freq

    if 0 < verbose < 4:
        print_start("Memory", threads_set, isa_set, precision_set, VLEN, LMUL)

    for threads in threads_set:
        for isa in isa_set:
            for precision in precision_set:
                if VLEN_aux > 1 and precision == "dp":
                    VLEN = VLEN_aux
                    LMUL = LMUL_aux
                if VLEN_aux > 1 and precision == "sp":
                    VLEN = VLEN_aux * 2
                    LMUL = LMUL_aux
                if isa not in vector_agnostic_ISA:
                    VLEN = 1
                    LMUL = 1
                if isa in ["sve"]:
                    LMUL = 1

                print_mid(verbose, "Memory", threads_set, isa_set, precision_set, threads, isa, precision, VLEN, LMUL)
                    
                Gbps = [0] * len(test_sizes)
                InstCycle = [0] * len(test_sizes)

                if verbose > 3:
                    make_verb_flag = ""
                else:
                    make_verb_flag = "-s"

                if verbose == 4:
                    print("-------------------------------------------------")
                    print("Debug output:")
        
                os.system(f"make {make_verb_flag} -C {script_dir} clean && make {make_verb_flag} -C {script_dir} isa={isa}")

                num_reps = int(int(l1_size)*1024/(tl1*2*mem_inst_size[isa][precision]*(num_ld+num_st)*VLEN*LMUL))

                os.system(str(bench_ex) + " -test MEM -num_LD " + str(num_ld) + " -num_ST " + str(num_st) + " -precision " + precision + " -num_rep " + str(num_reps) + " -Vlen " + str(VLEN) + " -LMUL " + str(LMUL) + " -verbose " + str(verbose) + " -num_runs " + str(num_runs))
                
                no_freq_measure = 0

                if(interleaved):
                    result = subprocess.run([test_ex, "-threads", str(threads), "-freq", str(freq), "-measure_freq", str(no_freq_measure), "--interleaved"], stdout=subprocess.PIPE)
                else:
                    result = subprocess.run([test_ex, "-threads", str(threads), "-freq", str(freq), "-measure_freq", str(no_freq_measure)], stdout=subprocess.PIPE)

                no_freq_measure = 1
                out = result.stdout.decode('utf-8').split(',')

                freq_real = float(out[2])
                if isa in x86_ISAs:
                    freq_nominal = float(out[3])

                i=0
                for size in test_sizes:
                    if verbose == 4:
                        print("-------------------------------------------------")
                        print("Debug output:")
                    if verbose > 2:
                        print("Testing with", size, "Kb")
                    num_reps = int(size*1024/(mem_inst_size[isa][precision]*(num_ld+num_st)*VLEN*LMUL))

                    os.system(str(bench_ex) + " -test MEM -num_LD " + str(num_ld) + " -num_ST " + str(num_st) + " -precision " + precision + " -num_rep " + str(num_reps) + " -Vlen " + str(VLEN) + " -LMUL " + str(LMUL) + " -verbose " + str(verbose) + " -num_runs " + str(num_runs))
                
                    if(interleaved):
                        result = subprocess.run([test_ex, "-threads", str(threads), "-freq", str(freq_real), "-measure_freq", str(no_freq_measure), "--interleaved"], stdout=subprocess.PIPE)
                    else:
                        result = subprocess.run([test_ex, "-threads", str(threads), "-freq", str(freq_real), "-measure_freq", str(no_freq_measure)], stdout=subprocess.PIPE)

                    out = result.stdout.decode('utf-8').split(',')
                    inner_loop_reps = float(out[1])
                    
                    if isa in x86_ISAs:
                        cycles = float(out[0])
                        Gbps[i] = float((threads*num_reps*(num_ld+num_st)*mem_inst_size[isa][precision]*freq_real*inner_loop_reps)/(cycles*float(freq_real/freq_nominal)))
                        InstCycle[i] = float((threads*num_reps*(num_ld+num_st)*inner_loop_reps)/(cycles*float(freq_real/freq_nominal)))
                    else:
                        time_ms = float(out[0])
                        Gbps[i] = (float((threads*num_reps*(num_ld+num_st)*mem_inst_size[isa][precision]*VLEN*inner_loop_reps)/(1000000000))/((time_ms/1000)))
                        InstCycle[i] = float((threads*num_reps*(num_ld+num_st)*inner_loop_reps)/((time_ms/1000)*freq_real*1000000000))
                    i += 1
                i = 0
                if (verbose > 1):
                    print("--------------------RESULTS----------------------")
                    print("ISA:", isa,  "| Number of Threads:", threads, " | Precision:", precision, "| Interleaved:", inter, "| Number of Loads:", num_ld, "| Number of Stores:", num_st, "| Memory Instruction Size:", mem_inst_size[isa][precision]*VLEN)
                    if isa in x86_ISAs:
                        print("Max Recorded Frequency (GHz):", ut.custom_round(freq_real), "| Nominal Frequency (GHz):", ut.custom_round(freq_nominal), "| Actual Frequency to Nominal Frequency Ratio:", ut.custom_round(float(freq_real/freq_nominal)))
                    else:
                        print("Max Recorded Frequency (GHz):", ut.custom_round(freq_real))
                        if isa in ["rvv0.7", "rvv1.0"]:
                            print("Vector Length:", VLEN, "Elements | Vector LMUL:", LMUL)
                        elif isa in ["sve"]:
                            print("Vector Length:", VLEN, "Elements")
                    for size in test_sizes:
                        print("Size (per thread):", size, "Kb | Gbps:", ut.custom_round(Gbps[i]), "| Instructions per Cycle:", ut.custom_round(InstCycle[i]))
                        i += 1
                    if 1 < verbose < 4:
                        print("-------------------------------------------------")

                if(os.path.isdir('carm_results') == False):
                    os.mkdir('carm_results')
                if(os.path.isdir('carm_results/memory_curve') == False):
                    os.mkdir('carm_results/memory_curve')

                ct = datetime.datetime.now()
                date = ct.strftime('%Y-%m-%d %H:%M:%S')
                update_memory_csv(Gbps, InstCycle, date, name, l1_size, l2_size, l3_size, isa, precision, num_ld, num_st, threads, interleaved, VLEN, LMUL)
                
                if plot:
                    if plot_numpy == None:
                        print("No Matplotlib and/or Numpy found, in order to draw CARM graphs make sure to install them.")
                        return

                    #Find position of nearest value to l1_size in test_sizes array
                    if l1_size > 0:
                        nearest_value = min(test_sizes, key=lambda x: abs(x - int(l1_size)))
                        l1_position = test_sizes.index(nearest_value)
                    #Find position of nearest value to l2_size in test_sizes array
                    if l2_size > 0:
                        nearest_value = min(test_sizes, key=lambda x: abs(x - int(l2_size)))
                        l2_position = test_sizes.index(nearest_value)
                    #Find position of nearest value to l3_size in test_sizes array
                    if l3_size > 0:
                        nearest_value = min(test_sizes, key=lambda x: abs(x - int(l3_size)))
                        l3_position = test_sizes.index(nearest_value)

                    test_sizes_str = list(map(str, test_sizes))

                    #Plotting
                    fig, ax = plt.subplots(figsize=(14*1.5,3.75*1.5))
                    plt.plot(test_sizes_str, Gbps, marker='o', linestyle='-', color="blue")
                    if(interleaved):
                        plt.title(name + ' Memory Bandwidth Curve: ' + str(isa) + " " + str(precision) + " " + str(threads) + " Threads " + str(num_ld) + " Load " + str(num_st) + " Store Interleaved", fontsize=18)
                    else:
                        plt.title(name + ' Memory Bandwidth Curve: ' + str(isa) + " " + str(precision) + " " + str(threads) + " Threads " + str(num_ld) + " Load " + str(num_st) + " Store", fontsize=18)
                    
                    plt.xlabel('Test Data Sizes Per Thread (Kb)', fontsize=18)
                    plt.ylabel('Measure Bandwidth (Gbps)', fontsize=18, color="blue")

                    plt.grid(True)
                    #Manually set tick positions
                    plt.xticks(test_sizes_str)
                    ax.set_xticklabels(test_sizes_str, rotation = len(test_sizes))
                    plt.xlim(left = 0)
                    plt.xlim(right = (len(test_sizes)-1))
                    if l1_size > 0:
                        plt.axvline(x = l1_position, color = 'k', lw = 3, label = 'L1 Cache Size')
                    if l2_size > 0:
                        plt.axvline(x = l2_position, color = 'grey', lw = 3, label = 'L2 Cache Size')
                    if l3_size > 0:
                        plt.axvline(x = l3_position, color = 'k',linestyle='dashed', lw = 3, label = 'L3 Cache Size')

                    plt.legend(bbox_to_anchor = (1.0, 1), loc = 'best', fontsize=18)
                    ax2 = ax.twinx()
                    color = 'tab:red'
                    ax2.set_ylabel('Instructions per Cycle', fontsize=18, color=color)
                    ax2.plot(test_sizes_str, InstCycle, color=color, marker='o', linestyle='-')
                    ax2.tick_params(axis='y')
                    new_rc_params = {'text.usetex': False,"svg.fonttype": 'none'}
                    plt.rcParams.update(new_rc_params)
                    plt.tight_layout()
                    
                    date = ct.strftime('%Y-%m-%d_%H-%M-%S')
                    if interleaved:
                        plt.savefig('carm_results/memory_curve/' + name + '_memory_curve_' + date + "_" + isa + "_" + str(precision) + "_" + str(threads) + "_Threads_" + str(num_ld) + "Load_" + str(num_st) + "Store_" + "Interleaved" + '.svg')
                    else:
                        plt.savefig('carm_results/memory_curve/' + name + '_memory_curve_' + date + "_" + isa + "_" + str(precision) + "_" + str(threads) + "_Threads_" + str(num_ld) + "Load_" + str(num_st) + "Store" + '.svg')
                    
def update_memory_csv(Gbps, InstCycle, date, name, l1_size, l2_size, l3_size, isa, precision, num_ld, num_st, threads, interleaved, VLEN, LMUL):

    csv_path = f"./carm_results/memory_curve/{name}_memory_curve.csv"
    
    #Concatenate each test size with itself
    duplicated_test_sizes = [size for size in test_sizes for _ in range(2)]

    secondary_headers = ['Name:', name, 'L1 Size:', l1_size, 'L2 Size:', l2_size, 'L3 Size:', l3_size] + duplicated_test_sizes
    primary_headers = ['Date', 'ISA', 'Precision', 'Threads', 'Loads', 'Stores', 'Interleaved'] + [''] + ["Gbps", "I/Cycle"] * len(test_sizes)

    #Check if the file exists
    if os.path.exists(csv_path):
        mode = 'a'
    else:
        mode = 'w'

    with open(csv_path, mode, newline='') as csvfile:
        writer = csv.writer(csvfile)
        if mode == 'w':  #If writing mode, write header
            writer.writerow(secondary_headers)
            writer.writerow(primary_headers)
        #Write the results to CSV
        if (isa in ["rvv0.7", "rvv1.0"]):
            isa = str(isa) + "_vl" + str(VLEN) + "_lmul" + str(LMUL)
        elif isa in ["sve"]:
            isa = str(isa) + "_vl" + str(VLEN)
        results = [date, isa, precision, threads, num_ld, num_st, "Yes" if interleaved else "No"] + ['']
        for gbps, inst_cycle in zip(Gbps, InstCycle):
            results.append(ut.custom_round(gbps))
            results.append(ut.custom_round(inst_cycle))
        writer.writerow(results)

#Run Mixed Benchmark
def run_mixed(name, freq, l1_size, l2_size, l3_size, inst, isa_set, precision_set, num_ld, num_st, num_fp, threads_set, interleaved, num_ops, l3_bytes, dram_bytes, dram_auto, test_type, verbose, set_freq, no_freq_measure, VLEN, tl1, tl2, LMUL, num_runs):
    
    isa_set, l1_size, l2_size, l3_size, VLEN, LMUL = check_hardware(isa_set, freq, set_freq, verbose, precision_set, l1_size, l2_size, l3_size, VLEN, LMUL)
    VLEN_aux = VLEN
    LMUL_aux = LMUL
    dram_bytes_aux = dram_bytes
    freq_nominal = freq
    freq_real = freq
    
    if 0 < verbose < 4:
        print_start(test_type, threads_set, isa_set, precision_set, VLEN, LMUL)

    for threads in threads_set:
        for isa in isa_set:
            for precision in precision_set:            
                dram_bytes = dram_bytes_aux
                if VLEN_aux > 1 and precision == "dp":
                    VLEN = VLEN_aux
                    LMUL = LMUL_aux
                if VLEN_aux > 1 and precision == "sp":
                    VLEN = VLEN_aux * 2
                    LMUL = LMUL_aux
                if isa not in  vector_agnostic_ISA:
                    VLEN = 1
                    LMUL = 1
                if isa in ["sve"]:
                    LMUL = 1

                print_mid(verbose, test_type, threads_set, isa_set, precision_set, threads, isa, precision, VLEN, LMUL)
                
                if test_type == "mixedL1":
                    if l1_size > 0:
                        num_reps = int(int(l1_size)*1024/(tl1*2*mem_inst_size[isa][precision]*(num_ld+num_st)*VLEN*LMUL))
                        test_size = (int(l1_size))/(tl1*2)
                    else:
                        print("ERROR: No L1 Size Found, you can use the -l1 <l1_size> argument, or a configuration file to specify it.")
                        return
                elif test_type == "mixedL2":
                    if l2_size > 0:
                        num_reps = int(1024*int(l2_size)/tl2)/(mem_inst_size[isa][precision]*(num_ld+num_st)*VLEN*LMUL)
                        test_size = int(int(l2_size)/tl2)
                    else:
                        print("ERROR: No L2 Size Found, you can use the -l2 <l2_size> argument, or a configuration file to specify it.")
                        return
                elif test_type == "mixedL3":
                    if l3_size == 0:
                        print("ERROR: No L3 Size Found, you can use the -l3 <l3_size> argument, or a configuration file to specify it.")
                        return
                    if l3_bytes > 0:
                        num_reps = l3_bytes*1024/(threads*mem_inst_size[isa][precision]*(num_ld+num_st)*VLEN*LMUL)
                        test_size = int(l3_bytes/(threads))
                    elif l3_size > 0 and (int(l2_size)*threads + (int(l3_size) - int(l2_size)*threads)/2)/threads > l2_size:
                        num_reps = int(1024*(int(l2_size)*threads + (int(l3_size) - int(l2_size)*threads)/2)/(threads*mem_inst_size[isa][precision]*(num_ld+num_st)*VLEN*LMUL))
                        test_size = int((int(l2_size)*threads + (int(l3_size) - int(l2_size)*threads)/2)/threads)
                    else:
                        num_reps = int(1024*(int(l2_size*1.2))/(mem_inst_size[isa][precision]*(num_ld+num_st)*VLEN*LMUL))
                        test_size = int(l2_size*1.2)
                        if verbose > 0:
                            print("WARNING: L3 size insuficient to give each thread a memory slice significantly larger than L2 without exceeding L3 size.\n"
                            "For a more detailed memory analysis run '--test MEM' to identify the most appropriate size for the L3 test.\n"
                            "Then use the argument '--l3_kbytes <test_size>' to enforce a custom test size")
                    
                    if verbose > 3:
                        print(f"Total L3 Test Size: {test_size*threads}Kb | L3 Size: {l3_size}kb | L3 Test Size per Thread: {test_size}Kb | L2 Size: {l2_size}Kb")
                elif test_type == "mixedDRAM":
                    if (dram_auto) and l3_size > 0 and ((int(dram_bytes)/threads) < (int(l3_size)*2)):
                        num_reps = int((int(l3_size)*2)*1024/(mem_inst_size[isa][precision]*(num_ld+num_st)*VLEN*LMUL))
                        test_size = int((int(l3_size)*2))
                        dram_bytes = int(l3_size)*2*threads
                    else:
                        num_reps = int(dram_bytes*1024/(threads*mem_inst_size[isa][precision]*(num_ld+num_st)*VLEN*LMUL))
                        test_size = int(dram_bytes/(threads))

                        if int(test_size) <= int(l3_size) and verbose > 0:
                            print("WARNING: DRAM test size per thread is not sufficient to guarantee best results, to guarantee best results consider changing the default test size.")
                            print("By using --dram_bytes", int(l3_size)*2*int(threads), "(", ut.custom_round(float((int(l3_size)*2*int(threads))/1048576)), "Gb) the minimum test size necessary for", threads, "threads is achieved, using the --dram_auto flag will automatically apply this adjustement.")
                    if verbose > 3:
                        print("DRAM Test Size per Thread:", test_size, "Kb | L3 Size:", l3_size, "Kb | Total DRAM Test Size:", ut.custom_round(float((test_size*threads)/1048576)), "Gb")

                if inst == "fma":
                    FP_factor = 2
                else:
                    FP_factor = 1

                if verbose > 3:
                    make_verb_flag = ""
                else:
                    make_verb_flag = "-s"

                if verbose == 4:
                    print("-------------------------------------------------")
                    print("Debug output:")

                os.system(f"make {make_verb_flag} -C {script_dir} clean && make {make_verb_flag} -C {script_dir} isa={isa}")
                os.system(str(bench_ex) + " -test MIXED -num_LD " + str(num_ld) + " -num_ST " + str(num_st) + " -num_FP " + str(num_fp) + " -op " + inst + " -precision " + precision + " -num_rep " + str(num_reps) + " -Vlen " + str(VLEN) + " -LMUL " + str(LMUL) + " -verbose " + str(verbose) + " -num_runs " + str(num_runs))

                if(interleaved):
                    result = subprocess.run([test_ex, "-threads", str(threads), "-freq", str(freq), "-measure_freq", str(no_freq_measure), "--interleaved"], stdout=subprocess.PIPE)
                    inter = "Yes"
                else:
                    result = subprocess.run([test_ex, "-threads", str(threads), "-freq", str(freq), "-measure_freq", str(no_freq_measure)], stdout=subprocess.PIPE)
                    inter = "No"
                out = result.stdout.decode('utf-8').split(',')

                inner_loop_reps = float(out[1])
                freq_real = float(out[2])

                if isa in x86_ISAs:
                    cycles = float(out[0])
                    freq_nominal = float(out[3])
                    gflops = float(threads*num_reps*num_fp*FP_factor*ops_fp[isa][precision]*freq_real*inner_loop_reps)/(cycles*float(freq_real/freq_nominal))
                    ai = float((num_fp*FP_factor*ops_fp[isa][precision])/((num_ld+num_st)*mem_inst_size[isa][precision]))
                    bandwidth = float((threads*num_reps*(num_ld+num_st)*mem_inst_size[isa][precision]*freq_real*inner_loop_reps)/(cycles*float(freq_real/freq_nominal)))
                    time_ms = float (cycles / (freq_nominal * 1e6))
                    if (verbose > 1):
                        print("--------------------RESULTS----------------------")
                        print ("ISA:", isa,  "| Number of Threads:", threads, "| Allocated Size:", test_size, "Kb | Precision:", precision, "| Interleaved:", inter, "| Number of Loads:", num_ld, "| Number of Stores:", num_st, "| Number of FP:", num_fp, "| Memory Instruction Size:", mem_inst_size[isa][precision]*VLEN, "| FP Operations per Instruction:", (FP_factor*ops_fp[isa][precision])*VLEN)
                        print("Best Average Cycles:", int(cycles), "| Best Average Time (in ms):", ut.custom_round(time_ms))
                        print("Instructions Per Cycle:", ut.custom_round(threads*num_reps*(num_ld+num_st+num_fp)*inner_loop_reps/(cycles*float(freq_real/freq_nominal))),
                              " | FP Instructions Per Cycle:", ut.custom_round(threads*num_reps*(num_fp)*inner_loop_reps/(cycles*float(freq_real/freq_nominal))),
                              " | Memory Instructions per Cycle:", ut.custom_round(threads*num_reps*(num_ld+num_st)*inner_loop_reps/(cycles*float(freq_real/freq_nominal))))
                        print("Bandwidth (GB/s):", ut.custom_round(bandwidth), " | GFLOP/s:", ut.custom_round(gflops))
                        print("Total Flops:", int(num_fp*FP_factor*ops_fp[isa][precision]*num_reps*inner_loop_reps),
                              " | Total Bytes:", int(((num_ld+num_st)*mem_inst_size[isa][precision])*num_reps*inner_loop_reps))
                        print("Arithmetic Intensity:", ut.custom_round(ai))
                        print("Max Recorded Frequency (GHz):", ut.custom_round(freq_real), "| Nominal Frequency (GHz):", ut.custom_round(freq_nominal), "| Actual Frequency to Nominal Frequency Ratio:", ut.custom_round(float(freq_real/freq_nominal)))
                        if verbose == 4:
                            print("Results Debug -> Total Inner Loop Reps:", int(inner_loop_reps), "| Number of reps:", num_reps)
                        elif 0 < verbose < 4:
                            print("-------------------------------------------------")
                else:
                    time_ms = float(out[0])
                    gflops = float((threads*num_reps*num_fp*FP_factor*ops_fp[isa][precision]*VLEN*inner_loop_reps)/(1000000000))/((time_ms/1000))
                    ai = float((num_fp*FP_factor*ops_fp[isa][precision])/((num_ld+num_st)*mem_inst_size[isa][precision]))
                    bandwidth = (float((threads*num_reps*(num_ld+num_st)*mem_inst_size[isa][precision]*VLEN*inner_loop_reps)/(1000000000))/((time_ms/1000)))
                    if (verbose > 1):
                        print("--------------------RESULTS----------------------")
                        print ("ISA:", isa,  "| Number of Threads:", threads, "| Allocated Size:", test_size, "Kb | Precision:", precision, "| Interleaved:", inter, "| Number of Loads:", num_ld, "| Number of Stores:", num_st, "| Number of FP:", num_fp, "| Memory Instruction Size:", mem_inst_size[isa][precision]*VLEN, "| FP Operations per Instruction:", (FP_factor*ops_fp[isa][precision])*VLEN)
                        print("Best Average Time (in ms):", ut.custom_round(time_ms))
                        print("Instructions Per Cycle:", ut.custom_round((threads*num_reps*(num_ld+num_st+num_fp)*inner_loop_reps)/((time_ms/1000)*freq_real*1000000000)))
                        print("FP Instructions Per Cycle:", ut.custom_round((threads*num_reps*(num_fp)*inner_loop_reps)/((time_ms/1000)*freq_real*1000000000)))
                        print("Memory Instructions Per Cycle:", ut.custom_round((threads*num_reps*(num_ld+num_st)*inner_loop_reps)/((time_ms/1000)*freq_real*1000000000)))
                        print("Bandwidth (GB/s):", ut.custom_round(bandwidth))
                        print("GFLOP/s:", ut.custom_round(gflops))
                        print("Total Flops:", int(num_fp*FP_factor*ops_fp[isa][precision]*num_reps*inner_loop_reps*VLEN))
                        print("Total Bytes:", int(((num_ld+num_st)*mem_inst_size[isa][precision])*num_reps*inner_loop_reps*VLEN))
                        print("Arithmetic Intensity:", ut.custom_round(ai))
                        print("Max Recorded Frequency (GHz):", ut.custom_round(freq_real))
                        if isa == "rvv0.7" or isa == "rvv1.0":
                            print("Vector Length:", VLEN, "Elements | Vector LMUL:", LMUL)
                        if verbose == 4:
                            print("Results Debug -> Total Inner Loop Reps:", int(inner_loop_reps), "| Number of reps:", num_reps)
                        elif 0 < verbose < 4:
                            print("-------------------------------------------------")
                
                ct = datetime.datetime.now()
                date = ct.strftime('%Y-%m-%d %H:%M:%S')
                test_details = test_type + "_" + str(num_fp) + "FP_" + str(num_ld) + "LD_" + str(num_st) + "ST_" + inst
                ut.update_csv(name, "/home/mixed", gflops, ai, bandwidth, time_ms, test_details, date, isa, precision, threads, "MIX", VLEN, LMUL)

def main():
    parser = argparse.ArgumentParser(description='Script to run micro-benchmarks to construct Cache-Aware Roofline Model')
    parser.add_argument('--test', default='roofline', nargs='?', choices=['FP', 'L1', 'L2', 'L3', 'DRAM', 'roofline', 'MEM', 'mixedL1', 'mixedL2', 'mixedL3', 'mixedDRAM'], help='Type of the test. Roofline test measures the bandwidth of the different memory levels and FP Performance, MEM test measures the bandwidth of various memory sizes, mixed test measures bandwidth and FP performance for a combination of memory acceses (to L1, L2, L3, or DRAM) and FP operations (Default: roofline)')
    parser.add_argument('-nr', '--num_runs', default=1024, nargs='?', type=ut.positive_int, help='Number of repetitions for the benchmarks (Default: 1024)')
    parser.add_argument('-ops', '--num_ops',  default=32768, nargs='?', type=ut.positive_int, help='Number of FP operations to be used in FP test (Default: 32768)')
    
    parser.add_argument('--isa', default=['auto'], nargs='+', choices=['avx512', 'avx2', 'sse', 'scalar', 'neon', 'armscalar', 'sve', 'riscvscalar', 'rvv0.7', 'rvv1.0', 'auto'], help='set of ISAs to test, if auto will test all available ISAs (Default: auto)')
    parser.add_argument('-p', '--precision', default=['dp'], nargs='+', choices=['dp', 'sp'], help='Data Precision (Default: dp)')
    parser.add_argument('-t', '--threads', default=[1], nargs='+', type=ut.positive_int, help='Set of number of threads for the micro-benchmarking, insert multiple thread valus by spacing them, no commas (Default: [1])')
    parser.add_argument('-i', '--interleaved',  dest='interleaved', action='store_const', const=1, default=0, help='For thread binding when cores are interleaved between NUMA domains (Default: 0)')
    parser.add_argument('--inst', default='add', nargs='?', choices=['add', 'mul', 'div', 'fma'], help='FP Instruction (Default: add), FMA performance is measured by default too.')
    parser.add_argument('-vl', '--vector_length',  default=1, nargs='?', type=ut.positive_int, help='Vector Length in double/single precision elements (if running dp and sp in one run, double precision elements will be assumed) for RVV configuration (Default: Max available)')
    parser.add_argument('-vlmul', '--vector_lmul', default=1, nargs='?', type = int, choices=[1, 2, 4, 8], help='Vector Register Grouping for RVV configuration (Default: 1)')
    
    parser.add_argument('-ldst', '--ld_st_ratio',  default=2, nargs='?', type=ut.positive_int, help='Load/Store Ratio (Default: 2)')
    parser.add_argument('-fpldst', '--fp_ld_st_ratio',  default=1, nargs='?', type=ut.positive_int, help='FP to Load/Store Ratio, for mixed test (Default: 1)')
    parser.add_argument('--only_ld',  dest='only_ld', action='store_const', const=1, default=0, help='Run only loads in mem test (ld_st_ratio is ignored)')
    parser.add_argument('--only_st',  dest='only_st', action='store_const', const=1, default=0, help='Run only stores in mem test (ld_st_ratio is ignored)')
    
    parser.add_argument('--l3_kbytes',  default=0, nargs='?', type=ut.positive_int, help='Total Size of the array for the L3 test in KiB')
    parser.add_argument('--dram_kbytes',  default=524288, nargs='?', type=ut.positive_int, help='Total Size of the array for the DRAM test in KiB (Default: 524288 (512 MiB))')
    parser.add_argument('--dram_auto',  dest='dram_auto', action='store_const', const=1, default=0, help='Automatically calculate the DRAM test size needed to ensure data does not fit in L3, can require a lot of memory in some cases, make sure it fits in the DRAM of your system (Default: 0)')

    parser.add_argument('-tl1', '--threads_per_l1',  default=1, nargs='?', type = int, help='Expected amount of threads that will have to share the same L1 cache (Default: 1)')
    parser.add_argument('-tl2', '--threads_per_l2',  default=2, nargs='?', type = int, help='Expected amount of threads that will have to share the same L2 cache (Default: 2)')
    parser.add_argument('-l1', '--l1_size',  default=0, nargs='?', type = int, help='L1 size per core')
    parser.add_argument('-l2', '--l2_size',  default=0, nargs='?', type = int, help='L2 size per core')
    parser.add_argument('-l3', '--l3_size',  default=0, nargs='?', type = int, help='L3 total size')
    parser.add_argument('config', nargs='?', help='Path for the system configuration file')

    parser.add_argument('--no_freq_measure',  dest='no_freq_measure', action='store_const', const=1, default=0, help='Measure CPU frequency or not')
    parser.add_argument('--freq', default='2.0', nargs='?', type = float, help='Expected/Desired CPU frequency during test (if no_freq_measure or set_freq is enabled)')
    parser.add_argument('--set_freq',  dest='set_freq', action='store_const', const=1, default=0, help='Set Processor frequency to indicated one (x86 only, might require admin priviliges and might not work for certain systems)')

    parser.add_argument('-v', '--verbose', default=3, nargs='?', type = int, choices=[0, 1, 2, 3, 4], help='Level of terminal output details (0 -> No Output 1 -> Only ISA/Configuration Errors and Test Specifications, 2 -> Test Results, 3 -> Configuration Values Selected/Detected, 4 -> Debug Output)')
    parser.add_argument('--name', default='unnamed', nargs='?', type = str, help='Name for results file (if not using config file)')
    parser.add_argument('--plot',  dest='plot', action='store_const', const=1, default=0, help='Create CARM plot SVG for each test result')
    parser.add_argument('-out', '--output', nargs='?', default='./carm_results', help='Path to a folder to save roofline results to (Default: ./carm_results | Only applies to roofline results)')

    args = parser.parse_args()
    if (args.verbose == 4):
        print(args)
    l1_size = args.l1_size
    l2_size = args.l2_size
    l3_size = args.l3_size
    freq = 0
    name = ''
    
    if (args.config != None):
        name, l1_size, l2_size, l3_size = read_config(args.config)

    freq = args.freq
    verbose = args.verbose
    precision_set = args.precision
    vlen = args.vector_length
    
    #uses name from arguments if not present in config file
    if (name == ''):
        name = args.name

    if(args.only_ld == 1):
        num_ld = 1
        if args.test in ["mixedL1", "mixedL2", "mixedL3", "mixedDRAM"]:
            num_ld = args.ld_st_ratio
        num_st = 0
    elif(args.only_st == 1):
        num_ld = 0
        num_st = 1
    else:
        num_ld = args.ld_st_ratio
        num_st = 1
    if args.test in ["mixedL1", "mixedL2", "mixedL3", "mixedDRAM"]:
        num_fp = args.fp_ld_st_ratio
    else:
        num_fp = 0
    
    isa_set = args.isa
    if 'rvv0.7' in isa_set and 'rvv1.0' in isa_set:
        print("ERROR: Only one RVV version must be specified. Exiting Program.")
        sys.exit(1)
    if 'auto' in isa_set and len(isa_set) > 1:
        if (verbose > 0):
            print("WARNING: When using auto ISA detection do not specify additional ISAs, setting ISA argument to just auto and removing additional ISAs.")
        isa_set = ["auto"]
    if ("sp" in precision_set and "dp" not in precision_set):
        if (vlen%2 == 0):
            vlen = vlen/2
        else:
            if (verbose > 0):
                print("WARNING: Please specify an even number for --vector_length (-vl) argument when using single precision.")
        
    if args.test in ["mixedL1", "mixedL2", "mixedL3", "mixedDRAM"]:
        run_mixed(name, freq, l1_size, l2_size, l3_size, args.inst, isa_set, precision_set, num_ld, num_st, num_fp, args.threads, args.interleaved, args.num_ops, args.l3_kbytes, args.dram_kbytes, args.dram_auto, args.test, verbose, args.set_freq, args.no_freq_measure, vlen, args.threads_per_l1,  args.threads_per_l2, args.vector_lmul, args.num_runs)
    elif args.test == 'MEM':
        run_memory(name, freq, args.set_freq, l1_size, l2_size, l3_size, isa_set, precision_set, num_ld, num_st, args.threads, args.interleaved, verbose, args.no_freq_measure, vlen, args.threads_per_l1, args.plot, args.vector_lmul, args.num_runs)
    else:
        run_roofline(name, freq, l1_size, l2_size, l3_size, args.inst, isa_set, precision_set, num_ld, num_st, args.threads, args.interleaved, args.num_ops, args.l3_kbytes, args.dram_kbytes, args.dram_auto, args.test, verbose, args.set_freq, args.no_freq_measure, vlen, args.threads_per_l1,  args.threads_per_l2, args.plot, args.vector_lmul, args.output, args.num_runs)

if __name__ == "__main__":
    main()

