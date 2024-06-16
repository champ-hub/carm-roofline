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
import re
import sys

import DBI_AI_Calculator

riscv_vector_compiler_path = "gcc"
#riscv_vector_compiler_path = "/home/inesc/llvm-EPI-0.7-development-toolchain-native/bin/clang"

#Mapping between ISA and memory transfer size
mem_inst_size = {"avx512": {"sp": 64, "dp": 64}, "avx": {"sp": 32, "dp": 32}, "avx2": {"sp": 32, "dp": 32}, "sse": {"sp": 16, "dp": 16}, "scalar": {"sp": 4, "dp": 8}, "neon": {"sp": 16, "dp": 16}, "armscalar": {"sp": 4, "dp": 8}, "riscvscalar": {"sp": 4, "dp": 8}, "rvv0.7": {"sp": 4, "dp": 8}, "rvv1.0": {"sp": 4, "dp": 8}}
ops_fp = {"avx512": {"sp": 16, "dp": 8}, "avx": {"sp": 8, "dp": 4}, "avx2": {"sp": 8, "dp": 4}, "sse": {"sp": 4, "dp": 2}, "scalar": {"sp": 1, "dp": 1}, "neon":{"sp": 4, "dp": 2}, "armscalar": {"sp": 1, "dp": 1}, "riscvscalar": {"sp": 1, "dp": 1}, "rvv0.7": {"sp": 1, "dp": 1}, "rvv1.0": {"sp": 1, "dp": 1}}
test_sizes = [2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 600, 768, 1024, 1536, 2048, 3072, 4096, 5120, 6144, 8192, 10240, 12288, 16384, 24567, 32768, 65536, 98304, 131072, 262144, 393216, 524288]

def check_hardware(isa_set, freq, set_freq, verbose, precision, l1_size, l2_size, l3_size, VLEN, LMUL):
    CPU_Type = platform.machine()

    if CPU_Type == "x86_64":
        
        auto_args = autoconf(freq*1000000, set_freq)
        #If user defines no ISA in arguments, uses all of the best ones
        if (isa_set[0] == "auto"):
            #If avx512 is supported
            if auto_args[0] == "avx512":
                isa_set[0] = auto_args[0]
                #We can then use 32 registers for avx2
                if auto_args[1] == "avx2":
                    isa_set.append("avx")
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
                if item in ["neon", "armscalar", "rvv1.0", "rvv0.7", "riscvscalar"]:
                    print("WARNING: Selected ISA " + item + " was detected and removed since it is not supported by " + CPU_Type + " architectures.")
                else:
                    supported_isas.append(item)
            isa_set = supported_isas
            supported_isas = []

            for isa in isa_set:
                #If user defines ISA, check support on the machine
                if(isa == "avx512" and auto_args[0] != "avx512"):
                    print("AVX512 not supported on this machine.")
                    continue
                if((isa == "avx2" or isa == "avx") and auto_args[0] == "avx512" and auto_args[1] == "avx2"):
                    #To use more registers with avx2
                    isa = "avx"
                if ((isa == "avx2" or isa == "avx") and auto_args[1] != "avx2"):
                    print("AVX2 not supported on this machine.")
                    continue
                if (isa == "sse" and auto_args[2] != "sse"):
                    print("SSE not supported on this machine.")
                    continue
                supported_isas.append(isa)
            #If no ISA specified is valid, default to Scalar
            if not supported_isas:
                supported_isas = ["scalar"]

            isa_set = supported_isas
        if (verbose > 2):
            print ("-----------------CPU INFORMATION-----------------")
            print("Vector Instruction ISAs Supported:", auto_args[0], auto_args[1], auto_args[2])
            print("CPU Vendor:", auto_args[3])
            print("L1 cache size:", auto_args[4], "KB")
            print("L2 cache size:", auto_args[5], "KB")
            print("L3 cache size:", auto_args[6], "KB")
            print ("-------------------------------------------------")
        #uses cache sizes from probing if not present in config file
        if (l1_size == 0):
            l1_size = auto_args[4]
        if (l2_size == 0):
            l2_size = auto_args[5]
        if (l3_size == 0):
            l3_size = auto_args[6]

        if VLEN != 1:
            print("WARNING: --vector_length (-vlen) argument is only used for RVV benchmarks.")
            VLEN = 1
        if LMUL != 1:
            print("WARNING: --vector_lmul (-vlmul) argument is only used for RVV benchmarks.")
            LMUL = 1
        
        return isa_set, int(l1_size), int(l2_size), int(l3_size), int(VLEN), int(LMUL)

    elif CPU_Type == "aarch64":
        #if we have an ARM CPU
        supported_isas = []

        for item in isa_set:
            if item in ["sse", "avx2", "avx512", "rvv1.0", "rvv0.7", "riscvscalar"]:
                print("WARNING: Selected ISA " + item + " was detected and removed since it is not supported by " + CPU_Type + " architectures.")
            else:
                supported_isas.append(item)
        isa_set = supported_isas
        if len(isa_set) == 0:
            isa_set.append("auto")

        if VLEN != 1:
            print("WARNING: --vector_length (-vlen) argument is only used for RVV benchmarks.")
            VLEN = 1
        if LMUL != 1:
            print("WARNING: --vector_lmul (-vlmul) argument is only used for RVV benchmarks.")
            LMUL = 1
        isa_set = ["armscalar" if x == "scalar" else x for x in isa_set]

        if (isa_set[0] == "auto"):
            isa_set[0] = "neon"
            isa_set.append("armscalar")

            return isa_set, int(l1_size), int(l2_size), int(l3_size), int(VLEN), int(LMUL)
        return isa_set, int(l1_size), int(l2_size), int(l3_size), int(VLEN), int(LMUL)
    elif CPU_Type == "riscv64":
        #if we have a RISCV CPU
        supported_isas = []

        for item in isa_set:
            if item in ["neon", "armscalar", "sse", "avx2", "avx512"]:
                print("WARNING: Selected ISA " + item + " was detected and removed since it is not supported by " + CPU_Type + " architectures.")
            else:
                supported_isas.append(item)
        isa_set = supported_isas
        if len(isa_set) == 0:
            isa_set.append("auto")
        
        isa_set = ["riscvscalar" if x == "scalar" else x for x in isa_set]
        if (isa_set[0] == "auto" or "rvv0.7" in isa_set or "rvv1.0" in isa_set):
            if "rvv0.7" in isa_set:
                subprocess.run([riscv_vector_compiler_path, "-o", "./config/auto_config/RISCV_Vector", "./config/auto_config/RISCV_Vector.c", "-march=rv64gcv0p7"])
            elif "rvv1.0" in isa_set:
                subprocess.run([riscv_vector_compiler_path, "-o", "./config/auto_config/RISCV_Vector", "./config/auto_config/RISCV_Vector.c", "-march=rv64gcv"])
            #subprocess.run([riscv_vector_compiler_path, "-o", "./config/auto_config/RISCV_Vector", "./config/auto_config/RISCV_Vector.c", "-march=rv64gcv0p7", "-menable-experimental-extensions"])
            result = subprocess.run(["./config/auto_config/RISCV_Vector", "dp"], stdout=subprocess.PIPE)
            VLEN_Check = int(result.stdout.decode('utf-8'))
            os.remove("./config/auto_config/RISCV_Vector")
            if VLEN < VLEN_Check:
                if VLEN == 1:
                    VLEN = VLEN_Check
                if (verbose > 2):
                    print("RISCV Vector with " + str(VLEN) + " elements (dp).")
                if "riscvscalar" not in isa_set and isa_set[0] == "auto":
                    isa_set.append("riscvscalar")
                if "rvv0.7" not in isa_set and "rvv1.0" not in isa_set:
                    print("WARNING: Automatic detection of RVV version not available yet, please specify RVV version using the --isa argument.")
            else:
                print("WARNING: RISCV Vector Support Not Detected in the System")
                isa_set[0] = "riscvscalar"
                if VLEN != 1:
                    print("WARNING: --vector_length (-vlen) argument is only used for RVV benchmarks.")
                    VLEN = 1
                if LMUL != 1:
                    print("WARNING: --vector_lmul (-vlmul) argument is only used for RVV benchmarks.")
                    LMUL = 1
            return isa_set, int(l1_size), int(l2_size), int(l3_size), int(VLEN), int(LMUL)
        else:
            if VLEN != 1:
                print("WARNING: --vector_length (-vlen) argument is only used for RVV benchmarks.")
                VLEN = 1
            if LMUL != 1:
                print("WARNING: --vector_lmul (-vlmul) argument is only used for RVV benchmarks.")
                LMUL = 1
            isa_set[0] = "riscvscalar"
            return isa_set, int(l1_size), int(l2_size), int(l3_size), int(VLEN), int(LMUL)
    else:
        print("ERROR: Unsupported architecture " + CPU_Type + " detected. Exiting Program.")
        sys.exit(1)
    
#Call system probing and frequency setting code (x86 Only)
def autoconf(new_max_freq, set_freq):

    subprocess.run(["gcc", "-o", "./config/auto_config/autoconfig", "./config/auto_config/autoconfig.c"])

    if (set_freq == 0):
        new_max_freq = 0
        
    result = subprocess.run(["./config/auto_config/autoconfig", str(new_max_freq)], stdout=subprocess.PIPE)
    arguments = result.stdout.decode('utf-8').split('\n')

    os.remove("./config/auto_config/autoconfig")

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


def round_power_of_2(number):
    if number > 1:
        for i in range(1, int(number)):
            if (2 ** i > number):
                return 2 ** i
    else:
        return 1

def carm_eq(ai, bw, fp):
    return np.minimum(ai*bw, fp)


def plot_roofline(name, data, date, isa, precision, threads, num_ld, num_st, inst, interleaved):
    if plot_numpy == None:
        print("No Matplotlib and/or Numpy found, in order to draw CARM graphs make sure to install them.")
        return
    fig, ax = plt.subplots(figsize=(7.1875*1.5,3.75*1.5))
    plt.xlim(0.015625, 256)
    if inst == "fma":
        plt.ylim(0.25, round_power_of_2(int(data['FP'])))
    else:
        plt.ylim(0.25, round_power_of_2(int(data['FP_FMA'])))
    ai = np.linspace(0.00390625, 256, num=200000)

    #Ploting Lines
    if inst == "fma":
        plt.plot(ai, carm_eq(ai, data['L1'], data['FP']), 'k', lw = 3, label='L1')
        plt.plot(ai, carm_eq(ai, data['L2'], data['FP']), 'grey', lw = 3, label='L2')
        plt.plot(ai, carm_eq(ai, data['L3'], data['FP']), 'k', linestyle='dashed', lw = 3, label='L3')
        plt.plot(ai, carm_eq(ai, data['DRAM'], data['FP']), 'k', linestyle='dotted', lw = 3, label='DRAM')
    else:
        plt.plot(ai, carm_eq(ai, data['L1'], data['FP_FMA']), 'k', lw = 3, label='L1')
        plt.plot(ai, carm_eq(ai, data['L2'], data['FP_FMA']), 'grey', lw = 3, label='L2')
        plt.plot(ai, carm_eq(ai, data['L3'], data['FP_FMA']), 'k', linestyle='dashed', lw = 3, label='L3')
        plt.plot(ai, carm_eq(ai, data['DRAM'], data['FP_FMA']), 'k', linestyle='dotted', lw = 3, label='DRAM')
        plt.plot(ai, carm_eq(ai, data['L1'], data['FP']), 'k', linestyle='dashdot', lw = 3, label=inst)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if(interleaved):
        plt.title(name + ' CARM: ' + str(isa) + " " + str(precision) + " " + str(threads) + " Threads " + str(num_ld) + " Load " + str(num_st) + " Store " + inst + " Interleaved", fontsize=18)
    else:
        plt.title(name + ' CARM: ' + str(isa) + " " + str(precision) + " " + str(threads) + " Threads " + str(num_ld) + " Load " + str(num_st) + " Store " + inst, fontsize=18)
    plt.ylabel('Performance [GFLOPS/s]', fontsize=18)
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
        plt.savefig('Results/Roofline/' + name + '_roofline_' + str(date) + '_' + isa + "_" + str(precision) + "_" + str(threads) + "_Threads_" + str(num_ld) + "Load_" + str(num_st) + "Store_" + inst + "_Interleaved"'.svg')
    else:
        plt.savefig('Results/Roofline/' + name + '_roofline_' + str(date) + '_' + isa + "_" + str(precision) + "_" + str(threads) + "_Threads_" + str(num_ld) + "Load_" + str(num_st) + "Store_" + inst +'.svg')

def custom_round(value, digits=4):
    if value == 0:
        return 0  #Directly return 0 if the value is 0
    elif abs(value) >= 1:
        #For numbers greater than or equal to 1, round normally
        return round(value, digits)
    else:
        #For numbers less than 1, find the position of the first non-zero digit after the decimal
        str_value = str(value)
        if 'e' in str_value or 'E' in str_value:  #Check for scientific notation
            return round(value, digits)
        
        #Count positions until first non-zero digit after the decimal
        decimal_part = str_value.split('.')[1]
        leading_zeros = 0
        for char in decimal_part:
            if char == '0':
                leading_zeros += 1
            else:
                break
        
        #Adjust the number of digits based on the position of the first significant digit
        total_digits = digits + leading_zeros
        return round(value, total_digits)
    
def update_csv(name, test_type, data, data_cycles, date, isa, precision, threads, num_ld, num_st, inst, interleaved, l1_size, l2_size, l3_size, dram_bytes):

    csv_path = f"./Results/Roofline/{name}_{test_type.replace(' ', '_')}.csv"

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
        results.append(custom_round(data["L1"]))
        results.append(custom_round(data_cycles["L1"]))
    else:
        results.append(0)
        results.append(0)
    if l2_size > 0:
        results.append(custom_round(data["L2"]))
        results.append(custom_round(data_cycles["L2"]))
    else:
        results.append(0)
        results.append(0)
    if l1_size > 0:
        results.append(custom_round(data["L3"]))
        results.append(custom_round(data_cycles["L3"]))
    else:
        results.append(0)
        results.append(0)

    results.append(custom_round(data["DRAM"]))
    results.append(custom_round(data_cycles["DRAM"]))
    results.append(custom_round(data["FP"]))
    results.append(custom_round(data_cycles["FP"]))


    if not inst == "fma":
        results.append(custom_round(data["FP_FMA"]))
        results.append(custom_round(data_cycles["FP_FMA"]))
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

def print_results(isa, test_type, test_data, data_cycles, num_reps, test_size, inner_loop_reps, freq_real, cycles, freq_nominal, time_ms, threads, num_ld, num_st, inst, FP_factor, precision, VLEN, interleaved):
    if interleaved:
        inter = "Yes"
    else:
        inter = "No"

    print(f"---------{test_type} RESULTS-----------")
    if test_type not in ["FP", "FP_FMA"]:
        print("ISA:", isa,  "| Number of Threads:", threads, "| Allocated Size:", test_size, "Kb | Precision:", precision, "| Interleaved:", inter, "| Number of Loads:", num_ld, "| Number of Stores:", num_st, "| Memory Instruction Size:", mem_inst_size[isa][precision]*VLEN, "| Total Inner Loop Reps:", int(inner_loop_reps),  "| Number of Reps:", num_reps)
    else:
        print("ISA:", isa,  "| Number of Threads:", threads, "| Instruction:", inst, "| Precision:", precision, "| Interleaved:", inter, "| FP Operations per Instruction:", (FP_factor*ops_fp[isa][precision])*VLEN, "| Total Inner Loop Reps:", int(inner_loop_reps), "| Number of reps:", num_reps)
    if isa not in ["neon", "armscalar", "riscvscalar", "rvv0.7", "rvv1.0"]:
        print("Best Average Cycles:", cycles, "| Best Average Time (in ms):", time_ms)
    else:
        print("Best Average Time (in ms):", time_ms)
    
    print("Instructions per Cycle:", data_cycles)

    if test_type not in ["FP", "FP_FMA"]:
        print("Bytes per Cycle:", data_cycles*mem_inst_size[isa][precision]*VLEN)
        print("Bandwidth (Gbps):", test_data)
    else:
        print("Flops per Cycle:", data_cycles*ops_fp[isa][precision]*FP_factor*VLEN)
        print("GFLOPS:", test_data)
    if isa not in ["neon", "armscalar", "riscvscalar", "rvv0.7", "rvv1.0"]:
        print("Max Recorded Frequency (GHz):", freq_real, "| Nominal Frequency (GHz):", freq_nominal, "| Actual Frequency to Nominal Frequency Ratio:", float(freq_real/freq_nominal))
    else:
        print("Max Recorded Frequency (GHz):", freq_real)
    if isa == "rvv0.7" or isa == "rvv1.0":
        print("Vector Length:", VLEN, "Elements")
    print("------------------------------")

#Run Roofline tests
def run_roofline(name, freq, l1_size, l2_size, l3_size, inst, isa_set, precision_set, num_ld, num_st, threads_set, interleaved, num_ops, dram_bytes, dram_auto, test_type, verbose, set_freq, no_freq_measure, VLEN, tl1, tl2, plot, LMUL):
    
    num_reps = {}
    test_size = {}
    data = {}
    data_cycles = {}
    #LMUL = 1
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

    if verbose == 1:
        print("------------------------------")
        print("Running Benchmarks for the Following Threads Counts:", threads_set)
        print("On the Following ISA extensions: ", isa_set)
        print("Using the Following Precisions:", precision_set)
        print("------------------------------")
        

    for threads in threads_set:
        for isa in isa_set:
            #Compile benchmark generator
            os.system("make clean && make isa=" + isa )
            for precision in precision_set:
                if verbose > 1:
                    print("------------------------------")
                    print("Running Benchmarks for the Following Threads Counts:", threads_set)
                    print("On the Following ISA extensions: ", isa_set)
                    print("Using the Following Precisions:", precision_set)
                    print("Now Testing:", threads, "Threads on", isa, "with", precision)
                if verbose == 1:
                    print("------------------------------")
                    print("Now Testing:", threads, "Threads on", isa, "with", precision)
                
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
                if isa not in ["rvv0.7", "rvv1.0"]:
                    VLEN = 1
                    LMUL = 1
                if verbose > 2 and isa in ["rvv0.7", "rvv1.0"]:
                    print("VLEN IS:", VLEN, "| LMUL IS:", LMUL)
           

                #Calculate number of repetitions for each test
                if (l1_size > 0):
                    num_reps["L1"] = int(int(l1_size)*1024/(tl1*mem_inst_size[isa][precision]*(num_ld+num_st)*VLEN*LMUL))
                    test_size["L1"] = (int(l1_size))/tl1
                else:
                    print("WARNING: No L1 Size Found, you can use the -l1 <l1_size> argument, or a configuration file to specify it.")

                if (l2_size > 0):
                    num_reps["L2"] = int(int(1024*int(l2_size)/tl2)/(mem_inst_size[isa][precision]*(num_ld+num_st)*VLEN*LMUL))
                    test_size["L2"] = int(int(l2_size)/tl2)
                else:
                    print("WARNING: No L2 Size Found, you can use the -l2 <l2_size> argument, or a configuration file to specify it.")

                if (l3_size > 0):
                    num_reps["L3"] = int(1024*(int(l2_size)*threads + (int(l3_size) - int(l2_size)*threads)/2)/(threads*mem_inst_size[isa][precision]*(num_ld+num_st)*VLEN*LMUL))
                    test_size["L3"] = int((int(l2_size)*threads + (int(l3_size) - int(l2_size)*threads)/2)/threads)
                else:
                    print("WARNING: No L3 Size Found, you can use the -l3 <l3_size> argument, or a configuration file to specify it.")
                if (dram_auto and l3_size > 0 and dram_bytes/(threads) < (l3_size*2)):
                    num_reps["DRAM"] = int((int(l3_size)*2)*1024/(mem_inst_size[isa][precision]*(num_ld+num_st)*VLEN*LMUL))
                    test_size["DRAM"] = int((int(l3_size)*2))
                    dram_bytes = int(l3_size)*2*threads
                else:
                    if (dram_bytes > 0):
                        num_reps["DRAM"] = int(dram_bytes*1024/(threads*mem_inst_size[isa][precision]*(num_ld+num_st)*VLEN*LMUL))
                        test_size["DRAM"] = int(dram_bytes/(threads))
                        if int(test_size["DRAM"]) <= int(l3_size) and verbose > 0:
                            print("WARNING: DRAM test size per thread is not sufficient to guarantee best results, to guarantee best results consider changing the default test size.")
                            print("By using --dram_bytes", int(l3_size)*2*int(threads), "(", custom_round(float((int(l3_size)*2*int(threads))/1048576)), "Gb) the minimum test size necessary for", threads, "threads is achieved, using the --dram_auto flag will automatically apply this adjustement.")
                if verbose > 2:
                    print("DRAM Test Size per Thread:", test_size["DRAM"], "Kb | L3 Size:", l3_size, "Kb | Total DRAM Test Size:", custom_round(float((test_size["DRAM"]*threads)/1048576)), "Gb")

                num_reps["FP"] = int(num_ops/(FP_factor*ops_fp[isa][precision]*LMUL*VLEN))
                if inst != "fma":
                    num_reps["FP_FMA"] = int(num_ops/(2*ops_fp[isa][precision]*LMUL*VLEN))

                print("NUM OPS FP: " + num_reps["FP"])
                print("NUM OPS FP FMA: " + num_reps["FP_FMA"])

                if verbose > 0:
                    print("------------------------------")

                

                if (test_type == 'L1' or test_type == 'roofline') and l1_size > 0:
                    #Run L1 Test

                    os.system("./Bench/Bench -test MEM -num_LD " + str(num_ld) + " -num_ST " + str(num_st) + " -precision " + precision + " -num_rep " + str(num_reps["L1"]) + " -Vlen " + str(VLEN) + " -LMUL " + str(LMUL))
                    
                    if(interleaved):
                        result = subprocess.run(["./bin/test", "-threads", str(threads), "-freq", str(freq), "-measure_freq", str(no_freq_measure), "--interleaved"], stdout=subprocess.PIPE)
                    else:
                        result = subprocess.run(["./bin/test", "-threads", str(threads), "-freq", str(freq), "-measure_freq", str(no_freq_measure)], stdout=subprocess.PIPE)
                
                    out = result.stdout.decode('utf-8').split(',')
                    
                    inner_loop_reps = float(out[1])
                    freq_real = float(out[2])
                    if isa not in ["neon", "armscalar", "riscvscalar", "rvv0.7", "rvv1.0"]:
                        cycles = float(out[0])
                        freq_nominal = float(out[3])
                        data['L1'] = float((threads*num_reps["L1"]*(num_ld+num_st)*mem_inst_size[isa][precision]*freq_real*inner_loop_reps)/(cycles*float(freq_real/freq_nominal)))
                        data_cycles['L1'] = float((threads*num_reps["L1"]*(num_ld+num_st)*inner_loop_reps)/(cycles*float(freq_real/freq_nominal)))
                        time_ms = float (cycles / (freq_nominal * 1e6))
                    else:
                        time_ms = float(out[0])
                        data['L1'] = (float((threads*num_reps["L1"]*(num_ld+num_st)*mem_inst_size[isa][precision]*VLEN*LMUL*inner_loop_reps)/(1000000000))/((time_ms/1000)))
                        data_cycles['L1'] = float((threads*num_reps["L1"]*(num_ld+num_st)*LMUL*inner_loop_reps)/((time_ms/1000)*freq_real*1000000000))
                    
                    if (verbose > 1):
                        print_results(isa, "L1", data["L1"], data_cycles["L1"], num_reps["L1"], test_size["L1"], inner_loop_reps, freq_real, cycles, freq_nominal, time_ms, threads, num_ld, num_st, inst, FP_factor, precision, VLEN, interleaved)
                    
                if (test_type == 'L2' or test_type == 'roofline') and l2_size > 0:
                    #Run L2 Test

                    os.system("./Bench/Bench -test MEM -num_LD " + str(num_ld) + " -num_ST " + str(num_st) + " -precision " + precision + " -num_rep " + str(num_reps["L2"]) + " -Vlen " + str(VLEN) + " -LMUL " + str(LMUL))
                    
                    if test_type == 'roofline' and l1_size > 0:
                        no_freq_measure = 1
                    
                    if(interleaved):
                        result = subprocess.run(["./bin/test", "-threads", str(threads), "-freq", str(freq), "-measure_freq", str(no_freq_measure), "--interleaved"], stdout=subprocess.PIPE)
                    else:
                        result = subprocess.run(["./bin/test", "-threads", str(threads), "-freq", str(freq), "-measure_freq", str(no_freq_measure)], stdout=subprocess.PIPE)
                    
                    out = result.stdout.decode('utf-8').split(',')
                    inner_loop_reps = float(out[1])
                    if no_freq_measure == 0:
                        freq_real = float(out[2])
                    if isa not in ["neon", "armscalar", "riscvscalar", "rvv0.7", "rvv1.0"]:
                        cycles = float(out[0])
                        if no_freq_measure == 0:
                            freq_nominal = float(out[3])
                        data['L2'] = float((threads*num_reps["L2"]*(num_ld+num_st)*mem_inst_size[isa][precision]*freq_real*inner_loop_reps)/(cycles*float(freq_real/freq_nominal)))
                        data_cycles['L2'] = float((threads*num_reps["L2"]*(num_ld+num_st)*inner_loop_reps)/(cycles*float(freq_real/freq_nominal)))
                        time_ms = float (cycles / (freq_nominal * 1e6))
                    else:
                        time_ms = float(out[0])
                        data['L2'] = (float((threads*num_reps["L2"]*(num_ld+num_st)*mem_inst_size[isa][precision]*VLEN*LMUL*inner_loop_reps)/(1000000000))/((time_ms/1000)))
                        data_cycles['L2'] = float((threads*num_reps["L2"]*(num_ld+num_st)*LMUL*inner_loop_reps)/((time_ms/1000)*freq_real*1000000000))
                    if (verbose > 1):
                        print_results(isa, "L2", data["L2"], data_cycles["L2"], num_reps["L2"], test_size["L2"], inner_loop_reps, freq_real, cycles, freq_nominal, time_ms, threads, num_ld, num_st, inst, FP_factor, precision, VLEN, interleaved)

                if ((test_type == 'L3' or test_type == 'roofline') and int(l3_size) > 0):
                    #Run L3 Test 

                    os.system("./Bench/Bench -test MEM -num_LD " + str(num_ld) + " -num_ST " + str(num_st) + " -precision " + precision + " -num_rep " + str(num_reps["L3"]) + " -Vlen " + str(VLEN) + " -LMUL " + str(LMUL))
                    
                    if test_type == 'roofline' and (l1_size > 0 or l2_size > 0):
                        no_freq_measure = 1

                    if(interleaved):
                        result = subprocess.run(["./bin/test", "-threads", str(threads), "-freq", str(freq), "-measure_freq", str(no_freq_measure), "--interleaved"], stdout=subprocess.PIPE)
                    else:
                        result = subprocess.run(["./bin/test", "-threads", str(threads), "-freq", str(freq), "-measure_freq", str(no_freq_measure)], stdout=subprocess.PIPE)
                
                    out = result.stdout.decode('utf-8').split(',')
                    inner_loop_reps = float(out[1])
                    if no_freq_measure == 0:
                        freq_real = float(out[2])
                    if (isa != "neon" and isa != "armscalar" and isa != "riscvscalar" and isa != "rvv0.7", "rvv1.0"):
                        cycles = float(out[0])
                        if no_freq_measure == 0:
                            freq_nominal = float(out[3])
                        data['L3'] = float((threads*num_reps["L3"]*(num_ld+num_st)*mem_inst_size[isa][precision]*freq_real*inner_loop_reps)/(cycles*float(freq_real/freq_nominal)))
                        data_cycles['L3'] = float((threads*num_reps["L3"]*(num_ld+num_st)*inner_loop_reps)/(cycles*float(freq_real/freq_nominal)))
                        time_ms = float (cycles / (freq_nominal * 1e6))
                    else:
                        time_ms = float(out[0])
                        data['L3'] = (float((threads*num_reps["L3"]*(num_ld+num_st)*mem_inst_size[isa][precision]*VLEN*LMUL*inner_loop_reps)/(1000000000))/((time_ms/1000)))
                        data_cycles['L3'] = float((threads*num_reps["L3"]*(num_ld+num_st)*LMUL*inner_loop_reps)/((time_ms/1000)*freq_real*1000000000))
                    if (verbose > 1):
                        print_results(isa, "L3", data["L3"], data_cycles["L3"], num_reps["L3"], test_size["L3"], inner_loop_reps, freq_real, cycles, freq_nominal, time_ms, threads, num_ld, num_st, inst, FP_factor, precision, VLEN, interleaved)

                if (test_type == 'DRAM' or test_type == 'roofline'):
                    #Run DRAM Test
                
                    os.system("./Bench/Bench -test MEM -num_LD " + str(num_ld) + " -num_ST " + str(num_st) + " -precision " + precision + " -num_rep " + str(num_reps["DRAM"]) + " -Vlen " + str(VLEN) + " -LMUL " + str(LMUL))
                    
                    if test_type == 'roofline' and (l1_size > 0 or l2_size > 0 or l3_size > 0):
                        no_freq_measure = 1

                    if(interleaved):
                        result = subprocess.run(["./bin/test", "-threads", str(threads), "-freq", str(freq), "-measure_freq", str(no_freq_measure), "--interleaved"], stdout=subprocess.PIPE)
                    else:
                        result = subprocess.run(["./bin/test", "-threads", str(threads), "-freq", str(freq), "-measure_freq", str(no_freq_measure)], stdout=subprocess.PIPE)
                    
                    out = result.stdout.decode('utf-8').split(',')
                    inner_loop_reps = float(out[1])
                    if no_freq_measure == 0:
                        freq_real = float(out[2])
                    if (isa != "neon" and isa != "armscalar" and isa != "riscvscalar" and isa != "rvv0.7", "rvv1.0"):
                        cycles = float(out[0])
                        if no_freq_measure == 0:
                            freq_nominal = float(out[3])
                        data['DRAM'] = float((threads*num_reps["DRAM"]*(num_ld+num_st)*mem_inst_size[isa][precision]*freq_real*inner_loop_reps)/(cycles*float(freq_real/freq_nominal)))
                        data_cycles['DRAM'] = float((threads*num_reps["DRAM"]*(num_ld+num_st)*inner_loop_reps)/(cycles*float(freq_real/freq_nominal)))
                        time_ms = float (cycles / (freq_nominal * 1e6))
                    else:
                        time_ms = float(out[0])
                        data['DRAM'] = (float((threads*num_reps["DRAM"]*(num_ld+num_st)*mem_inst_size[isa][precision]*VLEN*LMUL*inner_loop_reps)/(1000000000))/((time_ms/1000)))
                        data_cycles['DRAM'] = float((threads*num_reps["DRAM"]*(num_ld+num_st)*LMUL*inner_loop_reps)/((time_ms/1000)*freq_real*1000000000))
                    if (verbose > 1):
                        print_results(isa, "DRAM", data["DRAM"], data_cycles["DRAM"], num_reps["DRAM"], test_size["DRAM"], inner_loop_reps, freq_real, cycles, freq_nominal, time_ms, threads, num_ld, num_st, inst, FP_factor, precision, VLEN, interleaved)

                if (test_type == 'FP' or test_type == 'roofline'):
                    #Run FP Test
                
                    os.system("./Bench/Bench -test FLOPS -op " + inst + " -precision " + precision + " -fp " + str(num_reps["FP"]) + " -Vlen " + str(VLEN) + " -LMUL " + str(LMUL))
                    
                    if test_type == 'roofline' and (l1_size > 0 or l2_size > 0 or l3_size > 0 or dram_bytes > 0):
                        no_freq_measure = 1

                    if(interleaved):
                        result = subprocess.run(["./bin/test", "-threads", str(threads), "-freq", str(freq), "-measure_freq", str(no_freq_measure), "--interleaved"], stdout=subprocess.PIPE)
                    else:
                        result = subprocess.run(["./bin/test", "-threads", str(threads), "-freq", str(freq), "-measure_freq", str(no_freq_measure)], stdout=subprocess.PIPE)
                    
                    out = result.stdout.decode('utf-8').split(',')
                    inner_loop_reps = float(out[1])
                    if no_freq_measure == 0:
                        freq_real = float(out[2])
                    if (isa != "neon" and isa != "armscalar" and isa != "riscvscalar" and isa != "rvv0.7", "rvv1.0"):
                        cycles = float(out[0])
                        if no_freq_measure == 0:
                            freq_nominal = float(out[3])
                        data['FP'] = float(threads*num_reps["FP"]*FP_factor*ops_fp[isa][precision]*freq_real*inner_loop_reps)/(cycles*float(freq_real/freq_nominal))
                        data_cycles['FP'] = float((threads*num_reps["FP"]*inner_loop_reps)/(cycles*float(freq_real/freq_nominal)))
                        time_ms = float (cycles / (freq_nominal * 1e6))
                        
                    else:
                        time_ms = float(out[0])
                        data['FP'] = float((threads*num_reps["FP"]*FP_factor*ops_fp[isa][precision]*inner_loop_reps*VLEN)/(1000000000))/((time_ms/1000))
                        data_cycles['FP'] = float((threads*num_reps["FP"]*inner_loop_reps)/((time_ms/1000)*freq_real*1000000000))
                    if (verbose > 1):
                        print_results(isa, "FP", data["FP"], data_cycles["FP"], num_reps["FP"], test_size["DRAM"], inner_loop_reps, freq_real, cycles, freq_nominal, time_ms, threads, num_ld, num_st, inst, FP_factor, precision, VLEN, interleaved)

                
                if (test_type == 'roofline' and inst != 'fma'):
                    #Run FP FMA Test

                    inst_fma = 'fma'
                    FP_factor = 2
                
                    os.system("./Bench/Bench -test FLOPS -op " + inst_fma + " -precision " + precision + " -fp " + str(num_reps["FP_FMA"]) + " -Vlen " + str(VLEN) + " -LMUL " + str(LMUL))
                    
                    if test_type == 'roofline' and (l1_size > 0 or l2_size > 0 or l3_size > 0 or dram_bytes > 0):
                        no_freq_measure = 1

                    if(interleaved):
                        result = subprocess.run(["./bin/test", "-threads", str(threads), "-freq", str(freq), "-measure_freq", str(no_freq_measure), "--interleaved"], stdout=subprocess.PIPE)
                    else:
                        result = subprocess.run(["./bin/test", "-threads", str(threads), "-freq", str(freq), "-measure_freq", str(no_freq_measure)], stdout=subprocess.PIPE)
                    
                    out = result.stdout.decode('utf-8').split(',')

                    inner_loop_reps = float(out[1])
                    if no_freq_measure == 0:
                        freq_real = float(out[2])
                    if (isa != "neon" and isa != "armscalar" and isa != "riscvscalar" and isa != "rvv0.7", "rvv1.0"):
                        cycles = float(out[0])
                        if no_freq_measure == 0:
                            freq_nominal = float(out[3])
                        data['FP_FMA'] = float(threads*num_reps["FP_FMA"]*FP_factor*ops_fp[isa][precision]*freq_real*inner_loop_reps)/(cycles*float(freq_real/freq_nominal))
                        data_cycles['FP_FMA'] = float((threads*num_reps["FP_FMA"]*inner_loop_reps)/(cycles*float(freq_real/freq_nominal)))
                        time_ms = float (cycles / (freq_nominal * 1e6))
                    else:
                        time_ms = float(out[0])
                        data['FP_FMA'] = float((threads*num_reps["FP_FMA"]*FP_factor*ops_fp[isa][precision]*inner_loop_reps*VLEN)/(1000000000))/((time_ms/1000))
                        data_cycles['FP_FMA'] = float((threads*num_reps["FP_FMA"]*inner_loop_reps)/((time_ms/1000)*freq_real*1000000000))
                    if (verbose > 1):
                        print_results(isa, "FP_FMA", data["FP_FMA"], data_cycles["FP_FMA"], num_reps["FP_FMA"], test_size["DRAM"], inner_loop_reps, freq_real, cycles, freq_nominal, time_ms, threads, num_ld, num_st, inst_fma, FP_factor, precision, VLEN, interleaved)

                #Save Results if a full Roofline test is done
                if (test_type == 'roofline'):
                    if(os.path.isdir('Results') == False):
                        os.mkdir('Results')
                    if(os.path.isdir('Results/Roofline') == False):
                        os.mkdir('Results/Roofline')
                
                    ct = datetime.datetime.now()
                    if plot:
                        date = ct.strftime('%Y-%m-%d_%H-%M-%S')
                        plot_roofline(name, data, date, isa, precision, threads, num_ld, num_st, inst, interleaved)
                    
                    date = ct.strftime('%Y-%m-%d %H:%M:%S')
                    update_csv(name, "Roofline", data, data_cycles, date, isa, precision, threads, num_ld, num_st, inst, interleaved, l1_size, l2_size, l3_size, dram_bytes)

#Run Memory Bandwidth tests
def run_memory(name, freq, set_freq, l1_size, l2_size, l3_size, isa_set, precision_set, num_ld, num_st, threads_set, interleaved, verbose, no_freq_measure, VLEN, tl1, plot, LMUL):
    if interleaved:
        inter = "Yes"
    else:
        inter = "No"

    isa_set, l1_size, l2_size, l3_size, VLEN, LMUL = check_hardware(isa_set, freq, set_freq, verbose, precision_set, l1_size, l2_size, l3_size, VLEN, LMUL)
    
    VLEN_aux = VLEN
    LMUL_aux = LMUL
    freq_nominal = freq
    freq_real = freq

    if verbose == 1:
        print("------------------------------")
        print("Running Benchmarks for the Following Threads Counts:", threads_set)
        print("On the Following ISA extensions: ", isa_set)
        print("Using the Following Precisions:", precision_set)
        print("------------------------------")
        

    for threads in threads_set:
        for isa in isa_set:
            for precision in precision_set:
                if verbose > 1:
                    print("------------------------------")
                    print("Running Benchmarks for the Following Threads Counts:", threads_set)
                    print("On the Following ISA extensions: ", isa_set)
                    print("Using the Following Precisions:", precision_set)
                    print("Now Testing:", threads, "Threads on", isa, "with", precision)
                if verbose == 1:
                    print("------------------------------")
                    print("Now Testing:", threads, "Threads on", isa, "with", precision)
                
                if VLEN_aux > 1 and precision == "dp":
                    VLEN = VLEN_aux
                    LMUL = LMUL_aux
                if VLEN_aux > 1 and precision == "sp":
                    VLEN = VLEN_aux * 2
                    LMUL = LMUL_aux
                if isa not in ["rvv0.7", "rvv1.0"]:
                    VLEN = 1
                    LMUL = 1
                if verbose > 2 and isa in ["rvv0.7", "rvv1.0"]:
                    print("VLEN IS:", VLEN, "| LMUL IS:", LMUL)
                if verbose > 0:
                    print("------------------------------")
                    
                Gbps = [0] * len(test_sizes)
                InstCycle = [0] * len(test_sizes)

                os.system("make clean && make isa=" + isa)

                num_reps = int(int(l1_size)*1024/(tl1*mem_inst_size[isa][precision]*(num_ld+num_st)*VLEN*LMUL))

                os.system("./Bench/Bench -test MEM -num_LD " + str(num_ld) + " -num_ST " + str(num_st) + " -precision " + precision + " -num_rep " + str(num_reps) + " -Vlen " + str(VLEN) + " -LMUL " + str(LMUL))
                
                no_freq_measure = 0

                if(interleaved):
                    result = subprocess.run(["./bin/test", "-threads", str(threads), "-freq", str(freq), "-measure_freq", str(no_freq_measure), "--interleaved"], stdout=subprocess.PIPE)
                else:
                    result = subprocess.run(["./bin/test", "-threads", str(threads), "-freq", str(freq), "-measure_freq", str(no_freq_measure)], stdout=subprocess.PIPE)

                no_freq_measure = 1
                out = result.stdout.decode('utf-8').split(',')

                freq_real = float(out[2])
                if isa not in ["neon", "armscalar", "riscvscalar", "rvv0.7", "rvv1.0"]:
                    freq_nominal = float(out[3])

                i=0
                for size in test_sizes:
                    num_reps = int(size*1024/(mem_inst_size[isa][precision]*(num_ld+num_st)*VLEN*LMUL))
                    

                    os.system("./Bench/Bench -test MEM -num_LD " + str(num_ld) + " -num_ST " + str(num_st) + " -precision " + precision + " -num_rep " + str(num_reps) + " -Vlen " + str(VLEN) + " -LMUL " + str(LMUL))
                
                    if(interleaved):
                        result = subprocess.run(["./bin/test", "-threads", str(threads), "-freq", str(freq_real), "-measure_freq", str(no_freq_measure), "--interleaved"], stdout=subprocess.PIPE)
                    else:
                        result = subprocess.run(["./bin/test", "-threads", str(threads), "-freq", str(freq_real), "-measure_freq", str(no_freq_measure)], stdout=subprocess.PIPE)

                    out = result.stdout.decode('utf-8').split(',')
                    inner_loop_reps = float(out[1])
                    if verbose > 2:
                        print(size, "Kb")
                    if isa not in ["neon", "armscalar", "riscvscalar", "rvv0.7", "rvv1.0"]:
                        cycles = float(out[0])
                        Gbps[i] = float((threads*num_reps*(num_ld+num_st)*mem_inst_size[isa][precision]*freq_real*inner_loop_reps)/(cycles*float(freq_real/freq_nominal)))
                        InstCycle[i] = float((threads*num_reps*(num_ld+num_st)*inner_loop_reps)/(cycles*float(freq_real/freq_nominal)))
                    else:
                        time_ms = float(out[0])
                        Gbps[i] = (float((threads*num_reps*(num_ld+num_st)*mem_inst_size[isa][precision]*VLEN*LMUL*inner_loop_reps)/(1000000000))/((time_ms/1000)))
                        InstCycle[i] = float((threads*num_reps*(num_ld+num_st)*LMUL*inner_loop_reps)/((time_ms/1000)*freq_real*1000000000))
                    i += 1
                i = 0
                if (verbose > 1):
                    print("ISA:", isa,  "| Number of Threads:", threads, " | Precision:", precision, "| Interleaved:", inter, "| Number of Loads:", num_ld, "| Number of Stores:", num_st, "| Memory Instruction Size:", mem_inst_size[isa][precision]*VLEN)
                    if isa not in ["neon", "armscalar", "riscvscalar", "rvv0.7", "rvv1.0"]:
                        print("Max Recorded Frequency (GHz):", freq_real, "| Nominal Frequency (GHz):", freq_nominal, "| Actual Frequency to Nominal Frequency Ratio:", float(freq_real/freq_nominal))
                    else:
                        print("Max Recorded Frequency (GHz):", freq_real)
                    for size in test_sizes:
                        print("Size:", size, "Kb | Gbps:", custom_round(Gbps[i]), "| Instructions per Cycle:", custom_round(InstCycle[i]))
                        i += 1

                if(os.path.isdir('Results') == False):
                    os.mkdir('Results')
                if(os.path.isdir('Results/MemoryCurve') == False):
                    os.mkdir('Results/MemoryCurve')

                ct = datetime.datetime.now()
                date = ct.strftime('%Y-%m-%d %H:%M:%S')
                update_memory_csv(Gbps, InstCycle, date, name, l1_size, l2_size, l3_size, isa, precision, num_ld, num_st, threads, interleaved)
                
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
                        plt.savefig('Results/MemoryCurve/' + name + '_Memory_Curve_' + date + "_" + isa + "_" + str(precision) + "_" + str(threads) + "_Threads_" + str(num_ld) + "Load_" + str(num_st) + "Store_" + "Interleaved" + '.svg')
                    else:
                        plt.savefig('Results/MemoryCurve/' + name + '_Memory_Curve_' + date + "_" + isa + "_" + str(precision) + "_" + str(threads) + "_Threads_" + str(num_ld) + "Load_" + str(num_st) + "Store" + '.svg')
                    
def update_memory_csv(Gbps, InstCycle, date, name, l1_size, l2_size, l3_size, isa, precision, num_ld, num_st, threads, interleaved):

    csv_path = f"./Results/MemoryCurve/{name}_Memory_Curve.csv"
    
    #Concatenate each test size with itself
    duplicated_test_sizes = [size for size in test_sizes for _ in range(2)]

    secondary_headers = ['Name:', name, 'L1 Size:', l1_size, 'L2 Size:', l2_size, 'L3 Size:', l3_size] + duplicated_test_sizes
    primary_headers = ['Date', 'ISA', 'Precision', 'Threads', 'Loads', 'Stores', 'Interleaved'] + [''] + ["Gbps", "I/Cycle"] * len(test_sizes)

    #Check if the file exists
    if os.path.exists(csv_path):
        mode = 'a'  #Append mode
    else:
        mode = 'w'  #Write mode

    with open(csv_path, mode, newline='') as csvfile:
        writer = csv.writer(csvfile)
        if mode == 'w':  # Ifwriting mode, write header
            writer.writerow(secondary_headers)
            writer.writerow(primary_headers)
        #Write the results to CSV
        results = [date, isa, precision, threads, num_ld, num_st, "Yes" if interleaved else "No"] + ['']
        for gbps, inst_cycle in zip(Gbps, InstCycle):
            results.append(custom_round(gbps))
            results.append(custom_round(inst_cycle))
        writer.writerow(results)

#Run Mixed Benchmark
def run_mixed(name, freq, l1_size, l2_size, l3_size, inst, isa_set, precision_set, num_ld, num_st, num_fp, threads_set, interleaved, num_ops, dram_bytes, dram_auto, test_type, verbose, set_freq, no_freq_measure, VLEN, tl1, tl2, LMUL):
    
    isa_set, l1_size, l2_size, l3_size, VLEN, LMUL = check_hardware(isa_set, freq, set_freq, verbose, precision_set, l1_size, l2_size, l3_size, VLEN, LMUL)
    VLEN_aux = VLEN
    LMUL_aux = LMUL
    dram_bytes_aux = dram_bytes
    freq_nominal = freq
    freq_real = freq
    
    if verbose == 1:
        print("------------------------------")
        print("Running Benchmarks for the Following Threads Counts:", threads_set)
        print("On the Following ISA extensions: ", isa_set)
        print("Using the Following Precisions:", precision_set)
        print("------------------------------")
        

    for threads in threads_set:
        for isa in isa_set:
            for precision in precision_set:
                if verbose > 1:
                    print("------------------------------")
                    print("Running Benchmarks for the Following Threads Counts:", threads_set)
                    print("On the Following ISA extensions: ", isa_set)
                    print("Using the Following Precisions:", precision_set)
                    print("Now Testing:", threads, "Threads on", isa, "with", precision)
                if verbose == 1:
                    print("------------------------------")
                    print("Now Testing:", threads, "Threads on", isa, "with", precision)
                
                dram_bytes = dram_bytes_aux
                if VLEN_aux > 1 and precision == "dp":
                    VLEN = VLEN_aux
                    LMUL = LMUL_aux
                if VLEN_aux > 1 and precision == "sp":
                    VLEN = VLEN_aux * 2
                    LMUL = LMUL_aux
                if not isa in  ["rvv0.7", "rvv1.0"]:
                    VLEN = 1
                    LMUL = 1
                if verbose > 2 and isa in ["rvv0.7", "rvv1.0"]:
                    print("VLEN IS:", VLEN, "| LMUL IS:", LMUL)
        
                os.system("make clean && make isa=" + isa)
                if test_type == "mixedL1":
                    if l1_size > 0:
                        num_reps = int(int(l1_size)*1024/(tl1*mem_inst_size[isa][precision]*(num_ld+num_st)*VLEN*LMUL))
                        test_size = (int(l1_size))/tl1
                    else:
                        print("WARNING: No L1 Size Found, you can use the -l1 <l1_size> argument, or a configuration file to specify it.")
                        return
                elif test_type == "mixedL2":
                    if l2_size > 0:
                        num_reps = int(1024*int(l2_size)/tl2)/(mem_inst_size[isa][precision]*(num_ld+num_st)*VLEN*LMUL)
                        test_size = int(int(l2_size)/tl2)
                    else:
                        print("WARNING: No L2 Size Found, you can use the -l2 <l2_size> argument, or a configuration file to specify it.")
                        return
                elif test_type == "mixedL3":
                    if l3_size > 0:
                        num_reps = int(1024*(int(l2_size)*threads + (int(l3_size) - int(l2_size)*threads)/2)/(threads*mem_inst_size[isa][precision]*(num_ld+num_st)*VLEN*LMUL))
                        test_size = int((int(l2_size)*threads + (int(l3_size) - int(l2_size)*threads)/2)/threads)
                    else:
                        print("WARNING: No L3 Size Found, you can use the -l3 <l3_size> argument, or a configuration file to specify it.")
                        return
                elif test_type == "mixedDRAM":
                    if (dram_auto) and ((int(dram_bytes)/threads) < (int(l3_size)*2)):
                        num_reps = int((int(l3_size)*2)*1024/(mem_inst_size[isa][precision]*(num_ld+num_st)*VLEN*LMUL))
                        test_size = int((int(l3_size)*2))
                        dram_bytes = int(l3_size)*2*threads
                    else:
                        num_reps = int(dram_bytes*1024/(threads*mem_inst_size[isa][precision]*(num_ld+num_st)*VLEN*LMUL))
                        test_size = int(dram_bytes/(threads))

                        if int(test_size) <= int(l3_size) and verbose > 0:
                            print("WARNING: DRAM test size per thread is not sufficient to guarantee best results, to guarantee best results consider changing the default test size.")
                            print("By using --dram_bytes", int(l3_size)*2*int(threads), "(", custom_round(float((int(l3_size)*2*int(threads))/1048576)), "Gb) the minimum test size necessary for", threads, "threads is achieved, using the --dram_auto flag will automatically apply this adjustement.")
                    if verbose > 2:
                        print("DRAM Test Size per Thread:", test_size, "Kb | L3 Size:", l3_size, "Kb | Total DRAM Test Size:", custom_round(float((test_size*threads)/1048576)), "Gb")
                
                if verbose > 0:
                    print("------------------------------")

                if inst == "fma":
                    FP_factor = 2
                else:
                    FP_factor = 1

                os.system("./Bench/Bench -test MIXED -num_LD " + str(num_ld) + " -num_ST " + str(num_st) + " -num_FP " + str(num_fp) + " -op " + inst + " -precision " + precision + " -num_rep " + str(num_reps) + " -Vlen " + str(VLEN) + " -LMUL " + str(LMUL))

                if(interleaved):
                    result = subprocess.run(["./bin/test", "-threads", str(threads), "-freq", str(freq), "-measure_freq", str(no_freq_measure), "--interleaved"], stdout=subprocess.PIPE)
                    inter = "Yes"
                else:
                    result = subprocess.run(["./bin/test", "-threads", str(threads), "-freq", str(freq), "-measure_freq", str(no_freq_measure)], stdout=subprocess.PIPE)
                    inter = "No"
                out = result.stdout.decode('utf-8').split(',')

                inner_loop_reps = float(out[1])
                freq_real = float(out[2])

                if isa not in ["neon", "armscalar", "riscvscalar", "rvv0.7", "rvv1.0"]:
                    cycles = float(out[0])
                    freq_nominal = float(out[3])
                    gflops = float(threads*num_reps*num_fp*FP_factor*ops_fp[isa][precision]*freq_real*inner_loop_reps)/(cycles*float(freq_real/freq_nominal))
                    ai = float((num_fp*FP_factor*ops_fp[isa][precision])/((num_ld+num_st)*mem_inst_size[isa][precision]))
                    bandwidth = float((threads*num_reps*(num_ld+num_st)*mem_inst_size[isa][precision]*freq_real*inner_loop_reps)/(cycles*float(freq_real/freq_nominal)))
                    time_ms = float (cycles / (freq_nominal * 1e6))
                    if (verbose > 1):
                        print ("---------RESULTS-----------")
                        print ("ISA:", isa,  "| Number of Threads:", threads, "| Allocated Size:", test_size, "Kb | Precision:", precision, "| Interleaved:", inter, "| Number of Loads:", num_ld, "| Number of Stores:", num_st, "| Number of FP:", num_fp, "| Memory Instruction Size:", mem_inst_size[isa][precision]*VLEN, "| FP Operations per Instruction:", (FP_factor*ops_fp[isa][precision])*VLEN, "| Total Inner Loop Reps:", int(inner_loop_reps),  "| Number of Reps:", num_reps)
                        print ("Best Average Cycles:", cycles, "| Best Average Time (in ms):", time_ms)
                        print ("Instructions Per Cycle:", threads*num_reps*(num_ld+num_st+num_fp)*inner_loop_reps/(cycles*float(freq_real/freq_nominal)))
                        print ("FP Instructions Per Cycle:", threads*num_reps*(num_fp)*inner_loop_reps/(cycles*float(freq_real/freq_nominal)))
                        print ("Memory Instructions per Cycle:", threads*num_reps*(num_ld+num_st)*inner_loop_reps/(cycles*float(freq_real/freq_nominal)))
                        print ("Bandwidth (Gbps):", bandwidth)
                        print ("GFLOPS::", gflops)
                        print ("Total Flops:", int(num_fp*FP_factor*ops_fp[isa][precision]*num_reps*inner_loop_reps))
                        print ("Total Bytes:", int(((num_ld+num_st)*mem_inst_size[isa][precision])*num_reps*inner_loop_reps))
                        print ("Arithmetic Intensity:", ai)
                        print ("Max Recorded Frequency (GHz):", freq_real, "| Nominal Frequency (GHz):", freq_nominal, "| Actual Frequency to Nominal Frequency Ratio:", float(freq_real/freq_nominal))
                        print ("------------------------------")
                else:
                    time_ms = float(out[0])
                    gflops = float((threads*num_reps*num_fp*FP_factor*ops_fp[isa][precision]*VLEN*LMUL*inner_loop_reps)/(1000000000))/((time_ms/1000))
                    ai = float((num_fp*FP_factor*ops_fp[isa][precision])/((num_ld+num_st)*mem_inst_size[isa][precision]))
                    bandwidth = (float((threads*num_reps*(num_ld+num_st)*mem_inst_size[isa][precision]*VLEN*LMUL*inner_loop_reps)/(1000000000))/((time_ms/1000)))
                    if (verbose > 1):
                        print ("---------RESULTS-----------")
                        print ("ISA:", isa,  "| Number of Threads:", threads, "| Allocated Size:", test_size, "Kb | Precision:", precision, "| Interleaved:", inter, "| Number of Loads:", num_ld, "| Number of Stores:", num_st, "| Number of FP:", num_fp, "| Memory Instruction Size:", mem_inst_size[isa][precision]*VLEN, "| FP Operations per Instruction:", (FP_factor*ops_fp[isa][precision])*VLEN, "| Total Inner Loop Reps:", int(inner_loop_reps),  "| Number of Reps:", num_reps)
                        print ("Best Average Time (in ms):", time_ms)
                        print ("Instructions Per Cycle:", (threads*num_reps*(num_ld+num_st+num_fp)*LMUL*inner_loop_reps)/((time_ms/1000)*freq_real*1000000000))
                        print ("FP Instructions Per Cycle:", (threads*num_reps*(num_fp)*LMUL*inner_loop_reps)/((time_ms/1000)*freq_real*1000000000))
                        print ("Memory Instructions Per Cycle:", (threads*num_reps*(num_ld+num_st)*LMUL*inner_loop_reps)/((time_ms/1000)*freq_real*1000000000))
                        print ("Bandwidth (Gbps):", bandwidth)
                        print ("GFLOPS::", gflops)
                        print ("Total Flops:", int(num_fp*FP_factor*ops_fp[isa][precision]*num_reps*inner_loop_reps*VLEN*LMUL))
                        print ("Total Bytes:", int(((num_ld+num_st)*mem_inst_size[isa][precision])*num_reps*inner_loop_reps*VLEN*LMUL))
                        print ("Arithmetic Intensity:", ai)
                        print ("Max Recorded Frequency (GHz):", freq_real)
                        print ("------------------------------")
                
                ct = datetime.datetime.now()
                date = ct.strftime('%Y-%m-%d %H:%M:%S')
                test_details = test_type + "_" + str(num_fp) + "FP_" + str(num_ld) + "LD_" + str(num_st) + "ST_" + inst
                DBI_AI_Calculator.update_csv(name, "/home/mixed", gflops, ai, bandwidth, time_ms, test_details, date, isa, precision, threads, "MIX")

def main():
    parser = argparse.ArgumentParser(description='Script to run micro-benchmarks to construct Cache-Aware Roofline Model')
    parser.add_argument('--test', default='roofline', nargs='?', choices=['FP', 'L1', 'L2', 'L3', 'DRAM', 'roofline', 'MEM', 'mixedL1', 'mixedL2', 'mixedL3', 'mixedDRAM'], help='Type of the test. Roofline test measures the bandwidth of the different memory levels and FP Performance, MEM test measures the bandwidth of various memory sizes, mixed test measures bandwidth and FP performance for a combination of memory acceses (to L1, L2, L3, or DRAM) and FP operations (Default: roofline)')
    parser.add_argument('--no_freq_measure',  dest='no_freq_measure', action='store_const', const=1, default=0, help='Measure CPU frequency or not')
    parser.add_argument('--freq', default='2.0', nargs='?', type = float, help='Expected/Desired CPU frequency during test, (if no_freq_measure or set_freq is enabled)')
    parser.add_argument('--set_freq',  dest='set_freq', action='store_const', const=1, default=0, help='Set Processor frequency to indicated one (x86 only)')
    parser.add_argument('--name', default='unnamed', nargs='?', type = str, help='Name for results file (if not using config file)')
    parser.add_argument('-v', '--verbose', default=1, nargs='?', type = int, choices=[0, 1, 2, 3], help='Level of terminal output (0 -> No Output 1 -> Only ISA Errors and Test Details, 2 -> Intermediate Test Results, 3 -> Configuration Values Selected/Detected)')
    parser.add_argument('--inst', default='add', nargs='?', choices=['add', 'mul', 'div', 'fma'], help='FP Instruction (Default: add), FMA performance is measured by default too.')
    parser.add_argument('-vl', '--vector_length',  default=1, nargs='?', type = int, help='Vector Length in double-precision elements for RVV configuration (Default: 1)')
    parser.add_argument('-vlmul', '--vector_lmul', default=1, nargs='?', type = int, choices=[1, 2, 4, 8], help='Vector Register Grouping for RVV configuration (Default: 1 for FP benchmarks, 8 for memory/mixed benchmarks)')
    parser.add_argument('--isa', default=['auto'], nargs='+', choices=['avx512', 'avx2', 'sse', 'scalar', 'neon', 'armscalar', 'riscvscalar', 'rvv0.7', 'rvv1.0', 'auto'], help='set of ISAs to test, if auto will test all available ISAs (Default: auto)')
    parser.add_argument('-p', '--precision', default=['dp'], nargs='+', choices=['dp', 'sp'], help='Data Precision (Default: dp)')
    parser.add_argument('-ldst', '--ld_st_ratio',  default=2, nargs='?', type = int, help='Load/Store Ratio (Default: 2)')
    parser.add_argument('-fpldst', '--fp_ld_st_ratio',  default=1, nargs='?', type = int, help='FP to Load/Store Ratio, for mixed test (Default: 1)')
    parser.add_argument('--only_ld',  dest='only_ld', action='store_const', const=1, default=0, help='Run only loads in mem test (ld_st_ratio is ignored)')
    parser.add_argument('--only_st',  dest='only_st', action='store_const', const=1, default=0, help='Run only stores in mem test (ld_st_ratio is ignored)')
    parser.add_argument('config', nargs='?', help='Path for the system configuration file')
    parser.add_argument('-t', '--threads', default=[1], nargs='+', type = int, help='Set of number of threads for the micro-benchmarking, insert multiple thread valus by spacing them, no commas (Default: [1])')
    parser.add_argument('-i', '--interleaved',  dest='interleaved', action='store_const', const=1, default=0, help='For thread binding when cores are interleaved between NUMA domains (Default: 0)')
    parser.add_argument('-ops', '--num_ops',  default=32768, nargs='?', type = int, help='Number of FP operations to be used in FP test (Default: 32768)')
    parser.add_argument('--dram_bytes',  default=524288, nargs='?', type = int, help='Size of the array for the DRAM test in KiB (Default: 524288 (512 MiB))')
    parser.add_argument('--dram_auto',  dest='dram_auto', action='store_const', const=1, default=0, help='Automatically calculate the DRAM test size needed to ensure data does not fit in L3, can require a lot of memory in some cases, make sure it fits in the DRAM of your system (Default: 0)')
    parser.add_argument('--plot',  dest='plot', action='store_const', const=1, default=0, help='Create CARM plot SVG for each test result')

    parser.add_argument('-tl1', '--threads_per_l1',  default=2, nargs='?', type = int, help='Expected amount of threads that will have to share the same L1 cache (Default: 2)')
    parser.add_argument('-tl2', '--threads_per_l2',  default=2, nargs='?', type = int, help='Expected amount of threads that will have to share the same L2 cache (Default: 2)')
    parser.add_argument('-l1', '--l1_size',  default=0, nargs='?', type = int, help='L1 size per core')
    parser.add_argument('-l2', '--l2_size',  default=0, nargs='?', type = int, help='L2 size per core')
    parser.add_argument('-l3', '--l3_size',  default=0, nargs='?', type = int, help='L3 total size')

    args = parser.parse_args()
    if (args.verbose == 3):
        print(args)
    l1_size = args.l1_size
    l2_size = args.l2_size
    l3_size = args.l3_size
    freq = 0
    name = ''
    
    if (args.config != None):
        name, l1_size, l2_size, l3_size = read_config(args.config)

    freq = args.freq
    
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
        print("WARNING: When using auto ISA detection do not specify additional ISAs, setting ISA argument to just auto and removing additional ISAs.")
        isa_set = ["auto"]

    if args.test in ["mixedL1", "mixedL2", "mixedL3", "mixedDRAM"]:
        run_mixed(name, freq, l1_size, l2_size, l3_size, args.inst, isa_set, args.precision, num_ld, num_st, num_fp, args.threads, args.interleaved, args.num_ops, args.dram_bytes, args.dram_auto, args.test, args.verbose, args.set_freq, args.no_freq_measure, args.vector_length, args.threads_per_l1,  args.threads_per_l2, args.vector_lmul)
    elif args.test == 'MEM':
        run_memory(name, freq, args.set_freq, l1_size, l2_size, l3_size, isa_set, args.precision, num_ld, num_st, args.threads, args.interleaved, args.verbose, args.no_freq_measure, args.vector_length, args.threads_per_l1, args.plot, args.vector_lmul)
    else:
        run_roofline(name, freq, l1_size, l2_size, l3_size, args.inst, isa_set, args.precision, num_ld, num_st, args.threads, args.interleaved, args.num_ops, args.dram_bytes, args.dram_auto, args.test, args.verbose, args.set_freq, args.no_freq_measure, args.vector_length, args.threads_per_l1,  args.threads_per_l2, args.plot, args.vector_lmul)

if __name__ == "__main__":
    main()

