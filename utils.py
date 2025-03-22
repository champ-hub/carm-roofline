import os
import sys
import math
from decimal import Decimal
import argparse
import csv
plot_numpy = 1
try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    plot_numpy = None

CONFIG_FILE = "./config/auto_config/config.txt"

def read_library_path(tag):
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as file:
            for line in file:
                if line.strip() == "":
                    continue
                parts = line.strip().split("=")
                if len(parts) == 2:
                    key, value = parts
                    if key == tag:
                        return value
    return None

def write_library_path(tag, path):
    with open(CONFIG_FILE, "a") as file:
        file.write(f"{tag}={path}\n")

def make_power_of_two_ticks(min_val, max_val):
    #Ensure min_val > 0 to avoid log2 errors. If <=0, adjust logic as needed.
    min_val = max(min_val, 0.0000000001)
    max_val = max(max_val, 0.0000000001)
    start_exp = math.floor(math.log2(min_val))
    end_exp = math.ceil(math.log2(max_val))
    tickvals = [2**i for i in range(start_exp, end_exp+1)]
    ticktext = [f"2<sup>{i}</sup>" for i in range(start_exp, end_exp+1)]
    return tickvals, ticktext

def ensure_list(marker_dict, attr_name, default_value, n_points):
        #If marker[attr_name] doesn't exist or is not a list, convert it to a repeated list.
        if attr_name not in marker_dict:
            return [default_value] * n_points
        
        val = marker_dict[attr_name]
        if isinstance(val, list):
            return val
        else:
            return [val] * n_points

def custom_round(value, digits=4):
    if value == 0:
        return 0  #Directly return 0 if the value is 0
    elif abs(value) >= 1:
        #For numbers greater than or equal to 1, round normally
        return round(value, digits)
    else:
        #For numbers less than 1, find the position of the first non-zero digit after the decimal
        dec_val = Decimal(str(value))
        str_val = format(dec_val, 'f')
        if 'e' in str_val or 'E' in str_val:  #Check for scientific notation
            return round(value, digits)
        
        #Count positions until first non-zero digit after the decimal
        decimal_part = str_val.split('.')[1]
        leading_zeros = 0
        for char in decimal_part:
            if char == '0':
                leading_zeros += 1
            else:
                break
        
        #Adjust the number of digits based on the position of the first significant digit
        total_digits = digits + leading_zeros
        return round(value, total_digits)
    
def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid positive int value")
    return ivalue


def round_power_of_2(number):
    if number > 1:
        for i in range(1, int(number)):
            if (2 ** i > number):
                return 2 ** i
    else:
        return 1

def carm_eq(ai, bw, fp):
    import numpy as np
    return np.minimum(ai*bw, fp)

def parse_title_line(line):
    parts = line.split()
    title = {
        "name": parts[0],
        "isa": parts[2],
        "precision": parts[3],
        "threads": int(parts[4]),
        "load": int(parts[6]),
        "store": int(parts[8]),
        "inst": parts[10]
    }
    return title

#Read CARM results data
def read_roofline_data(filename):
    title = {}
    data = {}
    data_cycles = {}

    with open(filename, 'r') as file:
        title_line = file.readline().strip()
        title = parse_title_line(title_line)

        for line in file:
            if ':' not in line:
                continue

            label, value = line.strip().split(': ')

            if label == 'L1':
                data["L1"] = float(value)
            elif label == 'L2':
                data["L2"] = float(value)
            elif label == 'L3':
                data["L3"] = float(value)
            elif label == 'DRAM':
                data["DRAM"] = float(value)
            elif label == 'FP':
                data["FP"] = float(value)
            elif label == 'FP_FMA':
                data["FP_FMA"] = float(value)
            elif label == 'L1 Instruction Per Cycle':
                data_cycles["L1"] = float(value)
            elif label == 'L2 Instruction Per Cycle':
                data_cycles["L2"] = float(value)
            elif label == 'L3 Instruction Per Cycle':
                data_cycles["L3"] = float(value)
            elif label == 'DRAM Instruction Per Cycle':
                data_cycles["DRAM"] = float(value)
            elif label == 'FP Instruction Per Cycle':
                data_cycles["FP"] = float(value)
            elif label == 'FP_FMA Instruction Per Cycle':
                data_cycles["FP_FMA"] = float(value)

    return title, data, data_cycles


def read_data_from_files(directory, autochoice):
    #List all files with .out extension in the directory
    files = [file for file in os.listdir(directory) if file.endswith('.out') and 'roofline' in file]

    #Print the list of files with their indices
    print("Available files:")
    for i, file in enumerate(files):
        print(f"{i + 1}. {file}")

    #Prompt the user to select a file by its index
    if autochoice == 0:
        while True:
            try:
                choice = int(input("Enter the number corresponding to the file you want to read (or 0 to quit): "))
                if 1 <= choice <= len(files):
                    selected_file = files[choice - 1]
                    break
                elif choice == 0:
                    print("No Roofline will be drawn, terminating program.")
                    sys.exit(1)
                else:
                    print("Invalid choice. Please enter a number within the range.")
            except EOFError:
                print("No automatic roofline choice provided, no Roofline will be drawn. Exiting program.")
                sys.exit(1)
            except ValueError:
                print("Invalid input. Please enter a number.")
    else:
        choice = autochoice
        if 1 <= choice <= len(files):
            selected_file = files[choice - 1]
        else:
            print("Autochoice does not match any file. No Roofline will be drawn, terminating program.")
            sys.exit(1)

    return read_roofline_data(os.path.join(directory, selected_file))

def plot_roofline_with_dot(executable_path, exec_flops, exec_ai, choice, roi, date, method):

    title = {}
    data = {}
    data_cycles = {}
    if plot_numpy == None:
        print("No Matplotlib and/or Numpy found, in order to draw CARM graphs make sure to install them.")

    executable_name = os.path.basename(executable_path)

    min_ai = 0.015625
    min_flops = 0.25
    while exec_ai < min_ai :
        min_ai /= 2
    
    while exec_flops < min_flops :
        min_flops /= 2
    
    script_dir = os.path.dirname(os.path.abspath(__file__))

    #Construct the path to the build folder
    result_dir = os.path.join(script_dir, "carm_results/roofline")

    title, data, data_cycles = read_data_from_files(result_dir, choice)

    fig, ax = plt.subplots(figsize=(7.1875*1.5,3.75*1.5))
    plt.xlim(min_ai, 256)
    plt.ylim(min_flops, round_power_of_2(int(data["FP_FMA"])))
    ai = np.linspace(0.00390625, 256, num=200000)

    #Ploting Lines
    if title["inst"] == "fma":
        plt.plot(ai, carm_eq(ai, data["L1"], data["FP"]), 'k', lw = 3, label='L1')
        plt.plot(ai, carm_eq(ai, data["L2"], data["FP"]), 'grey', lw = 3, label='L2')
        plt.plot(ai, carm_eq(ai, data["L3"], data["FP"]), 'k', linestyle='dashed', lw = 3, label='L3')
        plt.plot(ai, carm_eq(ai, data["DRAM"], data["FP"]), 'k', linestyle='dotted', lw = 3, label='DRAM')
    else:
        plt.plot(ai, carm_eq(ai, data["L1"], data["FP_FMA"]), 'k', lw = 3, label='L1')
        plt.plot(ai, carm_eq(ai, data["L2"], data["FP_FMA"]), 'grey', lw = 3, label='L2')
        plt.plot(ai, carm_eq(ai, data["L3"], data["FP_FMA"]), 'k', linestyle='dashed', lw = 3, label='L3')
        plt.plot(ai, carm_eq(ai, data["DRAM"], data["FP_FMA"]), 'k', linestyle='dotted', lw = 3, label='DRAM')
        plt.plot(ai, carm_eq(ai, data["L1"], data["FP"]), 'k', linestyle='dashdot', lw = 3, label=title["inst"])
    
    #Plot dot at exec_gflops and exec_ai
    plt.scatter(exec_ai, exec_flops, color='red', label=executable_name, zorder=5)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if method == "dbi":
        if roi:
            plt.title(executable_name + " (" + title["name"] + ")" + ' DBI ROI CARM: ' + str(title["isa"]) + " " + str(title["precision"]) + " " + str(title["threads"]) + " Threads " + str(title["load"]) + " Load " + str(title["store"]) + " Store " + title["inst"], fontsize=18)
        else:
            plt.title(executable_name + " (" + title["name"] + ")" + ' DBI CARM: ' + str(title["isa"]) + " " + str(title["precision"]) + " " + str(title["threads"]) + " Threads " + str(title["load"]) + " Load " + str(title["store"]) + " Store " + title["inst"], fontsize=18)
    elif method == "pmu":
        plt.title(executable_name + " (" + title["name"] + ")" + ' PMU CARM: ' + str(title["isa"]) + " " + str(title["precision"]) + " " + str(title["threads"]) + " Threads " + str(title["load"]) + " Load " + str(title["store"]) + " Store " + title["inst"], fontsize=18)


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
    if(os.path.isdir('carm_results') == False):
            os.mkdir('carm_results')
    if(os.path.isdir('carm_results/applications') == False):
        os.mkdir('carm_results/applications')
    if method == "dbi":
        if roi:
            plt.savefig('carm_results/applications/' + executable_name + "_" + title["name"] + '_DBI_ROI_roofline_analysis_' + date + '_' + str(title["isa"]) + "_" + str(title["precision"]) + "_" + str(title["threads"]) + "_Threads_" + str(title["load"]) + "Load_" + str(title["store"]) + "Store_" + title["inst"] + '.svg')
        else:
            plt.savefig('carm_results/applications/' + executable_name + "_" + title["name"] + '_DBI_roofline_analysis_' + date + '_' + str(title["isa"]) + "_" + str(title["precision"]) + "_" + str(title["threads"]) + "_Threads_" + str(title["load"]) + "Load_" + str(title["store"]) + "Store_" + title["inst"] + '.svg')
    elif method == "pmu":
        plt.savefig('carm_results/applications/' + executable_name + "_" + title["name"] + '_PMU_roofline_analysis_' + date + '_' + str(title["isa"]) + "_" + str(title["precision"]) + "_" + str(title["threads"]) + "_Threads_" + str(title["load"]) + "Load_" + str(title["store"]) + "Store_" + title["inst"] + '.svg')

def update_csv(machine, executable_path, exec_flops, exec_ai, bandwidth, time, name, date, isa, precision, threads, method, VLEN, LMUL):

    csv_path = f"./carm_results/applications/{machine}_applications.csv"

    if name == "":
        name = os.path.basename(executable_path)

    if(os.path.isdir('carm_results') == False):
        os.mkdir('carm_results')
    if(os.path.isdir('carm_results/applications') == False):
        os.mkdir('carm_results/applications')
    
    if (isa in ["rvv0.7", "rvv1.0"]):
        isa = str(isa) + "_vl" + str(VLEN) + "_lmul" + str(LMUL)

    results = [
        date,
        method,
        name,
        isa,
        precision,
        threads,
        custom_round(exec_ai),
        custom_round(exec_flops),
        custom_round(bandwidth),
        custom_round(time)
    ]

    headers = ['Date', 'Method', 'Name', 'ISA', 'Precision', 'Threads', 'AI', 'Gflops', 'Bandwidth', 'Time']

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
            writer.writerow(headers)
            writer.writerow(results)