import argparse
import subprocess
import os
plot_numpy = 1
try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    plot_numpy = None
import datetime
import time
import math
import re
import sys
import json
import platform
import threading
import run
import csv

#Define a global set to store the names of JSON files that have been read
read_files = set()

def runPAPI_async(executable_path, additional_args):
    thread = threading.Thread(target=runPAPI, args=(executable_path, additional_args))
    thread.start()

#Check if PAPI is present
def check_PAPI_exists(PAPI_path):
    PAPI_src = os.path.join(PAPI_path, "src")
    if os.path.exists(PAPI_src):
        print(f"PAPI src folder found in: '{PAPI_path}'.")
    else:
        print(f"No PAPI src folder found in: '{PAPI_src}'.")
        sys.exit(1)

#Run PAPI with provided application
def runPAPI(executable_path, additional_args=None):
    additional_args = additional_args or []
    #Construct the command with the provided paths and additional arguments
    command = [executable_path, *additional_args]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    PAPI_output = os.path.join(script_dir, "PMU_Readings")
    os.environ["PAPI_OUTPUT_DIRECTORY"] = PAPI_output
    #Setup environment to test Memory Operations
    PAPI_Event = "PAPI_LST_INS"
    os.environ["PAPI_EVENTS"] = PAPI_Event

    print("\n------------------------------")
    print("Running Provided Application For Memory Instructions PMU Data\n")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("Error executing the command:", e)
    
    total_real_time_nsec_mem, total_papi_mem_ins, thread_count = analysePAPI(PAPI_Event)

    #Modify environment to test SP FP Operations
    PAPI_Event = "PAPI_SP_OPS"
    os.environ["PAPI_EVENTS"] = PAPI_Event
    print("------------------------------")
    print("Running Provided Application For SP FP Operations PMU Data\n")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("Error executing the command:", e)
    
    total_real_time_nsec_sp, total_papi_sp_ops, thread_count = analysePAPI(PAPI_Event)

    #Modify environment to test DP FP Operations
    PAPI_Event = "PAPI_DP_OPS"
    os.environ["PAPI_EVENTS"] = PAPI_Event

    print("------------------------------")
    print("Running Provided Application For DP FP Operations PMU Data\n")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("Error executing the command:", e)
    
    total_real_time_nsec_dp, total_papi_dp_ops, thread_count = analysePAPI(PAPI_Event)

    return float((total_real_time_nsec_mem + total_real_time_nsec_sp + total_real_time_nsec_dp)/3), total_papi_mem_ins, total_papi_sp_ops, total_papi_dp_ops, thread_count
    

def analysePAPI(PAPI_Event):
    global read_files

    script_dir = os.path.dirname(os.path.abspath(__file__))
    PAPI_output = os.path.join(script_dir, "PMU_Readings/papi_hl_output")
    files = os.listdir(PAPI_output)

    #Filter out files that have already been read
    new_files = [file for file in files if file not in read_files]

    #Ensure there's exactly one new file in the folder
    if len(new_files) != 1:
        print("Error: Folder should contain exactly one new file.")
        return None
    
    #Get the path of the new JSON file
    new_file_path = os.path.join(PAPI_output, new_files[0])
    
    #Ensure the file is a JSON file
    if not new_file_path.endswith('.json'):
        print("Error: File in folder is not a JSON file.")
        return None
    
    #Read and return the JSON data
    with open(new_file_path, 'r') as file:
        json_data = json.load(file)
        #Add the file name to the set of read files
        read_files.add(new_files[0])
    
    thread_count = len(json_data['threads'])
    total_papi_sp_ops = 0
    total_real_time_nsec = 0
    
    for thread_id, thread_info in json_data['threads'].items():
        regions = thread_info['regions']
        thread_papi_sp_ops = sum(int(region_info.get(PAPI_Event, 0)) for region_info in regions.values())
        thread_real_time_nsec = sum(int(region_info.get('real_time_nsec', 0)) for region_info in regions.values())
        total_papi_sp_ops += int(thread_papi_sp_ops)
        total_real_time_nsec += int(thread_real_time_nsec)
    total_real_time_nsec = total_real_time_nsec / thread_count

    return total_real_time_nsec, total_papi_sp_ops, thread_count

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

    print("Available files:")
    for i, file in enumerate(files):
        print(f"{i + 1}. {file}")

    if autochoice == 0:
    #Prompt the user to select a file by its index
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
    return read_roofline_data(os.path.join(directory, selected_file))

def carm_eq(ai, bw, fp):
    return np.minimum(ai*bw, fp)

def round_power_of_2(number):
    if number > 1:
        for i in range(1, int(number)):
            if (2 ** i > number):
                return 2 ** i
    else:
        return 1

def plot_roofline_with_dot(executable_path, exec_flops, exec_ai, choice, date):

    title = {}
    data = {}
    data_cycles = {}

    executable_name = os.path.basename(executable_path)
    min_ai = 0.015625
    min_flops = 0.25
    while exec_ai < min_ai :
        min_ai /= 2
    
    while exec_flops < min_flops :
        min_flops /= 2

    script_dir = os.path.dirname(os.path.abspath(__file__))

    result_dir = os.path.join(script_dir, "Results/Roofline")

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
    if(os.path.isdir('Results') == False):
            os.mkdir('Results')
    if(os.path.isdir('Results/Applications') == False):
        os.mkdir('Results/Applications')
    plt.savefig('Results/Applications/' + executable_name + "_" + title["name"] + '_PMU_roofline_analysis_' + date + '_' + str(title["isa"]) + "_" + str(title["precision"]) + "_" + str(title["threads"]) + "_Threads_" + str(title["load"]) + "Load_" + str(title["store"]) + "Store_" + title["inst"] + '.svg')

def update_csv(machine, executable_path, exec_flops, exec_ai, bandwidth, time, name, date, isa, precision, threads):

    csv_path = f"./Results/Applications/{machine}_Applications.csv"

    if name == "":
        name = os.path.basename(executable_path)

    if(os.path.isdir('Results') == False):
        os.mkdir('Results')
    if(os.path.isdir('Results/Applications') == False):
        os.mkdir('Results/Applications')

    results = [
        date,
        "PMU",
        name,
        isa,
        precision,
        threads,
        run.custom_round(exec_ai),
        run.custom_round(exec_flops),
        run.custom_round(bandwidth),
        run.custom_round(time)
    ]

    headers = ['Date', 'Method', 'Name', 'ISA', 'Precision', 'Threads', 'AI', 'Gflops', 'Bandwidth', 'Time']

    if os.path.exists(csv_path):
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(results)
    else:
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerow(results)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run an executable with PAPI instrumentation.")

    parser.add_argument("PAPI_path", nargs = "?", help="Path to the PAPI directory.")
    parser.add_argument("executable_path", help="Path to the executable provided by the user.")
    parser.add_argument('-dr', '--drawroof',  dest='drawroof', action='store_const', const=1, default=0, help='Plot application in a chosen roofline chart localy (work in progress).')
    parser.add_argument('-c', '--choice', default=0, nargs='?', type = int, help='Automatically choose a roofline chart for the application analysis, --drawroof is required for this (Default: 0).')
    parser.add_argument("additional_args", nargs="...", help="Additional arguments for the user's application.")
    parser.add_argument('-n','--name', default='unnamed', nargs='?', type = str, help='Name for the machine running the app. (Default: unnamed)')
    parser.add_argument('-an','--app_name', default='', nargs='?', type = str, help='Name for the app.')
    parser.add_argument('--isa', default='', nargs='?', choices=['avx512', 'avx', 'avx2', 'sse', 'scalar', 'neon', 'armscalar', 'riscvscalar', 'riscvvector', ''], help='Main ISA used by the application, if not sure leave blank (optional only for naming facilitation).')    

    args = parser.parse_args()

    CPU_Type = platform.machine()
    if CPU_Type != "x86_64" and CPU_Type != "aarch64":
        print("No PMU analysis support on non x86 / ARM CPUS.")
        sys.exit(1)

    if not (args.PAPI_path is None):
        check_PAPI_exists(args.PAPI_path)

    total_time_nsec, total_mem, total_sp, total_dp, thread_count = runPAPI(args.executable_path, args.additional_args)

    total_fp = total_sp + total_dp

    sp_ratio = float (total_sp / total_fp)
    dp_ratio = float (total_dp / total_fp)

    memory_bytes = total_mem * (sp_ratio * 4 + dp_ratio * 8)

    if dp_ratio > 0.9:
        precision = "dp"
    elif sp_ratio > 0.9:
        precision = "sp"
    else:
        precision = "mixed"

    ai = float (total_fp / memory_bytes)

    gflops = float(total_fp / total_time_nsec)
    bandwidth = float((memory_bytes) / total_time_nsec)

    print("\n---------PMU RESULTS-----------")
    print("Total FP Operations:", total_fp)
    print("Calculated Total Memory Bytes:", memory_bytes)
    #print("Simple AI:", float(total_fp / total_mem))
    print("SP FLOP Ratio: " + str(sp_ratio) + " DP FLOP Ration: " + str(dp_ratio))
    print("Threads Used:", thread_count)

    print("\nExecution Time (seconds):" + str(float(total_time_nsec / 1e9)))
    print("GFLOP/s: " + str(gflops))
    print("Bandwidth (GB/s): " + str(bandwidth))
    print("Arithmetic Intensity:", ai)
    print("------------------------------")

    ct = datetime.datetime.now()
    
    #Plot Roofline
    if args.drawroof:
        if not plot_numpy == None:
            print("Manual application plotting not implemented iet, results can be viewed using the GUI")
            #date = ct.strftime('%Y-%m-%d_%H-%M-%S')
            #plot_roofline_with_dot(args.executable_path, gflops, ai, args.choice, date)
        else:
            print("No Matplotlib and/or Numpy found, in order to draw CARM graphs make sure to install them.")

    date = ct.strftime('%Y-%m-%d %H:%M:%S')
    update_csv(args.name, args.executable_path, gflops, ai, bandwidth, float(total_time_nsec / 1e9), args.app_name, date, args.isa, precision, thread_count)