import argparse
import subprocess
import os
import datetime
import time
import sys
import json
import platform
import shutil

import utils as ut

if not hasattr(time, 'time_ns'):
    time.time_ns = lambda: int(time.time() * 1e9)

#Define a global set to store the names of JSON files that have been read
read_files = set()

#Run PAPI with provided application
def runPAPI(executable_path, debug, additional_args=None):
    additional_args = additional_args or []
    #Construct the command with the provided paths and additional arguments
    command = [executable_path, *additional_args]

    PAPI_output_folder = "carm_pmu_output"
    os.environ["PAPI_OUTPUT_DIRECTORY"] = PAPI_output_folder

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

    #Remove intermediate output when not in debug mode
    if not debug:
        if os.path.isdir(PAPI_output_folder):
            shutil.rmtree(PAPI_output_folder)
        else:
            print(f"Warning: Directory {PAPI_output_folder} does not exist, something went wrong")

    return float((total_real_time_nsec_mem + total_real_time_nsec_sp + total_real_time_nsec_dp)/3), total_papi_mem_ins, total_papi_sp_ops, total_papi_dp_ops, thread_count
    

def analysePAPI(PAPI_Event):
    global read_files

    PAPI_output = "./carm_pmu_output/papi_hl_output"
    try:
        files = os.listdir(PAPI_output)
        # Continue with processing `files`...
        
    except FileNotFoundError:
        print(f"Error: The directory '{PAPI_output}' was not found, does the analyzed executable contain the necessary ROI definitions using the PAPI high-level interface?")
        sys.exit(1)  # Exit the program with a non-zero status

    # If you need to handle the case where it exists but is not a directory:
    except NotADirectoryError:
        print(f"Error: The directory '{PAPI_output}' was not found, does the analyzed executable contain the necessary ROI definitions using the PAPI high-level interface?")
        sys.exit(1)
    #files = os.listdir(PAPI_output)

    #Filter out files that have already been read
    new_files = [file for file in files if file not in read_files]

    #Ensure there's exactly one new file in the folder
    if len(new_files) != 1:
        print("Error: Folder should contain exactly one new file.")
        sys.exit(1)
    
    #Get the path of the new JSON file
    new_file_path = os.path.join(PAPI_output, new_files[0])
    
    #Ensure the file is a JSON file
    if not new_file_path.endswith('.json'):
        print("Error: File in folder is not a JSON file.")
        sys.exit(1)
    
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run an executable with PAPI instrumentation.")

    parser.add_argument("executable_path", help="Path to the executable provided by the user.")
    parser.add_argument('-d', '--debug',  dest='debug', action='store_const', const=1, default=0, help='Will conserve the raw PMU output from PAPI for debugging')
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


    total_time_nsec, total_mem, total_sp, total_dp, thread_count = runPAPI(args.executable_path, args.debug, args.additional_args)

    time_taken_seconds = float (total_time_nsec / 1e9)

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
    print("Total FP Operations:", ut.custom_round(total_fp))
    print("Calculated Total Memory Bytes:", ut.custom_round(memory_bytes))
    print("SP FLOP Ratio: " + str(ut.custom_round(sp_ratio)) + " DP FLOP Ration: " + str(ut.custom_round(dp_ratio)))
    print("Threads Used:", thread_count)
    print("\nExecution Time (seconds):",ut.custom_round(time_taken_seconds))
    print("GFLOP/s: " + str(ut.custom_round(gflops)))
    print("Bandwidth (GB/s): " + str(ut.custom_round(bandwidth)))
    print("Arithmetic Intensity:", ut.custom_round(ai))
    print("------------------------------")

    ct = datetime.datetime.now()
    date = ct.strftime('%Y-%m-%d %H:%M:%S')
    
    #Plot Roofline
    if args.drawroof:
        print("Manual application plotting not implemented iet, results can be viewed using the GUI")
        #ut.plot_roofline_with_dot(args.executable_path, gflops, ai, args.choice, date, "pmu")

    ut.update_csv(args.name, args.executable_path, gflops, ai, bandwidth, time_taken_seconds, args.app_name, date, args.isa, precision, thread_count, "PMU", 1, 1)