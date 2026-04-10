#!/usr/bin/env python3

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

def _load_json_objects(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    decoder = json.JSONDecoder()
    index = 0
    objects = []

    while index < len(content):
        while index < len(content) and content[index].isspace():
            index += 1

        if index >= len(content):
            break

        obj, next_index = decoder.raw_decode(content, index)
        objects.append(obj)
        index = next_index

    return objects


def _find_json_files(root_dir):
    json_files = []
    for root, _, files in os.walk(root_dir):
        for file_name in files:
            if file_name.endswith('.json'):
                json_files.append(os.path.join(root, file_name))
    return json_files

def _run_single_papi_event(command, papi_event, run_label, papi_output_folder):
    if os.path.isdir(papi_output_folder):
        shutil.rmtree(papi_output_folder)
    os.makedirs(papi_output_folder, exist_ok=True)

    os.environ["PAPI_OUTPUT_DIRECTORY"] = papi_output_folder
    os.environ["PAPI_EVENTS"] = papi_event

    print("------------------------------")
    print(f"Running Provided Application For {run_label} PMU Data\n")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("Error executing the command:", e)

    result = analysePAPI(papi_event, papi_output_folder)

    if os.path.isdir(papi_output_folder):
        shutil.rmtree(papi_output_folder)

    return result


#Run PAPI with provided application
def runPAPI(executable_path, _debug, additional_args=None):
    additional_args = additional_args or []
    #Construct the command with the provided paths and additional arguments
    command = [executable_path, *additional_args]

    PAPI_output_folder = "carm_pmu_output"
    event_sequence = [
        ("PAPI_LST_INS", "Memory Instructions"),
        ("PAPI_SP_OPS", "SP FP Operations"),
        ("PAPI_DP_OPS", "DP FP Operations"),
    ]

    print("\n------------------------------")
    event_results = {}
    thread_count = 0
    for event_name, event_label in event_sequence:
        total_real_time_nsec, total_event_ops, current_thread_count = _run_single_papi_event(
            command, event_name, event_label, PAPI_output_folder
        )
        event_results[event_name] = (total_real_time_nsec, total_event_ops)
        thread_count = current_thread_count

    total_real_time_nsec_mem, total_papi_mem_ins = event_results["PAPI_LST_INS"]
    total_real_time_nsec_sp, total_papi_sp_ops = event_results["PAPI_SP_OPS"]
    total_real_time_nsec_dp, total_papi_dp_ops = event_results["PAPI_DP_OPS"]

    return float((total_real_time_nsec_mem + total_real_time_nsec_sp + total_real_time_nsec_dp)/3), total_papi_mem_ins, total_papi_sp_ops, total_papi_dp_ops, thread_count


def analysePAPI(PAPI_Event, papi_output_root):
    if not os.path.isdir(papi_output_root):
        print(f"Error: The directory '{papi_output_root}' was not found, does the analyzed executable contain the necessary ROI definitions using the PAPI high-level interface?")
        sys.exit(1)

    json_files = _find_json_files(papi_output_root)
    if not json_files:
        print(f"Error: No JSON output found in '{papi_output_root}'.")
        sys.exit(1)

    total_papi_event_ops = 0
    total_real_time_nsec = 0
    thread_count = 0

    for file_path in json_files:
        try:
            json_objects = _load_json_objects(file_path)
        except json.JSONDecodeError as err:
            print(f"Error: Failed to parse JSON data in '{file_path}': {err}")
            sys.exit(1)

        for json_data in json_objects:
            threads = json_data.get('threads', {}) if isinstance(json_data, dict) else {}
            if not isinstance(threads, dict):
                continue

            thread_count += len(threads)
            for thread_info in threads.values():
                regions = thread_info.get('regions', {}) if isinstance(thread_info, dict) else {}
                if not isinstance(regions, dict):
                    continue

                for region_info in regions.values():
                    total_papi_event_ops += int(region_info.get(PAPI_Event, 0))
                    total_real_time_nsec += int(region_info.get('real_time_nsec', 0))

    if thread_count == 0:
        print(f"Error: No thread data found while parsing '{papi_output_root}'.")
        sys.exit(1)

    total_real_time_nsec = total_real_time_nsec / thread_count
    return total_real_time_nsec, total_papi_event_ops, thread_count

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run an executable with PAPI instrumentation.")

    parser.add_argument("executable_path", help="Path to the executable provided by the user.")
    parser.add_argument('-d', '--debug',  dest='debug', action='store_const', const=1, default=0, help='Ignored: output is always cleaned after each PAPI event run')
    parser.add_argument('-dr', '--drawroof',  dest='drawroof', action='store_const', const=1, default=0, help='Plot application in a chosen roofline chart localy (work in progress).')
    parser.add_argument('-c', '--choice', default=0, nargs='?', type = int, help='Automatically choose a roofline chart for the application analysis, --drawroof is required for this (Default: 0).')
    parser.add_argument("additional_args", nargs="...", help="Additional arguments for the user's application.")
    parser.add_argument('-n','--name', default='unnamed', nargs='?', type = str, help='Name for the machine running the app. (Default: unnamed)')
    parser.add_argument('-an','--app_name', default='', nargs='?', type = str, help='Name for the app.')
    parser.add_argument('--isa', default='', nargs='?', choices=['avx512', 'avx', 'avx2', 'sse', 'scalar', 'neon', 'armscalar', 'riscvscalar', 'riscvvector', ''], help='Main ISA used by the application, if not sure leave blank (optional only for naming facilitation).')
    parser.add_argument('--vlen', nargs='?', type=int, help="scales the ld/st size")

    args = parser.parse_args()

    if args.vlen is None:
        scale = 1
    else:
        scale = args.vlen // 8


    CPU_Type = platform.machine()
    if CPU_Type != "x86_64" and CPU_Type != "aarch64":
        print("No PMU analysis support on non x86 / ARM CPUS.")
        sys.exit(1)


    total_time_nsec, total_mem, total_sp, total_dp, thread_count = runPAPI(args.executable_path, args.debug, args.additional_args)

    time_taken_seconds = float (total_time_nsec / 1e9)

    total_fp = total_sp + total_dp

    if total_fp == 0:
        print("Error: Total FP operations is zero (SP + DP).")
        print("This usually means the target did not emit PAPI_SP_OPS/PAPI_DP_OPS in ROI regions.")
        sys.exit(1)

    sp_ratio = float (total_sp / total_fp)
    dp_ratio = float (total_dp / total_fp)

    memory_bytes = total_mem * (sp_ratio * 4 + dp_ratio * 8) * scale

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
