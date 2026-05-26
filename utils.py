from __future__ import annotations
import os
import sys
import math
from decimal import Decimal
import argparse
import csv

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
