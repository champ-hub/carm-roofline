#!/usr/bin/env python3
from dotenv import load_dotenv
import os
import shutil
import argparse
import subprocess
import sys
import pandas as pd
import csv
import datetime
load_dotenv(os.path.dirname(os.path.realpath(__file__))+ '/GPU/gpu.env')

DEVICE = os.getenv('DEVICE')
ROCM_PATH = os.getenv('ROCM_PATH')
ROCPROFV3_PATH = os.getenv('ROCPROFV3_PATH')

COUNTERS_PATH = os.path.dirname(os.path.realpath(__file__))+ '/GPU/rocm_counters.txt'

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

def process_metrics(report_dir, kernel_name, level):
	execution_time = 0
	if level == "app":
		# First group of counters: vector cores
		try:
			data = pd.read_csv(report_dir + "/pmc_1/tmp_counter_collection.csv", sep=',')
		except Exception:
			print(f"There is no kernel to profile with the name {kernel_name} or profiling failed.")
			shutil.rmtree(report_dir)
			sys.exit(4)

		grouped_data = data.groupby("Start_Timestamp")["End_Timestamp"]
		for key in grouped_data.groups.keys():
			execution_time += grouped_data.first()[key] - key

		# find all the rows that share a metric name and sum their metric values
		grouped_data = data.groupby("Counter_Name")["Counter_Value"].sum()

		half_flops = 64 * (grouped_data['SQ_INSTS_VALU_ADD_F16']+2*grouped_data['SQ_INSTS_VALU_FMA_F16']+grouped_data['SQ_INSTS_VALU_MUL_F16'])

		float_flops = 64 * (grouped_data['SQ_INSTS_VALU_ADD_F32']+2*grouped_data['SQ_INSTS_VALU_FMA_F32']+grouped_data['SQ_INSTS_VALU_MUL_F32'])

		# fp64 + matrix
		try:
			data = pd.read_csv(report_dir + "/pmc_2/tmp_counter_collection.csv", sep=',')
		except Exception:
			print(f"There is no kernel to profile with the name {kernel_name} or profiling failed.")
			shutil.rmtree(report_dir)
			sys.exit(4)

		grouped_data = data.groupby("Start_Timestamp")["End_Timestamp"]
		for key in grouped_data.groups.keys():
			execution_time += grouped_data.first()[key] - key

		grouped_data = data.groupby("Counter_Name")["Counter_Value"].sum()

		double_flops = 64 * (grouped_data['SQ_INSTS_VALU_ADD_F64']+2*grouped_data['SQ_INSTS_VALU_FMA_F64']+grouped_data['SQ_INSTS_VALU_MUL_F64'])

		tensor_flops = 512 * (grouped_data['SQ_INSTS_VALU_MFMA_MOPS_F16'] + grouped_data['SQ_INSTS_VALU_MFMA_MOPS_BF16'] + grouped_data['SQ_INSTS_VALU_MFMA_MOPS_F32'] + grouped_data['SQ_INSTS_VALU_MFMA_MOPS_F64'] + grouped_data['SQ_INSTS_VALU_MFMA_MOPS_I8'])

		# bytes
		try:
			data = pd.read_csv(report_dir + "/pmc_3/tmp_counter_collection.csv", sep=',')
		except Exception:
			print(f"There is no kernel to profile with the name {kernel_name} or profiling failed.")
			shutil.rmtree(report_dir)
			sys.exit(4)

		grouped_data = data.groupby("Start_Timestamp")["End_Timestamp"]
		for key in grouped_data.groups.keys():
			execution_time += grouped_data.first()[key] - key

		grouped_data = data.groupby("Counter_Name")["Counter_Value"].sum()

		bytes_requested = (grouped_data['SQ_LDS_IDX_ACTIVE']- grouped_data['SQ_LDS_BANK_CONFLICT']) * 4 * 32 + grouped_data['TCP_TOTAL_CACHE_ACCESSES_sum'] * 64

		results = [{"execution_time": execution_time/3, "bytes_requested": bytes_requested, "tensor_flops": tensor_flops, "half_flops": half_flops, "float_flops": float_flops, "double_flops": double_flops}]

	else:
		with open(report_dir + "/pmc_1/tmp_counter_collection.csv", "a") as baseFile:
			with open(report_dir + "/pmc_2/tmp_counter_collection.csv", "r") as copiedFile:
				next(copiedFile)
				shutil.copyfileobj(copiedFile, baseFile)
			with open(report_dir + "/pmc_3/tmp_counter_collection.csv", "r") as copiedFile:
				next(copiedFile)
				shutil.copyfileobj(copiedFile, baseFile)

		# First group of counters: vector cores
		try:
			data = pd.read_csv(report_dir + "/pmc_1/tmp_counter_collection.csv", sep=',')
		except Exception:
			print(f"There is no kernel to profile with the name {kernel_name} or profiling failed.")
			shutil.rmtree(report_dir)
			sys.exit(4)

		reps = data.groupby("Kernel_Name")["Correlation_Id"].nunique()

		results = []

		grouped_data = data.groupby(['Kernel_Name', 'Counter_Name'])["Counter_Value"].sum()

		time_data = data.groupby(['Kernel_Name', "Start_Timestamp"])["End_Timestamp"]

		for kernel_name in grouped_data.index.levels[0]:
			for key in time_data.groups.keys():
				if key[0] == kernel_name:
					execution_time += time_data.first()[key] - key[1]

			half_flops = 64 * (grouped_data[kernel_name,'SQ_INSTS_VALU_ADD_F16']+2*grouped_data[kernel_name,'SQ_INSTS_VALU_FMA_F16']+grouped_data[kernel_name,'SQ_INSTS_VALU_MUL_F16'])

			float_flops = 64 * (grouped_data[kernel_name,'SQ_INSTS_VALU_ADD_F32']+2*grouped_data[kernel_name,'SQ_INSTS_VALU_FMA_F32']+grouped_data[kernel_name,'SQ_INSTS_VALU_MUL_F32'])

			double_flops = 64 * (grouped_data[kernel_name,'SQ_INSTS_VALU_ADD_F64']+2*grouped_data[kernel_name,'SQ_INSTS_VALU_FMA_F64']+grouped_data[kernel_name,'SQ_INSTS_VALU_MUL_F64'])

			tensor_flops = 512 * (grouped_data['SQ_INSTS_VALU_MFMA_MOPS_F16'] + grouped_data['SQ_INSTS_VALU_MFMA_MOPS_BF16'] + grouped_data['SQ_INSTS_VALU_MFMA_MOPS_F32'] + grouped_data['SQ_INSTS_VALU_MFMA_MOPS_F64'] + grouped_data['SQ_INSTS_VALU_MFMA_MOPS_I8'])

			bytes_requested = (grouped_data[kernel_name,'SQ_LDS_IDX_ACTIVE']- grouped_data[kernel_name,'SQ_LDS_BANK_CONFLICT']) * 4 * 32 + grouped_data[kernel_name,'TCP_TOTAL_CACHE_ACCESSES_sum'] * 64

			tmp={"kernel_name": kernel_name, "calls": reps[kernel_name], "execution_time": execution_time/3, "bytes_requested": bytes_requested, "tensor_flops": tensor_flops, "half_flops": half_flops, "float_flops": float_flops, "double_flops": double_flops}
			execution_time = 0
			results.append(tmp)

	return results

def update_csv(machine_name, app_name, performance, ai, bandwidth, execution_time, date, target, precision):
	csv_path = os.path.dirname(os.path.realpath(__file__)) + f"/Results/Applications/{machine_name}_Applications.csv"

	if(os.path.isdir(os.path.dirname(os.path.realpath(__file__)) +'/Results') == False):
		os.mkdir(os.path.dirname(os.path.realpath(__file__)) +'/Results')
	if(os.path.isdir(os.path.dirname(os.path.realpath(__file__)) +'/Results/Applications') == False):
		os.mkdir(os.path.dirname(os.path.realpath(__file__)) +'/Results/Applications')

	results = [date, "Rocprofv3", app_name, target, precision, 0, custom_round(ai), custom_round(performance), custom_round(bandwidth), custom_round(execution_time)]

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



def run_ncu(machine_name, app_name, executable_path, no_tensor, level, kernel_name = "", additional_args = []):
	tmp_file_path = os.path.dirname(os.path.realpath(__file__)) +'/counters'
	kernel = "" if kernel_name == "" else f' --kernel-include-regex {kernel_name}'

	options = f'-i {COUNTERS_PATH} -o tmp -d {tmp_file_path} -T{kernel} --'.split(' ')

	command = [ROCPROFV3_PATH, *options, executable_path, *additional_args]

	result = subprocess.run(command)
	if result.returncode != 0:
		print("Error profilling application.")
		shutil.rmtree(tmp_file_path)
		sys.exit(3)

	results = process_metrics(tmp_file_path,kernel_name, level) # Analyse metrics from kernels

	if app_name == "":
		app_name = os.path.basename(executable_path)

	for data in results:

		vector_flops = data["half_flops"] + data["float_flops"] + data["double_flops"]
		total_flops = vector_flops + data["tensor_flops"]

		ai = float(total_flops / data["bytes_requested"])
		gflops = float(total_flops / data["execution_time"])
		gbw = float(data["bytes_requested"] / data["execution_time"])

		if level == "app":
			print("\n----------ROCPROFV3 Results----------")

			print("Total FLOPS:", total_flops)
			print("\tTotal Vector Core FLOPS:", vector_flops)
			print("\t\t - Half:", data["half_flops"])
			print("\t\t - Single:", data["float_flops"])
			print("\t\t - Double:", data["double_flops"])
			print("\tTotal Matrix Core FLOPS:", data["tensor_flops"])
			print("Total Transfered Bytes:", data["bytes_requested"], "\n")
			
			print("Execution Time (s):", float(data["execution_time"]/1e9))
			print("Performance (GFLOPS/s):", gflops)
			print("Bandwidth (GB/s):", gbw)
			print("Arithmetic Intensity:", ai)
			print("---------------------------------------")

		ct = datetime.datetime.now()
		date = ct.strftime('%Y-%m-%d %H:%M:%S')

		target = 'mixed'
		if (total_flops != 0):
			if (vector_flops / total_flops) > 0.9:
				target = 'vector'
			elif (data["tensor_flops"] / total_flops) > 0.9:
				target = 'matrix'
		else:
			target = 'na'

		if level == "kernel":
			if "(" in data["kernel_name"]:
				app_name_csv = app_name + f"/{data['kernel_name'][:data['kernel_name'].find('(')]}({data.get('calls', 0)})"
			else:
				app_name_csv = app_name + f"/{data['kernel_name']}({data.get('calls', 0)})"
		else:
			app_name_csv = app_name

		update_csv(machine_name, app_name_csv, gflops, ai, gbw, float(data["execution_time"] / 1e9), date, target, 'na')

		app_name = app_name
	# TODO: Needs discussion on threads and precision

	shutil.rmtree(tmp_file_path)



def main():
	parser = argparse.ArgumentParser(description="Profile a GPU application with rocprofv3.")

	parser.add_argument("executable_path", help="Path to the executable provided by the user.")
	parser.add_argument("additional_args", nargs="...", help='Additional arguments for target application.')
	parser.add_argument("-n", '--name', default="unnamed", nargs='?', type=str, help='Name for the machine running the application.')
	parser.add_argument("-an", '--app_name', default='', nargs='?', type=str, help="Name for the target app.")
	parser.add_argument("--no_tensor", action='store_const', const=1, default=0, help='Disable Tensor Core profilling for applications that do not need it')
	parser.add_argument("-k", "--kernel_name", default="", nargs='?', help='Name of target kernel when profilling a single kernel.')
	parser.add_argument("-l", "--level", default="app", choices=["app", "kernel"], help='Level of profiling. Choose between app or kernel. Default is app. Kernel level seperates the metrics per kernel.')

	args=parser.parse_args()

	# Test if NVIDIA GPU is available
	try:
		subprocess.run('amd-smi', stdout=subprocess.PIPE)
		print('AMD GPU detected')

	except Exception:
		print('AMD GPU not detected')
		sys.exit(1)

	# Test if rocprofv3 exists
	try:
		subprocess.run([ROCPROFV3_PATH, '-h'], stdout=subprocess.PIPE)
		print('Rocprofv3 detected.')
	except Exception:
		print(f'Rocprofv3 not detected. Double check the Rocprofv3 path in GPU/gpu.env.')
		sys.exit(2)

	run_ncu(args.name, args.app_name, args.executable_path, args.no_tensor, args.level, args.kernel_name, args.additional_args)

if __name__ == "__main__":
	main()