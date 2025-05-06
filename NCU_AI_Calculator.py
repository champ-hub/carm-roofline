#!/usr/bin/env python3
from dotenv import load_dotenv
import os
import argparse
import subprocess
import sys
import pandas as pd
import csv
import datetime

load_dotenv('GPU/gpu.env')

DEVICE = os.getenv('DEVICE')
CUDA_PATH = os.getenv('CUDA_PATH')

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

def preprocess_output(ncu_report):
	key = '"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit","Metric Value"'
	save = False

	with open(ncu_report, 'r+') as fr:
		lines = fr.readlines()
	
	with open(ncu_report, 'w+') as fw:
		for line in lines:
			if line.strip('\n') == key:
				save = True
			if save:
				fw.write(line)

def process_metrics(ncu_report, kernel_name, level):
	try:
		data = pd.read_csv(ncu_report, sep=',')
	except pd.errors.EmptyDataError:
		print(f"There is no kernel to profile with the name {kernel_name}")
		os.remove(ncu_report)
		sys.exit(4)


	data.replace(',','', regex=True, inplace=True)
	data['Metric Value'] = pd.to_numeric(data['Metric Value'])

	if level == "app":
		# find all the rows that share a metric name and sum their metric values
		grouped_data = data.groupby('Metric Name')['Metric Value'].sum()
		
		execution_time = grouped_data['gpu__time_duration.avg']

		bytes_requested = grouped_data.filter(like="sm__sass_data_bytes_mem").sum()

		tensor_flops = grouped_data.filter(like="sm__ops_path_tensor_src").sum()

		half_flops = grouped_data['sm__sass_thread_inst_executed_op_hadd_pred_on.sum']+2*grouped_data['sm__sass_thread_inst_executed_op_hfma_pred_on.sum']+grouped_data['sm__sass_thread_inst_executed_op_hmul_pred_on.sum']

		float_flops = grouped_data['sm__sass_thread_inst_executed_op_fadd_pred_on.sum']+2*grouped_data['sm__sass_thread_inst_executed_op_ffma_pred_on.sum']+grouped_data['sm__sass_thread_inst_executed_op_fmul_pred_on.sum']

		double_flops = grouped_data['sm__sass_thread_inst_executed_op_dadd_pred_on.sum']+2*grouped_data['sm__sass_thread_inst_executed_op_dfma_pred_on.sum']+grouped_data['sm__sass_thread_inst_executed_op_dmul_pred_on.sum']

		results = [{"execution_time": execution_time, "bytes_requested": bytes_requested, "tensor_flops": tensor_flops, "half_flops": half_flops, "float_flops": float_flops, "double_flops": double_flops}]

	else:
		reps = data.groupby('Kernel Name')['ID'].nunique()
		grouped_data = data.groupby(['Kernel Name', 'Metric Name'])['Metric Value'].sum()

		results = []

		for kernel_name in grouped_data.index.levels[0]:

			execution_time = grouped_data[kernel_name,'gpu__time_duration.avg']

			bytes_requested = grouped_data[kernel_name].filter(like="sm__sass_data_bytes_mem").sum()

			tensor_flops = grouped_data[kernel_name].filter(like="sm__ops_path_tensor_src").sum()

			half_flops = grouped_data[kernel_name, 'sm__sass_thread_inst_executed_op_hadd_pred_on.sum']+2*grouped_data[kernel_name, 'sm__sass_thread_inst_executed_op_hfma_pred_on.sum']+grouped_data[kernel_name, 'sm__sass_thread_inst_executed_op_hmul_pred_on.sum']

			float_flops = grouped_data[kernel_name, 'sm__sass_thread_inst_executed_op_fadd_pred_on.sum']+2*grouped_data[kernel_name, 'sm__sass_thread_inst_executed_op_ffma_pred_on.sum']+grouped_data[kernel_name, 'sm__sass_thread_inst_executed_op_fmul_pred_on.sum']

			double_flops = grouped_data[kernel_name, 'sm__sass_thread_inst_executed_op_dadd_pred_on.sum']+2*grouped_data[kernel_name, 'sm__sass_thread_inst_executed_op_dfma_pred_on.sum']+grouped_data[kernel_name, 'sm__sass_thread_inst_executed_op_dmul_pred_on.sum']

			tmp={"kernel_name": kernel_name, "calls": reps[kernel_name], "execution_time": execution_time, "bytes_requested": bytes_requested, "tensor_flops": tensor_flops, "half_flops": half_flops, "float_flops": float_flops, "double_flops": double_flops}
			
			results.append(tmp)


	return results

def update_csv(machine_name, app_name, performance, ai, bandwidth, execution_time, date, target, precision):
	csv_path = f"./Results/Applications/{machine_name}_Applications.csv"

	if(os.path.isdir('Results') == False):
		os.mkdir('Results')
	if(os.path.isdir('Results/Applications') == False):
		os.mkdir('Results/Applications')

	results = [date, "NCU", app_name, target, precision, 0, custom_round(ai), custom_round(performance), custom_round(bandwidth), custom_round(execution_time)]

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
	tmp_file_path = 'tmp_report.csv'
	ncu_path = f'{CUDA_PATH}/bin/ncu'
	kernel = "" if kernel_name == "" else f' -k {kernel_name}'
	options = f'--replay-mode kernel --clock-control none --print-units base --csv --log-file {tmp_file_path} --devices {DEVICE}{kernel} --metrics'.split(' ')

	cuda_core_metrics = 'sm__sass_data_bytes_mem_global.sum,sm__sass_data_bytes_mem_local.sum,sm__sass_data_bytes_mem_shared.sum,gpu__time_duration.avg,sm__sass_thread_inst_executed_op_fadd_pred_on.sum,sm__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__sass_thread_inst_executed_op_fmul_pred_on.sum,sm__sass_thread_inst_executed_op_hadd_pred_on.sum,sm__sass_thread_inst_executed_op_hfma_pred_on.sum,sm__sass_thread_inst_executed_op_hmul_pred_on.sum,sm__sass_thread_inst_executed_op_dadd_pred_on.sum,sm__sass_thread_inst_executed_op_dfma_pred_on.sum,sm__sass_thread_inst_executed_op_dmul_pred_on.sum'
	tensor_core_metrics = 'sm__ops_path_tensor_src_bf16_dst_fp32.sum,sm__ops_path_tensor_src_fp16_dst_fp16.sum,sm__ops_path_tensor_src_fp16_dst_fp32.sum,sm__ops_path_tensor_src_fp64.sum,sm__ops_path_tensor_src_int1.sum,sm__ops_path_tensor_src_int4.sum,sm__ops_path_tensor_src_int8.sum,sm__ops_path_tensor_src_tf32_dst_fp32.sum'
	
	if not no_tensor:
		cuda_core_metrics += ',' + tensor_core_metrics

	command = [ncu_path, *options, cuda_core_metrics, executable_path, *additional_args]

	result = subprocess.run(command)
	if result.returncode != 0:
		print("Error profilling application.")
		sys.exit(3)

	preprocess_output(tmp_file_path) # Remove unnecessary headers from csv report

	results = process_metrics(tmp_file_path,kernel_name, level) # Analyse metrics from kernels

	for data in results:

		cuda_flops = data["half_flops"] + data["float_flops"] + data["double_flops"]
		total_flops = cuda_flops + data["tensor_flops"]

		ai = float(total_flops / data["bytes_requested"])
		gflops = float(total_flops / data["execution_time"])
		gbw = float(data["bytes_requested"] / data["execution_time"])

		if level == "app":
			print("\n----------NCU Results----------")

			print("Total FLOPS:", total_flops)
			print("\tTotal Cuda Core FLOPS:", cuda_flops)
			print("\t\t - Half:", data["half_flops"])
			print("\t\t - Single:", data["float_flops"])
			print("\t\t - Double:", data["double_flops"])
			print("\tTotal Tensor Core FLOPS:", data["tensor_flops"])
			print("Total Transfered Bytes:", data["bytes_requested"], "\n")
			
			print("Execution Time (s):", float(data["execution_time"]/1e9))
			print("Performance (GFLOPS/s):", gflops)
			print("Bandwidth (GB/s):", gbw)
			print("Arithmetic Intensity:", ai)
			print("------------------------------")

		ct = datetime.datetime.now()
		date = ct.strftime('%Y-%m-%d %H:%M:%S')

		target = 'mixed'
		if (cuda_flops / total_flops) > 0.9:
			target = 'cuda'
		elif (data["tensor_flops"] / total_flops) > 0.9:
			target = 'tensor'

		if app_name == "":
			app_name = os.path.basename(executable_path)

		if level == "kernel":
			if "(" in data["kernel_name"]:
				app_name += f"/{data['kernel_name'][:data['kernel_name'].find('(')]}({data.get('calls', 0)})"
			else:
				app_name += f"/{data['kernel_name']}({data.get('calls', 0)})"

		update_csv(machine_name, app_name, gflops, ai, gbw, float(data["execution_time"] / 1e9), date, target, 'na')
	# TODO: Needs discussion on threads and precision

	os.remove(tmp_file_path)



def main():
	parser = argparse.ArgumentParser(description="Profile a GPU application with NCU.")

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
		subprocess.run('nvidia-smi', stdout=subprocess.PIPE)
		print('NVIDIA GPU detected')

	except Exception:
		print('NVIDIA GPU not detected')
		sys.exit(1)

	# Test if NCU exists
	try:
		subprocess.run(f'{CUDA_PATH}/bin/ncu', stdout=subprocess.PIPE)
		print('NCU Detected')
	except Exception:
		print(f'NCU not detected in {CUDA_PATH}/bin/ncu. Double check your CUDA path on GPU/gpu.env')
		sys.exit(2)

	run_ncu(args.name, args.app_name, args.executable_path, args.no_tensor, args.level, args.kernel_name, args.additional_args)

if __name__ == "__main__":
	main()