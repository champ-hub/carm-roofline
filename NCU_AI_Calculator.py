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

def process_metrics(ncu_report, kernel_name):
	try:
		data = pd.read_csv(ncu_report, sep=',')
	except pd.errors.EmptyDataError:
		print(f"There is no kernel to profile with the name {kernel_name}")
		os.remove(ncu_report)
		sys.exit(4)


	data.replace(',','', regex=True, inplace=True)
	data['Metric Value'] = pd.to_numeric(data['Metric Value'])

	# find all the rows that share a metric name and sum their metric values
	grouped_data = data.groupby('Metric Name')['Metric Value'].sum()
	
	execution_time = grouped_data['gpu__time_duration.avg']

	bytes_requested = grouped_data.filter(like="sm__sass_data_bytes_mem").sum()

	tensor_flops = grouped_data.filter(like="sm__ops_path_tensor_src").sum()

	half_flops = grouped_data['sm__sass_thread_inst_executed_op_hadd_pred_on.sum']+2*grouped_data['sm__sass_thread_inst_executed_op_hfma_pred_on.sum']+grouped_data['sm__sass_thread_inst_executed_op_hmul_pred_on.sum']

	float_flops = grouped_data['sm__sass_thread_inst_executed_op_fadd_pred_on.sum']+2*grouped_data['sm__sass_thread_inst_executed_op_ffma_pred_on.sum']+grouped_data['sm__sass_thread_inst_executed_op_fmul_pred_on.sum']

	double_flops = grouped_data['sm__sass_thread_inst_executed_op_dadd_pred_on.sum']+2*grouped_data['sm__sass_thread_inst_executed_op_dfma_pred_on.sum']+grouped_data['sm__sass_thread_inst_executed_op_dmul_pred_on.sum']

	return execution_time, bytes_requested, tensor_flops, half_flops, float_flops, double_flops

def update_csv(machine_name, executable_path, app_name, performance, ai, bandwidth, execution_time, date, target, precision):
	csv_path = f"./Results/Applications/{machine_name}_Applications.csv"

	if app_name == "":
		app_name = os.path.basename(executable_path)

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



def run_ncu(machine_name, app_name, executable_path, no_tensor, kernel_name = "", additional_args = []):
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

	print("\n----------NCU Results----------")

	execution_time_nsec, bytes_requested, tensor_flops, half_flops, float_flops, double_flops = process_metrics(tmp_file_path,kernel_name) # Analyse metrics from kernels 

	cuda_flops = half_flops + float_flops + double_flops
	total_flops = cuda_flops + tensor_flops

	ai = float(total_flops/bytes_requested)
	gflops = float(total_flops/execution_time_nsec)
	gbw = float(bytes_requested/execution_time_nsec)

	print("Total FLOPS:", total_flops)
	print("\tTotal Cuda Core FLOPS:", cuda_flops)
	print("\t\t - Half:", half_flops)
	print("\t\t - Single:", float_flops)
	print("\t\t - Double:", double_flops)
	print("\tTotal Tensor Core FLOPS:", tensor_flops)
	print("Total Transfered Bytes:", bytes_requested, "\n")
	
	print("Execution Time (s):", float(execution_time_nsec/1e9))
	print("Performance (GFLOPS/s):", gflops)
	print("Bandwidth (GB/s):", gbw)
	print("Arithmetic Intensity:", ai)
	print("------------------------------")

	ct = datetime.datetime.now()
	date = ct.strftime('%Y-%m-%d %H:%M:%S')

	target = 'mixed'
	if (cuda_flops/total_flops) > 0.9:
		target = 'cuda'
	elif (tensor_flops/total_flops) > 0.9:
		target = 'tensor'

	update_csv(machine_name, executable_path, app_name, gflops, ai, gbw, float(execution_time_nsec/1e9), date, target, 'na')
	# TODO: Needs discussion on threads and precision

	os.remove(tmp_file_path)



def main():
	parser = argparse.ArgumentParser(description="Profile a GPU application with NCU.")

	parser.add_argument("executable_path", help="Path to the executable provided by the user.")
	parser.add_argument("additional_args", nargs="...", help='Additional arguments for target application.')
	parser.add_argument("-n", '--name', default="unnamed", nargs='?', type=str, help='Name for the machine running the application.')
	parser.add_argument("-an", '--app_name', default='', nargs='?', type=str, help="Name for the target app.")
	parser.add_argument("--no_tensor", action='store_const', const=1, default=0, help='Disable Tensor Core profilling for applications that do not need it')
	parser.add_argument("--kernel_name", default="", nargs='?', help='Name of target kernel when profilling a single kernel.')

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

	run_ncu(args.name, args.app_name, args.executable_path, args.no_tensor, args.kernel_name ,args.additional_args)

if __name__ == "__main__":
	main()