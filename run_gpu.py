#!/usr/bin/env python3
import argparse
import os
import subprocess
import datetime
import sys 
import csv
import json
from dotenv import load_dotenv

load_dotenv('GPU/gpu.env')

DEVICE = os.getenv('DEVICE')

def read_config(config_file):
	f = open(config_file, 'r')

	name = None

	for line in f:
		if line[0] == '#':
			continue

		l = line.split('=')
		if l[0] == 'name':
			name = l[1].rstrip()
	
	return name

def update_csv(name, test, results, date, target, precision, inst, threads, blocks, out_path):
	csv_path = f"{out_path}/Roofline/{name}_{test}.csv"
	output = [date, target, precision, threads, 1, 1, blocks, 0, inst]

	output.append(results["shared"])
	output.append(0) #I/Cycle not implemented yet

	output.append(0) #L1 cache not implemented yet
	output.append(0) #I/Cycle not implemented yet
	
	output.append(results["l2"])
	output.append(0) #I/Cycle not implemented yet

	output.append(results["global"])
	output.append(0) #I/Cycle not implemented yet

	if "flops" in results:
		output.append(results["flops"])
		output.append(0) #I/Cycle not implemented yet
	else:
		output.append(0)
		output.append(0)

	if "fma" in results:
		output.append(results["fma"])
		output.append(0) #I/Cycle not implemented yet
	else:
		output.append(0)
		output.append(0)

	secondary_headers = ["Name:", name, "Shared Memory/L1 Size:", "0", "L2 Size:", "0", "Global Size:", "0", "", "Shared", "Shared", "L1", "L1", "L2", "L2", "Global", "Global", "FP", "FP", "FP FMA", "FP FMA"]
	primary_headers = ["Date", "ISA", "Precision", "Threads per Block", "Loads", "Stores", "Number of Blocks", "DRAM Bytes", "FP Inst.", "GB/s", "I/Cycle", "GB/s", "I/Cycle", "GB/s", "I/Cycle", "GB/s", "I/Cycle", "Gflops/s", "I/Cycle", "Gflops/s", "I/Cycle"]

	#Check if the file exists
	if os.path.exists(csv_path):
		#If exists, append without header
		with open(csv_path, 'a', newline='') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(output)
	else:
		#If not, write with header and include secondary headers
		with open(csv_path, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(secondary_headers)
			writer.writerow(primary_headers)
			writer.writerow(output)

def check_hardware(verbose, set_freq, freq_sm, freq_mem, arch, target_vector, target_tensor):
	compute_capability = 0
	gpu_name = ''
	vector_precisions = ['int', 'hp', 'sp', 'dp']
	tensor_cores = False
	tensor_core_precisions = []

	if arch == 'nvidia':
		# Obtain compute capability
		result = subprocess.run(['nvidia-smi','--query-gpu=compute_cap', '-i', str(DEVICE), '--format=csv,noheader'], stdout=subprocess.PIPE)
		result = result.stdout.decode('utf-8').rstrip().split('.')
		compute_capability = int(result[0])*10+int(result[1])

		# Obtain GPU name to see if it's GTX (necessary as GTX 1660 is compute capability 75 and does not have TC)
		result = subprocess.run(['nvidia-smi','--query-gpu=gpu_name', '-i', str(DEVICE), '--format=csv,noheader'], stdout=subprocess.PIPE)
		gpu_name = result.stdout.decode('utf-8').rstrip()

		if compute_capability >= 70 and gpu_name.find('GTX') < 0:
			tensor_cores = True
			tensor_core_precisions.append('fp16_32')

			if compute_capability >= 75:
				tensor_core_precisions.append('fp16_16')
				tensor_core_precisions.append('int8')
				tensor_core_precisions.append('int4')

				if compute_capability >= 80:
					tensor_core_precisions.append('bf16')
					tensor_core_precisions.append('tf32')
					tensor_core_precisions.append('int1')
					vector_precisions.append('bf16')

					if compute_capability >= 89:
						pass #implement fp8
	else:
		result = subprocess.run(['amd-smi', 'static', '-g', str(DEVICE), '-a', '--json'], stdout=subprocess.PIPE).stdout.decode('utf-8').rstrip()
		compute_capability = json.loads(result)
		gpu_name = compute_capability[0]['asic']['market_name']
		compute_capability = compute_capability[0]['asic']['target_graphics_version']
		
		major = compute_capability[3:-2]

		if int(major) >= 9:
			vector_precisions.append('bf16')
			if compute_capability in ['gfx908', 'gfx90a', 'gfx942']:
				tensor_cores = True
				tensor_core_precisions.append('fp32')
				tensor_core_precisions.append('int8')
				tensor_core_precisions.append('fp16_32')
				tensor_core_precisions.append('bf16')
				
				if compute_capability in ['gfx90a', 'gfx942']:
					tensor_core_precisions.append('fp64')

					if compute_capability == 'gfx942':
						pass #implement fp8

	# Check valid arithmetic precisions
	if target_vector[0] == 'auto':
		target_vector = vector_precisions.copy()
	elif 'none' in target_vector:
		target_vector = []
	else:
		for item in target_vector:
			if item not in vector_precisions:
				print("WARNING: Selected Vector arithmetic precision", item, "was detected and removed since it is not supported by the tested GPU.")
				target_vector.remove(item)
	
	if target_tensor[0] == 'auto':
		target_tensor = tensor_core_precisions.copy()
	elif 'none' in target_tensor:
		target_tensor = []
	else:
		for item in target_tensor:
			if item not in tensor_core_precisions:
				print("WARNING: Selected Tensor arithmetic precision", item, "was detected and removed since it is not supported by the tested GPU.")
				target_tensor.remove(item)
	
	if arch == 'nvidia':
	# Configure SM and MEM frequency
		if set_freq and freq_sm !=0 and freq_mem != 0:
			# Turn on persistency mode
			result = subprocess.run(['nvidia-smi', '-i', str(DEVICE), '-pm', '1'], stdout=subprocess.PIPE)
			
			if result.returncode != 0:
				print(result.stdout.decode('utf-8').rstrip())
				sys.exit(2)
			
			# Configure SM freqency
			result = subprocess.run(['nvidia-smi', '-i', str(DEVICE), '-lgc', str(freq_sm)], stdout=subprocess.PIPE)

			if result.returncode != 0:
				print(result.stdout.decode('utf-8').rstrip())
				sys.exit(3)
			
			# Configure MEM frequency
			result = subprocess.run(['nvidia-smi', '-i', str(DEVICE), '-lmc', str(freq_mem)], stdout=subprocess.PIPE)

			if result.returncode != 0:
				print(result.stdout.decode('utf-8').rstrip())
				sys.exit(4)
			
	
	if verbose > 2:
		print("-----------------GPU INFORMATION-----------------")
		print("GPU:", gpu_name)
		print("Compute Capability:", compute_capability)
		print("Supported Vector Precisions:", ', '.join(vector_precisions))
		print("Tested Vector Precisions:", ', '.join(target_vector))
		if tensor_cores:
			print("Supported Tensor Precisions:", ', '.join(tensor_core_precisions))
			print("Tested Tensor Precisions:", ', '.join(target_tensor))
	
		if arch == 'nvidia':
			# Print current GPU frequencies
			result = subprocess.run(['nvidia-smi', '--query-gpu=clocks.sm,clocks.mem', '-i', str(DEVICE), '--format=csv,noheader'], stdout=subprocess.PIPE)
			result = result.stdout.decode('utf-8').rstrip().split(' ')
			real_freq_sm = result[0]
			real_freq_mem = result[2]
			print("Current SM Frequency:", real_freq_sm, "MHz")
			print("Current Memory Frequency:", real_freq_mem, "MHz")

	return compute_capability, target_vector, target_tensor



def run_roofline(verbose, name, out, set_freq, freq_sm, freq_mem, arch, target_vector, target_tensor, vector_op, threads, blocks):
	compute_capability, target_vector, target_tensor = check_hardware(verbose, set_freq, freq_sm, freq_mem, arch, target_vector, target_tensor)

	if verbose == 1:
		print("------------------------------")
		if arch == 'nvidia':
			print("Running Benchmarks for the CUDA Core Precisions: ", target_vector)
		else:
			print("Running Benchmarks for the Rocm Vector Core Precisions: ", target_vector)
		if not target_tensor:
			if arch == 'nvidia':
				print("Tensor Cores are not supported in this device.")
			else:
				print("Matrix Cores are not supported in this device.")
		else:
			if arch == 'nvidia':
				print("On the Following Tensor Core Precisions: ", target_tensor)
			else:
				print("On the Following Matrix Core Precisions: ", target_tensor)
		print("------------------------------")


	# Compile benchmark generator
	os.system("cd GPU && make -s clean && make -s")

	# Cuda Core benchmarks
	for precision in target_vector:
		outputs = {}
		# Generate benchmarks
		#FLOPS
		if vector_op != "fma":
			result =  subprocess.run(["./GPU/Bench/Bench", "--test", "FLOPS","--target", "vector", "--arch", str(arch), "--operation", vector_op, "--precision", precision, "--compute", str(compute_capability),"--threads", str(threads), "--blocks", str(blocks), "--device", str(DEVICE)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			if result.returncode != 0:
				print(result.stderr.decode('utf-8').rstrip())
				sys.exit(5)

			result = subprocess.run(["./GPU/bin/test"], stdout=subprocess.PIPE)
			if result.returncode != 0:
				print(result.stdout.decode('utf-8').rstrip())
				exit(8)
			
			outputs["flops"] = result.stdout.decode('utf-8').split(' ')[0]
			print("Performance(" + precision + ", " + vector_op + "): ", result.stdout.decode('utf-8').rstrip())


		# Always execute FMA
		result =  subprocess.run(["./GPU/Bench/Bench", "--test", "FLOPS","--target", "vector", "--arch", str(arch), "--operation", "fma", "--precision", precision, "--compute", str(compute_capability),"--threads", str(threads), "--blocks", str(blocks), "--device", str(DEVICE)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		if result.returncode != 0:
			print(result.stderr.decode('utf-8').rstrip())
			sys.exit(5)

		result = subprocess.run(["./GPU/bin/test"], stdout=subprocess.PIPE)
		if result.returncode != 0:
			print(result.stdout.decode('utf-8').rstrip())
			exit(8)
		
		outputs["fma"] = result.stdout.decode('utf-8').split(' ')[0]
		print("Performance(" + precision + ", " + "fma" + "): ", result.stdout.decode('utf-8').rstrip())

		#MEM Shared
		result =  subprocess.run(["./GPU/Bench/Bench", "--test", "MEM","--target", "shared", "--precision", precision, "--compute", str(compute_capability), "--threads", str(threads), "--blocks", str(blocks), "--device", str(DEVICE)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		if result.returncode != 0:
			print(result.stderr.decode('utf-8').rstrip())
			sys.exit(9)

		result = subprocess.run(["./GPU/bin/test"], stdout=subprocess.PIPE)
		if result.returncode != 0:
			print(result.stdout.decode('utf-8').rstrip())
			exit(10)

		outputs["shared"] = result.stdout.decode('utf-8').split(' ')[0]
		print("Bandwith Shared Memory(" + precision + "): ", result.stdout.decode('utf-8').rstrip())

		# MEM L2
		result =  subprocess.run(["./GPU/Bench/Bench", "--test", "MEM","--target", "L2", "--precision", precision, "--compute", str(compute_capability), "--threads", str(threads), "--blocks", str(blocks), "--device", str(DEVICE)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		if result.returncode != 0:
			print(result.stderr.decode('utf-8').rstrip())
			sys.exit(23)

		result = subprocess.run(["./GPU/bin/test"], stdout=subprocess.PIPE)
		if result.returncode != 0:
			print(result.stdout.decode('utf-8').rstrip())
			exit(24)

		outputs["l2"] = result.stdout.decode('utf-8').split(' ')[0]
		print("Bandwith L2 Cache(" + precision +"): ", result.stdout.decode('utf-8').rstrip())

		#MEM Global
		result =  subprocess.run(["./GPU/Bench/Bench", "--test", "MEM","--target", "global", "--precision", precision, "--compute", str(compute_capability), "--threads", str(threads), "--blocks", str(blocks), "--device", str(DEVICE)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		if result.returncode != 0:
			print(result.stderr.decode('utf-8').rstrip())
			sys.exit(9)

		result = subprocess.run(["./GPU/bin/test"], stdout=subprocess.PIPE)
		if result.returncode != 0:
			print(result.stdout.decode('utf-8').rstrip())
			exit(10)
			
		outputs["global"] = result.stdout.decode('utf-8').split(' ')[0]
		print("Bandwith Global Memory(" + precision + "): ", result.stdout.decode('utf-8').rstrip())


		# Save results
		if out == './Results':
			if os.path.isdir('Results') == False:
				os.mkdir('Results')
			if os.path.isdir('Results/Roofline') == False:
				os.mkdir('Results/Roofline')
		else:
			if os.path.isdir(out):
				if os.path.isdir(out + "/Roofline") == False:
					os.mkdir(out + "/Roofline")
			else:
				print("ERROR: Provided output path does not exist")

		ct = datetime.datetime.now()
		date = ct.strftime('%Y-%m-%d %H:%M:%S')
		update_csv(name, "Roofline", outputs, date, "cuda" if (arch == "nvidia") else "vector-amd", precision, vector_op, threads, blocks, out)
		print("--------------------------------------------------")


	# Tensor Core benchmarks
	for precision in target_tensor:
		if (arch == "amd"):
			break
		outputs = {}
		# TENSOR FLOPS
		result =  subprocess.run(["./GPU/Bench/Bench", "--test", "FLOPS","--target", "tensor", "--precision", precision, "--compute", str(compute_capability),"--threads", str(threads), "--blocks", str(blocks), "--device", str(DEVICE)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		if result.returncode != 0:
			print(result.stderr.decode('utf-8').rstrip())
			sys.exit(16)
		
		result = subprocess.run(["./GPU/bin/test"], stdout=subprocess.PIPE)
		if result.returncode != 0:
			print(result.stdout.decode('utf-8').rstrip())
			exit(17)
		
		outputs["flops"] = result.stdout.decode('utf-8').split(' ')[0]
		print("Performance Tensor(" + precision + "): ", result.stdout.decode('utf-8').rstrip())

		#MEM Shared
		result =  subprocess.run(["./GPU/Bench/Bench", "--test", "MEM","--target", "shared", "--precision", precision, "--compute", str(compute_capability), "--threads", str(threads), "--blocks", str(blocks), "--device", str(DEVICE)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		if result.returncode != 0:
			print(result.stderr.decode('utf-8').rstrip())
			sys.exit(18)

		result = subprocess.run(["./GPU/bin/test"], stdout=subprocess.PIPE)
		if result.returncode != 0:
			print(result.stdout.decode('utf-8').rstrip())
			exit(19)

		outputs["shared"] = result.stdout.decode('utf-8').split(' ')[0]
		print("Bandwith Shared Memory(" + precision + "): ", result.stdout.decode('utf-8').rstrip())

		# MEM L2
		result =  subprocess.run(["./GPU/Bench/Bench", "--test", "MEM","--target", "L2", "--precision", precision, "--compute", str(compute_capability), "--device", str(DEVICE)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		if result.returncode != 0:
			print(result.stderr.decode('utf-8').rstrip())
			sys.exit(23)

		result = subprocess.run(["./GPU/bin/test"], stdout=subprocess.PIPE)
		if result.returncode != 0:
			print(result.stdout.decode('utf-8').rstrip())
			exit(24)

		outputs["l2"] = result.stdout.decode('utf-8').split(' ')[0]
		print("Bandwith L2 Cache(" + precision +"): ", result.stdout.decode('utf-8').rstrip())

		#MEM Global
		result =  subprocess.run(["./GPU/Bench/Bench", "--test", "MEM","--target", "global", "--precision", precision, "--compute", str(compute_capability), "--threads", str(threads), "--blocks", str(blocks), "--device", str(DEVICE)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		if result.returncode != 0:
			print(result.stderr.decode('utf-8').rstrip())
			sys.exit(20)

		result = subprocess.run(["./GPU/bin/test"], stdout=subprocess.PIPE)
		if result.returncode != 0:
			print(result.stdout.decode('utf-8').rstrip())
			exit(21)
			
		outputs["global"] = result.stdout.decode('utf-8').split(' ')[0]
		print("Bandwith Global Memory(" + precision + "): ", result.stdout.decode('utf-8').rstrip())

				# Save results
		if out == './Results':
			if os.path.isdir('Results') == False:
				os.mkdir('Results')
			if os.path.isdir('Results/Roofline') == False:
				os.mkdir('Results/Roofline')
		else:
			if os.path.isdir(out):
				if os.path.isdir(out + "/Roofline") == False:
					os.mkdir(out + "/Roofline")
			else:
				print("ERROR: Provided output path does not exist")

		ct = datetime.datetime.now()
		date = ct.strftime('%Y-%m-%d %H:%M:%S')
		update_csv(name, "Roofline", outputs, date, "tensor", precision, "mma", threads, blocks, out)
		print("--------------------------------------------------")


def shutdown(set_freq):
	if set_freq:
		result = subprocess.run(['nvidia-smi', '-i', str(DEVICE), '-pm', '0'], stdout=subprocess.PIPE)
		if result.returncode != 0:
			print("Important: It was not possible to disable Persistent Mode in GPU", DEVICE)


def main():
	# Parse arguments
	parser = argparse.ArgumentParser(description='Script to run GPU micro-benchmarks to construct the Cache-Aware Roofline Model for GPUs')
	parser.add_argument('--test', default='roofline', nargs= '?', choices=['FP', 'TC', 'roofline', 'MEM'], help='Type of test.Type of the test. Roofline test measures the bandwidth of the different memory levels and FP Performance, MEM test measures the bandwidth of various memory sizes, mixed test measures bandwidth and FP performance for a combination of memory acceses (to L1, L2, L3, or DRAM) and FP operations (Default: roofline) ')
	parser.add_argument('--name', default='unnamed', nargs= '?', help='Name of the GPU to be tested (if not using config file)')
	parser.add_argument('config', nargs='?', help='Path to the configuration file')
	parser.add_argument('-v', '--verbose', default=1, nargs='?', type=int, choices=[0, 1, 2, 3], help='Level of terminal output (0 -> No Output 1 -> Only Errors and Test Details, 2 -> Intermediate Test Results, 3 -> Configuration Values Selected/Detected)')
	parser.add_argument('-out', '--output', default='./Results', nargs='?', help='Path to the output directory')

	parser.add_argument('--freq_sm', dest='freq_sm', default=0, nargs='?', type = int, help='Desired SM frequency during test')
	parser.add_argument('--freq_mem', dest='freq_mem', default=0, nargs='?', type=int, help='Desired MEM frequency during test')
	parser.add_argument('--set_freq',  dest='set_freq', action='store_const', const=1, default=0, help='Set SM and MEM frequency to indicated one')

	parser.add_argument('--vector', default=['auto'], nargs='+', choices=['none','auto','hp', 'int', 'sp', 'dp', 'bf16'], help='Set of CUDA core arithmetic precisions to test. If auto, all will be tested.')
	parser.add_argument('--tensor', default=['auto'], nargs='+', choices=['none','auto', 'fp16_32', 'fp16_16', 'tf32', 'bf16', 'int8', 'int4', 'int1'], help='Set of Tensor Core arithmetic precisions to test. If auto, all will be tested.')
	parser.add_argument('--vector_op', dest='vector_op', default='add', nargs='?', choices=['fma', 'add', 'mul'], help="Desired operation to execute in CUDA Cores.")

	parser.add_argument('--threads', default=1024, nargs='?', type=int, help='Num of threads per block to execute in the benchmarks')
	parser.add_argument('--blocks', default=32768, nargs='?', type=int, help='Number of thread blocks to execute in the benchmarks')

	args = parser.parse_args()

	name = ''

	# Read configuration file
	if args.config != None:
		name = read_config(args.config)

	if name == '':
		name = args.name

	# Test if GPU is available
	arch = ''
	try:
		subprocess.run('nvidia-smi', stdout=subprocess.PIPE)
		print('NVIDIA GPU detected')
		arch = 'nvidia'

	except Exception:
		print('NVIDIA GPU not detected')
		try:
			subprocess.run('amd-smi', stdout=subprocess.PIPE)
			print('AMD GPU detected')
			arch = 'amd'

		except Exception:
			print('AMD GPU not detected. This tool requires the amd-smi CLI.')
			sys.exit(1)

	run_roofline(args.verbose, args.name, args.output, args.set_freq, args.freq_sm, args.freq_mem, arch, args.vector, args.tensor, args.vector_op, args.threads, args.blocks)

	shutdown(args.set_freq)

if __name__ == '__main__':
	main()
