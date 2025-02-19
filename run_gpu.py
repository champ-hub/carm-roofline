import argparse
import os
import subprocess
import datetime # check if used
import platform # check if used
import sys # check if used

# TODO: use some config file or option to define target GPU
device_id = 0

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

def check_hardware(verbose,freq, set_freq):
	compute_capability = 0
	gpu_name = ''
	cuda_precisions = ['scalar', 'hp', 'sp', 'dp']
	tensor_cores = False
	tensor_core_precisions = []

	# Obtain compute capability
	result = subprocess.run(['nvidia-smi','--query-gpu=compute_cap', '-i', str(device_id), '--format=csv,noheader'], stdout=subprocess.PIPE)
	result = result.stdout.decode('utf-8').rstrip().split('.')
	compute_capability = int(result[0])*10+int(result[1])

	# Obtain GPU name to see if it's GTX (necessary as GTX 1660 is compute capability 75 and does not have TC)
	result = subprocess.run(['nvidia-smi','--query-gpu=gpu_name', '-i', str(device_id), '--format=csv,noheader'], stdout=subprocess.PIPE)
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
				cuda_precisions.append('bf16')

				if compute_capability >= 89:
					pass #implement fp8
	
	if verbose > 2:
		print("-----------------GPU INFORMATION-----------------")
		print("GPU:", gpu_name)
		print("Compute Capability:", compute_capability)
		print("Supported CUDA Precisions:", ', '.join(cuda_precisions))
		if tensor_cores:
			print("Supported Tensor Core Precisions:", ', '.join(tensor_core_precisions))



def run_roofline(verbose):
	check_hardware(verbose,0,0)





def main():
	# Parse arguments
	parser = argparse.ArgumentParser(description='Script to run GPU micro-benchmarks to construct the Cache-Aware Roofline Model for GPUs')
	parser.add_argument('--test', default='roofline', nargs= '?', choices=['FP', 'TC', 'roofline', 'MEM'], help='Type of test.Type of the test. Roofline test measures the bandwidth of the different memory levels and FP Performance, MEM test measures the bandwidth of various memory sizes, mixed test measures bandwidth and FP performance for a combination of memory acceses (to L1, L2, L3, or DRAM) and FP operations (Default: roofline) ')
	parser.add_argument('--name', default='unnnamed_gpu', nargs= '?', help='Name of the GPU to be tested (if not using config file)')
	parser.add_argument('config', nargs='?', help='Path to the configuration file')
	parser.add_argument('-v', '--verbose', default=1, nargs='?', type=int, choices=[0, 1, 2, 3], help='Level of terminal output (0 -> No Output 1 -> Only Errors and Test Details, 2 -> Intermediate Test Results, 3 -> Configuration Values Selected/Detected)')
	parser.add_argument('-out', '--output', default='./Results', nargs='?', help='Path to the output directory')

	args = parser.parse_args()

	name = ''

	# Read configuration file
	if args.config != None:
		name = read_config(args.config)

	if name == '':
		name = args.name

	# Test if NVIDIA GPU is available
	try:
		subprocess.run('nvidia-smi', stdout=subprocess.PIPE)
		print('NVIDIA GPU detected')

	except Exception:
		print('NVIDIA GPU not detected')
		sys.exit(1)

	run_roofline(args.verbose)

if __name__ == '__main__':
	main()
