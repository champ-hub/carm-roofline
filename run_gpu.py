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

def check_hardware(verbose, set_freq, freq_sm, freq_mem, target_cuda, target_tensor):
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

	# Check valid arithmetic precisions
	if target_cuda[0] == 'auto':
		target_cuda = cuda_precisions.copy()
	else:
		for item in target_cuda:
			if item not in cuda_precisions:
				print("WARNING: Selected CUDA arithmetic precision", item, "was detected and removed since it is not supported by the tested GPU.")
				target_cuda.remove(item)
	
	if target_tensor[0] == 'auto':
		target_tensor = tensor_core_precisions.copy()
	else:
		for item in target_tensor:
			if item not in tensor_core_precisions:
				print("WARNING: Selected Tensor Core arithmetic precision", item, "was detected and removed since it is not supported by the tested GPU.")
				target_tensor.remove(item)
	

	# Configure SM and MEM frequency
	if set_freq and freq_sm !=0 and freq_mem != 0:
		# Turn on persistency mode
		result = subprocess.run(['nvidia-smi', '-i', str(device_id), '-pm', '1'], stdout=subprocess.PIPE)
		
		if result.returncode != 0:
			print(result.stdout.decode('utf-8').rstrip())
			sys.exit(2)
		
		# Configure SM freqency
		result = subprocess.run(['nvidia-smi', '-i', str(device_id), '-lgc', str(freq_sm)], stdout=subprocess.PIPE)

		if result.returncode != 0:
			print(result.stdout.decode('utf-8').rstrip())
			sys.exit(3)
		
		# Configure MEM frequency
		result = subprocess.run(['nvidia-smi', '-i', str(device_id), '-lmc', str(freq_mem)], stdout=subprocess.PIPE)

		if result.returncode != 0:
			print(result.stdout.decode('utf-8').rstrip())
			sys.exit(4)
		
	
	if verbose > 2:
		print("-----------------GPU INFORMATION-----------------")
		print("GPU:", gpu_name)
		print("Compute Capability:", compute_capability)
		print("Supported CUDA Precisions:", ', '.join(cuda_precisions))
		print("Tested CUDA Precisions:", ', '.join(target_cuda))
		if tensor_cores:
			print("Supported Tensor Core Precisions:", ', '.join(tensor_core_precisions))
			print("Tested Tensor Core Precisions:", ', '.join(target_tensor))
		
		# Print current GPU frequencies
		result = subprocess.run(['nvidia-smi', '--query-gpu=clocks.sm,clocks.mem', '-i', str(device_id), '--format=csv,noheader'], stdout=subprocess.PIPE)
		result = result.stdout.decode('utf-8').rstrip().split(' ')
		real_freq_sm = result[0]
		real_freq_mem = result[2]
		print("Current SM Frequency:", real_freq_sm, "MHz")
		print("Current Memory Frequency:", real_freq_mem, "MHz")

	return compute_capability, target_cuda, target_tensor



def run_roofline(verbose, set_freq, freq_sm, freq_mem, target_cuda, target_tensor):
	compute_capability, target_cuda, target_tensor = check_hardware(verbose, set_freq, freq_sm, freq_mem, target_cuda, target_tensor)

	if verbose == 1:
		print("------------------------------")
		print("Running Benchmarks for the CUDA Core Precisions", target_cuda)
		if not target_tensor:
			print("Tensor Cores are not supported in this device.")
		else:
			print("On the Following Tensor Core Precisions: ", target_tensor)
		print("------------------------------")

	# Compile benchmark generator
	os.system("cd GPU && make clean && make")

	# Cuda Core benchmarks
	for precision in target_cuda:
		pass

	# Tensor Core benchmarks
	for precision in target_tensor:
		pass


def shutdown(set_freq):
	if set_freq:
		result = subprocess.run(['nvidia-smi', '-i', str(device_id), '-pm', '0'], stdout=subprocess.PIPE)
		if result.returncode != 0:
			print("Important: It was not possible to disable Persistent Mode in GPU", device_id)


def main():
	# Parse arguments
	parser = argparse.ArgumentParser(description='Script to run GPU micro-benchmarks to construct the Cache-Aware Roofline Model for GPUs')
	parser.add_argument('--test', default='roofline', nargs= '?', choices=['FP', 'TC', 'roofline', 'MEM'], help='Type of test.Type of the test. Roofline test measures the bandwidth of the different memory levels and FP Performance, MEM test measures the bandwidth of various memory sizes, mixed test measures bandwidth and FP performance for a combination of memory acceses (to L1, L2, L3, or DRAM) and FP operations (Default: roofline) ')
	parser.add_argument('--name', default='unnnamed_gpu', nargs= '?', help='Name of the GPU to be tested (if not using config file)')
	parser.add_argument('config', nargs='?', help='Path to the configuration file')
	parser.add_argument('-v', '--verbose', default=1, nargs='?', type=int, choices=[0, 1, 2, 3], help='Level of terminal output (0 -> No Output 1 -> Only Errors and Test Details, 2 -> Intermediate Test Results, 3 -> Configuration Values Selected/Detected)')
	parser.add_argument('-out', '--output', default='./Results', nargs='?', help='Path to the output directory')

	parser.add_argument('--freq_sm', dest='freq_sm', default=0, nargs='?', type = int, help='Desired SM frequency during test')
	parser.add_argument('--freq_mem', dest='freq_mem', default=0, nargs='?', type=int, help='Desired MEM frequency during test')
	parser.add_argument('--set_freq',  dest='set_freq', action='store_const', const=1, default=0, help='Set SM and MEM frequency to indicated one')

	parser.add_argument('--cuda', default=['auto'], nargs='+', choices=['auto','hp', 'scalar', 'sp', 'dp', 'bf16'], help='Set of CUDA core arithmetic precisions to test. If auto, all will be tested.')
	parser.add_argument('--tensor', default=['auto'], nargs='+', choices=['auto', 'fp16_32', 'fp16_16', 'tf32', 'bf16', 'int8', 'int4', 'int1'], help='Set of Tensor Core arithmetic precisions to test. If auto, all will be tested.')

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

	run_roofline(args.verbose, args.set_freq, args.freq_sm, args.freq_mem, args.cuda, args.tensor)

	shutdown(args.set_freq)

if __name__ == '__main__':
	main()
