import argparse
import subprocess
import os
import datetime
import time
import re
import sys
import platform

import utils as ut

if not hasattr(time, 'time_ns'):
    time.time_ns = lambda: int(time.time() * 1e9)

#FP OPERATIONS
x86_Scalar_fp_operations = {
    "vmulsd": {"count": 0, "string": "Scalar (1x 64 bit)", "factor": 1},
    "mulsd": {"count": 0, "string": "Scalar (1x 64 bit)", "factor": 1},
    "vdivsd": {"count": 0, "string": "Scalar (1x 64 bit)", "factor": 1},
    "divsd": {"count": 0, "string": "Scalar (1x 64 bit)", "factor": 1},
    "vaddsd": {"count": 0, "string": "Scalar (1x 64 bit)", "factor": 1},
    "addsd": {"count": 0, "string": "Scalar (1x 64 bit)", "factor": 1},
    "vsubsd": {"count": 0, "string": "Scalar (1x 64 bit)", "factor": 1},
    "subsd": {"count": 0, "string": "Scalar (1x 64 bit)", "factor": 1},
    "vfmadd132sd": {"count": 0, "string": "Scalar (2x 64 bit)", "factor": 2},
    "vfmadd231sd": {"count": 0, "string": "Scalar (2x 64 bit)", "factor": 2},
    "vfmadd213sd": {"count": 0, "string": "Scalar (2x 64 bit)", "factor": 2},
    "vdivss": {"count": 0, "string": "Scalar (1x 32 bit)", "factor": 1},
    "divss": {"count": 0, "string": "Scalar (1x 32 bit)", "factor": 1},
    "vaddss": {"count": 0, "string": "Scalar (1x 32 bit)", "factor": 1},
    "addss": {"count": 0, "string": "Scalar (1x 32 bit)", "factor": 1},
    "vsubss": {"count": 0, "string": "Scalar (1x 32 bit)", "factor": 1},
    "subss": {"count": 0, "string": "Scalar (1x 32 bit)", "factor": 1},  
    "vmulss": {"count": 0, "string": "Scalar (1x 32 bit)", "factor": 1},
    "mulss": {"count": 0, "string": "Scalar (1x 32 bit)", "factor": 1},
    "vfmadd132ss": {"count": 0, "string": "Scalar (2x 32 bit)", "factor": 2},
    "vfmadd231ss": {"count": 0, "string": "Scalar (2x 32 bit)", "factor": 2},
    "vfmadd213ss": {"count": 0, "string": "Scalar (2x 32 bit)", "factor": 2},
}

x86_SSE_fp_operations = {
    "divpd": {"count": 0, "string": "SSE (2x 64 bit)", "factor": 2},
    "addpd": {"count": 0, "string": "SSE (2x 64 bit)", "factor": 2},
    "subpd": {"count": 0, "string": "SSE (2x 64 bit)", "factor": 2},
    "mulpd": {"count": 0, "string": "SSE (2x 64 bit)", "factor": 2},
    "vfmadd132pd": {"count": 0, "string": "SSE (4x 64 bit)", "factor": 4},
    "vfmadd231pd": {"count": 0, "string": "SSE (4x 64 bit)", "factor": 4},
    "vfmadd213pd": {"count": 0, "string": "SSE (4x 64 bit)", "factor": 4},
    "vfnmadd132pd": {"count": 0, "string": "SSE (4x 64 bit)", "factor": 4},
    "vfnmadd231pd": {"count": 0, "string": "SSE (4x 64 bit)", "factor": 4},
    "vfnmadd213pd": {"count": 0, "string": "SSE (4x 64 bit)", "factor": 4},
    "divps": {"count": 0, "string": "SSE (4x 32 bit)", "factor": 4},
    "addps": {"count": 0, "string": "SSE (4x 32 bit)", "factor": 4},
    "subps": {"count": 0, "string": "SSE (4x 32 bit)", "factor": 4},
    "vfmsub132ps": {"count": 0, "string": "SSE (8x 32 bit)", "factor": 8},
    "vfmsub231ps": {"count": 0, "string": "SSE (8x 32 bit)", "factor": 8},
    "vfmsub213ps": {"count": 0, "string": "SSE (8x 32 bit)", "factor": 8},
    "mulps": {"count": 0, "string": "SSE (4x 32 bit)", "factor": 4},
    "vfmadd132ps": {"count": 0, "string": "SSE (8x 32 bit)", "factor": 8},
    "vfmadd231ps": {"count": 0, "string": "SSE (8x 32 bit)", "factor": 8},
    "vfmadd213ps": {"count": 0, "string": "SSE (8x 32 bit)", "factor": 8},
    "vfnmadd132ps": {"count": 0, "string": "SSE (8x 32 bit)", "factor": 8},
    "vfnmadd231ps": {"count": 0, "string": "SSE (8x 32 bit)", "factor": 8},
    "vfnmadd213ps": {"count": 0, "string": "SSE (8x 32 bit)", "factor": 8},
    "vdivpd": {"count": 0, "string": "SSE (2x 64 bit)", "factor": 2},
    "vaddpd": {"count": 0, "string": "SSE (2x 64 bit)", "factor": 2},
    "vsubpd": {"count": 0, "string": "SSE (2x 64 bit)", "factor": 2},
    "vmulpd": {"count": 0, "string": "SSE (2x 64 bit)", "factor": 2},
    "vdivps": {"count": 0, "string": "SSE (4x 32 bit)", "factor": 4},
    "vaddps": {"count": 0, "string": "SSE (4x 32 bit)", "factor": 4},
    "vsubps": {"count": 0, "string": "SSE (4x 32 bit)", "factor": 4},
    "vmulps": {"count": 0, "string": "SSE (4x 32 bit)", "factor": 4},
}

x86_AVX2_fp_operations = {
    "vdivpd": {"count": 0, "string": "AVX2 (4x 64 bit)", "factor": 4},
    "vaddpd": {"count": 0, "string": "AVX2 (4x 64 bit)", "factor": 4},
    "vsubpd": {"count": 0, "string": "AVX2 (4x 64 bit)", "factor": 4},
    "vmulpd": {"count": 0, "string": "AVX2 (4x 64 bit)", "factor": 4},
    "vfmadd132pd": {"count": 0, "string": "AVX2 (8x 64 bit)", "factor": 8},
    "vfmadd231pd": {"count": 0, "string": "AVX2 (8x 64 bit)", "factor": 8},
    "vfmadd213pd": {"count": 0, "string": "AVX2 (8x 64 bit)", "factor": 8},
    "vfnmadd132pd": {"count": 0, "string": "AVX2 (8x 64 bit)", "factor": 8},
    "vfnmadd231pd": {"count": 0, "string": "AVX2 (8x 64 bit)", "factor": 8},
    "vfnmadd213pd": {"count": 0, "string": "AVX2 (8x 64 bit)", "factor": 8},
    "vdivps": {"count": 0, "string": "AVX2 (8x 32 bit)", "factor": 8},
    "vaddps": {"count": 0, "string": "AVX2 (8x 32 bit)", "factor": 8},
    "vsubps": {"count": 0, "string": "AVX2 (8x 32 bit)", "factor": 8},
    "vfmsub132ps": {"count": 0, "string": "AVX2 (16x 32 bit)", "factor": 16},
    "vfmsub231ps": {"count": 0, "string": "AVX2 (16x 32 bit)", "factor": 16},
    "vfmsub213ps": {"count": 0, "string": "AVX2 (16x 32 bit)", "factor": 16},
    "vmulps": {"count": 0, "string": "AVX2 (8x 32 bit)", "factor": 8},
    "vfmadd132ps": {"count": 0, "string": "AVX2 (16x 32 bit)", "factor": 16},
    "vfmadd231ps": {"count": 0, "string": "AVX2 (16x 32 bit)", "factor": 16},
    "vfmadd213ps": {"count": 0, "string": "AVX2 (16x 32 bit)", "factor": 16},
    "vfnmadd132ps": {"count": 0, "string": "AVX2 (16x 32 bit)", "factor": 16},
    "vfnmadd231ps": {"count": 0, "string": "AVX2 (16x 32 bit)", "factor": 16},
    "vfnmadd213ps": {"count": 0, "string": "AVX2 (16x 32 bit)", "factor": 16}
}

x86_AVX512_fp_operations = {
    "vdivpd": {"count": 0, "string": "AVX512 (8x 64 bit)", "factor": 8},
    "vaddpd": {"count": 0, "string": "AVX512 (8x 64 bit)", "factor": 8},
    "vsubpd": {"count": 0, "string": "AVX512 (8x 64 bit)", "factor": 8},
    "vmulpd": {"count": 0, "string": "AVX512 (8x 64 bit)", "factor": 8},
    "vfmadd132pd": {"count": 0, "string": "AVX512 (16x 64 bit)", "factor": 16},
    "vfmadd231pd": {"count": 0, "string": "AVX512 (16x 64 bit)", "factor": 16},
    "vfmadd213pd": {"count": 0, "string": "AVX512 (16x 64 bit)", "factor": 16},
    "vfnmadd132pd": {"count": 0, "string": "AVX512 (16x 64 bit)", "factor": 16},
    "vfnmadd231pd": {"count": 0, "string": "AVX512 (16x 64 bit)", "factor": 16},
    "vfnmadd213pd": {"count": 0, "string": "AVX512 (16x 64 bit)", "factor": 16},
    "vdivps": {"count": 0, "string": "AVX512 (16x 32 bit)", "factor": 16},
    "vaddps": {"count": 0, "string": "AVX512 (16x 32 bit)", "factor": 16},
    "vsubps": {"count": 0, "string": "AVX512 (16x 32 bit)", "factor": 16},
    "vfmsub132ps": {"count": 0, "string": "AVX512 (32x 32 bit)", "factor": 32},
    "vfmsub231ps": {"count": 0, "string": "AVX512 (32x 32 bit)", "factor": 32},
    "vfmsub213ps": {"count": 0, "string": "AVX512 (32x 32 bit)", "factor": 32},
    "vmulps": {"count": 0, "string": "AVX512 (16x 32 bit)", "factor": 16},
    "vfmadd132ps": {"count": 0, "string": "AVX512 (32x 32 bit)", "factor": 32},
    "vfmadd231ps": {"count": 0, "string": "AVX512 (32x 32 bit)", "factor": 32},
    "vfmadd213ps": {"count": 0, "string": "AVX512 (32x 32 bit)", "factor": 32},
    "vfnmadd132ps": {"count": 0, "string": "AVX512 (32x 32 bit)", "factor": 32},
    "vfnmadd231ps": {"count": 0, "string": "AVX512 (32x 32 bit)", "factor": 32},
    "vfnmadd213ps": {"count": 0, "string": "AVX512 (32x 32 bit)", "factor": 32}
}

#INTEGER OPERATIONS
x86_Scalar_int_operations = {
    "add": {"count": 0, "string": "Scalar (1x 32 bit)", "factor": 1},
    "imul": {"count": 0, "string": "Scalar (1x 32 bit)", "factor": 1},
    "sub": {"count": 0, "string": "Scalar (1x 32 bit)", "factor": 1},
    "mul": {"count": 0, "string": "Scalar (1x 32 bit)", "factor": 1},
    "div": {"count": 0, "string": "Scalar (1x 32 bit)", "factor": 1},
    "idiv": {"count": 0, "string": "Scalar (1x 32 bit)", "factor": 1},
    "xadd": {"count": 0, "string": "Scalar (1x 32 bit)", "factor": 1},
    
}

x86_SSE_int_operations = {
    "paddq": {"count": 0, "string": "SSE (2x 64 bit)", "factor": 2},
    "paddd": {"count": 0, "string": "SSE (4x 32 bit)", "factor": 4},
    "paddw": {"count": 0, "string": "SSE (8x 16 bit)", "factor": 8},
    "paddb": {"count": 0, "string": "SSE (16x 8 bit)", "factor": 16},
    "psubq": {"count": 0, "string": "SSE (2x 64 bit)", "factor": 2},
    "psubd": {"count": 0, "string": "SSE (4x 32 bit)", "factor": 4},
    "psubw": {"count": 0, "string": "SSE (8x 16 bit)", "factor": 8},
    "psubb": {"count": 0, "string": "SSE (16x 8 bit)", "factor": 16},
    "pdivq": {"count": 0, "string": "SSE (2x 64 bit)", "factor": 2},
    "pdivd": {"count": 0, "string": "SSE (4x 32 bit)", "factor": 4},
    "pdivw": {"count": 0, "string": "SSE (8x 16 bit)", "factor": 8},
    "pdivb": {"count": 0, "string": "SSE (16x 8 bit)", "factor": 16},
    "vpaddq": {"count": 0, "string": "SSE (2x 64 bit)", "factor": 2},
    "vpaddd": {"count": 0, "string": "SSE (4x 32 bit)", "factor": 4},
    "vpaddw": {"count": 0, "string": "SSE (8x 16 bit)", "factor": 8},
    "vpaddb": {"count": 0, "string": "SSE (16x 8 bit)", "factor": 16},
    "vpsubq": {"count": 0, "string": "SSE (2x 64 bit)", "factor": 2},
    "vpsubd": {"count": 0, "string": "SSE (4x 32 bit)", "factor": 4},
    "vpsubw": {"count": 0, "string": "SSE (8x 16 bit)", "factor": 8},
    "vpsubb": {"count": 0, "string": "SSE (16x 8 bit)", "factor": 16},
    "vpdivq": {"count": 0, "string": "SSE (2x 64 bit)", "factor": 2},
    "vpdivd": {"count": 0, "string": "SSE (4x 32 bit)", "factor": 4},
    "vpdivw": {"count": 0, "string": "SSE (8x 16 bit)", "factor": 8},
    "vpdivb": {"count": 0, "string": "SSE (16x 8 bit)", "factor": 16},
}

x86_AVX2_int_operations = {
    "vpaddq": {"count": 0, "string": "AVX2 (4x 64 bit)", "factor": 4},
    "vpaddd": {"count": 0, "string": "AVX2 (8x 32 bit)", "factor": 8},
    "vpaddw": {"count": 0, "string": "AVX2 (16x 16 bit)", "factor": 16},
    "vpaddb": {"count": 0, "string": "AVX2 (32x 8 bit)", "factor": 32},
    "vpsubq": {"count": 0, "string": "AVX2 (4x 64 bit)", "factor": 4},
    "vpsubd": {"count": 0, "string": "AVX2 (8x 32 bit)", "factor": 8},
    "vpsubw": {"count": 0, "string": "AVX2 (16x 16 bit)", "factor": 16},
    "vpsubb": {"count": 0, "string": "AVX2 (32x 8 bit)", "factor": 32},
    "vpdivq": {"count": 0, "string": "AVX2 (4x 64 bit)", "factor": 4},
    "vpdivd": {"count": 0, "string": "AVX2 (8x 32 bit)", "factor": 8},
    "vpdivw": {"count": 0, "string": "AVX2 (16x 16 bit)", "factor": 16},
    "vpdivb": {"count": 0, "string": "AVX2 (32x 8 bit)", "factor": 32},
}

x86_AVX512_int_operations = {
    "vpaddq": {"count": 0, "string": "AVX512 (8x 64 bit)", "factor": 8},
    "vpaddd": {"count": 0, "string": "AVX512 (16x 32 bit)", "factor": 16},
    "vpaddw": {"count": 0, "string": "AVX512 (32x 16 bit)", "factor": 32},
    "vpaddb": {"count": 0, "string": "AVX512 (64x 8 bit)", "factor": 64},
    "vpsubq": {"count": 0, "string": "AVX512 (8x 64 bit)", "factor": 8},
    "vpsubd": {"count": 0, "string": "AVX512 (16x 32 bit)", "factor": 16},
    "vpsubw": {"count": 0, "string": "AVX512 (32x 16 bit)", "factor": 32},
    "vpsubb": {"count": 0, "string": "AVX512 (64x 8 bit)", "factor": 64},
    "vpdivq": {"count": 0, "string": "AVX512 (8x 64 bit)", "factor": 8},
    "vpdivd": {"count": 0, "string": "AVX512 (16x 32 bit)", "factor": 16},
    "vpdivw": {"count": 0, "string": "AVX512 (32x 16 bit)", "factor": 32},
    "vpdivb": {"count": 0, "string": "AVX512 (64x 8 bit)", "factor": 64},
}

ARM_FP_operations = {}
ARM_INT_operations = {}

memory_operations = {}

x86_not_supported = {}
misc_operations = {}

#Check if SDE is present
def check_sde_exists(path_sde):
    sde_exec = os.path.join(path_sde, "sde64")
    #Check for SDE folder
    if os.path.exists(sde_exec):
        print(f"SDE executable found in: '{path_sde}'.")
        return True
        
    else:
        print(f"No SDE folder found in: '{sde_exec}'.")
        return False

#Check if DynamoRIO Client is present
def check_client_exists(path):
    
    drrun_path = os.path.join(path, 'bin64', 'drrun')

    #Check for the existence of the 'drrun' executable
    if os.path.exists(drrun_path):
        print(f"DynamoRIO executable 'drrun' found in: '{drrun_path}'.")
    else:
        print(f"No DynamoRIO executable 'drrun' found in: '{drrun_path}'.")
        return False

    #Check for build folder
    script_dir = os.path.dirname(os.path.abspath(__file__))

    cmake_command = f"cmake -DDynamoRIO_DIR={path}/cmake {script_dir}/CustomClient"
    
    

    #Construct the path to the build folder
    build_dir = "./carm_dbi_build"

    if os.path.exists("./carm_dbi_build"):
        print(f"The build folder is present.")
    else:
        print(f"The build folder does not exist. Building the DynamoRIO Client.")
        try:
            subprocess.run(f"mkdir carm_dbi_build && cd carm_dbi_build && {cmake_command} && make opcoder", check=True, shell=True)
        except subprocess.CalledProcessError as e:
            print("Error executing the command:", e)
    
    #Construct the path to the client file
    path_client = os.path.join(build_dir, "bin/libopcoder.so")

    #Check if the client file exists
    if os.path.exists(path_client):
        print(f"The opcode client exists in '{path_client}'.")
    else:
        print(f"The opcode client does not exist in the path '{path_client}'. Building the client.")
        try:
            subprocess.run(f"rm -rf carm_dbi_build && mkdir carm_dbi_build && cd carm_dbi_build && {cmake_command} && make opcoder", check=True, shell=True)
        except subprocess.CalledProcessError as e:
            print("Error executing the command:", e)
    return True
    
#Run SDE with provided application
def runSDE(sde_path, roi, executable_path, additional_args):
    #Construct the command with the provided paths and additional arguments
    if roi:
        command = f"{sde_path}/sde64 -iform -mix -dyn_mask_profile -start_ssc_mark FACE:repeat -stop_ssc_mark DEAD:repeat -- {executable_path}"
    else:
        command = f"{sde_path}/sde64 -iform -mix -dyn_mask_profile -- {executable_path}"

    #Add additional arguments to the command
    if additional_args != None:
        command += " " + " ".join(additional_args)
    command_args = command.split()

    print("------------------------------")
    print("Running Provided Application For Opcode Data\n")
    
    try:
        subprocess.run(command_args, check=True)
    except subprocess.CalledProcessError as e:
        print("Error executing the command:", e)


#Run DynamoRIO client with provided application
def runDynamoRIO(dynamo_path, roi, executable_path, additional_args):
    #Construct the command with the provided paths and additional arguments
    if roi:
        command = f"{dynamo_path}/bin64/drrun -c ./carm_dbi_build/bin/libopcoder.so -roi -- {executable_path}"
    else:
        command = f"{dynamo_path}/bin64/drrun -c ./carm_dbi_build/bin/libopcoder.so -- {executable_path}"

    # Add additional arguments to the command
    if additional_args != None:
        command += " " + " ".join(additional_args)
    command_args = command.split()
    print("------------------------------")
    print("Running Provided Application For Opcode Data\n")

    try:
        subprocess.run(command_args, check=True)
    except subprocess.CalledProcessError as e:
        print("Error executing the command:", e)
    
    #os.remove("timing_results.txt")

#Run provided application for timming measurements
def runApplication(roi, executable_path, additional_args):

    #Add additional arguments to the command
    if additional_args != None:
        executable_path += " " + " ".join(additional_args)
    print("\n------------------------------")
    print("Running Provided Application For Timming Data\n")
    if roi:
        try:
            subprocess.run(executable_path, check=True, shell=True)
        except subprocess.CalledProcessError as e:
            print("Error executing the command:", e)

        with open("carm_timing_results.txt", "r") as file:
            contents = file.read()
            #Extract the number of seconds
            match = re.search(r"Time Taken:\s*([\d.]+)\s*seconds", contents)
            if match:
                seconds = float(match.group(1))
            else:
                print("No match found in carm_timing_results.txt, stopping program.")
                sys.exit(1)
        file.close()

        os.remove("carm_timing_results.txt")
        return float(seconds * 1e9)
    else:
    
        try:
            start=time.time_ns()
            subprocess.run(executable_path, check=True, shell=True)
            end=time.time_ns()
        except subprocess.CalledProcessError as e:
            print("Error executing the command:", e)
        return end-start
    

def analyseSDE():
    #Regular expressions for each metric
    single_prec_flops_regex = r"Single prec\. FLOPs: (\d+)"
    double_prec_flops_regex = r"Double prec\. FLOPs: (\d+)"
    total_bytes_written_regex = r"Total bytes written: (\d+)"
    total_bytes_read_regex = r"Total bytes read: (\d+)"

    try:
        result = subprocess.run(['python3', "SDE_AI_Calculator.py"], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                print(line)

            single_prec_flops = int(re.findall(single_prec_flops_regex, result.stdout)[0])
            double_prec_flops = int(re.findall(double_prec_flops_regex, result.stdout)[0])
            total_bytes_written = int(re.findall(total_bytes_written_regex, result.stdout)[0])
            total_bytes_read = int(re.findall(total_bytes_read_regex, result.stdout)[0])

            os.remove("sde-dyn-mask-profile.txt")
            os.remove("sde-mix-out.txt")
            return single_prec_flops+double_prec_flops, total_bytes_read+total_bytes_written
        else:
            print("Error running script:", result.stderr)
            return None
    except Exception as e:
        print("Exception occurred:", e)
        return None
    

def analyseDynamoRIOx86():
    arith = False
    mem = False
    fp_ops = 0
    integer_ops = 0
    memory_bytes = 0
    with open('carm_dbi_output.txt', 'r') as file:
        for line in file:
            #Arithmetic Section       
            if "Floating Point and Integer opcode execution counts" in line:
                arith = True
                continue

            if arith:
                if "Memory opcode execution counts" in line:
                    arith = False
                    mem = True
                    continue

                parts = line.split(':')
                if len(parts) == 2:
                    count, rest = parts
                    count = count.strip()

                    #Check if the count is an integer
                    if count.isdigit():
                        count = int(count)
                        #Check if the "|" character is present in the rest of the line
                        if "|" in rest:
                            #If "|" is present, split the rest of the line based on the "|" character
                            opcode, description = rest.split("|")
                            opcode = opcode.strip()
                            description = description.strip()
                        else:
                            continue
                        if description == "Scalar":
                            if opcode in x86_Scalar_fp_operations:
                                x86_Scalar_fp_operations[opcode]["count"] += count
                                fp_ops += count*x86_Scalar_fp_operations[opcode]["factor"]
                            elif opcode in x86_Scalar_int_operations:
                                x86_Scalar_int_operations[opcode]["count"] += count
                                integer_ops += count*x86_Scalar_int_operations[opcode]["factor"]
                            else:
                                x86_not_supported[opcode] = count
                        elif description == "SSE":
                            if opcode in x86_SSE_fp_operations:
                                x86_SSE_fp_operations[opcode]["count"] += count
                                fp_ops += count*x86_SSE_fp_operations[opcode]["factor"]
                            elif opcode in x86_SSE_int_operations:
                                x86_SSE_int_operations[opcode]["count"] += count
                                integer_ops += count*x86_SSE_int_operations[opcode]["factor"]
                            else:
                                x86_not_supported[opcode] = count
                        elif description == "AVX2":
                            if opcode in x86_AVX2_fp_operations:
                                x86_AVX2_fp_operations[opcode]["count"] += count
                                fp_ops += count*x86_AVX2_fp_operations[opcode]["factor"]
                            elif opcode in x86_AVX2_int_operations:
                                x86_AVX2_int_operations[opcode]["count"] += count
                                integer_ops += count*x86_AVX2_int_operations[opcode]["factor"]
                            else:
                                x86_not_supported[opcode] = count
                        elif description == "AVX512":
                            if opcode in x86_AVX512_fp_operations:
                                x86_AVX512_fp_operations[opcode]["count"] += count
                                fp_ops += count*x86_AVX512_fp_operations[opcode]["factor"]
                            elif opcode in x86_AVX512_int_operations:
                                x86_AVX512_int_operations[opcode]["count"] += count
                                integer_ops += count*x86_AVX512_int_operations[opcode]["factor"]
                            else:
                                x86_not_supported[opcode] = count
                        else:
                            x86_not_supported[opcode] = count            
            #Memory Section
            elif mem:

                if "Miscellaneous Opcode execution counts" in line:
                    mem = False
                    continue

                line = line.strip()
                parts = line.split(':')
                if len(parts) == 2:
                    count, rest = parts
                    count = int(count.strip())
                    
                    #Check if the "|" character is present in the rest of the line
                    if "error" in rest:
                        continue
                    if "|" in rest:
                        #If "|" is present, split the rest of the line based on the "|" character
                        opcode, description = rest.split("|")
                        opcode = opcode.strip()
                        description = description.strip()
                        size, extrarest = description.split(" ")
                        memory_bytes += count*int(size)
                    else:
                        #If "|" is not present, check if the word "TOTAL" is in the rest of the line
                        if "TOTAL" in rest:
                            #If "TOTAL" is present, set the opcode to the part before "TOTAL" and description to "TOTAL"
                            parts = rest.split()
                            opcode = " ".join(parts[:-1]).strip()
                            description = "TOTAL"
                        else:
                            #If "TOTAL" is not present, set the opcode to the whole rest of the line and description to None
                            opcode = rest.strip()
                            description = None
                # Store the count, opcode, and description in the dictionary
                # Check if the opcode already exists in the dictionary
                if opcode in memory_operations:
                    #Check if the entry with the same count and description already exists
                        if (count, description) not in memory_operations[opcode]:
                            memory_operations[opcode].append((count, description))
                else:
                    #If the opcode doesn't exist, create a new list with the entry
                    memory_operations[opcode] = [(count, description)]
            #Others Section
            else:
                parts = line.split(':')
                if len(parts) == 2:
                    count, opcode = parts
                    count = count.strip()
                    opcode = opcode.strip()
                    if count.isdigit():
                        count = int(count)
                        misc_operations[opcode] = count
    return fp_ops, memory_bytes, integer_ops

def analyseDynamoRIOARM():
    arith = False
    mem = False
    fp_ops = 0
    integer_ops = 0
    memory_bytes = 0

    with open('carm_dbi_output.txt', 'r') as file:
        for line in file:
            # Arithmetic Section       
            if "Floating Point and Integer opcode execution counts" in line:
                arith = True
                continue

            if arith:
                if "Memory opcode execution counts" in line:
                    arith = False
                    mem = True
                    continue

                line = line.strip()
                parts = line.split(':')
                if len(parts) == 2:
                    count, rest = parts
                    count = int(count.strip())
                    
                    #Check if the "|" character is present in the rest of the line
                    if "error" in rest:
                        continue
                    if "|" in rest:
                        #If "|" is present, split the rest of the line based on the "|" character
                        opcode, description = rest.split("|")
                        opcode = opcode.strip()
                        description = description.strip()
                        match = re.search(r'(\d+)x', description)
                        if opcode[0] == "f":
                            if match:
                                if opcode == "fmla":
                                    fp_ops += count*int(match.group(1))*2
                                else:
                                    fp_ops += count*int(match.group(1))
                                #print("Opcode: " + opcode + "FP Count: " + str(count) + " | Factor: " +str(int(match.group(1))) + " | Total: " + str(count*int(match.group(1))))
                        else:
                            if match:
                                integer_ops += count*int(match.group(1))
                                #print("Integer Count: " + str(count) + " | Factor: " +str(int(match.group(1))) + " | Total: " + str(count*int(match.group(1))))
                    else:
                        #If "|" is not present, check if the word "TOTAL" is in the rest of the line
                        if "TOTAL" in rest:
                            #If "TOTAL" is present, set the opcode to the part before "TOTAL" and description to "TOTAL"
                            parts = rest.split()
                            opcode = " ".join(parts[:-1]).strip()
                            description = "TOTAL"
                        else:
                            #If "TOTAL" is not present, set the opcode to the whole rest of the line and description to None
                            opcode = rest.strip()
                            description = None
                #If Floating Point
                if opcode[0] == "f":

                    if opcode in ARM_FP_operations:
                        #Check if the entry with the same count and description already exists
                            if (count, description) not in ARM_FP_operations[opcode]:
                                ARM_FP_operations[opcode].append((count, description))
                    else:
                        #If the opcode doesn't exist, create a new list with the entry
                        ARM_FP_operations[opcode] = [(count, description)]
                #If Integer
                else:

                    if opcode in ARM_INT_operations:
                        #Check if the entry with the same count and description already exists
                            if (count, description) not in ARM_INT_operations[opcode]:
                                ARM_INT_operations[opcode].append((count, description))
                    else:
                        #If the opcode doesn't exist, create a new list with the entry
                        ARM_INT_operations[opcode] = [(count, description)]
                            
            #Memory Section
            elif mem:

                if "Miscellaneous Opcode execution counts" in line:
                    mem = False
                    continue

                line = line.strip()
                parts = line.split(':')
                if len(parts) == 2:
                    count, rest = parts
                    count = int(count.strip())
                    
                    #Check if the "|" character is present in the rest of the line
                    if "error" in rest:
                        continue
                    if "|" in rest:
                        #If "|" is present, split the rest of the line based on the "|" character
                        opcode, description = rest.split("|")
                        opcode = opcode.strip()
                        description = description.strip()
                        size, extrarest = description.split(" ")
                        memory_bytes += count*int(size)
                    else:
                        #If "|" is not present, check if the word "TOTAL" is in the rest of the line
                        if "TOTAL" in rest:
                            #If "TOTAL" is present, set the opcode to the part before "TOTAL" and description to "TOTAL"
                            parts = rest.split()
                            opcode = " ".join(parts[:-1]).strip()
                            description = "TOTAL"
                        else:
                            #If "TOTAL" is not present, set the opcode to the whole rest of the line and description to None
                            opcode = rest.strip()
                            description = None
                
                #Store the count, opcode, and description in the dictionary
                #Check if the opcode already exists in the dictionary
                if opcode in memory_operations:
                    #Check if the entry with the same count and description already exists
                        if (count, description) not in memory_operations[opcode]:
                            memory_operations[opcode].append((count, description))
                else:
                    #If the opcode doesn't exist, create a new list with the entry
                    memory_operations[opcode] = [(count, description)]
            else:
                parts = line.split(':')
                if len(parts) == 2:
                    count, opcode = parts
                    count = count.strip()
                    opcode = opcode.strip()

                    #Check if the count is an integer
                    if count.isdigit():
                        count = int(count)
                        misc_operations[opcode] = count
    return fp_ops, memory_bytes, integer_ops

def printDynamoRIOx86():
    global x
    print("\n---------OPCODE BREAKDOWN-----------")
    print("Memory Operations:")
    for opcode, entries in memory_operations.items():
        total_entry_printed = False
        for count, description in entries:
            if description != "TOTAL":
                if total_entry_printed:
                    #Indent subsequent entries after the TOTAL entry
                    print(f"{'':2} {count:12} : {opcode : <12} | {description}")
                else:
                    print(f"{count:12} : {opcode : <12} | {description}")
            else:
                print(f"\n{count:12} : {opcode : <12}  {description}")
                total_entry_printed = True

    #FLOATING POINT OPERATIONS        
    #AVX512 Floating Point Operations
    sorted_ops = sorted(x86_AVX512_fp_operations.items(), key=lambda item: item[1]["count"], reverse=False)
    all_zero = all(data["count"] == 0 for _, data in sorted_ops)

    if not all_zero:
        print("\nAVX512 Floating Point Operations:")

    #Print the sorted opcodes with counts greater than 0
    for opcode, data in sorted_ops:
        count = data["count"]
        description = data["string"]
        if count > 0:
            print(f"{count:12} : {opcode : <12} | {description}")


    #AVX2 Floating Point Operations
    sorted_ops = sorted(x86_AVX2_fp_operations.items(), key=lambda item: item[1]["count"], reverse=False)
    all_zero = all(data["count"] == 0 for _, data in sorted_ops)

    if not all_zero:
        print("\nAVX2 Floating Point Operations:")

    #Print the sorted opcodes with counts greater than 0
    for opcode, data in sorted_ops:
        count = data["count"]
        description = data["string"]
        if count > 0:
            print(f"{count:12} : {opcode : <12} | {description}")


    #SSE Floating Point Operations
    sorted_ops = sorted(x86_SSE_fp_operations.items(), key=lambda item: item[1]["count"], reverse=False)
    all_zero = all(data["count"] == 0 for _, data in sorted_ops)

    if not all_zero:
        print("\nSSE Floating Point Operations:")
        
    #Print the sorted opcodes with counts greater than 0
    for opcode, data in sorted_ops:
        count = data["count"]
        description = data["string"]
        if count > 0:
            print(f"{count:12} : {opcode : <12} | {description}")


    #Scalar Floating Point Operations
    sorted_ops = sorted(x86_Scalar_fp_operations.items(), key=lambda item: item[1]["count"], reverse=False)
    all_zero = all(data["count"] == 0 for _, data in sorted_ops)

    if not all_zero:
        print("\nScalar Floating Point Operations:")
        
    #Print the sorted opcodes with counts greater than 0
    for opcode, data in sorted_ops:
        count = data["count"]
        description = data["string"]
        if count > 0:
            print(f"{count:12} : {opcode : <12} | {description}")


    #INTEGER OPERATIONS
    #AVX512 Integer Operations
    sorted_ops = sorted(x86_AVX512_int_operations.items(), key=lambda item: item[1]["count"], reverse=False)
    all_zero = all(data["count"] == 0 for _, data in sorted_ops)

    if not all_zero:
        print("\nAVX512 Integer Operations:")
        
    #Print the sorted opcodes with counts greater than 0
    for opcode, data in sorted_ops:
        count = data["count"]
        description = data["string"]
        if count > 0:
            print(f"{count:12} : {opcode : <12} | {description}")


    #AVX2 Integer Operations
    sorted_ops = sorted(x86_AVX2_int_operations.items(), key=lambda item: item[1]["count"], reverse=False)
    all_zero = all(data["count"] == 0 for _, data in sorted_ops)

    if not all_zero:
        print("\nAVX2 Integer Operations:")
        
    #Print the sorted opcodes with counts greater than 0
    for opcode, data in sorted_ops:
        count = data["count"]
        description = data["string"]
        if count > 0:
            print(f"{count:12} : {opcode : <12} | {description}")


    #SSE Integer Operations
    sorted_ops = sorted(x86_SSE_int_operations.items(), key=lambda item: item[1]["count"], reverse=False)
    all_zero = all(data["count"] == 0 for _, data in sorted_ops)

    if not all_zero:
        print("\nSSE Integer Operations:")
        
    #Print the sorted opcodes with counts greater than 0
    for opcode, data in sorted_ops:
        count = data["count"]
        description = data["string"]
        if count > 0:
            print(f"{count:12} : {opcode : <12} | {description}")


    #Scalar Integer Operations
    sorted_ops = sorted(x86_Scalar_int_operations.items(), key=lambda item: item[1]["count"], reverse=False)
    all_zero = all(data["count"] == 0 for _, data in sorted_ops)

    if not all_zero:
        print("\nScalar Integer Operations:")
        
    #Print the sorted opcodes with counts greater than 0
    for opcode, data in sorted_ops:
        count = data["count"]
        description = data["string"]
        if count > 0:
            print(f"{count:12} : {opcode : <12} | {description}")


    #MISC OPERATIONS
    sorted_ops = sorted(misc_operations.items(), key=lambda item: item[1], reverse=False)
    #Print misc opcodes with counts greater than 0
    print("\nMiscellaneous operations")
    for opcode, data in sorted(sorted_ops, key=lambda item: item[1], reverse=False):
        if data > 0:
            print(f"{data:12} : {opcode}")

    #NOT SUPPORTED OPERATIONS
    sorted_ops = sorted(x86_not_supported.items(), key=lambda item: item[1], reverse=False)
    #Print misc opcodes with counts greater than 0
    print("\nNot supported operations")
    for opcode, data in sorted(sorted_ops, key=lambda item: item[1], reverse=False):
        if data > 0:
            print(f"{data:12} : {opcode}")

    print("------------------------------")

def printDynamoRIOARM():
    print("\n---------OPCODE BREAKDOWN-----------")
    print("Memory Operations:\n")
    for opcode, entries in memory_operations.items():
        total_entry_printed = False
        for count, description in entries:
            if description != "TOTAL":
                if total_entry_printed:
                    #Indent subsequent entries after the TOTAL entry
                    print(f"{'':2} {count:12} : {opcode : <12} | {description}")
                else:
                    print(f"{count:12} : {opcode : <12} | {description}")
            else:
                print(f"\n{count:12} : {opcode : <12}  {description}")
                total_entry_printed = True

    #FLOATING POINT OPERATIONS        
    print("\nFloating Point Operations:")
    for opcode, entries in ARM_FP_operations.items():
        total_entry_printed = False
        for count, description in entries:
            if description != "TOTAL":
                if total_entry_printed:
                    # Indent subsequent entries after the TOTAL entry
                    print(f"{'':2} {count:12} : {opcode : <12} | {description}")
                else:
                    print(f"{count:12} : {opcode : <12} | {description}")
            else:
                print(f"\n{count:12} : {opcode : <12}  {description}")
                total_entry_printed = True

    #FLOATING POINT OPERATIONS        
    print("\nInteger Operations:\n")
    for opcode, entries in ARM_INT_operations.items():
        total_entry_printed = False
        for count, description in entries:
            if description != "TOTAL":
                if total_entry_printed:
                    # Indent subsequent entries after the TOTAL entry
                    print(f"{'':2} {count:12} : {opcode : <12} | {description}")
                else:
                    print(f"{count:12} : {opcode : <12} | {description}")
            else:
                print(f"\n{count:12} : {opcode : <12}  {description}")
                total_entry_printed = True

    #MISC OPERATIONS
    misc_operations_sorted = sorted(misc_operations.items(), key=lambda item: item[1], reverse=False)
    #print not supported things
    print("\nNot supported operations:\n")
    for opcode, data in sorted(misc_operations_sorted, key=lambda item: item[1], reverse=False):
        if data > 0:
            print(f"{data:12} : {opcode}")
    print("------------------------------")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run an executable with DynamoRIO.")
    parser.add_argument("dbi_path", nargs="?", help="Path to the DynamoRIO directory")
    parser.add_argument('--roi',  dest='roi', action='store_const', const=1, default=0, help='Measure only Region of Interest, or not.')
    parser.add_argument('--sde',  dest='sde', action='store_const', const=1, default=0, help='Measure using Intel SDE, instead of DynamoRIO.')
    parser.add_argument('-dr', '--drawroof',  dest='drawroof', action='store_const', const=1, default=0, help='Plot application in a chosen roofline chart localy (work in progress).')
    parser.add_argument('-c', '--choice', default=0, nargs='?', type = int, help='Automatically choose a roofline chart for the application opcode analysis, --drawroof is required for this (Default: 0).')
    parser.add_argument('-n','--name', default='unnamed', nargs='?', type = str, help='Name for the machine running the app. (Default: unnamed)')
    parser.add_argument('-an','--app_name', default='', nargs='?', type = str, help='Name for the app.')
    parser.add_argument('--isa', default='', nargs='?', choices=['avx512', 'avx', 'avx2', 'sse', 'scalar', 'neon', 'armscalar', 'riscvscalar', 'riscvvector', ''], help='Main ISA used by the application, if not sure leave blank (optional only for naming facilitation).')
    parser.add_argument('-t', '--threads', default='0', nargs='?', type = int, help='Number of threads used by the application (optional only for naming facilitation).')
    parser.add_argument('-p', '--precision', default='', nargs='?', choices=['dp', 'sp'], help='Data Precision used by the application (optional only for naming facilitation).')
    parser.add_argument("executable_path", help="Path to the executable provided by the user")
    
    parser.add_argument("additional_args", nargs="...", help="Additional arguments for the user's application.")
   

    args = parser.parse_args()

    CPU_Type = platform.machine()
    if CPU_Type != "x86_64" and CPU_Type != "aarch64":
        print("No opcode analysis support on non x86 / ARM CPUS.")
        sys.exit(1)

    if args.sde:
        if CPU_Type == "aarch64":
            print("No SDE opcode analysis support on non x86 CPUs.")
            sys.exit(1)
        if not (check_sde_exists(args.dbi_path)):
            sys.exit(1)
    else:
        #Check if DynamoRIO client is present
        if not (check_client_exists(args.dbi_path)):
            sys.exit(1)

    #Run the application to get time taken
    exec_time = runApplication(args.roi, args.executable_path, args.additional_args)

    if args.sde:
        runSDE(args.dbi_path, args.roi, args.executable_path, args.additional_args)
        fp_ops, memory_bytes = analyseSDE()
        method = "SDE"
        if args.roi:
            method += "-ROI"
                
    else:
        #Run the client with the provided executable and arguments
        runDynamoRIO(args.dbi_path, args.roi, args.executable_path, args.additional_args)
        
        if CPU_Type == "x86_64":
            fp_ops, memory_bytes, integer_ops = analyseDynamoRIOx86()
            printDynamoRIOx86()
        elif CPU_Type == "aarch64":
            fp_ops, memory_bytes, integer_ops = analyseDynamoRIOARM()
            printDynamoRIOARM()
        else:
            print("No opcode analysis support on this architecture.")
            sys.exit(1)
        method = "DR"
        if args.roi:
            method += "-ROI"

    time_taken_seconds = float (exec_time / 1e9)

    flops = fp_ops/time_taken_seconds

    gflops = flops / 1e9

    ai = float(fp_ops/memory_bytes)
    bandwidth = float((memory_bytes) / exec_time)

    print("\n---------DBI RESULTS-----------")
    print("Total FP operations:", ut.custom_round(fp_ops))
    print("Total memory bytes:", ut.custom_round(memory_bytes))
    if (not args.sde):
        print("Total integer operations:", ut.custom_round(integer_ops))

    print("\nExecution time (seconds):", ut.custom_round(time_taken_seconds))
    print("GFLOP/s:", ut.custom_round(gflops))
    print("Bandwidth (GB/s): " + str(ut.custom_round(bandwidth)))
    print("Arithmetic Intensity:", ut.custom_round(ai))
    print("------------------------------")

    ct = datetime.datetime.now()
    date = ct.strftime('%Y-%m-%d %H:%M:%S')

    #Plot Roofline
    if args.drawroof:
        print("Manual application plotting not implemented iet, results can be viewed using the GUI")
        #ut.plot_roofline_with_dot(args.executable_path, gflops, ai, args.choice, args.roi, date, "dbi")
    
    ut.update_csv(args.name, args.executable_path, gflops, ai, bandwidth, time_taken_seconds, args.app_name, date, args.isa, args.precision, args.threads, method, 1, 1)
