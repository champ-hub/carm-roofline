import sys
import argparse
import subprocess
import tempfile
import os
import platform
import threading

#Default compiler option, change with your preference
compiler = "gcc"

def compile_file_async(file_path, PMU_or_DBI, PAPI_path, additional_arguments):
    thread = threading.Thread(target=compile_c_file, args=(file_path, PMU_or_DBI, PAPI_path, additional_arguments))
    thread.start()

#Check if PAPI is present
def check_PAPI_exists(PAPI_path):
    PAPI_src = os.path.join(PAPI_path, "src")
    #Check for SDE folder
    if os.path.exists(PAPI_src):
        print(f"PAPI src folder found in: '{PAPI_path}'.")
        return True
        
    else:
        print(f"No PAPI src folder found in: '{PAPI_src}'.")
        return False
    
def check_CPU_Type():
    CPU_Type = platform.machine()
    if CPU_Type != "x86_64" and CPU_Type != "aarch64":
        print("No PMU/opcode analysis support on non x86 / ARM CPUS.")
        return False
    else:
        return CPU_Type
    
def inject_code(input_file, PMUDBI, create_file):
    
    CPU_Type = check_CPU_Type()
    if not CPU_Type:
        return "CPUError"

    
    directory, filename = os.path.split(input_file)
    filename_without_extension, extension = os.path.splitext(filename)

    if extension != ".c" and extension != ".cpp":
        print("Unsupported file type. Please provide path to a .c or .cpp file.")
        sys.exit(1)
    if create_file:
        injected_filename = filename_without_extension + '_injected' + extension
        injected_file = os.path.join(directory, injected_filename)
    else:
        injected_file = input_file
    for PMU_or_DBI in PMUDBI:
        print(PMU_or_DBI)
        roi_start = False
        roi_end = False

        if (PMU_or_DBI == "dbi"):
            if (CPU_Type == "x86_64"):
                injected_declaration_code = """\n//Injected Marker Declaration Code
#ifndef __SSC_MARK
#define __SSC_MARK(tag)\
__asm__ __volatile__("movl %0, %%ebx; .byte 0x64, 0x67, 0x90 "\
::"i"(tag) : "%ebx")
#endif"""
            elif (CPU_Type == "aarch64"):
                injected_declaration_code = """\n//Injected Marker Declaration Code
#ifndef __SSC_MARK
#define __SSC_MARK(tag)\
__asm__ __volatile__("mov x9, %0; .inst 0x9b03e03f"\
::"i"(tag) : "%x9")
#endif"""
            injected_instrumentation_start_code = r"""
//ROI START
struct timespec t_start, t_end;
clock_gettime(CLOCK_MONOTONIC, &t_start);
__SSC_MARK(0xFACE);
/*----------------------------------------*/
            """

            injected_instrumentation_end_code = r"""
//ROI END
__SSC_MARK(0xDEAD);
clock_gettime(CLOCK_MONOTONIC, &t_end);
double timing_duration = ((t_end.tv_sec + ((double) t_end.tv_nsec / 1000000000)) - (t_start.tv_sec + ((double) t_start.tv_nsec / 1000000000)));
FILE *timing_results_file = fopen("timing_results.txt", "w");
fprintf(timing_results_file, "Time Taken:\t%0.9lf seconds\\n", timing_duration);
fclose(timing_results_file);
/*----------------------------------------*/
            """
        elif str(PMU_or_DBI) == "pmu":
            injected_instrumentation_start_code = r"""
            //ROI START
            int retval = PAPI_hl_region_begin("roi");
            if ( retval != PAPI_OK ){
                printf("HL begin error\n");
                exit(0);
            }
            /*----------------------------------------*/
            """
            injected_instrumentation_end_code = r"""
            //ROI END
            retval = PAPI_hl_region_end("roi");
            if ( retval != PAPI_OK ){
                printf("HL end error\n");
                exit(0);
            }
            /*----------------------------------------*/
            """
        
        with open(input_file, 'r') as f:
            lines = f.readlines()
            if (PMU_or_DBI == "dbi"):
                time_included = any(line.strip() == '#include <time.h>' for line in lines)
                if not time_included:
                    #Add time.h at the end of the #include section
                    includes_index = next((i for i, line in enumerate(lines) if not line.startswith('#include')), len(lines))
                    
                    lines.insert(includes_index, injected_declaration_code)
                    lines.insert(includes_index, '\n//Injected Time Dependency\n#include <time.h>\n')
                else:
                    includes_index = next((i for i, line in enumerate(lines) if not line.startswith('#include')), len(lines))
                    lines.insert(includes_index, injected_declaration_code)
            elif PMU_or_DBI == "pmu":
                stdlib_included = any(line.strip() == '#include <stdlib.h>' for line in lines)
                if not stdlib_included:
                    #Add papi.h at the end of the #include section
                    includes_index = next((i for i, line in enumerate(lines) if not line.startswith('#include')), len(lines))
            
                    lines.insert(includes_index, '\n//Injected PAPI Dependency\n#include <stdlib.h>\n')
                else:
                    includes_index = next((i for i, line in enumerate(lines) if not line.startswith('#include')), len(lines))

                papi_included = any(line.strip() == '#include <papi.h>' for line in lines)
                if not papi_included:
                    #Add papi.h at the end of the #include section
                    includes_index = next((i for i, line in enumerate(lines) if not line.startswith('#include')), len(lines))

                    lines.insert(includes_index, '\n//Injected PAPI Dependency\n#include <papi.h>\n')
                else:
                    includes_index = next((i for i, line in enumerate(lines) if not line.startswith('#include')), len(lines))

        with open(injected_file, 'w') as f:
            roi_start = False
            roi_end = False
            for line in lines:
                if "//CARM ROI START" in line:
                    roi_start = True
                    f.write(injected_instrumentation_start_code + '\n')
                elif "//CARM ROI END" in line:
                    roi_end = True
                    f.write(injected_instrumentation_end_code + '\n')
                else:
                    f.write(line)
        
        #Check if the ROI Flags were found
        if not roi_start and create_file:
            print("No ROI start flag found. Please add //CARM ROI START line to the source code.")
            os.remove(injected_file)
            return False
        if not roi_end and create_file:
            print("No ROI end flag found. Please add //CARM ROI END line to the source code.")
            os.remove(injected_file)
            return False
    return True

def compile_c_file(c_file, PMU_or_DBI, PAPI_Path, compiler_flags=None):
    
    CPU_Type = check_CPU_Type()
    if not CPU_Type:
        return "CPUError"
    compiler_flags = compiler_flags or []
    filename, file_extension = os.path.splitext(c_file)
    compiled_executable = filename
    if PMU_or_DBI == "DBI":
        try:
            command = [compiler, *compiler_flags, c_file, "-o", compiled_executable]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                return stderr.strip()
            else:
                return None
        except Exception as e:
            return str(e)
        
    elif PMU_or_DBI == "PMU":
        if CPU_Type == "x86_64":
            PAPI_src = os.path.join(PAPI_Path, "src")
            PAPI_lib = os.path.join(PAPI_Path, "src/libpapi.a")
            PAPI_argument1 = "-I" + PAPI_src
        elif CPU_Type == "aarch64":
            PAPI_lib = "-lpapi"
            PAPI_argument1 = "-I/${PAPI_DIR}/include -L/${PAPI_DIR}/lib"
        
        try:
            command = [compiler, *compiler_flags, PAPI_argument1, c_file, "-o", compiled_executable, PAPI_lib]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                return stderr.strip()
            else:
                return None
        except Exception as e:
            return str(e)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Inject instrumentation code into the ROI of provided source code, and produce injected executable.")
    parser.add_argument('--PMU',  dest='pmu', action='store_const', const=1, default=0, help='Inject code for PMU measuring, intead of binary instrumentation.')
    parser.add_argument('--new_file',  dest='new_file', action='store_const', const=1, default=0, help='Create a new injected source file instead of injecting directly on the provided source file')
    parser.add_argument('--compile',  dest='compile', action='store_const', const=1, default=0, help='Automatically compile source file after injection')

    parser.add_argument("PAPI_path", nargs='?', help="Path to the PAPI instalation")
    parser.add_argument("source_path", help="Path to the source code")
    parser.add_argument("comp_args", nargs="...", help="Additional arguments for the compiler")

    args = parser.parse_args()
    PMU_or_DBI = "DBI"

    if (args.pmu):
        PMU_or_DBI = "PMU"
    else:
        PMU_or_DBI = "DBI"

    CPU_Type = platform.machine()
    if CPU_Type != "x86_64" and CPU_Type != "aarch64":
        print("No PMU/opcode analysis support on non x86 / ARM CPUS.")
        sys.exit(1)

    source_file = args.source_path

    directory, filename = os.path.split(source_file)
    filename_without_extension, extension = os.path.splitext(filename)

    #Check for correct file type
    if extension != ".c" and extension != ".cpp":
        print("Unsupported file type. Please provide path to a .c or .cpp file.")
        sys.exit(1)

    #Inject ROI instrumentation code
    inject_code(source_file, PMU_or_DBI, args.new_file)

    if args.compile:
        if args.new_file:
            injected_filename = filename_without_extension + '_injected' + extension
            injected_file = os.path.join(directory, injected_filename)
        else:
            injected_file = source_file
        error = compile_c_file(injected_file, PMU_or_DBI, args.PAPI_path, args.comp_args)
        if error:
            print("Compilation failed with the following error:\n", error)
        else:
            print("Compilation successful.")
