#!/usr/bin/env python
"""
This script computes FLOPs executed by any application using the Intel Software
Development Emulator (Intel SDE).

Example:
  $ sde64 -iform -mix -dyn_mask_profile -start_ssc_mark FACE:repeat
          -stop_ssc_mark DEAD:repeat -- <your_app>
  <output of your_app>
  $ python intel_sde_flops.py
  <output of FLOPs information>

In addition, add the following markers to your code to select the sections for
which to count the FLOPs:
- __SSC_MARK(0xFACE) // Starts/resumes profiling
- __SSC_MARK(0xDEAD) // Ends/pauses profiling

Author: Georg Zitzlsberger (georg.zitzlsberger<ad>vsb.cz)

Copyright (C) 2017-2018 Georg Zitzlsberger, IT4Innovations,
                        VSB-Technical University of Ostrava, Czech Republic

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.


History:
- 1.1: Added BF16 and 4FMA support
- 1.0: Initial version (never labled like that)
"""

import re
import sys

__version__ = "1.1"

def usage():
    print("Usage:\npython %s [<sde_mix_out> <sde_dyn_mask_profile>]\n" %  sys.argv[0])
    print("If no arguments are used, defaults are:")
    print("  <sde_mix_out>: sde-mix-out.txt")
    print("  <sde_dyn_mask_profile>: sde-dyn-mask-profile.txt")
    print("If arguments are used, specify both in correct order:")
    print("  <sde_mix_out> <sde_dyn_mask_profile>\n")
    print("The files <sde_mix_out> and <sde_dyn_mask_profile> are created by "
          "Intel SDE's '-mix -iform' and '-dyn_mask_profile' options, "
          "respectively.")


def flops_mix(mix_file):
    "Calculate the double/single FLOPS indicated in 'mix_file'"
    lines = []
    try:
        with open(mix_file, 'rt') as in_file:
            for line in in_file:
                lines.append(line)
    except:
        print("Error: File '%s' does not exist!\n" % mix_file)
        usage()
        exit(1)

    # Brief validity check whether it's the correct file type...
    if not any(re.match(r'^# EMIT_DYNAMIC_STATS FOR TID', line)
               for line in lines):
        print("Error: File '%s' does not seem to be created from Intel SDE's "
              "'-mix' option!\n" % mix_file)
        usage()
        exit(1)

    tid_end = -1
    result = []
    while True:  # iterate over all threads
        tid = -1
        os_tid = -1
        # Find start line for thread (tid_start)
        tid_start = -1
        for i in range(tid_end, len(lines)):
            mobj = re.match(r'^# EMIT_DYNAMIC_STATS FOR TID\s+([0-9]+)\s'
                            r'+OS-TID\s+([0-9]+)\s+EMIT', lines[i])
            if mobj:
                tid_start = i
                tid = int(mobj.group(1))
                os_tid = int(mobj.group(2))
                break
        if (tid_start == -1):
            break

        # Find end line for thread (tid_end)
        old_tid_end = tid_end
        for i in range(tid_start, len(lines)):
            mobj = re.match(r'# END_DYNAMIC_STATS', lines[i])
            if mobj:
                tid_end = i
                break
        if (old_tid_end == tid_end):
            print("Error: END_DYNAMIC_STATS not found!")
            exit(1)

        # Find line "# $dynamic-counts"
        gdc_line = -1  # zero-based!
        for i in range(tid_start, tid_end):
            mobj = re.match(r'# \$dynamic-counts$', lines[i])
            if mobj:
                gdc_line = i
                break
        if gdc_line == -1:
            break

        # Find line "# iform count" (if not there, SDE did not use -iform)
        iform_line = -1  # zero-based!
        for i in range(gdc_line, tid_end):
            mobj = re.match(r'#\s+iform\s+count', lines[i])
            if mobj:
                iform_line = i
                break
        if iform_line == -1:
            print("Error: File '%s' does not seem to be created from Intel "
                  "SDE's '-iform' option!\n" % mix_file)
            usage()
            exit(1)

        # Read the instruction groups and counts
        instruction_group_count = {}
        for i in range(iform_line, tid_end):
            mobj = re.match(r'^([*a-zA-Z0-9-_]+)\s+([0-9]+)$', lines[i])
            if mobj:
                instruction_group_count[mobj.group(1)] = eval(mobj.group(2))

        # Compute FLOPs below...
        total_fmas = 0
        total_single_fp = 0
        total_double_fp = 0

        # General FP (*elements_fp_[double|single]_[1|2|4|8|16])
        for cnt in [1, 2, 4, 8]:
            key = '*elements_fp_double_' + str(cnt)
            if key in instruction_group_count:
                total_double_fp += instruction_group_count[key] * cnt
        for cnt in [1, 2, 4, 8, 16]:
            key = '*elements_fp_single_' + str(cnt)
            if key in instruction_group_count:
                total_single_fp += instruction_group_count[key] * cnt

        # Get total executed instructions...
        total_inst = 0
        key = '*total'
        if key in instruction_group_count:
            total_inst = instruction_group_count[key]

        # Get bytes written...
        total_written = 0

        key = '*mem-write-1'
        if key in instruction_group_count:
            total_written += instruction_group_count[key]

        key = '*mem-write-2'
        if key in instruction_group_count:
            total_written += (instruction_group_count[key]*2)

        key = '*mem-write-4'
        if key in instruction_group_count:
            total_written += (instruction_group_count[key]*4)

        key = '*mem-write-8'
        if key in instruction_group_count:
            total_written += (instruction_group_count[key]*8)

        key = '*mem-write-16'
        if key in instruction_group_count:
            total_written += (instruction_group_count[key]*16)

        key = '*mem-write-32'
        if key in instruction_group_count:
            total_written += (instruction_group_count[key]*32)

        key = '*mem-write-64'
        if key in instruction_group_count:
            total_written += (instruction_group_count[key]*64)

        # Get bytes read...
        total_read = 0

        key = '*mem-read-1'
        if key in instruction_group_count:
            total_read += instruction_group_count[key]
        
        key = '*mem-read-2'
        if key in instruction_group_count:
            total_read += (instruction_group_count[key]*2)
        
        key = '*mem-read-4'
        if key in instruction_group_count:
            total_read += (instruction_group_count[key]*4)
        
        key = '*mem-read-8'
        if key in instruction_group_count:
            total_read += (instruction_group_count[key]*8)

        key = '*mem-read-16'
        if key in instruction_group_count:
            total_read += (instruction_group_count[key]*16)

        key = '*mem-read-32'
        if key in instruction_group_count:
            total_read += (instruction_group_count[key]*32)

        key = '*mem-read-64'
        if key in instruction_group_count:
            total_read += (instruction_group_count[key]*64)

        # FMAs
        # Note: AVX512 FMAs are all only masked versions which will be taken
        #       care of by the function "flops_masked" properly (using
        #       "comp_count")
        fma_double_xmm = [
            'VFMADD132PD_XMMdq_XMMdq_MEMdq',
            'VFMADD132PD_XMMdq_XMMdq_XMMdq',
            'VFMADD213PD_XMMdq_XMMdq_MEMdq',
            'VFMADD213PD_XMMdq_XMMdq_XMMdq',
            'VFMADD231PD_XMMdq_XMMdq_MEMdq',
            'VFMADD231PD_XMMdq_XMMdq_XMMdq',
            'VFMADDSUB132PD_XMMdq_XMMdq_MEMdq',
            'VFMADDSUB132PD_XMMdq_XMMdq_XMMdq',
            'VFMADDSUB213PD_XMMdq_XMMdq_MEMdq',
            'VFMADDSUB213PD_XMMdq_XMMdq_XMMdq',
            'VFMADDSUB231PD_XMMdq_XMMdq_MEMdq',
            'VFMADDSUB231PD_XMMdq_XMMdq_XMMdq',
            'VFMSUB132PD_XMMdq_XMMdq_MEMdq',
            'VFMSUB132PD_XMMdq_XMMdq_XMMdq',
            'VFMSUB213PD_XMMdq_XMMdq_MEMdq',
            'VFMSUB213PD_XMMdq_XMMdq_XMMdq',
            'VFMSUB231PD_XMMdq_XMMdq_MEMdq',
            'VFMSUB231PD_XMMdq_XMMdq_XMMdq',
            'VFMSUBADD132PD_XMMdq_XMMdq_MEMdq',
            'VFMSUBADD132PD_XMMdq_XMMdq_XMMdq',
            'VFMSUBADD213PD_XMMdq_XMMdq_MEMdq',
            'VFMSUBADD213PD_XMMdq_XMMdq_XMMdq',
            'VFMSUBADD231PD_XMMdq_XMMdq_MEMdq',
            'VFMSUBADD231PD_XMMdq_XMMdq_XMMdq',
            'VFNMADD132PD_XMMdq_XMMdq_MEMdq',
            'VFNMADD132PD_XMMdq_XMMdq_XMMdq',
            'VFNMADD213PD_XMMdq_XMMdq_MEMdq',
            'VFNMADD213PD_XMMdq_XMMdq_XMMdq',
            'VFNMADD231PD_XMMdq_XMMdq_MEMdq',
            'VFNMADD231PD_XMMdq_XMMdq_XMMdq',
            'VFNMSUB132PD_XMMdq_XMMdq_MEMdq',
            'VFNMSUB132PD_XMMdq_XMMdq_XMMdq',
            'VFNMSUB213PD_XMMdq_XMMdq_MEMdq',
            'VFNMSUB213PD_XMMdq_XMMdq_XMMdq',
            'VFNMSUB231PD_XMMdq_XMMdq_MEMdq',
            'VFNMSUB231PD_XMMdq_XMMdq_XMMdq'
            ]
        for cnt in fma_double_xmm:
            if cnt in instruction_group_count:
                total_double_fp += instruction_group_count[cnt] * 2
                total_fmas += instruction_group_count[cnt]

        fma_double_ymm = [
            'VFMADD132PD_YMMqq_YMMqq_MEMqq',
            'VFMADD132PD_YMMqq_YMMqq_YMMqq',
            'VFMADD213PD_YMMqq_YMMqq_MEMqq',
            'VFMADD213PD_YMMqq_YMMqq_YMMqq',
            'VFMADD231PD_YMMqq_YMMqq_MEMqq',
            'VFMADD231PD_YMMqq_YMMqq_YMMqq',
            'VFMADDSUB132PD_YMMqq_YMMqq_MEMqq',
            'VFMADDSUB132PD_YMMqq_YMMqq_YMMqq',
            'VFMADDSUB213PD_YMMqq_YMMqq_MEMqq',
            'VFMADDSUB213PD_YMMqq_YMMqq_YMMqq',
            'VFMADDSUB231PD_YMMqq_YMMqq_MEMqq',
            'VFMADDSUB231PD_YMMqq_YMMqq_YMMqq',
            'VFMSUB132PD_YMMqq_YMMqq_MEMqq',
            'VFMSUB132PD_YMMqq_YMMqq_YMMqq',
            'VFMSUB213PD_YMMqq_YMMqq_MEMqq',
            'VFMSUB213PD_YMMqq_YMMqq_YMMqq',
            'VFMSUB231PD_YMMqq_YMMqq_MEMqq',
            'VFMSUB231PD_YMMqq_YMMqq_YMMqq',
            'VFMSUBADD132PD_YMMqq_YMMqq_MEMqq',
            'VFMSUBADD132PD_YMMqq_YMMqq_YMMqq',
            'VFMSUBADD213PD_YMMqq_YMMqq_MEMqq',
            'VFMSUBADD213PD_YMMqq_YMMqq_YMMqq',
            'VFMSUBADD231PD_YMMqq_YMMqq_MEMqq',
            'VFMSUBADD231PD_YMMqq_YMMqq_YMMqq',
            'VFNMADD132PD_YMMqq_YMMqq_MEMqq',
            'VFNMADD132PD_YMMqq_YMMqq_YMMqq',
            'VFNMADD213PD_YMMqq_YMMqq_MEMqq',
            'VFNMADD213PD_YMMqq_YMMqq_YMMqq',
            'VFNMADD231PD_YMMqq_YMMqq_MEMqq',
            'VFNMADD231PD_YMMqq_YMMqq_YMMqq',
            'VFNMSUB132PD_YMMqq_YMMqq_MEMqq',
            'VFNMSUB132PD_YMMqq_YMMqq_YMMqq',
            'VFNMSUB213PD_YMMqq_YMMqq_MEMqq',
            'VFNMSUB213PD_YMMqq_YMMqq_YMMqq',
            'VFNMSUB231PD_YMMqq_YMMqq_MEMqq',
            'VFNMSUB231PD_YMMqq_YMMqq_YMMqq'
            ]
        for cnt in fma_double_ymm:
            if cnt in instruction_group_count:
                total_double_fp += instruction_group_count[cnt] * 4
                total_fmas += instruction_group_count[cnt]

        fma_double_scalar = [
            'VFMADD132SD_XMMdq_XMMq_MEMq',
            'VFMADD132SD_XMMdq_XMMq_XMMq',
            'VFMADD213SD_XMMdq_XMMq_MEMq',
            'VFMADD213SD_XMMdq_XMMq_XMMq',
            'VFMADD231SD_XMMdq_XMMq_MEMq',
            'VFMADD231SD_XMMdq_XMMq_XMMq',
            'VFMSUB132SD_XMMdq_XMMq_MEMq',
            'VFMSUB132SD_XMMdq_XMMq_XMMq',
            'VFMSUB213SD_XMMdq_XMMq_MEMq',
            'VFMSUB213SD_XMMdq_XMMq_XMMq',
            'VFMSUB231SD_XMMdq_XMMq_MEMq',
            'VFMSUB231SD_XMMdq_XMMq_XMMq',
            'VFNMADD132SD_XMMdq_XMMq_MEMq',
            'VFNMADD132SD_XMMdq_XMMq_XMMq',
            'VFNMADD213SD_XMMdq_XMMq_MEMq',
            'VFNMADD213SD_XMMdq_XMMq_XMMq',
            'VFNMADD231SD_XMMdq_XMMq_MEMq',
            'VFNMADD231SD_XMMdq_XMMq_XMMq',
            'VFNMSUB132SD_XMMdq_XMMq_MEMq',
            'VFNMSUB132SD_XMMdq_XMMq_XMMq',
            'VFNMSUB213SD_XMMdq_XMMq_MEMq',
            'VFNMSUB213SD_XMMdq_XMMq_XMMq',
            'VFNMSUB231SD_XMMdq_XMMq_MEMq',
            'VFNMSUB231SD_XMMdq_XMMq_XMMq'
            ]
        for cnt in fma_double_scalar:
            if cnt in instruction_group_count:
                total_double_fp += instruction_group_count[cnt]
                total_fmas += instruction_group_count[cnt]

        fma_single_xmm = [
            'VFMADD132PS_XMMdq_XMMdq_MEMdq',
            'VFMADD132PS_XMMdq_XMMdq_XMMdq',
            'VFMADD213PS_XMMdq_XMMdq_MEMdq',
            'VFMADD213PS_XMMdq_XMMdq_XMMdq',
            'VFMADD231PS_XMMdq_XMMdq_MEMdq',
            'VFMADD231PS_XMMdq_XMMdq_XMMdq',
            'VFMADDSUB132PS_XMMdq_XMMdq_MEMdq',
            'VFMADDSUB132PS_XMMdq_XMMdq_XMMdq',
            'VFMADDSUB213PS_XMMdq_XMMdq_MEMdq',
            'VFMADDSUB213PS_XMMdq_XMMdq_XMMdq',
            'VFMADDSUB231PS_XMMdq_XMMdq_MEMdq',
            'VFMADDSUB231PS_XMMdq_XMMdq_XMMdq',
            'VFMSUB132PS_XMMdq_XMMdq_MEMdq',
            'VFMSUB132PS_XMMdq_XMMdq_XMMdq',
            'VFMSUB213PS_XMMdq_XMMdq_MEMdq',
            'VFMSUB213PS_XMMdq_XMMdq_XMMdq',
            'VFMSUB231PS_XMMdq_XMMdq_MEMdq',
            'VFMSUB231PS_XMMdq_XMMdq_XMMdq',
            'VFMSUBADD132PS_XMMdq_XMMdq_MEMdq',
            'VFMSUBADD132PS_XMMdq_XMMdq_XMMdq',
            'VFMSUBADD213PS_XMMdq_XMMdq_MEMdq',
            'VFMSUBADD213PS_XMMdq_XMMdq_XMMdq',
            'VFMSUBADD231PS_XMMdq_XMMdq_MEMdq',
            'VFMSUBADD231PS_XMMdq_XMMdq_XMMdq',
            'VFNMADD132PS_XMMdq_XMMdq_MEMdq',
            'VFNMADD132PS_XMMdq_XMMdq_XMMdq',
            'VFNMADD213PS_XMMdq_XMMdq_MEMdq',
            'VFNMADD213PS_XMMdq_XMMdq_XMMdq',
            'VFNMADD231PS_XMMdq_XMMdq_MEMdq',
            'VFNMADD231PS_XMMdq_XMMdq_XMMdq',
            'VFNMSUB132PS_XMMdq_XMMdq_MEMdq',
            'VFNMSUB132PS_XMMdq_XMMdq_XMMdq',
            'VFNMSUB213PS_XMMdq_XMMdq_MEMdq',
            'VFNMSUB213PS_XMMdq_XMMdq_XMMdq',
            'VFNMSUB231PS_XMMdq_XMMdq_MEMdq',
            'VFNMSUB231PS_XMMdq_XMMdq_XMMdq'
            ]
        for cnt in fma_single_xmm:
            if cnt in instruction_group_count:
                total_single_fp += instruction_group_count[cnt] * 4
                total_fmas += instruction_group_count[cnt]

        fma_single_ymm = [
            'VFMADD132PS_YMMqq_YMMqq_MEMqq',
            'VFMADD132PS_YMMqq_YMMqq_YMMqq',
            'VFMADD213PS_YMMqq_YMMqq_MEMqq',
            'VFMADD213PS_YMMqq_YMMqq_YMMqq',
            'VFMADD231PS_YMMqq_YMMqq_MEMqq',
            'VFMADD231PS_YMMqq_YMMqq_YMMqq',
            'VFMADDSUB132PS_YMMqq_YMMqq_MEMqq',
            'VFMADDSUB132PS_YMMqq_YMMqq_YMMqq',
            'VFMADDSUB213PS_YMMqq_YMMqq_MEMqq',
            'VFMADDSUB213PS_YMMqq_YMMqq_YMMqq',
            'VFMADDSUB231PS_YMMqq_YMMqq_MEMqq',
            'VFMADDSUB231PS_YMMqq_YMMqq_YMMqq',
            'VFMSUB132PS_YMMqq_YMMqq_MEMqq',
            'VFMSUB132PS_YMMqq_YMMqq_YMMqq',
            'VFMSUB213PS_YMMqq_YMMqq_MEMqq',
            'VFMSUB213PS_YMMqq_YMMqq_YMMqq',
            'VFMSUB231PS_YMMqq_YMMqq_MEMqq',
            'VFMSUB231PS_YMMqq_YMMqq_YMMqq',
            'VFMSUBADD132PS_YMMqq_YMMqq_MEMqq',
            'VFMSUBADD132PS_YMMqq_YMMqq_YMMqq',
            'VFMSUBADD213PS_YMMqq_YMMqq_MEMqq',
            'VFMSUBADD213PS_YMMqq_YMMqq_YMMqq',
            'VFMSUBADD231PS_YMMqq_YMMqq_MEMqq',
            'VFMSUBADD231PS_YMMqq_YMMqq_YMMqq',
            'VFNMADD132PS_YMMqq_YMMqq_MEMqq',
            'VFNMADD132PS_YMMqq_YMMqq_YMMqq',
            'VFNMADD213PS_YMMqq_YMMqq_MEMqq',
            'VFNMADD213PS_YMMqq_YMMqq_YMMqq',
            'VFNMADD231PS_YMMqq_YMMqq_MEMqq',
            'VFNMADD231PS_YMMqq_YMMqq_YMMqq',
            'VFNMSUB132PS_YMMqq_YMMqq_MEMqq',
            'VFNMSUB132PS_YMMqq_YMMqq_YMMqq',
            'VFNMSUB213PS_YMMqq_YMMqq_MEMqq',
            'VFNMSUB213PS_YMMqq_YMMqq_YMMqq',
            'VFNMSUB231PS_YMMqq_YMMqq_MEMqq',
            'VFNMSUB231PS_YMMqq_YMMqq_YMMqq'
            ]
        for cnt in fma_single_ymm:
            if cnt in instruction_group_count:
                total_single_fp += instruction_group_count[cnt] * 8
                total_fmas += instruction_group_count[cnt]

        fma_single_scalar = [
            'VFMADD132SS_XMMdq_XMMd_MEMd',
            'VFMADD132SS_XMMdq_XMMd_XMMd',
            'VFMADD213SS_XMMdq_XMMd_MEMd',
            'VFMADD213SS_XMMdq_XMMd_XMMd',
            'VFMADD231SS_XMMdq_XMMd_MEMd',
            'VFMADD231SS_XMMdq_XMMd_XMMd',
            'VFMSUB132SS_XMMdq_XMMd_MEMd',
            'VFMSUB132SS_XMMdq_XMMd_XMMd',
            'VFMSUB213SS_XMMdq_XMMd_MEMd',
            'VFMSUB213SS_XMMdq_XMMd_XMMd',
            'VFMSUB231SS_XMMdq_XMMd_MEMd',
            'VFMSUB231SS_XMMdq_XMMd_XMMd',
            'VFNMADD132SS_XMMdq_XMMd_MEMd',
            'VFNMADD132SS_XMMdq_XMMd_XMMd',
            'VFNMADD213SS_XMMdq_XMMd_MEMd',
            'VFNMADD213SS_XMMdq_XMMd_XMMd',
            'VFNMADD231SS_XMMdq_XMMd_MEMd',
            'VFNMADD231SS_XMMdq_XMMd_XMMd',
            'VFNMSUB132SS_XMMdq_XMMd_MEMd',
            'VFNMSUB132SS_XMMdq_XMMd_XMMd',
            'VFNMSUB213SS_XMMdq_XMMd_MEMd',
            'VFNMSUB213SS_XMMdq_XMMd_XMMd',
            'VFNMSUB231SS_XMMdq_XMMd_MEMd',
            'VFNMSUB231SS_XMMdq_XMMd_XMMd'
            ]
        for cnt in fma_single_scalar:
            if cnt in instruction_group_count:
                total_single_fp += instruction_group_count[cnt]
                total_fmas += instruction_group_count[cnt]

        result.append([tid, os_tid, total_single_fp, total_double_fp,
                       total_inst, total_fmas, total_written, total_read])
    # end while
    return result


def flops_dyn(dyn_file):
    "Calculate the masked double/single FLOPS indicated in 'dyn_file'"
    warn4FMA = True
    warnBF16 = True
    lines = []
    try:
        with open(dyn_file, 'rt') as in_file:
            for line in in_file:
                lines.append(line)
    except:
        print("Error: File '%s' does not exist!\n" % dyn_file)
        usage()
        exit(1)

    # Brief validity check whether it's the correct file type...
    if not any(re.match(r'\s+<thread-number>', line)
               for line in lines):
        print("Error: File '%s' does not seem to be created from Intel SDE's "
              "'-dyn_mask_profile' option!\n" % dyn_file)
        usage()
        exit(1)

    tid_end = -1
    result = []
    while True:  # iterate over all threads
        tid = -1
        # Find start line for thread (tid_start)
        tid_start = -1
        for i in range(tid_end, len(lines)):
            mobj = re.match(r'<thread>', lines[i])
            if mobj:
                tid_start = i
                break
        if (tid_start == -1):
            break

        # Find end line for thread (tid_end)
        old_tid_end = tid_end
        for i in range(tid_start, len(lines)):
            mobj = re.match(r'</thread>', lines[i])
            if mobj:
                tid_end = i
                break
        if (old_tid_end == tid_end):
            print("Error: </thread> not found!")
            exit(1)

        # Find line "<thread-number>" for TID
        th_num = -1  # zero-based!
        for i in range(tid_start, tid_end):
            mobj = re.match(r'\s+<thread-number>\s+([0-9]+)\s+'
                            r'</thread-number>', lines[i])
            if mobj:
                th_num = i
                tid = int(mobj.group(1))
                break
        if th_num == -1:
            print("Error: <thread-number> not found!")
            exit(1)

        # Find line "<summarytable>"
        sum_line = -1  # zero-based!
        for i in range(tid_start, tid_end):
            mobj = re.match(r'\s+<summarytable>', lines[i])
            if mobj:
                sum_line = i
                break
        if sum_line == -1:
            print("Error: <summarytable> not found!")
            exit(1)

        # Find line "</summarytable>"
        endsum_line = -1  # zero-based!
        for i in range(sum_line, tid_end):
            mobj = re.match(r'\s+</summarytable>', lines[i])
            if mobj:
                endsum_line = i
                break
        if endsum_line == -1:
            print("Error: </summarytable> not found!")
            exit(1)

        # Compute masked FLOPs below...
        total_fmas = 0
        total_single_fp = 0
        total_double_fp = 0
        total_single_fp_m = 0
        total_double_fp_m = 0

        # Read the masked instruction counts (comp_count) for "fp" types
        for i in range(sum_line, endsum_line):
            mobj = re.match(r'^\s+masked\s+mask\s+[0-9]+b\s+[0-9]+elem\s+'
                            r'([0-9]+)b\s+fp\s+[|]\s+[0-9]+\s+([0-9]+)\s+'
                            r'[0-9.]+$', lines[i])
            if mobj:
                fp_type_bits = int(mobj.group(1))  # 32 (single) or 64 (double)
                if (fp_type_bits == 32):
                    total_single_fp_m += int(mobj.group(2))  # comp_count
                elif (fp_type_bits == 64):
                    total_double_fp_m += int(mobj.group(2))  # comp_count
                else:
                    print("Error: Unkown element_s!")
                    exit(1)

        # Read the individual instruction details and find FMAs
        idet_end = tid_start
        while True:  # iterate over all instruction details
            # Find start line for instruction details (idet_start)
            idet_start = -1
            for i in range(idet_end, tid_end):
                mobj = re.match(r'\s+<instruction-details>', lines[i])
                if mobj:
                    idet_start = i
                    break
            if (idet_start == -1):
                break

            # Find end line for instruction details (idet_end)
            old_idet_end = idet_end
            for i in range(idet_start, tid_end):
                mobj = re.match(r'\s+</instruction-details>', lines[i])
                if mobj:
                    idet_end = i
                    break
            if (old_idet_end == idet_end):
                print("Error: </instruction-details> not found!")
                exit(1)

            # Identify if instruction is FMA (or FMS)
            for i in range(idet_start, idet_end):
                mobj = re.match(r'\s+<disassembly>\s+'
                                r'(vf(m|nm)(add|sub)[0-9a-z]+)\s+', lines[i])
                if mobj:
                    # For each found, get computation count
                    comp_count = 0
                    for j in range(idet_start, idet_end):
                        mobjj = re.match(r'\s+<computation-count>\s+'
                                         r'([0-9]+)\s+', lines[j])
                        if mobjj:
                            comp_count = int(mobjj.group(1))
                            break
                    # For each found, get execution count
                    exec_count = 0
                    for j in range(idet_start, idet_end):
                        mobjj = re.match(r'\s+<execution-counts>\s+'
                                         r'([0-9]+)\s+', lines[j])
                        if mobjj:
                            exec_count = int(mobjj.group(1))
                            break

                    mobjm = re.match(r'\s+<disassembly>\s+'
                                     r'vf.*(\{k[0-9]+\})', lines[i])
                    is_masked = False
                    if mobjm is not None:
                        is_masked = True

                    total_fmas += exec_count
                    # For each found, distinguish single and double prec.
                    if (mobj.group(1)[-1:] == "s"):
                        if is_masked:
                            total_single_fp_m += comp_count
                        else:
                            total_single_fp += comp_count
                    elif (mobj.group(1)[-1:] == "d"):
                        if is_masked:
                            total_double_fp_m += comp_count
                        else:
                            total_double_fp += comp_count
                    else:
                        print("Error: Unknown FP type for FMA!")
                        exit(1)
                    break

            # Identify if instruction is 4FMA
            for i in range(idet_start, idet_end):
                mobj = re.match(r'\s+<disassembly>\s+'
                                r'(v4f(m|nm)add[a-z]+)\s+', lines[i])
                if mobj:
                    # For each found, get computation count
                    comp_count = 0
                    for j in range(idet_start, idet_end):
                        mobjj = re.match(r'\s+<computation-count>\s+'
                                         r'([0-9]+)\s+', lines[j])
                        if mobjj:
                            comp_count = int(mobjj.group(1))
                            break
                    # For each found, get execution count
                    exec_count = 0
                    for j in range(idet_start, idet_end):
                        mobjj = re.match(r'\s+<execution-counts>\s+'
                                         r'([0-9]+)\s+', lines[j])
                        if mobjj:
                            exec_count = int(mobjj.group(1))
                            break

                    mobjm = re.match(r'\s+<disassembly>\s+'
                                     r'v4f.*(\{k[0-9]+\})', lines[i])
                    is_masked = False
                    if mobjm is not None:
                        is_masked = True

                    total_fmas += 4 * exec_count
                    # For each found, increase single prec. count.
                    # There is no double prec. support for this instruction.
                    if (mobj.group(1)[-1:] == "s"):
                        # TODO:
                        # Once supported within Intel SDE, validate!
                        # Current Intel SDE 8.50 does not support 4FMA for
                        # -icx, but should!?
                        if is_masked:
                            total_single_fp_m += 3 * comp_count
                        else:
                            total_single_fp += 4 * comp_count

                        if warn4FMA:
                            print("Warning: 4FMA is currently experimental!")
                            warn4FMA = False
                    else:
                        print("Error: Unknown FP type for 4FMA!")
                        exit(1)
                    break

            # Identify if instruction is DPBF16
            # Note:
            # We only consider DP (dot product) instructions and ignore type
            # converts (BF16 -> FP32)!
            for i in range(idet_start, idet_end):
                mobj = re.match(r'\s+<disassembly>\s+'
                                r'(vdpbf16[a-z]+)\s+', lines[i])
                if mobj:
                    # For each found, get computation count
                    comp_count = 0
                    for j in range(idet_start, idet_end):
                        mobjj = re.match(r'\s+<computation-count>\s+'
                                         r'([0-9]+)\s+', lines[j])
                        if mobjj:
                            comp_count = int(mobjj.group(1))
                            break
                    # For each found, get execution count
                    exec_count = 0
                    for j in range(idet_start, idet_end):
                        mobjj = re.match(r'\s+<execution-counts>\s+'
                                         r'([0-9]+)\s+', lines[j])
                        if mobjj:
                            exec_count = int(mobjj.group(1))
                            break

                    # This is a workaround since (current?) Intel SDE is
                    # inconsistent for BF16 instructions when differentiating
                    # between masked and un-masked versions. For other
                    # instructions, un-masked ones will be properly counted
                    # in the sde-mix-out.txt report, but BF16 ones are not.
                    mobjm = re.match(r'\s+<disassembly>\s+'
                                     r'vdpbf16.*(\{k[0-9]+\})', lines[i])
                    is_masked = False
                    if mobjm is not None:
                        is_masked = True

                    # For each found, increase single prec. count.
                    # There is no double prec. support for this instruction.
                    if (mobj.group(1)[-1:] == "s"):
                        if is_masked:
                            total_single_fp_m += 3 * comp_count
                        else:
                            total_single_fp += 4 * comp_count

                        if warnBF16:
                            print("Warning: BF16 is currently experimental!")
                            warnBF16 = False
                    else:
                        print("Error: Unknown FP type for DPBF16!")
                        exit(1)
                    break

        result.append([tid, total_single_fp_m, total_double_fp_m, total_fmas,
                       total_single_fp, total_double_fp])
    # end while
    return result

print("Version: %s" % __version__)
if (len(sys.argv) == 3):
    str(sys.argv)
    # User selected profiling files (note the order!)
    file_sde_mix = sys.argv[1]
    file_sde_dyn = sys.argv[2]
elif (len(sys.argv) == 1):
    # Default profiling files
    file_sde_mix = "sde-mix-out.txt"
    file_sde_dyn = "sde-dyn-mask-profile.txt"
else:
    print("Error: Incorrect arguments!\n")
    usage()
    exit(1)

result_mix = flops_mix(file_sde_mix)
result_dyn = flops_dyn(file_sde_dyn)

sum_single_flops = 0
sum_double_flops = 0
sum_total_inst = 0
sum_total_written = 0
sum_total_read = 0
sum_total_fmas = 0
for i in range(0, len(result_mix)):
    masked_idx = -1
    for j in range(0, len(result_dyn)):
        if (result_dyn[j][0] == result_mix[i][0]):
            masked_idx = j
            break
    print("TID: %d (OS-TID: %d):" % (result_mix[i][0],
                                     result_mix[i][1]))
    sum_single_flops += result_mix[i][2] + result_dyn[i][4]
    print("\tUnmasked single prec. FLOPs: %d" % (result_mix[i][2] +
                                                 result_dyn[i][4]))
    sum_single_flops += result_dyn[masked_idx][1]
    print("\tMasked single prec. FLOPs: %d" % result_dyn[masked_idx][1])
    sum_double_flops += result_mix[i][3] + result_dyn[i][5]
    print("\tUnmasked double prec. FLOPs: %d" % (result_mix[i][3] +
                                                 result_dyn[i][5]))
    sum_double_flops += result_dyn[masked_idx][2]
    print("\tMasked double prec. FLOPs: %d" % result_dyn[masked_idx][2])
    sum_total_inst += result_mix[i][4]
    print("\tInstructions executed: %d" % result_mix[i][4])
    sum_total_fmas += (result_mix[i][5] + result_dyn[masked_idx][3])
    print("\tFMA instructions executed: %d" % (result_mix[i][5] +
                                               result_dyn[masked_idx][3]))
    sum_total_written += result_mix[i][6]
    print("\tTotal bytes written: %d" % result_mix[i][6])
    sum_total_read += result_mix[i][7]
    print("\tTotal bytes read: %d" % result_mix[i][7])
    print("\tArithmetic intensity (approx.): %f (EXPERIMENTAL)" % ((result_mix[i][2] + result_dyn[i][4] + result_dyn[masked_idx][1] + result_mix[i][3] + result_dyn[i][5] + result_dyn[masked_idx][2])/float(result_mix[i][6] + result_mix[i][7])))
print("=============================================\nSum:")
print("\tSingle prec. FLOPs: %d" % sum_single_flops)
print("\tDouble prec. FLOPs: %d" % sum_double_flops)
print("\tTotal instructions executed: %d" % sum_total_inst)
print("\tTotal FMA instructions executed: %d" % sum_total_fmas)
print("\tTotal bytes written: %d" % sum_total_written)
print("\tTotal bytes read: %d" % sum_total_read)
if (sum_total_read + sum_total_written > 0):
    print("\tTotal arithmetic intensity (approx.): %f (EXPERIMENTAL)" % ((sum_single_flops + sum_double_flops)/float(sum_total_written + sum_total_read)))
