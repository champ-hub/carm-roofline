/* ******************************************************************************
 * Copyright (c) 2015-2018 Google, Inc.  All rights reserved.
 * ******************************************************************************/

/*
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of VMware, Inc. nor the names of its contributors may be
 *   used to endorse or promote products derived from this software without
 *   specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL VMWARE, INC. OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 */

/* Code Manipulation API Sample:
 * opcoder.c
 *
 * Reports the dynamic count of the total number of instructions executed
 * broken down by opcode.
 */

#include "dr_api.h"
#include "drmgr.h"
#include "drx.h"
#include <stdlib.h> /* qsort */
#include <string.h>

#ifdef WINDOWS
#    define DISPLAY_STRING(msg) dr_messagebox(msg)
#else
#    define DISPLAY_STRING(msg) dr_printf("%s\n", msg);
#endif

#define NULL_TERMINATE(buf) (buf)[(sizeof((buf)) / sizeof((buf)[0])) - 1] = '\0'

/* We keep a separate execution count per opcode.
 *
 * XXX: our counters are racy on ARM.  We use DRX_COUNTER_LOCK to make them atomic
 * (at a perf cost) on x86.
 *
 * XXX: we're using 32-bit counters.  64-bit counters are more challenging: they're
 * harder to make atomic on 32-bit x86, and drx does not yet support them on ARM.
 */
enum {
#ifdef X86
    ISA_X86_32,
    ISA_X86_64,
#elif defined(ARM)
    ISA_ARM_A32,
    ISA_ARM_THUMB,
#elif defined(AARCH64)
    ISA_ARM_A64,
#elif defined(RISCV64)
    ISA_RV64IMAFDC,
#endif
    NUM_ISA_MODE,
};


int MemoryMapping[9] = {0, 1, 2, 4, 8, 16, 32, 64, 128};

bool countopcodes = true;

char messagesMem[9][64] = {
        "Total",
        "1 byte",
        "2 bytes",
        "4 bytes",
        "8 bytes",
        "16 bytes",
        "32 bytes",
        "64 bytes",
        "128 bytes / error",
    };
#ifdef X86
    char messagesArithx86[6][64] = {
            "Total",
            "Scalar",
            "SSE",
            "AVX2",
            "AVX512",
            "Error",
        };
#elif defined(AARCH64)
    char messagesArithARM[11][64] = {
        "Total",
        "(2x 64 bit)",
        "(1x 64 bit)",
        "(4x 32 bit)",
        "(2x 32 bit)",
        "(1x 32 bit)",
        "(8x 16 bit)",
        "(4x 16 bit)",
        "(16x 8 bit)",
        "(8x 8 bit)",
        "Error"
        };
    struct ARMIopsMapping {
        const char *reg;
        const char *code;
        int Iop;
        int coding;
    };

    int Iop = 0;

    struct ARMIopsMapping AddIops[] = {
            {"%q", "$0x03", 2, 1},   // 2x 64 bit Integer/Float
            {"%x", "$0x00",	1, 2},   // 1x 64 bit Integer
            {"%q", "$0x02", 4, 3},   // 4x 32 bit Integer/Float
            {"%d", "$0x02", 2, 4},   // 2x 32 bit Integer/Float
            {"%w", "$0x00", 1, 5},	 // 1x 32 bit Integer
            {"%q", "$0x01", 8, 6},   // 8x 16 bit Integer
            {"%d", "$0x01", 4, 7},   // 4x 16 bit Integer
            {"%q", "$0x00", 16, 8},  // 16x 8 bit Integer
            {"%d", "$0x00", 8, 9},   // 8x 8 bit Integer
    };
#endif

static uint64 count[NUM_ISA_MODE][OP_LAST + 1];
static uint64 countMem[NUM_ISA_MODE][OP_LAST + 1][9];
#ifdef X86
    static uint64 countArithx86[NUM_ISA_MODE][OP_LAST + 1][6];
#elif defined(AARCH64)
    static uint64 countArithARM[NUM_ISA_MODE][OP_LAST + 1][11];
#endif

#define NUM_COUNT sizeof(count[0]) / sizeof(count[0][0])

#define NUM_COUNT_SHOW 300
static bool roi_enabled = false;

static void
event_exit(void);
static dr_emit_flags_t
event_app_instruction(void *drcontext, void *tag, instrlist_t *bb, instr_t *instr,
                      bool for_trace, bool translating, void *user_data);

DR_EXPORT void
dr_client_main(client_id_t id, int argc, const char *argv[])
{
    dr_set_client_name("DynamoRIO Sample Client 'opcoder'",
                       "http://dynamorio.org/issues");
    if (!drmgr_init())
        DR_ASSERT(false);
    drx_init();

    /* Parse command line arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-roi") == 0) {
            roi_enabled = true;
            countopcodes = false;
            fprintf(stderr, "Region of Interest Enabled\n");
            break;/* Optionally, you can break here if -roi doesn't take any additional arguments */
        }
        else{
            fprintf(stderr, "Region of Interest Disabled\n");
        }
    }

    /* Register events: */
    dr_register_exit_event(event_exit);
    if (!drmgr_register_bb_instrumentation_event(NULL, event_app_instruction, NULL))
        DR_ASSERT(false);

    /* Make it easy to tell from the log file which client executed. */
    dr_log(NULL, DR_LOG_ALL, 1, "Client 'opcoder' initializing\n");
#ifdef SHOW_RESULTS
    /* Also give notification to stderr. */
    if (dr_is_notify_on()) {
#    ifdef WINDOWS
        /* Ask for best-effort printing to cmd window.  Must be called at init. */
        dr_enable_console_printing();
#    endif
        dr_fprintf(STDERR, "Client opcoder is running\n");
    }
#endif
}

#ifdef SHOW_RESULTS
/* We use cur_isa to iterate each ISA counters in event_exit, so there will be
 * no race on accessing it in compare_counts.
 */
static uint cur_isa;
static int
compare_counts(const void *a_in, const void *b_in)
{
    const uint64 a = *(const uint64 *)a_in;
    const uint64 b = *(const uint64 *)b_in;
    if (count[cur_isa][a] > count[cur_isa][b])
        return 1;
    if (count[cur_isa][a] < count[cur_isa][b])
        return -1;
    return 0;
}

static int
compare_countsMem(const void *a_in, const void *b_in)
{
    const uint64 a = *(const uint64 *)a_in;
    const uint64 b = *(const uint64 *)b_in;
    if (countMem[cur_isa][a][0] > countMem[cur_isa][b][0])
        return 1;
    if (countMem[cur_isa][a][0] < countMem[cur_isa][b][0])
        return -1;
    return 0;
}

#ifdef X86
    static int
    compare_countsArith(const void *a_in, const void *b_in)
    {
        const uint64 a = *(const uint64 *)a_in;
        const uint64 b = *(const uint64 *)b_in;
        if (countArithx86[cur_isa][a][0] > countArithx86[cur_isa][b][0])
            return 1;
        if (countArithx86[cur_isa][a][0] < countArithx86[cur_isa][b][0])
            return -1;
        return 0;
    }
#elif defined(AARCH64)
    static int
    compare_countsArith(const void *a_in, const void *b_in)
    {
        const uint64 a = *(const uint64 *)a_in;
        const uint64 b = *(const uint64 *)b_in;
        if (countArithARM[cur_isa][a][0] > countArithARM[cur_isa][b][0])
            return 1;
        if (countArithARM[cur_isa][a][0] < countArithARM[cur_isa][b][0])
            return -1;
        return 0;
    }
#endif
void opcode_totalMem()
{
    int i, j;
    for (j=0; j<=OP_LAST; j++){
            countMem[cur_isa][j][0] = 0;
            for (i=1; i<9; i++){
                    countMem[cur_isa][j][0] += countMem[cur_isa][j][i];

            }
    }
}

#ifdef X86
    void opcode_totalArith()
    {
        int i, j;
        for (j=0; j<=OP_LAST; j++){
                countArithx86[cur_isa][j][0] = 0;
                for (i=1; i<6; i++){
                        countArithx86[cur_isa][j][0] += countArithx86[cur_isa][j][i];

                }
        }
    }
#elif defined(AARCH64)
    void opcode_totalArith()
    {
        int i, j;
        for (j=0; j<=OP_LAST; j++){
                countArithARM[cur_isa][j][0] = 0;
                for (i=1; i<11; i++){
                        countArithARM[cur_isa][j][0] += countArithARM[cur_isa][j][i];

                }
        }
    }
#endif

static const char *
get_isa_mode_name(uint isa_mode)
{
#    ifdef X86
    return (isa_mode == ISA_X86_32) ? "32-bit X86" : "64-bit AMD64";
#    elif defined(ARM)
    return (isa_mode == ISA_ARM_A32) ? "32-bit ARM" : "32-bit Thumb";
#    elif defined(AARCH64)
    return "64-bit AArch64";
#    else
    return "unknown";
#    endif
}
#endif

static void
event_exit(void)
{
#ifdef SHOW_RESULTS
file_t file = dr_open_file("output.txt", DR_FILE_WRITE_OVERWRITE);
    char msg[NUM_COUNT_SHOW * 80];
    int len, i, j, index;
    size_t sofar = 0;

    /* First, sort the counts */
    uint64 indices[NUM_COUNT];
    for (cur_isa = 0; cur_isa < NUM_ISA_MODE; cur_isa++) {
        opcode_totalArith();
        opcode_totalMem();
        sofar = 0;
        //FP and INT OPCODES
        for (i = 0; i <= OP_LAST; i++)
            indices[i] = i;
        qsort(indices, NUM_COUNT, sizeof(indices[0]), compare_countsArith);

        #ifdef X86
            if (countArithx86[cur_isa][indices[OP_LAST]] == 0)
                continue;
            len = dr_snprintf(msg + sofar, sizeof(msg) / sizeof(msg[0]) - sofar,
                    "Floating Point and Integer opcode execution counts in %s mode:\n",
                    get_isa_mode_name(cur_isa));
            DR_ASSERT(len > 0);
            sofar += len;
            for (i = 0; i <= OP_LAST; i++) {
                int totalflag = 0;
                if (countArithx86[cur_isa][indices[i]][0] != 0) {
                    for (j = 1; j<6; j++){
                        if (countArithx86[cur_isa][indices[i]][j] != 0) {
                            totalflag += 1;
                        } 
                    }
                    if (totalflag > 1){
                        len = dr_snprintf(msg + sofar, sizeof(msg) / sizeof(msg[0]) - sofar,
                            "\n  %15llu : %-15s \t TOTAL\n", countArithx86[cur_isa][indices[i]][0],
                            decode_opcode_name(indices[i]));
                    DR_ASSERT(len > 0);
                    sofar += len;
                    for (j = 1; j<6; j++){
                        if (countArithx86[cur_isa][indices[i]][j] != 0) {
                            len = dr_snprintf(msg + sofar, sizeof(msg) / sizeof(msg[0]) - sofar,
                                    " \t %15llu : %-15s | %s\n", countArithx86[cur_isa][indices[i]][j],
                                    decode_opcode_name(indices[i]), messagesArithx86[j]);
                            DR_ASSERT(len > 0);
                            sofar += len;
                        }
                    }
                        len = dr_snprintf(msg + sofar, sizeof(msg) / sizeof(msg[0]) - sofar, "\n");
                            DR_ASSERT(len > 0);
                            sofar += len;
                    }
                    else{
                        for (j = 1; j<6; j++){
                        if (countArithx86[cur_isa][indices[i]][j] != 0) {
                            len = dr_snprintf(msg + sofar, sizeof(msg) / sizeof(msg[0]) - sofar,
                                    "  %15llu : %-15s | %s\n", countArithx86[cur_isa][indices[i]][j],
                                    decode_opcode_name(indices[i]), messagesArithx86[j]);
                            DR_ASSERT(len > 0);
                            sofar += len;
                        }
                    }
                    }
                }
            }
        #elif defined(AARCH64)
            if (countArithARM[cur_isa][indices[OP_LAST]] == 0)
                continue;
            len = dr_snprintf(msg + sofar, sizeof(msg) / sizeof(msg[0]) - sofar,
                    "Floating Point and Integer opcode execution counts in %s mode:\n",
                    get_isa_mode_name(cur_isa));
            DR_ASSERT(len > 0);
            sofar += len;
            for (i = 0; i <= OP_LAST; i++) {
                int totalflag = 0;
                if (countArithARM[cur_isa][indices[i]][0] != 0) {
                    for (j = 1; j<11; j++){
                        if (countArithARM[cur_isa][indices[i]][j] != 0) {
                            totalflag += 1;
                        } 
                    }
                    if (totalflag > 1){
                        len = dr_snprintf(msg + sofar, sizeof(msg) / sizeof(msg[0]) - sofar,
                            "\n  %15llu : %-15s \t TOTAL\n", countArithARM[cur_isa][indices[i]][0],
                            decode_opcode_name(indices[i]));
                    DR_ASSERT(len > 0);
                    sofar += len;
                    for (j = 1; j<11; j++){
                        if (countArithARM[cur_isa][indices[i]][j] != 0) {
                            len = dr_snprintf(msg + sofar, sizeof(msg) / sizeof(msg[0]) - sofar,
                                    " \t %15llu : %-15s | %s\n", countArithARM[cur_isa][indices[i]][j],
                                    decode_opcode_name(indices[i]), messagesArithARM[j]);
                            DR_ASSERT(len > 0);
                            sofar += len;
                        }
                    }
                        len = dr_snprintf(msg + sofar, sizeof(msg) / sizeof(msg[0]) - sofar, "\n");
                            DR_ASSERT(len > 0);
                            sofar += len;
                    }
                    else{
                        for (j = 1; j<11; j++){
                        if (countArithARM[cur_isa][indices[i]][j] != 0) {
                            len = dr_snprintf(msg + sofar, sizeof(msg) / sizeof(msg[0]) - sofar,
                                    "  %15llu : %-15s | %s\n", countArithARM[cur_isa][indices[i]][j],
                                    decode_opcode_name(indices[i]), messagesArithARM[j]);
                            DR_ASSERT(len > 0);
                            sofar += len;
                        }
                    }
                    }
                }
            }

        #endif
        //MEMORY, FP and INT OPCODES
        for (i = 0; i <= OP_LAST; i++)
            indices[i] = i;
        qsort(indices, NUM_COUNT, sizeof(indices[0]), compare_countsMem);

        if (countMem[cur_isa][indices[OP_LAST]] == 0)
            continue;
        len = dr_snprintf(msg + sofar, sizeof(msg) / sizeof(msg[0]) - sofar,
                "\nMemory opcode execution counts in %s mode:\n",
                get_isa_mode_name(cur_isa));
        DR_ASSERT(len > 0);
        sofar += len;
        for (i = 0; i <= OP_LAST; i++) {
            int totalflag = 0;
            if (countMem[cur_isa][indices[i]][0] != 0) {
                for (j = 1; j<9; j++){
                    if (countMem[cur_isa][indices[i]][j] != 0) {
                        totalflag += 1;
                    } 
                }
                if (totalflag > 1){
                    len = dr_snprintf(msg + sofar, sizeof(msg) / sizeof(msg[0]) - sofar,
                        "\n  %15llu : %-15s \t TOTAL\n", countMem[cur_isa][indices[i]][0],
                        decode_opcode_name(indices[i]));
                DR_ASSERT(len > 0);
                sofar += len;
                for (j = 1; j<9; j++){
                    if (countMem[cur_isa][indices[i]][j] != 0) {
                        len = dr_snprintf(msg + sofar, sizeof(msg) / sizeof(msg[0]) - sofar,
                                " \t %15llu : %-15s | %s\n", countMem[cur_isa][indices[i]][j],
                                decode_opcode_name(indices[i]), messagesMem[j]);
                        DR_ASSERT(len > 0);
                        sofar += len;
                    }
                }
                    len = dr_snprintf(msg + sofar, sizeof(msg) / sizeof(msg[0]) - sofar, "\n");
                        DR_ASSERT(len > 0);
                        sofar += len;
                }
                else{
                    for (j = 1; j<9; j++){
                    if (countMem[cur_isa][indices[i]][j] != 0) {
                        len = dr_snprintf(msg + sofar, sizeof(msg) / sizeof(msg[0]) - sofar,
                                "  %15llu : %-15s | %s\n", countMem[cur_isa][indices[i]][j],
                                decode_opcode_name(indices[i]), messagesMem[j]);
                        DR_ASSERT(len > 0);
                        sofar += len;
                    }
                }
                }
            }
        }
        //Miscellaneous OPCODES
        for (i = 0; i <= OP_LAST; i++)
            indices[i] = i;
        qsort(indices, NUM_COUNT, sizeof(indices[0]), compare_counts);

        if (count[cur_isa][indices[OP_LAST]] == 0)
            continue;
        len = dr_snprintf(msg + sofar, sizeof(msg) / sizeof(msg[0] - sofar),
                          "\nMiscellaneous Opcode execution counts in %s mode:\n",
                          get_isa_mode_name(cur_isa));
        DR_ASSERT(len > 0);
        sofar += len;
        for (i = 0; i <= OP_LAST; i++) {
            if (count[cur_isa][indices[i]] != 0) {
                len = dr_snprintf(msg + sofar, sizeof(msg) / sizeof(msg[0]) - sofar,
                                  "  %15llu : %-15s\n", count[cur_isa][indices[i]],
                                  decode_opcode_name(indices[i]));
                DR_ASSERT(len > 0);
                sofar += len;
            }
        }

        NULL_TERMINATE(msg);
        dr_write_file(file, msg, sofar);
                dr_close_file(file);
    }
#endif /* SHOW_RESULTS */
    if (!drmgr_unregister_bb_insertion_event(event_app_instruction))
        DR_ASSERT(false);
    drx_exit();
    drmgr_exit();
}

static uint
get_count_isa_idx(void *drcontext)
{
    switch (dr_get_isa_mode(drcontext)) {
#ifdef X86
    case DR_ISA_X86: return ISA_X86_32;
    case DR_ISA_AMD64: return ISA_X86_64;
#elif defined(ARM)
    case DR_ISA_ARM_A32: return ISA_ARM_A32; break;
    case DR_ISA_ARM_THUMB: return ISA_ARM_THUMB;
#elif defined(AARCH64)
    case DR_ISA_ARM_A64: return ISA_ARM_A64;
#endif
    default: DR_ASSERT(false); /* NYI */
    }
    return 0;
}

/* This is called separately for each instruction in the block. */

static dr_emit_flags_t
event_app_instruction(void *drcontext, void *tag, instrlist_t *bb, instr_t *instr,
                      bool for_trace, bool translating, void *user_data)
{
    drmgr_disable_auto_predication(drcontext, bb);
    if (drmgr_is_first_instr(drcontext, instr)) {
        instr_t *ins;
        
        uint isa_idx = get_count_isa_idx(drcontext);
        char disas_instr[256];
        
        for (ins = instrlist_first_app(bb); ins != NULL; ins = instr_get_next_app(ins)) {
            #if defined(AARCH64)
                Iop=1;
                int i;
                int j;
                int size = 0;
                int optype = 0;
                int codeMem = 8;
                int codeArith = 5;
                size = instr_memory_reference_size(ins);

                if (size > 0){
                    //Memory section
                    for (j = 0; j < 9; j++){
                        if (size == MemoryMapping[j]){
                            codeMem = j;
                            break;
                        }
                    }
                    drx_insert_counter_update(drcontext, bb, instr,
                            SPILL_SLOT_MAX + 1,
                            IF_AARCHXX_(SPILL_SLOT_MAX + 1) &
                                countMem[isa_idx][instr_get_opcode(ins)][codeMem],
                            1,
                            /* DRX_COUNTER_LOCK is not yet supported on ARM */
                            DRX_COUNTER_64BIT | DRX_COUNTER_REL_ACQ);
                }else{
                    //FP and Integer Section (and other operations)
                    instr_disassemble_to_buffer(drcontext, ins, disas_instr, 256);
                    if (strstr(disas_instr, "add") != NULL || strstr(disas_instr, "sub") != NULL || strstr(disas_instr, "mul") != NULL || strstr(disas_instr, "fmla") != NULL || strstr(disas_instr, "madd") != NULL){
                        
                        //1x64 and 1x32 Float Special Case
                        if (disas_instr[0] == 'f' && strstr(disas_instr, "$0x") == NULL && (strstr(disas_instr, "%d") != NULL) || (strstr(disas_instr, "%s") != NULL)){
                            //1x64 Float
                            if (strstr(disas_instr, "%d") != NULL){
                                Iop = 1;
                                codeArith = 2;
                            }
                            //1x32 Float
                            else {
                                Iop = 1;
                                codeArith = 5;
                            }
                        }
                        else{
                            for (i = 0; i < 9; ++i) {
                                if (strstr(disas_instr, AddIops[i].reg) != NULL && strstr(disas_instr, AddIops[i].code) != NULL) {
                                    Iop = AddIops[i].Iop;
                                    codeArith = AddIops[i].coding;
                                    break;
                                }	
                            }
                        }
                        drx_insert_counter_update(drcontext, bb, instr,
                                SPILL_SLOT_MAX + 1,
                                IF_AARCHXX_(SPILL_SLOT_MAX + 1) &
                                countArithARM[isa_idx][instr_get_opcode(ins)][codeArith],
                                1,
                                /* DRX_COUNTER_LOCK is not yet supported on ARM */
                                DRX_COUNTER_64BIT | DRX_COUNTER_REL_ACQ);
                    }else {
                        //OTHER OPERATIONS SECTION
                        drx_insert_counter_update(drcontext, bb, instr,
                                /* We're using drmgr, so these slots
                                * here won't be used: drreg's slots will be.
                                */
                                SPILL_SLOT_MAX + 1,
                                IF_AARCHXX_(SPILL_SLOT_MAX + 1) &
                                count[isa_idx][instr_get_opcode(ins)],
                                1,
                                /* DRX_COUNTER_LOCK is not yet supported on ARM */
                                DRX_COUNTER_64BIT | DRX_COUNTER_REL_ACQ);
                    }
                }
            #else
                int size = 0;
                int j;
                int codeMem = 8;
                int codeArith = 5;
                
                instr_disassemble_to_buffer(drcontext, ins, disas_instr, 256);
                if (strstr(disas_instr, "face") != NULL){
                    countopcodes = true;
                }

                if (countopcodes == true){
                    if (strstr(disas_instr, "dead") != NULL && roi_enabled){
                        countopcodes = false;
                    }
                    size = instr_memory_reference_size(ins);
                    if (size > 0){
                        if (strstr(disas_instr, "add") != NULL || strstr(disas_instr, "sub") != NULL || strstr(disas_instr, "mul") != NULL || strstr(disas_instr, "div") != NULL){
                            //AVX2 and AVX512 Floating Point Section
                            if (size == 32){
                                codeArith = 3;
                            } else if (size == 64){
                                codeArith = 4;
                            } else if (size == 16){
                                codeArith = 2;
                            } else {
                                codeArith = 1;
                            }
                            drx_insert_counter_update(drcontext, bb, instr,
                                                SPILL_SLOT_MAX + 1,
                                                IF_AARCHXX_OR_RISCV64_(SPILL_SLOT_MAX + 1) &
                                                countArithx86[isa_idx][instr_get_opcode(ins)][codeArith],
                                                1,
                                                //DRX_COUNTER_LOCK is not yet supported on ARM 
                                                DRX_COUNTER_64BIT | DRX_COUNTER_LOCK);
                        }
                        else{
                            //MEMORY SECTION
                            for (j = 0; j < 9; j++){
                                if (size == MemoryMapping[j]){
                                    codeMem = j;
                                    break;
                                }
                            }
                            
                            drx_insert_counter_update(drcontext, bb, instr,
                                                    /* We're using drmgr, so these slots
                                                    * here won't be used: drreg's slots will be.
                                                    */
                                                    SPILL_SLOT_MAX + 1,
                                                    IF_AARCHXX_OR_RISCV64_(SPILL_SLOT_MAX + 1) &
                                                    countMem[isa_idx][instr_get_opcode(ins)][codeMem],
                                                    1,
                                                    /* DRX_COUNTER_LOCK is not yet supported on ARM */
                                                    DRX_COUNTER_64BIT | DRX_COUNTER_LOCK);
                        }
                    }
                    else{
                        //FP and Integer section
                        if (strstr(disas_instr, "add") != NULL || strstr(disas_instr, "sub") != NULL || strstr(disas_instr, "mul") != NULL || strstr(disas_instr, "div") != NULL){

                            if (strstr(disas_instr, "xmm") != NULL){
                                if (strstr(disas_instr, "byte") != NULL){
                                    codeArith = 1;
                                }else{
                                    codeArith = 2;
                                }
                            }else if(strstr(disas_instr, "ymm") != NULL){
                                codeArith = 3;
                            }else if(strstr(disas_instr, "zmm") != NULL){
                                codeArith = 4;
                            }else {
                                if (strstr(disas_instr, "byte") != NULL){
                                }
                                codeArith = 1;
                            }

                            drx_insert_counter_update(drcontext, bb, instr,
                                                /* We're using drmgr, so these slots
                                                * here won't be used: drreg's slots will be.
                                                */
                                                SPILL_SLOT_MAX + 1,
                                                IF_AARCHXX_OR_RISCV64_(SPILL_SLOT_MAX + 1) &
                                                    countArithx86[isa_idx][instr_get_opcode(ins)][codeArith],
                                                1,
                                                /* DRX_COUNTER_LOCK is not yet supported on ARM */
                                                DRX_COUNTER_64BIT | DRX_COUNTER_LOCK);

                        }
                        //Other instructions
                        else{
                            drx_insert_counter_update(drcontext, bb, instr,
                                                SPILL_SLOT_MAX + 1,
                                                IF_AARCHXX_OR_RISCV64_(SPILL_SLOT_MAX + 1) &
                                                    count[isa_idx][instr_get_opcode(ins)],
                                                1,
                                                //DRX_COUNTER_LOCK is not yet supported on ARM 
                                                DRX_COUNTER_64BIT | DRX_COUNTER_LOCK);
                        }
                        
                    }
                }
                else{
                    continue;
                }
            #endif
        }
    
    }
    return DR_EMIT_DEFAULT;
}
