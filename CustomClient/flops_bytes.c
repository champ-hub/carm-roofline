
#include <stdio.h>
#include <stddef.h> /* for offsetof */
#include <string.h> /* for offsetof */
#include <sched.h>
#include "dr_api.h"
#include "drmgr.h"
#include "drutil.h"
#include "drreg.h"
#include "drx.h"
#include <stdint.h>


#define NEXT_ITERATION_INSTR  0x2520e060

enum {
    REF_TYPE_FPINS = 0,
    REF_TYPE_MEMINS = 1,
    REF_TYPE_INTINS = 2,
};
/* Struct to hold floating point operation and memory load/store counts for each thread */
typedef struct _thread_counters_t {
	uint64_t fp_ops;
	uint64_t int_ops;
	uint64_t bytes;
	uint64_t fp_ins;
	uint64_t int_ins;
} thread_counters_t;

typedef struct _sve_predicated_instr_t {
	ushort type; /* equal to 0 if floating-point instruction, 1 if memory instruction */
	ushort elem_size; /* number of operations per element (flops for fpins, bytes for memins) */
	int active_elem_count; /* number of active elements */
} sve_predicated_instr_t;

#define MAX_NUM_MEM_REFS 4096
/* The maximum size of buffer for holding mem_refs. */
#define MEM_BUF_SIZE (sizeof(sve_predicated_instr_t) * MAX_NUM_MEM_REFS)

#ifdef AARCH64
static bool reported_sg_warning = false;
#endif


typedef struct {
    byte *seg_base;
    sve_predicated_instr_t *buf_base;
    thread_counters_t *counters;
    bool countopcodes;
} per_thread_t;

static void *mutex;     /* for multithread support */

char filename[256];
FILE *logfile;
uint64_t global_flops_count=0;
uint64_t global_bytes_count=0;
uint64_t global_fpins_count=0;
uint64_t global_intops_count=0;
uint64_t global_intins_count=0;
static int tls_idx;
static app_pc exe_start; /* Application module address used to avoid tracing*/

static bool roi_enabled=false;
bool countopcodes = true;
bool firstpass_countopcodes = true;

/* Allocated TLS slot offsets */
enum {
    MEMTRACE_TLS_OFFS_BUF_PTR,
    MEMTRACE_TLS_COUNT, /* total number of TLS slots allocated */
};
static reg_id_t tls_seg;
static uint tls_offs;
static int tls_buf;
#define TLS_SLOT(tls_base, enum_val) (void **)((byte *)(tls_base) + tls_offs + (enum_val))
#define BUF_PTR(tls_base) *(sve_predicated_instr_t **)TLS_SLOT(tls_base, MEMTRACE_TLS_OFFS_BUF_PTR)

#define MINSERT instrlist_meta_preinsert


static dr_emit_flags_t
event_bb_app2app(void *drcontext, void *tag, instrlist_t *bb, bool for_trace, bool translating);

static dr_emit_flags_t
event_app_instruction(void *drcontext, void *tag, instrlist_t *bb, instr_t *instr,
		bool for_trace, bool translating,void *user_data);

static dr_emit_flags_t
event_bb_analysis(void *drcontext, void *tag, instrlist_t *bb, bool for_trace,
                  bool translating, void **user_data);

static void
event_thread_init(void *drcontext);

static void
event_thread_exit(void *drcontext);

static void
event_exit(void);

/* ------------- CLEAN CALL FUNCTIONS --------------------- */
	static void
count_call(uint64_t flops_count,uint64_t bytes_count,uint64_t fpins_count,uint64_t intins_count,uint64_t intops_count)
{
	void *drcontext = dr_get_current_drcontext();
	//thread_counters_t *data = (thread_counters_t *) drmgr_get_tls_field(drcontext, tls_idx);
	

	per_thread_t *data = drmgr_get_tls_field(drcontext, tls_buf);
	//if (countopcodes/* && data->countopcodes*/) {
	data->counters = (thread_counters_t *) drmgr_get_tls_field(drcontext, tls_idx);
	data->counters->fp_ops += flops_count;
	data->counters->fp_ins += fpins_count;
	data->counters->bytes += bytes_count;
	data->counters->int_ops += intops_count;
	data->counters->int_ins += intins_count;
	//}
}

	static void
count_sve_predicated_instructions(void *drcontext)
{
	per_thread_t *data;
	sve_predicated_instr_t *sve_predicated_instr, *buf_ptr;

	data = drmgr_get_tls_field(drcontext, tls_buf);
	data->counters = (thread_counters_t *) drmgr_get_tls_field(drcontext, tls_idx);
	buf_ptr = BUF_PTR(data->seg_base);

	for (sve_predicated_instr = (sve_predicated_instr_t *)data->buf_base; sve_predicated_instr < buf_ptr; sve_predicated_instr++) {
			if (sve_predicated_instr->type == 0) { //Is a floating point instruction
				data->counters->fp_ops += sve_predicated_instr->elem_size * sve_predicated_instr->active_elem_count;
			}
			if (sve_predicated_instr->type == 2) { //Is a int ops instruction
				data->counters->int_ops += sve_predicated_instr->elem_size * sve_predicated_instr->active_elem_count;
			}
			else  {//Is a mem instruction 
				data->counters->bytes += sve_predicated_instr->elem_size * sve_predicated_instr->active_elem_count;
			}
	}
	BUF_PTR(data->seg_base) = data->buf_base;
}
static void
enable_roi()
{
	void *drcontext = dr_get_current_drcontext();
	per_thread_t *data;
	data = drmgr_get_tls_field(drcontext, tls_buf);
	data->countopcodes = true;

}
static void
disable_roi()
{
	void *drcontext = dr_get_current_drcontext();
	per_thread_t *data;
	data = drmgr_get_tls_field(drcontext, tls_buf);
	data->countopcodes = false;

}
static void
sve_instr_call()
{
	void *drcontext = dr_get_current_drcontext();
	count_sve_predicated_instructions(drcontext);

}
/* ------------- UTILITY FUNCTIONS --------------------- */

/* utility function to determine if an instruction is a floating-point operation */
	bool 
is_flop (instr_t *instr)
{
	int opc = instr_get_opcode(instr);
	return (     opc == OP_fabd 
			|| (opc >= OP_fadd && opc <= OP_faddp) 
			|| (opc >= OP_fdiv && opc <= OP_fmadd) 
			|| (opc >= OP_fmla && opc <= OP_fmlsl2) 
			|| (opc >= OP_fmsub && opc <= OP_fmulx) 
			|| (opc >= OP_fnmadd && opc <= OP_fnmul) 
			|| (opc >= OP_frsqrte && opc <= OP_fsub)
			|| (opc >= OP_fexpa && opc <= OP_ftsmul)
			|| (opc >= OP_fadda && opc <= OP_faddv)
			|| (opc >= OP_fdivr && opc <= OP_fmsb)
			|| (opc == OP_fnmsb)
			|| (opc >= OP_fmlalb && opc <= OP_fmlslt)
		   ); 
}
	bool 
may_be_marker (instr_t *instr)
{
	int opc = instr_get_opcode(instr);
	return (opc == OP_movz); 
}


/* utility function to determine if an instruction is a int operation */
	bool 
is_intop (instr_t *instr)
{
	int opc = instr_get_opcode(instr);
	return ((opc >= OP_add && opc <= OP_addv) 
			//|| (opc >= OP_ldadd && opc <= OP_ldaddlh) 
			|| (opc >= OP_madd && opc <= OP_mls) 
			|| (opc >= OP_msub && opc <= OP_mul) 
			|| (opc >= OP_pmul && opc <= OP_pmull2) 
			|| (opc >= OP_raddhn && opc <= OP_raddhn2)
			|| (opc >= OP_rsubhn && opc <= OP_sbcs)
			|| (opc == OP_shadd )
			|| (opc == OP_shsub )
			|| (opc == OP_smaddl )
			|| (opc >= OP_smlal && opc <= OP_smlsl2)
			|| (opc >= OP_smsubl && opc <= OP_smull2)
			|| (opc >= OP_sqadd && opc <= OP_sqdmull2)
			|| (opc >= OP_sqrdmlah && opc <= OP_sqrdmulh)
			|| (opc == OP_sqsub)
			|| (opc == OP_srhadd)
			|| (opc >= OP_ssubl && opc <= OP_ssubw2)
			|| (opc >= OP_sub && opc <= OP_suqadd)
			|| (opc >= OP_uaba && opc <= OP_uaddw2)
			|| (opc >= OP_udiv && opc <= OP_umaddl)
			|| (opc >= OP_umlal && opc <= OP_umlsl2)
			|| (opc >= OP_umsubl && opc <= OP_uqadd)
			|| (opc == OP_urhadd)
			|| (opc == OP_usqadd)
			|| (opc >= OP_usubl && opc <= OP_usubw2)
			|| (opc >= OP_usubl && opc <= OP_usubw2)
			); 
}

bool is_int_fma (instr_t *instr)
{
	int opc = instr_get_opcode(instr);
	if ((opc >= OP_madd && opc <= OP_mls)  //multiply-add on vector 
			|| opc == OP_msub  // multiply-sub
			|| opc == OP_smaddl // multiply-add with negation
			|| (opc >= OP_smlal && opc <= OP_smlsl2)  //multiply-add on vector 
			|| opc == OP_smsubl // multiply-sub with negation
			|| (opc >= OP_sqdmlal && opc <= OP_sqdmlsl2)  //multiply-add on vector 
			|| opc == OP_sqrdmlah // multiply-sub with negation
			|| opc == OP_umaddl // multiply-sub with negation
			|| (opc >= OP_umlal && opc <= OP_umlsl2)  //multiply-add on vector 
			|| opc == OP_umsubl // multiply-sub with negation
	   )
		return true;
	return false;
}

bool is_fma (instr_t *instr)
{
	int opc = instr_get_opcode(instr);
	if (opc == OP_fmadd // multiply-add
			|| (opc >= OP_fmla && opc <= OP_fmlsl2)  //multiply-add on vector 
			|| opc == OP_fmsub  // multiply-sub
			|| opc == OP_fnmadd // multiply-add with negation
			|| opc == OP_fnmsub // multiply-sub with negation
			|| (opc >= OP_fmad && opc <= OP_fmsb)  //multiply-add on vector 
			|| (opc >= OP_fcadd && opc <= OP_fcmla)  //multiply-add on vector 
			|| opc == OP_fnmsb // multiply-sub with negation
			|| (opc >= OP_fmlalb && opc <= OP_fmlslt)  //multiply-add on vector 


	   )
		return true;
	return false;
}


bool is_sve_instr(instr_t *instr)
{
	reg_id_t reg;
	int i;
	for (i = 0; i < instr_num_srcs(instr); i++) {
		if (opnd_is_reg(instr_get_src(instr,i))) {
			reg = opnd_get_reg(instr_get_src(instr,i));
			if ((DR_REG_Z0 <= reg && reg <= DR_REG_Z31))
				return true;
		}
	}
	for (i = 0; i < instr_num_dsts(instr); i++) {
		if (opnd_is_reg(instr_get_dst(instr,i))) {
			reg = opnd_get_reg(instr_get_dst(instr,i));
			if ((DR_REG_Z0 <= reg && reg <= DR_REG_Z31))
				return true;
		}
	}
	return false;
}	

bool is_unsupported_instr(instr_t *instr)
{
	int opc = instr_get_opcode(instr);
	return false;
}

bool is_predicated_instr(instr_t *instr)
{
	reg_id_t reg;
	int i;
	for (i = 0; i < instr_num_srcs(instr); i++) {
		if (opnd_is_reg(instr_get_src(instr,i))) {
			reg = opnd_get_reg(instr_get_src(instr,i));
			if (DR_REG_P0 <= reg && reg <= DR_REG_P15)
				return true;
		}
	}
	return false;
}

opnd_t get_predicate_operand(instr_t *instr)
{
	reg_id_t reg;
	int i;
	for (i = 0; i < instr_num_srcs(instr); i++) {
		if (opnd_is_reg(instr_get_src(instr,i))) {
			reg = opnd_get_reg(instr_get_src(instr,i));
			if (reg >= DR_REG_P0 && reg <= DR_REG_P15) {
				return instr_get_src(instr,i);
			}
		}
	}
	for (i = 0; i < instr_num_dsts(instr); i++) {
		if (opnd_is_reg(instr_get_dst(instr,i))) {
			reg = opnd_get_reg(instr_get_dst(instr,i));
			if (reg >= DR_REG_P0 && reg <= DR_REG_P15)
				return instr_get_dst(instr,i);
		}
	}
}

/* Test if the instruction is working on a NEON vector register */ 
bool is_neon_instr(instr_t *instr)
{
	int reg = opnd_get_reg(instr_get_dst(instr,0));
	if  ((reg >= DR_REG_Q0 && reg <= DR_REG_Q31) 
				|| (reg >= DR_REG_D0 && reg <= DR_REG_D31) 
				&& opnd_is_immed_int(instr_get_src(instr,instr_num_srcs(instr) - 1)) )
		return true;
	return false;
}

/* Return the number of elements in a NEON vector 
 * Destination register size can be 8 or 16 bytes
 * Elements in vector size can be 2, 4 or 8 bytes according to the value in the third source register (added by dynamorio ?) */
int number_of_elements_in_neon_vector(instr_t *instr)
{
	int vec_size_in_bytes = opnd_size_in_bytes(opnd_get_size(instr_get_dst(instr,0)));
	int data_size_in_bytes = 0;//opnd_size_in_bytes(opnd_get_vector_element_size(instr_get_dst(instr,0)));
	int value_in_last_reg = opnd_get_immed_int(instr_get_src(instr,instr_num_srcs(instr) - 1));
	if (value_in_last_reg == 0)
		data_size_in_bytes = 1;
	if (value_in_last_reg == 1)
		data_size_in_bytes = 2;
	if (value_in_last_reg == 2)
		data_size_in_bytes = 4;
	if (value_in_last_reg == 3)
		data_size_in_bytes = 8;
	return vec_size_in_bytes/data_size_in_bytes;	
}

/* Return number of FLOP done by a floating-point instruction (SVE predicated instructions not supported) */
int n_flops(instr_t *instr)
{
	int value;
	if (is_fma(instr) || is_int_fma(instr)) 
		value = 2;
	else
		value = 1;
	if (is_sve_instr(instr)) {
		value *= opnd_size_in_bytes(opnd_get_size(instr_get_dst(instr,0)))/opnd_size_in_bytes(opnd_get_vector_element_size(instr_get_dst(instr,0))); 	
	}
	else if (is_neon_instr(instr))
		value *= number_of_elements_in_neon_vector(instr); 
	return value;    
}



/* ------------- INSTRUMENTATION FUNCTIONS --------------------- */
static void
insert_load_buf_ptr(void *drcontext, instrlist_t *ilist, instr_t *where, reg_id_t reg_ptr)
{
    dr_insert_read_raw_tls(drcontext, ilist, where, tls_seg,
                           tls_offs + MEMTRACE_TLS_OFFS_BUF_PTR, reg_ptr);
}

static void
insert_update_buf_ptr(void *drcontext, instrlist_t *ilist, instr_t *where,
                      reg_id_t reg_ptr, int adjust)
{
	MINSERT(
			ilist, where,
			XINST_CREATE_add(drcontext, opnd_create_reg(reg_ptr), OPND_CREATE_INT16(adjust)));
	dr_insert_write_raw_tls(drcontext, ilist, where, tls_seg,
			tls_offs + MEMTRACE_TLS_OFFS_BUF_PTR, reg_ptr);
}

static void
insert_save_type(void *drcontext, instrlist_t *ilist, instr_t *where, reg_id_t base,
                 reg_id_t scratch, ushort type)
{
    scratch = reg_resize_to_opsz(scratch, OPSZ_2);
    MINSERT(ilist, where,
            XINST_CREATE_load_int(drcontext, opnd_create_reg(scratch),
                                  OPND_CREATE_INT16(type)));
    MINSERT(ilist, where,
            XINST_CREATE_store_2bytes(drcontext,
                                      OPND_CREATE_MEM16(base, offsetof(sve_predicated_instr_t, type)),
                                      opnd_create_reg(scratch)));
}

static void
insert_save_elem_size(void *drcontext, instrlist_t *ilist, instr_t *where, reg_id_t base,
                 reg_id_t scratch, ushort elem_size)
{
    scratch = reg_resize_to_opsz(scratch, OPSZ_2);
    MINSERT(ilist, where,
            XINST_CREATE_load_int(drcontext, opnd_create_reg(scratch),
                                  OPND_CREATE_INT16(elem_size)));
    MINSERT(ilist, where,
            XINST_CREATE_store_2bytes(drcontext,
                                      OPND_CREATE_MEM16(base, offsetof(sve_predicated_instr_t, elem_size)),
                                      opnd_create_reg(scratch)));
}

static void
insert_save_active_elem_count(void *drcontext, instrlist_t *ilist, instr_t *where, opnd_t ref, reg_id_t base,
                 reg_id_t scratch)
{
	scratch = reg_resize_to_opsz(scratch, OPSZ_8);

	/*opnd_t opnd_tmp = opnd_create_reg_element_vector(DR_REG_P15,OPSZ_1);
	MINSERT(ilist, where,
			INSTR_CREATE_ptrue_sve(drcontext,opnd_tmp,opnd_create_immed_pred_constr(DR_PRED_CONSTR_ALL)));*/
	instr_t *instr = INSTR_CREATE_cntp_sve_pred(drcontext, opnd_create_reg(scratch),opnd_create_reg_element_vector(opnd_get_reg(ref),OPSZ_8),opnd_create_reg_element_vector(opnd_get_reg(ref),OPSZ_8));
	//instr_t *instr = INSTR_CREATE_cntp_sve_pred(drcontext, opnd_create_reg(scratch),ref ,opnd_tmp);
	MINSERT(ilist, where,
			instr);
	insert_load_buf_ptr(drcontext, ilist, where, base);
	MINSERT(ilist, where,
			XINST_CREATE_store(drcontext,
				OPND_CREATE_MEMPTR(base, offsetof(sve_predicated_instr_t, active_elem_count)),
				opnd_create_reg(scratch)));
}


/* insert inline code to add an SVE predicated floating-point instruction into the buffer */
static void
instrument_sve_predicated_fpins(void *drcontext, instrlist_t *ilist, instr_t *where,instr_t *instr, ushort flop_per_elem)
{
    /* We need two scratch registers */
    reg_id_t reg_ptr, reg_tmp;
    /* we don't want to predicate this, because an instruction fetch always occurs */
    instrlist_set_auto_predicate(ilist, DR_PRED_NONE);
    if (drreg_reserve_register(drcontext, ilist, where, NULL, &reg_ptr) !=
            DRREG_SUCCESS ||
        drreg_reserve_register(drcontext, ilist, where, NULL, &reg_tmp) !=
            DRREG_SUCCESS) {
        DR_ASSERT(false); /* cannot recover */
        return;
    }

    insert_save_active_elem_count(drcontext, ilist, where, get_predicate_operand(instr), reg_ptr, reg_tmp);
    insert_save_type(drcontext, ilist, where, reg_ptr, reg_tmp,0);
    insert_save_elem_size(drcontext, ilist, where, reg_ptr, reg_tmp, flop_per_elem);
    insert_update_buf_ptr(drcontext, ilist, where, reg_ptr, sizeof(sve_predicated_instr_t));

    /* Restore scratch registers */
    if (drreg_unreserve_register(drcontext, ilist, where, reg_ptr) != DRREG_SUCCESS ||
        drreg_unreserve_register(drcontext, ilist, where, reg_tmp) != DRREG_SUCCESS)
        DR_ASSERT(false);
    instrlist_set_auto_predicate(ilist, instr_get_predicate(where));
}

/* insert inline code to add an SVE predicated floating-point instruction into the buffer */
static void
instrument_sve_predicated_intins(void *drcontext, instrlist_t *ilist, instr_t *where,instr_t *instr, ushort flop_per_elem)
{
    /* We need two scratch registers */
    reg_id_t reg_ptr, reg_tmp;
    /* we don't want to predicate this, because an instruction fetch always occurs */
    instrlist_set_auto_predicate(ilist, DR_PRED_NONE);
    if (drreg_reserve_register(drcontext, ilist, where, NULL, &reg_ptr) !=
            DRREG_SUCCESS ||
        drreg_reserve_register(drcontext, ilist, where, NULL, &reg_tmp) !=
            DRREG_SUCCESS) {
        DR_ASSERT(false); /* cannot recover */
        return;
    }

    insert_save_active_elem_count(drcontext, ilist, where, get_predicate_operand(instr), reg_ptr, reg_tmp);
    insert_save_type(drcontext, ilist, where, reg_ptr, reg_tmp,2);
    insert_save_elem_size(drcontext, ilist, where, reg_ptr, reg_tmp, flop_per_elem);
    insert_update_buf_ptr(drcontext, ilist, where, reg_ptr, sizeof(sve_predicated_instr_t));

    /* Restore scratch registers */
    if (drreg_unreserve_register(drcontext, ilist, where, reg_ptr) != DRREG_SUCCESS ||
        drreg_unreserve_register(drcontext, ilist, where, reg_tmp) != DRREG_SUCCESS)
        DR_ASSERT(false);
    instrlist_set_auto_predicate(ilist, instr_get_predicate(where));
}


/* insert inline code to add an SVE predicated floating-point instruction into the buffer */
static void
instrument_sve_predicated_mem(void *drcontext, instrlist_t *ilist, instr_t *where,instr_t *instr, opnd_t ref)
{
    /* We need two scratch registers */
    reg_id_t reg_ptr, reg_tmp;
    /* we don't want to predicate this, because an instruction fetch always occurs */
    instrlist_set_auto_predicate(ilist, DR_PRED_NONE);
    if (drreg_reserve_register(drcontext, ilist, where, NULL, &reg_ptr) !=
            DRREG_SUCCESS ||
        drreg_reserve_register(drcontext, ilist, where, NULL, &reg_tmp) !=
            DRREG_SUCCESS) {
        DR_ASSERT(false); /* cannot recover */
        return;
    }
    insert_save_active_elem_count(drcontext, ilist, where, get_predicate_operand(instr), reg_ptr, reg_tmp);
    insert_save_elem_size(drcontext, ilist, where, reg_ptr, reg_tmp, opnd_size_in_bytes(opnd_get_vector_element_size(ref)));
    insert_save_type(drcontext, ilist, where, reg_ptr, reg_tmp,1);
    insert_update_buf_ptr(drcontext, ilist, where, reg_ptr, sizeof(sve_predicated_instr_t));
    /* Restore scratch registers */
    if (drreg_unreserve_register(drcontext, ilist, where, reg_ptr) != DRREG_SUCCESS ||
        drreg_unreserve_register(drcontext, ilist, where, reg_tmp) != DRREG_SUCCESS)
        DR_ASSERT(false);
    instrlist_set_auto_predicate(ilist, instr_get_predicate(where));
}

	static dr_emit_flags_t
event_bb_app2app(void *drcontext, void *tag, instrlist_t *bb, bool for_trace, bool translating)
{
	if (!drutil_expand_rep_string(drcontext, bb)) {
		DR_ASSERT(false);
		// in release build, carry on: we'll just miss per-iter refs 
	}
	if (!drx_expand_scatter_gather(drcontext, bb, NULL)) {
		DR_ASSERT(false);
	}
	return DR_EMIT_DEFAULT;
}


	static dr_emit_flags_t
event_bb_analysis(void *drcontext, void *tag, instrlist_t *bb, bool for_trace,
		bool translating, void **user_data)

{

	//drmgr_disable_auto_predication(drcontext, bb);
	//module_data_t *mod = dr_lookup_module(dr_fragment_app_pc(tag));
	//if (mod != NULL) {
	//	bool from_exe = (mod->start == exe_start);
	//	dr_free_module_data(mod);
	//	if (!from_exe)
	//		*user_data=NULL;
	//		return DR_EMIT_DEFAULT;
	//}
	instr_t *cursor;
	thread_counters_t *counter =(thread_counters_t *)dr_thread_alloc(drcontext, sizeof(thread_counters_t));
	int i;
	counter->fp_ins = 0;
	counter->fp_ops = 0;
	counter->bytes = 0;
	counter->int_ins = 0;
	counter->int_ops = 0;
        char disas_instr[256];

	for (cursor = instrlist_first(bb); cursor != NULL;
			cursor = instr_get_next(cursor)) {
		if (roi_enabled) {
			if (may_be_marker(cursor)) {
				instr_disassemble_to_buffer(drcontext, cursor, disas_instr, 256);
				if (strstr(disas_instr, "face") != NULL){

					if (dr_get_thread_id(drcontext) == dr_get_process_id()) {
						countopcodes = true;
						printf("In ROI \n");
					}
					continue;
				}

				if (strstr(disas_instr, "dead") != NULL && roi_enabled){
					if (dr_get_thread_id(drcontext) == dr_get_process_id()) {
						countopcodes = false;
						printf("Out of ROI \n");
					}
					continue;
				}
			}
			
		}

		if (countopcodes) {
			if (!instr_is_app(cursor))
				continue;
			if (is_intop(cursor)) {
				counter->int_ins++;
				if (is_predicated_instr(cursor)) {
					//counter->sve_instr++;
					continue;
				}
				counter->int_ops += n_flops(cursor);
				continue;
			}
			if (is_flop(cursor)) {
				counter->fp_ins++;
				if (is_predicated_instr(cursor)) {
					//counter->sve_instr++;
					continue;
				}
				counter->fp_ops += n_flops(cursor);
				continue;
			}

			if (instr_reads_memory(cursor) || instr_writes_memory(cursor)) {
				//counter->bytes +=(ushort)instr_length(drcontext, cursor);
				if (is_predicated_instr(cursor)) {
					//counter->sve_instr++;
					continue;
				}
				for (i = 0; i < instr_num_srcs(cursor); i++) {
					const opnd_t src = instr_get_src(cursor, i);
					if (opnd_is_memory_reference(src) ) {
						if (opnd_is_base_disp(src) && (reg_is_z(opnd_get_base(src)) || reg_is_z(opnd_get_index(src))))
							continue;
						counter->bytes += (ushort)opnd_size_in_bytes(opnd_get_size(src));
					}
				}
				for (i = 0; i < instr_num_dsts(cursor); i++) {
					const opnd_t dst = instr_get_dst(cursor, i);
					if (opnd_is_memory_reference(dst) ) {

						if (opnd_is_base_disp(dst) && (reg_is_z(opnd_get_base(dst)) || reg_is_z(opnd_get_index(dst))))
							continue;
						counter->bytes += (ushort)opnd_size_in_bytes(opnd_get_size(dst));
					}
				}
				continue;
			}	
		}
	}

	if (counter->fp_ops != 0 || counter->bytes != 0 || counter->int_ins != 0)//|| counter->sve_instr != 0)
		*user_data = (void *)(thread_counters_t *)counter;
	else
		*user_data = NULL;
	return DR_EMIT_DEFAULT;
}

	static dr_emit_flags_t
event_app_instruction(void *drcontext, void *tag, instrlist_t *bb, instr_t *instr,
		bool for_trace, bool translating,void *user_data)
{
	int i;
	drmgr_disable_auto_predication(drcontext, bb);
        char disas_instr[256];
	
	if (user_data != NULL && drmgr_is_first_instr(drcontext,instr)) {
		thread_counters_t *counter;
		counter = (thread_counters_t *)user_data;

		dr_insert_clean_call(drcontext, bb, instrlist_first_app(bb),
				(void *)count_call, false, 5,
				OPND_CREATE_INT(counter->fp_ops),
				OPND_CREATE_INT(counter->bytes),
				OPND_CREATE_INT(counter->fp_ins),
				OPND_CREATE_INT(counter->int_ins),
				OPND_CREATE_INT(counter->int_ops)
				);
		//dr_thread_free(drcontext, counter, sizeof(thread_counters_t));
	}

	instr_t *instr_fetch = drmgr_orig_app_instr_for_fetch(drcontext);

	if (roi_enabled) {
		if (may_be_marker(instr)) {
			instr_disassemble_to_buffer(drcontext, instr, disas_instr, 256);
			if (strstr(disas_instr, "face") != NULL){

				if (dr_get_thread_id(drcontext) == dr_get_process_id()) {
					countopcodes = true;
					printf("In ROI \n");
				}
				//dr_insert_clean_call(drcontext, bb, instr,
				//		(void *)enable_roi, false, 0
				//		);
				return DR_EMIT_DEFAULT;
			}

			if (strstr(disas_instr, "dead") != NULL && roi_enabled){
				if (dr_get_thread_id(drcontext) == dr_get_process_id()) {
					countopcodes = false;
					printf("Out of ROI \n");
				}
				//dr_insert_clean_call(drcontext, bb, instr,
				//		(void *)disable_roi, false, 0
				//		);
				return DR_EMIT_DEFAULT;
			}
		}
	}
	if (instr_fetch != NULL) {
		if (!(is_sve_instr(instr_fetch) && is_predicated_instr(instr_fetch)))
			return DR_EMIT_DEFAULT;
		instr_t *instr_operands = drmgr_orig_app_instr_for_operands(drcontext);

		if (instr_operands == NULL ||
				(!instr_reads_memory(instr_operands) && !instr_writes_memory(instr_operands) && !is_intop(instr_operands) && !is_flop(instr_operands) && !instr_is_app(instr_operands)))
			return DR_EMIT_DEFAULT;
		if (is_intop(instr_operands)) {
			if (is_int_fma(instr_operands))
				instrument_sve_predicated_intins(drcontext, bb, instr,instr_operands, 2);
			else
				instrument_sve_predicated_intins(drcontext, bb, instr,instr_operands, 1);

		}
		if (is_flop(instr_operands)) {
			if (is_fma(instr_operands))
				instrument_sve_predicated_fpins(drcontext, bb, instr,instr_operands, 2);
			else
				instrument_sve_predicated_fpins(drcontext, bb, instr,instr_operands, 1);
		}

		if (instr_reads_memory(instr_operands) || instr_writes_memory(instr_operands)) {
			/* Insert code to add an entry for each memory reference opnd. */
			for (i = 0; i < instr_num_srcs(instr_operands); i++) {
				const opnd_t src = instr_get_src(instr_operands, i);
				if (opnd_is_memory_reference(src)) {
					//if (opnd_is_base_disp(src) && (reg_is_z(opnd_get_base(src)) || reg_is_z(opnd_get_index(src))))
					//	continue;

					instrument_sve_predicated_mem(drcontext, bb, instr,instr_operands,instr_get_dst(instr_operands, 0));
				}
			}
			for (i = 0; i < instr_num_dsts(instr_operands); i++) {
				const opnd_t dst = instr_get_dst(instr_operands, i);
				if (opnd_is_memory_reference(dst) ) {
					//if (opnd_is_base_disp(dst) && (reg_is_z(opnd_get_base(dst)) || reg_is_z(opnd_get_index(dst))))
					//	continue;

					instrument_sve_predicated_mem(drcontext, bb, instr,instr_operands, instr_get_src(instr_operands, 0));
				}
			}
		}
		if (/* XXX i#1698: there are constraints for code between ldrex/strex pairs,
		     * so we minimize the instrumentation in between by skipping the clean call.
		     * As we're only inserting instrumentation on a memory reference, and the
		     * app should be avoiding memory accesses in between the ldrex...strex,
		     * the only problematic point should be before the strex.
		     * However, there is still a chance that the instrumentation code may clear the
		     * exclusive monitor state.
		     * Using a fault to handle a full buffer should be more robust, and the
		     * forthcoming buffer filling API (i#513) will provide that.
		     */
				IF_AARCHXX_ELSE(!instr_is_exclusive_store(instr_operands), true))
			dr_insert_clean_call(drcontext, bb, instr,
					(void *)sve_instr_call, false, 0
					);
		return DR_EMIT_DEFAULT;

	}
	
	return DR_EMIT_DEFAULT;
}

DR_EXPORT void
dr_client_main(client_id_t id, int argc, const char *argv[])
{
	drreg_options_t ops = { sizeof(ops), 3 /*max slots needed: aflags*/, false };
	dr_set_client_name("DynamoRIO Sample Client 'OI'",
			"http://dynamorio.org/issues");
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-roi") == 0) {
			roi_enabled = true;
			countopcodes = false;
			firstpass_countopcodes = false;
			fprintf(stderr, "Region of Interest Enabled\n");
			break;/* Optionally, you can break here if -roi doesn't take any additional arguments */
		}
	}

	if (!drmgr_init() || !drx_init() || drreg_init(&ops) != DRREG_SUCCESS || !drutil_init())
		DR_ASSERT(false);


	drmgr_register_bb_app2app_event(event_bb_app2app, NULL);
	drmgr_register_bb_instrumentation_event(event_bb_analysis,event_app_instruction,NULL);
	drmgr_register_thread_init_event(event_thread_init);
	drmgr_register_thread_exit_event(event_thread_exit);
	dr_register_exit_event(event_exit);

	/* Create thread-local storage index */
	tls_idx = drmgr_register_tls_field();
	tls_buf = drmgr_register_tls_field();

	if (!dr_raw_tls_calloc(&tls_seg, &tls_offs, MEMTRACE_TLS_COUNT, 0))
		DR_ASSERT(false);

	module_data_t *exe = dr_get_main_module();
	if (exe != NULL)
		exe_start = exe->start;
	dr_free_module_data(exe);
	mutex = dr_mutex_create();

	sprintf(filename,"output_%d.txt",dr_get_process_id());
	logfile = fopen(filename,"w");
	fprintf(logfile,"Flops,Bytes,FPINS,Intops,INTINS\n");
}


	static void
event_thread_init(void *drcontext)
{

	per_thread_t *data = dr_thread_alloc(drcontext, sizeof(per_thread_t));
	DR_ASSERT(data != NULL);
	drmgr_set_tls_field(drcontext, tls_buf, data);
	if (roi_enabled)
		data->countopcodes = false;
	else
		data->countopcodes = true;


	/* Keep seg_base in a per-thread data structure so we can get the TLS
	 * slot and find where the pointer points to in the buffer.
	 */
	data->counters = (thread_counters_t *)dr_thread_alloc(drcontext, sizeof(thread_counters_t));
	data->counters->fp_ops = 0;
	data->counters->fp_ins = 0;
	data->counters->int_ops = 0;
	data->counters->int_ins = 0;
	data->counters->bytes = 0;

	drmgr_set_tls_field(drcontext, tls_idx, data->counters);
	data->seg_base = dr_get_dr_segment_base(tls_seg);
	data->buf_base =
		dr_raw_mem_alloc(MEM_BUF_SIZE, DR_MEMPROT_READ | DR_MEMPROT_WRITE, NULL);
	DR_ASSERT(data->seg_base != NULL && data->buf_base != NULL);
	/* put buf_base to TLS as starting buf_ptr */
	BUF_PTR(data->seg_base) = data->buf_base;
}

	static void
event_thread_exit(void *drcontext)
{
	/* Process and free SVE instr buffers*/
	count_sve_predicated_instructions(drcontext); /* Dump last entries in buffer */
	per_thread_t *data;
	data = drmgr_get_tls_field(drcontext, tls_buf);

	/* Process and get global counter value */
	/*dr_printf("Thread %d: Floating point operations - %llu, floating point instructions - %llu, bytes - %llu, OI %f\n",
	  dr_get_thread_id(drcontext), data->counters->fp_ops, data->counters->fp_ins, data->counters->bytes, (double)((double)data->counters->fp_ops/data->counters->bytes));*/
	dr_mutex_lock(mutex);
	//printf("appname,%i,%lu,%lu,%lu,%lu,%lu\n",dr_get_thread_id(drcontext),data->counters->fp_ops,data->counters->bytes,data->counters->fp_ins,data->counters->int_ops,data->counters->int_ins);
	global_flops_count+=data->counters->fp_ops;
	global_fpins_count+=data->counters->fp_ins;
	global_intops_count+=data->counters->int_ops;
	global_intins_count+=data->counters->int_ins;
	global_bytes_count+=data->counters->bytes;
	dr_mutex_unlock(mutex);

	dr_thread_free(drcontext, data->counters, sizeof(thread_counters_t));
	dr_raw_mem_free(data->buf_base, MEM_BUF_SIZE);
	dr_thread_free(drcontext, data, sizeof(per_thread_t));
}

	static void
event_exit(void)
{
    if (!dr_raw_tls_cfree(tls_offs, MEMTRACE_TLS_COUNT))
        DR_ASSERT(false);
    fprintf(logfile,"%llu,%llu,%llu,%llu,%llu\n",global_flops_count,global_bytes_count,global_fpins_count,global_intops_count,global_intins_count);
    if (!drmgr_unregister_tls_field(tls_idx) ||
		    !drmgr_unregister_tls_field(tls_buf) ||
		    !drmgr_unregister_thread_init_event(event_thread_init) ||
		    !drmgr_unregister_thread_exit_event(event_thread_exit) ||
		    !drmgr_unregister_bb_insertion_event(event_app_instruction) ||
		    !drmgr_unregister_bb_app2app_event(event_bb_app2app))
	    DR_ASSERT(false);
    fclose(logfile);
    drutil_exit();
    drx_exit();
    drreg_exit();
    drmgr_exit();
}

