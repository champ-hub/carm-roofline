#define _GNU_SOURCE
#include "dr_api.h"
#include "dr_ir_instr.h"
#include "dr_ir_opnd.h"
#include "drmgr.h"
#include "drwrap.h"
#if !defined(X86)
#error "The CARM DynamoRIO instruction counter currently supports x86 only."
#endif

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define OP_COUNT 5
#define PREC_COUNT 2
#define WIDTH_COUNT 4
#define CATEGORY_COUNT (OP_COUNT * PREC_COUNT * WIDTH_COUNT)

enum { OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_FMA };
enum { PREC_F32, PREC_F64 };
enum { WIDTH_SCALAR, WIDTH_128, WIDTH_256, WIDTH_512 };
static const char *const op_names[OP_COUNT] = {"add", "sub", "mul", "div", "fma"};
static const char *const precision_names[PREC_COUNT] = {"f32", "f64"};
static const char *const width_names[WIDTH_COUNT] = {"scalar", "128", "256", "512"};

struct counts {
    uint64_t categories[CATEGORY_COUNT];
    uint64_t read_bytes, write_bytes;
};

struct region_frame {
    char *name;
    struct counts counters;
    struct region_frame *parent;
};

struct thread_state {
    uint64_t process_index, process_id, thread_index, thread_id;
    uint64_t instance;
    bool has_process, has_thread;
    unsigned int marker_depth;
    app_pc marker_return_pc;
    struct region_frame *top;
};

struct identity_entry {
    uint64_t instance, thread_index, thread_id;
    struct identity_entry *next;
};

struct result {
    uint64_t process_index, process_id, thread_index, thread_id;
    char *name;
    struct counts counters;
    struct result *next;
};

static int tls_index = -1;
static file_t output_file = INVALID_FILE;
static void *result_lock;
static void *identity_lock;
static struct result *results;
static struct identity_entry *thread_identities;
static uint64_t next_thread_instance;
static uint64_t process_index, process_id;
static bool process_identity_set;
static uint64_t open_regions;
static void marker_post(void *wrapcxt, void *user_data);
static void process_identity_pre(void *wrapcxt, void **user_data);
static void thread_identity_pre(void *wrapcxt, void **user_data);
static void begin_pre(void *wrapcxt, void **user_data);
static void end_pre(void *wrapcxt, void **user_data);
static void fail_client(const char *message)
{
    dr_fprintf(STDERR, "CARM DynamoRIO client error: %s\n", message);
    dr_abort_with_code(2);
    dr_abort();
}

static void append_format(char *line, size_t capacity, size_t *length, const char *format, ...)
{
    if (*length >= capacity)
        fail_client("JSON record exceeds the output buffer");
    va_list args;
    va_start(args, format);
    int written = vsnprintf(line + *length, capacity - *length, format, args);
    va_end(args);
    if (written < 0 || (size_t) written >= capacity - *length)
        fail_client("JSON record exceeds the output buffer");
    *length += (size_t) written;
}

static void append_json_string(char *line, size_t capacity, size_t *length, const char *s)
{
    append_format(line, capacity, length, "\"");
    for (const unsigned char *p = (const unsigned char *) s; *p != '\0'; ++p) {
        switch (*p) {
        case '"':
            append_format(line, capacity, length, "\\\"");
            break;
        case '\\':
            append_format(line, capacity, length, "\\\\");
            break;
        case '\n':
            append_format(line, capacity, length, "\\n");
            break;
        case '\r':
            append_format(line, capacity, length, "\\r");
            break;
        case '\t':
            append_format(line, capacity, length, "\\t");
            break;
        default:
            if (*p < 0x20)
                append_format(line, capacity, length, "\\u%04x", *p);
            else
                append_format(line, capacity, length, "%c", *p);
        }
    }
    append_format(line, capacity, length, "\"");
}

static void append_line(const char *line, size_t length)
{
    struct flock lock = {0};
    lock.l_type = F_WRLCK;
    lock.l_whence = SEEK_SET;
    while (fcntl(output_file, F_SETLKW, &lock) != 0) {
        if (errno == EINTR)
            continue;
        fail_client("cannot lock shared output file");
    }
    size_t written = 0;
    while (written < length) {
        ssize_t n = dr_write_file(output_file, line + written, length - written);
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0) {
            lock.l_type = F_UNLCK;
            fcntl(output_file, F_SETLK, &lock);
            fail_client("cannot append complete JSON record");
        }
        written += (size_t) n;
    }
    lock.l_type = F_UNLCK;
    if (fcntl(output_file, F_SETLK, &lock) != 0)
        fail_client("cannot unlock shared output file");
}

static struct thread_state *get_state(void *drcontext)
{
    return (struct thread_state *) drmgr_get_tls_field(drcontext, tls_index);
}

static char *copy_string(const char *s)
{
    size_t size = strlen(s) + 1;
    char *copy = dr_global_alloc(size);
    memcpy(copy, s, size);
    return copy;
}

static void free_string(char *s)
{
    if (s != NULL)
        dr_global_free(s, strlen(s) + 1);
}

static char *safe_string(const char *app_string)
{
    char buf[4096];
    size_t used = 0;
    if (app_string == NULL)
        return NULL;
    while (used + 1 < sizeof(buf)) {
        char ch;
        size_t got = 0;
        if (!dr_safe_read(app_string + used, 1, &ch, &got) || got != 1)
            return NULL;
        if (ch == '\0')
            break;
        buf[used++] = ch;
    }
    if (used + 1 == sizeof(buf))
        return NULL;
    buf[used] = '\0';
    return copy_string(buf);
}

static void merge_result(struct thread_state *state, struct region_frame *frame)
{
    dr_mutex_lock(result_lock);
    struct result *r;
    for (r = results; r != NULL; r = r->next) {
        if (r->process_id == state->process_id && r->thread_id == state->thread_id && strcmp(r->name, frame->name) == 0)
            break;
    }
    if (r == NULL) {
        r = dr_global_alloc(sizeof(*r));
        memset(r, 0, sizeof(*r));
        r->process_index = state->process_index;
        r->process_id = state->process_id;
        r->thread_index = state->thread_index;
        r->thread_id = state->thread_id;
        r->name = copy_string(frame->name);
        r->next = results;
        results = r;
    }
    for (int i = 0; i < CATEGORY_COUNT; ++i)
        r->counters.categories[i] += frame->counters.categories[i];
    r->counters.read_bytes += frame->counters.read_bytes;
    r->counters.write_bytes += frame->counters.write_bytes;
    dr_mutex_unlock(result_lock);
}

static void write_results(void)
{
    if (output_file == INVALID_FILE)
        return;
    dr_mutex_lock(result_lock);
    for (struct result *r = results; r != NULL; r = r->next) {
        char line[32768];
        size_t length = 0;
        append_format(
                line,
                sizeof(line),
                &length,
                "{\"type\":\"region\",\"process_index\":%llu,\"thread_index\":%llu,\"process_id\":%llu,\"thread_id\":%"
                "llu,\"thread_id_stable\":true,\"name\":",
                (unsigned long long) r->process_index,
                (unsigned long long) r->thread_index,
                (unsigned long long) r->process_id,
                (unsigned long long) r->thread_id);
        append_json_string(line, sizeof(line), &length, r->name);
        append_format(line, sizeof(line), &length, ",\"instruction_counts\":{");
        bool first = true;
        for (int op = 0; op < OP_COUNT; ++op) {
            for (int prec = 0; prec < PREC_COUNT; ++prec) {
                for (int width = 0; width < WIDTH_COUNT; ++width) {
                    int i = (op * PREC_COUNT + prec) * WIDTH_COUNT + width;
                    if (r->counters.categories[i] == 0)
                        continue;
                    append_format(
                            line,
                            sizeof(line),
                            &length,
                            "%s\"%s_%s_%s\":%llu",
                            first ? "" : ",",
                            op_names[op],
                            width_names[width],
                            precision_names[prec],
                            (unsigned long long) r->counters.categories[i]);
                    first = false;
                }
            }
        }
        append_format(
                line,
                sizeof(line),
                &length,
                "},\"read_bytes\":%llu,\"write_bytes\":%llu,\"elapsed_ns\":null,\"masked_lanes\":\"full_encoded_width_"
                "estimate\"}\n",
                (unsigned long long) r->counters.read_bytes,
                (unsigned long long) r->counters.write_bytes);
        append_line(line, length);
    }
    dr_mutex_unlock(result_lock);
}

static void thread_init(void *drcontext)
{
    struct thread_state *state = dr_thread_alloc(drcontext, sizeof(*state));
    memset(state, 0, sizeof(*state));
    dr_mutex_lock(identity_lock);
    state->instance = ++next_thread_instance;
    dr_mutex_unlock(identity_lock);
    drmgr_set_tls_field(drcontext, tls_index, state);
}

static void thread_exit(void *drcontext)
{
    struct thread_state *state = get_state(drcontext);
    if (state != NULL && (state->top != NULL || state->marker_depth != 0))
        fail_client("thread exited with an unfinished region or marker call");
    if (state != NULL)
        dr_thread_free(drcontext, state, sizeof(*state));
}

static void count_instruction(int category, int read_bytes, int write_bytes, int repeat, app_pc instruction_pc)
{
    void *drcontext = dr_get_current_drcontext();
    struct thread_state *state = get_state(drcontext);
    if (state == NULL || state->top == NULL || (state->marker_depth != 0 && instruction_pc != state->marker_return_pc))
        return;
    uint64_t iterations = 1;
    if (repeat) {
        dr_mcontext_t mc = {sizeof(mc), DR_MC_INTEGER};
        if (!dr_get_mcontext(drcontext, &mc))
            return;
        iterations = (uint64_t) reg_get_value(DR_REG_XCX, &mc);
    }
    if (category >= 0 && category < CATEGORY_COUNT)
        ++state->top->counters.categories[category];
    state->top->counters.read_bytes += (uint64_t) read_bytes * iterations;
    state->top->counters.write_bytes += (uint64_t) write_bytes * iterations;
}

/* DynamoRIO's current opcode catalog uses these stems only for FMA3/FMA4 forms. */
static bool is_fma_base(const char *base)
{
    static const char *const stems[] = {"fmadd", "fmsub", "fnmadd", "fnmsub"};
    for (size_t i = 0; i < sizeof(stems) / sizeof(stems[0]); ++i) {
        if (strncmp(base, stems[i], strlen(stems[i])) == 0)
            return true;
    }
    return false;
}

static int classify_fp(instr_t *instr)
{
    const char *decoded = decode_opcode_name(instr_get_opcode(instr));
    if (decoded == NULL)
        return -1;
    const char *name = decoded[0] == 'v' ? decoded + 1 : decoded;
    size_t len = strlen(name);
    if (len < 3)
        return -1;
    const char *suffix = name + len - 2;
    int prec, width, op;
    bool scalar;
    if (strcmp(suffix, "ss") == 0) {
        prec = PREC_F32;
        scalar = true;
    } else if (strcmp(suffix, "sd") == 0) {
        prec = PREC_F64;
        scalar = true;
    } else if (strcmp(suffix, "ps") == 0) {
        prec = PREC_F32;
        scalar = false;
    } else if (strcmp(suffix, "pd") == 0) {
        prec = PREC_F64;
        scalar = false;
    } else {
        return -1;
    }

    char base[32];
    size_t base_len = len - 2;
    if (base_len >= sizeof(base))
        return -1;
    memcpy(base, name, base_len);
    base[base_len] = '\0';
    if (is_fma_base(base)) {
        op = OP_FMA;
    } else if (strcmp(base, "add") == 0) {
        op = OP_ADD;
    } else if (strcmp(base, "sub") == 0) {
        op = OP_SUB;
    } else if (strcmp(base, "mul") == 0) {
        op = OP_MUL;
    } else if (strcmp(base, "div") == 0) {
        op = OP_DIV;
    } else {
        return -1;
    }

    if (scalar) {
        width = WIDTH_SCALAR;
    } else {
        uint max_bytes = 0;
        int ndst = instr_num_dsts(instr), nsrc = instr_num_srcs(instr);
        for (int i = 0; i < ndst; ++i) {
            opnd_t o = instr_get_dst(instr, i);
            if (opnd_is_reg(o) && opnd_size_in_bytes(opnd_get_size(o)) > max_bytes)
                max_bytes = opnd_size_in_bytes(opnd_get_size(o));
        }
        for (int i = 0; i < nsrc; ++i) {
            opnd_t o = instr_get_src(instr, i);
            if (opnd_is_reg(o) && opnd_size_in_bytes(opnd_get_size(o)) > max_bytes)
                max_bytes = opnd_size_in_bytes(opnd_get_size(o));
        }
        if (max_bytes == 64)
            width = WIDTH_512;
        else if (max_bytes == 32)
            width = WIDTH_256;
        else if (max_bytes == 16)
            width = WIDTH_128;
        else
            return -1;
    }
    return (op * PREC_COUNT + prec) * WIDTH_COUNT + width;
}

static bool is_marker_call(instr_t *instr)
{
    if (!instr_is_call_direct(instr))
        return false;
    app_pc target = instr_get_branch_target_pc(instr);
    return drwrap_is_wrapped(target, begin_pre, marker_post) || drwrap_is_wrapped(target, end_pre, marker_post) ||
           drwrap_is_wrapped(target, process_identity_pre, marker_post) ||
           drwrap_is_wrapped(target, thread_identity_pre, marker_post);
}

static dr_emit_flags_t instrument_instruction(
        void *drcontext, void *tag, instrlist_t *bb, instr_t *instr, bool for_trace, bool translating, void *user_data)
{
    if (!instr_is_app(instr))
        return DR_EMIT_DEFAULT;
    if (is_marker_call(instr))
        return DR_EMIT_DEFAULT;
    int read_bytes = 0, write_bytes = 0;
    bool reads_memory = instr_reads_memory(instr);
    bool writes_memory = instr_writes_memory(instr);
    if (reads_memory) {
        for (int i = 0; i < instr_num_srcs(instr); ++i) {
            opnd_t opnd = instr_get_src(instr, i);
            if (opnd_is_memory_reference(opnd)) {
                uint size = opnd_size_in_bytes(opnd_get_size(opnd));
                if (size > 0 && size <= INT32_MAX)
                    read_bytes += (int) size;
            }
        }
    }
    if (writes_memory) {
        for (int i = 0; i < instr_num_dsts(instr); ++i) {
            opnd_t opnd = instr_get_dst(instr, i);
            if (opnd_is_memory_reference(opnd)) {
                uint size = opnd_size_in_bytes(opnd_get_size(opnd));
                if (size > 0 && size <= INT32_MAX)
                    write_bytes += (int) size;
            }
        }
    }
    int category = classify_fp(instr);
    bool repeat = instr_is_rep_string_op(instr);
    if (category < 0 && read_bytes == 0 && write_bytes == 0)
        return DR_EMIT_DEFAULT;
    dr_insert_clean_call(
            drcontext,
            bb,
            instr,
            (void *) count_instruction,
            false,
            5,
            OPND_CREATE_INT32(category),
            OPND_CREATE_INT32(read_bytes),
            OPND_CREATE_INT32(write_bytes),
            OPND_CREATE_INT32(repeat),
            OPND_CREATE_INTPTR(instr_get_app_pc(instr)));
    return DR_EMIT_DEFAULT;
}

/* The first caller instruction may be instrumented before drwrap's post callback runs. */
static void marker_pre(void *wrapcxt, void **user_data, struct thread_state *state)
{
    *user_data = state->marker_return_pc;
    state->marker_return_pc = drwrap_get_retaddr(wrapcxt);
    ++state->marker_depth;
}

static void set_identity_pre(void *wrapcxt, void **user_data, bool process)
{
    void *drcontext = drwrap_get_drcontext(wrapcxt);
    struct thread_state *state = get_state(drcontext);
    marker_pre(wrapcxt, user_data, state);
    uint64_t index = (uint64_t) (ptr_uint_t) drwrap_get_arg(wrapcxt, 0);
    uint64_t id = (uint64_t) (ptr_uint_t) drwrap_get_arg(wrapcxt, 1);
    dr_mutex_lock(identity_lock);
    if (process) {
        if ((process_identity_set && (process_index != index || process_id != id)) ||
            (state->has_process && (state->process_index != index || state->process_id != id)))
            fail_client("process identity changed or conflicts within one process");
        process_index = index;
        process_id = id;
        process_identity_set = true;
        state->process_index = index;
        state->process_id = id;
        state->has_process = true;
    } else {
        struct identity_entry *entry;
        for (entry = thread_identities; entry != NULL; entry = entry->next) {
            if (entry->instance == state->instance) {
                if (entry->thread_index != index || entry->thread_id != id)
                    fail_client("thread identity changed after registration");
                break;
            }
            if (entry->thread_index == index || entry->thread_id == id)
                fail_client("duplicate thread index or ID in one process");
        }
        if (entry == NULL) {
            entry = dr_global_alloc(sizeof(*entry));
            entry->instance = state->instance;
            entry->thread_index = index;
            entry->thread_id = id;
            entry->next = thread_identities;
            thread_identities = entry;
        }
        if (state->has_thread && (state->thread_index != index || state->thread_id != id))
            fail_client("thread identity changed after registration");
        state->thread_index = index;
        state->thread_id = id;
        state->has_thread = true;
    }
    dr_mutex_unlock(identity_lock);
}

static void marker_post(void *wrapcxt, void *user_data)
{
    struct thread_state *state = get_state(drwrap_get_drcontext(wrapcxt));
    if (state != NULL && state->marker_depth > 0) {
        --state->marker_depth;
        state->marker_return_pc = (app_pc) user_data;
    }
}

static void process_identity_pre(void *c, void **u)
{
    set_identity_pre(c, u, true);
}

static void thread_identity_pre(void *c, void **u)
{
    set_identity_pre(c, u, false);
}

static void begin_pre(void *wrapcxt, void **user_data)
{
    void *drcontext = drwrap_get_drcontext(wrapcxt);
    struct thread_state *state = get_state(drcontext);
    marker_pre(wrapcxt, user_data, state);
    if (!state->has_process || !state->has_thread)
        fail_client("region begin before process/thread identity");
    char *name = safe_string((const char *) drwrap_get_arg(wrapcxt, 0));
    if (name == NULL || name[0] == '\0') {
        free_string(name);
        fail_client("invalid or unreadable region name");
    }
    struct region_frame *frame = dr_thread_alloc(drcontext, sizeof(*frame));
    if (frame == NULL)
        fail_client("cannot allocate region frame");
    memset(frame, 0, sizeof(*frame));
    frame->name = name;
    frame->parent = state->top;
    state->top = frame;
    dr_mutex_lock(result_lock);
    ++open_regions;
    dr_mutex_unlock(result_lock);
}

static void end_pre(void *wrapcxt, void **user_data)
{
    void *drcontext = drwrap_get_drcontext(wrapcxt);
    struct thread_state *state = get_state(drcontext);
    marker_pre(wrapcxt, user_data, state);
    char *name = safe_string((const char *) drwrap_get_arg(wrapcxt, 0));
    if (state->top == NULL || name == NULL || strcmp(state->top->name, name) != 0) {
        free_string(name);
        fail_client("region end does not match the innermost open region");
    }
    struct region_frame *frame = state->top;
    state->top = frame->parent;
    merge_result(state, frame);
    free_string(frame->name);
    dr_thread_free(drcontext, frame, sizeof(*frame));
    free_string(name);
    dr_mutex_lock(result_lock);
    if (open_regions == 0)
        fail_client("internal region accounting underflow");
    --open_regions;
    dr_mutex_unlock(result_lock);
}

static void module_load(void *drcontext, const module_data_t *module, bool loaded)
{
    generic_func_t f = dr_get_proc_address(module->handle, "carm_region_begin");
    if (f != NULL)
        drwrap_wrap((app_pc) f, begin_pre, marker_post);
    f = dr_get_proc_address(module->handle, "carm_region_end");
    if (f != NULL)
        drwrap_wrap((app_pc) f, end_pre, marker_post);
    f = dr_get_proc_address(module->handle, "carm_region_set_process_identity");
    if (f != NULL)
        drwrap_wrap((app_pc) f, process_identity_pre, marker_post);
    f = dr_get_proc_address(module->handle, "carm_region_set_thread_identity");
    if (f != NULL)
        drwrap_wrap((app_pc) f, thread_identity_pre, marker_post);
}

static void exit_event(void)
{
    dr_mutex_lock(result_lock);
    if (open_regions != 0) {
        dr_mutex_unlock(result_lock);
        fail_client("process exited with unfinished region markers");
    }
    dr_mutex_unlock(result_lock);
    write_results();
    for (struct result *r = results; r != NULL;) {
        struct result *next = r->next;
        free_string(r->name);
        dr_global_free(r, sizeof(*r));
        r = next;
    }
    while (thread_identities != NULL) {
        struct identity_entry *next = thread_identities->next;
        dr_global_free(thread_identities, sizeof(*thread_identities));
        thread_identities = next;
    }
    if (output_file != INVALID_FILE)
        dr_close_file(output_file);
    dr_mutex_destroy(result_lock);
    dr_mutex_destroy(identity_lock);
    drwrap_exit();
    drmgr_unregister_tls_field(tls_index);
    drmgr_exit();
}

DR_EXPORT void dr_client_main(client_id_t id, int argc, const char *argv[])
{
    const char *output_path = NULL;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc)
            output_path = argv[++i];
        else
            fail_client("usage: -c <client> -o <output.jsonl> -- <app>");
    }
    if (output_path == NULL)
        fail_client("DynamoRIO client requires -o <output.jsonl>");
    output_file = dr_open_file(output_path, DR_FILE_WRITE_APPEND);
    if (output_file == INVALID_FILE)
        fail_client("cannot open client output for append");
    if (!drmgr_init() || !drwrap_init())
        fail_client("cannot initialize DynamoRIO extensions");
    tls_index = drmgr_register_tls_field();
    if (tls_index < 0)
        fail_client("cannot allocate DynamoRIO TLS field");
    result_lock = dr_mutex_create();
    identity_lock = dr_mutex_create();
    if (result_lock == NULL || identity_lock == NULL)
        fail_client("cannot create client locks");
    if (!drmgr_register_thread_init_event(thread_init) || !drmgr_register_thread_exit_event(thread_exit) ||
        !drmgr_register_module_load_event(module_load) ||
        !drmgr_register_bb_instrumentation_event(NULL, instrument_instruction, NULL))
        fail_client("cannot register DynamoRIO instrumentation events");
    dr_register_exit_event(exit_event);
}
