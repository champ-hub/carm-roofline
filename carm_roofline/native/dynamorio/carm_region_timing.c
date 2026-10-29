#define _POSIX_C_SOURCE 200809L
#include "carm_region.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <time.h>
#include <unistd.h>

struct region_frame {
    char *name;
    uint64_t elapsed_ns;
    uint64_t started_ns;
    struct region_frame *parent;
};
struct thread_state {
    uint64_t process_index, process_id, thread_index, thread_id, instance;
    int has_process_identity, has_thread_identity;
    struct region_frame *top;
};
struct identity_entry {
    uint64_t instance, thread_index, thread_id;
    struct identity_entry *next;
};
struct timing_record {
    uint64_t process_index, process_id, thread_index, thread_id, elapsed_ns;
    char *name;
    struct timing_record *next;
};

static pthread_key_t state_key;
static pthread_once_t key_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t records_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t identity_lock = PTHREAD_MUTEX_INITIALIZER;
static struct timing_record *records;
static struct identity_entry *thread_identities;
static const char *output_path;
static uint64_t next_thread_instance;
static uint64_t process_index, process_id, open_regions;
static int process_identity_set, timing_enabled;

static void fail(const char *message)
{
    fprintf(stderr, "CARM region marker error: %s\n", message);
    abort();
}

static uint64_t now_ns(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        fail("clock_gettime(CLOCK_MONOTONIC) failed");
    return (uint64_t) ts.tv_sec * UINT64_C(1000000000) + (uint64_t) ts.tv_nsec;
}

static void free_thread(void *ptr)
{
    struct thread_state *state = ptr;
    if (state == NULL)
        return;
    if (state->top != NULL)
        fail("thread exited with an unfinished region");
    free(state);
}

static void make_key(void)
{
    if (pthread_key_create(&state_key, free_thread) != 0)
        fail("pthread_key_create failed");
}

static struct thread_state *get_state(void)
{
    if (pthread_once(&key_once, make_key) != 0)
        fail("pthread_once failed");
    struct thread_state *state = pthread_getspecific(state_key);
    if (state == NULL) {
        state = calloc(1, sizeof(*state));
        if (state == NULL)
            fail("cannot allocate thread state");
        pthread_mutex_lock(&identity_lock);
        state->instance = ++next_thread_instance;
        pthread_mutex_unlock(&identity_lock);
        if (pthread_setspecific(state_key, state) != 0)
            fail("pthread_setspecific failed");
    }
    return state;
}

static void require_ids(const struct thread_state *state)
{
    if (!state->has_process_identity || !state->has_thread_identity)
        fail("process and thread identities must be set before region markers");
}

static void add_record(const struct thread_state *state, const char *name, uint64_t ns)
{
    pthread_mutex_lock(&records_lock);
    struct timing_record *record;
    for (record = records; record != NULL; record = record->next) {
        if (record->process_id == state->process_id && record->thread_id == state->thread_id &&
            strcmp(record->name, name) == 0) {
            if (record->process_index != state->process_index || record->thread_index != state->thread_index) {
                pthread_mutex_unlock(&records_lock);
                fail("one stable ID maps to multiple index values");
            }
            record->elapsed_ns += ns;
            pthread_mutex_unlock(&records_lock);
            return;
        }
    }
    record = calloc(1, sizeof(*record));
    if (record == NULL || (record->name = strdup(name)) == NULL)
        fail("cannot allocate timing record");
    record->process_index = state->process_index;
    record->process_id = state->process_id;
    record->thread_index = state->thread_index;
    record->thread_id = state->thread_id;
    record->elapsed_ns = ns;
    record->next = records;
    records = record;
    pthread_mutex_unlock(&records_lock);
}

static void set_identity(uint64_t index, uint64_t id, int process)
{
    struct thread_state *state = get_state();
    pthread_mutex_lock(&identity_lock);
    if (process) {
        if ((process_identity_set && (process_index != index || process_id != id)) ||
            (state->has_process_identity && (state->process_index != index || state->process_id != id))) {
            pthread_mutex_unlock(&identity_lock);
            fail("process identity changed or conflicts within one process");
        }
        process_index = index;
        process_id = id;
        process_identity_set = 1;
        state->process_index = index;
        state->process_id = id;
        state->has_process_identity = 1;
    } else {
        struct identity_entry *entry;
        for (entry = thread_identities; entry != NULL; entry = entry->next) {
            if (entry->instance == state->instance) {
                if (entry->thread_index != index || entry->thread_id != id) {
                    pthread_mutex_unlock(&identity_lock);
                    fail("thread identity changed after registration");
                }
                break;
            }
            if (entry->thread_index == index || entry->thread_id == id) {
                pthread_mutex_unlock(&identity_lock);
                fail("duplicate thread index or ID in one process");
            }
        }
        if (entry == NULL) {
            entry = calloc(1, sizeof(*entry));
            if (entry == NULL) {
                pthread_mutex_unlock(&identity_lock);
                fail("cannot allocate thread identity record");
            }
            entry->instance = state->instance;
            entry->thread_index = index;
            entry->thread_id = id;
            entry->next = thread_identities;
            thread_identities = entry;
        }
        if (state->has_thread_identity && (state->thread_index != index || state->thread_id != id)) {
            pthread_mutex_unlock(&identity_lock);
            fail("thread identity changed after registration");
        }
        state->thread_index = index;
        state->thread_id = id;
        state->has_thread_identity = 1;
    }
    pthread_mutex_unlock(&identity_lock);
}

void carm_region_set_process_identity(uint64_t index, uint64_t id)
{
    set_identity(index, id, 1);
}

void carm_region_set_thread_identity(uint64_t index, uint64_t id)
{
    set_identity(index, id, 0);
}

static void append_format(char *line, size_t capacity, size_t *length, const char *format, ...)
{
    if (*length >= capacity)
        fail("timing JSON record exceeds output buffer");
    va_list args;
    va_start(args, format);
    int written = vsnprintf(line + *length, capacity - *length, format, args);
    va_end(args);
    if (written < 0 || (size_t) written >= capacity - *length)
        fail("timing JSON record exceeds output buffer");
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

static void append_record(int fd, const struct timing_record *record)
{
    char line[32768];
    size_t length = 0;
    append_format(
            line,
            sizeof(line),
            &length,
            "{\"type\":\"region\",\"process_index\":%llu,\"thread_index\":%llu,\"process_id\":%llu,\"thread_id\":%llu,"
            "\"thread_id_stable\":true,\"name\":",
            (unsigned long long) record->process_index,
            (unsigned long long) record->thread_index,
            (unsigned long long) record->process_id,
            (unsigned long long) record->thread_id);
    append_json_string(line, sizeof(line), &length, record->name);
    append_format(line, sizeof(line), &length, ",\"elapsed_ns\":%llu}\n", (unsigned long long) record->elapsed_ns);

    if (flock(fd, LOCK_EX) != 0)
        fail("cannot lock native timing output");
    size_t written = 0;
    while (written < length) {
        ssize_t n = write(fd, line + written, length - written);
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0)
            fail("cannot append complete native timing record");
        written += (size_t) n;
    }
    if (flock(fd, LOCK_UN) != 0)
        fail("cannot unlock native timing output");
}

__attribute__((constructor)) static void initialize(void)
{
    output_path = getenv("CARM_REGION_TIMING_OUTPUT");
    timing_enabled = output_path != NULL && output_path[0] != '\0';
}

__attribute__((destructor)) static void write_records(void)
{
    if (!timing_enabled)
        return;
    pthread_mutex_lock(&records_lock);
    pthread_mutex_lock(&identity_lock);
    if (open_regions != 0) {
        pthread_mutex_unlock(&identity_lock);
        pthread_mutex_unlock(&records_lock);
        fail("process exited with unfinished regions");
    }
    pthread_mutex_unlock(&identity_lock);
    int fd = open(output_path, O_CREAT | O_WRONLY | O_APPEND | O_CLOEXEC, 0666);
    if (fd < 0) {
        pthread_mutex_unlock(&records_lock);
        fail("cannot open CARM_REGION_TIMING_OUTPUT for append");
    }
    for (struct timing_record *record = records; record != NULL; record = record->next)
        append_record(fd, record);
    int close_error = close(fd);
    pthread_mutex_unlock(&records_lock);
    if (close_error != 0)
        fail("cannot close native timing output");
}

void carm_region_begin(const char *name)
{
    if (name == NULL || name[0] == '\0')
        fail("region name must be nonempty");
    struct thread_state *state = get_state();
    require_ids(state);
    if (!timing_enabled)
        return;

    uint64_t now = now_ns();
    if (state->top != NULL)
        state->top->elapsed_ns += now - state->top->started_ns;
    struct region_frame *frame = calloc(1, sizeof(*frame));
    if (frame == NULL || (frame->name = strdup(name)) == NULL)
        fail("cannot allocate region frame");
    frame->started_ns = now;
    frame->parent = state->top;
    state->top = frame;
    pthread_mutex_lock(&identity_lock);
    ++open_regions;
    pthread_mutex_unlock(&identity_lock);
}

void carm_region_end(const char *name)
{
    struct thread_state *state = get_state();
    require_ids(state);
    if (!timing_enabled)
        return;
    if (state->top == NULL || name == NULL || strcmp(state->top->name, name) != 0)
        fail("region end must match the innermost open region on this thread");

    uint64_t now = now_ns();
    struct region_frame *frame = state->top;
    frame->elapsed_ns += now - frame->started_ns;
    add_record(state, frame->name, frame->elapsed_ns);
    state->top = frame->parent;
    free(frame->name);
    free(frame);
    pthread_mutex_lock(&identity_lock);
    if (open_regions == 0) {
        pthread_mutex_unlock(&identity_lock);
        fail("internal open-region accounting underflow");
    }
    --open_regions;
    pthread_mutex_unlock(&identity_lock);
    if (state->top != NULL)
        state->top->started_ns = now;
}
