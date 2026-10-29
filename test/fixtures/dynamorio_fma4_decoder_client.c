#define dr_client_main carm_dynamorio_client_main
#include "../../carm_roofline/native/dynamorio/carm_dynamorio_client.c"
#undef dr_client_main

struct fma4_opcode_case {
    byte encoding[6];
    const char *name;
    int precision;
    int width;
};

static const struct fma4_opcode_case fma4_opcodes[] = {
        {{0xc4, 0xe3, 0xe5, 0x68, 0xe1, 0x20}, "vfmaddps", PREC_F32, WIDTH_256},
        {{0xc4, 0xe3, 0xe1, 0x69, 0xe1, 0x20}, "vfmaddpd", PREC_F64, WIDTH_128},
        {{0xc4, 0xe3, 0xe1, 0x6a, 0xe1, 0x20}, "vfmaddss", PREC_F32, WIDTH_SCALAR},
        {{0xc4, 0xe3, 0xe1, 0x6b, 0xe1, 0x20}, "vfmaddsd", PREC_F64, WIDTH_SCALAR},
        {{0xc4, 0xe3, 0xe5, 0x6c, 0xe1, 0x20}, "vfmsubps", PREC_F32, WIDTH_256},
        {{0xc4, 0xe3, 0xe1, 0x6d, 0xe1, 0x20}, "vfmsubpd", PREC_F64, WIDTH_128},
        {{0xc4, 0xe3, 0xe1, 0x6e, 0xe1, 0x20}, "vfmsubss", PREC_F32, WIDTH_SCALAR},
        {{0xc4, 0xe3, 0xe1, 0x6f, 0xe1, 0x20}, "vfmsubsd", PREC_F64, WIDTH_SCALAR},
        {{0xc4, 0xe3, 0xe5, 0x78, 0xe1, 0x20}, "vfnmaddps", PREC_F32, WIDTH_256},
        {{0xc4, 0xe3, 0xe1, 0x79, 0xe1, 0x20}, "vfnmaddpd", PREC_F64, WIDTH_128},
        {{0xc4, 0xe3, 0xe1, 0x7a, 0xe1, 0x20}, "vfnmaddss", PREC_F32, WIDTH_SCALAR},
        {{0xc4, 0xe3, 0xe1, 0x7b, 0xe1, 0x20}, "vfnmaddsd", PREC_F64, WIDTH_SCALAR},
        {{0xc4, 0xe3, 0xe5, 0x7c, 0xe1, 0x20}, "vfnmsubps", PREC_F32, WIDTH_256},
        {{0xc4, 0xe3, 0xe1, 0x7d, 0xe1, 0x20}, "vfnmsubpd", PREC_F64, WIDTH_128},
        {{0xc4, 0xe3, 0xe1, 0x7e, 0xe1, 0x20}, "vfnmsubss", PREC_F32, WIDTH_SCALAR},
        {{0xc4, 0xe3, 0xe1, 0x7f, 0xe1, 0x20}, "vfnmsubsd", PREC_F64, WIDTH_SCALAR},
        {{0xc4, 0xe3, 0xe5, 0x5c, 0xe1, 0x20}, "vfmaddsubps", PREC_F32, WIDTH_256},
        {{0xc4, 0xe3, 0xe1, 0x5d, 0xe1, 0x20}, "vfmaddsubpd", PREC_F64, WIDTH_128},
        {{0xc4, 0xe3, 0xe5, 0x5e, 0xe1, 0x20}, "vfmsubaddps", PREC_F32, WIDTH_256},
        {{0xc4, 0xe3, 0xe1, 0x5f, 0xe1, 0x20}, "vfmsubaddpd", PREC_F64, WIDTH_128},
};

static bool fma4_decoder_tested;

static dr_emit_flags_t test_fma4_decoder(
        void *drcontext, void *tag, instrlist_t *bb, instr_t *instr, bool for_trace, bool translating, void *user_data)
{
    if (fma4_decoder_tested || !instr_is_app(instr))
        return DR_EMIT_DEFAULT;
    fma4_decoder_tested = true;
    for (size_t i = 0; i < sizeof(fma4_opcodes) / sizeof(fma4_opcodes[0]); ++i) {
        const struct fma4_opcode_case *test = &fma4_opcodes[i];
        instr_t *decoded = instr_create(drcontext);
        if (decoded == NULL || decode_from_copy(drcontext, test->encoding, instr_get_app_pc(instr), decoded) == NULL)
            fail_client("DynamoRIO failed to decode an FMA4 test opcode");
        const char *name = decode_opcode_name(instr_get_opcode(decoded));
        int expected = (OP_FMA * PREC_COUNT + test->precision) * WIDTH_COUNT + test->width;
        if (name == NULL || strcmp(name, test->name) != 0 || classify_fp(decoded) != expected)
            fail_client("DynamoRIO FMA4 opcode name, precision, or width classification mismatch");
        instr_destroy(drcontext, decoded);
    }
    return DR_EMIT_DEFAULT;
}

DR_EXPORT void dr_client_main(client_id_t id, int argc, const char *argv[])
{
    carm_dynamorio_client_main(id, argc, argv);
    if (!drmgr_register_bb_instrumentation_event(NULL, test_fma4_decoder, NULL))
        fail_client("cannot register FMA4 decoder test callback");
}
