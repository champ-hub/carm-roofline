# Legacy Benchmark Generator Documentation (`legacy_bench_gen/`)

> **For AI Agents**: This file documents the **deprecated** C-based benchmark generator. All new development uses `benchmark/generation/` (Python). Read this to understand the migration baseline, the test comparison infrastructure in `refactor_tests/`, or when investigating behavior differences between old and new output.

---

## Table of Contents

1. [Overview](#overview)
2. [File Map](#file-map)
3. [Build System](#build-system)
4. [Execution Flow](#execution-flow)
5. [config_test.h — ISA Configuration Macros](#config_testh--isa-configuration-macros)
6. [select_op.c — Operation and Register Selection](#select_opc--operation-and-register-selection)
7. [calc_param.c — Loop Parameter Calculation](#calc_paramc--loop-parameter-calculation)
8. [create_bench.c — Benchmark Orchestration](#create_benchc--benchmark-orchestration)
9. [write_asm.c — Assembly Code Generation](#write_asmc--assembly-code-generation)
10. [Generated Output: Test/test_params.h](#generated-output-testtest_paramsh)
11. [Two-Phase Build: Test/ Directory](#two-phase-build-test-directory)
12. [Bench/ Directory — Identical Active Copy](#bench-directory--identical-active-copy)
13. [New System: benchmark/generation/](#new-system-benchmarkgeneration)
14. [Key Differences vs New Python System](#key-differences-vs-new-python-system)
15. [Migration Notes](#migration-notes)
16. [Test Infrastructure: refactor_tests/](#test-infrastructure-refactor_tests)

---

## Overview

The legacy benchmark generator is a **C program** that generates inline assembly microbenchmarks at runtime by writing C source (containing `__asm__ __volatile__` blocks) to `Test/test_params.h`, then invoking `make` to compile the final benchmark binary.

**Key characteristics:**
- ISA is baked in at **compile time** of the generator itself via preprocessor defines (e.g., `-DAVX2`)
- One binary per ISA was compiled: `bench_avx2`, `bench_sse`, `bench_neon`, etc.
- The generator binary, when run, writes assembly to disk and shells out to `make`
- Supports three test modes: `FLOPS` (arithmetic), `MEM` (memory bandwidth), `MIXED` (combined)
- Supports operations: `add`, `mul`, `div`, `fma`, `mad` (mul+add interleaved)
- Supports precisions: `dp` (double), `sp` (single float)

---

## File Map

```
legacy_bench_gen/
├── bench.c          # Main entry point — CLI parsing, dispatches to create_bench.c
├── config_test.h    # ISA-specific macros: register names, instruction mnemonics, alignment, flop counts
├── functions.h      # Function declarations for all modules
├── select_op.c      # Maps (operation, precision) → instruction mnemonic + register prefix
├── calc_param.c     # Calculates loop iteration counts (inner/outer split)
├── create_bench.c   # Orchestrates: calls select_op, write_asm, then shells out to make
├── write_asm.c      # Generates the inline assembly C code (largest file, 1275 lines)
└── Makefile         # Builds one binary per ISA (bench_avx2, bench_neon, etc.)
```

---

## Build System

**File**: `legacy_bench_gen/Makefile`

Builds 11 separate ISA-specific binaries from the same source using preprocessor flag injection:

```makefile
ISAS = scalar sse avx avx2 avx512 armscalar neon sve riscvscalar rvv0.7 rvv1.0

ISA_FLAGS.scalar      = -DSCALAR
ISA_FLAGS.sse         = -DSSE
ISA_FLAGS.avx         = -DAVX
ISA_FLAGS.avx2        = -DAVX2
ISA_FLAGS.avx512      = -DAVX512
ISA_FLAGS.armscalar   = -DASCALAR
ISA_FLAGS.neon        = -DNEON
ISA_FLAGS.sve         = -DSVE
ISA_FLAGS.riscvscalar = -DRISCVSCALAR
ISA_FLAGS.rvv0.7      = -DRVV07
ISA_FLAGS.rvv1.0      = -DRVV1
```

Each binary is compiled as `bench_<isa>`. The ISA define activates the corresponding `#elif` block in `config_test.h`, setting all instruction mnemonics and register names at preprocessing time.

Also adds `-DPROJECT_DIR="$(CURRENT_DIR)"` so `write_asm.c` can compute the output path to `Test/test_params.h`.

---

## Execution Flow

```
./bench_avx2 -test FLOPS -op add -fp 1024 -precision dp -num_runs 5
  │
  ├─ bench.c:main()           — parse argv into typed variables
  │
  ├─ create_bench.c:create_benchmark_flops()
  │    ├─ select_op.c:select_ISA_flops()       — resolve instruction mnemonic (e.g. "vaddpd")
  │    ├─ select_op.c:select_ISA_flops_register() — resolve register prefix (e.g. "ymm")
  │    ├─ write_asm.c:write_asm_fp()           — write Test/test_params.h containing:
  │    │     - #define directives (NUM_RUNS, FP_INST, OPS, PRECISION, ALIGN, ...)
  │    │     - static inline test_function() with __asm__ __volatile__ body
  │    └─ system("make -C ../Test -f Makefile_Benchmark isa=avx2")
  │
  └─ Test/bin/test             — compiled benchmark binary, runs timing
```

For MEM tests: `create_benchmark_mem()` → `write_asm_mem()`
For MIXED tests: `create_benchmark_mixed()` → `write_asm_mixed()`

---

## config_test.h — ISA Configuration Macros

**File**: `legacy_bench_gen/config_test.h` (327 lines)

The central ISA configuration header. All ISA-specific constants are set here via `#if defined(...)` chains and consumed by all other modules as string literals and integer constants.

### Global constants (line 8–9)

```c
#define BASE_LOOP_SIZE 256   // Maximum instructions per inner loop unroll
#define INST_LOOP_SIZE 256   // (Same value, used in some paths)
```

### Per-ISA blocks define these macros

| Macro | Type | Meaning |
|---|---|---|
| `ISA` | string | ISA name string (e.g., `"avx2"`) used in make command |
| `NUM_REGISTER` | int | Number of available FP registers |
| `MEM_SP_REGISTER` | string | Register name prefix for SP memory ops (e.g., `"ymm"`) |
| `MEM_DP_REGISTER` | string | Register name prefix for DP memory ops |
| `FP_SP_REGISTER` | string | Register name prefix for SP FP ops |
| `FP_DP_REGISTER` | string | Register name prefix for DP FP ops |
| `DP_ALIGN` | int | Byte alignment for double-precision arrays |
| `SP_ALIGN` | int | Byte alignment for single-precision arrays |
| `COBLERED` | string | Clobber list string for `__asm__ __volatile__` |
| `DP_OPS` | int | FP operations per instruction (for throughput: `DP_ALIGN / 8`) |
| `SP_OPS` | int | FP operations per instruction (for throughput: `SP_ALIGN / 4`) |
| `DP_DIV/ADD/MUL/FMA` | string | DP instruction mnemonics |
| `SP_DIV/ADD/MUL/FMA` | string | SP instruction mnemonics |
| `DP_LOAD/STORE` | string | DP memory instruction mnemonics |
| `SP_LOAD/STORE` | string | SP memory instruction mnemonics |

### ISA coverage (ISA define → register prefix)

| Define | ISA string | Register prefix (DP) | `NUM_REGISTER` |
|---|---|---|---|
| `AVX512` | `"avx512"` | `zmm` | 32 |
| `AVX` | `"avx"` | `ymm` | 32 |
| `AVX2` | `"avx2"` | `ymm` | 16 |
| `SSE` | `"sse"` | `xmm` | 16 |
| `SVE` | `"sve"` | `.d` / `.s` | 32 |
| `NEON` | `"neon"` | `q` / `d` | 32 |
| `ASCALAR` | `"armscalar"` | `d` / `s` | 32 |
| `RISCVSCALAR` | `"riscvscalar"` | `f` | 32 |
| `RVV07` | `"rvv0.7"` | `v` | 32 |
| `RVV1` | `"rvv1.0"` | `v` | 32 |
| *(default)* | `"scalar"` | `xmm` | 16 |

**Special cases:**
- SVE memory alignment: `DP_ALIGN=8`, `SP_ALIGN=4` (element size, not vector width — SVE is length-agnostic)
- RISC-V RVV: `DP_ALIGN=8`, `SP_ALIGN=4` (same, elements)
- `DP_OPS=1` for all scalar/SVE/NEON/RISC-V (SIMD lanes are runtime-variable)

---

## select_op.c — Operation and Register Selection

**File**: `legacy_bench_gen/select_op.c`

Four functions that translate `(operation_string, precision_string)` into instruction mnemonics and register prefixes by reading the preprocessor macros from `config_test.h` at compile time.

### Functions

```c
// Returns flop count per instruction (*flop) and instruction mnemonic (*assembly_op)
void select_ISA_flops(int *flop, char **assembly_op, char *operation, char *precision);

// Returns register prefix string (e.g. "ymm", "v", "f") for FP registers
void select_ISA_flops_register(char **registr, char *precision);

// Returns alignment, ops-per-inst, and load/store mnemonic for memory operations
void select_ISA_mem(int *align, int *ops, char **assembly_op, char *operation, char *precision);

// Returns register prefix string for memory registers
void select_ISA_mem_register(char **registr, char *precision);
```

### Logic

- Input `operation`: `"add"`, `"mul"`, `"div"`, `"fma"` (for `select_ISA_flops`)
- Input `operation`: `"load"`, `"store"` (for `select_ISA_mem`)
- Input `precision`: `"dp"` or `"sp"`
- Allocates the returned strings on the heap with `malloc` — caller is responsible for `free()`
- The `mad` operation (mul+add) is handled in `create_bench.c` by calling `select_ISA_flops` twice: once for `"mul"` and once for `"add"`, returning two separate mnemonics (`assembly_op_flops_1`, `assembly_op_flops_2`)

---

## calc_param.c — Loop Parameter Calculation

**File**: `legacy_bench_gen/calc_param.c`

Two functions that calculate how many loop iterations are needed to execute the requested number of operations.

### `flops_math` (line 4)

```c
long long flops_math(long long fp);
```

- Calculates the outer-loop iteration count for arithmetic benchmarks
- If `fp > BASE_LOOP_SIZE (256)`: `iter = floor(fp / BASE_LOOP_SIZE)`
- Otherwise: `iter = 1` (no outer loop needed; emit all ops in one pass)
- The inner loop always emits exactly `BASE_LOOP_SIZE` instructions when `iter > 1`

### `mem_math` (line 14)

```c
long long mem_math(long long num_rep, int num_ld, int num_st, int *num_aux, int align);
```

- Calculates outer iteration count and `num_aux` (inner unroll factor) for memory benchmarks
- `num_aux` is incremented until `num_aux * (num_ld + num_st) >= BASE_LOOP_SIZE`
- Outer iterations: `iter = floor(num_rep / num_aux)`
- **ARM AArch64 constraint** (`ASCALAR` or `NEON`): limits inner unroll so that `num_aux * (num_ld+num_st) * align < 4096 - (num_ld+num_st)*align` — prevents exceeding 12-bit signed offset limit in AArch64 load/store immediates
- **RISC-V scalar constraint** (`RISCVSCALAR`): same logic but limit is `2048` — 12-bit signed immediate in RISC-V I-format
- **RVV** (`RVV07`, `RVV1`): no alignment constraint on `num_aux` (uses register-indirect addressing)
- **x86 and others**: no offset constraint
- Minimum: `iter = max(iter, 1)` to prevent zero iterations

---

## create_bench.c — Benchmark Orchestration

**File**: `legacy_bench_gen/create_bench.c`

Three functions that orchestrate the full generation pipeline for each test type.

### `create_benchmark_flops` (line 8)

```c
void create_benchmark_flops(char *op, char *precision, long long fp,
                             int Vlen, int LMUL, int verbose, int num_runs);
```

1. If `op == "mad"`: call `select_ISA_flops` twice (for `"mul"` and `"add"`)
2. Else: call `select_ISA_flops` once
3. Call `select_ISA_flops_register`
4. Call `write_asm_fp()` — generates `Test/test_params.h`
5. Shell out: `make [-s] isa=<ISA> -C <PROJECT_DIR>/../Test -f Makefile_Benchmark`

### `create_benchmark_mem` (line 41)

```c
void create_benchmark_mem(long long num_rep, int num_ld, int num_st,
                           char *precision, int Vlen, int LMUL, int verbose, int num_runs);
```

1. Call `select_ISA_mem` for `"load"` and `"store"` separately
2. Call `select_ISA_mem_register`
3. Call `write_asm_mem()` — generates `Test/test_params.h`
4. Shell out to `make`

### `create_benchmark_mixed` (line 77)

```c
void create_benchmark_mixed(char *op, long long num_rep, int num_ld, int num_st, int num_fp,
                             char *precision, int Vlen, int LMUL, int verbose, int num_runs);
```

1. Combines FP and MEM selection
2. Calls `write_asm_mixed()`
3. Shells out to `make`

**Verbosity**: `verbose > 3` → make is run without `-s` (shows build output)

**Error handling**: only checks `system()` return for `-1` (fork failure); non-zero exit from make is silently ignored

---

## write_asm.c — Assembly Code Generation

**File**: `legacy_bench_gen/write_asm.c` (1275 lines)

The core of the legacy system. Three functions (`write_asm_fp`, `write_asm_mem`, `write_asm_mixed`) that write C source containing inline assembly to `Test/test_params.h`.

### Output file

```c
char path[8192];
snprintf(path, sizeof(path), "%s/../Test/test_params.h", PROJECT_DIR);
file_header = fopen(path, "w");
```

### write_asm_fp (line 8)

Generates arithmetic benchmark function.

**`#define` directives written** (lines ~18–55):
```c
#define NUM_RUNS <num_runs>
// ISA-specific: ARM/SVE/RISCV/RISCVVECTOR + VLEN/VLMUL
#define DIV 1           // only if op == "div"
#define NUM_LD 1        // only if op == "div" (redundant, legacy quirk)
#define NUM_ST 0        // only if op == "div"
#define OPS <flops>
#define NUM_REP 1
#define PRECISION double|float
#define ALIGN <DP_ALIGN|SP_ALIGN>
#define FP_INST <fp>
```

**Function signature generated**:
- `op == "div"`: `void test_function(PRECISION *test_var, long long num_rep_max)`
- others: `void test_function(long long num_rep_max)`

**Loop structure** (for `fp > BASE_LOOP_SIZE`):
```
outer loop (num_rep_max):
  inner loop (iter = floor(fp / BASE_LOOP_SIZE)):
    BASE_LOOP_SIZE × instruction (cycling through registers 0..NUM_REGISTER-1)
  end inner
  remaining (fp % iter) × instruction
end outer
```

**Register cycling**: `j` increments from 0..`NUM_REGISTER-1`, wraps at `NUM_REGISTER`. For RVV, `j += LMUL` to respect register grouping.

**Architecture-specific inline assembly syntax** (handled with `#if` chains throughout):

| Arch | Outer-loop counter setup | Inner-loop counter | Branch instruction |
|---|---|---|---|
| x86 | `movq %0, %%r8` | `movl $<iter>, %%edi` | `jnz Loop1_%=` |
| ARM AArch64 | `mov w0, %w0` | `ldr w1, =<iter>` | `cbnz w1, Loop1_%=` |
| RISC-V | `ld t0, %0` | `li t1, <iter>` | `bgtz t1, Loop1_%=` |

**FP instruction format per ISA** (inner-loop body, lines ~120–175):

| ISA | Format example |
|---|---|
| x86 AVX/AVX2/AVX512 | `"vaddpd %%ymm0, %%ymm1, %%ymm1\n\t\t"` |
| x86 SSE | `"addpd %%xmm0, %%xmm1;"` (semicolons instead of `\n\t\t`) |
| ARM Scalar | `"fadd d0, d1, d1\n\t"` |
| ARM NEON | `"fadd V0.2d, V0.2d, V0.2d\n\t"` |
| ARM SVE | `"fadd z0.d, p0/m, z0.d, z0.d\n\t"` |
| RISC-V Scalar | `"fadd.d f0, f1, f1\n\t"` |
| RISC-V RVV | `"vfadd.vv v0, v1, v1\n\t"` |

**SVE preamble** (line ~77): `ptrue p0.d` or `ptrue p0.s` before the loop
**RVV preamble** (line ~67): `li t4, <Real_Vlen>` + `vsetvli t0, t4, e64, m<LMUL>` before the loop

### write_asm_mem (line ~335)

Generates memory benchmark function.

**`#define` directives written**:
```c
#define MEM 1
#define NUM_LD <num_ld>
#define NUM_ST <num_st>
#define OPS <ops>
#define NUM_REP <num_rep>
#define PRECISION double|float
#define ALIGN <align>
#define FP_INST 1
```

**Memory access pattern**: unrolls `num_aux` repetitions of the LD/ST pattern per inner-loop iteration. Each consecutive instruction uses an increasing immediate offset (`offset += align`). After the inner loop body, a pointer-bump instruction advances the base pointer by `num_aux * (num_ld + num_st) * align`.

**AArch64 constraint**: offset must stay within 12-bit signed range (enforced in `calc_param.c:mem_math`)

### write_asm_mixed (line ~800, approximately)

Generates combined arithmetic + memory benchmark. Interleaves FP instructions and LD/ST instructions in the same loop body. Parameters `num_fp`, `num_ld`, `num_st` control the ratio.

---

## Generated Output: Test/test_params.h

This file is **overwritten on every generator run**. It contains:

1. `#define` configuration constants (architecture type flags, loop counts, precision, alignment)
2. A single `static inline __attribute__((always_inline)) void test_function(...)` body containing the `__asm__ __volatile__` block with all generated instructions

The generated function is `#include`d by `Test/main_test.c` (or `test_arith.c` / `test_mem.c`) to produce the actual benchmark binary.

**Key defines controlling test harness behavior in Test/main_test.c:**

| Define | Meaning |
|---|---|
| `NUM_RUNS` | Number of timing repetitions |
| `FP_INST` | Total FP instructions per invocation (for FLOP/s calculation) |
| `NUM_LD` / `NUM_ST` | Load/store counts per rep |
| `OPS` | Operations per instruction (SIMD width factor) |
| `NUM_REP` | Memory repetitions per invocation |
| `PRECISION` | `double` or `float` |
| `ALIGN` | Array element alignment in bytes |
| `MEM` | Set to 1 if memory benchmark |
| `DIV` | Set to 1 if division operation |
| `ARM` / `SVE` / `RISCV` / `RISCVVECTOR` | Architecture selection |
| `VLEN` / `VLMUL` | Vector length configuration for SVE/RVV |

---

## Two-Phase Build: Test/ Directory

The generator uses a two-phase build:

**Phase 1**: `legacy_bench_gen/bench_<isa>` runs → writes `Test/test_params.h`

**Phase 2**: `make -C Test -f Makefile_Benchmark isa=<isa>` compiles:
```
Test/main_test.c + Test/CoreClockChecker<arch>.s → bin/test
```

`Test/Makefile_Benchmark` selects:
- Architecture-specific clock-reading assembly (`CoreClockCheckerx86.s`, `CoreClockCheckerARM.s`, `CoreClockCheckerRISCV.s`)
- Compiler flags per ISA (e.g., `-march=armv8-a+sve` for SVE, `-march=rv64gcv` for RVV v1.0)
- Always uses `-Ofast -pthread`

Output binary: `bin/test`

---

## Bench/ Directory — Identical Active Copy

**`Bench/`** is a near-identical copy of `legacy_bench_gen/` with the same 8 source files and virtually the same content. Differences:

- `Bench/Makefile` builds a **single** target `./Bench` (choosing ISA via `isa=` variable at make time), not the full set of pre-built ISA binaries
- `Bench/Bench.c` = `legacy_bench_gen/bench.c` with identical logic
- All other files (`calc_param.c`, `config_test.h`, `create_bench.c`, `functions.h`, `select_op.c`, `write_asm.c`) are identical content

The `Bench/` directory appears to be the **original working location** used by `run.py` and the legacy `test_bench` build chain, while `legacy_bench_gen/` is the copy preserved for regression testing. Both directories generate to the same `Test/` output directory (resolved via `PROJECT_DIR`).

---

## New System: benchmark/generation/

**Location**: `benchmark/generation/`
**Documentation**: `benchmark/generation/README.md`

The replacement is a pure-Python code generation system:

```
benchmark/generation/
├── isa.py           # BaseISA abstract class, InlineASM dataclass, loop-split helpers
├── x86.py           # X86Scalar, X86SSE, X86AVX, X86AVX2, X86AVX512
├── arm.py           # ArmScalar, ArmNeon, ArmSVE
├── riscv.py         # RISCVScalar, RISCV_RVV_071, RISCV_RVV
├── parameters.py    # ArithmeticBenchmarkParams, MemoryBenchmarkParams (typed dataclasses)
├── code_gen/        # Instruction, Register, Operation, DataType abstractions
└── __init__.py      # ALL_ISAS tuple, ISA_NAME_TO_CLASS dict, INCOMPATIBLE_ISAS
```

**Key exports** from `benchmark/generation/__init__.py`:
- `ALL_ISAS: tuple[type[BaseISA], ...]` — all registered ISA classes
- `ISA_NAME_TO_CLASS: dict[str, type[BaseISA]]` — name → class lookup
- `INCOMPATIBLE_ISAS` — pairs of ISAs that cannot be used together

---

## Key Differences vs New Python System

| Aspect | Legacy (`legacy_bench_gen/`) | New (`benchmark/generation/`) |
|---|---|---|
| **Implementation language** | C | Python |
| **ISA selection mechanism** | Preprocessor define at generator compile time; separate binary per ISA | Runtime class selection; single unified pipeline |
| **Output artifact** | Writes `Test/test_params.h` file to disk | Returns `MicrobenchmarkFunctionSpec` Python object in memory |
| **Compilation trigger** | Shells out `system("make ...")` from within benchmark execution | Controlled by `test_bench/builder.py` separately |
| **ISA encoding** | Macros in `config_test.h` (string literals, ints) | Typed Python dataclasses with inheritance hierarchy |
| **Loop splitting** | Hardcoded `BASE_LOOP_SIZE=256` in `calc_param.c` | Configurable `instruction_limit` per ISA class (`BaseISA.instruction_limit`, default 2048) |
| **Register cycling** | `j` counter cycling 0..`NUM_REGISTER-1` in C loops | Managed by `code_gen/` register abstraction objects |
| **AArch64 offset constraint** | Hardcoded `< 4096` check in `mem_math` | Encapsulated in `ArmScalar.max_unique_offsets()` / `offset_increment()` |
| **RISC-V offset constraint** | Hardcoded `< 2048` check in `mem_math` | Encapsulated in `RISCVScalar` override methods |
| **MAD operation** | Dual-mnemonic path: calls `select_ISA_flops` twice | Single operation type, fused at instruction level |
| **Type safety** | None — `char*` strings everywhere | Typed enums: `DataType`, `ArithmeticOperation`, `Operation` |
| **Test types** | `FLOPS`, `MEM`, `MIXED` | `ArithmeticBenchmarkParams`, `MemoryBenchmarkParams` (MIXED not directly exposed as a named type) |
| **VLEN/LMUL** | CLI flags passed as raw `int` through call chain | Typed constructor parameters in `ArmSVE(vlen_bits=...)` and `RISCV_RVV(vlen_bits=..., lmul=...)` |
| **div operation quirk** | Generates extra `NUM_LD=1`, `NUM_ST=0`, `MEM` defines and a memory preload instruction even for pure FLOPS test | Not preserved; treated as a pure arithmetic operation |
| **SSE semicolons** | Uses `;` separator instead of `\n\t\t` — inconsistency | Uniform newline separator |
| **Clobber list** | Hardcoded string macro `COBLERED` in `config_test.h` | Dynamic clobber list built from register set |
| **Error handling** | Silent on many failures; only checks `system()` return for -1 | Raises `BenchParamError` on invalid params |

---

## Migration Notes

### What was preserved

- **Loop structure semantics**: inner loop of `BASE_LOOP_SIZE` (256) ops, outer `num_rep_max` loop — the new system emits functionally equivalent loop structure for arithmetic tests, verified by `refactor_tests/test_bench_gen.py`
- **Instruction mnemonics**: same ISA instructions (`vfmadd231pd`, `fld`, `vfadd.vv`, etc.)
- **Register naming conventions**: same register prefixes per ISA
- **AArch64/RISC-V offset constraints**: same numeric limits, now encoded in class methods
- **Vector preamble**: `ptrue p0.d` for SVE; `vsetvli` for RVV — both preserved
- **Benchmark output `#define` constants**: same set of constants written to output header (verified via `compare_asm` in `refactor_tests/asm_comparison.py`)

### What changed fundamentally

- **No more disk I/O in generator**: new system builds an in-memory function spec, not a file
- **No `system("make")` call inside generator**: build is orchestrated externally via `test_bench/builder.py`
- **Single binary, multiple ISAs**: no more per-ISA generator binaries
- **`instruction_limit` raised from 256 → 2048**: allows denser benchmarks
- **`MIXED` not directly named**: functionality subsumed by `MemoryBenchmarkParams` with FP component
- **`mad` implemented differently**: legacy interleaves `mul`/`add` as alternating instructions with two separate mnemonics; new system's exact behavior depends on `code_gen/` implementation
- **`div` operation no longer inserts spurious MEM defines**: legacy `write_asm_fp` always set `DIV=1`, `NUM_LD=1`, `NUM_ST=0`, `OPS`, `NUM_REP`, `PRECISION`, `ALIGN` even for pure FLOPS tests with div — this was a legacy quirk not replicated

### Known divergences (documented in test infra)

- `num_rep=0` for MEM tests: legacy does not fail; new raises `BenchParamError` — documented exception in `test_bench_gen.py:compare_tests` (line ~69)
- `op=div` for FLOPS: legacy adds redundant memory-related `#define`s and a preload instruction; new does not — documented exception in `asm_comparison.py:compare_asm` (line ~333)
- `op=mad`: register interleaving differs between legacy (alternating registers) and new (potentially sequential) — `mad_exception` flag in `ParsedLoops.deep_comparison` skips exact sequence check
- ARM NEON: register format comparison skipped (`arm_neon_exception` flag) due to differing but functionally equivalent register naming

---

## Test Infrastructure: refactor_tests/

### test_bench_gen.py

**File**: `refactor_tests/test_bench_gen.py` (127 lines)

Regression test driver that runs both legacy and new generators over the same parameter grid and compares outputs.

**Test grid** (combinatorial product):
```python
FLOP_COMBINATION_SET = {
    "-precision": ("dp", "sp"),
    "-test": ("FLOPS",),
    "-op": ("add", "mul", "div", "fma", "mad"),
    "-fp": (0, 1, 32, 1024),
}
MEM_COMBINATION_SET = {
    "-test": ("MEM",),
    "-num_LD": (0, 1, 2),
    "-num_ST": (0, 1, 2),
    "-num_rep": (0, 1, 32, 1024),
    ...
}
```

RVV additionally covers `VLEN ∈ {1, 512, 1024}` and `LMUL ∈ {1, 2, 8}`.

**Requires**: pre-built legacy binaries at `legacy_bench_gen/bench_<isa>` (built via `make` in that directory).

### asm_comparison.py

**File**: `refactor_tests/asm_comparison.py` (420 lines)

Structured ASM comparison engine. Does not do string equality — instead parses both outputs into `ParsedLoops` and compares semantically:

```python
@dataclass
class ParsedLoops:
    sequence: dict[str, int]   # {mnemonic: normalized_count}
    iterations: int             # inner-loop iteration count
    inner_insts: int            # instructions in inner loop body
    outer_insts: int            # instructions outside inner loop
    ptr_increment: int          # bytes advanced per inner iteration (MEM only)
    register_format: str        # register prefix in use
```

**ISA-specific regex patterns** (`ISAPatterns` subclasses):
- `X86Patterns` — matches `movq`/`movl` counter setup, `jnz` branches
- `ArmScalarPatterns` — matches `mov w0/x0` setup, `cbnz` branches
- `ArmNeonPatterns` — extends `ArmScalarPatterns` with NEON register format
- `RISCVScalarPatterns` — matches `ld`/`li` setup, `bgtz` branches
- `RVV0_7Patterns` / `RVV1_0Patterns` — adds `vsetvli` preamble and vector load/store patterns

**Comparison checks** (`ParsedLoops.deep_comparison`):
1. Total instruction count must match (unless `mad` exception)
2. Pointer increment must match
3. Instruction sequence (mnemonic → count map, GCD-normalized) must match
4. Register format must match (unless ARM NEON exception)
5. Reports loop structure difference (inner/outer split differs) as warning, not error

**Entry point** for comparison: `compare_asm(test, legacy_bench, new_bench) -> (bool, str)`

**ISA pattern dispatch table** (line ~385):
```python
ISA_PATTERNS_MAP: dict[str, ISAPatterns] = {
    "riscvscalar": RISCVScalarPatterns(),
    "rvv0.7":      RVV0_7Patterns(),
    "rvv1.0":      RVV1_0Patterns(),
    "scalar":      X86Patterns(),
    "sse":         X86Patterns(),
    "avx":         X86Patterns(),
    "avx2":        X86Patterns(),
    "avx512":      X86Patterns(),
    "armscalar":   ArmScalarPatterns(),
    "neon":        ArmNeonPatterns(),
    "sve":         ArmNeonPatterns(),
}
```
