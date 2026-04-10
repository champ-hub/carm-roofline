# GPU Benchmarking and AI Calculators

**Scope**: `run_gpu.py`, `ROC_AI_Calculator.py`, `NCU_AI_Calculator.py`, `SDE_AI_Calculator.py`, `DBI_AI_Calculator.py`, `PMU_AI_Calculator.py`, and the `GPU/` directory.

---

## Section 1: GPU Benchmarking (`run_gpu.py`)

### Purpose

`run_gpu.py` is the **legacy GPU entry point** for constructing the Cache-Aware Roofline Model on GPU platforms. It supports both **NVIDIA (CUDA)** and **AMD (ROCm/HIP)** GPUs. It:

1. Auto-detects the GPU architecture via `nvidia-smi` / `amd-smi`
2. Detects compute capability and available precision modes (vector + tensor/matrix cores)
3. Compiles and runs GPU micro-benchmarks from `GPU/Bench/`
4. Measures FLOP/s and memory bandwidth (shared, L2, global) per precision
5. Writes results to CSV files in `{output}/Roofline/`

### CLI Arguments

```
python run_gpu.py [config] [options]
```

| Argument | Default | Choices / Type | Description |
|---|---|---|---|
| `config` | (none) | positional, optional | Path to config file (`name=<value>` format) |
| `--test` | `roofline` | `FP`, `TC`, `roofline`, `MEM` | Type of test: roofline measures FP + all memory levels |
| `--name` | `unnamed` | str | GPU name label when no config file is used |
| `-v` / `--verbose` | `1` | `0`–`3` | Output detail: 0=none, 1=errors+details, 2=intermediate results, 3=configuration |
| `-out` / `--output` | `./Results` | path | Output directory (creates `{output}/Roofline/` subdirectory) |
| `--freq_sm` | `0` | int (MHz) | Target SM frequency (requires `--set_freq`) |
| `--freq_mem` | `0` | int (MHz) | Target memory frequency (requires `--set_freq`) |
| `--set_freq` | off | flag | Enables frequency locking via `nvidia-smi -lgc`/`-lmc` and persistency mode |
| `--vector` | `auto` | `none auto hp hp2 int sp dp bf16` | Vector core precisions to test |
| `--tensor` | `auto` | `none auto fp16_32 fp16_16 tf32 bf16 fp8 int8 int4 int1 fp64 fp32` | Tensor/matrix core precisions to test |
| `--vector_op` | `add` | `fma add mul` | Arithmetic operation for vector core benchmarks |
| `--threads` | `1024` | int | Threads per block |
| `--blocks` | `32768` | int | Number of thread blocks |

### Execution Pipeline

```
main()
  1. Auto-detect GPU: try nvidia-smi → arch="nvidia", else try amd-smi → arch="amd"
  2. check_hardware(verbose, set_freq, freq_sm, freq_mem, arch, target_vector, target_tensor)
     a. For NVIDIA: query compute capability via nvidia-smi --query-gpu=compute_cap
     b. For AMD: query via `amd-smi static --json`, parse target_graphics_version (gfxNNN)
     c. Build supported vector_precisions list (int, hp, hp2, sp, dp; +bf16 for CC≥80 / GFX9+)
     d. Build tensor_core_precisions list:
        - NVIDIA: CC≥70 (not GTX): fp16_32; CC≥75: +fp16_16,int8,int4; CC≥80: +bf16,tf32,int1; CC=80: +fp64
        - AMD: gfx908/gfx90a/gfx942 only: fp32,int8,fp16_32,bf16; gfx90a+: +fp64; gfx942: +tf32,fp8
     e. Filter target_vector and target_tensor to supported subset (warn on unsupported)
     f. For NVIDIA + set_freq: enable persistency mode → set SM freq → set MEM freq
  3. run_roofline(...)
     a. Compile GPU benchmark generator: `cd GPU && make -s clean && make -s`
     b. For each precision in target_vector:
        - If vector_op != "fma": run Bench --test FLOPS --target vector --operation {vector_op}
        - Always run: Bench --test FLOPS --target vector --operation fma
        - Run: Bench --test MEM --target shared
        - Run: Bench --test MEM --target L2
        - Run: Bench --test MEM --target global
        - Each Bench invocation generates a kernel binary at ./GPU/bin/test
        - Execute ./GPU/bin/test, capture stdout as GB/s or Gflops/s
        - Append to outputs dict: {flops, fma, shared, l2, global}
        - Call update_csv(name, "Roofline", outputs, date, "cuda"/"vector-amd", precision, ...)
     c. For each precision in target_tensor:
        - Run: Bench --test FLOPS --target tensor
        - Run: Bench --test MEM --target shared/L2/global (fp32 fallback for fp8)
        - Call update_csv(..., "tensor", precision, "mma", ...)
  4. shutdown(set_freq): disable nvidia-smi persistency mode if set_freq was used
```

### CSV Output Format

Written to `{output}/Roofline/{name}_Roofline.csv`.

Two header rows (on first write), then one data row per precision/test:

```
[secondary_headers row]  # Name, sizes
[primary_headers row]    # Date, ISA, Precision, Threads, Loads, Stores, Blocks, DRAM Bytes, FP Inst., GB/s, I/Cycle, ...
[Date, ISA, Precision, ThreadsPerBlock, 1, 1, Blocks, 0, inst, shared_bw, 0, 0, 0, l2_bw, 0, global_bw, 0, flops, 0, fma, 0]
```

Memory columns layout: Shared (GB/s + I/Cycle), L1 (placeholder 0), L2 (GB/s + I/Cycle), Global (GB/s + I/Cycle), FP (Gflops/s + I/Cycle), FP FMA (Gflops/s + I/Cycle).

### GPU/ Directory Structure

```
GPU/
├── gpu.env              # Environment config loaded by run_gpu, ROC_AI_Calculator, NCU_AI_Calculator
│                        # Variables: CUDA_PATH, ROCM_PATH, ROCPROFV3_PATH, DEVICE
├── Makefile             # Delegates: cd Bench && make
├── Bench/               # GPU benchmark generator (C++)
│   ├── Bench.cpp        # Main CLI: --test FLOPS/MEM --target vector/tensor/shared/L2/global
│   │                    # --arch nvidia/amd --operation --precision --compute --threads
│   │                    # --blocks --device → generates CUDA/HIP kernel at ./GPU/bin/test
│   ├── create_bench.cpp # Kernel code generation
│   └── functions.h      # Benchmark function templates
├── Test/
│   ├── nvidia/          # Subdirs: flops/, mem/ (test kernel sources)
│   └── amd/
└── rocm_counters.txt    # rocprofv3 counter groups (3 pmc lines):
                         # pmc 1: SQ_INSTS_VALU_{ADD,MUL,FMA}_{F16,F32}
                         # pmc 2: SQ_INSTS_VALU_{ADD,MUL,FMA}_F64 + MFMA_MOPS_{F16,BF16,F32,F64,I8}
                         # pmc 3: SQ_LDS_{BANK_CONFLICT,IDX_ACTIVE} + TCP_TOTAL_CACHE_ACCESSES_sum
```

### Dependencies

| Component | Requirement |
|---|---|
| NVIDIA path | `nvidia-smi` in PATH, CUDA toolkit at `CUDA_PATH` from `gpu.env` |
| AMD path | `amd-smi` in PATH, ROCm at `ROCM_PATH`, `rocprofv3` at `ROCPROFV3_PATH` from `gpu.env` |
| Build | `make` + HIP/CUDA compiler available |
| Python | `python-dotenv`, standard library only |
| Config file | `GPU/gpu.env` must set `DEVICE` (GPU index, default 0) |

### Precision Availability Matrix

| Precision | NVIDIA CC | AMD Target |
|---|---|---|
| int | any | not supported |
| hp / hp2 | any | GFX9+ |
| sp / dp | any | any |
| bf16 | CC≥80 | GFX9+ |
| fp16_32 TC | CC≥70, not GTX | gfx908/90a/942 |
| fp16_16 TC | CC≥75 | — |
| tf32 TC | CC≥80 | gfx942 |
| int8/int4/int1 TC | CC≥75 | gfx908/90a/942 (int8 only) |
| fp64 TC | CC=80 | gfx90a/942 |
| fp8 TC | CC≥89 (not yet) | gfx942 |

---

## Section 2: AI Calculators (Analysis Tools)

### Role and Common Pattern

The `*_AI_Calculator.py` scripts are **standalone application profilers** that compute the Arithmetic Intensity (AI) and performance of an existing user application. They are used to **place application dots** on the roofline chart. They are independent from `carm.py`/`run_gpu.py` which measure roofline ceilings.

**Shared output pattern**: All calculators call `ut.update_csv()` or their own `update_csv()` to write to a machine-specific CSV:
- CPU calculators: `Results/Applications/{machine_name}_Applications.csv`
- GPU calculators: `Results/Applications/{machine_name}_Applications.csv`

**Common CSV columns**: `Date, Method, Name, ISA, Precision, Threads, AI, Gflops, Bandwidth, Time`

**Derived metrics** (computed identically across all calculators):
```python
ai        = total_flops / total_bytes
gflops    = total_flops / execution_time_ns      # result is GFLOP/s when time is in ns
bandwidth = total_bytes / execution_time_ns       # result is GB/s
```

---

### `PMU_AI_Calculator.py` — PAPI Hardware Counters

**Backend**: PAPI high-level API (requires PAPI library installed and app instrumented).

**Supported platforms**: x86_64, aarch64.

**How it works**:
1. Runs the target application **3 times** sequentially, once per PAPI event:
   - `PAPI_LST_INS` — Load/Store instructions
   - `PAPI_SP_OPS` — Single-precision FP operations
   - `PAPI_DP_OPS` — Double-precision FP operations
2. Sets `PAPI_OUTPUT_DIRECTORY` and `PAPI_EVENTS` env vars before each run
3. Reads per-thread JSON files from `carm_pmu_output/` (emitted by PAPI high-level API)
4. Sums `real_time_nsec` and event counts across all threads and regions
5. Execution time = average of the 3 runs' `real_time_nsec / thread_count`

**Memory byte calculation**:
```python
memory_bytes = PAPI_LST_INS * (sp_ratio * 4 + dp_ratio * 8) * scale
# scale = vlen // 8  (default 1 when --vlen not set)
```

**Application requirements**: Target must use PAPI high-level API (`PAPI_hl_region_begin` / `PAPI_hl_region_end`) to define ROI regions.

**CLI**:
```
python PMU_AI_Calculator.py <executable_path> [additional_args...]
  -d / --debug         # Ignored (output always cleaned)
  -dr / --drawroof     # Plot flag (work in progress)
  -c / --choice        # Roofline chart selector (requires --drawroof)
  -n / --name          # Machine name for CSV (default: unnamed)
  -an / --app_name     # Application name for CSV
  --isa                # ISA label: avx512|avx|avx2|sse|scalar|neon|armscalar|...
  --vlen               # Scales ld/st byte count: memory_bytes *= (vlen // 8)
```

**Output method label in CSV**: `PMU`

---

### `DBI_AI_Calculator.py` — DynamoRIO / Intel SDE Binary Instrumentation

**Backend**: DynamoRIO (cross-platform) or Intel SDE (x86 only). Selected via `--sde` flag.

**Supported platforms**: x86_64, aarch64 (DynamoRIO only for aarch64).

#### DynamoRIO Backend

1. **Setup**: Builds `CustomClient/opcoder.c` as `libopcoder.so` using cmake against the provided DynamoRIO installation (`carm_dbi_build/bin/libopcoder.so`)
2. **Execution**: `drrun -c libopcoder.so [--roi] -- <executable>`
3. **Output file**: `carm_dbi_output.txt` — structured sections:
   - "Floating Point and Integer opcode execution counts" — per opcode with vectorization tier label
   - "Memory opcode execution counts" — with byte size per access
   - "Miscellaneous Opcode execution counts"
4. **Parsing** (`analyseDynamoRIOx86` / `analyseDynamoRIOARM`): dispatches each opcode to one of the static instruction dicts by ISA tier:
   - `x86_Scalar_fp_operations`, `x86_SSE_fp_operations`, `x86_AVX2_fp_operations`, `x86_AVX512_fp_operations`
   - Same tiers for int operations
   - Memory: `count * byte_size` from opcode description field

#### SDE Backend (`--sde`)

- Calls `runSDE()` → runs `sde64 -iform -mix -dyn_mask_profile [--roi markers] -- <executable>`
- Calls `analyseSDE()` → invokes `SDE_AI_Calculator.py` as subprocess, parses stdout for regex patterns:
  - `Single prec. FLOPs: (\d+)`
  - `Double prec. FLOPs: (\d+)`
  - `Total bytes written: (\d+)` / `Total bytes read: (\d+)`

#### ROI Mode (`--roi`)

- DynamoRIO: passes `-roi` flag to client
- SDE: adds `-start_ssc_mark FACE:repeat -stop_ssc_mark DEAD:repeat`
- Timing: reads `carm_timing_results.txt` written by app (format: `Time Taken: X.Y seconds`)
- Non-ROI timing: wall-clock via `time.time_ns()`

**CLI**:
```
python DBI_AI_Calculator.py <dbi_path> <executable_path> [additional_args...]
  --roi                # Measure only region of interest
  --sde                # Use Intel SDE instead of DynamoRIO (x86 only)
  -n / --name          # Machine name (default: unnamed)
  -an / --app_name     # Application name for CSV
  --isa                # ISA label for CSV
  -t / --threads       # Thread count label for CSV
  -p / --precision     # dp|sp label for CSV
  -dr / --drawroof     # Plot flag (work in progress)
  -c / --choice        # Chart selector (requires --drawroof)
```

**Output method labels in CSV**: `DR`, `DR-ROI`, `SDE`, `SDE-ROI`

---

### `SDE_AI_Calculator.py` — Intel SDE FLOP Counter (standalone parser)

**Purpose**: Parses two SDE output files and reports per-thread FLOPs, bytes, and AI. Not an argparse script.

**Invocation**:
```bash
# Generate SDE output first:
sde64 -iform -mix -dyn_mask_profile -start_ssc_mark FACE:repeat -stop_ssc_mark DEAD:repeat -- <app>

# Parse:
python SDE_AI_Calculator.py [sde-mix-out.txt] [sde-dyn-mask-profile.txt]
# Defaults: sde-mix-out.txt + sde-dyn-mask-profile.txt
```

**Parsing pipeline**:
- `flops_mix(mix_file)`: Parses `sde-mix-out.txt` (`-mix -iform` output)
  - Per TID block between `EMIT_DYNAMIC_STATS` / `END_DYNAMIC_STATS`
  - Counts `*elements_fp_double_N` and `*elements_fp_single_N` groups
  - Also counts FMA instructions and byte reads/writes
- `flops_dyn(dyn_file)`: Parses `sde-dyn-mask-profile.txt` (`-dyn_mask_profile` output)
  - Counts **masked** FP operations (AVX-512 masking)
  - Special BF16 handling via `vdpbf16` with mask detection (marked experimental)

**Output** (printed per TID, then summed):
```
Single prec. FLOPs: N
Double prec. FLOPs: N
Total bytes written: N
Total bytes read: N
Total arithmetic intensity (approx.): X.XXXXXX (EXPERIMENTAL)
```

**Note**: This script is also called as a subprocess by `DBI_AI_Calculator.py` (`analyseSDE()` function).

**Limitations**: x86_64 only. BF16 AVX-512 masked counting is experimental. No argparse — positional files only.

---

### `ROC_AI_Calculator.py` — ROCm Application Profiler (rocprofv3)

**Backend**: `rocprofv3` (ROCm profiler v3).

**Configuration** (from `GPU/gpu.env`):
- `DEVICE` — GPU device index
- `ROCM_PATH` — ROCm installation path
- `ROCPROFV3_PATH` — Full path to rocprofv3 binary (e.g. `/usr/bin/rocprofv3`)

**Counter collection**: Uses `GPU/rocm_counters.txt` (3 `pmc` groups collected in separate passes):
- Pass 1: `SQ_INSTS_VALU_{ADD,MUL,FMA}_{F16,F32}` → half + float FLOPs
- Pass 2: `SQ_INSTS_VALU_{ADD,MUL,FMA}_F64` + `SQ_INSTS_VALU_MFMA_MOPS_{F16,BF16,F32,F64,I8}` → double + matrix FLOPs
- Pass 3: `SQ_LDS_{BANK_CONFLICT,IDX_ACTIVE}` + `TCP_TOTAL_CACHE_ACCESSES_sum` → bytes

**FLOP computation**:
```python
half_flops   = 64 * (SQ_INSTS_VALU_ADD_F16 + 2*SQ_INSTS_VALU_FMA_F16 + SQ_INSTS_VALU_MUL_F16)
float_flops  = 64 * (SQ_INSTS_VALU_ADD_F32 + 2*SQ_INSTS_VALU_FMA_F32 + SQ_INSTS_VALU_MUL_F32)
double_flops = 64 * (SQ_INSTS_VALU_ADD_F64 + 2*SQ_INSTS_VALU_FMA_F64 + SQ_INSTS_VALU_MUL_F64)
tensor_flops = 512 * (MFMA_MOPS_F16 + MFMA_MOPS_BF16 + MFMA_MOPS_F32 + MFMA_MOPS_F64 + MFMA_MOPS_I8)
```

**Byte computation**:
```python
bytes_requested = (SQ_LDS_IDX_ACTIVE - SQ_LDS_BANK_CONFLICT) * 4 * 32 + TCP_TOTAL_CACHE_ACCESSES_sum * 64
```

**Profiling levels**:
- `app`: aggregates all kernels, execution_time averaged across 3 counter passes
- `kernel`: groups by `Kernel_Name`, reports each kernel separately with call count

**Execution flow** (`run_ncu()` — note: function is misnamed; it wraps rocprofv3):
```
rocprofv3 -i rocm_counters.txt -o tmp -d ./counters -T [--kernel-include-regex NAME] -- <executable>
→ parse CSV files in ./counters/pmc_{1,2,3}/tmp_counter_collection.csv
→ compute metrics
→ update_csv(...)
→ shutil.rmtree(./counters)
```

**CLI**:
```
python ROC_AI_Calculator.py <executable_path> [additional_args...]
  -n / --name          # Machine name
  -an / --app_name     # Application label
  --no_tensor          # Skip MFMA counter collection
  -k / --kernel_name   # Filter to single kernel (regex)
  -l / --level         # app (default) | kernel
```

**Output method label in CSV**: `Rocprofv3`

---

### `NCU_AI_Calculator.py` — NVIDIA Nsight Compute Profiler

**Backend**: `ncu` (Nsight Compute CLI) from `{CUDA_PATH}/bin/ncu`.

**Configuration** (from `GPU/gpu.env`):
- `DEVICE` — GPU device index
- `CUDA_PATH` — CUDA installation path (e.g. `/usr/local/cuda-12.2`)

**Metrics collected** (single-pass profiling):
```
# CUDA core metrics (always collected):
sm__sass_data_bytes_mem_global.sum
sm__sass_data_bytes_mem_local.sum
sm__sass_data_bytes_mem_shared.sum
gpu__time_duration.avg
sm__sass_thread_inst_executed_op_{fadd,ffma,fmul,hadd,hfma,hmul,dadd,dfma,dmul}_pred_on.sum

# Tensor core metrics (unless --no_tensor):
sm__ops_path_tensor_src_{bf16_dst_fp32,fp16_dst_fp16,fp16_dst_fp32,fp64,int1,int4,int8,tf32_dst_fp32}.sum
```

**ncu invocation flags**: `--replay-mode kernel --clock-control none --print-units base --csv --log-file tmp_report.csv --devices {DEVICE}`

**FLOP computation**:
```python
half_flops   = hadd + 2*hfma + hmul
float_flops  = fadd + 2*ffma + fmul
double_flops = dadd + 2*dfma + dmul
tensor_flops = sum of all sm__ops_path_tensor_src_*.sum
```

**Byte computation**:
```python
bytes_requested = sm__sass_data_bytes_mem_global + sm__sass_data_bytes_mem_local + sm__sass_data_bytes_mem_shared
```

**Profiling levels**:
- `app`: groups by `Metric Name`, sums across all kernels
- `kernel`: groups by `(Kernel Name, Metric Name)`, reports per-kernel with call count

**Flow** (`run_ncu()`):
```
ncu [flags] --metrics <metric_list> -- <executable>
→ preprocess_output(): strips non-CSV header lines from tmp_report.csv
→ process_metrics(): parse with pandas
→ print results + update_csv()
→ os.remove(tmp_report.csv)
```

**CLI**:
```
python NCU_AI_Calculator.py <executable_path> [additional_args...]
  -n / --name          # Machine name
  -an / --app_name     # Application label
  --no_tensor          # Disable tensor core metric collection
  -k / --kernel_name   # Single-kernel filter (passed as -k to ncu)
  -l / --level         # app (default) | kernel
```

**Output method label in CSV**: `NCU`

---

## Quick Reference: Which Calculator to Use

| Scenario | Tool | Notes |
|---|---|---|
| CPU app, PAPI-instrumented (ROI defined with PAPI HL API) | `PMU_AI_Calculator.py` | Most accurate; requires app recompilation with PAPI |
| CPU app, want opcode-level breakdown, x86 or ARM | `DBI_AI_Calculator.py --sde=0` | Zero instrumentation needed; DynamoRIO must be installed |
| CPU app, x86 only, no DynamoRIO, want masked SIMD counting | `DBI_AI_Calculator.py --sde` | Requires Intel SDE; slower (emulated) |
| CPU app, raw SDE file analysis only | `SDE_AI_Calculator.py` | Parses existing sde-mix-out.txt + sde-dyn-mask-profile.txt |
| AMD GPU app | `ROC_AI_Calculator.py` | Requires ROCm + rocprofv3 |
| NVIDIA GPU app | `NCU_AI_Calculator.py` | Requires CUDA + ncu (Nsight Compute) |
| GPU roofline ceilings (peak BW + FLOP/s) | `run_gpu.py` | Builds and runs GPU micro-benchmarks |

## Environment Setup (`GPU/gpu.env`)

All GPU scripts load `GPU/gpu.env` via `python-dotenv`. Required variables:

```dotenv
CUDA_PATH=/usr/local/cuda-12.2   # NVIDIA: path to CUDA installation
ROCM_PATH=/opt/rocm              # AMD: path to ROCm installation
ROCPROFV3_PATH=/usr/bin/rocprofv3 # AMD: full path to rocprofv3 binary
DEVICE=0                         # GPU device index
```
