# `run.py` — Legacy Entry Point Documentation

> **Status**: DEPRECATED. Replaced by `carm.py`. Emits `DeprecationWarning` at import time ([run.py L3–7](../../run.py#L3-L7)).
> **Do not add new features here.** All new work goes in `carm.py` + refactored modules.

---

## What `run.py` Was

Monolithic legacy entry point for CARM CPU benchmarking. It handled:
- Hardware auto-detection (ISA capabilities, cache sizes, vector lengths)
- Benchmark code generation via `make` + the `Bench/Bench` C binary
- Timing harness execution via the `bin/test` binary
- Manual computation of GB/s and GFLOP/s from raw timing output
- CSV result writing and optional matplotlib SVG plotting

All logic lived in a single ~1440-line Python file with no abstraction layers. No type hints. No tests.

---

## File References

| Component | Location |
|-----------|----------|
| Entry point / `main()` | [run.py L1348–1441](../../run.py#L1348-L1441) |
| Hardware detection | [run.py L44–397](../../run.py#L44-L397) |
| `autoconf()` (x86 probe) | [run.py L399–414](../../run.py#L399-L414) |
| `read_config()` (INI parser) | [run.py L416–430](../../run.py#L416-L430) |
| `run_roofline()` | [run.py L564–1000](../../run.py#L564) |
| `run_memory()` | [run.py L1002–1150](../../run.py#L1002) |
| `run_mixed()` | [run.py L1195–1340](../../run.py#L1195) |
| `update_csv()` (roofline CSV) | [run.py L470–530](../../run.py#L470) |
| `update_memory_csv()` | [run.py L1152–1193](../../run.py#L1152) |
| `plot_roofline()` | [run.py L432–468](../../run.py#L432) |
| Global ISA/size tables | [run.py L33–40](../../run.py#L33-L40) |

---

## CLI Arguments

All arguments to `main()` parsed via `argparse.ArgumentParser`.

### Test Selection

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--test` | str (choices) | `'roofline'` | Test type: `FP`, `L1`, `L2`, `L3`, `DRAM`, `roofline`, `MEM`, `mixedL1`, `mixedL2`, `mixedL3`, `mixedDRAM` |
| `config` | positional str | `None` | Path to INI-style config file (see format below) |

### ISA & Precision

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--isa` | `nargs='+'` str list | `['auto']` | ISA(s) to test. Choices: `avx512`, `avx2`, `sse`, `scalar`, `neon`, `armscalar`, `sve`, `riscvscalar`, `rvv0.7`, `rvv1.0`, `auto` |
| `-p` / `--precision` | `nargs='+'` str list | `['dp']` | Precision(s). Choices: `dp` (double), `sp` (single) |
| `-vl` / `--vector_length` | `positive_int` | `1` | Vector length in dp/sp elements. For RVV and SVE only. `1` means use hardware max. |
| `-vlmul` / `--vector_lmul` | int (choices) | `1` | RVV register grouping LMUL. Choices: `1`, `2`, `4`, `8` |

### Threading

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-t` / `--threads` | `nargs='+'` `positive_int` list | `[1]` | Thread count(s) to sweep (space-separated, no commas) |
| `-i` / `--interleaved` | store_const int | `0` | Enable NUMA-interleaved thread binding (cores 0,2,4,… on node 0) |

### Arithmetic Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--inst` | str (choices) | `'add'` | FP instruction for arithmetic/FP tests. Choices: `add`, `mul`, `div`, `fma`. FMA is always measured additionally in `roofline` mode. |
| `-ops` / `--num_ops` | `positive_int` | `32768` | Number of FP operations in FP/arithmetic test |
| `-nr` / `--num_runs` | `positive_int` | `1024` | Repetitions passed to `Bench/Bench` binary |

### Memory Access Pattern

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-ldst` / `--ld_st_ratio` | `positive_int` | `2` | Number of loads per store (sets `num_ld = ratio`, `num_st = 1`) |
| `--only_ld` | store_const int | `0` | Load-only mode; sets `num_ld=1, num_st=0` (overrides `--ld_st_ratio`) |
| `--only_st` | store_const int | `0` | Store-only mode; sets `num_ld=0, num_st=1` (overrides `--ld_st_ratio`) |
| `-fpldst` / `--fp_ld_st_ratio` | `positive_int` | `1` | FP-to-LD+ST ratio for `mixed*` tests only (sets `num_fp`) |

### Cache / Memory Sizing

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-l1` / `--l1_size` | int | `0` | L1 cache size override in KiB (0 = auto-detect) |
| `-l2` / `--l2_size` | int | `0` | L2 cache size override in KiB (0 = auto-detect) |
| `-l3` / `--l3_size` | int | `0` | L3 total size override in KiB (0 = auto-detect) |
| `-tl1` / `--threads_per_l1` | int | `1` | Expected threads sharing a single L1 (used in rep count calculation) |
| `-tl2` / `--threads_per_l2` | int | `2` | Expected threads sharing a single L2 (used in rep count calculation) |
| `--l3_kbytes` | `positive_int` | `0` | Manual total array size for L3 test in KiB (0 = auto-calculate) |
| `--dram_kbytes` | `positive_int` | `524288` | Total array size for DRAM test in KiB (default: 512 MiB) |
| `--dram_auto` | store_const int | `0` | Auto-scale DRAM test size to `2 × L3_size × threads` to guarantee off-L3 access |

### Frequency

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--freq` | float (GHz) | `2.0` | Nominal/expected CPU frequency; used when `--no_freq_measure` or `--set_freq` is active |
| `--no_freq_measure` | store_const int | `0` | Skip at-runtime frequency measurement (use `--freq` value directly) |
| `--set_freq` | store_const int | `0` | x86 only: set CPU max frequency to `--freq` via `autoconf` binary (may require root) |

### Output & Verbosity

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-v` / `--verbose` | int (choices) | `3` | Verbosity: `0`=silent, `1`=errors+config, `2`=results, `3`=detection info, `4`=debug |
| `--name` | str | `'unnamed'` | Name used in CSV filenames and plot titles (overridden by config file `name=` field) |
| `--plot` | store_const int | `0` | Save per-test SVG roofline/memory-curve plots to `carm_results/Roofline/` or `carm_results/memory_curve/` |
| `-out` / `--output` | str | `'./carm_results'` | Output directory for roofline CSV results (only applies to `roofline` test) |

---

## Config File Format (INI-style)

Positional `config` argument points to a plain-text file:

```
name=my_machine
l1_cache=32
l2_cache=256
l3_cache=8192
```

- Fields: `name`, `l1_cache`, `l2_cache`, `l3_cache` (all in KiB for cache fields)
- Missing fields default to `0` (triggers auto-detection for sizes)
- CLI arguments `--l1_size`, `--l2_size`, `--l3_size`, `--name` override config file values
- Parsed by `read_config()` at [run.py L416–430](../../run.py#L416-L430)

**In `carm.py`**: replaced by JSON config (`config.json`) with full CLI override support.

---

## Execution Flow / Pipeline

```
main()
  ├── argparse.parse_args()
  ├── read_config(args.config)          # optional INI config
  ├── Resolve num_ld / num_st           # from --ld_st_ratio / --only_ld / --only_st
  └── Dispatch:
       ├── args.test in [mixedL1/L2/L3/DRAM] → run_mixed()
       ├── args.test == 'MEM'           → run_memory()
       └── else (FP/L1/L2/L3/DRAM/roofline) → run_roofline()

run_roofline() / run_memory() / run_mixed():
  ├── check_hardware()                  # ISA validation + cache auto-detection
  │    ├── platform.machine() → x86_64 / aarch64 / riscv64
  │    ├── autoconf()                   # x86: compiles+runs config/auto_config/autoconfig.c
  │    │    └── gcc -o autoconfig autoconfig.c && ./autoconfig <freq> <set_freq>
  │    ├── SVE/RVV probe compilation   # ARM/RISCV: compiles+runs vector detection programs
  │    └── Returns: validated isa_set, l1_size, l2_size, l3_size, VLEN, LMUL
  │
  └── for threads in threads_set:
       for isa in isa_set:
         for precision in precision_set:
           ├── Calculate num_reps per cache level
           │    └── Formula: (cache_size_bytes) / (inst_size × num_ld+st × VLEN × LMUL)
           ├── os.system("make -C <dir> clean && make -C <dir> isa=<isa>")
           │    └── Recompiles Bench/Bench C binary for the target ISA
           ├── For each sub-test (L1/L2/L3/DRAM/FP/FP_FMA):
           │    ├── os.system("Bench/Bench -test MEM|FLOPS -num_LD ... -num_rep ...")
           │    │    └── Writes microbenchmark config to shared state / file
           │    ├── subprocess.run(["bin/test", "-threads", ...])
           │    │    └── Executes threaded timing harness; returns CSV: cycles,inner_reps,freq_real,freq_nominal
           │    └── Manually compute GB/s or GFLOP/s in Python
           ├── update_csv() or update_memory_csv()  # append to carm_results/.../name_*.csv
           └── plot_roofline() or matplotlib inline  # optional SVG (--plot)
```

### Performance Calculation (Manual, In-Python)

For x86 ISAs (cycle-based):
```python
data['L1'] = (threads * num_reps * (num_ld+num_st) * mem_inst_size[isa][precision]
              * freq_real * inner_loop_reps) / (cycles * (freq_real/freq_nominal))
```

For non-x86 (time-based):
```python
data['L1'] = (threads * num_reps * (num_ld+num_st) * mem_inst_size[isa][precision]
              * VLEN * LMUL * inner_loop_reps / 1e9) / (time_ms / 1000)
```

These formulas are scattered across `run_roofline()`, `run_memory()`, and `run_mixed()` with no shared abstraction.

---

## Output Files

### Roofline / per-test CSV

- Path: `<output>/roofline/<name>_roofline.csv` (or `<name>_<test_type>.csv`)
- Written by `update_csv()` at [run.py L470](../../run.py#L470)
- Schema (two header rows + data rows):
  ```
  Row 0 (secondary): Name: <n>, L1 Size: <kb>, L2 Size: <kb>, L3 Size: <kb>, ..., L1, L1, L2, L2, L3, L3, DRAM, DRAM, FP, FP, FP FMA, FP_FMA
  Row 1 (primary):   Date, ISA, Precision, Threads, Loads, Stores, Interleaved, DRAM Bytes, FP Inst., GB/s, I/Cycle, GB/s, I/Cycle, ...
  Row N (data):      <values>
  ```
- Appends to existing file without re-writing headers

### Memory Curve CSV

- Path: `./carm_results/memory_curve/<name>_memory_curve.csv`
- Written by `update_memory_csv()` at [run.py L1152](../../run.py#L1152)
- Fixed test sizes: `[2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 600, 768, 1024, ...]` KiB (35 sizes total, hardcoded at [run.py L36](../../run.py#L36))

### SVG Plots

- Roofline: `carm_results/Roofline/<name>_roofline_<date>_<isa>_<prec>_<N>_Threads_<ld>Load_<st>Store_<inst>[_Interleaved].svg`
- Memory curve: `carm_results/memory_curve/<name>_memory_curve_<date>_<isa>_...svg`
- Requires matplotlib + numpy; silently skipped if not installed

---

## Key Differences from `carm.py`

### Architecture

| Aspect | `run.py` | `carm.py` |
|--------|----------|-----------|
| Structure | Monolithic single-file | Modular (`Architecture`, `Benchmarking`, `RunConfig`, `ExecutionInterface`, `CARMContext`) |
| Type hints | None | Full mypy-strict type hints on all new code |
| Test coverage | None | pytest in `refactor_tests/` |
| Config file format | Custom INI (`key=value`) | JSON (`config.json`) |
| Hardware detection | Inline Python in `check_hardware()` | `Architecture` class at `architecture/` with C probes |
| `ExecutionInterface` | Direct `os.system()` / `subprocess.run()` | Abstracted through `ExecutionInterface` (supports cross-compilation/simulation) |
| ISA code generation | Pre-built `Bench/Bench` C binary via `make isa=<isa>` recompilation per test | New `benchmark/generation/` ISA system generating `test_bench/microbenchmarks.h` |
| Timing harness | Separate `bin/test` binary | Unified `test_bench/` Python builder with inline C header generation |
| Performance calculation | Manual per-architecture formulas in Python | Encapsulated in benchmark suite objects |
| Output formats | CSV only (hardcoded two-row schema) | `csv`, `json`, `table` selectable via `--output-format` |
| Output path | Hardcoded `carm_results/roofline/<name>_*.csv` | `carm_results/<test_type>/results.<ext>` or `--output-file` |

### Test Types

| `run.py --test` | `carm.py --test` | Notes |
|-----------------|------------------|-------|
| `roofline` | `roofline` | Full roofline model |
| `FP` | `arithmetic` | Arithmetic peak only |
| `L1`, `L2`, `L3`, `DRAM` | `memory --mem-target L1/L2/L3/DRAM` | Per-level memory bandwidth |
| `MEM` | `memory` (targets all) | Memory bandwidth sweep |
| `mixedL1/L2/L3/DRAM` | `mixed` | Combined memory+FP test |
| *(not present)* | `memory_sweep` | New in `carm.py` |

### ISA Names

| `run.py` | `carm.py` (new system) |
|----------|------------------------|
| `avx512` | `x86_avx512` |
| `avx2` | `x86_avx2` |
| `sse` | `x86_sse` |
| `scalar` | `x86_scalar` |
| `neon` | `arm_neon` |
| `armscalar` | `arm_scalar` |
| `sve` | `arm_sve` |
| `riscvscalar` | `riscv_scalar` |
| `rvv0.7` | `riscv_rvv07` |
| `rvv1.0` | `riscv_rvv` |

### Argument Mapping

| `run.py` | `carm.py` / `Benchmarking` | Notes |
|----------|---------------------------|-------|
| `--test roofline/FP/L1/MEM/mixed*` | `--test arithmetic/memory/roofline/mixed` | Different test names |
| `--isa avx2 ...` (list) | `--isa x86_avx2 ...` (list) | Renamed ISA strings |
| `-p dp sp` (list) | `-d f64` / `-d f32` (DataType enum) | Precision → data type, single value |
| `-t 1 2 4` (list sweep) | `--threads 4` (single value) | No thread sweep in `carm.py` |
| `--inst add` (default) | `--instruction fma` (default) | Default changed to fma |
| `-ops 32768` | `-o/--num-ops 32768` | Same default |
| `-nr/--num_runs 1024` | *(removed)* | Replaced by `--test-time 25.0` (time-based) |
| `-ldst 2` | `--ld-st-ratio 2:1` | Format changed to `N:M` |
| `--only_ld` | `--ld-st-ratio 1:0` | Expressed as ratio |
| `--only_st` | `--ld-st-ratio 0:1` | Expressed as ratio |
| `-fpldst/--fp_ld_st_ratio 1` | `--arith-mem-ratio 2` | Similar concept, different default |
| `--l3_kbytes`, `--dram_kbytes` | `--mem-test-sizes` | Unified per-level size list |
| `--dram_auto` | *(implicit in `auto` sizing)* | `auto` keyword in `--mem-test-sizes` |
| `-tl1`, `-tl2` | *(removed from CLI)* | Internal to `Architecture` detection |
| `-l1/-l2/-l3 <KiB>` | `--caches <sizes...>` | Syntax changed |
| `--freq 2.0` | Handled by `Architecture` | Moved to architecture module |
| `--set_freq` | Handled by `Architecture` | Moved to architecture module |
| `--no_freq_measure` | Handled by `Architecture` | Moved to architecture module |
| `-v 3` (default) | `-v 0` (default) | Default verbosity changed |
| `--name unnamed` | `--name` | Same |
| `--plot` | `--plot` | Same flag, different implementation |
| `-out ./carm_results` | `--output-dir` / `--output-file` | Split into dir vs explicit file |
| *(not present)* | `--output-format csv/json/table` | New in `carm.py` |
| *(not present)* | `--dry-run` | New in `carm.py` |
| *(not present)* | `--sim-cmd` | Cross-compilation simulation |
| *(not present)* | `--compiler` | Custom compiler path |
| *(not present)* | `--emit-config` | Generate template TOML config |

---

## Functionality Dropped / Not Migrated

- **Thread-count sweep**: `run.py` accepted `-t 1 2 4 8` to run all thread counts in one invocation. `carm.py` takes a single `--threads N`.
- **Precision sweep**: `run.py` accepted `-p dp sp`. `carm.py` takes single `--data-type`.
- **Fine-grained DRAM sizing**: `--dram_kbytes` and `--dram_auto` were explicit; replaced by `--mem-test-sizes`.
- **`mixedL1/L2/L3/DRAM` granularity**: Four separate test names → single `--test mixed` with `--mem-target`.
- **Direct `Bench/Bench` binary path**: `run.py` hardcoded `Bench/Bench` and `bin/test` relative to script dir. New system generates and compiles inline.
- **`plot_roofline()` with matplotlib**: SVG output with direct matplotlib in `run.py`; `carm.py` uses `gui/dashboard.py` (Dash/Plotly) via `--plot`.
- **INI config file**: `read_config()` parsing `name=`, `l1_cache=`, etc. Replaced by JSON `config.json`.
- **`--set_freq` / CPU frequency pinning**: Was an integrated feature with root-level x86 frequency setter (`autoconfig.c`). Now separated into `Architecture` module.
- **`mixed` CSV output**: `run_mixed()` called `ut.update_csv()` with a hardcoded path `/home/mixed` (apparent bug at [run.py L1338](../../run.py#L1338)).

---

## Special Behaviors / Workarounds

- **RVV both versions check**: If both `rvv0.7` and `rvv1.0` are specified, `run.py` exits with error ([run.py L1370](../../run.py#L1370)).
- **`auto` ISA + explicit ISAs**: If `auto` is in the ISA list alongside explicit entries, emits a warning and strips to `['auto']` only ([run.py L1374](../../run.py#L1374)).
- **Frequency measurement skipping in multi-test roofline**: After the first sub-test (L1) measures frequency, subsequent sub-tests set `no_freq_measure = 1` to reuse the detected frequency ([run.py L749–750](../../run.py#L749-L750)).
- **SVE VLEN always forced to hardware max**: User-specified `--vector_length` for SVE is overridden to hardware maximum with a warning. Only RVV respects user VLEN within hardware limits.
- **SP VLEN halving**: When `sp` precision is used for RVV/SVE (without `dp`), `vlen` is halved since it represents sp elements ([run.py L1381](../../run.py#L1381)).
- **`bin/test` output CSV format**: `out[0]` = cycles (x86) or time_ms (non-x86), `out[1]` = inner_loop_reps, `out[2]` = freq_real, `out[3]` = freq_nominal (x86 only).
- **DeprecationWarning at import**: `warnings.warn()` fires at module-level import, not just at `main()` call ([run.py L3–7](../../run.py#L3-L7)).
- **`os.system()` for make**: Uses `os.system()` (not `subprocess`) for make invocations — no error checking on compilation failure.
- **Mixed test CSV bug**: `ut.update_csv(name, "/home/mixed", ...)` at [run.py L1338](../../run.py#L1338) uses a hardcoded absolute path `/home/mixed` as the output subdirectory — apparent hardcoded leftover from development.
