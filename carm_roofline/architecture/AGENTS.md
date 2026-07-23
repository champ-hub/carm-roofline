# Architecture Module Documentation

This module handles hardware detection, ISA abstraction, and architecture configuration for the CARM roofline tool.

## Module Overview

The architecture system provides automatic hardware detection and ISA configuration:

- **[architecture.py](architecture.py)** - Core `Architecture` class with argument parsing and auto-detection
- **[detect.py](detect.py)** - Hardware detection orchestration and test probe compilation
- **[x86.py](x86.py), [arm.py](arm.py), [riscv.py](riscv.py)** - Architecture-specific detectors
- **[frequency.py](frequency.py)** - CPU frequency utilities (sysfs-based, x86 primarily)
- **[tests/](tests/)** - C-based hardware probes in hierarchical structure
- **[__init__.py](__init__.py)** - Context variable exports for ExecutionInterface sharing

## Core Architecture Class (architecture.py)

### Architecture Class

Manages hardware configuration with auto-detection and argument parsing via `InsertsArguments` pattern.

**Fields:**
- `isa: list[type[BaseISA]]` - List of ISA classes to use
- `memory_topology: MemoryTopologyLike` - Unified memory topology interface (detected or fallback)
- `frequency: ISAFrequencies` - Per-ISA frequency mapping
- `nominal_frequency: Quantity | None` - TSC-based nominal frequency (x86 only)
- `vector_length: int | None` - Vector register length in bytes (ARM SVE, RISC-V RVV)
- `vector_lmul: int | None` - LMUL multiplier (RISC-V RVV only)
- `memory_topology` always exposes per-level thread sharing and affinity planning

**Initialization Flow:**

```python
def __init__(args):
    1. Call configure_verbosity() from args
    2. Extract num_threads from args for detection
    3. If --isa argument not provided:
         → Call native_detect(num_threads) to auto-detect
    4. Else:
         → Call detect_for_isa(first_isa, num_threads)
    5. Use BaseISA.from_name() for name-to-class lookup
    6. Call _replace_and_warn() to merge detected + user args
    7. If --isa is explicitly provided, validate compatibility via check_isa_compatibility()
    8. Build ISA class list and ISAFrequencies object
```

**Methods:**
- `get_frequency_for_isa(isa_name: str) -> Frequency` - Retrieve frequency for specific ISA
- `_replace_and_warn()` (private) - Merge detection with user args and warn on overrides
- `insert_arguments()` (static) - Add CLI arguments to parser

### Helper Classes

**SimpleMemoryTopology:**
TOML-backed fallback topology used when detailed sysfs topology is unavailable.
It stores explicit cache instance counts per level and uses synthetic CPU placement.

```python
hierarchy = SimpleMemoryTopology(
    [32*KB, 256*KB, 8*MB, 256*GB],
    instances_per_level=[8, 4, 2, 1],
    total_cpus=16,
    smt_degree=2,
    cpu_offset=0,
)
```

The levels are numbered 1..N in the order supplied; the final entry is named `DRAM`
when iterating hierarchy metadata (`__iter__`) and in string representations.

**Key methods:**
- `available_cache_levels() -> list[int]` - Extract numeric hierarchy levels (L1/L2/L3/DRAM → [1,2,3,4])
- `plan_thread_affinity(n_threads, cache_level) -> CacheAwareThreadAffinity` - Returns fallback affinity plan
- `__iter__()` - Iterate over `MemoryLevelInfo` objects

**MemoryTopologyLike (Protocol):**
Common interface implemented by both `MemoryTopology` and `SimpleMemoryTopology`. This allows consumers to use one type and still access:
- memory levels (`__iter__`, `available_cache_levels`)
- thread placement (`plan_thread_affinity` returning `CacheAwareThreadAffinity`)

**MemoryTopology:**
Detailed topology parsed from Linux sysfs (`/sys/devices/system/cpu`). Includes package/core/cache-sharing domains and provides topology-aware affinity planning.
`__iter__()` exposes cache levels followed by a final `DRAM` level using detected system memory.

**ISAFrequencies:**
Per-ISA frequency storage with unit-aware Unit objects (core/units module):
```python
# Construction methods:
ISAFrequencies({"x86_avx2": 2.4*GHz, "x86_avx512": 2.0*GHz})
ISAFrequencies.from_base_frequency(2.4*GHz, [X86AVX2, X86AVX512])  # All get same freq
```
- Stores unit-aware Unit objects (core/units module)
- Access via dictionary interface
- Used for per-ISA frequency variation (x86 AVX-512 downclocking)

### Standalone Functions

No standalone memory hierarchy builder is used in the refactored path. Memory topology objects are created as:
- `MemoryTopology()` for native Linux auto-detection (in `detect.py`)
- `SimpleMemoryTopology(...)` for TOML-provided fallback (`--topology-config`)

`SimpleMemoryTopology` validates level/instance consistency and uses explicit cache instance counts from TOML.

**positive_po2_int(arg: str) -> int:**
Validates power-of-two integers:
- Used for vector_lmul (1, 2, 4, 8)
- Raises ValueError if not power of 2

**check_isa_compatibility(isas: list[str]):**
Validates ISA selection using family-based approach:
1. Ensures all ISAs from same family (no mixing x86 + ARM)
2. Checks `isa.INCOMPATIBLE_ISAS` for within-family conflicts
3. Raises ValueError with descriptive message on incompatibility

### Command-Line Arguments

Via `insert_arguments()` static method:
- `-i/--isa` - ISA names (nargs="*", zero or more, choices from BaseISA.names())
- `--topology-config` - Path to TOML topology file (`total_cpus`/`smt_degree`/`cpu_offset` + `[[cache_levels]]`)
    - If omitted: auto-detected from hardware topology when available
    - Example: `--topology-config cpu-topology.toml`
- `--frequency` - Base frequency (e.g., "2.4GHz", "3200MHz")
- `--vector-length` - Vector register length in bytes
- `--vector-lmul` - LMUL multiplier (choices: 1, 2, 4, 8)

## Hardware Detection (detect.py)

### TestContext

Dataclass for hierarchical test discovery:
```python
@dataclass
class TestContext:
    family: str          # "x86", "arm", "riscv"
    isa: str | None      # Specific ISA name or None

    def find_test(test_name: str) -> Path | None:
        # Searches: ISA-specific → family-specific → generic
```

**Hierarchical Search:**
1. `tests/{family}/{isa}/{test}.c` or `.h`
2. `tests/{family}/{test}.c`
3. `tests/{test}.c`

### DetectedArchitecture

Type-safe container for auto-detected hardware parameters with normalized typed fields.

**Fields:**
- `isa: list[str] | None` - Detected ISA names (e.g., ["x86", "x86_avx2"])
- `memory_topology: MemoryTopology | None` - Detailed topology from sysfs (native Linux), otherwise `None`
- `vector_length: int | None` - Vector register length in bytes
- `frequency: Frequency | None` - Detected frequency
- `frequency_nominal: Frequency | None` - Nominal/base frequency (x86 TSC-based)
- `isa_frequencies: dict[str, Frequency] | None` - Per-ISA frequency mapping
- `arch: str | None` - Architecture string (e.g., "x86_64", "aarch64")
- `vendor: str | None` - CPU vendor string

**Notes:**
- No raw `*_hz` or `caches_kib` fields are exposed on this dataclass in the refactored flow.
- Memory topology is passed through directly as `MemoryTopology` and consumed by `Architecture`.

**Usage:**
```python
# From sysfs: topology is preserved
detected = DetectedArchitecture(
    memory_topology=MemoryTopology()
)
# Result: topology contains cache-sharing sets and affinity planning

# Optional: when no topology available (cross/sim), memory_topology may be None
detected = DetectedArchitecture(memory_topology=None)
```

### Detection Functions

**native_detect(threads: int = 1) -> DetectedArchitecture:**
Auto-detects host platform ISAs via C probes:
```python
def native_detect(threads):
    machine = platform.machine()
    if machine in ["x86_64", "AMD64"]:
        return x86.detect(threads)
    elif machine in ["aarch64", "arm64"]:
        return arm.detect(threads)
    elif machine.startswith("riscv"):
        return riscv.detect(threads)
    else:
        raise ValueError(f"Unsupported platform: {machine}")
```

**detect_for_isa(isa: type[BaseISA], threads: int = 1) -> DetectedArchitecture:**
ISA-driven detection (supports cross-compilation/simulation):
```python
def detect_for_isa(isa, threads):
    family = isa.family  # Get ISA family ("x86", "arm", "riscv")
    ctx = TestContext(family=family, isa=isa.name)

    # Dispatch to family-specific detector
    if family == "x86":
        return x86.detect(threads)
    elif family == "arm":
        return arm.detect(threads)
    elif family == "riscv":
        return riscv.detect(threads)
```

Enables cross-platform detection (e.g., detect RISC-V ISAs on x86 host using QEMU).

**run_test(src: Path, ctx: TestContext, threads: int = 1) -> dict:**
Compiles and runs architecture probe tests:
1. Calls `_ensure_test_built(src, ctx)` to compile if needed
2. Executes via `ExecutionInterface.run()` (respects --sim-cmd)
3. Parses JSON output from test
4. Returns dict with detected features

Probe binaries are cached under a writable temporary directory (`$TMPDIR` fallback), not beside packaged source files.

**run_generic_tests(ctx: TestContext, threads: int = 1) -> DetectedArchitecture:**
Main coordinator for generic detection tests (features, cache, vlen, frequency):

```python
builder = DetectionBuilder()

features_fields = detect_features(ctx)
if features_fields:
    builder.merge_fields("detect_features", features_fields)

cache_fields = detect_cache(ctx)
if cache_fields:
    builder.merge_fields("detect_cache", cache_fields)

vlen_fields = detect_vlen(ctx)
if vlen_fields:
    builder.merge_fields("detect_vlen", vlen_fields)

frequency_fields = detect_frequency(ctx, threads=threads)
if frequency_fields:
    builder.merge_fields("detect_frequency", frequency_fields)

return builder.build()
```

**Cache Detection Priority**:
1. **Native execution** (`sim_cmd is None`): `detect_cache()` builds `MemoryTopology()` from Linux sysfs.
    - Returns `{"memory_topology": MemoryTopology(...)}` when successful.
    - Raises `ValueError` when native sysfs parsing fails.
2. **Non-native execution** (cross/simulated): `detect_cache()` returns `{}`.
    - User must provide `--memory-levels`, `--memory-names` (optional), and `--threads-per-levels`.

Returns merged detection fields such as:
- `memory_topology`
- `isa`
- `vector_length`
- `frequency`, `frequency_nominal`, `isa_frequencies`

### Helper Functions

**detect_cache(ctx: TestContext) -> dict:**
Primary cache detection entrypoint in refactored code.
- Native: parses `MemoryTopology` from `/sys/devices/system/cpu/`
- Non-native: returns empty dict, deferring to CLI-provided fallback topology

**MemoryTopology internals (in `memory.py`):**
- `_parse_cpu_list()` parses sysfs CPU range strings (e.g., `0-3,6`)
- `_parse_size_kb()` parses sysfs cache sizes (`32K`, `8M`, ...)
- Cache instances are registered by level and sharing CPU-set
- `plan_thread_affinity()` returns topology-aware `CacheAwareThreadAffinity`

### Constants

- `ROOT` - Path to architecture module directory
- `GENERIC_PROBE_SRC` - Path to generic probe source files

## Architecture-Specific Detectors

### x86.py

**detect(threads: int = 1) -> dict:**

Returns rich detection results:
```python
{
    "isa": ["x86_scalar", "x86_sse", "x86_avx2", ...],  # Detected ISAs
    "caches": [32*KB, 256*KB, 8*MB],                     # As unit-aware Unit
    "frequency": 2.4*GHz,                                # Real frequency
    "frequency_nominal": 2.5*GHz,                        # TSC-based nominal
    "isa_frequencies": {"x86_avx512": 2.0*GHz, ...},    # Per-ISA map
    "arch": "Intel(R) Xeon(R) ...",                      # CPU model
    "set_frequency": False,                              # Whether to set freq
}
```

**Per-ISA Frequency Detection:**
- Iterates through detected ISAs
- Checks for ISA-specific `frequency.h` headers in `tests/x86/{isa_name}/`
- Runs frequency.c with ISA-specific context if header exists
- Returns `isa_frequencies` dict mapping ISA names to frequencies
- Supports AVX-512 frequency downclocking detection

**Uses:** ExecutionInterface.compile() for building probes.

### arm.py

**detect(threads: int = 1) -> dict:**

Returns:
```python
{
    "isa": ["arm_scalar", "arm_neon", "arm_sve"],  # Based on detected features
    "caches": [32*KB, 256*KB, 4*MB],
    "vector_length": 256,                           # SVE vector length in bytes
    "frequency": 2.0*GHz,                           # Real frequency (wall-clock)
}
```

**ISA List Building:**
- Imports ISA classes from `isa` (ArmScalar, ArmNeon, ArmSVE)
- Always includes `ArmScalar.name` as base
- Conditionally adds `ArmNeon.name` and `ArmSVE.name` based on detected features
- Uses class names to ensure consistency with generation system

**Note:** ARM uses wall-clock timing (no TSC), so only returns `frequency` (not `frequency_nominal`).

### riscv.py

**detect(threads: int = 1) -> dict:**

Returns:
```python
{
    "isa": ["riscv_scalar", "riscv_rvv"],  # Or "riscv_rvv_071" for v0.7.1
    "caches": [32*KB, 256*KB],
    "vector_length": 128,                   # RVV VLEN in bytes
    "vector_lmul": 2,                       # LMUL multiplier
    "frequency": 1.5*GHz,                   # Real frequency (wall-clock)
}
```

**RVV Version Detection:**
- Uses tempfile for compilation tests
- Tries compiling with `-DRISCV_RVV` (v1.0) first
- Falls back to `-DRISCV_RVV_0_7_1` (v0.7.1) if v1.0 fails
- Uses `bench_gen.RISCV_RVV.name` or `bench_gen.RISCV_RVV_071.name` for ISA list

**Helper Function:**
- `_try_compile_rvv()` - Attempts to compile RVV probe with specific flags

**Note:** RISC-V uses wall-clock timing (no TSC), so only returns `frequency` (not `frequency_nominal`).

## Test Probe Structure (tests/)

### Hierarchical Organization

```
tests/
├── frequency.c                    # Universal frequency test (uses family/ISA headers)
├── x86/
│   ├── features.c                 # x86 CPUID feature detection
│   ├── cache.c                    # x86 CPUID cache detection
│   ├── x86_avx512/
│   │   └── frequency.h           # AVX-512 specific frequency detection
│   └── x86_avx2/
│       └── frequency.h           # AVX2 specific frequency detection
├── arm/
│   ├── features.c                 # ARM feature register detection
│   └── vlen.c                     # SVE vector length detection
└── riscv/
    ├── version.c                  # RVV version detection
    └── vlen.c                     # RVV vector length detection
```

**Pattern:** `tests/{family}/{test}.c` for family-specific tests, `tests/{family}/{isa}/{test}.h` for ISA-specific headers.

### Frequency Detection

**Multi-threaded Measurement:**
- Supports `--threads N` argument
- Uses pthread barriers for synchronization
- Each thread measures independently, results aggregated

**ISA-Specific Headers:**
- Example: `tests/x86/x86_avx512/frequency.h` defines AVX-512 specific timing loop
- Each header defines `has_nominal_frequency()` and `calculate_frequencies()`
- x86: Returns both real (wall-clock) and nominal (TSC-based) frequencies
- ARM/RISC-V: Return only real frequency

**Critical Constraint:**
- If an ISA returns both `frequency` and `frequency_nominal` (x86 via TSC), test_bench MUST use TSC timing
- If an ISA returns only `frequency` (ARM/RISC-V), test_bench MUST use wall-clock timing
- Timing method must match frequency measurement method for accurate results

**External Measurement:**
- test_bench NO LONGER measures frequency internally
- Frequency is measured externally and passed via `--freq` argument
- `run_microbenchmarks()` prioritizes `nominal_frequency`, falls back to ISA-specific frequency

## Frequency Utilities (frequency.py)

**set_cpu_frequency():**
Sets maximum CPU frequency via sysfs (`/sys/devices/system/cpu`):
- Requires root/sudo access
- x86-specific implementation
- Used when `set_frequency=True` from detection

**read_cpu_frequencies():**
Reads current CPU frequencies for verification:
- Returns dict mapping CPU ID → frequency
- Used for diagnostics

**_iter_scaling_max_freq_files():**
Helper to iterate over CPU frequency sysfs files.

**Script Mode:**
When run as `python -m architecture.frequency`, prints current frequencies.

## Context Variable Pattern (__init__.py)

### ExecutionInterface Sharing

Uses `contextvars` to avoid parameter threading and circular imports:

```python
import contextvars

_exec_context = contextvars.ContextVar('execution', default=None)

def set_execution_interface(exec_interface: ExecutionInterface):
    return _exec_context.set(exec_interface)

def get_execution_interface() -> ExecutionInterface:
    return _exec_context.get()
```

**Usage in detect.py:**
```python
def run_test(src: Path):
    from . import get_execution_interface  # Lazy import
    exec_iface = get_execution_interface()
    exec_iface.compile(...)
```

**Exports:**
- Exports `Architecture` from `.architecture` (not `.base` - that file doesn't exist)
- Avoids circular imports via lazy imports in detect functions

## Common Workflows

### Adding Support for New Architecture

1. Create `architecture/<arch>.py` module (e.g., `loongarch.py`)
2. Implement `detect(threads: int = 1)` function returning dict with:
    - `isa`: list of supported ISA name strings
    - `caches`: list of unit-aware Quantity objects (in bytes)
    - `vector_length`: vector register length in bytes (or None)
    - `frequency`: detected frequency as a unit-aware Quantity
3. Add architecture-specific probes under `tests/<arch>/`
4. Wire detector into `detect.py`:
   - Add to `native_detect()` platform.machine() dispatch
   - Add to `detect_for_isa()` family dispatch
5. Test: `python -c "from architecture import Architecture; import argparse; args = argparse.Namespace(isa=None, ...); arch = Architecture(args); print(arch.isa)"`

### Cross-Platform Detection

Enable ISA detection on non-native platforms:

```bash
# Detect RISC-V ISAs on x86 host using QEMU
./carm.py benchmark --isa riscv_rvv --compiler riscv64-linux-gnu-gcc --sim-cmd "qemu-riscv64 {binary}"

# Detect ARM SVE on x86 host using SDE
./carm.py benchmark --isa arm_sve --compiler aarch64-linux-gnu-gcc --sim-cmd "sde -sde-arm-sve -- {binary}"
```

The `detect_for_isa()` function uses the ISA's `family` attribute to dispatch to the correct detector, which then uses ExecutionInterface for cross-compilation/simulation.

### Overriding Auto-Detection

Provide explicit values via CLI to override detection:

```bash
# Classic 3-level cache + DRAM
./carm.py benchmark --test arithmetic --memory-levels 32KiB 256KiB 8MiB 16GiB --threads-per-levels 1 2 8 16

# Optane system with explicit levels
./carm.py benchmark --test roofline \
  --memory-levels 32KiB 256KiB 8MiB 32GiB \
    --memory-names L1 L2 L3 Optane \
    --threads-per-levels 1 2 8 16

# Override frequency
./carm.py benchmark --test arithmetic --isa arm_neon --frequency 2.4GHz

# Override vector length (SVE/RVV)
./carm.py benchmark --test arithmetic --isa riscv_rvv --vector-length 128 --vector-lmul 4
```

The `_replace_and_warn()` method merges user values with detection and warns about conflicts.

### Memory Topology Examples

**Auto-detection (default):**
```bash
./carm.py benchmark --test arithmetic
# Detects native platform topology from /sys
```

**Explicit cache+memory levels (fallback mode):**
```bash
./carm.py benchmark --test memory \
    --memory-levels 32KiB 256KiB 8MiB 16GiB \
    --threads-per-levels 1 2 8 16
# Uses exactly the levels provided above (no inferred extra level)
```

**Custom names with explicit per-level sharing:**
```bash
./carm.py benchmark --test roofline \
  --memory-levels 32KiB 256KiB 8MiB 32GiB \
    --memory-names L1 L2 LLC DRAM \
    --threads-per-levels 1 2 8 16
```

**Error handling:**
```bash
# This fails: --threads-per-levels is required when --memory-levels is provided
./carm.py benchmark --test arithmetic \
    --memory-levels 32KiB 256KiB 8MiB 16GiB
    # Error: --threads-per-levels is required when providing --memory-levels
```

## Integration with Other Modules

### Dependencies
- Removed: previously used `pint` for unit-aware parsing; no longer a dependency
- **isa** - ISA classes (for name consistency and compatibility checking)
- **exec_interface** - Cross-compilation/simulation support via ExecutionInterface
- **output_utils** - Logging (configure_verbosity, detail, debug, warn)

### Used By
- **carm.py** - Creates Architecture object from args
- **context.py** - Architecture is a field in CARMContext
- **benchmark/interface.py** - Accesses ISA list and frequencies for generation

## See Also

- **[../benchmark/generation/README.md](../benchmark/generation/README.md)** - ISA class definitions
- **[../test_bench/README.md](../test_bench/README.md)** - Timing method constraints
- **[../exec_interface.py](../exec_interface.py)** - Execution abstraction

---

**When modifying this module:** Update this documentation to reflect changes in detection logic, new architectures, or test probe structure.
