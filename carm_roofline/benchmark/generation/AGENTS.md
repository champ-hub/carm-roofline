# Benchmark Generation Module Documentation

This module provides ISA-specific code generation for CARM microbenchmarks, implementing a flexible abstraction layer for x86, ARM, and RISC-V architectures.

## Module Overview

The generation system separates ISA-specific details from benchmark logic:

- **[isa.py](isa.py)** - Base `BaseISA` class and inline assembly generation
- **[x86.py](x86.py), [arm.py](arm.py), [riscv.py](riscv.py)** - ISA implementations
- **[parameters.py](parameters.py)** - Benchmark parameter classes
- **[code_gen/](code_gen/)** - Abstract instruction/register/operation/data type models
- **[__init__.py](__init__.py)** - Exports and ISA registry

## ISA System Architecture

### BaseISA Class (isa.py)

Abstract base class for all ISA implementations, handling C code generation with inline assembly.

**Class Attributes:**
- `family: str` - ISA family ("x86", "arm", "riscv")
- `name: str` - Unique ISA identifier (e.g., "x86_avx2", "arm_sve", "riscv_rvv")
- `INNER_LOOP_LABEL: str` - Inner loop label for assembly (default: "2")
- `OUTER_LOOP_LABEL: str` - Outer loop label for assembly (default: "1")
- `unroll_loop: bool` - Whether to unroll loops (default: False)
- `instruction_limit: int` - Maximum instructions per benchmark (default: 2048)

**Abstract Properties (must be implemented by subclasses):**
- `ControlInsts` - Control flow instructions (branches, load-imm, add-imm)
- `TypedRegisterSets` - Register sets per data type (f32, f64)
- `TypedInstructions` - Instruction formats per data type and operation

**Methods:**

**`from_architecture(arch: Architecture) -> BaseISA`** (class method):
Factory method for ISA instantiation with architecture-specific parameters:
```python
# Default implementation (scalar ISAs):
return cls()

# Vector ISAs (ARM SVE, RISC-V RVV) override to use vector_length:
return cls(vlen_bits=arch.vector_length * 8, lmul=arch.vector_lmul)
```

**`setup_assembly(data_type: DataType) -> str`**:
Returns ISA-specific setup instructions executed before benchmark:
```python
# x86/ARM scalar: empty string
# RISC-V RVV: vsetvli configuration
```

**`format_iasm_input(var: str) -> str`**:
Formats inline assembly input variables (ISA-specific syntax):
```python
# x86: f"%{var}"
# ARM/RISC-V: f"{var}"
```

**`ops_per_inst(data_type: DataType, op: Operation) -> int`**:
Returns operations per instruction for throughput calculation:
```python
# Scalar: 1 (or 2 for FMA/MAD)
# Vector: (vector_bytes // data_type.bytes()) * ops_per_element
```

**`bytes_per_inst(data_type: DataType) -> int`**:
Returns bytes transferred per memory instruction:
```python
# Scalar: data_type.bytes()
# Vector: vector register size in bytes
```

**`max_unique_offsets(data_type: DataType) -> int`**:
Returns maximum unique memory offsets (for array sizing):
```python
# Typically: num_general_purpose_registers - reserved_registers
```

**`offset_increment(data_type: DataType) -> int`**:
Returns offset increment per memory instruction:
```python
# Typically: bytes_per_inst(data_type)
# SVE: overridden to handle predicated memory
```

**Generation Methods:**

**`generate_arithmetic_benchmark(params: ArithmeticBenchmarkParams) -> MicrobenchmarkFunctionSpec`**:
Generates arithmetic benchmark with specified operation and repetition count.

**`generate_memory_benchmark(params: MemoryBenchmarkParams) -> MicrobenchmarkFunctionSpec`**:
Generates memory benchmark with load/store pattern and repetition count.

**Loop and Memory Helpers (BaseISA):**
- **`_split_loop(num_ops: int, max_loop_size: int) -> LoopSplitConfig`** - Shared inner/outer loop splitting helper.
- **`_validate_memory_size(...) -> SizeInfo`** - Validates memory sizes and derives working set sizes.
- **`_calculate_loop_configuration(...) -> LoopConfig`** - Computes loop sizing and repeat configuration.
- **`_generate_memory_instruction_stream(...) -> list[str]`** - Builds per-repeat memory instruction stream.
- **`_extract_thread_sharing_config(...) -> ThreadConfig`** - Cache-level thread sharing selection.

### Helper Dataclasses (isa.py)

**`LoopSplitConfig`** - Configuration for splitting operations into inner/outer loops:
```python
@dataclass
class LoopSplitConfig:
    instance_inner_loop: bool  # Whether an inner loop should be instanced
    inner_repeats: int         # Number of repeats in inner loop
    outer_repeats: int         # Number of repeats in outer loop
    num_iterations: int        # Number of inner loop iterations
```

**`SizeInfo`** - Derived sizes for memory benchmark generation:
```python
@dataclass
class SizeInfo:
    repeats: int                    # Number of repeats per benchmark
    bytes_per_repeat: int           # Bytes transferred per repeat
    actual_working_set_size: int    # Total working set size in bytes
```

**`LoopConfig`** - Loop configuration for memory benchmark generation:
```python
@dataclass
class LoopConfig:
    block_size_offsets: int         # Number of unique memory offsets per block
    bytes_per_block: int            # Total bytes per block
    mem_insts_per_loop: int         # Memory instructions per loop
    max_loop_size: int              # Max repeats per loop
    num_iterations: int             # Number of inner loop iterations
    instance_inner_loop: bool       # Whether to instance inner loop
    inner_repeats: int              # Repeats in inner loop
    outer_repeats: int              # Repeats in outer loop
    loop_instruction_limit: int     # Maximum instructions per loop body
```

**`ThreadConfig`** - Thread/cache sharing configuration for memory benchmarks:
```python
@dataclass
class ThreadConfig:
    cache_level: str                 # Cache level (L1, L2, L3, DRAM)
    threads_sharing_cache: int       # Number of threads sharing this cache level
```

### InlineASM Class (isa.py)

Represents inline assembly blocks in C code:

```python
@dataclass
class InlineASM:
    @dataclass
    class Input:
        c_variable: str    # C variable name
        asm_variable: str  # Assembly variable placeholder

    body: list[str]        # Assembly instructions
    inputs: list[Input]    # Input variable mappings
    clobbers: list[str]    # Clobbered registers/memory

    def format() -> str:
        # Returns formatted inline assembly string

    def as_function_body() -> str:
        # Wraps in function with variable declarations
```

**Example:**
```python
iasm = InlineASM(
    body=["vfmadd231pd %%ymm0, %%ymm1, %%ymm2"],
    inputs=[InlineASM.Input("data", "r")],
    clobbers=["ymm0", "ymm1", "ymm2"],
)
```

### Helper Functions

**`_add_if(condition: bool, items: list, item) -> list`**:
Conditionally adds item to list (used for assembly generation).

## ISA Implementations

### x86.py

**ISA Hierarchy:**
```
BaseX86 (common x86 configuration)
├── X86Scalar (BITS = 64, scalar double/float)
├── X86SSE (BITS = 128, SSE registers)
├── X86AVX (BITS = 256, AVX registers)
├── X86AVX2 (BITS = 256, AVX2 optimizations)
└── X86AVX512 (BITS = 512, AVX-512 registers)
```

**Class Attributes:**
- `BITS: int` - Register width in bits (determines vector operations per instruction)
- `family = "x86"`
- `name` - Unique per subclass ("x86_scalar", "x86_sse", "x86_avx2", "x86_avx512")

**Register Management:**
- General purpose: `%rax`, `%rbx`, `%rcx`, etc. (used for pointers/counters)
- Vector registers: `%xmm0-%xmm15` (SSE), `%ymm0-%ymm15` (AVX), `%zmm0-%zmm31` (AVX-512)
- Reserved: `%rsp` (stack pointer), `%rbp` (frame pointer)

**Instruction Formats:**
```python
TypedInstructions = {
    DataType.f32: {
        ArithmeticOperation.add: "vaddps {src1}, {src2}, {dst}",
        ArithmeticOperation.mul: "vmulps {src1}, {src2}, {dst}",
        ArithmeticOperation.fma: "vfmadd231ps {src1}, {src2}, {dst}",
    },
    DataType.f64: { ... }, # Similar with 'pd' suffix
}
```

**Control Instructions:**
- Branch: `"jg {label}"`
- Load immediate: `"movq ${imm}, {dst}"`
- Add immediate: `"addq ${imm}, {dst}"`

### arm.py

**ISA Hierarchy:**
```
BaseArm (common ARM configuration)
├── ArmScalar (scalar double/float)
├── ArmNeon (NEON SIMD, 128-bit)
└── ArmSVE (Scalable Vector Extension, variable width)
```

**Class Attributes:**
- `family = "arm"`
- `name` - Unique per subclass ("arm_scalar", "arm_neon", "arm_sve")

**ArmSVE Specifics:**
```python
class ArmSVE(BaseArm):
    def __init__(self, vlen_bits: int):
        self.vlen_bits = vlen_bits  # Vector length in bits
        self.vector_bytes = vlen_bits // 8
```

**Predicate Registers (SVE):**
- `setup_assembly()` configures predicate registers: `"ptrue p0.d"` or `"ptrue p0.s"`
- Memory operations use predicates: `ld1d {z0.d}, p0/z, [{ptr}]`

**Instruction Formats:**
```python
# NEON:
ArithmeticOperation.add: "fadd {dst}.2d, {src1}.2d, {src2}.2d"

# SVE:
ArithmeticOperation.add: "fadd {dst}.d, p0/m, {src1}.d, {src2}.d"
```

**Methods:**
- `offset_increment(data_type)` - Overridden in SVE to handle predicate granularity
- `max_unique_offsets(data_type)` - Overridden in SVE to account for predicate registers

### riscv.py

**ISA Hierarchy:**
```
BaseRISCV (common RISC-V configuration)
├── RISCVScalar (scalar double/float)
├── RISCV_RVV_071 (RVV v0.7.1, older spec)
└── RISCV_RVV (RVV v1.0, current spec)
```

**Class Attributes:**
- `family = "riscv"`
- `name` - Unique per subclass ("riscv_scalar", "riscv_rvv_071", "riscv_rvv")

**RVV Specifics:**
```python
class RISCV_RVV(BaseRISCV):
    def __init__(self, vlen_bits: int, lmul: int = 1):
        self.vlen_bits = vlen_bits
        self.lmul = lmul
        self.vector_bytes = (vlen_bits * lmul) // 8
```

**VSETVLI Configuration:**
- `setup_assembly()` returns `vsetvli` instruction:
  ```python
  # v1.0: "vsetvli zero, zero, e64, m2, ta, ma"
  # v0.7.1: "vsetvli zero, zero, e64m2"
  ```

**TYPE_TO_VSETVL Mapping (v0.7.1):**
```python
TYPE_TO_VSETVL = {
    DataType.f32: "e32m1",
    DataType.f64: "e64m1",
}
```

**Instruction Formats:**
```python
# Scalar:
ArithmeticOperation.add: "fadd.d {dst}, {src1}, {src2}"

# RVV:
ArithmeticOperation.add: "vfadd.vv {dst}, {src1}, {src2}"
```

**Methods:**
- `from_architecture(arch)` - Extracts `vlen_bits` and `lmul` from Architecture

## Code Generation Abstractions (code_gen/)

### instruction.py

**Instruction Hierarchy:**

```
_Instruction (abstract base)
├── LoadImm - Load immediate value
├── AddImm - Add immediate to register
├── Branch - Conditional/unconditional branch
├── Memory - Memory access
│   ├── LoadWord - Load from memory
│   └── StoreWord - Store to memory
└── Arithmetic - Arithmetic operations
    ├── BinaryOp - Two-operand arithmetic
    └── TernaryOp - Three-operand arithmetic (FMA)
```

**Memory Addressing Modes:**
```python
class Memory(_Instruction):
    class AddressingMode(Enum):
        ptr_offset = "ptr_offset"  # [ptr + offset]
        ptr_only = "ptr_only"      # [ptr]
```

**MemoryModeTracker:**
Validates consistent memory addressing mode usage:
```python
tracker = MemoryModeTracker()
tracker.require_mode(AddressingMode.ptr_offset)  # Lock to ptr_offset
tracker.check(AddressingMode.ptr_offset)  # OK
tracker.check(AddressingMode.ptr_only)  # Raises ValueError
```

**escape_for_inline_asm(asm_str: str) -> str:**
Escapes `%` characters for inline assembly (x86 specific):
```python
escape_for_inline_asm("movq %rax, %rbx")  # Returns "movq %%rax, %%rbx"
```

### register.py

**RegisterSet:**
Manages register allocation with name templates and ranges:
```python
# x86 vector registers:
RegisterSet("%%ymm{}", [(0, 16)])  # ymm0-ymm15

# ARM scalar registers:
RegisterSet("d{}", [(0, 32)])  # d0-d31

# Multiple ranges:
RegisterSet("%%xmm{}", [(0, 8), (12, 16)])  # xmm0-7, xmm12-15
```

**Methods:**
- `allocate(n: int)` - Allocates n registers, returns list of names
- `free(name: str)` - Frees a register
- `available()` - Returns count of available registers

**HelperRegisterSet:**
Disjoint set for counters and pointers (prevents conflicts):
```python
helpers = HelperRegisterSet("%%r{}", [(8, 12)])  # r8-r11 for helpers
regs = RegisterSet("%%r{}", [(0, 8)])  # r0-r7 for benchmarks
# No overlap between helper and benchmark registers
```

### operation.py

**Operation Types:**

Operations are split into two distinct enums for type safety:

```python
class ArithmeticOperation(Enum):
    add = auto()   # Addition (1 op)
    mul = auto()   # Multiplication (1 op)
    div = auto()   # Division (1 op)
    fma = auto()   # Fused multiply-add (2 ops)

class MemoryOperation(Enum):
    ld = auto()    # Load from memory
    st = auto()    # Store to memory
```

**Type alias:** `Operation = ArithmeticOperation | MemoryOperation`

Used as dict keys in `TypedInstructions`. The separate types enable compile-time type checking and prevent mixing arithmetic and memory operations.

### data_type.py

**DataType Enum:**
```python
class DataType(Enum):
    # Floating-point
    f32 = "f32"
    f64 = "f64"

    # Integer (for future use)
    i8 = "i8"
    i16 = "i16"
    i32 = "i32"
    i64 = "i64"
```

**Methods:**
- `bytes() -> int` - Returns size in bytes
- `bits() -> int` - Returns size in bits
- `to_c_type() -> str` - Returns C type name ("float", "double")
- `check_validity(dt) -> None` (static) - Validates DataType value

Used as dict keys in `TypedRegisterSets` and `TypedInstructions`.

## Parameter Classes (parameters.py)

### BenchmarkParams

Base class for all benchmark parameters:
```python
@dataclass
class BenchmarkParams:
    data_type: DataType  # f32 or f64
    # Note: NO test_id field (removed from earlier design)
```

### ArithmeticBenchmarkParams

Parameters for arithmetic benchmarks:
```python
@dataclass
class ArithmeticBenchmarkParams(BenchmarkParams):
    operation: ArithmeticOperation  # ADD, MUL, FMA, etc.
    num_ops: int                    # Number of operations
```

**Note:** The `operation` field is typed to `ArithmeticOperation` only (not generic `Operation`). Memory operations are rejected at type-check time.

### MemoryBenchmarkParams

Parameters for memory benchmarks:
```python
@dataclass
class MemoryBenchmarkParams(BenchmarkParams):
    load_store_ratio: LoadStoreRatio
    size_per_thread: Bytes
    memory_level_name: str
    layout_mode: MemoryLayoutMode = MemoryLayoutMode.split
```

`num_ld` and `num_st` are derived from `load_store_ratio` during validation.

### MemoryLayoutMode

Typed memory layout selection for memory benchmarks:
```python
class MemoryLayoutMode(Enum):
    single = "single"  # one shared read/write array
    split = "split"    # separate read and write arrays
```

### BenchParamError

Exception for invalid parameters:
```python
raise BenchParamError("num_ops must be positive")
```

## Module Exports (__init__.py)

### ISA Registry

**ISA Registry:**

ISAs register themselves via `register=True` on the class statement. The registry is accessible through `BaseISA`:

- `BaseISA.from_name(name)` → Look up an ISA class by its name string
- `BaseISA.names()` → List all registered ISA names
- `BaseISA.all()` → Tuple of all registered ISA classes

**INCOMPATIBLE_ISAS:**
Set of ISA pairs that cannot be used together:
```python
INCOMPATIBLE_ISAS = {
    frozenset({RISCV_RVV_071, RISCV_RVV}),  # Class references, not strings
}
```

### Other Exports

- `DataType` enum
- `Operation` enum
- Parameter classes: `BenchmarkParams`, `ArithmeticBenchmarkParams`, `MemoryBenchmarkParams`, `BenchParamError`
- `MemoryLayoutMode` enum

## Adding a New ISA

### Step-by-Step Guide

1. **Create ISA class** in appropriate file (x86.py, arm.py, riscv.py, or new file):
   ```python
   class X86AVX10(BaseX86):
       name = "x86_avx10"
       BITS = 512  # Or appropriate width
   ```

2. **Set family attribute** (usually inherited from base):
   ```python
   # Inherited from BaseX86:
   family = "x86"
   ```

3. **Define ControlInsts** (control flow instructions):
   ```python
   ControlInsts = ControlInstFormats(
       branch=Branch("jg {label}"),
       load_imm=LoadImm("movq ${imm}, {dst}"),
       add_imm=AddImm("addq ${imm}, {dst}"),
   )
   ```

4. **Define TypedRegisterSets** (registers per data type):
   ```python
   TypedRegisterSets = {
       DataType.f32: RegisterSet("%%zmm{}", [(0, 32)]),
       DataType.f64: RegisterSet("%%zmm{}", [(0, 32)]),
   }
   ```

5. **Define TypedInstructions** (instruction formats):
   ```python
   TypedInstructions = {
       DataType.f32: {
           ArithmeticOperation.add: "vaddps {src1}, {src2}, {dst}",
           ArithmeticOperation.fma: "vfmadd231ps {src1}, {src2}, {dst}",
           MemoryOperation.ld: "movss {off}({ptr}), {reg}",
           MemoryOperation.st: "movss {reg}, {off}({ptr})",
       },
       DataType.f64: {
           ArithmeticOperation.add: "vaddpd {src1}, {src2}, {dst}",
           ArithmeticOperation.fma: "vfmadd231pd {src1}, {src2}, {dst}",
           MemoryOperation.ld: "movsd {off}({ptr}), {reg}",
           MemoryOperation.st: "movsd {reg}, {off}({ptr})",
       },
   }
   ```

6. **Implement from_architecture()** if ISA needs architecture params:
   ```python
   @classmethod
   def from_architecture(cls, arch: Architecture):
       if arch.vector_length:
           return cls(vlen_bits=arch.vector_length * 8)
       return cls()
   ```

7. **Override methods** if needed:
   ```python
   def setup_assembly(self, data_type: DataType) -> str:
       return "vsetvli zero, zero, e64, m2"  # Example for RVV

   def ops_per_inst(self, data_type: DataType, op: Operation) -> int:
       base = super().ops_per_inst(data_type, op)
       return base * self.lmul  # Example for RVV LMUL
   ```

8. **Register the ISA**: Add `register=True` to the class definition:
   ```python
   class X86AVX10(BaseX86, register=True):
       name = "x86_avx10"
       BITS = 512
   ```

   No changes to `__init__.py` are needed — registration is automatic.

   Add to `INCOMPATIBLE_ISAS` if needed:
   ```python
   INCOMPATIBLE_ISAS = {
       ...,
       frozenset({"x86_avx512", "x86_avx10"}),  # Example
   }
   ```

9. **Create detection probe** (optional) in `architecture/tests/`:
   ```c
   // architecture/tests/x86/x86_avx10/features.c
   // Detection code using CPUID or other methods
   ```

10. **Update architecture detector** in `architecture/{arch}.py`:
    ```python
    # In x86.py detect() function:
    if has_avx10_feature:
        isas.append("x86_avx10")
    ```

### Testing New ISA

```bash
# Generate benchmarks only (dry-run)
./run2.py --isa x86_avx10 --test arithmetic --dry-run -v 4

# Check generated code
cat test_bench/microbenchmarks.h

# Run full benchmark
./run2.py --isa x86_avx10 --test arithmetic --num_ops 100
```

## Common Patterns

### Type-Safe Instruction Retrieval

```python
isa = X86AVX2()
add_inst = isa.TypedInstructions[DataType.f64][ArithmeticOperation.add]
# Returns: "vaddpd {src1}, {src2}, {dst}"

# Memory instruction retrieval:
mem_inst = isa.bench_instructions[DataType.f32][MemoryOperation.ld]
# Returns: "movss {off}({ptr}), {reg}"
```

### Register Allocation Pattern

```python
regs = isa.TypedRegisterSets[DataType.f32]
allocated = regs.allocate(3)  # Returns ["%%ymm0", "%%ymm1", "%%ymm2"]
# ... use registers ...
for reg in allocated:
    regs.free(reg)
```

### Assembly Generation Pattern

```python
iasm = InlineASM(
    body=[inst.fmt(dst=d, src1=s1, src2=s2)
          for inst, d, s1, s2 in zip(insts, dsts, src1s, src2s)],
    inputs=[InlineASM.Input("data", "r")],
    clobbers=allocated_regs,
)

function_code = iasm.as_function_body()
```

## Integration with Other Modules

### Dependencies
- **pydantic** or **dataclasses** - Parameter class definitions
- **enum** - Operation and DataType enums
- **abc** - Abstract base classes

### Used By
- **[../benchmark.py](../benchmark.py)** - Uses parameter classes
- **[../interface.py](../interface.py)** - Calls `generate_*_benchmark()` methods
- **[../suites/](../suites/README.md)** - Suite classes instantiate ISAs via `from_architecture()`
- **[../../architecture/](../../architecture/README.md)** - ISA class names used for detection

## See Also

- **[../README.md](../README.md)** - Benchmark module overview
- **[../suites/README.md](../suites/README.md)** - Suite system using generated benchmarks
- **[../../architecture/README.md](../../architecture/README.md)** - ISA detection and configuration
- **[../../test_bench/README.md](../../test_bench/README.md)** - Execution of generated benchmarks

---

**When modifying this module:** Update this documentation when adding new ISAs, changing abstractions, or modifying the generation interface. Keep ISA registry exports synchronized with actual implementations.
