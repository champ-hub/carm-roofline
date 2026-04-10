# Benchmark Suites Module Documentation

This module provides the suite system for organizing and managing collections of benchmarks per ISA.

## Module Overview

The suite system organizes benchmarks by type and provides utilities for peak performance extraction:

- **[base.py](base.py)** - `ISABenchmarkSuite` abstract base class
- **[arithmetic.py](arithmetic.py)** - `ArithmeticBenchmarkSuite` for FP operations
- **[memory.py](memory.py)** - `MemoryBenchmarkSuite` for cache hierarchy tests
- **[roofline.py](roofline.py)** - `RooflineBenchmarkSuite` combining arithmetic + memory
- **[mixed.py](mixed.py)** - `MixedBenchmarkSuite` for arithmetic+memory stress (stub)

## Suite System Purpose

Suites serve as:
1. **Organizational containers** - Group benchmarks by ISA and test type
2. **Generation orchestrators** - Implement `generate()` class method to create benchmarks
3. **Result aggregators** - Provide methods to extract peak performance metrics
4. **Merging utilities** - Combine results from multiple runs

## Base Class (base.py)

### ISABenchmarkSuite

Abstract base class for all suite types:

```python
@dataclass
class ISABenchmarkSuite(ABC):
    isa_name: str
    benchmarks: dict[str, Benchmark] = field(default_factory=dict)
```

**Abstract Methods:**

**`generate(context: CARMContext, isa_name: str) -> ISABenchmarkSuite`** (classmethod):
Generates suite with benchmarks for the given ISA name. Must be implemented by subclasses.

### Core Methods

**`add_benchmark(name: str, benchmark: BaseBenchmark) -> None`**:
Adds benchmark to suite under the given key.

**`get_arithmetic_benchmarks() -> dict[str, ArithmeticBenchmark]`**:
Returns arithmetic benchmarks keyed by name (filters by type).

**`get_memory_benchmarks() -> dict[str, MemoryBenchmark]`**:
Returns memory benchmarks keyed by name (filters by type).

**`get_mixed_benchmarks() -> dict[str, MixedBenchmark]`**:
Returns mixed benchmarks keyed by name (filters by type).

**`all_results_populated() -> bool`**:
Checks if all benchmarks have results populated (for validation).

**`merge(other: ISABenchmarkSuite) -> None`**:
Merges benchmarks from another suite with the same ISA:
```python
suite1.merge(suite2)  # Adds suite2's benchmarks to suite1
```

**`merge_suites(*suite_dicts: dict[str, ISABenchmarkSuite]) -> dict[str, ISABenchmarkSuite]`** (staticmethod):
Merges multiple ISA-to-suite dicts into one:
```python
merged = ISABenchmarkSuite.merge_suites(suites_dict1, suites_dict2)
```

### Peak Performance Properties

**`peak_arithmetic_gops: float | None`**:
Property returning peak GOPS across all arithmetic benchmarks:
```python
suite.peak_arithmetic_gops  # Returns max gops or None if no results
```

**`peak_bandwidth_gb_per_s: float | None`**:
Property returning peak bandwidth across all memory benchmarks:
```python
suite.peak_bandwidth_gb_per_s  # Returns max bandwidth or None
```

### Peak Extraction Methods

**`get_peak_gops(operation: ArithmeticOperation | None = None) -> float | None`**:
Returns peak GOPS for specific operation (or all if None):
```python
suite.get_peak_gops(ArithmeticOperation.fma)  # Peak FMA performance
suite.get_peak_gops()                          # Overall peak arithmetic
```

**`get_peak_bandwidth(cache_level: str | None = None) -> float | None`**:
Returns peak bandwidth for specific cache level (or all if None):
```python
suite.get_peak_bandwidth("L1")   # L1 bandwidth
suite.get_peak_bandwidth()       # Overall peak bandwidth
```

**`get_peak_bandwidth_by_level() -> dict[str, float]`**:
Returns dict mapping cache levels to peak bandwidths:
```python
{"L1": 250.5, "L2": 120.3, "L3": 45.2, "DRAM": 25.8}
```

## Arithmetic Suite (arithmetic.py)

### ArithmeticBenchmarkSuite

Generates pure floating-point arithmetic benchmarks.

**Generation:**

```python
@classmethod
def generate(cls, context: CARMContext, isa_name: str) -> ArithmeticBenchmarkSuite:
    isa_class = next(c for c in context.architecture.isa if c.name == isa_name)
    isa_instance = isa_class.from_architecture(context.architecture)
    suite = cls(isa_name=isa_instance.name)

    # Generate benchmarks for selected operation
    operation = context.benchmarking.instruction or ArithmeticOperation.fma
    for data_type in [context.benchmarking.precision]:
        params = ArithmeticBenchmarkParams(
            operation=operation,
            num_ops=context.benchmarking.num_ops,
            data_type=data_type,
            num_threads=1,
        )
        spec = isa_instance.generate_arithmetic_benchmark(params)
        benchmark = ArithmeticBenchmark(params=params, spec=spec)
        suite.add_benchmark(benchmark)

    return suite
```

**Key Features:**
- Iterates over operations (all or specified via `--instruction`)
- Uses configured precision (f32/f64)
- Calls ISA's `generate_arithmetic_benchmark()` for code generation
- Creates `ArithmeticBenchmark` objects with params and spec

**Additional Methods:**

**`get_gops_by_operation() -> dict[ArithmeticOperation, float]`**:
Returns dict mapping operations to their peak GOPS:
```python
{ArithmeticOperation.add: 150.2, ArithmeticOperation.mul: 145.8, ArithmeticOperation.fma: 290.5}
```

## Memory Suite (memory.py)

### MemoryBenchmarkSuite

Generates memory bandwidth benchmarks targeting specific cache levels.

**Generation:**

```python
@classmethod
def generate(cls, context: CARMContext, isa_name: str) -> MemoryBenchmarkSuite:
    isa_class = next(c for c in context.architecture.isa if c.name == isa_name)
    isa_instance = isa_class.from_architecture(context.architecture)
    suite = cls(isa_name=isa_instance.name)

    # Generate all levels from topology in hierarchy order
    for level_idx in context.architecture.memory_topology.available_cache_levels():
        params = MemoryBenchmarkParams(
            data_type=context.benchmarking.data_type,
            thread_affinity=thread_affinity.cpu_ids,
            load_store_ratio=context.benchmarking.ld_st_ratio,
            size_per_thread=size_per_thread,
            memory_level_name=level_name,
            layout_mode=(MemoryLayoutMode.single if first_generated else MemoryLayoutMode.split),
        )
        spec = isa_instance.generate_memory(params, context)
        benchmark = MemoryBenchmark(
            params=params,
            spec=spec,
            cache_level=level_name,
        )
        suite.add_benchmark(benchmark.name, benchmark)

    return suite
```

**Key Features:**
- Targets all memory levels exposed by topology iteration (including final-level `DRAM` when provided)
- Uses 80% of per-thread available size for non-final targets
- Uses exactly 2x the previous level's per-thread size for the final target level
- Configures load/store ratio via `--ld_st_ratio`
- Uses typed `MemoryLayoutMode` policy: first generated level (after `mem_target` filtering) uses `single`, all later
    generated levels use `split`
- Suppresses lower-level-fit warnings for DRAM targets to avoid false positives
- Assigns topology-derived `cache_level` names to each benchmark

**Additional Methods:**

**`get_benchmarks_by_cache_level(level: str) -> list[MemoryBenchmark]`**:
Returns benchmarks for specific cache level:
```python
l1_benchmarks = suite.get_benchmarks_by_cache_level("L1")
```

**`get_peak_bandwidth_by_level() -> dict[str, float]`**:
Returns peak bandwidth for each cache level tested:
```python
{"L1": 250.5, "L2": 120.3, "L3": 45.2, "DRAM": 25.8}
```

## Roofline Suite (roofline.py)

### RooflineBenchmarkSuite

Combines arithmetic and memory suites for roofline analysis.

**Generation:**

```python
@classmethod
def generate(cls, context: CARMContext, isa_name: str) -> RooflineBenchmarkSuite:
    suite = cls(isa_name=isa_name)

    # Generate arithmetic benchmarks
    arith_suite = ArithmeticBenchmarkSuite.generate(context, isa_name)
    suite.merge(arith_suite)

    # Generate memory benchmarks for all cache levels
    mem_suite = MemoryBenchmarkSuite.generate(context, isa_name)
    suite.merge(mem_suite)

    return suite
```

**Key Features:**
- Delegates to ArithmeticBenchmarkSuite and MemoryBenchmarkSuite
- Merges both into single suite
- Provides complete view of performance (compute + memory)

**Additional Methods:**

**`compute_ridge_points() -> dict[str, float]`**:
Computes ridge points for each cache level:
```python
{
    "L1": 0.8,    # GOPS / GB/s
    "L2": 2.5,
    "L3": 6.4,
    "DRAM": 11.6,
}
```

Ridge point = `peak_gops / peak_bandwidth` (arithmetic intensity threshold)

## Mixed Suite (mixed.py)

### MixedBenchmarkSuite

Generates combined arithmetic+memory stress tests (currently stub implementation).

**Status:** Placeholder for future mixed workload benchmarks.

**Planned Features:**
- Configurable arithmetic:memory ratio via `--arith_mem_ratio`
- Simultaneous arithmetic and memory operations
- Stress test for sustained performance under mixed load

## Integration with Pipeline

### Usage in interface.py

```python
# In generate_microbenchmarks():
suite_class_map = {
    TestType.ARITHMETIC: ArithmeticBenchmarkSuite,
    TestType.MEMORY: MemoryBenchmarkSuite,
    TestType.ROOFLINE: RooflineBenchmarkSuite,
    TestType.MIXED: MixedBenchmarkSuite,
}

suite_class = suite_class_map[context.benchmarking.test]
suite = suite_class.generate(context, isa_name)
return suite
```

### Usage in run_full_benchmark()

```python
# Step 1: Generate suites for all ISAs
isa_suites = {}
for isa_class in context.architecture.isa:
    suite = generate_microbenchmarks(context, isa_class.name)
    isa_suites[isa_class.name] = suite

# Step 2: Flatten for compilation
flat_benchmarks = {
    name: bench
    for suite in isa_suites.values()
    for name, bench in suite.benchmarks.items()
}

# ... compile, run, parse ...

# Step 3: Return suites with populated results
return isa_suites
```

## Common Workflows

### Extracting Peak Performance

```python
# After run_full_benchmark():
for isa_name, suite in result_suites.items():
    print(f"{isa_name}:")
    print(f"  Peak GOPS: {suite.peak_arithmetic_gops}")
    print(f"  Peak Bandwidth: {suite.peak_bandwidth_gb_per_s}")

    # Per-operation breakdown
    for op, gops in suite.get_gops_by_operation().items():
        print(f"    {op.name}: {gops} GOPS")

    # Per-cache-level breakdown
    for level, bw in suite.get_peak_bandwidth_by_level().items():
        print(f"    {level}: {bw} GB/s")
```

### Merging Results from Multiple Runs

```python
# Combine results from different runs
run1_suites = run_full_benchmark(context1)
run2_suites = run_full_benchmark(context2)

merged = {}
for isa_name in run1_suites:
    merged[isa_name] = ISABenchmarkSuite.merge_suites([
        run1_suites[isa_name],
        run2_suites[isa_name],
    ])
```

### Adding Custom Suite Type

1. Create new suite class in `benchmark/suites/`:
   ```python
   class CustomBenchmarkSuite(ISABenchmarkSuite):
       @classmethod
       def generate(cls, context, isa_name):
           suite = cls(isa_name=isa_name)
           # Generate custom benchmarks
           return suite
   ```

2. Add to `TestType` enum in `benchmark/benchmarking.py`:
   ```python
   class TestType(Enum):
       CUSTOM = "custom"
   ```

3. Register in `interface.py`:
   ```python
   suite_class_map = {
       ...,
       TestType.CUSTOM: CustomBenchmarkSuite,
   }
   ```

## Error Handling

- **ValueError** - Raised when merging suites with different ISA names
- **AttributeError** - Raised when accessing peak metrics before results populated

**Best Practice:** Always check `all_results_populated()` before accessing peak metrics.

## See Also

- **[../README.md](../README.md)** - Benchmark module overview
- **[../generation/README.md](../generation/README.md)** - ISA code generation
- **[../interface.py](../interface.py)** - Pipeline integration
- **[../../test_bench/README.md](../../test_bench/README.md)** - Benchmark execution

---

**When modifying this module:** Update this documentation when adding new suite types, changing generation logic, or adding peak extraction methods. Ensure suite class map in interface.py stays synchronized with TestType enum.
