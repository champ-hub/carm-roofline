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

**Generation:** Calls `ArithmeticBenchmarkSuite.generate(context, isa_name)`, which iterates over the
Cartesian product of `context.benchmarking.data_type × threads × instructions`, produces
`ArithmeticBenchmarkParams` per combination, delegates code generation to
`isa_instance.generate_arithmetic(params, context)`, and wraps each result in an `ArithmeticBenchmark`.

**Key Features:** Cartesian product over ``data_type × threads × instructions``;
``get_gops_by_operation()`` provides per-operation peak GOPS.

**Additional Methods:**

**`get_gops_by_operation() -> dict[ArithmeticOperation, float]`**:
Returns dict mapping operations to their peak GOPS:
```python
{ArithmeticOperation.add: 150.2, ArithmeticOperation.mul: 145.8, ArithmeticOperation.fma: 290.5}
```

## Memory Suite (memory.py)

### MemoryBenchmarkSuite

Generates memory bandwidth benchmarks targeting specific cache levels.

**Generation:** Calls ``MemoryBenchmarkSuite.generate(context, isa_name)``, which iterates over the

Cartesian product of ``context.benchmarking.data_type × threads × ld_st_ratio`` and for each combination
produces one benchmark per memory level (filtered by ``mem_target``). Per-level sizing uses 80% of
available cache capacity per thread (or 2× previous level for the final target), with
``MemoryLayoutMode.split`` for the final level and ``single`` for the rest. Delegates code generation to
``isa_instance.generate_memory(params, context)`` and wraps each result in a ``MemoryBenchmark``.

**Additional Methods:**

**``get_benchmarks_by_cache_level(level: str) -> list[MemoryBenchmark]``**:
Returns benchmarks for a specific cache level.

**``get_peak_bandwidth_by_level() -> dict[str, float]``**:
Returns peak bandwidth per cache level (e.g. ``{"L1": 250.5, "L2": 120.3, "DRAM": 25.8}``).

## Roofline Suite (roofline.py)

### RooflineBenchmarkSuite

Combines arithmetic and memory suites for roofline analysis.

**Generation:** Calls ``RooflineBenchmarkSuite.generate(context, isa_name)``, which creates an empty suite,
delegates to ``ArithmeticBenchmarkSuite.generate`` and ``MemoryBenchmarkSuite.generate``, and merges both
into the result via ``suite.merge()``.

**Key Features:** Delegates to ``ArithmeticBenchmarkSuite.generate`` and
``MemoryBenchmarkSuite.generate`` and merges via ``suite.merge()``.

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

In ``generate_microbenchmarks()`` (``interface.py``), the ``TestType`` is mapped to the corresponding
suite class via ``suite_class_map`` and ``suite_class.generate(context, isa_name)`` is called.

### Usage in run_full_benchmark()
``run_full_benchmark()`` (``interface.py``) generates suites for each ISA via
``generate_microbenchmarks()``, flattens all benchmarks into a single dict for compilation and
execution, then returns the per-ISA suite dict with results populated.

## Common Workflows
Use ``suite.peak_arithmetic_gops`` and ``suite.peak_bandwidth_gb_per_s`` for aggregate metrics.
``suite.get_gops_by_operation()`` and ``suite.get_peak_bandwidth_by_level()`` provide per-operation
and per-cache-level breakdowns respectively.

### Merging Results from Multiple Runs

``ISABenchmarkSuite.merge_suites()`` combines results from multiple runs by ISA name:
``merged[isa] = ISABenchmarkSuite.merge_suites([suite1, suite2])``.

### Adding Custom Suite Type

1. Subclass ``ISABenchmarkSuite`` and implement ``generate()``.
2. Add the ``TestType`` enum entry in ``benchmark/benchmarking.py``.
3. Register in ``interface.py``'s ``suite_class_map``.

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
