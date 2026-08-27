"""Memory benchmark suite for cache hierarchy bandwidth measurement."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

from carm_roofline.core import Bytes, MemoryOperation, UserError
from carm_roofline.output_utils import debug, warn

from ..benchmark import MemoryBenchmark
from ..generation import MemoryBenchmarkParams, MemoryLayoutMode
from .base import ISABenchmarkSuite

if TYPE_CHECKING:
    from carm_roofline.context import CARMContext
    from carm_roofline.isa import BaseISA

# Fraction of a cache level's capacity to target, to avoid eviction effects.
_CACHE_COVERAGE = 0.75


def iter_memory_benchmark_params(context: CARMContext, isa: BaseISA) -> Iterator[MemoryBenchmarkParams]:
    """Yield memory benchmark parameters with the shared memory sizing policy."""
    benchmark = context.benchmarking
    architecture = context.architecture
    mem_level_indices = architecture.memory_topology.available_cache_levels()

    for data_type in benchmark.data_type:
        available_operations = isa.bench_instructions.available_operations(data_type)
        required_operations = {
            operation
            for operation, count in ((MemoryOperation.ld, ratio.loads) for ratio in benchmark.ld_st_ratio)
            if count
        } | {
            operation
            for operation, count in ((MemoryOperation.st, ratio.stores) for ratio in benchmark.ld_st_ratio)
            if count
        }
        if not required_operations.issubset(available_operations):
            warn(
                f"Skipping memory benchmarks for data type '{data_type.name}' on ISA '{isa.name}': "
                f"no required load/store instructions available"
            )
            continue
        for thread_count, ratio in product(benchmark.threads, benchmark.ld_st_ratio):
            if (ratio.loads and MemoryOperation.ld not in available_operations) or (
                ratio.stores and MemoryOperation.st not in available_operations
            ):
                warn(
                    f"Skipping memory benchmark for data type '{data_type.name}' on ISA '{isa.name}': "
                    f"load/store ratio {ratio} is not supported"
                )
                continue
            previous_size_per_thread: Bytes | None = None
            user_sizes: Iterator[Bytes | None] | None = (
                iter(benchmark.mem_test_sizes) if benchmark.mem_test_sizes else None
            )
            for mem_level_idx, mem_level_info in zip(mem_level_indices, architecture.memory_topology):
                if mem_level_info.name not in benchmark.mem_target:
                    continue
                thread_affinity = architecture.memory_topology.plan_thread_affinity(thread_count, mem_level_idx)
                avail_size_per_thread: Bytes = thread_affinity.total_cache_bytes // thread_affinity.num_threads
                is_final_target = mem_level_idx == mem_level_indices[-1]
                target_size_per_thread = (
                    previous_size_per_thread * 16
                    if is_final_target and previous_size_per_thread is not None
                    else avail_size_per_thread * _CACHE_COVERAGE
                )
                if user_sizes is not None:
                    user_size = next(user_sizes, None)
                    if user_size is not None:
                        target_size_per_thread = user_size
                total_used = target_size_per_thread * thread_affinity.num_threads
                for lower_level in range(1, mem_level_idx):
                    lower_bytes = thread_affinity.cache_bytes_per_level[lower_level]
                    if total_used <= lower_bytes:
                        warn(
                            f"{mem_level_info.name} memory benchmark: dataset "
                            f"({target_size_per_thread}/thread) fits in "
                            f"L{lower_level} ({lower_bytes // thread_affinity.num_threads}/thread). "
                            f"Data may be served from L{lower_level} instead."
                        )
                yield MemoryBenchmarkParams(
                    data_type=data_type,
                    thread_affinity=thread_affinity.cpu_ids,
                    load_store_ratio=ratio,
                    size_per_thread=target_size_per_thread,
                    memory_level_name=mem_level_info.name,
                    layout_mode=MemoryLayoutMode.split if is_final_target else MemoryLayoutMode.single,
                )
                previous_size_per_thread = avail_size_per_thread


@dataclass
class MemoryBenchmarkSuite(ISABenchmarkSuite):
    """Suite for memory bandwidth benchmarks (TestType.MEMORY).

    Measures memory bandwidth across cache hierarchy (L1/L2/L3/DRAM).
    """

    @classmethod
    def generate(cls, context: CARMContext, isa_name: str) -> MemoryBenchmarkSuite:
        """Generate memory microbenchmarks for a single ISA across cache levels.

        Creates benchmarks for each cache level (L1, L2, L3, DRAM) based on
        architecture configuration and mem_target setting.

        Args:
            context: CARM context with benchmarking and architecture config.
            isa_name: Name of the ISA to generate benchmarks for (e.g., "avx2").

        Returns:
            MemoryBenchmarkSuite with memory benchmarks for specified cache levels.

        Raises:
            ValueError: Unsupported data type or unknown ISA name.
        """
        benchmark = context.benchmarking
        architecture = context.architecture

        # Find ISA class by name
        isa_class = next(
            (isa_cls for isa_cls in architecture.isa if isa_cls.name == isa_name),
            None,
        )
        if isa_class is None:
            raise ValueError(f"Unknown ISA: {isa_name}")

        # Use factory method to instantiate ISA with architecture-specific parameters
        isa: BaseISA = isa_class.from_architecture(architecture)

        # Create ISA suite
        suite = cls(isa_name=isa.name)

        debug(f"Generating memory benchmarks for ISA '{isa.name}'")

        for params in iter_memory_benchmark_params(context, isa):
            debug(
                f"  [{params.memory_level_name}] size_per_thread={params.size_per_thread}, "
                f"threads={len(params.thread_affinity)}, cpu_ids={params.thread_affinity}, "
                f"ld_st_ratio={params.load_store_ratio}"
            )
            bench_spec = isa.generate_memory(params, context)
            test_size = bench_spec.read_array_size + bench_spec.write_array_size
            total_size = test_size * len(params.thread_affinity)
            mem_bench = MemoryBenchmark(
                params=params,
                spec=bench_spec,
                working_set_bytes=total_size,
                cache_level=params.memory_level_name,
            )
            debug(
                f"[{params.memory_level_name}] benchmark '{mem_bench.name}' added "
                f"(working set={total_size}, per thread={test_size})"
            )
            suite.add_benchmark(mem_bench.name, mem_bench)

        if not suite.benchmarks:
            raise UserError(
                f"No memory benchmarks can be generated for ISA '{isa_name}' with data type(s) "
                f"{[dt.name for dt in benchmark.data_type]}"
            )
        return suite

    def get_benchmarks_by_cache_level(self) -> dict[str, list[MemoryBenchmark]]:
        """Group memory benchmarks by cache level.

        Returns:
            Dictionary mapping cache level (L1/L2/L3/DRAM) to list of benchmarks.
        """
        result: defaultdict[str, list[MemoryBenchmark]] = defaultdict(list)
        for bench in self.get_memory_benchmarks().values():
            result[bench.cache_level or "UNKNOWN"].append(bench)
        return dict(result)
