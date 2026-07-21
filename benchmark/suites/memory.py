"""Memory benchmark suite for cache hierarchy bandwidth measurement."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

from core import Bytes
from output_utils import debug, warn

from ..benchmark import MemoryBenchmark
from ..generation import MemoryBenchmarkParams, MemoryLayoutMode
from .base import ISABenchmarkSuite

if TYPE_CHECKING:
    from context import CARMContext
    from isa import BaseISA

# Fraction of a cache level's capacity to target, to avoid eviction effects.
_CACHE_COVERAGE = 0.75


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

        mem_level_indices = architecture.memory_topology.available_cache_levels()
        for data_type, thread_count, ratio in product(benchmark.data_type, benchmark.threads, benchmark.ld_st_ratio):
            previous_size_per_thread: Bytes | None = None
            user_sizes: Iterator[Bytes | None] | None = (
                iter(benchmark.mem_test_sizes) if benchmark.mem_test_sizes else None
            )
            for mem_level_idx, mem_level_info in zip(mem_level_indices, architecture.memory_topology):
                if mem_level_info.name not in benchmark.mem_target:
                    continue
                # Plan thread affinity for this memory level
                thread_affinity = architecture.memory_topology.plan_thread_affinity(thread_count, mem_level_idx)
                avail_size_per_thread: Bytes = thread_affinity.total_cache_bytes // thread_affinity.num_threads
                _is_first_target = mem_level_idx == mem_level_indices[0]
                is_final_target = mem_level_idx == mem_level_indices[-1]

                if is_final_target and previous_size_per_thread is not None:
                    target_size_per_thread = previous_size_per_thread * 16
                else:
                    target_size_per_thread = avail_size_per_thread * _CACHE_COVERAGE

                # Override with user-provided size (if any) for this cache level.
                if user_sizes is not None:
                    user_size = next(user_sizes, None)
                    if user_size is not None:
                        target_size_per_thread = user_size

                # Warn if the per-thread dataset also fits in a smaller cache level.
                total_used = target_size_per_thread * thread_affinity.num_threads
                lower_levels = range(1, mem_level_idx)
                for lower_level in lower_levels:
                    lower_bytes = thread_affinity.cache_bytes_per_level[lower_level]
                    if total_used <= lower_bytes:
                        warn(
                            f"{mem_level_info.name} memory benchmark: dataset "
                            f"({target_size_per_thread}/thread) fits in "
                            f"L{lower_level} ({lower_bytes // thread_affinity.num_threads}/thread). "
                            f"Data may be served from L{lower_level} instead."
                        )

                # Generate memory benchmark for this level
                layout_mode = MemoryLayoutMode.split if is_final_target else MemoryLayoutMode.single
                params = MemoryBenchmarkParams(
                    data_type=data_type,
                    thread_affinity=thread_affinity.cpu_ids,
                    load_store_ratio=ratio,
                    size_per_thread=target_size_per_thread,
                    memory_level_name=mem_level_info.name,
                    layout_mode=layout_mode,
                )

                debug(
                    f"  [{mem_level_info.name}] size_per_thread={target_size_per_thread}, "
                    f"threads={thread_affinity.num_threads}, cpu_ids={thread_affinity.cpu_ids}, "
                    f"ld_st_ratio={ratio}"
                )

                bench_spec = isa.generate_memory(params, context)
                test_size = bench_spec.read_array_size + bench_spec.write_array_size
                total_size = test_size * thread_affinity.num_threads

                mem_bench = MemoryBenchmark(
                    params=params,
                    spec=bench_spec,
                    working_set_bytes=total_size,
                    cache_level=mem_level_info.name,
                )
                debug(
                    f"[{mem_level_info.name}] benchmark '{mem_bench.name}' added (working set={total_size}, "
                    f"per thread={test_size})"
                )
                suite.add_benchmark(mem_bench.name, mem_bench)
                previous_size_per_thread = avail_size_per_thread

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
