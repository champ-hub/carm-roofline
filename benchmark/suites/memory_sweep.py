"""Memory sweep benchmark suite for continuous cache hierarchy bandwidth measurement.

Generates 32 logarithmically-spaced working-set sizes spanning the full memory
hierarchy — from 80 % of L1 capacity up to 4x the last real cache level (DRAM
benchmark point) — so that bandwidth can be plotted as a continuous function of
working-set size rather than at a handful of fixed cache-level targets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from output_utils import debug
from units import Bandwidth, Bytes

from ..benchmark import MemoryBenchmark
from ..generation import MemoryBenchmarkParams, MemoryLayoutMode
from .base import ISABenchmarkSuite

if TYPE_CHECKING:
    from architecture.memory import CacheAwareThreadAffinity
    from context import CARMContext

# Number of logarithmically-spaced sweep points.
NUM_SWEEP_POINTS: int = 48


@dataclass
class MemorySweepBenchmarkSuite(ISABenchmarkSuite):
    """Suite for a continuous memory-bandwidth sweep across the cache hierarchy.

    Instead of one benchmark per cache level, this suite generates
    ``NUM_SWEEP_POINTS`` benchmark with logarithmically spaced working-set sizes.
    Each benchmark is annotated with the cache level it falls into
    (smallest level whose per-thread capacity ≥ working-set size per thread),
    defaulting to the last topology level (DRAM) for oversized working sets.
    """

    @classmethod
    def generate(cls, context: CARMContext, isa_name: str) -> MemorySweepBenchmarkSuite:
        """Generate logarithmically-spaced memory sweep benchmarks for a single ISA.

        Args:
            context: CARM context with benchmarking and architecture config.
            isa_name: Name of the ISA to generate benchmarks for (e.g., ``"x86_avx2"``).

        Returns:
            ``MemorySweepBenchmarkSuite`` containing ``NUM_SWEEP_POINTS`` benchmarks.

        Raises:
            ValueError: If the ISA name is unknown or the topology has fewer than
                one level.
        """
        from benchmark.generation.isa import BaseISA

        benchmark = context.benchmarking
        architecture = context.architecture
        data_type = benchmark.data_type

        # Resolve ISA class by name.
        isa_class = next(
            (isa_cls for isa_cls in architecture.isa if isa_cls.name == isa_name),
            None,
        )
        if isa_class is None:
            raise ValueError(f"Unknown ISA: {isa_name}")

        isa: BaseISA = isa_class.from_architecture(architecture)
        suite = cls(isa_name=isa.name)

        debug(
            f"Generating memory sweep benchmarks for ISA '{isa.name}', "
            f"data_type={data_type}, threads={benchmark.threads}, "
            f"n_points={NUM_SWEEP_POINTS}"
        )

        # ------------------------------------------------------------------
        # Step 1 - Build per-level metadata table.
        #
        # For each topology level we record:
        #   level_num          - numeric level index (1-based, for plan_thread_affinity)
        #   level_name         - canonical name ("L1", "L2", …, "DRAM")
        #   avail_per_thread   - per-thread capacity at this level
        #   affinity           - pre-computed CacheAwareThreadAffinity
        # ------------------------------------------------------------------
        topology = architecture.memory_topology

        # The last entry is the DRAM (or last topology level) sentinel.
        all_targets: list[tuple[int, str]] = [
            (level_num, level_info.name)
            for level_num, level_info in zip(
                topology.available_cache_levels(),
                iter(topology),
            )
        ]

        if not all_targets:
            raise ValueError("Memory topology has no levels; cannot generate sweep.")

        # Pre-compute affinity and per-thread capacity for every level.
        level_table: list[tuple[int, str, Bytes, CacheAwareThreadAffinity]] = []
        for level_num, level_name in all_targets:
            affinity = topology.plan_thread_affinity(benchmark.threads, level_num)
            avail: Bytes = affinity.total_cache_bytes // affinity.num_threads
            level_table.append((level_num, level_name, avail, affinity))

        # ------------------------------------------------------------------
        # Step 2 - Determine sweep range.
        #
        # min_size: 0.1 x L1 per-thread capacity
        # max_size: 32 x capacity of the *last real cache* before DRAM
        #
        # Following the convention in MemoryBenchmarkSuite, the DRAM benchmark
        # point is placed at 32 x the penultimate (last non-DRAM) cache capacity.
        # If there is only one level, we treat it as both L1 and the last real
        # cache.
        # ------------------------------------------------------------------
        _, _, l1_avail, _ = level_table[0]
        min_size_bytes: float = 0.1 * float(l1_avail.value)

        # The "last real cache" is the second-to-last entry when there are ≥2
        # levels (the last being DRAM).  With only one level we reuse it.
        if len(level_table) >= 2:
            _, _, last_cache_avail, _ = level_table[-2]
        else:
            _, _, last_cache_avail, _ = level_table[-1]

        max_size_bytes: float = 32.0 * float(last_cache_avail.value)

        if max_size_bytes <= min_size_bytes:
            raise ValueError(
                f"Sweep range is degenerate: min={min_size_bytes}, max={max_size_bytes}. "
                "Check cache topology configuration."
            )

        debug(
            f"  sweep range: min_size={Bytes(int(min_size_bytes))}, "
            f"max_size={Bytes(int(max_size_bytes))}, n_points={NUM_SWEEP_POINTS}"
        )

        log_min = math.log(min_size_bytes)
        log_max = math.log(max_size_bytes)

        # Cache levels available for "belongs to" classification (all levels
        # except DRAM, which is the fallback).
        cache_levels_for_classification = level_table[:-1]  # all but last
        _, dram_level_name, _, dram_affinity = level_table[-1]

        # ------------------------------------------------------------------
        # Step 3 - Generate one benchmark per sweep point.
        # ------------------------------------------------------------------
        for i in range(NUM_SWEEP_POINTS):
            # Log-linear interpolation between min and max.
            t = i / (NUM_SWEEP_POINTS - 1) if NUM_SWEEP_POINTS > 1 else 0.0
            size_bytes = math.exp(log_min + t * (log_max - log_min))
            size_per_thread = Bytes(int(size_bytes))

            # Classify: find the smallest cache level whose per-thread capacity
            # is >= the sweep size. Fall back to DRAM label if none matches.
            assigned_level_name: str = dram_level_name
            assigned_affinity: CacheAwareThreadAffinity = dram_affinity
            for _lvl_num, lvl_name, avail, aff in cache_levels_for_classification:
                if size_per_thread <= avail:
                    assigned_level_name = lvl_name
                    assigned_affinity = aff
                    # L1 can suffer from 4k aliasing - use single-array layout to prevent speculative forwarding issues
                    mem_layout = MemoryLayoutMode.single if _lvl_num == 1 else MemoryLayoutMode.split
                    break

            # Use sweep{i:02d} as memory_level_name to guarantee unique C
            # function names even when rounded sizes coincide.
            sweep_label = f"sweep{i:02d}"

            params = MemoryBenchmarkParams(
                data_type=data_type,
                thread_affinity=assigned_affinity.cpu_ids,
                load_store_ratio=benchmark.ld_st_ratio,
                size_per_thread=size_per_thread,
                memory_level_name=sweep_label,
                layout_mode=mem_layout,
            )

            bench_spec = isa.generate_memory(params, context)

            working_set = size_per_thread * assigned_affinity.num_threads

            mem_bench = MemoryBenchmark(
                params=params,
                spec=bench_spec,
                working_set_bytes=working_set,
                cache_level=assigned_level_name,
            )

            debug(
                f"  [sweep{i:02d}] size_per_thread={size_per_thread}, "
                f"cache_level={assigned_level_name}, "
                f"threads={assigned_affinity.num_threads}, "
                f"cpu_ids={assigned_affinity.cpu_ids}, "
                f"bench='{mem_bench.name}'"
            )

            suite.add_benchmark(mem_bench.name, mem_bench)

        return suite

    # ------------------------------------------------------------------
    # Public interface consumed by the output handler.
    # ------------------------------------------------------------------

    def get_sweep_data(self) -> list[tuple[Bytes, str, Bandwidth | None]]:
        """Return sweep results sorted by ascending working-set size.

        Each element is a 3-tuple:

        * ``working_set_bytes`` - total bytes across all threads.
        * ``cache_level`` - classified level name (e.g. ``"L1"``, ``"DRAM"``).
        * ``bandwidth`` - measured bandwidth, or ``None`` if not yet executed.

        Returns:
            List of ``(Bytes, str, Bandwidth | None)`` tuples in ascending
            working-set-size order.
        """
        rows: list[tuple[Bytes, str, Bandwidth | None]] = []
        for bench in self.get_memory_benchmarks().values():
            bw: Bandwidth | None = bench.results.bandwidth if bench.results is not None else None
            level = bench.cache_level if bench.cache_level is not None else "UNKNOWN"
            rows.append((bench.working_set_bytes, level, bw))

        rows.sort(key=lambda t: t[0].value)
        return rows
