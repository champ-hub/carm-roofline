"""Tests for MemoryBenchmarkSuite generation policy and DRAM handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

from carm_roofline.architecture.memory import CacheAwareThreadAffinity, MemoryLevelInfo
from carm_roofline.benchmark.benchmarking import LoadStoreRatio
from carm_roofline.benchmark.generation import MemoryLayoutMode
from carm_roofline.core import DataType
from carm_roofline.benchmark.suites.memory import MemoryBenchmarkSuite
from carm_roofline.test_bench.builder import MicrobenchmarkFunctionSpec
from carm_roofline.core import Bytes, Frequency


@dataclass
class _DummyBenchmarking:
    data_type: list[DataType] = field(default_factory=lambda: [DataType.f64])
    threads: list[int] = field(default_factory=lambda: [4])
    ld_st_ratio: list[LoadStoreRatio] = field(default_factory=lambda: [LoadStoreRatio(loads=2, stores=1)])
    mem_test_sizes: list[Bytes | None] | None = None
    mem_target: list[str] = field(default_factory=lambda: ["L1", "L2", "L3", "DRAM"])


class _TopologyWithDRAMFinal:
    """Minimal topology stub with final level explicitly named DRAM."""

    def __iter__(self):
        yield MemoryLevelInfo(size=Bytes.from_string("32KiB"), name="L1", num_sharing_threads=1, instances=4)
        yield MemoryLevelInfo(size=Bytes.from_string("256KiB"), name="L2", num_sharing_threads=2, instances=2)
        yield MemoryLevelInfo(size=Bytes.from_string("8MiB"), name="DRAM", num_sharing_threads=4, instances=1)

    def available_cache_levels(self) -> list[int]:
        return [1, 2, 3]

    def plan_thread_affinity(
        self, n_threads: int, cache_level: int, prefer_no_smt: bool = True
    ) -> CacheAwareThreadAffinity:
        cpu_ids = list(range(n_threads))

        if cache_level == 1:
            cache_bytes_per_level = {
                1: Bytes.from_string("128KiB"),
            }
        elif cache_level == 2:
            cache_bytes_per_level = {
                1: Bytes.from_string("128KiB"),
                2: Bytes.from_string("1MiB"),
            }
        else:
            # Purposefully oversized L2 to ensure old lower-level-fit checks would trigger on DRAM.
            cache_bytes_per_level = {
                1: Bytes.from_string("128KiB"),
                2: Bytes.from_string("20MiB"),
                3: Bytes.from_string("32MiB"),
            }

        return CacheAwareThreadAffinity(
            cache_level=cache_level,
            cpu_ids=cpu_ids,
            cache_bytes_per_level=cache_bytes_per_level,
        )


class _DummyISA:
    name = "dummy_isa"

    @classmethod
    def from_architecture(cls, architecture):
        return cls()

    def generate_memory(self, params, context):
        return MicrobenchmarkFunctionSpec(
            function_name=f"mem_{params.memory_level_name.lower()}_{params.size_per_thread.value}",
            body="",
            read_array_size=Bytes(0),
            write_array_size=Bytes(0),
            frequency=Frequency(3e9),
            thread_affinity=params.thread_affinity,
        )


def _make_context() -> SimpleNamespace:
    topology = _TopologyWithDRAMFinal()
    architecture = SimpleNamespace(isa=[_DummyISA], memory_topology=topology)
    benchmarking = _DummyBenchmarking()
    return SimpleNamespace(architecture=architecture, benchmarking=benchmarking)


def test_memory_suite_uses_dram_name_for_final_topology_level():
    context = _make_context()

    suite = MemoryBenchmarkSuite.generate(context, "dummy_isa")
    levels = {bench.cache_level for bench in suite.get_memory_benchmarks().values()}

    assert levels == {"L1", "L2", "DRAM"}


def test_memory_suite_final_level_uses_geq_16x_previous_size(monkeypatch):
    context = _make_context()
    warnings: list[str] = []

    def _capture_warn(message: str) -> None:
        warnings.append(message)

    monkeypatch.setattr("carm_roofline.benchmark.suites.memory.warn", _capture_warn)

    suite = MemoryBenchmarkSuite.generate(context, "dummy_isa")
    by_level = {bench.cache_level: bench for bench in suite.get_memory_benchmarks().values()}

    l1_size = by_level["L1"].params.size_per_thread
    l2_size = by_level["L2"].params.size_per_thread
    dram_size = by_level["DRAM"].params.size_per_thread

    assert l1_size == Bytes(int((Bytes.from_string("128KiB") // 4).value * 0.75))
    assert l2_size == Bytes(int((Bytes.from_string("1MiB") // 4).value * 0.75))
    assert dram_size >= l2_size * 16


def test_memory_suite_warns_if_dataset_fits_in_lower_cache(monkeypatch):
    # Create a topology where the L1 cache is large enough that an L2-targeted
    # dataset (with default 4 threads) will fit entirely in L1.  The warning
    # logic should flag this condition when generating the L2 benchmark.
    class _TopologyWithLowerFit:
        def __iter__(self):
            yield MemoryLevelInfo(size=Bytes.from_string("8GiB"), name="L1", num_sharing_threads=1, instances=1)
            yield MemoryLevelInfo(size=Bytes.from_string("8GiB"), name="L2", num_sharing_threads=1, instances=1)
            yield MemoryLevelInfo(size=Bytes.from_string("32GiB"), name="DRAM", num_sharing_threads=1, instances=1)

        def available_cache_levels(self) -> list[int]:
            return [1, 2, 3]

        def plan_thread_affinity(
            self, n_threads: int, cache_level: int, prefer_no_smt: bool = True
        ) -> CacheAwareThreadAffinity:
            cpu_ids = list(range(n_threads))
            if cache_level == 1:
                cache_bytes_per_level = {1: Bytes.from_string("8GiB")}
            elif cache_level == 2:
                cache_bytes_per_level = {1: Bytes.from_string("8GiB"), 2: Bytes.from_string("8GiB")}
            else:
                cache_bytes_per_level = {
                    1: Bytes.from_string("8GiB"),
                    2: Bytes.from_string("8GiB"),
                    3: Bytes.from_string("32GiB"),
                }
            return CacheAwareThreadAffinity(
                cache_level=cache_level,
                cpu_ids=cpu_ids,
                cache_bytes_per_level=cache_bytes_per_level,
            )

    context = _make_context()
    context.architecture.memory_topology = _TopologyWithLowerFit()

    warnings: list[str] = []
    monkeypatch.setattr("carm_roofline.benchmark.suites.memory.warn", lambda msg: warnings.append(msg))

    _suite = MemoryBenchmarkSuite.generate(context, "dummy_isa")

    # There should be at least one warning mentioning that the L2 benchmark
    # fits in L1.  The exact numeric sizes are not important for this test.
    assert any("L2 memory benchmark" in m and "fits in L1" in m for m in warnings), (
        f"expected lower-cache warning, got: {warnings}"
    )


def test_memory_suite_uses_single_then_split_for_last_level():
    context = _make_context()

    suite = MemoryBenchmarkSuite.generate(context, "dummy_isa")
    generated = list(suite.get_memory_benchmarks().values())

    assert len(generated) == 3
    assert generated[0].params.layout_mode == MemoryLayoutMode.single
    assert generated[1].params.layout_mode == MemoryLayoutMode.single
    assert generated[2].params.layout_mode == MemoryLayoutMode.split
