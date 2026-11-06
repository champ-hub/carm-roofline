"""Tests for MemoryBenchmarkSuite generation policy and DRAM handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
from carm_roofline.architecture.memory import CacheAwareThreadAffinity, DarwinMemoryTopology, MemoryLevelInfo
from carm_roofline.benchmark.benchmarking import LoadStoreRatio
from carm_roofline.benchmark.generation import MemoryLayoutMode
from carm_roofline.benchmark.generation.code_gen import TypedInstructions
from carm_roofline.core import (
    ArithmeticOperation,
    AutoMemoryLayout,
    Bytes,
    DataType,
    Frequency,
    MemoryLayoutSetting,
    MemoryOperation,
    UserError,
)
from carm_roofline.benchmark.suites.memory import MemoryBenchmarkSuite
from carm_roofline.benchmark.suites.memory_sweep import MemorySweepBenchmarkSuite
from carm_roofline.test_bench.builder import MicrobenchmarkFunctionSpec


pytestmark = pytest.mark.unit


@dataclass
class _DummyBenchmarking:
    data_type: list[DataType] = field(default_factory=lambda: [DataType.f64])
    threads: list[int] = field(default_factory=lambda: [4])
    ld_st_ratio: list[LoadStoreRatio] = field(default_factory=lambda: [LoadStoreRatio(loads=2, stores=1)])
    mem_target: list[str] = field(default_factory=lambda: ["L1", "L2", "L3", "DRAM"])
    mem_test_sizes: list[Bytes | None] | None = None
    mem_layouts: tuple[MemoryLayoutSetting, ...] | None = None


class _TopologyWithDRAMFinal:
    """Minimal topology stub with final level explicitly named DRAM."""

    def __iter__(self):
        yield MemoryLevelInfo(size=Bytes.from_string("32KiB"), name="L1", num_sharing_threads=1, instances=4)
        yield MemoryLevelInfo(size=Bytes.from_string("256KiB"), name="L2", num_sharing_threads=2, instances=2)
        yield MemoryLevelInfo(size=Bytes.from_string("8MiB"), name="DRAM", num_sharing_threads=4, instances=1)
    memory_layouts = (AutoMemoryLayout.auto, AutoMemoryLayout.auto, AutoMemoryLayout.auto)

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

    bench_instructions = TypedInstructions(
        {
            DataType.f64: {
                ArithmeticOperation.add: "vaddsd {}, {}, {}",
                MemoryOperation.ld: "movsd {off}({ptr}), {reg}",
                MemoryOperation.st: "movsd {reg}, {off}({ptr})",
            }
        }
    )

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

        memory_layouts = (AutoMemoryLayout.auto, AutoMemoryLayout.auto, AutoMemoryLayout.auto)
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


def test_memory_suite_auto_splits_the_final_selected_level():
    context = _make_context()
    context.benchmarking.mem_target = ["L1"]

    generated = list(MemoryBenchmarkSuite.generate(context, "dummy_isa").get_memory_benchmarks().values())

    assert len(generated) == 1
    assert generated[0].params.layout_mode == MemoryLayoutMode.split
    assert generated[0].params.size_per_thread == Bytes(int((Bytes.from_string("128KiB") // 4).value * 0.75))


def test_memory_suite_cli_layouts_override_topology_config_including_auto():
    context = _make_context()
    context.architecture.memory_topology.memory_layouts = (
        MemoryLayoutMode.split,
        MemoryLayoutMode.split,
        MemoryLayoutMode.single,
    )
    context.benchmarking.mem_layouts = (
        AutoMemoryLayout.auto,
        MemoryLayoutMode.single,
        AutoMemoryLayout.auto,
    )

    generated = list(MemoryBenchmarkSuite.generate(context, "dummy_isa").get_memory_benchmarks().values())

    assert [bench.params.layout_mode for bench in generated] == [
        MemoryLayoutMode.single,
        MemoryLayoutMode.single,
        MemoryLayoutMode.split,
    ]


def test_memory_suite_rejects_partial_cli_layout_list():
    context = _make_context()
    context.benchmarking.mem_layouts = (MemoryLayoutMode.single, MemoryLayoutMode.split)

    with pytest.raises(UserError, match="one value per topology level"):
        MemoryBenchmarkSuite.generate(context, "dummy_isa")


def test_memory_sweep_layout_applies_to_fallback_level(monkeypatch):
    context = _make_context()
    context.architecture.memory_topology.memory_layouts = (
        AutoMemoryLayout.auto,
        AutoMemoryLayout.auto,
        MemoryLayoutMode.single,
    )
    monkeypatch.setattr(
        "carm_roofline.benchmark.suites.memory_sweep.NUM_SWEEP_POINTS",
        2,
    )

    generated = list(MemorySweepBenchmarkSuite.generate(context, "dummy_isa").get_memory_benchmarks().values())
    fallback = generated[-1]

    assert fallback.cache_level == "DRAM"
    assert fallback.params.layout_mode == MemoryLayoutMode.single


def test_memory_sweep_uses_configured_cache_level_layout(monkeypatch):
    context = _make_context()
    context.architecture.memory_topology.memory_layouts = (
        MemoryLayoutMode.split,
        AutoMemoryLayout.auto,
        AutoMemoryLayout.auto,
    )
    monkeypatch.setattr("carm_roofline.benchmark.suites.memory_sweep.NUM_SWEEP_POINTS", 2)

    generated = list(MemorySweepBenchmarkSuite.generate(context, "dummy_isa").get_memory_benchmarks().values())
    first = generated[0]

    assert first.cache_level == "L1"
    assert first.params.layout_mode == MemoryLayoutMode.split


def test_memory_sweep_cli_auto_overrides_cache_level_config(monkeypatch):
    context = _make_context()
    context.architecture.memory_topology.memory_layouts = (
        MemoryLayoutMode.split,
        AutoMemoryLayout.auto,
        AutoMemoryLayout.auto,
    )
    context.benchmarking.mem_layouts = (
        AutoMemoryLayout.auto,
        AutoMemoryLayout.auto,
        AutoMemoryLayout.auto,
    )
    monkeypatch.setattr("carm_roofline.benchmark.suites.memory_sweep.NUM_SWEEP_POINTS", 2)

    generated = list(MemorySweepBenchmarkSuite.generate(context, "dummy_isa").get_memory_benchmarks().values())
    first = generated[0]

    assert first.cache_level == "L1"
    assert first.params.layout_mode == MemoryLayoutMode.single


def test_memory_sweep_auto_fallback_keeps_split_heuristic(monkeypatch):
    context = _make_context()
    monkeypatch.setattr("carm_roofline.benchmark.suites.memory_sweep.NUM_SWEEP_POINTS", 2)

    generated = list(MemorySweepBenchmarkSuite.generate(context, "dummy_isa").get_memory_benchmarks().values())
    fallback = generated[-1]

    assert fallback.cache_level == "DRAM"
    assert fallback.params.layout_mode == MemoryLayoutMode.split

def test_memory_suite_generates_with_darwin_topology(monkeypatch: pytest.MonkeyPatch) -> None:
    sysctls = {
        "hw.nperflevels": 1,
        "hw.ncpu": 2,
        "hw.memsize": 16 * 1024**3,
        "hw.perflevel0.physicalcpu": 2,
        "hw.perflevel0.logicalcpu": 2,
        "hw.perflevel0.l1dcachesize": 32 * 1024,
        "hw.perflevel0.l2cachesize": 256 * 1024,
        "hw.perflevel0.cpusperl2": 2,
    }
    monkeypatch.setattr("carm_roofline.architecture.memory.sysctl_int_or_none", sysctls.get)
    context = _make_context()
    topology = DarwinMemoryTopology()
    context.architecture.memory_topology = topology
    context.benchmarking.threads = [1]
    context.benchmarking.mem_target = [level.name for level in topology]

    generated = list(MemoryBenchmarkSuite.generate(context, "dummy_isa").get_memory_benchmarks().values())

    assert [bench.params.layout_mode for bench in generated] == [
        MemoryLayoutMode.single,
        MemoryLayoutMode.single,
        MemoryLayoutMode.split,
    ]
