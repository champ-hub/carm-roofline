"""Unit tests for execution-time deduplication of identical microbenchmarks.

Covers the measurement identity on MicrobenchmarkFunctionSpec and the
grouping/result-propagation helpers in carm_roofline.benchmark.interface.
"""

from __future__ import annotations

import pytest

from carm_roofline.benchmark import MemoryBenchmark
from carm_roofline.benchmark.benchmarking import LoadStoreRatio
from carm_roofline.benchmark.generation import MemoryBenchmarkParams
from carm_roofline.benchmark.interface import _group_duplicates, _propagate_results
from carm_roofline.core import Bytes, DataType, Frequency, Seconds
from carm_roofline.test_bench.builder import MicrobenchmarkFunctionSpec


def make_spec(
    name: str = "test",
    frequency_ghz: float = 2.5,
    thread_affinity: list[int] | None = None,
    read_array_size: int = 1024,
    write_array_size: int = 0,
    body: str = "// test body",
    nominal_frequency: Frequency | None = None,
) -> MicrobenchmarkFunctionSpec:
    """Helper to create a minimal MicrobenchmarkFunctionSpec for testing."""
    return MicrobenchmarkFunctionSpec(
        function_name=name,
        body=body,
        read_array_size=Bytes(read_array_size),
        write_array_size=Bytes(write_array_size),
        frequency=Frequency(frequency_ghz * 1e9),
        thread_affinity=thread_affinity if thread_affinity is not None else [0],
        nominal_frequency=nominal_frequency,
    )


def make_memory_benchmark(spec: MicrobenchmarkFunctionSpec) -> MemoryBenchmark:
    """Build a memory-style benchmark wrapping the given spec."""
    params = MemoryBenchmarkParams(
        data_type=DataType.i32,
        load_store_ratio=LoadStoreRatio(loads=4, stores=0),
        size_per_thread=Bytes(spec.read_array_size.value),
        memory_level_name="L1",
        thread_affinity=spec.thread_affinity,
    )
    return MemoryBenchmark(
        params=params,
        spec=spec,
        working_set_bytes=params.size_per_thread,
        cache_level="L1",
    )


@pytest.mark.unit
def test_measurement_key_ignores_function_name():
    """Specs differing only in function_name have equal measurement keys.

    The body embeds the function name once (like the generated C signature),
    so the key must strip the first occurrence to see through the rename.
    """
    spec_a = make_spec(name="mem_a")
    spec_a.body = "static void mem_a(void) { return; }"
    spec_b = make_spec(name="mem_b")
    spec_b.body = "static void mem_b(void) { return; }"

    assert spec_a.measurement_key() == spec_b.measurement_key()


@pytest.mark.unit
@pytest.mark.parametrize(
    "variant_kwargs",
    [
        {"body": "static void other(void) { return; }"},
        {"read_array_size": 2048},
        {"write_array_size": 16},
        {"thread_affinity": [1]},
        {"frequency_ghz": 3.0},
        {"nominal_frequency": Frequency(2.0e9)},
    ],
    ids=["body", "read_array_size", "write_array_size", "thread_affinity", "frequency", "nominal_frequency"],
)
def test_measurement_key_differs_when_field_changes(variant_kwargs):
    """A change to any single experiment field must change the key."""
    base = make_spec()
    variant = make_spec(**variant_kwargs)

    assert base.measurement_key() != variant.measurement_key()


@pytest.mark.unit
def test_group_duplicates_merges_only_equal_keys_first_inserted_wins():
    """Identical experiments group together; the first-inserted benchmark leads."""
    spec_a = make_spec(name="mem_a")
    spec_b = make_spec(name="mem_b")  # identical experiment, different name
    spec_c = make_spec(name="mem_c", read_array_size=4096)  # distinct experiment

    bench_a = make_memory_benchmark(spec_a)
    bench_b = make_memory_benchmark(spec_b)
    bench_c = make_memory_benchmark(spec_c)

    groups = _group_duplicates({"mem_a": bench_a, "mem_b": bench_b, "mem_c": bench_c})

    assert len(groups) == 2
    duplicate_group = groups[spec_a.measurement_key()]
    assert duplicate_group[0] is bench_a  # first-inserted benchmark is canonical
    assert duplicate_group[1] is bench_b
    assert groups[spec_c.measurement_key()] == [bench_c]


@pytest.mark.unit
def test_propagate_results_copies_canonical_measurement_to_aliases():
    """Aliases receive the canonical timing and a bandwidth recomputed from it."""
    spec_a = make_spec(name="mem_a")
    spec_b = make_spec(name="mem_b")
    bench_a = make_memory_benchmark(spec_a)
    bench_b = make_memory_benchmark(spec_b)

    groups = _group_duplicates({"mem_a": bench_a, "mem_b": bench_b})
    bench_a.process_results(Seconds.from_milliseconds(10.0), 500)
    canonical = bench_a.results
    assert canonical is not None

    _propagate_results(groups)

    assert bench_b.results is not None
    assert bench_b.results.time_taken == canonical.time_taken
    assert bench_b.results.num_repetitions == canonical.num_repetitions
    assert bench_b.results.bandwidth.value == pytest.approx(canonical.bandwidth.value)


@pytest.mark.unit
def test_propagate_results_leaves_aliases_untouched_when_canonical_has_no_result():
    """A group whose canonical has no result leaves every alias unset."""
    spec_a = make_spec(name="mem_a")
    spec_b = make_spec(name="mem_b")
    bench_a = make_memory_benchmark(spec_a)
    bench_b = make_memory_benchmark(spec_b)

    groups = _group_duplicates({"mem_a": bench_a, "mem_b": bench_b})
    _propagate_results(groups)

    assert bench_a.results is None
    assert bench_b.results is None
