"""Unified JSONL output writer for benchmark results.

Each line is a self-describing JSON object containing all benchmark
parameters and computed metrics. Mixed benchmarks use one line per fixed
configuration with a list of measured points.
This writer works across all test types (arithmetic, memory, roofline, etc.)
with a single unified schema.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from carm_roofline.core import Cycles
from carm_roofline.output_utils import info, warn
from carm_roofline.roofline_assembly import RecordType

if TYPE_CHECKING:
    from carm_roofline.benchmark.benchmark import ArithmeticBenchmark, MemoryBenchmark, MixedBenchmark
    from carm_roofline.benchmark.suites.base import ISABenchmarkSuite
    from carm_roofline.context import CARMContext
    from carm_roofline.core import Frequency
    from carm_roofline.isa import BaseISA


def _serialize_arithmetic(
    bench: ArithmeticBenchmark,
    isa_name: str,
    frequency: Frequency,
    isa_instance: BaseISA,
    timestamp: str,
    machine: str,
    actual_frequency_hz: int | None = None,
    frequency_overridden: bool = False,
) -> dict[str, object]:
    """Serialize an arithmetic benchmark to a flat dict for JSONL output."""
    from carm_roofline.benchmark.benchmark import ArithmeticBenchmarkResult

    res = bench.results
    assert isinstance(res, ArithmeticBenchmarkResult)

    ops_per_inst = isa_instance.ops_per_inst(bench.params.data_type, bench.params.operation)
    total_insts = (int(bench.params.num_ops) // ops_per_inst) * res.num_repetitions if ops_per_inst else 0
    cycles = Cycles.from_time_and_frequency(res.time_taken, frequency)
    ipc = 0.0 if cycles.value == 0 else total_insts / cycles.value

    d: dict[str, object] = {
        "type": RecordType.ARITHMETIC,
        "name": bench.name,
        "isa": isa_name,
        "data_type": bench.params.data_type.name,
        "num_threads": bench.params.num_threads,
        "thread_affinity": bench.params.thread_affinity,
        "timestamp": timestamp,
        "machine": machine,
        "operation": bench.params.operation.name,
        "num_ops": int(bench.params.num_ops),
        "performance_gops": float(res.performance) / 1e9,
        "ipc": ipc,
        "frequency_hz": float(frequency),
        "ops_per_instruction": ops_per_inst,
        "ops_per_cycle": ops_per_inst * ipc,
        "time_seconds": float(res.time_taken),
        "repetitions": res.num_repetitions,
        "cycles": int(cycles),
        "frequency_overridden": frequency_overridden,
    }
    if actual_frequency_hz is not None:
        d["actual_frequency_hz"] = actual_frequency_hz
    return d


def _serialize_memory(
    bench: MemoryBenchmark,
    isa_name: str,
    frequency: Frequency,
    isa_instance: BaseISA,
    timestamp: str,
    machine: str,
    actual_frequency_hz: int | None = None,
    frequency_overridden: bool = False,
) -> dict[str, object]:
    """Serialize a memory benchmark to a flat dict for JSONL output."""
    from carm_roofline.benchmark.benchmark import MemoryBenchmarkResult

    res = bench.results
    assert isinstance(res, MemoryBenchmarkResult)

    bpi = isa_instance.bytes_per_inst(bench.params.data_type)
    total_insts = (int(bench.working_set_bytes) // bpi) * res.num_repetitions
    cycles = Cycles.from_time_and_frequency(res.time_taken, frequency)
    ipc = 0.0 if cycles.value == 0 else total_insts / cycles.value

    d: dict[str, object] = {
        "type": RecordType.MEMORY,
        "name": bench.name,
        "isa": isa_name,
        "data_type": bench.params.data_type.name,
        "num_threads": bench.params.num_threads,
        "thread_affinity": bench.params.thread_affinity,
        "timestamp": timestamp,
        "machine": machine,
        "load_store_ratio": f"{bench.params.num_ld}:{bench.params.num_st}",
        "num_loads": bench.params.num_ld,
        "num_stores": bench.params.num_st,
        "cache_level": bench.cache_level,
        "memory_level_name": bench.params.memory_level_name,
        "size_per_thread_bytes": int(bench.params.size_per_thread),
        "working_set_bytes": int(bench.working_set_bytes),
        "layout_mode": bench.params.layout_mode.value,
        "bandwidth_gbps": float(res.bandwidth) / 1e9,
        "frequency_hz": float(frequency),
        "ipc": ipc,
        "time_seconds": float(res.time_taken),
        "repetitions": res.num_repetitions,
        "cycles": int(cycles),
        "frequency_overridden": frequency_overridden,
    }
    if actual_frequency_hz is not None:
        d["actual_frequency_hz"] = actual_frequency_hz
    return d


def _serialize_mixed(
    bench: MixedBenchmark,
    isa_name: str,
    frequency: Frequency,
    isa_instance: BaseISA,
    timestamp: str,
    machine: str,
    actual_frequency_hz: int | None = None,
    frequency_overridden: bool = False,
) -> dict[str, object]:
    """Serialize one mixed benchmark point for grouped JSONL output."""
    from carm_roofline.benchmark.benchmark import MixedBenchmarkResult

    res = bench.results
    assert isinstance(res, MixedBenchmarkResult)
    cycles = Cycles.from_time_and_frequency(res.time_taken, frequency)
    params = bench.params
    return {
        "type": RecordType.MIXED,
        "name": bench.name,
        "isa": isa_name,
        "data_type": params.data_type.name,
        "num_threads": params.num_threads,
        "thread_affinity": params.thread_affinity,
        "timestamp": timestamp,
        "machine": machine,
        "operation": params.operation.name,
        "load_store_ratio": f"{params.num_ld}:{params.num_st}",
        "arith_mem_ratio": f"{params.arith_mem_ratio[0]}:{params.arith_mem_ratio[1]}",
        "num_loads": params.num_ld,
        "num_stores": params.num_st,
        "cache_level": bench.cache_level,
        "memory_level_name": params.memory_level_name,
        "size_per_thread_bytes": int(params.size_per_thread),
        "working_set_bytes": int(bench.working_set_bytes),
        "layout_mode": params.layout_mode.value,
        "point_index": params.point_index,
        "requested_arithmetic_intensity": float(params.requested_arithmetic_intensity),
        "arithmetic_intensity": float(res.arithmetic_intensity),
        "performance_gops": float(res.performance) / 1e9,
        "num_arithmetic_instructions": params.num_arithmetic_instructions,
        "memory_pattern_repeats": params.memory_pattern_repeats,
        "operations_per_thread": int(bench.operations_per_thread),
        "frequency_hz": float(frequency),
        "time_seconds": float(res.time_taken),
        "repetitions": res.num_repetitions,
        "cycles": int(cycles),
        "frequency_overridden": frequency_overridden,
        **({"actual_frequency_hz": actual_frequency_hz} if actual_frequency_hz is not None else {}),
    }


_MIXED_SHARED_FIELDS = (
    "type",
    "isa",
    "data_type",
    "num_threads",
    "thread_affinity",
    "timestamp",
    "machine",
    "operation",
    "load_store_ratio",
    "arith_mem_ratio",
    "num_loads",
    "num_stores",
    "cache_level",
    "memory_level_name",
    "size_per_thread_bytes",
    "working_set_bytes",
    "layout_mode",
    "frequency_hz",
    "frequency_overridden",
    "actual_frequency_hz",
)


def _mixed_series_key(record: dict[str, object]) -> tuple[object, ...]:
    """Return the fixed configuration identity for one mixed point."""
    values: list[object] = []
    for field in _MIXED_SHARED_FIELDS:
        value = record.get(field)
        values.append(tuple(value) if isinstance(value, list) else value)
    return tuple(values)


def _group_mixed_record(record: dict[str, object]) -> dict[str, object]:
    """Create a grouped mixed series record from one point record."""
    series = {field: record[field] for field in _MIXED_SHARED_FIELDS if field in record}
    series["points"] = []
    return series


def _mixed_point(record: dict[str, object]) -> dict[str, object]:
    """Extract point-specific values from one mixed record."""
    return {field: value for field, value in record.items() if field not in _MIXED_SHARED_FIELDS}


def write_jsonl_benchmarks(
    context: CARMContext,
    isa_suites: dict[str, ISABenchmarkSuite],
    output_dir: Path | None = None,
) -> None:
    """Write benchmark results to a single JSONL file.

    Each arithmetic or memory benchmark produces one object. Mixed points
    share one object when their fixed configuration is identical.

    One JSON object per line is appended to
    ``<output_dir>/<machine_name>/benchmarks.jsonl``.

    Args:
        context: The CARM execution context.
        isa_suites: Benchmark results grouped by ISA name.
        output_dir: Override output directory (default: context.run_config.output_dir).
    """

    base_out_dir = Path(output_dir) if output_dir is not None else context.run_config.output_dir
    machine = context.run_config.name
    out_dir = base_out_dir / machine
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "benchmarks.jsonl"
    timestamp = datetime.now().isoformat()

    isa_instances: dict[str, BaseISA] = {
        isa_cls.name: isa_cls.from_architecture(context.architecture) for isa_cls in context.architecture.isa
    }

    lines: list[str] = []
    mixed_series: dict[tuple[object, ...], tuple[dict[str, object], list[dict[str, object]]]] = {}

    for isa, suite in sorted(isa_suites.items(), key=lambda kv: kv[0]):
        frequency = context.architecture.get_frequency_for_isa(isa)
        actual_freq_hz = context.architecture.actual_frequency_hz
        freq_overridden = bool(context.architecture.set_frequency)
        isa_instance = isa_instances.get(isa)
        if isa_instance is None:
            warn(f"No ISA instance available for {isa}; skipping benchmarks in JSONL output")
            continue

        # Arithmetic benchmarks
        for abench in suite.get_arithmetic_benchmarks().values():
            if abench.results is None:
                continue
            entry = _serialize_arithmetic(
                abench, isa, frequency, isa_instance, timestamp, machine, actual_freq_hz, freq_overridden
            )
            lines.append(json.dumps(entry, sort_keys=True))

        # Memory benchmarks
        for mbench in suite.get_memory_benchmarks().values():
            if mbench.results is None:
                continue
            entry = _serialize_memory(
                mbench, isa, frequency, isa_instance, timestamp, machine, actual_freq_hz, freq_overridden
            )
            lines.append(json.dumps(entry, sort_keys=True))

        for mixed_bench in suite.get_mixed_benchmarks().values():
            if mixed_bench.results is None:
                continue
            entry = _serialize_mixed(
                mixed_bench, isa, frequency, isa_instance, timestamp, machine, actual_freq_hz, freq_overridden
            )
            key = _mixed_series_key(entry)
            if key not in mixed_series:
                mixed_series[key] = (_group_mixed_record(entry), [])
            mixed_series[key][1].append(_mixed_point(entry))

    for series, points in mixed_series.values():
        series["points"] = points
        lines.append(json.dumps(series, sort_keys=True))

    if not lines:
        info(f"No benchmark results to write; created file at: {jsonl_path}")

    with open(jsonl_path, "a", encoding="utf-8") as f:
        for entry_line in lines:
            f.write(entry_line)
            f.write("\n")

    info(f"JSONL benchmarks appended to: {jsonl_path}")
