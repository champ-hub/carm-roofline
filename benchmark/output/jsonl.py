"""Unified JSONL output writer for benchmark results.

Each line is a self-describing JSON object containing all benchmark
parameters and computed metrics. One line per benchmark.
This writer works across all test types (arithmetic, memory, roofline, etc.)
with a single unified schema.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from output_utils import info, warn
from roofline_assembly import RecordType
from units import Cycles

if TYPE_CHECKING:
    from benchmark.benchmark import ArithmeticBenchmark, MemoryBenchmark
    from benchmark.generation.isa import BaseISA
    from benchmark.suites.base import ISABenchmarkSuite
    from context import CARMContext
    from units import Frequency


def _serialize_arithmetic(
    bench: ArithmeticBenchmark,
    isa_name: str,
    frequency: Frequency,
    isa_instance: BaseISA,
    timestamp: str,
    machine: str,
) -> dict[str, object]:
    """Serialize an arithmetic benchmark to a flat dict for JSONL output."""
    from benchmark.benchmark import ArithmeticBenchmarkResult

    res = bench.results
    assert isinstance(res, ArithmeticBenchmarkResult)

    ops_per_inst = isa_instance.ops_per_inst(bench.params.data_type, bench.params.operation)
    total_insts = (bench.params.num_ops.value // ops_per_inst) * res.num_repetitions if ops_per_inst else 0
    cycles = Cycles.from_time_and_frequency(res.time_taken, frequency)
    ipc = 0.0 if cycles.value == 0 else total_insts / cycles.value

    return {
        "type": RecordType.ARITHMETIC,
        "name": bench.name,
        "isa": isa_name,
        "data_type": bench.params.data_type.name,
        "num_threads": bench.params.num_threads,
        "thread_affinity": bench.params.thread_affinity,
        "timestamp": timestamp,
        "machine": machine,
        "operation": bench.params.operation.name,
        "num_ops": int(bench.params.num_ops.value),
        "performance_gops": res.performance.value / 1e9,
        "ipc": ipc,
        "frequency_hz": frequency.value,
        "ops_per_instruction": ops_per_inst,
        "ops_per_cycle": ops_per_inst * ipc,
        "time_seconds": res.time_taken.value,
        "repetitions": res.num_repetitions,
        "cycles": cycles.value,
    }


def _serialize_memory(
    bench: MemoryBenchmark,
    isa_name: str,
    frequency: Frequency,
    isa_instance: BaseISA,
    timestamp: str,
    machine: str,
) -> dict[str, object]:
    """Serialize a memory benchmark to a flat dict for JSONL output."""
    from benchmark.benchmark import MemoryBenchmarkResult

    res = bench.results
    assert isinstance(res, MemoryBenchmarkResult)

    bpi = isa_instance.bytes_per_inst(bench.params.data_type)
    total_insts = (bench.working_set_bytes.value // bpi) * res.num_repetitions
    cycles = Cycles.from_time_and_frequency(res.time_taken, frequency)
    ipc = 0.0 if cycles.value == 0 else total_insts / cycles.value

    return {
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
        "size_per_thread_bytes": bench.params.size_per_thread.value,
        "working_set_bytes": bench.working_set_bytes.value,
        "layout_mode": bench.params.layout_mode.value,
        "bandwidth_gbps": res.bandwidth.value / 1e9,
        "ipc": ipc,
        "time_seconds": res.time_taken.value,
        "repetitions": res.num_repetitions,
        "cycles": cycles.value,
    }


def write_jsonl_benchmarks(
    context: CARMContext,
    isa_suites: dict[str, ISABenchmarkSuite],
    output_dir: Path | None = None,
) -> None:
    """Write all benchmark results to a single JSONL file.

    One JSON object per line, appended to
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

    for isa, suite in sorted(isa_suites.items(), key=lambda kv: kv[0]):
        frequency = context.architecture.get_frequency_for_isa(isa)
        isa_instance = isa_instances.get(isa)
        if isa_instance is None:
            warn(f"No ISA instance available for {isa}; skipping benchmarks in JSONL output")
            continue

        # Arithmetic benchmarks
        for abench in suite.get_arithmetic_benchmarks().values():
            if abench.results is None:
                continue
            entry = _serialize_arithmetic(abench, isa, frequency, isa_instance, timestamp, machine)
            lines.append(json.dumps(entry, sort_keys=True))

        # Memory benchmarks
        for mbench in suite.get_memory_benchmarks().values():
            if mbench.results is None:
                continue
            entry = _serialize_memory(mbench, isa, frequency, isa_instance, timestamp, machine)
            lines.append(json.dumps(entry, sort_keys=True))

        # NOTE: Mixed benchmarks are not yet generated by any suite.
        # When they are, add a branch here using suite.get_mixed_benchmarks().

    if not lines:
        info(f"No benchmark results to write; created file at: {jsonl_path}")

    with open(jsonl_path, "a", encoding="utf-8") as f:
        for entry_line in lines:
            f.write(entry_line)
            f.write("\n")

    info(f"JSONL benchmarks appended to: {jsonl_path}")
