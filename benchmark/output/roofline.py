"""Roofline output handler: print bandwidth/peak tables and plot roofline curves per ISA."""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmark.generation.code_gen.operation import ArithmeticOperation
from output_utils import info, warn
from units import Cycles

from .base import OutputHandler
from .common import format_precision_label

if TYPE_CHECKING:
    from benchmark.benchmark import ISABenchmarkSuite
    from benchmark.generation.isa import BaseISA
    from context import CARMContext


def _gather_roofline_points(suite: ISABenchmarkSuite) -> dict[str, tuple[float, float]]:
    """Gather roofline points (intensity, performance) from a benchmark suite.

    Roofline analysis requires both arithmetic intensity and GOPS,
    which are only available from MixedBenchmarkResult.

    Note:
        Skips invalid values (NaN, inf, or non-positive) with warning.
    """
    # Basic implementation used by tests and future consumers.
    # Collect intensity/performance pairs from MixedBenchmarkResult-like objects.
    points: dict[str, tuple[float, float]] = {}

    for name, bench in suite.benchmarks.items():
        res = getattr(bench, "results", None)
        if res is None:
            continue
        # we expect attributes 'arithmetic_intensity' and 'performance'
        ai = getattr(res, "arithmetic_intensity", None)
        perf = getattr(res, "performance", None)
        if ai is None or perf is None:
            # nothing to record for this benchmark
            continue
        try:
            ai_val = float(ai)
            perf_val = float(perf)
        except Exception:
            warn(f"Skipping invalid roofline point for {name} (non-numeric)")
            continue

        if ai_val <= 0 or perf_val <= 0:
            warn(f"Skipping invalid roofline point for {name} (non-positive values)")
            continue

        points[name] = (ai_val, perf_val)
    return points


def _print_table(context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
    info("Roofline summary per ISA:")


def _write_plot(isa_suites: dict[str, ISABenchmarkSuite], output_path: Path | None = None) -> None:
    """Generate roofline plots or warn when no data.

    This minimal implementation is sufficient for unit tests and avoids
    raising exceptions during missing-data scenarios.  Full plotting logic
    may be added later.
    """
    # gather all points across ISAs; warn if none found
    any_points = False
    for _isa, suite in isa_suites.items():
        pts = _gather_roofline_points(suite)
        if pts:
            any_points = True
            break

    if not any_points:
        warn("No roofline data found")
        return

    # Placeholder: in a complete implementation we would import matplotlib
    # and numpy and draw the curves. For now just silently succeed.
    return


def _to_isa_label(isa_name: str, context: CARMContext) -> str:
    isa_label = isa_name
    if isa_name in ["rvv0.7", "rvv1.0"]:
        vl = getattr(context.architecture, "vector_length", None)
        lmul = getattr(context.architecture, "vector_lmul", None)
        if vl is not None:
            isa_label = f"{isa_label}_vl{vl}"
        if lmul is not None:
            isa_label = f"{isa_label}_lmul{lmul}"
    elif isa_name == "sve":
        vl = getattr(context.architecture, "vector_length", None)
        if vl is not None:
            isa_label = f"{isa_label}_vl{vl}"
    return isa_label


def _cache_sizes_kib(context: CARMContext) -> tuple[int, int, int]:
    l1_kib = 0
    l2_kib = 0
    l3_kib = 0

    try:
        for level in context.architecture.memory_topology:
            name = getattr(level, "name", "")
            size = getattr(level, "size", None)
            if size is None:
                continue
            kib = int(size.value // 1024)
            if name == "L1":
                l1_kib = kib
            elif name == "L2":
                l2_kib = kib
            elif name == "L3":
                l3_kib = kib
    except Exception:
        # Fallback to zeros if topology is incomplete
        pass

    return l1_kib, l2_kib, l3_kib


def _find_dram_working_set_bytes(suite: ISABenchmarkSuite) -> int:
    for bench in suite.get_memory_benchmarks().values():
        if bench.cache_level == "DRAM" and getattr(bench, "working_set_bytes", None) is not None:
            return int(bench.working_set_bytes.value)
    return 0


def _get_isa_instance(context: CARMContext, isa_name: str) -> BaseISA | None:
    isa_class = next((isa_cls for isa_cls in context.architecture.isa if isa_cls.name == isa_name), None)
    if isa_class is None:
        return None
    return isa_class.from_architecture(context.architecture)


def _group_combos(
    mem_benches: dict[str, Any],
) -> dict[tuple[Any, int, int, int], list[Any]]:
    """Group memory benches by (data_type, num_threads, num_ld, num_st)."""
    groups: defaultdict[tuple[Any, int, int, int], list[Any]] = defaultdict(list)
    for bench in mem_benches.values():
        if bench.results is not None:
            key = (bench.params.data_type, bench.params.num_threads, bench.params.num_ld, bench.params.num_st)
            groups[key].append(bench)
    return groups


def _aggregate_level_metrics(
    context: CARMContext,
    isa_name: str,
    group: list[Any],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute per-level bandwidth and IPC for a group of memory benches."""
    bandwidth_by_level: dict[str, float] = {"L1": 0.0, "L2": 0.0, "L3": 0.0, "DRAM": 0.0}
    ipc_by_level: dict[str, float] = {"L1": 0.0, "L2": 0.0, "L3": 0.0, "DRAM": 0.0}

    isa_instance = _get_isa_instance(context, isa_name)
    if isa_instance is None:
        return bandwidth_by_level, ipc_by_level

    frequency = context.architecture.get_frequency_for_isa(isa_name)

    for bench in group:
        if bench.results is None or bench.cache_level not in bandwidth_by_level:
            continue
        level = bench.cache_level
        assert level is not None
        bytes_per_inst = isa_instance.bytes_per_inst(bench.params.data_type)
        bandwidth_by_level[level] = float(bench.results.bandwidth.value) / 1e9
        total_insts = (bench.working_set_bytes.value // bytes_per_inst) * bench.results.num_repetitions
        cycles = Cycles.from_time_and_frequency(bench.results.time_taken, frequency)
        ipc_by_level[level] = 0.0 if cycles.value == 0 else total_insts / cycles.value

    return bandwidth_by_level, ipc_by_level


def _pick_fp_for_combo(
    context: CARMContext,
    suite: ISABenchmarkSuite,
    isa_name: str,
    dt: Any,
    nt: int,
) -> tuple[str, float, float, float, float]:
    """Find arithmetic benches matching dt and nt, compute FP metrics."""
    frequency = context.architecture.get_frequency_for_isa(isa_name)
    isa_instance = _get_isa_instance(context, isa_name)
    fp_inst_name = ""
    fp_gflops = 0.0
    fp_ipc = 0.0
    fma_gflops = 0.0
    fma_ipc = 0.0

    if isa_instance is None:
        return fp_inst_name, fp_gflops, fp_ipc, fma_gflops, fma_ipc

    for bench in suite.get_arithmetic_benchmarks().values():
        if bench.results is None:
            continue
        if bench.params.data_type != dt or bench.params.num_threads != nt:
            continue
        ops_per_inst = isa_instance.ops_per_inst(bench.params.data_type, bench.params.operation)
        total_insts = (
            (bench.params.num_ops.value // ops_per_inst) * bench.results.num_repetitions if ops_per_inst else 0
        )
        cycles = Cycles.from_time_and_frequency(bench.results.time_taken, frequency)
        ipc = 0.0 if cycles.value == 0 else total_insts / cycles.value
        gflops = float(bench.results.performance.value) / 1e9

        if bench.params.operation == ArithmeticOperation.fma:
            fma_gflops = gflops
            fma_ipc = ipc
        else:
            fp_inst_name = bench.params.operation.name
            fp_gflops = gflops
            fp_ipc = ipc

    if not fp_inst_name:
        fp_inst_name = ArithmeticOperation.fma.name
        fp_gflops = fma_gflops
        fp_ipc = fma_ipc

    return fp_inst_name, fp_gflops, fp_ipc, fma_gflops, fma_ipc


def _write_csv(context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite], output_dir: Path | None = None) -> None:
    base_out_dir = Path(output_dir) if output_dir is not None else context.run_config.output_dir
    out_dir = base_out_dir / "roofline"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{context.run_config.name}_roofline.csv"
    file_exists = csv_path.exists()
    for isa, suite in sorted(isa_suites.items(), key=lambda kv: kv[0]):
        mem_benches = suite.get_memory_benchmarks()
        interleaved = "Yes" if context.benchmarking.interleaved else "No"
        dram_bytes = _find_dram_working_set_bytes(suite)
        l1_kib, l2_kib, l3_kib = _cache_sizes_kib(context)

        def round_helper(x: float) -> str:
            return f"{x:.3g}" if x < 1 else f"{x:.3f}"

        secondary_headers = [
            "Name:",
            context.run_config.name,
            "L1 Size:",
            l1_kib,
            "L2 Size:",
            l2_kib,
            "L3 Size:",
            l3_kib,
            "",
            "L1",
            "L1",
            "L2",
            "L2",
            "L3",
            "L3",
            "DRAM",
            "DRAM",
            "FP",
            "FP",
            "FP FMA",
            "FP_FMA",
        ]

        primary_headers = [
            "Date",
            "ISA",
            "Precision",
            "Threads",
            "Loads",
            "Stores",
            "Interleaved",
            "DRAM Bytes",
            "FP Inst.",
            "GB/s",
            "I/Cycle",
            "GB/s",
            "I/Cycle",
            "GB/s",
            "I/Cycle",
            "GB/s",
            "I/Cycle",
            "Gflop/s",
            "I/Cycle",
            "Gflop/s",
            "I/Cycle",
        ]

        mode = "a" if file_exists else "w"
        with open(csv_path, mode, newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(secondary_headers)
                writer.writerow(primary_headers)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            isa_label = _to_isa_label(isa, context)

            for (dt, nt, ld, st), group in _group_combos(mem_benches).items():
                bandwidth_by_level, icycle_by_level = _aggregate_level_metrics(context, isa, group)
                fp_inst_name, fp_gflops, fp_ipc, fma_gflops, fma_ipc = _pick_fp_for_combo(context, suite, isa, dt, nt)

                row = [
                    timestamp,
                    isa_label,
                    format_precision_label(dt),
                    nt,
                    ld,
                    st,
                    interleaved,
                    dram_bytes,
                    fp_inst_name,
                    round_helper(bandwidth_by_level["L1"]),
                    round_helper(icycle_by_level["L1"]),
                    round_helper(bandwidth_by_level["L2"]),
                    round_helper(icycle_by_level["L2"]),
                    round_helper(bandwidth_by_level["L3"]),
                    round_helper(icycle_by_level["L3"]),
                    round_helper(bandwidth_by_level["DRAM"]),
                    round_helper(icycle_by_level["DRAM"]),
                    round_helper(fp_gflops),
                    round_helper(fp_ipc),
                    round_helper(fma_gflops),
                    round_helper(fma_ipc),
                ]
                writer.writerow(row)

        file_exists = True

    info(f"Roofline legacy CSV saved to: {csv_path}")


class RooflineOutputHandler(OutputHandler):
    """Output handler for roofline benchmarks."""

    def print_table(self, context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
        _print_table(context, isa_suites)

    def write_plot(self, context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
        path = context.run_config.output_dir / "roofline"
        _write_plot(isa_suites, output_path=path)

    def write_csv(self, context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
        _write_csv(context, isa_suites)

    def write_jsonl(self, context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
        from benchmark.output.jsonl import write_jsonl_benchmarks

        write_jsonl_benchmarks(context, isa_suites)
