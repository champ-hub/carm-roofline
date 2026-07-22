"""Roofline output handler: print bandwidth/peak tables and plot roofline curves per ISA."""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.table import Table

from carm_roofline.core import ArithmeticIntensity, ArithmeticOperation, Bandwidth, Cycles, Performance
from carm_roofline.output_utils import error, get_console, info, warn

from . import arithmetic
from .base import OutputHandler
from .common import format_precision_label, safe_matplotlib_import, save_or_show_plot

if TYPE_CHECKING:
    from carm_roofline.benchmark.benchmark import ISABenchmarkSuite
    from carm_roofline.context import CARMContext
    from carm_roofline.isa import BaseISA


def _print_table(context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
    """Print a merged roofline summary table.

    One row per ISA x (data_type, threads, loads, stores) combo, showing peak
    intensity. Values are displayed via their unit class __str__ methods
    (e.g., "8.95 GOPS/s", "380.00 GB/s", "0.02 FLOP/B").
    At verbose >= 3, also prints the arithmetic and memory detail tables.
    """
    table = Table(title="Roofline Summary")
    table.add_column("ISA", style="cyan")
    table.add_column("Prec.", justify="left")
    table.add_column("Thr.", style="magenta", justify="right")
    table.add_column("Peak", justify="right")
    for level in _CACHE_LEVELS:
        table.add_column(level, justify="right")
        table.add_column(f"{level} AI", justify="right")

    for isa, suite in sorted(isa_suites.items(), key=lambda kv: kv[0]):
        mem_benches = suite.get_memory_benchmarks()
        arith_benches = suite.get_arithmetic_benchmarks()

        for (dt, nt, _ld, _st), group in sorted(_group_combos(mem_benches).items()):
            # Collect Bandwidth per cache level from this combo's memory benchmarks
            bw_by_level: dict[str, Bandwidth | None] = dict.fromkeys(_CACHE_LEVELS)
            for bench in group:
                if bench.results is not None and bench.cache_level in bw_by_level:
                    bw_by_level[bench.cache_level] = bench.results.bandwidth

            # Find peak Performance from arithmetic benchmarks matching this combo
            peak_perf: Performance | None = None
            for bench in arith_benches.values():
                if bench.results is None:
                    continue
                if bench.params.data_type != dt or bench.params.num_threads != nt:
                    continue
                if peak_perf is None or bench.results.performance > peak_perf:
                    peak_perf = bench.results.performance

            row: list[str] = [isa, format_precision_label(dt), str(nt)]
            if peak_perf is not None:
                row.append(str(peak_perf))
            else:
                row.append("-")

            for level in _CACHE_LEVELS:
                bw = bw_by_level[level]
                if bw is not None and bw.value > 0 and peak_perf is not None:
                    ridge = ArithmeticIntensity(peak_perf.value / bw.value)
                    row.append(str(bw))
                    row.append(str(ridge))
                else:
                    row.append("-")
                    row.append("-")
            table.add_row(*row)

    get_console().print(table)

    if context.run_config.verbose >= 3:
        from . import memory

        arithmetic._print_table(context, isa_suites)
        memory._print_table(context, isa_suites)


_ROOFLINE_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
_LINE_STYLES = {"L1": "-", "L2": "--", "L3": "-.", "DRAM": ":"}
_CACHE_LEVELS = ("L1", "L2", "L3", "DRAM")


def _plot_roofline_to_axis(
    roofline_data: dict[str, tuple[float, dict[str, float]]],
    ax: Any,
    np: Any,
) -> None:
    """Draw roofline curves on a matplotlib log-log axis.

    For each ISA: horizontal ceiling at peak Gflop/s, and one bandwidth slope
    per cache level (y = bandwidth x AI, clipped at the ceiling).
    """
    all_ridges: list[float] = []
    for peak_gflops, bw_gbps in roofline_data.values():
        for bw in bw_gbps.values():
            if bw > 0:
                all_ridges.append(peak_gflops / bw)

    if not all_ridges:
        warn("No valid roofline ridge points to plot")
        return

    ai_min = min(all_ridges) / 10
    ai_max = max(all_ridges) * 10
    ai_values = np.logspace(np.log10(ai_min), np.log10(ai_max), 200)
    if not isinstance(ai_values, list):
        ai_values = list(ai_values)

    for idx, (isa, (peak_gflops, bw_gbps)) in enumerate(sorted(roofline_data.items())):
        color = _ROOFLINE_COLORS[idx % len(_ROOFLINE_COLORS)]

        # Compute ridge points for this ISA to clip the ceiling
        isa_ridges = [peak_gflops / bw for level in _CACHE_LEVELS if (bw := bw_gbps.get(level, 0)) > 0]
        if isa_ridges:
            min_ridge = min(isa_ridges)
            ceiling_ai = [ai for ai in ai_values if ai >= min_ridge]
        else:
            ceiling_ai = []
        if ceiling_ai:
            ax.plot(
                ceiling_ai,
                [peak_gflops] * len(ceiling_ai),
                color=color,
                linestyle="-",
                linewidth=1.5,
                label=f"{isa} ceiling",
            )
        for level in _CACHE_LEVELS:
            bw = bw_gbps.get(level, 0)
            if bw <= 0:
                continue
            perf = [min(bw * ai, peak_gflops) for ai in ai_values]
            ax.plot(
                ai_values,
                perf,
                color=color,
                linestyle=_LINE_STYLES[level],
                linewidth=1,
                alpha=1,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax.set_ylabel("Performance (Gflop/s)")
    ax.set_title("Roofline Model")
    ax.legend(fontsize=8)


def _write_plot(isa_suites: dict[str, ISABenchmarkSuite], output_path: Path | None = None) -> None:
    """Generate and save roofline plot.

    Draws log-log roofline curves (bandwidth slopes + compute ceiling) per ISA.
    Gracefully handles missing data, invalid values, and I/O errors.
    """
    roofline_data: dict[str, tuple[float, dict[str, float]]] = {}
    for isa, suite in sorted(isa_suites.items(), key=lambda kv: kv[0]):
        try:
            peak = suite.get_peak_performance()
            bw_by_level = suite.get_bandwidth_by_level()
        except ValueError:
            warn(f"Skipping {isa} in roofline plot: missing or incomplete benchmark results")
            continue
        peak_gflops = float(peak) / 1e9
        bw_gbps = {level: float(bw) / 1e9 for level, bw in bw_by_level.items()}
        roofline_data[isa] = (peak_gflops, bw_gbps)

    if not roofline_data:
        warn("No roofline data found")
        return

    plt, np = safe_matplotlib_import()
    if plt is None:
        return

    fig = None
    try:
        fig, ax = plt.subplots(figsize=(10, 7))
        _plot_roofline_to_axis(roofline_data, ax, np)
        plt.tight_layout()
        save_or_show_plot(output_path, "roofline.png", plt=plt)
    except Exception as e:
        error(f"Failed to generate roofline plot: {e}")
    finally:
        if fig is not None:
            plt.close(fig)


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
            kib = int(size) // 1024
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
            return int(bench.working_set_bytes)
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
        bandwidth_by_level[level] = float(bench.results.bandwidth) / 1e9
        total_insts = (int(bench.working_set_bytes) // bytes_per_inst) * bench.results.num_repetitions
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
        total_insts = (int(bench.params.num_ops) // ops_per_inst) * bench.results.num_repetitions if ops_per_inst else 0
        cycles = Cycles.from_time_and_frequency(bench.results.time_taken, frequency)
        ipc = 0.0 if cycles.value == 0 else total_insts / cycles.value
        gflops = float(bench.results.performance) / 1e9

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
        from carm_roofline.benchmark.output.jsonl import write_jsonl_benchmarks

        write_jsonl_benchmarks(context, isa_suites)
