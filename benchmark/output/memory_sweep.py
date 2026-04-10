"""Memory sweep output handler: prints bandwidth vs working-set-size table and plots the sweep curve."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from rich.table import Table

from benchmark.benchmark import MemoryBenchmarkResult
from benchmark.generation.isa import BaseISA
from benchmark.suites import MemorySweepBenchmarkSuite
from output_utils import error, info, warn
from units import Cycles

from .base import NON_ROOFLINE_CSV_ERROR_MSG, OutputHandler
from .common import safe_matplotlib_import, save_or_show_plot

if TYPE_CHECKING:
    from benchmark.benchmark import ISABenchmarkSuite
    from context import CARMContext

# Fixed color map for cache levels — consistent across all plots.
LEVEL_COLORS: dict[str, str] = {
    "L1": "green",
    "L2": "blue",
    "L3": "orange",
    "DRAM": "red",
}


def _format_bytes(n: int) -> str:
    """Format a byte count as a human-readable string (KB / MB / GB)."""
    if n >= 1024**3:
        return f"{n / 1024**3:.1f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def _format_bytes_mib(n: int) -> str:
    """Format a byte count as a human-readable string using IEC (KiB / MiB / GiB)."""
    if n >= 1024**3:
        return f"{n / 1024**3:.1f} GiB"
    if n >= 1024**2:
        return f"{n / 1024**2:.1f} MiB"
    if n >= 1024:
        return f"{n / 1024:.1f} KiB"
    return f"{n} B"


def _print_table(context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
    """Print a tabular summary of bandwidth vs working-set size for each ISA.

    Rows are ordered by ISA name then ascending working-set size.  Benchmarks
    without results are skipped with a warning so the table is always valid.
    """

    table = Table(title="Memory Bandwidth Sweep")
    table.add_column("ISA", style="cyan")
    table.add_column("Working Set", justify="right")
    table.add_column("Cache Level", justify="left")
    table.add_column("Bandwidth", justify="right")
    table.add_column("IPC", justify="right")

    # ISA instances needed for ops-per-instruction calculations
    isa_instances: dict[str, BaseISA] = {
        isa_cls.name: isa_cls.from_architecture(context.architecture) for isa_cls in context.architecture.isa
    }

    for isa, suite in sorted(isa_suites.items(), key=lambda kv: kv[0]):
        assert isinstance(suite, MemorySweepBenchmarkSuite)  # for type checker
        mem_benchmarks = suite.get_memory_benchmarks()
        isa_instance = isa_instances[isa]
        bytes_per_inst = isa_instance.bytes_per_inst(context.benchmarking.data_type)
        frequency = context.architecture.get_frequency_for_isa(isa)
        for _name, bench in sorted(mem_benchmarks.items(), key=lambda kv: kv[0]):
            assert bench.results is not None  # for type checker
            total_insts = (bench.working_set_bytes.value // bytes_per_inst) * bench.results.num_repetitions
            cycles = Cycles.from_time_and_frequency(bench.results.time_taken, frequency)
            cache_level = bench.results.cache_level or "unknown"
            table.add_row(
                isa,
                str(bench.working_set_bytes),
                cache_level,
                str(bench.results.bandwidth),
                f"{total_insts / cycles.value:.2f}",
            )

    from output_utils import get_console

    get_console().print(table)


def _collect_sweep_series(
    isa_suites: dict[str, ISABenchmarkSuite],
) -> dict[str, list[tuple[int, str, float]]]:
    """Collect per-ISA lists of (working_set_bytes_int, cache_level, bandwidth_gbps).

    The lists are sorted ascending by working-set size.  Entries with invalid /
    missing bandwidth are excluded.
    """
    from benchmark.suites.memory_sweep import MemorySweepBenchmarkSuite  # local import avoids circular dep

    series: dict[str, list[tuple[int, str, float]]] = {}

    for isa, suite in sorted(isa_suites.items(), key=lambda kv: kv[0]):
        points: list[tuple[int, str, float]] = []

        if isinstance(suite, MemorySweepBenchmarkSuite):
            for ws_bytes, cache_level, bandwidth in suite.get_sweep_data():
                if bandwidth is None:
                    continue
                bw_val = float(bandwidth.value) / 1e9
                if not math.isfinite(bw_val):
                    continue
                points.append((ws_bytes.value, cache_level, bw_val))
        else:
            mem_benchmarks = suite.get_memory_benchmarks()
            for _name, bench in sorted(mem_benchmarks.items(), key=lambda kv: kv[0]):
                res = bench.results
                if not isinstance(res, MemoryBenchmarkResult):
                    continue
                bw_val = float(res.bandwidth.value) / 1e9
                if not math.isfinite(bw_val):
                    continue
                ws = bench.working_set_bytes
                cache_level = res.cache_level or "unknown"
                points.append((ws.value, cache_level, bw_val))

        if points:
            points.sort(key=lambda t: t[0])
            series[isa] = points

    return series


def _plot_to_axis(isa_suites: dict[str, ISABenchmarkSuite], ax: Any) -> None:
    """Plot the bandwidth-vs-working-set sweep to *ax*.

    One line per ISA; scatter points colour-coded by cache level.
    Suitable for embedding inside combined (mixed) plots.
    """
    series = _collect_sweep_series(isa_suites)
    if not series:
        warn("No memory sweep data available for plotting")
        return

    # ISA line styles — cycle through a small set of line styles for readability.
    line_styles = ["-", "--", "-.", ":"]
    isa_line_handles: list[Any] = []

    for idx, (isa, points) in enumerate(series.items()):
        xs = [p[0] for p in points]
        ys = [p[2] for p in points]
        levels = [p[1] for p in points]
        colors = [LEVEL_COLORS.get(lvl, "gray") for lvl in levels]

        ls = line_styles[idx % len(line_styles)]
        (line,) = ax.plot(xs, ys, linestyle=ls, color="black", alpha=0.4, zorder=1, label=isa)
        ax.scatter(xs, ys, c=colors, zorder=2, s=60)
        isa_line_handles.append(line)

    # Build legend: ISA lines + cache-level colour patches.
    import matplotlib.patches as mpatches  # matplotlib already confirmed available at this point

    level_handles = [mpatches.Patch(color=color, label=level) for level, color in LEVEL_COLORS.items()]
    ax.legend(handles=isa_line_handles + level_handles, loc="upper right", fontsize="small")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.xaxis.set_major_formatter(
        __import__("matplotlib.ticker", fromlist=["FuncFormatter"]).FuncFormatter(lambda x, _pos: _format_bytes(int(x)))
    )
    ax.set_xlabel("Working Set Size")
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title("Memory Bandwidth Sweep")


def _write_plot(isa_suites: dict[str, ISABenchmarkSuite], output_path: Path | None = None) -> None:
    """Generate and save a bandwidth-vs-working-set-size line plot.

    Args:
        isa_suites: Benchmark results organised by ISA name.
        output_path: Directory to save the plot.  ``None`` shows it interactively.

    Note:
        Gracefully handles missing data, invalid values, and I/O errors.
        Will not raise exceptions.
    """
    plt, _ = safe_matplotlib_import()
    if plt is None:
        return

    series = _collect_sweep_series(isa_suites)
    if not series:
        warn("No memory sweep data available for plotting")
        return

    fig: Any = None
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        _plot_to_axis(isa_suites, ax)
        plt.tight_layout()
        save_or_show_plot(output_path, "memory_sweep_bandwidth.png", plt=plt)
    except Exception as e:
        error(f"Failed to generate memory sweep bandwidth plot: {e}")
    finally:
        if fig is not None:
            plt.close(fig)


def _write_json(context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
    # ISA instances needed for instruction-count based IPC calculations.
    isa_instances: dict[str, BaseISA] = {
        isa_cls.name: isa_cls.from_architecture(context.architecture) for isa_cls in context.architecture.isa
    }

    results: dict[str, list[dict[str, object]]] = {}

    for isa, suite in sorted(isa_suites.items(), key=lambda kv: kv[0]):
        mem_benchmarks = suite.get_memory_benchmarks()
        isa_instance = isa_instances[isa]
        bytes_per_inst = isa_instance.bytes_per_inst(context.benchmarking.data_type)
        frequency = context.architecture.get_frequency_for_isa(isa)

        rows: list[dict[str, object]] = []
        for name, bench in sorted(mem_benchmarks.items(), key=lambda kv: kv[0]):
            if bench.results is None:
                continue

            total_insts = (bench.working_set_bytes.value // bytes_per_inst) * bench.results.num_repetitions
            cycles = Cycles.from_time_and_frequency(bench.results.time_taken, frequency)
            ipc = 0.0 if cycles.value == 0 else total_insts / cycles.value

            rows.append(
                {
                    "benchmark": name,
                    "cache_level": bench.results.cache_level,
                    "working_set_bytes": int(bench.working_set_bytes.value),
                    "bandwidth_gbps": float(bench.results.bandwidth.value) / 1e9,
                    "ipc": ipc,
                    "time_seconds": float(bench.results.time_taken.value),
                    "repetitions": int(bench.results.num_repetitions),
                    "cycles": float(cycles.value),
                }
            )

        rows.sort(key=lambda row: cast(int, row["working_set_bytes"]))
        results[isa] = rows

    out_dir = context.run_config.output_dir / "memory_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{context.run_config.name}_memory_sweep.json"
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump({"results": results}, json_file, indent=2, sort_keys=True)

    info(f"Memory sweep JSON saved to: {json_path}")


class MemorySweepOutputHandler(OutputHandler):
    """Output handler for memory sweep benchmarks."""

    def print_table(self, context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
        _print_table(context, isa_suites)

    def write_plot(self, context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
        path = context.run_config.output_dir / "memory_sweep"
        _write_plot(isa_suites, output_path=path)

    def write_csv(self, context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
        warn(NON_ROOFLINE_CSV_ERROR_MSG)

    def write_json(self, context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
        _write_json(context, isa_suites)
