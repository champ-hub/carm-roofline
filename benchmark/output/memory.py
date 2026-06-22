"""Memory output handler: prints bandwidth numbers and optionally plots bandwidth by cache level."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.table import Table

from benchmark.benchmark import MemoryBenchmarkResult
from benchmark.generation.isa import BaseISA
from output_utils import error, warn
from units import Cycles

from .base import NON_ROOFLINE_CSV_ERROR_MSG, OutputHandler
from .common import (
    safe_matplotlib_import,
    save_or_show_plot,
)

if TYPE_CHECKING:
    from benchmark.benchmark import ISABenchmarkSuite
    from context import CARMContext


def _collect_bandwidth_by_label(isa_suites: dict[str, ISABenchmarkSuite]) -> dict[str, float]:
    bandwidth_by_label: dict[str, float] = {}

    for isa, suite in sorted(isa_suites.items(), key=lambda kv: kv[0]):
        benches: dict[str, Any]
        if hasattr(suite, "get_memory_benchmarks"):
            benches = suite.get_memory_benchmarks()
        else:
            benches = getattr(suite, "benchmarks", {})

        for name, bench in sorted(benches.items(), key=lambda kv: kv[0]):
            res = getattr(bench, "results", None)
            if not isinstance(res, MemoryBenchmarkResult):
                continue

            level = res.cache_level or "unknown"
            bandwidth_value = float(res.bandwidth.value) / 1e9
            if not math.isfinite(bandwidth_value):
                warn(f"Skipping invalid bandwidth value for {isa}/{name}: {bandwidth_value}")
                continue

            label = f"{isa}:{level}"
            bandwidth_by_label[label] = bandwidth_value

    return bandwidth_by_label


def _print_table(context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
    """Print a tabular summary of memory bandwidth by ISA and cache level.

    The arithmetic handler prints a single row per ISA.  Memory tests may
    generate multiple results per ISA (one per cache level), so the output is
    structured differently: we create a ``rich.Table`` with one row per
    <ISA,level> pair and show only the bandwidth to keep the table compact.
    """

    # ISA instances needed for ops-per-instruction calculations
    isa_instances: dict[str, BaseISA] = {
        isa_cls.name: isa_cls.from_architecture(context.architecture) for isa_cls in context.architecture.isa
    }

    table = Table(title="Memory Bandwidth Summary")
    table.add_column("ISA", style="cyan")
    table.add_column("Level", justify="left")
    table.add_column("Threads", style="magenta")
    table.add_column("Bandwidth", justify="right")
    table.add_column("IPC", justify="right")

    # Walk through each ISA and its memory benchmarks.  Sort by ISA name for
    # deterministic output, then by benchmark name (which implicitly encodes
    # the cache level).
    for isa, suite in sorted(isa_suites.items(), key=lambda kv: kv[0]):
        mem_benchmarks = suite.get_memory_benchmarks()
        isa_instance = isa_instances[isa]
        frequency = context.architecture.get_frequency_for_isa(isa)

        for _name, bench in sorted(mem_benchmarks.items(), key=lambda kv: kv[0]):
            assert bench.results is not None  # for type checker
            level = bench.results.cache_level or "unknown"
            bytes_per_inst = isa_instance.bytes_per_inst(bench.params.data_type)
            total_insts = (bench.working_set_bytes.value // bytes_per_inst) * bench.results.num_repetitions
            cycles = Cycles.from_time_and_frequency(bench.results.time_taken, frequency)

            ipc = total_insts / cycles.value

            table.add_row(isa, level, str(bench.params.num_threads), str(bench.results.bandwidth), f"{ipc:.2f}")

    # Print the table using the shared console helper (rich.Console instance).
    from output_utils import get_console

    get_console().print(table)


def _plot_to_axis(isa_suites: dict[str, ISABenchmarkSuite], ax: Any) -> None:
    """Plot memory bandwidth bars to a given matplotlib axis.

    Args:
        isa_suites: Benchmark results organized by ISA
        ax: Matplotlib axis object to plot on

    Note:
        Internal helper for creating combined plots (e.g., in mixed handler).
        Assumes matplotlib is available.
    """
    bandwidth = _collect_bandwidth_by_label(isa_suites)
    if not bandwidth:
        warn("No memory bandwidth data available for plotting")
        return

    labels = list(bandwidth.keys())
    vals = list(bandwidth.values())

    # Validate we have plottable data
    if all(v == 0.0 for v in vals):
        warn("All bandwidth values are zero; plot may not be meaningful")

    ax.bar(labels, vals)
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title("Memory Bandwidth by ISA")


def _write_plot(isa_suites: dict[str, ISABenchmarkSuite], output_path: Path | None = None) -> None:
    """Generate and save memory bandwidth plot.

    Args:
        isa_suites: Benchmark results organized by ISA
        output_path: Directory to save plot (None = display interactively)

    Note:
        Gracefully handles missing data, invalid values, and I/O errors.
        Will not raise exceptions.
    """
    plt, _ = safe_matplotlib_import()
    if plt is None:
        return

    bandwidth = _collect_bandwidth_by_label(isa_suites)
    if not bandwidth:
        warn("No memory bandwidth data available for plotting")
        return

    labels = list(bandwidth.keys())

    fig = None
    try:
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))
        _plot_to_axis(isa_suites, ax)
        plt.tight_layout()

        save_or_show_plot(output_path, "memory_bandwidth.png", plt=plt)
    except Exception as e:
        error(f"Failed to generate memory bandwidth plot: {e}")
    finally:
        # Explicitly close the figure to prevent matplotlib from retaining it in memory.
        if fig is not None:
            plt.close(fig)


class MemoryOutputHandler(OutputHandler):
    """Output handler for memory benchmarks."""

    def print_table(self, context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
        _print_table(context, isa_suites)

    def write_plot(self, context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
        path = context.run_config.output_dir / "memory"
        _write_plot(isa_suites, output_path=path)

    def write_csv(self, context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
        warn(NON_ROOFLINE_CSV_ERROR_MSG)

    def write_jsonl(self, context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
        from benchmark.output.jsonl import write_jsonl_benchmarks

        write_jsonl_benchmarks(context, isa_suites)
