"""Arithmetic output handler: prints GOPS tables and (optional) simple bar plots."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.table import Table

from output_utils import error, get_console, warn
from units import Cycles

from .base import NON_ROOFLINE_CSV_ERROR_MSG, OutputHandler
from .common import (
    safe_matplotlib_import,
    save_or_show_plot,
)

if TYPE_CHECKING:
    from benchmark.benchmark import ISABenchmarkSuite
    from context import CARMContext


def _collect_gops_by_isa(isa_suites: dict[str, ISABenchmarkSuite]) -> dict[str, float]:
    from benchmark.benchmark import ArithmeticBenchmarkResult

    gops_by_isa: dict[str, float] = {}

    for isa, suite in sorted(isa_suites.items(), key=lambda kv: kv[0]):
        values: list[float] = []
        for bench in suite.get_arithmetic_benchmarks().values():
            if not isinstance(bench.results, ArithmeticBenchmarkResult):
                continue

            gops_value = float(bench.results.performance.value) / 1e9
            if not math.isfinite(gops_value):
                warn(f"Skipping invalid GOPS value for ISA {isa}: {gops_value}")
                continue
            values.append(gops_value)

        if values:
            gops_by_isa[isa] = max(values)

    return gops_by_isa


def _print_table(context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
    from benchmark.benchmark import ArithmeticBenchmarkResult
    from benchmark.generation.isa import BaseISA

    print_extra = context.run_config.verbose >= 3

    # ISA instances needed for ops-per-instruction calculations
    isa_instances: dict[str, BaseISA] = {
        isa_cls.name: isa_cls.from_architecture(context.architecture) for isa_cls in context.architecture.isa
    }

    arithmetic_metrics: list[dict[str, Any]] = []

    for isa, suite in sorted(isa_suites.items(), key=lambda kv: kv[0]):
        frequency = context.architecture.get_frequency_for_isa(isa)
        isa_instance = isa_instances.get(isa)
        if isa_instance is None:
            warn(f"No ISA instance available for {isa}; skipping metrics")
            continue

        benches = suite.get_arithmetic_benchmarks()
        for benchmark_name, benchmark in benches.items():
            res = benchmark.results
            assert isinstance(res, ArithmeticBenchmarkResult), (
                "Expected ArithmeticBenchmarkResult for arithmetic summary"
            )

            ops_per_inst = isa_instance.ops_per_inst(benchmark.params.data_type, benchmark.params.operation)
            total_insts = (benchmark.params.num_ops.value // ops_per_inst) * res.num_repetitions if ops_per_inst else 0
            cycles = Cycles.from_time_and_frequency(res.time_taken, frequency)
            ipc = 0.0 if cycles.value == 0 else total_insts / cycles.value
            ops_per_cycle = ops_per_inst * ipc

            arithmetic_metrics.append(
                {
                    "isa": isa,
                    "operation": benchmark.params.operation.name,
                    "benchmark": benchmark_name,
                    "threads": benchmark.params.num_threads,
                    "gops": float(res.performance.value) / 1e9,
                    "gops_display": str(res.performance),
                    "ipc": ipc,
                    "frequency_hz": float(frequency.value),
                    "frequency_display": str(frequency),
                    "ops_per_instruction": ops_per_inst,
                    "ops_per_cycle": ops_per_cycle,
                    "time_seconds": float(res.time_taken.value),
                    "time_display": str(res.time_taken),
                    "repetitions": int(res.num_repetitions),
                    "cycles": float(cycles.value),
                    "cycles_display": str(cycles),
                }
            )

    table = Table(title="Arithmetic Performance Summary")

    # normal output columns
    table.add_column("ISA", style="cyan")
    table.add_column("Op", style="magenta")
    table.add_column("Threads", style="magenta")
    table.add_column("GOPS", justify="right")
    table.add_column("IPC", justify="right")
    table.add_column("Frequency", justify="right")

    # extra columns for verbose output
    if print_extra:
        table.add_column("OPC", justify="right")  # ops per cycle (derived from IPC and frequency)
        table.add_column("Time (ms)", justify="right")
        table.add_column("Repetitions", justify="right")
        table.add_column("Cycles", justify="right")

    # sort rows by ISA and operation
    for metrics in sorted(arithmetic_metrics, key=lambda row: (row["isa"], row["operation"])):
        ipc = float(metrics["ipc"])

        if print_extra:
            extra_args: tuple[str, ...] | tuple[()] = (
                f"{float(metrics['ops_per_cycle']):.2f}",
                str(metrics["time_display"]),
                str(metrics["repetitions"]),
                str(metrics["cycles_display"]),
            )
        else:
            extra_args = ()

        table.add_row(
            metrics["isa"],
            metrics["operation"],
            str(metrics["threads"]),
            str(metrics["gops_display"]),
            f"{ipc:.2f}",
            str(metrics["frequency_display"]),
            *extra_args,
        )

    get_console().print(table)


def _plot_to_axis(isa_suites: dict[str, ISABenchmarkSuite], ax: Any) -> None:
    """Plot arithmetic GOPS bars to a given matplotlib axis.

    Args:
        isa_suites: Benchmark results organized by ISA
        ax: Matplotlib axis object to plot on

    Note:
        Internal helper for creating combined plots (e.g., in mixed handler).
        Assumes matplotlib is available.
    """
    gops = _collect_gops_by_isa(isa_suites)
    if not gops:
        warn("No arithmetic data available for plotting")
        return

    labels = list(gops.keys())
    values = [gops[k] for k in labels]

    # Validate we have plottable data
    if all(v == 0.0 for v in values):
        warn("All GOPS values are zero; plot may not be meaningful")

    ax.bar(labels, values)
    ax.set_ylabel("GOPS")
    ax.set_title("Arithmetic: GOPS by ISA")


def _write_plot(isa_suites: dict[str, ISABenchmarkSuite], output_path: Path | None = None) -> None:
    """Generate and save arithmetic performance plot.

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

    gops = _collect_gops_by_isa(isa_suites)
    if not gops:
        warn("No arithmetic data available for plotting")
        return

    labels = list(gops.keys())

    fig = None
    try:
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))
        _plot_to_axis(isa_suites, ax)
        plt.tight_layout()

        save_or_show_plot(output_path, "arithmetic_gops.png", plt=plt)
    except Exception as e:
        error(f"Failed to generate arithmetic plot: {e}")
    finally:
        # Explicitly close the figure to prevent matplotlib from retaining it in memory.
        if fig is not None:
            plt.close(fig)


class ArithmeticOutputHandler(OutputHandler):
    def print_table(self, context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
        _print_table(context, isa_suites)

    def write_plot(self, context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
        path = context.run_config.output_dir / "arithmetic"
        _write_plot(isa_suites, output_path=path)

    def write_csv(self, context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
        warn(NON_ROOFLINE_CSV_ERROR_MSG)

    def write_jsonl(self, context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
        from benchmark.output.jsonl import write_jsonl_benchmarks

        write_jsonl_benchmarks(context, isa_suites)
