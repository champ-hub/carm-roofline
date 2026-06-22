"""Mixed output handler: combines arithmetic and memory summaries."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from output_utils import error, info, warn

from . import arithmetic, memory
from .base import NON_ROOFLINE_CSV_ERROR_MSG, OutputHandler
from .common import (
    safe_matplotlib_import,
    save_or_show_plot,
)

if TYPE_CHECKING:
    from benchmark.benchmark import ISABenchmarkSuite
    from context import CARMContext


def _print_table(context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
    info("Mixed test summary:")


def _write_plot(isa_suites: dict[str, ISABenchmarkSuite], output_path: Path | None = None) -> None:
    """Generate and save combined arithmetic and memory plot.

    Creates a single figure with two side-by-side subplots:
    - Left: Arithmetic GFLOPS by ISA
    - Right: Memory bandwidth by ISA

    Args:
        isa_suites: Benchmark results organized by ISA
        output_path: Directory to save plot (None = display interactively)

    Note:
        Gracefully handles errors from sub-handlers. Will not raise exceptions.
        If either subplot fails, the other will still be displayed.
    """
    plt, _ = safe_matplotlib_import()
    if plt is None:
        return

    fig = None
    try:
        # Create a single figure with 2 subplots side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Plot arithmetic GFLOPS to left subplot
        try:
            arithmetic._plot_to_axis(isa_suites, axes[0])
        except Exception as e:
            warn(f"Failed to generate arithmetic subplot in mixed plot: {e}")
            axes[0].text(0.5, 0.5, "Arithmetic data unavailable", ha="center", va="center", transform=axes[0].transAxes)

        # Plot memory bandwidth to right subplot
        try:
            memory._plot_to_axis(isa_suites, axes[1])
        except Exception as e:
            warn(f"Failed to generate memory subplot in mixed plot: {e}")
            axes[1].text(0.5, 0.5, "Memory data unavailable", ha="center", va="center", transform=axes[1].transAxes)

        plt.suptitle("Mixed test: Arithmetic + Memory")
        plt.tight_layout()
        save_or_show_plot(output_path, "mixed_summary.png", plt=plt)

    except Exception as e:
        error(f"Failed to generate mixed summary plot: {e}")
    finally:
        # Explicitly close the figure to prevent matplotlib from retaining it in memory.
        if fig is not None:
            plt.close(fig)


class MixedOutputHandler(OutputHandler):
    """Output handler for mixed benchmarks."""

    def print_table(self, context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
        _print_table(context, isa_suites)

    def write_plot(self, context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
        path = context.run_config.output_dir / "mixed"
        _write_plot(isa_suites, output_path=path)

    def write_csv(self, context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
        warn(NON_ROOFLINE_CSV_ERROR_MSG)

    def write_jsonl(self, context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
        from benchmark.output.jsonl import write_jsonl_benchmarks

        write_jsonl_benchmarks(context, isa_suites)
