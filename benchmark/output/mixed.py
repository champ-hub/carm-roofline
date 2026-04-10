"""Mixed output handler: combines arithmetic and memory summaries."""

from __future__ import annotations

import json
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


def _write_json(context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
    results: dict[str, list[dict[str, object]]] = {}

    for isa, suite in sorted(isa_suites.items(), key=lambda kv: kv[0]):
        rows: list[dict[str, object]] = []
        for name, bench in sorted(suite.benchmarks.items(), key=lambda kv: kv[0]):
            res = getattr(bench, "results", None)
            if res is None:
                continue

            row: dict[str, object] = {
                "benchmark": name,
                "time_seconds": float(res.time_taken.value),
                "repetitions": int(res.num_repetitions),
            }

            performance = getattr(res, "performance", None)
            if performance is not None:
                row["performance_gflops"] = float(performance.value) / 1e9

            arithmetic_intensity = getattr(res, "arithmetic_intensity", None)
            if arithmetic_intensity is not None:
                row["arithmetic_intensity"] = float(arithmetic_intensity.value)

            bandwidth = getattr(res, "bandwidth", None)
            if bandwidth is not None:
                row["bandwidth_gbps"] = float(bandwidth.value) / 1e9

            cache_level = getattr(res, "cache_level", None)
            if cache_level is not None:
                row["cache_level"] = cache_level

            rows.append(row)

        results[isa] = rows

    out_dir = context.run_config.output_dir / "mixed"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{context.run_config.name}_mixed.json"
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump({"results": results}, json_file, indent=2, sort_keys=True)

    info(f"Mixed JSON saved to: {json_path}")


class MixedOutputHandler(OutputHandler):
    """Output handler for mixed benchmarks."""

    def print_table(self, context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
        _print_table(context, isa_suites)

    def write_plot(self, context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
        path = context.run_config.output_dir / "mixed"
        _write_plot(isa_suites, output_path=path)

    def write_csv(self, context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
        warn(NON_ROOFLINE_CSV_ERROR_MSG)

    def write_json(self, context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
        _write_json(context, isa_suites)
