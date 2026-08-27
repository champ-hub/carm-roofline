"""Mixed benchmark output."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from rich.table import Table

from carm_roofline.output_utils import get_console, warn

from .base import NON_ROOFLINE_CSV_ERROR_MSG, OutputHandler
from .common import safe_matplotlib_import, save_or_show_plot

if TYPE_CHECKING:
    from carm_roofline.benchmark.benchmark import ISABenchmarkSuite, MixedBenchmark
    from carm_roofline.context import CARMContext


def _rows(isa_suites: dict[str, ISABenchmarkSuite]) -> list[tuple[str, MixedBenchmark]]:
    rows: list[tuple[str, MixedBenchmark]] = []
    for isa, suite in isa_suites.items():
        for benchmark in suite.get_mixed_benchmarks().values():
            if benchmark.results is not None:
                rows.append((isa, benchmark))
    return sorted(
        rows,
        key=lambda item: (
            item[0],
            item[1].cache_level,
            item[1].params.data_type.name,
            item[1].params.operation.name,
            item[1].params.num_threads,
            item[1].params.load_store_ratio.loads,
            item[1].params.load_store_ratio.stores,
            item[1].params.point_index,
        ),
    )


def _print_table(context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
    table = Table(title="Mixed benchmark results")
    for column in (
        "ISA",
        "Memory Level",
        "Data Type",
        "Operation",
        "Threads",
        "Load:Store",
        "Requested AI",
        "Achieved AI",
        "Performance",
    ):
        table.add_column(column)
    for isa, benchmark in _rows(isa_suites):
        results = benchmark.results
        assert results is not None
        ratio = benchmark.params.load_store_ratio
        table.add_row(
            isa,
            benchmark.cache_level,
            benchmark.params.data_type.name,
            benchmark.params.operation.name,
            str(benchmark.params.num_threads),
            f"{ratio.loads}:{ratio.stores}",
            f"{float(benchmark.params.requested_arithmetic_intensity):g}",
            f"{float(results.arithmetic_intensity):g}",
            f"{float(results.performance) / 1e9:.3f} GOPS",
        )
    get_console().print(table)


def _write_plot(isa_suites: dict[str, ISABenchmarkSuite], output_path: Path | None = None) -> None:
    plt = safe_matplotlib_import()
    if plt is None:
        return
    series: dict[tuple[str, str, str, str, int, int, int], list[tuple[float, float]]] = defaultdict(list)
    for isa, benchmark in _rows(isa_suites):
        results = benchmark.results
        assert results is not None
        ratio = benchmark.params.load_store_ratio
        key = (
            isa,
            benchmark.cache_level,
            benchmark.params.data_type.name,
            benchmark.params.operation.name,
            benchmark.params.num_threads,
            ratio.loads,
            ratio.stores,
        )
        series[key].append((float(results.arithmetic_intensity), float(results.performance) / 1e9))
    fig, axis = plt.subplots()
    for key, points in series.items():
        points.sort()
        axis.plot(*zip(*points), marker="o", label=" ".join(map(str, key)))
    axis.set_xscale("log", base=2)
    axis.set_yscale("log", base=2)
    axis.set_xlabel("Arithmetic intensity (FLOP/B)")
    axis.set_ylabel("Performance (GOPS)")
    axis.legend()
    save_or_show_plot(output_path, "mixed_performance.png", plt=plt)
    plt.close(fig)


class MixedOutputHandler(OutputHandler):
    """Output handler for mixed benchmarks."""

    def print_table(self, context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
        _print_table(context, isa_suites)

    def write_plot(self, context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
        _write_plot(isa_suites, context.run_config.output_dir / "mixed")

    def write_csv(self, context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
        warn(NON_ROOFLINE_CSV_ERROR_MSG)

    def write_jsonl(self, context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
        from carm_roofline.benchmark.output.jsonl import write_jsonl_benchmarks

        write_jsonl_benchmarks(context, isa_suites)
