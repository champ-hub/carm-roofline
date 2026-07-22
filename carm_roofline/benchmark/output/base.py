from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from carm_roofline.benchmark.benchmark import ISABenchmarkSuite
    from carm_roofline.context import CARMContext


class OutputKind(Enum):
    """Requested output kinds supported by the output pipeline."""

    TABLE = "table"
    PLOT = "plot"
    JSONL = "jsonl"
    CSV = "csv"


NON_ROOFLINE_CSV_ERROR_MSG = "CSV output is strictly for backward comaptibility, and is roofline mode only"


@runtime_checkable
class OutputHandler(Protocol):
    """Protocol for test-type specific output strategy implementations."""

    def handle(self, context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
        """Handle output generation for the given context and benchmark results."""
        output_kinds = context.run_config.output_formats

        if OutputKind.TABLE in output_kinds:
            self.print_table(context, isa_suites)
        if OutputKind.PLOT in output_kinds:
            self.write_plot(context, isa_suites)
        if OutputKind.JSONL in output_kinds:
            self.write_jsonl(context, isa_suites)
        if OutputKind.CSV in output_kinds:
            self.write_csv(context, isa_suites)

    def print_table(
        self,
        context: CARMContext,
        isa_suites: dict[str, ISABenchmarkSuite],
    ) -> None:
        """Format and print CLI output."""

    def write_plot(
        self,
        context: CARMContext,
        isa_suites: dict[str, ISABenchmarkSuite],
    ) -> None:
        """Create and persist plot output."""

    def write_csv(
        self,
        context: CARMContext,
        isa_suites: dict[str, ISABenchmarkSuite],
    ) -> None:
        """Write CSV output for benchmark results."""

    def write_jsonl(
        self,
        context: CARMContext,
        isa_suites: dict[str, ISABenchmarkSuite],
    ) -> None:
        """Write JSONL output for benchmark results."""
