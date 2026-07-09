"""Benchmark result data structures and output formatting.

This module defines:
- BenchmarkMetadata: Configuration details for a benchmark
- BenchmarkResult: Spec + metadata + execution results
- Output formatters for CSV and modern formats
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from core import Seconds

if TYPE_CHECKING:
    from .benchmark import BaseBenchmark


class OutputFormat(Enum):
    """Supported output formats."""

    CSV = "csv"  # Legacy format for compatibility
    JSON = "json"  # Modern JSON format
    TABLE = "table"  # Human-readable table format
    JSONL = "jsonl"  # JSON Lines (streaming format)


def _parse_benchmark_result_line(line: str) -> tuple[str, Seconds, int]:
    """
    Parse a single line of benchmark output in the expected CSV format.
    Expected format: "benchmark_name, runtime_ms, num_repetitions"

    Args:
        line: A string containing the benchmark result line.
    Returns:
        A tuple of (benchmark_name, runtime as Seconds, num_repetitions as int
    """
    parts = [p.strip() for p in line.split(",")]

    if len(parts) != 3:
        raise ValueError(f"Expected 3 fields in result line, got {len(parts)}: {line}")

    function_name, runtime_ms_str, num_repetitions_str = parts

    try:
        runtime = Seconds.from_milliseconds(float(runtime_ms_str))
        num_repetitions = int(num_repetitions_str)
    except ValueError as e:
        raise ValueError(
            f"Failed to parse numeric fields in result line: {line}. "
            f"runtime_ms must be float, num_repetitions must be int."
        ) from e

    return function_name, runtime, num_repetitions


def parse_benchmark_output(benchmarks: dict[str, BaseBenchmark], output: str) -> None:
    """
    Parse benchmark output and populate results for known benchmarks.

    This function processes a multi-line benchmark output string, extracts benchmark
    results from each line, and stores them in the corresponding Benchmark objects.

    Args:
        benchmarks: A dictionary mapping benchmark names to Benchmark objects to be populated.
        output: A string containing benchmark results, with one result per line.

    Raises:
        RuntimeError: If a benchmark result is received for a benchmark name that is not
                      present in the benchmarks dictionary.

    Note:
        - Empty lines are skipped automatically.
        - Invalid lines that cannot be parsed are silently skipped (logged as debug).
        - Each valid line is expected to contain benchmark name, runtime in milliseconds,
          and number of repetitions, as parsed by _parse_benchmark_result_line().
    """
    for line in output.strip().split("\n"):
        stripped_line = line.strip()
        if stripped_line:  # Skip empty lines
            try:
                name, seconds, num_repetitions = _parse_benchmark_result_line(stripped_line)
            except ValueError:
                continue  # Skip invalid lines (debug logging)

            if name not in benchmarks:
                raise RuntimeError(f"Received result for unknown benchmark: {name}")
            benchmark = benchmarks[name]
            benchmark.process_results(seconds, num_repetitions)
