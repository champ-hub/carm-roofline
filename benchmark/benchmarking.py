"""Benchmark configuration and argument parsing. For data structures representing benchmark results, see

benchmark/benchmark.py."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum

from arguments import InsertsArguments, enum_action, positive_float, positive_int
from benchmark.generation.code_gen.operation import ArithmeticOperation
from output_utils import warn
from units import Operations

from .generation import DataType


class TestType(Enum):
    """Benchmark test types."""

    ARITHMETIC = "arithmetic"
    MEMORY = "memory"
    ROOFLINE = "roofline"
    MIXED = "mixed"
    MEMORY_SWEEP = "memory_sweep"


@dataclass
class LoadStoreRatio:
    loads: int
    stores: int


def ld_st_ratio_type(arg: str) -> LoadStoreRatio:
    """Parse a load-store ratio specifier.

    Supported forms:
      - "ld"    -> (ld, 1) meaning ld loads per store
      - "ld:st" -> (ld, st) explicit loads:stores

    Returns a tuple (loads, stores) where loads and/or stores may be 0.
    Raises argparse.ArgumentTypeError on invalid input.
    """
    s = str(arg).strip().lower()

    # Parse two intergers ("N:M" form)
    if ":" in s:
        parts = s.split(":")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"invalid ld-st ratio: {arg!r}")
        try:
            ld = int(parts[0])
            st = int(parts[1])
            if ld < 0 or st < 0:
                raise argparse.ArgumentTypeError(f"invalid ld-st ratio, values must be non-negative: {arg!r}")
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid ld-st ratio, could not parse integers: {arg!r}") from None
    else:
        try:
            ld = positive_int(s)
            st = 1
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid ld-st ratio: {arg!r}") from None

    return LoadStoreRatio(ld, st)


def mem_test_size_type(arg: str) -> int | str:
    """Parse a memory test size specifier.

    Accepts either:
      - A positive integer (KiB)
      - The string "auto" (case-insensitive) for automatic sizing

    Returns:
        int (KiB value) or str (literal "auto")

    Raises:
        argparse.ArgumentTypeError on invalid input.
    """
    s = str(arg).strip().lower()
    if s == "auto":
        return "auto"
    try:
        return positive_int(arg)
    except (ValueError, argparse.ArgumentTypeError) as e:
        raise argparse.ArgumentTypeError(
            f"invalid memory test size: {arg!r}. Must be a positive integer or 'auto'"
        ) from e


class Benchmarking(InsertsArguments):
    """Benchmark configuration and argument parsing."""

    num_ops: Operations

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.test: TestType = args.test
        self.mem_target: str = args.mem_target
        self.data_type: DataType = args.data_type
        self.threads: int = args.threads
        self.interleaved: bool = args.interleaved
        self.instructions: set[ArithmeticOperation] = set(args.instruction)
        self.num_ops: Operations = Operations(args.num_ops)
        self.ld_st_ratio: LoadStoreRatio = args.ld_st_ratio
        self.arith_mem_ratio: int = args.arith_mem_ratio
        self.mem_test_sizes: list[int | str] | None = args.mem_test_sizes
        self.verbose: int = args.verbose
        self.test_time: float = args.test_time
        if self.test_time < 10.0:
            warn(
                f"Target test time {self.test_time:.1f}s may be too low for accurate measurements. Consider using a "
                f"higher value (e.g. 25s) for more reliable results."
            )

    @staticmethod
    def insert_arguments(parser: argparse.ArgumentParser) -> None:
        # The specific memory level target has been moved to --mem-target
        parser.add_argument(
            "-t",
            "--test",
            default=TestType.ROOFLINE,
            action=enum_action(TestType),
            help="Type of the test. 'arithmetic' measures the performance of arithmetic operations, 'memory' "
            "measures the bandwidth of various memory sizes, 'roofline' combines both tests to construct the "
            "roofline model, 'mixed' measures bandwidth and FP performance for a combination of "
            "memory accesses.",
        )
        parser.add_argument(
            "-m",
            "--mem-target",
            default="all",
            nargs="?",
            choices=["L1", "L2", "L3", "DRAM", "all"],
            help="Target memory level for 'memory' and 'mixed' tests (Default: all)",
        )
        parser.add_argument(
            "-o",
            "--num-ops",
            default=32 * 1024,
            nargs="?",
            type=positive_int,
            help="Number of arithmetic operations to perform in the arithmetic test (Default: 32768)",
        )
        parser.add_argument(
            "-d",
            "--data-type",
            default=DataType.f32,
            nargs="?",
            action=enum_action(DataType),
            help="Data type for benchmark operations (Default: f32)",
        )
        parser.add_argument(
            "--threads", default=1, nargs="?", type=positive_int, help="Number of threads to benchmark (Default: 1)"
        )
        parser.add_argument(
            "--interleaved",
            action="store_true",
            help="Optimize thread affinity for NUMA systems "
            "where the core domain is interleaved (e.g. node 0 has cores 0, 2, 4, ...)",
        )
        parser.add_argument(
            "--instruction",
            default=[ArithmeticOperation.add, ArithmeticOperation.fma],
            nargs="+",
            action=enum_action(ArithmeticOperation),
            help="Arithmetic instruction(s) to benchmark (Default: add fma)",
        )
        parser.add_argument(
            "--ld-st-ratio",
            default=LoadStoreRatio(2, 1),
            nargs="?",
            type=ld_st_ratio_type,
            help="Load-to-store ratio for memory access patterns. Format: 'LD:ST' (e.g., '2:1') or a single "
            "integer for 'N:1' ratio. Use '1:0' for load-only or '0:1' for store-only tests. (Default: 2:1)",
        )
        parser.add_argument(
            "--arith-mem-ratio",
            default=2,
            nargs="?",
            type=positive_int,
            help="Ratio between arithmetic and memory operations for 'mixed' test (Default: 2)",
        )
        parser.add_argument(
            "--mem-test-sizes",
            nargs="*",
            type=mem_test_size_type,
            help="Size of the test arrays for each memory level in KiB. 'auto' can be used to automatically determine "
            "sizes based on the cache sizes, e.g. '--mem-test-sizes 32 256 2048 65536' / '--mem-test-sizes auto auto "
            "2048 auto' for L1, L2, L3, DRAM tests respectively. (Default: auto for all levels)",
        )
        parser.add_argument(
            "--test-time",
            default=25.0,
            nargs="?",
            type=positive_float,
            help="Target runtime for each individual microbenchmark, in seconds. Low runtime may lead to inaccurate "
            "results. (Default: 25.0)",
        )
