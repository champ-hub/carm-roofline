"""Profile subcommand configuration and argument parsing."""

from __future__ import annotations

import argparse
from enum import Enum
from pathlib import Path

from arguments import InsertsArguments, enum_action
from benchmark.generation import ISA_NAME_TO_CLASS, DataType
from results_paths import default_results_root


class BackendType(Enum):
    """Supported profiler backends."""

    PAPI = "papi"
    PERF = "perf"


class AggregationMode(Enum):
    """Supported aggregation strategies for multi-rank profiling results."""

    GLOBAL = "global"
    RANK = "rank"
    THREAD = "thread"
    REGION = "region"


class ProfileConfig(InsertsArguments):
    """Configuration for the profile subcommand.

    Attributes:
        command: The application command to profile (everything after --).
        backend: Profiler backend to use (papi or perf).
        aggregation: Aggregation strategy for multi-rank results.
        output_dir: Directory for output files.
        verbose: Verbosity level (0-4).
        name: Name prefix for output files.
        results_dir: Directory to scan for existing profiling result files.
        keep_artifacts: Whether to keep raw profiling output files.
        papi_events: Optional comma-separated PAPI event override.
        perf_events: Optional comma-separated perf event override.
        perf_interval: Sampling interval in ms for perf interval mode (None = full-run).
        isa: Dominant ISA for metric calculation.
        data_type: Dominant data type for metric calculation.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.command: list[str] = list(args.command)
        self.backend: BackendType = args.backend
        self.aggregation: AggregationMode = args.aggregation
        self.output_dir: Path = args.output_dir
        self.verbose: int = args.verbose
        self.name: str = args.name or "unnamed"
        self.results_dir: Path = args.results_dir
        self.keep_artifacts: bool = args.keep_artifacts
        self.papi_events: str | None = args.papi_events
        self.perf_events: str | None = args.perf_events
        self.perf_interval: int | None = args.perf_interval
        self.isa = ISA_NAME_TO_CLASS.get(args.isa)
        self.data_type: DataType = args.data_type

    @staticmethod
    def insert_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "command",
            nargs="*",
            help="Application command to profile (everything after -- is passed verbatim)",
        )
        parser.add_argument(
            "--backend",
            default=BackendType.PAPI,
            action=enum_action(BackendType),
            help=f"Profiler backend to use (default: {BackendType.PAPI.value})",
        )
        parser.add_argument(
            "--verbose",
            "-v",
            default=3,
            const=4,
            nargs="?",
            type=int,
            choices=(0, 1, 2, 3, 4),
            help="Level of detail of terminal output (0 -> None, 4 -> Debug info)",
        )
        parser.add_argument(
            "--aggregation",
            default=AggregationMode.GLOBAL,
            action=enum_action(AggregationMode),
            help=f"Aggregation strategy for multi-rank results (default: {AggregationMode.GLOBAL.value})",
        )
        parser.add_argument(
            "--name",
            default=None,
            type=str,
            help="Name for the results directory (default: auto-generated from CPU model)",
        )
        parser.add_argument(
            "--output-dir",
            default=default_results_root(),
            type=Path,
            help="Directory to write result files (default: platform user data dir for app 'carm')",
        )
        parser.add_argument(
            "--results-dir",
            default=default_results_root() / "profile",
            type=Path,
            help="Directory to scan for existing profiling result files (default: <output-dir>/profile)",
        )
        parser.add_argument(
            "--keep-artifacts",
            action="store_true",
            help="Keep raw profiling output files in the temporary directory after execution",
        )
        parser.add_argument(
            "--papi-events",
            default=None,
            type=str,
            help="Comma-separated PAPI event list override (default: auto-resolved from papi_decode -a)",
        )
        parser.add_argument(
            "--perf-events",
            default=None,
            type=str,
            help="Comma-separated perf event list override (default: auto-resolved from perf list -j)",
        )
        parser.add_argument(
            "--perf-interval",
            default=None,
            type=int,
            help="Sampling interval in milliseconds for perf interval mode (default: full-run, no -I flag)",
        )
        parser.add_argument(
            "--isa",
            default=None,
            choices=list(ISA_NAME_TO_CLASS.keys()),
            help="Dominant ISA to assume for metric calculation. If ideal counters are not present, this will be used "
            "to estimate the number of ops/bytes per instruction.",
        )
        parser.add_argument(
            "--data-type",
            default=DataType.f32,
            action=enum_action(DataType),
            help="Data type to assume for metric calculation (default: f32)",
        )
