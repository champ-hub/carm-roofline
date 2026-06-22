from __future__ import annotations

import argparse
from pathlib import Path

from arguments import InsertsArguments, enum_action
from benchmark.output.base import OutputKind
from output_utils import warn
from results_paths import default_results_root


class RunConfig(InsertsArguments):
    """General configuration for the run."""

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.verbose: int = args.verbose
        self.name: str = args.name
        self.dry_run: bool = args.dry_run
        self.output_dir: Path = args.output_dir
        self.output_formats: set[OutputKind] = set(args.output_fmt)
        self.keep_artifacts: bool = args.keep_artifacts
        if self.dry_run:
            warn("Dry run enabled: no benchmarks will be executed.")

    @staticmethod
    def insert_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--verbose",
            "-v",
            default=3,
            const=4,
            nargs="?",
            type=int,
            choices=(0, 1, 2, 3, 4),
            help="Level of detail of terminal output (0 -> None 1 -> Only ISA/configuration errors and test "
            "specifications, 2 -> Test results, 3 -> Configuration values selected/detected, 4 -> Debug info)",
        )
        parser.add_argument("--name", default="unnamed", nargs="?", type=str, help="Name of the results file")
        parser.add_argument(
            "--dry-run", action="store_true", help="Only generate the benchmark code, do not compile or run tests"
        )
        parser.add_argument(
            "--output-dir",
            default=default_results_root(),
            type=Path,
            help="Directory to write result files (default: platform user data dir for app 'carm')",
        )
        parser.add_argument(
            "--output-fmt",
            nargs="+",
            action=enum_action(OutputKind),
            default={OutputKind.TABLE, OutputKind.PLOT, OutputKind.JSONL, OutputKind.CSV},
            help="Output format(s): table, plot, jsonl, csv (default: table plot jsonl csv)",
        )
        parser.add_argument(
            "--keep-artifacts",
            action="store_true",
            help="Keep generated benchmark artifacts (source files, binaries) in the temporary directory after "
            "execution",
            default=False,
        )
