"""Profile subcommand configuration and argument parsing."""

from __future__ import annotations

import argparse
from enum import Enum
from pathlib import Path

from carm_roofline.architecture import MachineSignature, detect_machine_signature, generate_run_name
from carm_roofline.arguments import InsertsArguments, add_verbose_argument, enum_action
from carm_roofline.core import DataType
from carm_roofline.isa import BaseISA
from carm_roofline.results_paths import default_results_root


class BackendType(Enum):
    """Supported profiler backends."""

    PAPI = "papi"
    PERF = "perf"


class AggregationMode(Enum):
    """Supported aggregation strategies for multi-rank profiling results."""

    GLOBAL = "global"
    RANK = "rank"
    THREAD = "thread"
    REGION_MERGED = "region_merged"
    REGION_PER_THREAD = "region_per_thread"


def _default_app_name(command: list[str]) -> str:
    """Best-effort application name extracted from the profiled command.

    Among non-flag tokens (skipping ones starting with ``-``), prefer the last path-like token (containing ``/``) —
    launchers such as ``mpirun`` and ``srun`` pass the binary as a path (``./myapp``).  If no token has a ``/``, fall
    back to the first non-flag token (the direct-invocation case ``myapp --input foo`` → ``myapp``).  The basename is
    taken so ``./build/myapp`` → ``myapp``.  Returns ``"app"`` when the command is empty or has no non-flag token.
    """
    candidates = [tok for tok in command if tok and not tok.startswith("-")]
    if not candidates:
        return "app"
    pathlike = [tok for tok in candidates if "/" in tok]
    return Path(pathlike[-1] if pathlike else candidates[0]).name


class ProfileConfig(InsertsArguments):
    """Configuration for the profile subcommand.

    Attributes:
        command: The application command to profile (everything after --).
        backend: Profiler backend to use (papi or perf).
        aggregation: Aggregation strategy for multi-rank results.
        output_dir: Directory for output files.
        verbose: Verbosity level (0-4).
        machine_name: Name for the results directory (each machine gets its own subdirectory).
        app_name: Application name recorded in the output metadata.
        keep_artifacts: Whether to keep raw profiling output files.
        papi_events: Optional comma-separated PAPI event override.
        use_papi_cache: Whether to read/write the cached PAPI event catalog.
        perf_events: Optional comma-separated perf event override.
        perf_interval: Sampling interval in ms for perf interval mode (None = full-run).
        isas: ISA(s) the application exercises, as a tuple of BaseISA classes (empty when unspecified).
        data_type: Dominant data type for metric calculation.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.command: list[str] = list(args.command)
        self.backend: BackendType = args.backend
        self.aggregation: AggregationMode = args.aggregation
        self.verbose: int = args.verbose
        self.output_dir: Path = args.output_dir
        self.machine_signature: MachineSignature = detect_machine_signature()
        self.machine_name: str = (
            args.machine_name if args.machine_name is not None else generate_run_name(self.machine_signature)
        )
        self.app_name: str = args.app_name if args.app_name is not None else _default_app_name(args.command)
        self.keep_artifacts: bool = args.keep_artifacts
        self.papi_events: str | None = args.papi_events
        self.use_papi_cache: bool = not args.no_papi_cache
        self.perf_events: str | None = args.perf_events
        self.perf_interval: int | None = args.perf_interval
        self.isas: tuple[type[BaseISA], ...]
        if args.isa is not None:
            self.isas = tuple(BaseISA.from_name(name) for name in args.isa if BaseISA.exists(name))
        else:
            self.isas = ()
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
        add_verbose_argument(parser)
        parser.add_argument(
            "--aggregation",
            default=AggregationMode.GLOBAL,
            action=enum_action(AggregationMode),
            help=f"Aggregation strategy for multi-rank results (default: {AggregationMode.GLOBAL.value})",
        )
        parser.add_argument(
            "--machine-name",
            default=None,
            type=str,
            help="Name for the results directory (default: auto-generated from CPU model)",
        )
        parser.add_argument(
            "--app-name",
            default=None,
            type=str,
            help="Application name recorded in the output metadata (default: extracted from the profiled command)",
        )
        parser.add_argument(
            "--output-dir",
            default=default_results_root(),
            type=Path,
            help="Directory to write result files (default: platform user data dir for app 'carm')",
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
            help="Comma-separated PAPI event list override (default: auto-resolved from papi_xml_event_info)",
        )
        parser.add_argument(
            "--no-papi-cache",
            action="store_true",
            help="Do not read or write the cached PAPI event catalog (default: cache per machine configuration)",
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
            nargs="+",
            choices=BaseISA.names(),
            metavar="ISA",
            help="ISA(s) the application exercises (e.g., x86_avx2 x86_sse x86_scalar). "
            "Only the FP_ARITH counters matching these widths will be requested, "
            "leaving counter budget for PAPI_LST_INS. "
            "If ideal counters are not present, fall back to estimates.",
        )
        parser.add_argument(
            "--data-type",
            default=DataType.f32,
            action=enum_action(DataType),
            help="Data type to assume for metric calculation (default: f32)",
        )
