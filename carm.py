from __future__ import annotations

#!/usr/bin/env python3
import argparse
import sys
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from rich_argparse import RichHelpFormatter

from architecture import set_execution_interface
from arguments import InsertsArguments, TopLevelHelpFormatter
from benchmark.interface import run_full_benchmark
from benchmark.output import output_benchmark_results
from context import Architecture, Benchmarking, CARMContext, ExecutionInterface, RunConfig
from error import ConfigurationError
from gui.config import GUIConfig
from output_utils import configure_verbosity, debug, detail, error, info, warn

RichHelpFormatter.styles.update(
    {
        "argparse.args": "bold cyan",
        "argparse.prog": "bold cyan",
        "argparse.argument": "bold yellow",
        "argparse.groups": "bold green",
        "argparse.metavar": "cyan",
    }
)


def _handle_benchmark(args: argparse.Namespace) -> int:
    configure_verbosity(args.verbose)
    args_str = ", ".join(f"{k}={v}" for k, v in vars(args).items() if v is not None)
    debug(f"parsed benchmark arguments: {args_str}")

    # Handle --emit-config early (before expensive benchmark initialization)
    if args.emit_config:
        from architecture.config import emit_template_toml

        try:
            emit_template_toml(Path(args.emit_config))
            info(f"Generated template TOML configuration: {args.emit_config}")
            return 0
        except OSError as e:
            error(f"Failed to emit template config: {e}")
            return 1

    try:
        exec_iface = ExecutionInterface(args)
        # Make exec_iface globally available for architecture feature detection
        set_execution_interface(exec_iface)
        architecture = Architecture(args)
        benchmarking = Benchmarking(args)
        run_config = RunConfig(args)

        context = CARMContext(
            architecture=architecture, benchmarking=benchmarking, run_config=run_config, exec_interface=exec_iface
        )

        benchmark_suites = run_full_benchmark(context)

        if run_config.dry_run:
            detail("Dry run finished: benchmark code generated successfully.")
            return 0

        output_benchmark_results(context, benchmark_suites)
        return 0

    except ConfigurationError as e:
        error(str(e))
        debug(str(traceback.format_exc()))
        return 1
    except Exception as e:
        error(f"Unexpected error: {e.__class__.__name__}: {e}")
        debug(str(traceback.format_exc()))
        return 1


def _handle_gui(args: argparse.Namespace) -> int:
    gui_config = GUIConfig(args)
    results_dir = gui_config.results_dir.resolve()

    try:
        from gui import dashboard

        dashboard.set_results_root(results_dir)
        dashboard.app.run(host=gui_config.gui_host, port=gui_config.gui_port, debug=gui_config.gui_debug)
        return 0
    except ImportError:
        error('Failed to import GUI dependencies. Install optional GUI extras with: pip install "carm-roofline[gui]"')
        debug(str(traceback.format_exc()))
        return 1
    except Exception as e:
        error(f"Unexpected error: {e.__class__.__name__}: {e}")
        debug(str(traceback.format_exc()))
        return 1


def _handle_profile(_args: argparse.Namespace) -> int:
    warn("The 'profile' subcommand is not implemented yet. Use PMU_AI_Calculator.py or DBI_AI_Calculator.py for now.")
    return 1


@dataclass
class SubParserInfo:
    name: str
    help_short: str
    help_long: str
    handler: Callable[[argparse.Namespace], int]
    arg_inserting_modules: Sequence[type[InsertsArguments]] = ()


SUBPARSERS = (
    SubParserInfo(
        name="benchmark",
        help_short="Run CARM benchmarks",
        help_long="Run benchmarks to construct the CARM or related metrics",
        handler=_handle_benchmark,
        arg_inserting_modules=(Architecture, Benchmarking, RunConfig, ExecutionInterface),
    ),
    SubParserInfo(
        name="gui",
        help_short="Launch GUI mode",
        help_long="Launch the interactive CARM GUI to plot rooflines and applications",
        handler=_handle_gui,
        arg_inserting_modules=(GUIConfig,),
    ),
    SubParserInfo(
        name="profile",
        help_short="Profile applications (TBI)",
        help_long="Profile applications with CARM metrics (TBI)",
        handler=_handle_profile,
        arg_inserting_modules=(),
    ),
)


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="carm",
        description="CARM: Cache-Aware Roofline Model benchmarking and visualization toolkit",
        formatter_class=TopLevelHelpFormatter,
        epilog="Run 'carm <command> --help' for command-specific options.",
    )

    subparsers = parser.add_subparsers(
        dest="mode",
        title="Commands",
        metavar="<COMMAND>",
    )

    for subparser in SUBPARSERS:
        sp = subparsers.add_parser(
            subparser.name,
            help=subparser.help_short,
            description=subparser.help_long,
            formatter_class=RichHelpFormatter,
        )
        sp.set_defaults(handler=subparser.handler)
        for module in subparser.arg_inserting_modules:
            module.insert_arguments(sp)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _create_parser()
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) == 0:
        parser.print_help()
        return 0

    args = parser.parse_args(args=argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 2

    handler = cast(Callable[[argparse.Namespace], int], args.handler)
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
