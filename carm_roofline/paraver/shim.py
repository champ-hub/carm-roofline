"""Paraver compatibility entry point; translates the legacy Paraver CLI to ``carm gui``.

Paraver's plugin launches ``Paraver_CARM.py`` with the legacy tool's CLI
(``[--color_csv] [--mask_csv] [-ac] [--min_dur N] --csv <mask.csv> <trace.prv>``).
This thin, removable shim translates that argv into ``carm gui`` argv and
delegates to :func:`carm_roofline.carm.main`. Nothing else in the codebase
imports this module, so removal is exactly two deletions: this file and the
``[project.scripts]`` entry.
"""

from __future__ import annotations

import argparse
import importlib.metadata
from collections.abc import Sequence

from carm_roofline.carm import main as carm_main
from carm_roofline.output_utils import warn


def _carm_version() -> str:
    return importlib.metadata.version("carm-roofline")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Paraver_CARM.py",
        description="Paraver compatibility entry point; translates the legacy Paraver CLI to 'carm gui'",
    )
    parser.add_argument("-v", "--version", action="version", version=f"Paraver_CARM version {_carm_version()}")
    parser.add_argument("--min_dur", type=float, default=None)
    parser.add_argument("--color_csv", action="store_true")
    parser.add_argument("--mask_csv", action="store_true")
    parser.add_argument("-ac", action="store_true")
    parser.add_argument("--csv", required=True, help="Path to the mask CSV")
    parser.add_argument("trace_path", help="Path to the .prv file")
    parser.add_argument("-d", "--debug", action="store_true")
    return parser


def translate_args(args: argparse.Namespace) -> list[str]:
    """Translate legacy Paraver argv into ``carm gui`` argv."""
    translated = ["gui", "--paraver-trace", args.trace_path, "--paraver-window-csv", args.csv]
    if args.mask_csv:
        translated.append("--paraver-use-semantic-window")
    if args.debug:
        # -v must stay the final element so carm gui's nargs="?" never swallows a following option.
        translated.append("-v")
    if args.ac:
        warn("--ac (accumulate mode) is not supported by CARM; ignoring")
    if args.min_dur is not None:
        warn(f"--min_dur {args.min_dur} (duration filter) is not supported by CARM; ignoring")
    return translated


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return carm_main(translate_args(args))
