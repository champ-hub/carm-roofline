from __future__ import annotations

import argparse
from pathlib import Path

from arguments import InsertsArguments


class GUIConfig(InsertsArguments):
    """GUI launch configuration and argument parsing."""

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.results_dir: Path = args.results_dir
        self.gui_host: str = args.gui_host
        self.gui_port: int = args.gui_port
        self.gui_debug: bool = args.gui_debug

    @staticmethod
    def insert_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--results-dir",
            type=Path,
            default=Path("carm_results"),
            help="Results directory root (default: carm_results, relative to current working directory)",
        )
        parser.add_argument("--gui-host", default="127.0.0.1", help="Host address for the Dash server")
        parser.add_argument("--gui-port", type=int, default=8050, help="Port for the Dash server")
        parser.add_argument("--gui-debug", action="store_true", help="Enable Dash debug mode")
