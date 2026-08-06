from __future__ import annotations

import argparse
import json
import os
import tempfile
from enum import Enum, auto
from pathlib import Path

from carm_roofline.arguments import InsertsArguments, add_verbose_argument
from carm_roofline.core.error import UserError
from carm_roofline.gui.data import GUISettings
from carm_roofline.output_utils import warn
from carm_roofline.paraver import ParaverWindowMode, default_legend_path, parse_paraver_header
from carm_roofline.results_paths import default_results_root


class GUIConfig(InsertsArguments):
    """GUI launch configuration and argument parsing."""

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.verbose: int = args.verbose
        self.results_dir: Path = args.results_dir
        self.gui_host: str = args.gui_host
        self.gui_port: int = args.gui_port
        self.gui_debug: bool = args.gui_debug
        self.paraver_trace: Path | None = args.paraver_trace

        # Paraver-mode arguments.
        self.paraver_window_csv: Path | None = args.paraver_window_csv
        self.paraver_use_semantic_window: bool = args.paraver_use_semantic_window

        # Validate Paraver-only requirements.
        self._validate_paraver_args()

    @staticmethod
    def insert_arguments(parser: argparse.ArgumentParser) -> None:
        add_verbose_argument(parser)
        parser.add_argument(
            "--results-dir",
            type=Path,
            default=default_results_root(),
            help="Results directory root (default: <user-data-directory>/carm)",
        )
        parser.add_argument("--gui-host", default="0.0.0.0", help="Host address for the Dash server")
        parser.add_argument("--gui-port", type=int, default=8050, help="Port for the Dash server")
        parser.add_argument("--gui-debug", action="store_true", help="Enable Dash debug mode")
        parser.add_argument(
            "--paraver-trace",
            type=Path,
            default=None,
            help="Path to the paraver trace file; enables Paraver GUI mode when given",
        )
        parser.add_argument(
            "--paraver-window-csv",
            type=Path,
            default=None,
            help="Path to the Paraver window/mask CSV (required when --paraver-trace is given)",
        )
        parser.add_argument(
            "--paraver-use-semantic-window",
            action="store_true",
            help="Initialize time slider to the window CSV's semantic extent",
        )

    def _validate_paraver_args(self) -> None:
        """Validate Paraver-only CLI requirements, raising UserError for invalid combos."""
        if self.paraver_trace is None:
            return

        if self.paraver_window_csv is None:
            raise UserError("--paraver-window-csv is required when --paraver-trace is given")
        if not self.paraver_trace.is_file():
            raise UserError(f"paraver trace file not found: {self.paraver_trace}")
        if not self.paraver_window_csv.is_file():
            raise UserError(f"paraver window CSV not found: {self.paraver_window_csv}")
        # Code-mode windows require the derived legend CSV; gradient mode needs none.
        with open(self.paraver_window_csv, encoding="utf-8") as fh:
            header = parse_paraver_header(fh.readline().strip())
        if ParaverWindowMode.from_header(header.window_mode) == ParaverWindowMode.CODE:
            legend = default_legend_path(self.paraver_window_csv)
            if not legend.is_file():
                raise UserError(f"paraver legend CSV not found: {legend}")


class GUIMode(Enum):
    """Dashboard UI mode: CARM (benchmark points) or PARAVER (external trace)."""

    CARM = auto()
    PARAVER = auto()

    @property
    def show_app_dropdown(self) -> bool:
        """Whether the apps dropdown (benchmark applications) is shown."""
        return self is GUIMode.CARM

    @property
    def show_time_slider(self) -> bool:
        """Whether the Paraver time-window slider is shown."""
        return self is GUIMode.PARAVER

    @property
    def has_export_tab(self) -> bool:
        """Whether the Paraver export tab is available."""
        return self is GUIMode.PARAVER


def gui_settings_path() -> Path:
    """Return the path to the GUI settings JSON file."""
    from platformdirs import user_config_dir

    return Path(user_config_dir("carm", appauthor=None)) / "gui-settings.json"


def load_gui_settings(path: Path) -> GUISettings:
    """Load GUI settings from a JSON file, returning defaults on failure."""
    try:
        data = path.read_text()
    except FileNotFoundError:
        return GUISettings()
    except OSError as exc:
        warn(f"Failed to read GUI settings from {path}: {exc}")
        return GUISettings()

    try:
        return GUISettings.from_dict(json.loads(data))
    except json.JSONDecodeError as exc:
        warn(f"Corrupt GUI settings file {path}: {exc}")
        return GUISettings()


def save_gui_settings(path: Path, settings: GUISettings) -> None:
    """Save GUI settings to a JSON file, creating parent directories as needed.

    Uses an atomic write pattern (temp file + rename) to prevent file corruption
    when multiple Dash callbacks write settings concurrently.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(settings.to_dict(), f, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise
