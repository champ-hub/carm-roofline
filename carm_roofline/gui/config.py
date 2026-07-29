from __future__ import annotations

import argparse
import json
from pathlib import Path

from carm_roofline.arguments import InsertsArguments, add_verbose_argument
from carm_roofline.gui.data import GUISettings
from carm_roofline.output_utils import warn
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
    """Save GUI settings to a JSON file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings.to_dict(), sort_keys=True))
