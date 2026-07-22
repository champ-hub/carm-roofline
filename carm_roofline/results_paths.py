from __future__ import annotations

from pathlib import Path

from platformdirs import user_data_dir


def user_data_dir_for_results() -> Path:
    return Path(user_data_dir("carm", appauthor=None)).expanduser()


def default_results_root() -> Path:
    return user_data_dir_for_results()
