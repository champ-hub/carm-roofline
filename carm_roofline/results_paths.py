from __future__ import annotations

import os
import pwd
from pathlib import Path

from platformdirs import user_cache_dir, user_data_dir


def _platformdirs_dir(kind: str) -> Path:
    """Return the expanded XDG directory of *kind* for CARM."""
    if kind == "data":
        return Path(user_data_dir("carm", appauthor=None)).expanduser()
    return Path(user_cache_dir("carm", appauthor=None)).expanduser()


def _user_platformdirs_dir(kind: str) -> Path:
    """Return the XDG user directory of *kind* for CARM.

    When running under sudo, this returns the original (invoking) user's
    directory rather than root's, so CARM files land in the right place.
    """
    sudo_uid = os.environ.get("SUDO_UID")
    if sudo_uid is not None:
        try:
            pw = pwd.getpwuid(int(sudo_uid))
            old_home = os.environ.get("HOME")
            os.environ["HOME"] = pw.pw_dir
            try:
                return _platformdirs_dir(kind)
            finally:
                if old_home is not None:
                    os.environ["HOME"] = old_home
                else:
                    os.environ.pop("HOME", None)
        except (KeyError, ValueError, PermissionError, OSError):
            pass
    return _platformdirs_dir(kind)


def user_data_dir_for_results() -> Path:
    """Return the XDG user data directory for CARM results."""
    return _user_platformdirs_dir("data")


def user_cache_dir_for_carm() -> Path:
    """Return the XDG user cache directory for CARM."""
    return _user_platformdirs_dir("cache")


def default_results_root() -> Path:
    return user_data_dir_for_results()
