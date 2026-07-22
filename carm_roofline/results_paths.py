from __future__ import annotations

import os
import pwd
from pathlib import Path

from platformdirs import user_data_dir


def user_data_dir_for_results() -> Path:
    """Return the XDG user data directory for CARM results.

    When running under sudo, this returns the original (invoking) user's
    data directory rather than root's, so benchmark output files land in
    the right place.
    """
    sudo_uid = os.environ.get("SUDO_UID")
    if sudo_uid is not None:
        try:
            pw = pwd.getpwuid(int(sudo_uid))
            old_home = os.environ.get("HOME")
            os.environ["HOME"] = pw.pw_dir
            try:
                return Path(user_data_dir("carm", appauthor=None)).expanduser()
            finally:
                if old_home is not None:
                    os.environ["HOME"] = old_home
                else:
                    os.environ.pop("HOME", None)
        except (KeyError, ValueError, PermissionError, OSError):
            pass
    return Path(user_data_dir("carm", appauthor=None)).expanduser()


def default_results_root() -> Path:
    return user_data_dir_for_results()
