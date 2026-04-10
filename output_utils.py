"""Lightweight printing helpers with verbosity control.

This module centralizes user-facing output. Verbosity is stored in a
context variable so it can be temporarily overridden without mutating
module globals. Levels follow the existing convention (0-4):
 0 - no output
 1 - errors/warnings
 2 - test results
 3 - configuration/details
 4 - debug
"""

from __future__ import annotations

import contextvars
import inspect
import os
import sys
from enum import IntEnum
from typing import Any

from rich.console import Console
from rich.theme import Theme


# Console instances for stdout and stderr with custom theme
_theme = Theme(
    {
        "error": "bold red",
        "warn": "bold yellow",
        "info": "cyan",
        "detail": "green",
        "debug": "dim white",
    }
)
_console_out = Console(theme=_theme, file=sys.stdout, highlight=False, markup=False)
_console_err = Console(theme=_theme, file=sys.stderr, highlight=False, markup=False)

def get_console() -> Console:
    """Get the appropriate console based on current verbosity (errors to stderr, others to stdout)."""

    return _console_out


class Ansi:
    """Minimal ANSI color helpers.

    Deprecated: Kept for backward compatibility. Use Rich styles instead.
    """

    ERROR = "\033[91m"
    WARN = "\033[93m"
    INFO = "\033[96m"
    DETAIL = "\033[92m"
    DEBUG = "\033[90m"
    RESET = "\033[0m"


class Verbosity(IntEnum):
    QUIET = 0
    ERROR = 1
    RESULT = 2
    CONFIG = 3
    DEBUG = 4


# ContextVar keeps the verbosity scoped to the current context (thread/task).
_verbosity: contextvars.ContextVar[int] = contextvars.ContextVar("verbosity", default=Verbosity.CONFIG.value)


def set_verbosity(level: int | Verbosity) -> contextvars.Token[int]:
    """Set the current verbosity level and return the reset token."""

    level_int = int(level)
    if level_int < 0:
        raise ValueError("verbosity must be non-negative")
    return _verbosity.set(level_int)


def reset_verbosity(token: contextvars.Token[int]) -> None:
    """Reset verbosity using a previously returned token."""

    _verbosity.reset(token)


def configure_verbosity(level: int | Verbosity | None) -> contextvars.Token[int] | None:
    """Set verbosity if ``level`` is provided, otherwise leave unchanged."""

    if level is None:
        return None
    return set_verbosity(level)


def get_verbosity() -> int:
    """Return the current verbosity as an int."""

    return _verbosity.get()


def should_emit(level: int | Verbosity) -> bool:
    """Check whether a message at ``level`` should be printed."""

    return get_verbosity() >= int(level)


def _emit(
    level: int | Verbosity,
    *args: object,
    sep: str = " ",
    end: str = "\n",
    use_stderr: bool = False,
    prefix: str | None = None,
    style: str | None = None,
) -> None:
    """Print a message if verbosity allows it."""

    if not should_emit(level):
        return

    console = _console_err if use_stderr else _console_out

    # Build message from args
    message = sep.join(str(arg) for arg in args if str(arg) != "")

    # Prepend prefix if provided
    if prefix:
        message = f"{prefix} {message}" if message else prefix

    console.print(message, style=style, end=end)


def error(*args: object, **kwargs: Any) -> None:
    """Emit an error-level message (level 1) to stderr."""

    _emit(Verbosity.ERROR, *args, prefix="[error]", use_stderr=True, style="error", **kwargs)


def warn(*args: object, **kwargs: Any) -> None:
    """Emit a warning-level message (level 1) to stderr."""

    _emit(Verbosity.ERROR, *args, prefix="[warn]", use_stderr=True, style="warn", **kwargs)


def info(*args: object, **kwargs: Any) -> None:
    """Emit an info-level message (level 2) to stdout."""

    _emit(Verbosity.RESULT, *args, prefix="[info]", use_stderr=False, style="info", **kwargs)


def detail(*args: object, **kwargs: Any) -> None:
    """Emit a configuration/detail message (level 3) to stdout."""

    _emit(Verbosity.CONFIG, *args, prefix="[detail]", use_stderr=False, style="detail", **kwargs)


def debug(*args: object, **kwargs: Any) -> None:
    """Emit a debug-level message (level 4) to stdout."""

    # print caller file and line number
    frame = inspect.currentframe()
    if frame and frame.f_back:
        filename = frame.f_back.f_code.co_filename
        lineno = frame.f_back.f_lineno
        filename = os.path.relpath(filename)
        prefix = f"[debug] [{filename}:{lineno}]"
    else:
        prefix = "[debug]"

    _emit(Verbosity.DEBUG, *args, prefix=prefix, use_stderr=False, style="debug", **kwargs)


def format_if_container(obj: list[Any] | dict[Any, Any]) -> str:
    """Format a list or dict in a compact way, otherwise return str(obj)."""

    if isinstance(obj, dict):
        delim = ("{", "}")
        items = [f"{k}: {v}" for k, v in obj.items()]
    elif isinstance(obj, list):
        delim = ("[", "]")
        items = [str(item) for item in obj]
    else:
        return str(obj)
    return delim[0] + ", ".join(items) + delim[1]


__all__ = [
    "Ansi",
    "Verbosity",
    "configure_verbosity",
    "debug",
    "detail",
    "eprint",
    "error",
    "get_verbosity",
    "reset_verbosity",
    "set_verbosity",
    "should_emit",
    "info",
    "warn",
    "format_if_container",
]
