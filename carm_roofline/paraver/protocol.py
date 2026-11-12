"""Synchronous requests to the Paraver host."""

from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from threading import Lock


class ParaverCommand(str, Enum):
    GENERATE_CSV = "generate_csv"
    OPEN_CSV = "open_csv"


_PREFIX = "PARAVER_PROTOCOL"
_LOCK = Lock()


def command_line(command: ParaverCommand, *paths: Path) -> str:
    """Format one Paraver protocol command."""
    return " ".join((_PREFIX, command.value, *(str(path) for path in paths)))


def request(command: ParaverCommand, *paths: Path) -> None:
    """Send a request and wait for its corresponding completion response."""
    arguments = " ".join(str(path) for path in paths)
    with _LOCK:
        print(command_line(command, *paths), flush=True)  # noqa: T201 ; protocol stdout
        try:
            response = sys.stdin.readline()
        except OSError as exc:
            raise RuntimeError(
                f"Failed to read Paraver protocol response for {command.value} {arguments}: {exc}"
            ) from exc
        if not response:
            raise RuntimeError(f"Paraver protocol ended before a response for {command.value} {arguments}")
        if not response.endswith("\n"):
            raise RuntimeError(f"Malformed Paraver protocol response for {command.value} {arguments}: {response!r}")
        fields = response.split()
        if fields == [_PREFIX, "error"]:
            raise RuntimeError(f"Paraver protocol failed to {command.value} {arguments}")
        if fields != [_PREFIX, "succeed"]:
            raise RuntimeError(
                f"Malformed Paraver protocol response for {command.value} {arguments}: {response.rstrip()!r}"
            )
