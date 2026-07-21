from __future__ import annotations
import inspect


def get_file_line() -> str:
    return f"{inspect.stack()[1][1]}:{inspect.stack()[1][2]}"
