from __future__ import annotations

from enum import Enum
from typing import Union


class MemoryLayoutMode(Enum):
    """Resolved benchmark array allocation strategy."""

    single = "single"
    split = "split"

    def __str__(self) -> str:
        return self.value


class AutoMemoryLayout(Enum):
    """Configuration marker resolved using a test-specific default."""

    auto = "auto"

    def __str__(self) -> str:
        return self.value


MemoryLayoutSetting = Union[MemoryLayoutMode, AutoMemoryLayout]


def parse_memory_layout_setting(value: str) -> MemoryLayoutSetting:
    """Parse a textual layout setting into its auto marker or resolved mode."""
    try:
        return AutoMemoryLayout(value)
    except ValueError:
        return MemoryLayoutMode(value)
