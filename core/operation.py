from __future__ import annotations

from collections.abc import Mapping
from enum import Enum, auto
from typing import Union


class ArithmeticOperation(Enum):
    """Enumeration of arithmetic operations supported by benchmarks."""

    add = auto()
    mul = auto()
    div = auto()
    fma = auto()

    def ops(self) -> int:
        """Returns the number of operations performed by this instruction."""
        return _ARITHMETIC_OPS_COUNT[self]

    def __repr__(self) -> str:
        return self.name


_ARITHMETIC_OPS_COUNT: Mapping[ArithmeticOperation, int] = {
    ArithmeticOperation.add: 1,
    ArithmeticOperation.mul: 1,
    ArithmeticOperation.div: 1,
    ArithmeticOperation.fma: 2,
}


class MemoryOperation(Enum):
    """Enumeration of memory operations supported by benchmarks."""

    ld = auto()
    "Regular load from memory"
    st = auto()
    "Regular store to memory"
    sst = auto()
    "Streaming store to memory (non-temporal)"

    def __repr__(self) -> str:
        return self.name


Operation = Union[ArithmeticOperation, MemoryOperation]
