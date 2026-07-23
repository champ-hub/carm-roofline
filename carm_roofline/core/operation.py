from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto
from typing import Union

from carm_roofline.core import DataType


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


class TensorCoreOperation(Enum):
    """Tensor/matrix core MMA operation (separate from vector ArithmeticOperation)."""

    mma = auto()

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class TensorOperation:
    """Descriptor for a tensor/matrix core operation with precision and tile dimensions.

    ``TensorOperation`` is a dataclass (not an enum member) because tensor operations
    vary along multiple axes: precision triple, tile size, and FLOP count.
    """

    name: str
    precision_triple: tuple[DataType, DataType, DataType]  # (A_type, B_type, C_type)
    tile_mnk: tuple[int, int, int]  # (M, N, K)
    flops_per_mma: int  # = 2 * M * N * K

    def __post_init__(self) -> None:
        if any(d <= 0 for d in self.tile_mnk):
            raise ValueError(f"Tile dimensions must be positive: {self.tile_mnk}")
        if self.flops_per_mma <= 0:
            raise ValueError(f"flops_per_mma must be positive: {self.flops_per_mma}")


Operation = Union[ArithmeticOperation, MemoryOperation, TensorCoreOperation]
