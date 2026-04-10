from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Union

from .data_type import DataType

__all__ = [
    "CyclicRegisterSet",
    "HelperRegisterSet",
    "RegisterCollection",
    "TypedRegisterSets",
]

strLike = Union[str, int]
RegIndexSet = Iterable[Union[strLike, tuple[int, int]]]


class RegisterCollection:
    """Base class for immutable register sets with direct indexing.

    Provides a simple mapping from indices to formatted register names.
    """

    @staticmethod
    def _index_set(indices: RegIndexSet) -> list[strLike]:
        """Convert indices and inclusive ranges to a flat list.

        Tuples are interpreted as inclusive ranges (start, end).
        """
        result: list[strLike] = []

        for idx in indices:
            if isinstance(idx, tuple):
                result += list(range(idx[0], idx[1] + 1))
            elif isinstance(idx, (int, str)):
                result += [idx]
            else:
                raise TypeError(
                    f"Index sets can only contain indices or tuples of indices. Type '{type(idx).__name__}' is not "
                    "supported"
                )

        return result

    def __init__(self, format_: str, index_set: RegIndexSet):
        """Initialize a RegisterCollection with a format string and index set.

        Args:
            format_: Format string for register names (e.g., 'r{}').
            index_set: Iterable of indices or inclusive ranges as (start, end) tuples.

        Example:
            ```
            RegisterCollection('r{}', [0, (1, 3), "9"]) # can also contain strings, not just ints
            # creates registers: r0, r1, r2, r3, r9
            ```
        """
        self.indices: list[strLike] = self._index_set(index_set)
        self.register_format: str = format_

    def __getitem__(self, idx: int) -> str:
        """Allows indexing into the register set, returning a formatted register."""
        return self.register_format.format(self.indices[idx])

    def __iter__(self) -> Iterator[str]:
        """Iterator over formatted register names in the set."""
        for idx in self.indices:
            yield self.register_format.format(idx)


class CyclicRegisterSet(RegisterCollection):
    """Register set with cycling state for distributing operations across registers.

    This class maintains a running index that automatically cycles through the register
    set when calling get(). This is useful for benchmark register sets where operations
    need to be distributed across multiple registers to avoid dependencies.
    """

    def __init__(self, format_: str, index_set: RegIndexSet):
        """Initialize a CyclicRegisterSet with a format string and index set.

        Args:
            format_: Format string for register names (e.g., 'r{}').
            index_set: Iterable of indices or inclusive ranges as (start, end) tuples.

        Example:
            ```
            CyclicRegisterSet('r{}', [0, (1, 3), "9"]) # can also contain strings, not just ints
            # creates registers: r0, r1, r2, r3, r9
            ```
        """
        super().__init__(format_, index_set)
        self.running_index: int = 0

    def get(self) -> str:
        """
        Returns a formatted string for the current register, based on the `running_index`, which is then incremented.

        Repeatedly calling this method will cycle through all the registers in the set.
        """
        formatted = self[self.running_index]
        self.running_index = (self.running_index + 1) % len(self.indices)
        return formatted


class HelperRegisterSet(RegisterCollection):
    """Named register collection for loop helpers, pointers, and counters.

    Helper registers don't need cycling behavior - they have fixed semantic roles:
    outer/inner iterators, pointer, and pointer increment. This class simply provides
    convenient named access to specific registers in the set.
    """

    def __init__(self, format_: str, index_set: RegIndexSet):
        super().__init__(format_, index_set)

        if len(self.indices) < 5:
            raise ValueError(f"{HelperRegisterSet.__name__} does not have enough registers, 5 are required")

        self.outer_iterator = self[0]
        "Holds the iterator for the outer loop"
        self.inner_iterator = self[1]
        "Holds the iterator for the inner loop"
        self.pointer = self[2]
        "Holds the pointer to the read (load) buffer"
        self.pointer_increment = self[3]
        "Holds the increment to the pointer (for when memory instructions don't have an offset)"
        self.write_pointer = self[4]
        "Holds the pointer to the write (store) buffer"


class TypedRegisterSets:
    """Maps data types to their corresponding cyclic register sets for benchmarking."""

    def __init__(self, typed_reg_sets: dict[DataType, CyclicRegisterSet]):
        """
        Associates each `DataType` with a `CyclicRegisterSet`.

        Args:
            typed_reg_sets (dict[DataType, CyclicRegisterSet]): A dictionary mapping data types to cyclic register sets.
                Each register set will be used to cycle through registers when generating benchmark code.
        """
        for data_type, reg_set in typed_reg_sets.items():
            DataType.check_validity(data_type)
            if not isinstance(reg_set, CyclicRegisterSet):
                raise TypeError(
                    f"Type '{type(reg_set).__name__}' is not recognized as a '{CyclicRegisterSet.__name__}'"
                )

        self.typed_reg_sets = typed_reg_sets

    def __getitem__(self, arg: DataType) -> CyclicRegisterSet:
        return self.typed_reg_sets[arg]
