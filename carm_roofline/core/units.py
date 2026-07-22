"""Type-safe wrappers for unit-specific formatting and validation.

This module consolidates all unit handling across the codebase using a single
UnitRegistry instance. Each wrapper type (Bytes, Frequency, etc.) provides:
  - Strong type safety through dedicated classes
  - __str__() methods for pretty-printing (replacing format_quantity dispatch)
  - Validation to ensure correct units on construction
  - Property accessors for magnitude and unit conversion

Supported unit types:
  - Bytes: Memory sizes (cache, memory targets) with binary prefixes (KiB, MiB, GiB, ...)
  - Frequency: Clock frequencies with decimal prefixes (kHz, MHz, GHz, ...)
  - Bandwidth: Memory bandwidth in GB/s
  - Performance: Arithmetic performance in GOPS (billion OPS/sec)
  - Timing: Execution time in milliseconds
  - ArithmeticIntensity: FLOP/byte ratio (unitless)
"""

from __future__ import annotations

import re
from abc import ABC
from argparse import ArgumentTypeError
from typing import Any, Generic, TypeVar, overload

# Type variable for the magnitude (int or float)
T = TypeVar("T", int, float)
S = TypeVar("S", bound="Unit[Any]")


class Unit(ABC, Generic[T]):
    """Base class for units with automatic prefix selection for display."""

    _base_unit: str
    _prefixes: list[tuple[T, str]]
    _unit_value: T

    def __init__(self, value: T) -> None:
        """Initialize with a value in base units.

        Args:
            value: The magnitude in base units (bytes, Hz, etc.)
        """
        self._value: T = value

    @property
    def value(self) -> T:
        """Return the raw value in base units."""
        return self._value

    def __float__(self) -> float:
        """Coerce to float (works for both int and float magnitudes)."""
        return float(self._value)

    def __int__(self) -> int:
        """Coerce to int (IntUnit returns exact value; FloatUnit truncates toward zero)."""
        return int(self._value)

    def _select_prefix(self) -> tuple[T, str]:
        """Select the most appropriate prefix for the current value.

        Returns:
            Tuple of (divisor, prefix_string)
        """
        abs_value = abs(self._value)

        for threshold, prefix in self._prefixes:
            if abs_value >= threshold:
                return threshold, prefix

        # No prefix needed (base unit)
        return self._unit_value, ""

    @classmethod
    def from_string(cls: type[S], s: str) -> S:
        """Parse a value from a string like '2.5 GiB' or '100 MHz'.

        Args:
            s: String representation with optional prefix and unit

        Returns:
            New instance with parsed value

        Raises:
            ValueError: If string format is invalid or unit doesn't match
        """
        # Extract number and unit parts
        match = re.match(r"^\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*(.*)$", s.strip())
        if not match:
            raise ValueError(f"Invalid format: {s}")

        number_str, unit_str = match.groups()
        number = float(number_str)

        # Normalize unit string (remove spaces, lowercase for matching)
        unit_str = unit_str.strip().replace(" ", "")

        # Try to match against base unit + prefixes
        base_unit_lower = cls._base_unit.lower()

        # Check if it ends with base unit
        if not unit_str.lower().endswith(base_unit_lower):
            raise ValueError(f"Unit must end with '{cls._base_unit}' (got: {unit_str})")

        # Extract prefix part
        prefix_str = unit_str[: -len(cls._base_unit)].strip()

        # Find matching prefix
        if not prefix_str:
            # No prefix - base unit
            multiplier = cls._unit_value
        else:
            multiplier = None
            for threshold, prefix in cls._prefixes:
                if prefix.lower() == prefix_str.lower():
                    multiplier = threshold
                    break

        if multiplier is None:
            raise ValueError(f"Unknown prefix: {prefix_str}")

        # Calculate base value
        base_value = number * multiplier

        # Return proper instance (cast to int for IntUnit subclasses)
        if isinstance(cls._unit_value, int):
            return cls(int(base_value))
        else:
            return cls(base_value)

    @classmethod
    def from_argparse(cls: type[S], arg: str) -> S:
        """Convenience method for argparse type parsing."""
        try:
            return cls.from_string(arg)
        except ValueError as e:
            raise ArgumentTypeError(str(e)) from e

    def __str__(self) -> str:
        """Return human-readable string with appropriate prefix."""
        divisor, prefix = self._select_prefix()

        # Divide by the threshold to get the scaled value
        scaled_value = self._value / divisor

        # Show integer if it's a whole number (or very close to one)
        if abs(scaled_value - round(scaled_value)) < 0.01:
            return f"{round(scaled_value)} {prefix}{self._base_unit}"
        else:
            return f"{scaled_value:.2f} {prefix}{self._base_unit}"

    def __rich__(self) -> str:
        return self.__str__()

    def __repr__(self) -> str:
        """Return unambiguous representation."""
        return f"{self.__class__.__name__}({self._value})"

    def __hash__(self) -> int:
        return hash(self._value)

    def __eq__(self, other: object) -> bool:
        """Check equality based on value."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._value == other._value

    def __lt__(self, other: Unit[T]) -> bool:
        """Less than comparison."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._value < other._value

    def __le__(self, other: Unit[T]) -> bool:
        """Less than or equal comparison."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._value <= other._value

    def __gt__(self, other: Unit[T]) -> bool:
        """Greater than comparison."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._value > other._value

    def __ge__(self, other: Unit[T]) -> bool:
        """Greater than or equal comparison."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._value >= other._value

    def __add__(self: S, other: Unit[T]) -> S:
        """Add two units."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.__class__(self._value + other._value)

    def __sub__(self: S, other: Unit[T]) -> S:
        """Subtract two units."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.__class__(self._value - other._value)

    def __mul__(self: S, other: int | float) -> S:
        """Multiply unit by a scalar.

        Returns the same concrete `Unit` subclass as `self`.
        """
        return type(self)(self._value * other)

    def __truediv__(self: S, other: int | float) -> S:
        """Divide unit by a scalar.

        Returns the same concrete `Unit` subclass as `self`.
        """
        return type(self)(self._value / other)

    def __floordiv__(self: S, other: int | float) -> S:
        """Floor divide unit by a scalar.

        Returns the same concrete `Unit` subclass as `self`.
        """
        return type(self)(self._value // other)


@overload
def _make_prefixes(
    base: int, large_prefixes: list[str], small_prefixes: list[str] | None = None
) -> list[tuple[int, str]]: ...


@overload
def _make_prefixes(
    base: float, large_prefixes: list[str], small_prefixes: list[str] | None = None
) -> list[tuple[float, str]]: ...


def _make_prefixes(
    base: int | float, large_prefixes: list[str], small_prefixes: list[str] | None = None
) -> list[tuple[int, str]] | list[tuple[float, str]]:
    """Generate prefix list from base and prefix strings.

    Args:
        base: 1000 for SI/decimal, 1024 for binary
        large_prefixes: Prefixes for base^1, base^2, base^3, ...
        small_prefixes: Prefixes for base^-1, base^-2, base^-3, ...
    """

    result = []

    # Large prefixes (descending order for selection logic)
    for i, prefix in enumerate(reversed(large_prefixes), start=1):
        power = len(large_prefixes) - i + 1
        result.append((base**power, prefix))

    # Small prefixes (if any) - also descending
    if small_prefixes:
        result.append((base**0, ""))  # Base unit with no prefix

        for i, prefix in enumerate(small_prefixes, start=1):
            result.append((base**-i, prefix))

    return result


class IntUnit(Unit[int]):
    """Base class for integer units."""

    _unit_value = 1

    def __init__(self, value: int) -> None:
        self._value = int(value)


class FloatUnit(Unit[float]):
    """Base class for float units."""

    _unit_value = 1.0

    def __init__(self, value: float) -> None:
        self._value = float(value)


class Bytes(IntUnit):
    """Represents a quantity of bytes with binary prefixes (KiB, MiB, etc.)."""

    _base_unit = "B"
    _prefixes = _make_prefixes(1024, ["Ki", "Mi", "Gi", "Ti"])

    @classmethod
    def from_kibibytes(cls, kib: int) -> Bytes:
        """Create a Bytes instance from kibibytes."""
        return cls(kib * 1024)


class Operations(IntUnit):
    """Represents a quantity of operations."""

    _base_unit = "OPS"
    _prefixes = _make_prefixes(1000, ["k", "M", "G", "T"])


class ArithmeticIntensity(FloatUnit):
    """Represents arithmetic intensity in OPS/byte."""

    _base_unit = "FLOP/B"
    _prefixes = _make_prefixes(1000.0, ["k", "M", "G", "T"])

    @classmethod
    def from_ops_per_byte(cls, ops: Operations, bytes_: Bytes) -> ArithmeticIntensity:
        """Create an ArithmeticIntensity instance from operations per byte."""
        return cls(ops.value / bytes_.value)


class Seconds(FloatUnit):
    """Represents a duration in seconds with decimal prefixes"""

    _base_unit = "s"
    _prefixes = _make_prefixes(1000.0, ["k", "M", "G", "T"], ["m", "u", "n", "p"])

    @classmethod
    def from_milliseconds(cls, ms: float) -> Seconds:
        """Create a Seconds instance from milliseconds."""
        return cls(ms / 1000.0)


class Bandwidth(FloatUnit):
    """Represents memory bandwidth in GB/s with decimal prefixes."""

    _base_unit = "B/s"
    _prefixes = _make_prefixes(1000.0, ["k", "M", "G", "T"])

    @classmethod
    def from_bytes_per_second(cls, bytes_: Bytes, seconds: Seconds) -> Bandwidth:
        """Create a Bandwidth instance from bytes per second."""
        return cls(bytes_._value / seconds._value)


class Performance(FloatUnit):
    """Represents performance in arithmetic operations per second."""

    _base_unit = "OPS/s"
    _prefixes = _make_prefixes(1000.0, ["k", "M", "G", "T"])

    @classmethod
    def from_ops_per_second(cls, ops: Operations, seconds: Seconds) -> Performance:
        """Create a Performance instance from operations per second."""
        return cls(ops.value / seconds.value)


class Frequency(FloatUnit):
    """Represents a frequency with decimal prefixes (kHz, MHz, etc.)."""

    _base_unit = "Hz"
    _prefixes = _make_prefixes(1000.0, ["k", "M", "G", "T"])

    def as_gigahertz(self) -> float:
        """Return the frequency in GHz."""
        return self._value / 1e9

    def as_kilohertz(self) -> float:
        """Return the frequency in kHz."""
        return self._value / 1000


class Cycles(IntUnit):
    """Represents a number of CPU cycles."""

    _base_unit = ""
    _prefixes = _make_prefixes(1000, ["k", "M", "G", "T"])

    @classmethod
    def from_time_and_frequency(cls, time: Seconds, frequency: Frequency) -> Cycles:
        """Calculate cycles from time and frequency."""
        return cls(int(time._value * frequency._value))


__all__ = [
    "Bytes",
    "Cycles",
    "Frequency",
    "Operations",
    "Performance",
]
