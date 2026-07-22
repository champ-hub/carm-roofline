"""Tests for type-safe unit wrappers (__float__, __int__ coercion)."""

from __future__ import annotations

from typing import Any

import pytest

from carm_roofline.core import (
    ArithmeticIntensity,
    Bandwidth,
    Bytes,
    Cycles,
    Frequency,
    Operations,
    Performance,
    Seconds,
    Unit,
)

pytestmark = pytest.mark.unit


class TestFloatCoercion:
    """__float__ on Unit base covers all subclasses."""

    @pytest.mark.parametrize(
        "unit, expected",
        [
            (Bytes(1024), 1024.0),
            (Operations(1000), 1000.0),
            (Cycles(100), 100.0),
            (Frequency(3.0e9), 3.0e9),
            (Performance(1.5e9), 1.5e9),
            (Bandwidth(1.0e9), 1.0e9),
            (Seconds(0.5), 0.5),
            (ArithmeticIntensity(2.0), 2.0),
        ],
    )
    def test_float_coercion(self, unit: Unit[Any], expected: float) -> None:
        assert float(unit) == expected

    def test_float_converts_int_magnitude(self) -> None:
        """IntUnit subclasses also get __float__ via the Unit base."""
        b = Bytes(4096)
        result = float(b)
        assert isinstance(result, float)
        assert result == 4096.0

    def test_float_on_zero(self) -> None:
        assert float(Bytes(0)) == 0.0
        assert float(Frequency(0.0)) == 0.0

    def test_float_returns_python_float_not_unit(self) -> None:
        result = float(Bytes(8))
        assert type(result) is float
        assert not hasattr(result, "_value")


class TestIntCoercion:
    """__int__ on IntUnit (Bytes, Operations, Cycles) — exact value via _value."""

    @pytest.mark.parametrize(
        "unit, expected",
        [
            (Bytes(1024), 1024),
            (Operations(1000), 1000),
            (Cycles(1364391136), 1364391136),
        ],
    )
    def test_int_on_intunit(self, unit: Unit[Any], expected: int) -> None:
        assert int(unit) == expected

    def test_int_preserves_exact_value(self) -> None:
        """No float->int truncation path for IntUnit subclasses."""
        b = Bytes(1025)
        assert int(b) == 1025  # would be 1024 if going through float


class TestIntOnFloatUnit:
    """__int__ on FloatUnit (Frequency, Performance, Bandwidth, ...) — truncates."""

    @pytest.mark.parametrize(
        "unit, expected",
        [
            (Frequency(3.0e9), 3000000000),
            (Performance(1.5e9), 1500000000),
            (Bandwidth(999.999), 999),
            (Seconds(0.5), 0),
            (ArithmeticIntensity(2.99), 2),
        ],
    )
    def test_int_on_floatunit(self, unit: Unit[Any], expected: int) -> None:
        assert int(unit) == expected

    def test_int_truncates_toward_zero(self) -> None:
        """Like built-in int(float), __int__ on FloatUnit truncates toward zero."""
        assert int(Seconds(1.9)) == 1
        assert int(Seconds(-1.9)) == -1


class TestCoercionDoesNotMutate:
    """Calling float()/int() must not alter the wrapped value."""

    def test_float_does_not_alter(self) -> None:
        b = Bytes(2048)
        _ = float(b)
        assert b.value == 2048

    def test_int_does_not_alter(self) -> None:
        c = Cycles(42)
        _ = int(c)
        assert c.value == 42


class TestRoundtrip:
    """float()/int()/str() compose sensibly."""

    def test_from_int_to_float_to_int(self) -> None:
        """IntUnit -> float() -> int() preserves magnitude."""
        b = Bytes(8192)
        assert int(float(b)) == 8192

    def test_floatunit_int_round_trip_float_only(self) -> None:
        """A FloatUnit whose value has fractional part loses it through int()."""
        f = Performance(1234.5)
        assert int(f) == 1234
        assert float(f) == 1234.5
