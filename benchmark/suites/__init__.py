"""Benchmark suite classes for organizing benchmarks by type."""

from __future__ import annotations

from .arithmetic import ArithmeticBenchmarkSuite
from .base import ISABenchmarkSuite
from .memory import MemoryBenchmarkSuite
from .memory_sweep import MemorySweepBenchmarkSuite
from .mixed import MixedBenchmarkSuite
from .roofline import RooflineBenchmarkSuite

__all__ = [
    "ArithmeticBenchmarkSuite",
    "ISABenchmarkSuite",
    "MemoryBenchmarkSuite",
    "MemorySweepBenchmarkSuite",
    "MixedBenchmarkSuite",
    "RooflineBenchmarkSuite",
]
