"""CARM test bench module.

This module provides the test execution infrastructure for CARM microbenchmarks,
including code generation utilities for creating the microbenchmarks.h header file
and compilation/execution utilities for building and running the benchmark binary.
"""

from __future__ import annotations

from .builder import MicrobenchmarkFunctionSpec, compile_test_bench, create_microbenchmark_header, run_microbenchmarks

__all__ = ["MicrobenchmarkFunctionSpec", "compile_test_bench", "create_microbenchmark_header", "run_microbenchmarks"]
