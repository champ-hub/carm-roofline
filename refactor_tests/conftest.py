from __future__ import annotations
import os
import sys
from unittest.mock import Mock

import pytest

# Add parent directory to path for module imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from benchmark.generation import X86AVX, X86SSE, ArmNeon, RISCVScalar
from architecture import SimpleMemoryTopology
from units import Bytes, Frequency


@pytest.fixture
def x86avx_isa():
    """X86 AVX ISA instance for testing."""
    return X86AVX()


@pytest.fixture
def x86sse_isa():
    """X86 SSE ISA instance for testing."""
    return X86SSE()


@pytest.fixture
def arm_neon_isa():
    """ARM NEON ISA instance for testing."""
    return ArmNeon()


@pytest.fixture
def riscv_scalar_isa():
    """RISC-V scalar ISA instance for testing."""
    return RISCVScalar()


@pytest.fixture
def mock_context():
    """Create a minimal mock CARMContext for ISA generation tests."""
    context = Mock()

    # Mock architecture with necessary attributes
    context.architecture = Mock()
    # Explicit instance counts for deterministic cache coverage in tests.
    context.architecture.memory_topology = SimpleMemoryTopology(
        level_sizes=[
            Bytes.from_string("32KiB"),
            Bytes.from_string("256KiB"),
            Bytes.from_string("8MiB"),
        ],
        instances_per_level=[8, 4, 1],
        total_cpus=8,
        smt_degree=1,
    )

    # Mock get_frequency_for_isa to return a frequency
    def mock_get_frequency(isa_name: str) -> Frequency:
        return Frequency(3.0e9)  # 3 GHz

    context.architecture.get_frequency_for_isa = mock_get_frequency

    return context
