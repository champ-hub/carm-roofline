from __future__ import annotations

from pathlib import Path

import pytest

from carm_roofline.architecture.architecture import _validate_detected_memory_topology
from carm_roofline.architecture.memory import MemoryTopology
from carm_roofline.benchmark import TestType as BenchmarkTestType
from carm_roofline.core import UserError

pytestmark = pytest.mark.unit


def _topology_without_cache_metadata(tmp_path: Path) -> MemoryTopology:
    cpu_root = tmp_path / "cpu"
    cpu_root.mkdir()
    (cpu_root / "online").write_text("0\n")
    return MemoryTopology(cpu_root)


def test_arithmetic_allows_detected_topology_without_cache_metadata(tmp_path: Path) -> None:
    topology = _topology_without_cache_metadata(tmp_path)

    _validate_detected_memory_topology(topology, BenchmarkTestType.ARITHMETIC)


@pytest.mark.parametrize(
    "test",
    [BenchmarkTestType.MEMORY, BenchmarkTestType.ROOFLINE, BenchmarkTestType.MIXED, BenchmarkTestType.MEMORY_SWEEP],
)
def test_memory_tests_reject_detected_topology_without_cache_metadata(
    tmp_path: Path, test: BenchmarkTestType
) -> None:
    topology = _topology_without_cache_metadata(tmp_path)

    with pytest.raises(UserError, match="sysfs does not expose any data-cache levels"):
        _validate_detected_memory_topology(topology, test)
