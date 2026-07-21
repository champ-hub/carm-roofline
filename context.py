from __future__ import annotations

from dataclasses import dataclass

from architecture import Architecture
from benchmark import Benchmarking, TestType
from exec_interface import ExecutionInterface
from run_config import RunConfig

__all__ = ["Architecture", "Benchmarking", "CARMContext", "ExecutionInterface", "RunConfig", "TestType"]


@dataclass
class CARMContext:
    """Holds global context for the tool"""

    architecture: Architecture
    benchmarking: Benchmarking
    exec_interface: ExecutionInterface
    run_config: RunConfig

    def benchmark_is_native(self) -> bool:
        """Check if the given ISA is native to the architecture."""
        return self.exec_interface.sim_cmd is None
