"""Backend interface for profiler backends.

Defines the :class:`ProfilerBackend` abstract base class and the
:func:`create_backend` factory function.  Concrete backend implementations
live in :mod:`~.papi_backend` and :mod:`~.perf_backend`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from core import UserError

from .config import BackendType, ProfileConfig
from .shared import (
    MetricDefinition,
    MetricResolutionConfig,
    MetricType,
)

if TYPE_CHECKING:
    from .model import RankMetrics


class ProfilerBackend(ABC):
    """Abstract interface for profiler backends.

    Each backend knows how to:
    - Verify its prerequisites (e.g. PAPI library, environment).
    - Run the profiled command and produce profiling output files.
    - Parse its output files into the rank/thread/region model.
    """

    @abstractmethod
    def check_prerequisites(self) -> bool:
        """Check that all prerequisites for this backend are met.

        Raises:
            UserError: If prerequisites are not met (e.g. required library not found).

        Returns:
            True if the available metrics are ideal (e.g. exact FLOP counts), False if approximations are required.
        """

    @abstractmethod
    def run(self, command: list[str], cwd: Path) -> int:
        """Run the profiled command and collect profiling output.

        Args:
            command: The full application command (including launcher if any).
            cwd: Optional working directory for the command.

        Returns:
            Exit code of the command.
        """

    @property
    @abstractmethod
    def resolved_metrics(self) -> dict[MetricType, MetricDefinition]:
        """Metric implementations resolved for this system.

        Returns a dict mapping metric type (e.g. `MetricType.FLOPS`, `MetricType.BYTES`)
        to the best available :class:`MetricDefinition`.
        """

    @abstractmethod
    def parse_output(self) -> list[RankMetrics]:
        """Parse the profiling output files produced by :meth:`run`.

        Returns:
            List of RankMetrics parsed from the backend's output format.
        """

    @property
    @abstractmethod
    def run_method_name(self) -> str:
        """Human-readable label for the run metadata 'method' field."""


def create_backend(
    config: ProfileConfig,
    workspace: Path,
    resolution_cfg: MetricResolutionConfig,
) -> ProfilerBackend:
    """Factory: create the appropriate profiler backend for the given configuration.

    Args:
        config: Resolved profile configuration.
        workspace: Temporary workspace directory for profiling output.
        resolution_cfg: Metric resolution configuration.

    Returns:
        Initialized :class:`ProfilerBackend` instance.

    Raises:
        UserError: If *config.backend* is unknown.
    """
    if config.backend == BackendType.PAPI:
        from .papi_backend import PAPIHLBackend

        return PAPIHLBackend(workspace, resolution_config=resolution_cfg, events_override=config.papi_events)
    elif config.backend == BackendType.PERF:
        from .perf_backend import PerfBackend

        return PerfBackend(
            workspace,
            resolution_config=resolution_cfg,
            events_override=config.perf_events,
            interval_ms=config.perf_interval,
        )
    raise UserError(f"Unknown backend: {config.backend}")
