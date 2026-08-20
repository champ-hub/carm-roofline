"""Backend interface for profiler backends.

Defines the :class:`ProfilerBackend` abstract base class, the per-run
:class:`RunSpec`/:class:`RunResult` value objects, and the
:func:`create_backend` factory function.  Concrete backend implementations
live in :mod:`~.papi_backend` and :mod:`~.perf_backend`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from carm_roofline.core import UserError

from .config import ProfileConfig
from .shared import (
    BackendType,
    MetricDefinition,
    MetricResolutionConfig,
    MetricType,
)

if TYPE_CHECKING:
    from .model import RankMetrics


@dataclass(frozen=True)
class RunSpec:
    """Parameters for one profiled execution of the application.

    A backend executes one run per :class:`RunSpec`; the backend itself is
    session-scoped (one instance per ``carm profile`` invocation).
    """

    output_dir: Path
    """Directory for this run's profiling output files."""

    events: str | None = None
    """Comma-separated event list to collect, or None for the backend's resolved events."""


@dataclass(frozen=True)
class RunResult:
    """Outcome of one profiled execution: the exit code plus parsed ranks."""

    exit_code: int
    """Exit code of the profiled command."""

    ranks: list[RankMetrics]
    """Rank metrics parsed from this run's profiling output files."""


class ProfilerBackend(ABC):
    """Abstract interface for profiler backends.

    A backend is session-scoped: one instance per ``carm profile`` invocation.
    It verifies prerequisites and resolves metrics via
    :meth:`check_prerequisites` (once), then executes each application run via
    :meth:`profile` with a per-run :class:`RunSpec`.
    """

    @abstractmethod
    def check_prerequisites(self) -> bool:
        """Check that all prerequisites for this backend are met.

        This is the one-time session probe: it discovers the event catalog and
        resolves metric implementations. Call it once before :meth:`profile`.

        Raises:
            UserError: If prerequisites are not met (e.g. required library not found).

        Returns:
            True if the available metrics are ideal (e.g. exact FLOP counts), False if approximations are required.
        """

    @abstractmethod
    def profile(self, run_spec: RunSpec, command: list[str], cwd: Path) -> RunResult:
        """Run the profiled command and parse its profiling output.

        Args:
            run_spec: Per-run parameters (output directory, requested events).
            command: The full application command (including launcher if any).
            cwd: Optional working directory for the command.

        Returns:
            The command exit code together with the parsed rank metrics.
        """

    @property
    @abstractmethod
    def resolved_metrics(self) -> dict[MetricType, MetricDefinition]:
        """Metric implementations resolved for this system.

        Returns a dict mapping metric type (e.g. `MetricType.FLOPS`, `MetricType.BYTES`)
        to the best available :class:`MetricDefinition`.
        """

    @property
    @abstractmethod
    def available_events(self) -> frozenset[str]:
        """Hardware events available on this system.

        Returns the set of event names the backend can collect, populated by
        :meth:`check_prerequisites`.
        """

    @abstractmethod
    def can_collect(self, events: frozenset[str]) -> bool:
        """Return True if *events* can be collected together in one run.

        Args:
            events: Candidate event set to validate.

        Returns:
            True when all events fit in one run, False when the backend would
            have to drop or multiplex events and a partitioned run is needed.
        """

    @property
    @abstractmethod
    def run_method_name(self) -> str:
        """Human-readable label for the run metadata 'method' field."""


def create_backend(
    config: ProfileConfig,
    resolution_cfg: MetricResolutionConfig,
) -> ProfilerBackend:
    """Factory: create the profiler backend for the given configuration.

    The returned backend is session-scoped: it discovers and resolves metrics
    once via :meth:`ProfilerBackend.check_prerequisites`, then executes each
    application run via :meth:`ProfilerBackend.profile`.

    Args:
        config: Resolved profile configuration.
        resolution_cfg: Metric resolution configuration.

    Returns:
        Initialized :class:`ProfilerBackend` instance.

    Raises:
        UserError: If *config.backend* is unknown.
    """
    if config.backend == BackendType.PAPI:
        from .papi_backend import PAPIHLBackend

        return PAPIHLBackend(
            resolution_config=resolution_cfg,
            use_cache=config.use_papi_cache,
        )
    elif config.backend == BackendType.PERF:
        from .perf_backend import PerfBackend

        return PerfBackend(
            resolution_config=resolution_cfg,
            interval_ms=config.perf_interval,
        )
    raise UserError(f"Unknown backend: {config.backend}")
