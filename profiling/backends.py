"""Backend interface and PAPI-based profiler implementation.

The backend interface defines how profiling output files are produced by
an instrumented application. Currently supports PAPI High-Level (HL) API.

In the current model, users instrument their own application code with PAPI HL.
This module provides:
1. A :class:`ProfilerBackend` protocol for future backends to implement.
2. The :class:`PAPIHLBackend` that wraps a user command and collects its output.
"""

from __future__ import annotations

import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from error import UserError
from output_utils import debug, detail, error, info, warn

from .metrics import (
    MetricDefinition,
    MetricResolutionConfig,
    MetricType,
    parse_available_events,
    resolve_metrics,
)


class ProfilerBackend(ABC):
    """Abstract interface for profiler backends.

    Each backend knows how to:
    - Verify its prerequisites (e.g. PAPI library, environment).
    - Run the profiled command and produce profiling output files.
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
    def run(self, command: list[str], cwd: Path | None = None) -> int:
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

        Returns a dict mapping metric type (e.g. `MetricType.DP_FLOPS`, `MetricType.BYTES`)
        to the best available :class:`MetricDefinition`.
        """


class PAPIHLBackend(ProfilerBackend):
    """PAPI High-Level API profiler backend.

    Runs the user's command with ``PAPI_HL_OUTPUT_DIR`` set so that
    PAPI HL writes per-rank output files to a known location.

    Uses ``papi_decode -a`` at startup to discover available PAPI events,
    then resolves the best metric implementations for the current system.
    """

    def __init__(
        self,
        output_dir: Path,
        resolution_config: MetricResolutionConfig | None = None,
        events_override: str | None = None,
    ) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._resolution_config = resolution_config or MetricResolutionConfig()
        self._events_override = events_override
        self._available_events: frozenset[str] = frozenset()
        self._resolved_metrics: dict[MetricType, MetricDefinition] = {}

    @property
    def resolved_metrics(self) -> dict[MetricType, MetricDefinition]:
        return dict(self._resolved_metrics)

    def check_prerequisites(self) -> bool:
        """Verify that PAPI HL is available by checking the environment.

        We check for the presence of ``libpapi`` or the ``papi_hl_read`` utility
        as a proxy for PAPI HL availability.  After that, runs ``papi_decode -a``
        to discover available events and resolve metric implementations.
        """

        # Check if the papi shared library has the PAPI_hl_region_begin symbol (indicates PAPI HL support)
        try:
            command = "nm -D $(ldconfig -p | grep libpapi.so | head -1 | awk '{print $NF}')"
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if "PAPI_hl_region_begin" not in result.stdout:
                raise UserError("PAPI HL does not appear to be installed, cannot find libpapi.so or papi_hl_read.")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Discover available events and resolve metrics
        self._available_events = parse_available_events()
        self._resolved_metrics = resolve_metrics(self._available_events, self._resolution_config)
        detail(f"Available PAPI events: {len(self._available_events)}")
        for metric_type, impl in self._resolved_metrics.items():
            detail(f"  {metric_type.name} -> {impl.description}")
            if impl.warning is not None:
                detail(f"    Note: {impl.warning}")

        return any(impl.priority < 100 for impl in self._resolved_metrics.values())

    def _build_env(self) -> dict[str, str]:
        """Build the environment with PAPI HL output dir and resolved events."""
        env = os.environ.copy()
        # Configure PAPI
        env["PAPI_OUTPUT_DIRECTORY"] = str(self._output_dir)

        if self._events_override:
            # User explicitly specified events
            env["PAPI_EVENTS"] = self._events_override
            detail(f"Using user-specified PAPI events: {self._events_override}")
        elif self._resolved_metrics:
            # Collect all required events from resolved metrics
            all_events: set[str] = set()
            for impl in self._resolved_metrics.values():
                all_events |= impl.required_events
            events_str = ",".join(sorted(all_events))
            env["PAPI_EVENTS"] = events_str
            detail(f"Resolved PAPI events ({len(all_events)}): {events_str}")
        else:
            warn("No resolved metrics are available, cannot select PAPI events.")

        return env

    def run(self, command: list[str], cwd: Path | None = None) -> int:
        env = self._build_env()
        cmd_str = " ".join(command)
        info(f"Running profiled command: {cmd_str}")
        detail(f"PAPI HL output dir: {self._output_dir}")
        debug(f"cwd: {cwd}")
        try:
            result = subprocess.run(
                command,
                env=env,
                cwd=cwd,
                capture_output=False,  # Let stdout/stderr pass through
                check=False,
            )
        except FileNotFoundError as e:
            raise UserError(f"Command not found: {command[0]}") from e
        except OSError as e:
            error(f"Failed to execute command: {e}")
            raise RuntimeError(f"Failed to execute command: {e}") from e

        debug(f"Command exit code: {result.returncode}")
        if result.returncode != 0:
            warn(f"Profiled command exited with code {result.returncode}")

        return result.returncode
