"""Linux ``perf stat`` profiler backend.

Wraps the user command in ``perf stat -x,`` with auto-resolved hardware
events, supporting full-run and interval sampling modes.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from carm_roofline.core import UserError
from carm_roofline.output_utils import debug, detail, error, info, warn

from .backends import ProfilerBackend
from .model import RankMetrics
from .perf_loader import perf_csv_to_thread_metrics
from .perf_metrics import (
    parse_perf_available_events,
    resolve_perf_metrics,
)
from .shared import (
    MetricDefinition,
    MetricResolutionConfig,
    MetricType,
)


class PerfBackend(ProfilerBackend):
    """Linux ``perf stat`` profiler backend.

    Wraps the user command in ``perf stat -x,`` with auto-resolved hardware
    events, supporting two modes:

    - **Full-run**: ``perf stat -x, -e <events> -o <out_csv> <command>``
    - **Interval sampling**: ``perf stat -x, -I <ms> -e <events> -o <out_csv> <command>``

    Uses ``perf list -j`` at startup to discover available events and resolve
    the best metric implementations for the current system.
    """

    def __init__(
        self,
        output_dir: Path,
        resolution_config: MetricResolutionConfig,
        events_override: str | None = None,
        interval_ms: int | None = None,
    ) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._resolution_config = resolution_config or MetricResolutionConfig()
        self._events_override = events_override
        self._interval_ms = interval_ms
        self._available_events: frozenset[str] = frozenset()
        self._resolved_metrics: dict[MetricType, MetricDefinition] = {}

    @property
    def resolved_metrics(self) -> dict[MetricType, MetricDefinition]:
        return dict(self._resolved_metrics)

    @property
    def output_file(self) -> Path:
        """Path to the output CSV file that will be produced by perf."""
        return self._output_dir / "perf_stat.csv"

    @property
    def mode_name(self) -> str:
        """Human-readable mode name for metadata."""
        return f"perf {'interval' if self._interval_ms else 'full-run'}"

    @property
    def run_method_name(self) -> str:
        """Human-readable label for the run metadata 'method' field."""
        return self.mode_name

    def check_prerequisites(self) -> bool:
        """Verify that ``perf`` is available and discover hardware events.

        Runs ``perf list -j`` to discover available hardware events, then
        resolves the best metric implementations for this system.

        Raises:
            UserError: If ``perf`` is not found.
        """
        perf_bin = shutil.which("perf")
        if perf_bin is None:
            raise UserError(
                "perf not found. Install the 'perf' tools package (e.g. 'linux-tools-common' on Ubuntu, "
                "'perf' on Fedora)."
            )

        # Discover available events and resolve metrics
        self._available_events = parse_perf_available_events()
        self._resolved_metrics = resolve_perf_metrics(self._available_events, self._resolution_config)
        detail(f"Available perf events: {len(self._available_events)}")
        for metric_type, impl in self._resolved_metrics.items():
            detail(f"  {metric_type.name} -> {impl.description}")
            if impl.warning is not None:
                detail(f"    Note: {impl.warning}")

        if not self._resolved_metrics:
            warn("No perf metric implementations could be resolved. Flops/bytes will be zero.")
            return False

        return any(impl.priority < 100 for impl in self._resolved_metrics.values())

    def _build_command(self, command: list[str]) -> list[str]:
        """Build the perf command line.

        Constructs ``perf stat -x, -e <events> [-I <ms>] -o <out_csv> <command>``.
        """
        perf_bin = shutil.which("perf")
        assert perf_bin is not None  # checked in check_prerequisites

        cmd = [perf_bin, "stat", "-x,"]

        # Resolve event list
        if self._events_override:
            events_str = self._events_override
            detail(f"Using user-specified perf events: {self._events_override}")
        elif self._resolved_metrics:
            all_events: set[str] = set()
            for impl in self._resolved_metrics.values():
                all_events |= impl.required_events
            events_str = ",".join(sorted(all_events))
            detail(f"Resolved perf events ({len(all_events)}): {events_str}")
        else:
            warn("No resolved perf metrics are available. Running without -e; perf will use its default events.")
            events_str = ""
        # Always include duration_time for wall-clock time measurement.
        events_str = "duration_time" + ("," + events_str if events_str else "")

        if events_str:
            cmd.extend(["-e", events_str])

        # Interval mode
        if self._interval_ms is not None:
            cmd.extend(["-I", str(self._interval_ms)])

        # Output file
        cmd.extend(["-o", str(self.output_file)])

        # Append the user's command
        cmd.extend(command)

        return cmd

    def run(self, command: list[str], cwd: Path) -> int:
        perf_cmd = self._build_command(command)
        cmd_str = " ".join(perf_cmd)
        info(f"Running profiled command: {cmd_str}")
        detail(f"Perf output file: {self.output_file}")
        debug(f"cwd: {cwd}")

        try:
            result = subprocess.run(
                perf_cmd,
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

    def parse_output(self) -> list[RankMetrics]:
        """Parse perf stat CSV output into RankMetrics.

        Returns:
            List of RankMetrics parsed from the perf output CSV.
        """
        perf_file = self.output_file
        detail(f"Parsing perf output from: {perf_file}")

        thread = perf_csv_to_thread_metrics(perf_file, interval_ms=self._interval_ms)
        if thread is None:
            raise UserError("No profiling data found in perf output file.")

        rank = RankMetrics(rank_id=0, threads=[thread])
        info(f"Parsed perf output from {perf_file}")
        return [rank]
