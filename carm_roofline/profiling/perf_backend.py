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

from .backends import ProfilerBackend, RunResult, RunSpec
from .model import RankMetrics
from .perf_loader import multiplexed_events, perf_csv_to_thread_metrics
from .perf_metrics import (
    parse_perf_available_events,
    resolve_perf_metrics,
)
from .shared import (
    MetricDefinition,
    MetricResolutionConfig,
    MetricType,
    check_perf_event_paranoid,
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
        resolution_config: MetricResolutionConfig,
        *,
        interval_ms: int | None = None,
    ) -> None:
        self._resolution_config = resolution_config or MetricResolutionConfig()
        self._interval_ms = interval_ms
        self._available_events: frozenset[str] = frozenset()
        self._resolved_metrics: dict[MetricType, MetricDefinition] = {}

    @property
    def resolved_metrics(self) -> dict[MetricType, MetricDefinition]:
        return dict(self._resolved_metrics)

    @property
    def available_events(self) -> frozenset[str]:
        """Hardware events available on this system (from ``perf list -j``)."""
        return self._available_events

    def _output_file(self, run_spec: RunSpec) -> Path:
        """Path to the perf stat CSV that will be produced for this run."""
        return run_spec.output_dir / "perf_stat.csv"

    @property
    def mode_name(self) -> str:
        """Human-readable mode name for metadata."""
        return f"perf {'interval' if self._interval_ms else 'full-run'}"

    @property
    def run_method_name(self) -> str:
        """Human-readable label for the run metadata 'method' field."""
        return self.mode_name

    def can_collect(self, events: frozenset[str]) -> bool:
        """Return True if *events* can be counted together in one run without multiplexing.

        Runs a short ``perf stat`` probe over the event set and returns False
        when the kernel time-multiplexes any event or leaves any of them
        uncounted (``<not counted>``/``<not supported>`` rows). A missing perf
        binary or a failed probe returns True (proceed optimistically),
        matching the PAPI backend's fallback when ``papi_event_chooser`` is
        unavailable.
        """
        if not events:
            return True
        perf_bin = shutil.which("perf")
        if perf_bin is None:
            return True
        events_str = "duration_time," + ",".join(sorted(events))
        cmd = [perf_bin, "stat", "-x,", "-e", events_str, "--", "sleep", "0.05"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return True
        if result.returncode != 0:
            return False
        affected = multiplexed_events(result.stdout)
        debug(
            f"perf can_collect probe: {len(events)} events -> "
            f"{'fits' if not affected else 'would multiplex: ' + ', '.join(affected)}"
        )
        return not affected

    def check_prerequisites(self) -> bool:
        """Verify that ``perf`` is available and discover hardware events.

        Runs ``perf list -j`` to discover available hardware events, then
        resolves the best metric implementations for this system.

        Raises:
            UserError: If ``perf`` is not found, or the kernel blocks unprivileged hardware counters
                (``perf_event_paranoid`` above 2).
        """
        perf_bin = shutil.which("perf")
        if perf_bin is None:
            raise UserError(
                "perf not found. Install the 'perf' tools package (e.g. 'linux-tools-common' on Ubuntu, "
                "'perf' on Fedora)."
            )

        check_perf_event_paranoid()

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

        # Pre-flight: verify the resolved event set can be counted without multiplexing
        all_events: set[str] = set()
        for impl in self._resolved_metrics.values():
            all_events |= impl.required_events
        if not self.can_collect(frozenset(all_events)):
            warn(
                "The resolved perf event set may not fit in the available hardware counters without multiplexing. Perf "
                "results may be scaled or incomplete. Try specifying fewer ISAs with --isa."
            )

        return any(impl.priority < 100 for impl in self._resolved_metrics.values())

    def _build_command(self, run_spec: RunSpec, command: list[str]) -> list[str]:
        """Build the perf command line.

        Constructs ``perf stat -x, -e <events> [-I <ms>] -o <out_csv> <command>``.
        """
        perf_bin = shutil.which("perf")
        assert perf_bin is not None  # verified by the session probe's check_prerequisites

        cmd = [perf_bin, "stat", "-x,"]

        # Resolve event list
        if run_spec.events:
            events_str = run_spec.events
            detail(f"Resolved perf events ({len(run_spec.events.split(','))}): {run_spec.events}")
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
        cmd.extend(["-o", str(self._output_file(run_spec))])

        # Append the user's command
        cmd.extend(command)

        return cmd

    def profile(self, run_spec: RunSpec, command: list[str], cwd: Path) -> RunResult:
        """Run the profiled command under ``perf stat`` and parse its output.

        Args:
            run_spec: Per-run parameters (output directory, requested events).
            command: The full application command (including launcher if any).
            cwd: Working directory for the command.

        Returns:
            The command exit code together with the parsed rank metrics.
        """
        run_spec.output_dir.mkdir(parents=True, exist_ok=True)
        perf_cmd = self._build_command(run_spec, command)
        cmd_str = " ".join(perf_cmd)
        info(f"Running profiled command: {cmd_str}")
        detail(f"Perf output file: {self._output_file(run_spec)}")
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

        ranks = self._parse_output(run_spec)
        return RunResult(exit_code=result.returncode, ranks=ranks)

    def _parse_output(self, run_spec: RunSpec) -> list[RankMetrics]:
        """Parse perf stat CSV output into RankMetrics.

        Returns:
            List of RankMetrics parsed from the perf output CSV.
        """
        perf_file = self._output_file(run_spec)
        detail(f"Parsing perf output from: {perf_file}")

        # Post-hoc: verify every event was counted without time-multiplexing
        try:
            csv_text = perf_file.read_text()
        except OSError as exc:
            warn(f"Failed to read perf output file {perf_file}: {exc}")
            csv_text = ""
        affected = multiplexed_events(csv_text)
        if affected:
            warn(
                "The following perf events were not counted or were time-multiplexed during the run: "
                f"{affected}. Counts may be scaled or incomplete."
            )

        thread = perf_csv_to_thread_metrics(perf_file, interval_ms=self._interval_ms)
        if thread is None:
            raise UserError("No profiling data found in perf output file.")

        rank = RankMetrics(rank_id=0, threads=[thread])
        info(f"Parsed perf output from {perf_file}")
        return [rank]
