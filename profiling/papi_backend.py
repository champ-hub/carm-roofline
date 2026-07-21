"""PAPI High-Level (HL) API profiler backend.

Runs the user's command with ``PAPI_HL_OUTPUT_DIR`` set so that PAPI HL writes
per-rank output files to a known location, then parses them into the
:mod:`~profiling.model` hierarchy.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from core import UserError
from output_utils import debug, detail, error, info, warn

from .backends import ProfilerBackend
from .model import RankMetrics
from .papi_loader import load_all_ranks
from .papi_metrics import (
    PAPIMetricRegistry,
    parse_available_events,
    validate_event_set,
)
from .shared import (
    MetricDefinition,
    MetricResolutionConfig,
    MetricType,
)


def _find_papi_library_path() -> Path | None:
    """Locate ``libpapi.so`` via multiple discovery strategies.

    Tries in order:
      1. ``ldconfig -p`` (dynamic linker cache, standard installs)
      2. ``pkg-config`` (respects ``PKG_CONFIG_PATH``, custom prefixes)

    Returns:
        Path to ``libpapi.so``, or *None* if not found by any strategy.
    """
    # Strategy 1: ldconfig -p
    try:
        result = subprocess.run(
            ["ldconfig", "-p"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "libpapi.so" in line:
                    # ldconfig -p output format: "	libpapi.so (libc6,x86-64) => /usr/lib/libpapi.so"
                    parts = line.split("=>")
                    if len(parts) == 2:
                        path = Path(parts[1].strip())
                        if path.is_file():
                            return path
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # Strategy 2: pkg-config -> list dir, pick first libpapi.so*
    try:
        result = subprocess.run(
            ["pkg-config", "--variable=libdir", "papi"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            libdir = result.stdout.strip()
            if libdir:
                candidates = sorted(Path(libdir).glob("libpapi.so*"))
                if candidates:
                    return candidates[0]
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    return None


class PAPIHLBackend(ProfilerBackend):
    """PAPI High-Level API profiler backend.

    Runs the user's command with ``PAPI_HL_OUTPUT_DIR`` set so that PAPI HL writes per-rank output files to a known
    location.

    Uses ``papi_decode -a`` at startup to discover available PAPI events, then resolves the best metric implementations
    for the current system.
    """

    def __init__(
        self,
        output_dir: Path,
        resolution_config: MetricResolutionConfig,
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

    @property
    def run_method_name(self) -> str:
        """Human-readable label for the run metadata 'method' field."""
        return "PAPI HL"

    def check_prerequisites(self) -> bool:
        """Verify that PAPI HL is available by checking the environment.

        We check for the presence of ``libpapi`` (via :func:`_find_papi_library_path`) or the ``papi_hl_output_writer``
        utility as a proxy for PAPI HL availability. After that, runs ``papi_xml_event_info`` to discover available
        events and resolve metric implementations.
        """

        # Locate libpapi.so and verify it provides PAPI HL symbols
        papi_lib = _find_papi_library_path()
        if papi_lib is not None:
            try:
                result = subprocess.run(
                    ["nm", "-D", str(papi_lib)],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if "PAPI_hl_region_begin" not in result.stdout:
                    warn(
                        f"Found {papi_lib} but it lacks the PAPI_hl_region_begin symbol. "
                        "PAPI HL profiling will not work."
                    )
                    papi_lib = None
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
                warn(f"Could not check symbols in {papi_lib}: {exc}")
                papi_lib = None

        # Fallback: check for papi_hl_output_writer utility
        if papi_lib is None:
            hl_writer = shutil.which("papi_hl_output_writer")
            if hl_writer is not None:
                detail(f"Found PAPI HL via {hl_writer}")
            else:
                raise UserError(
                    "PAPI HL does not appear to be installed. Could not find libpapi.so (via ldconfig or pkg-config) "
                    "nor the papi_hl_output_writer utility."
                )

        # Build registry with ISA tailoring baked in at construction time
        self._available_events = parse_available_events()
        self._registry = PAPIMetricRegistry(self._resolution_config)
        self._resolved_metrics = self._registry.resolve(self._available_events)

        detail(f"Available PAPI events: {len(self._available_events)}")
        for metric_type, impl in self._resolved_metrics.items():
            detail(f"  {metric_type.name} -> {impl.description}")
            if impl.warning is not None:
                detail(f"    Note: {impl.warning}")

        # Log ISA-tailored resolution if active
        if self._resolution_config.isas:
            for mtype in (MetricType.FLOPS, MetricType.BYTES):
                if mtype in self._resolved_metrics:
                    impl = self._resolved_metrics[mtype]
                    detail(f"  {mtype.name} -> priority {impl.priority}")

        # Pre-flight: validate the resolved event set will fit
        if self._resolved_metrics:
            all_events: set[str] = set()
            for impl in self._resolved_metrics.values():
                all_events |= impl.required_events
            if not validate_event_set(frozenset(all_events)):
                warn(
                    "The resolved PAPI event set may not fit in the available hardware counters. PAPI may collect only "
                    "a subset. Try specifying fewer ISAs with --isa, or enable PAPI_MULTIPLEX=1."
                )

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

    def run(self, command: list[str], cwd: Path) -> int:
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

    def parse_output(self) -> list[RankMetrics]:
        """Parse PAPI HL profiling output files.

        Returns:
            List of RankMetrics parsed from the PAPI HL output directory.
        """
        papi_output_dir = self._output_dir / "papi_hl_output"
        detail(f"Scanning for PAPI HL profiling results in: {papi_output_dir}")
        ranks = load_all_ranks(papi_output_dir)

        if not ranks:
            raise UserError("No profiling result files found. Did the application run with PAPI HL instrumentation?")

        info(f"Loaded {len(ranks)} rank(s) from {papi_output_dir}")

        # Post-hoc: check all requested events were collected
        # Skip when the user supplied --events-override; resolved events are not what PAPI was told to collect.
        if ranks and self._resolved_metrics and not self._events_override:
            # Get requested event names from resolved metrics
            requested: set[str] = set()
            for impl in self._resolved_metrics.values():
                requested |= impl.required_events

            # Events collected by any region (union). Union—not intersection—because
            # multiplexed runs give regions different counter subsets, and metadata-only
            # regions have an empty counter set that would otherwise collapse the set.
            collected: set[str] = set()
            for rank in ranks:
                for thread in rank.threads:
                    for region in thread.regions:
                        collected |= set(region.counters.keys())

            missing = requested - collected
            if missing:
                warn(
                    "The following requested PAPI events were NOT collected by PAPI HL: "
                    f"{', '.join(sorted(missing))}. "
                    "This usually means the event set exceeds available hardware counters. Bytes and/or FLOPs may be "
                    "undercounted or zero. Try specifying fewer ISAs via --isa to reduce hardware counter pressure."
                )
                detail(
                    f"Requested: {len(requested)} events, "
                    f"Collected: {len(collected)} events, "
                    f"Missing: {', '.join(sorted(missing))}"
                )

        return ranks
