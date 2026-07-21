"""Perf-event-specific metric definitions, discovery, and resolution.

Extends the metric framework with perf hardware event names (e.g.
``fp_ret_sse_avx_ops.all``, ``ls_dispatch.ld_dispatch``) using the same
:class:`MetricDefinition` / :class:`MetricType` / :class:`MetricResolutionConfig`
abstractions defined in :mod:`.metrics`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Callable

from core import DataType
from output_utils import detail, warn

from .shared import (
    MetricDefinition,
    MetricResolutionConfig,
    MetricType,
    resolve_metrics as _resolve_with_registry,
)

# ---------------------------------------------------------------------------
# Perf metric definitions registry
# ---------------------------------------------------------------------------


def _data_type_match(dt: DataType) -> Callable[[MetricResolutionConfig], int]:
    """Return +15 when config.data_type matches *dt*, -15 otherwise."""
    return lambda cfg: 15 if cfg.data_type is dt else -15


def _build_perf_metric_definitions() -> dict[MetricType, list[MetricDefinition]]:
    """Build the full registry of perf metric implementations.

    Perf event names vary by microarchitecture; this function provides
    multiple implementations at different priority levels so the resolution
    logic picks whatever is available on the current system.

    Returns:
        ``{MetricType: [MetricDefinition, ...]}``
    """
    return {
        MetricType.FLOPS: [
            # AMD Zen: direct FLOP counter (uop-based, counts each FLOP individually)
            MetricDefinition(
                type=MetricType.FLOPS,
                required_events=frozenset({"fp_ret_sse_avx_ops.all"}),
                compute=lambda e, ctx: e["fp_ret_sse_avx_ops.all"],
                priority=100,
                description="Exact FLOPs from fp_ret_sse_avx_ops.all (AMD Zen)",
            ),
        ],
        MetricType.BYTES: [
            # AMD Zen: ls_dispatch counters
            MetricDefinition(
                type=MetricType.BYTES,
                required_events=frozenset(
                    {"ls_dispatch.ld_dispatch", "ls_dispatch.store_dispatch", "ls_dispatch.ld_st_dispatch"}
                ),
                compute=lambda e, ctx: (
                    (
                        e["ls_dispatch.ld_dispatch"]
                        + e["ls_dispatch.store_dispatch"]
                        + e["ls_dispatch.ld_st_dispatch"] * 2
                    )
                    * ctx.bytes_per_instruction
                ),
                priority=100,
                description="Bytes from ls_dispatch.ld/store/ld_st (AMD Zen) x bytes_per_inst",
            ),
        ],
    }


# Build once at module load time
_PERF_METRICS = _build_perf_metric_definitions()


def parse_perf_available_events() -> frozenset[str]:
    """Run ``perf list -j`` and parse the JSON output for hardware event names.

    Returns:
        frozenset of perf event name strings (e.g. ``fp_ret_sse_avx_ops.all``).
        Empty set if ``perf`` is not available or the command fails.
    """
    perf_bin = shutil.which("perf")
    if perf_bin is None:
        warn("perf not found - cannot determine available perf events")
        return frozenset()

    try:
        result = subprocess.run(
            [perf_bin, "list", "-j"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        warn(f"Failed to run 'perf list -j': {exc}")
        return frozenset()

    if result.returncode != 0:
        warn(f"'perf list -j' exited with code {result.returncode}: {result.stderr.strip()}")
        return frozenset()

    return _parse_perf_json_output(result.stdout)


def _parse_perf_json_output(output: str) -> frozenset[str]:
    """Parse the JSON output from ``perf list -j``.

    The JSON is a list of event groups, each with a ``"Symbol"`` key
    containing the event name.

    Args:
        output: Raw JSON stdout from ``perf list -j``.

    Returns:
        frozenset of event name strings.
    """
    try:
        data = json.loads(output)
    except json.JSONDecodeError as exc:
        warn(f"Failed to parse 'perf list -j' JSON output: {exc}")
        return frozenset()

    events: set[str] = set()
    if isinstance(data, list):
        for entry in data:
            symbol = entry.get("Symbol", "")
            if symbol:
                events.add(symbol)
            # Also collect event names for kernel PMU events that use "EventName" key
            event_name = entry.get("EventName", "")
            if event_name:
                events.add(event_name)
    elif isinstance(data, dict):
        # Some perf versions output a dict with event type keys
        for group in data.values():
            if isinstance(group, list):
                for entry in group:
                    symbol = entry.get("Symbol", "")
                    if symbol:
                        events.add(symbol)
                    event_name = entry.get("EventName", "")
                    if event_name:
                        events.add(event_name)

    detail(f"Parsed {len(events)} perf events from 'perf list -j'")
    return frozenset(events)


def resolve_perf_metrics(
    available_events: frozenset[str],
    config: MetricResolutionConfig | None = None,
) -> dict[MetricType, MetricDefinition]:
    """Pick the best available perf metric implementation for each roofline metric.

    Uses the same priority-based selection logic as :func:`resolve_metrics`,
    but operates on the perf-specific metric definitions registry.

    Args:
        available_events: Set of perf event names available on this system.
        config: Optional user preferences to bias resolution.

    Returns:
        Dict mapping metric type -> best ``MetricDefinition`` found.
    """
    return _resolve_with_registry(available_events, config, registry=_PERF_METRICS)
