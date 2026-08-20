"""Shared types and utilities for roofline metric resolution and computation.

This module contains the backend-agnostic types and algorithms shared across
all profiler backends:

- :class:`BackendType` — enumeration of profiler backends.
- :class:`MetricType` — enumeration of roofline metric categories.
- :class:`MetricResolutionConfig` — user preferences that influence resolution.
- :class:`MetricContext` — application/ISA-specific context for computation.
- :class:`MetricDefinition` — a concrete metric implementation.
- :func:`compute_region_point` / :func:`sum_roofline_points` — arithmetic helpers.
- :func:`sum_optional_bytes` — optional-metric bytes aggregation helper.
- :func:`resolve_metrics` — generic priority-based metric resolution.
- :func:`check_perf_event_paranoid` — pre-flight check that the kernel permits unprivileged hardware counters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable, cast

from carm_roofline.core import ArithmeticOperation, DataType, UserError
from carm_roofline.isa import BaseISA
from carm_roofline.output_utils import debug, detail, warn

if TYPE_CHECKING:
    from .optional_metrics import OptionalMetricName, ResolvedOptionalMetric


_PERF_PARANOID_PATH = Path("/proc/sys/kernel/perf_event_paranoid")
_PERF_PARANOID_LIMIT = 2


def perf_event_paranoid() -> int | None:
    """Return the current perf_event_paranoid value, or None when unreadable or non-integer."""
    try:
        return int(_PERF_PARANOID_PATH.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None


def check_perf_event_paranoid() -> None:
    """Raise UserError when the kernel blocks unprivileged hardware-counter access.

    Values above 2 block unprivileged per-process hardware counters for both
    the perf and PAPI backends. An unreadable or non-integer sysctl warns and
    proceeds optimistically.
    """
    paranoid = perf_event_paranoid()
    if paranoid is None:
        warn("Could not read /proc/sys/kernel/perf_event_paranoid; assuming perf hardware counters are permitted.")
    elif paranoid > _PERF_PARANOID_LIMIT:
        raise UserError(
            f"perf_event_paranoid is set to {paranoid}; hardware-counter profiling requires a value of 2 or lower. "
            "Run 'sudo sysctl kernel.perf_event_paranoid=-1' (or any value <= 2) and retry."
        )


class BackendType(Enum):
    """Supported profiler backends."""

    PAPI = "papi"
    PERF = "perf"


class MetricType(Enum):
    """Categories of roofline metrics that can be resolved and computed."""

    FLOPS = "flops"
    BYTES = "bytes"


@dataclass(frozen=True)
class MetricResolutionConfig:
    """User preferences that influence metric resolution.

    Attributes:
        data_type: Dominant data type of the application. When set, the
            resolution logic boosts/downgrades priorities of metric
            implementations that match/conflict with this type.
        isas: ISAs the application exercises. Used to refine
            bytes-per-instruction and ops-per-instruction estimates,
            and to build tailored FP_ARITH-based metrics.
    """

    data_type: DataType | None = None
    isas: tuple[type[BaseISA], ...] = ()


class MetricContext:
    """Application/ISA-specific context for computing roofline metrics from hardware counters."""

    def __init__(self, config: MetricResolutionConfig):
        cfg = config
        isa_classes = cfg.isas  # tuple; empty when user didn't specify --isa
        data_type = cfg.data_type

        if isa_classes and data_type is not None:
            # Use the ISA with the most bytes per instruction for scaling
            instances = [cls() for cls in isa_classes]
            isa_instance = max(instances, key=lambda isa: isa.bytes_per_inst(data_type))
            self.bytes_per_instruction = isa_instance.bytes_per_inst(data_type)
            self.ops_per_instruction = isa_instance.ops_per_inst(data_type, ArithmeticOperation.add)
        else:
            self.bytes_per_instruction = data_type.bytes() if data_type is not None else 8
            self.ops_per_instruction = 1

        if data_type is DataType.f64:
            self.double_ratio = 1.0
        else:
            self.double_ratio = 0.0

        self._config = cfg

        debug(
            f"MetricContext initialized with ISAs: "
            f"{', '.join(c.__name__ for c in isa_classes) if isa_classes else 'None'}, "
            f"DataType: {data_type.name if data_type else 'None'}, "
            f"Bytes/inst: {self.bytes_per_instruction}, Ops/inst: {self.ops_per_instruction}"
        )

    @property
    def config(self) -> MetricResolutionConfig:
        return self._config

    @property
    def single_ratio(self) -> float:
        return 1.0 - self.double_ratio


@dataclass(frozen=True)
class MetricDefinition:
    """A concrete implementation of a roofline metric using specific hardware events.

    Attributes:
        type: The type of the metric (e.g. ``MetricType.FLOPS``, ``MetricType.BYTES``).
        required_events: Set of event names this implementation requires.
        compute: Function that extracts the metric value from collected counter
            values. The dict key is the event name, the value is the counter reading.
        priority: Higher = preferred when multiple implementations are available.
        description: Human-readable description of this implementation.
    """

    type: MetricType
    required_events: frozenset[str]
    compute: Callable[[dict[str, float], MetricContext], float]
    priority: int = 0
    priority_modifier: Callable[[MetricResolutionConfig], int] = lambda _: 0
    description: str = ""
    warning: str | None = None


@dataclass(frozen=True)
class RooflinePoint:
    """Computed roofline metrics for a single region or aggregated set."""

    flops: float = 0.0
    bytes: float = 0.0
    time_s: float = 0.0
    optional_bytes: dict[str, dict[str, float]] = field(default_factory=dict)

    def __add__(self, other: RooflinePoint) -> RooflinePoint:
        return RooflinePoint(
            flops=self.flops + other.flops,
            bytes=self.bytes + other.bytes,
            time_s=self.time_s + other.time_s,
            optional_bytes=sum_optional_bytes([self.optional_bytes, other.optional_bytes]),
        )


def sum_optional_bytes(optional: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    """Sum per-metric, per-level optional bytes across points.

    Args:
        optional: Per-point ``optional_bytes`` dicts.

    Returns:
        ``{metric_name: {level: summed_bytes}}``; ``{}`` when the list is empty
        or all points carry no optional bytes.
    """
    result: dict[str, dict[str, float]] = {}
    for per_metric in optional:
        for name, levels in per_metric.items():
            target = result.setdefault(name, {})
            for level, value in levels.items():
                target[level] = target.get(level, 0.0) + value
    return result


# ---------------------------------------------------------------------------
# Computing roofline point data from raw counters
# ---------------------------------------------------------------------------


def compute_region_point(
    counters: dict[str, int],
    time_nsec: int,
    resolved: dict[MetricType, MetricDefinition],
    metric_ctx: MetricContext,
    resolved_optional: dict[OptionalMetricName, ResolvedOptionalMetric] | None = None,
) -> RooflinePoint:
    """Compute (flops, bytes, time_s) for a single region from raw counters.

    Only metrics whose required events are all present in *counters* are
    computed; missing events cause the metric to be silently skipped.
    Optional metrics are computed the same way: each resolved optional metric
    whose required events are all present contributes its per-level bytes.

    Args:
        counters: Raw event counter values keyed by event name.
        time_nsec: Wall-clock time in nanoseconds for the region.
        resolved: Resolved metric implementations from ``resolve_metrics()``.
        metric_ctx: Metric context for flops/bytes computation.
        resolved_optional: Resolved optional metrics (from
            ``resolve_optional_metrics()``); None or empty skips them.

    Returns:
        A `RooflinePoint` with flops, bytes, time_s, and optional_bytes.
    """
    float_counters = {k: float(v) for k, v in counters.items()}
    available_set = frozenset(counters)

    flops = 0.0
    if MetricType.FLOPS in resolved:
        impl = resolved[MetricType.FLOPS]
        if impl.required_events <= available_set:
            flops = impl.compute(float_counters, metric_ctx)

    bytes_val = 0.0
    if MetricType.BYTES in resolved:
        impl = resolved[MetricType.BYTES]
        if impl.required_events <= available_set:
            bytes_val = impl.compute(float_counters, metric_ctx)

    optional_bytes: dict[str, dict[str, float]] = {}
    if resolved_optional:
        for name, ro in resolved_optional.items():
            if ro.required_events <= available_set:
                levels = ro.metric.compute(float_counters, bytes_val, ro.role_events, metric_ctx.bytes_per_instruction)
                # The JSON-facing container is per-metric heterogeneous, so it stays dict[str, float];
                optional_bytes[name.value] = cast(dict[str, float], levels)

    return RooflinePoint(flops=flops, bytes=bytes_val, time_s=time_nsec / 1e9, optional_bytes=optional_bytes)


def sum_roofline_points(points: list[RooflinePoint]) -> RooflinePoint:
    """Sum a list of (flops, bytes, time_s) RooflinePoints.

    Flops, bytes, and time are all summed, all points in the list represent
    sequential execution within a single thread.
    """
    return sum(points, start=RooflinePoint())


def resolve_metrics(
    available_events: frozenset[str],
    config: MetricResolutionConfig | None = None,
    *,
    registry: dict[MetricType, list[MetricDefinition]],
) -> dict[MetricType, MetricDefinition]:
    """Pick the best available implementation for each roofline metric.

    For each logical metric (``FLOPS``, ``BYTES``), selects the
    highest-priority ``MetricDefinition`` whose required events are all present
    in *available_events*.

    When *config* is provided, each definition's ``priority_modifier`` is
    called to adjust the base priority, allowing user preferences (e.g.
    data type) to influence which implementation is chosen.

    Args:
        available_events: Set of event names available on this system.
        config: Optional user preferences to bias resolution.
        registry: Metric definitions registry (required, no default —
            each backend passes its own registry explicitly).

    Returns:
        Dict mapping metric type -> best ``MetricDefinition`` found.
    """
    resolved: dict[MetricType, MetricDefinition] = {}
    cfg = config or MetricResolutionConfig()

    for metric_type, implementations in registry.items():
        valid = [impl for impl in implementations if impl.required_events <= available_events]
        if not valid:
            detail(f"No available implementation for metric '{metric_type}' with events: {sorted(available_events)}")
            continue

        def _effective_key(impl: MetricDefinition) -> tuple[int, int]:
            return (impl.priority + impl.priority_modifier(cfg), len(impl.required_events))

        best = max(valid, key=_effective_key)
        resolved[metric_type] = best

    return resolved
