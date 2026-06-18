"""Shared types and utilities for roofline metric resolution and computation.

This module contains the backend-agnostic types and algorithms shared across
all profiler backends:

- :class:`MetricType` — enumeration of roofline metric categories.
- :class:`MetricResolutionConfig` — user preferences that influence resolution.
- :class:`MetricContext` — application/ISA-specific context for computation.
- :class:`MetricDefinition` — a concrete metric implementation.
- :func:`compute_region_point` / :func:`sum_roofline_points` — arithmetic helpers.
- :func:`resolve_metrics` — generic priority-based metric resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from benchmark.generation import BaseISA, DataType
from benchmark.generation.code_gen.operation import ArithmeticOperation
from output_utils import debug, detail


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
        isa: Dominant ISA of the application. Used to refine
            bytes-per-instruction and ops-per-instruction estimates.
    """

    data_type: DataType | None = None
    isa: type[BaseISA] | None = None


class MetricContext:
    """Application/ISA-specific context for computing roofline metrics from hardware counters."""

    def __init__(self, config: MetricResolutionConfig):
        cfg = config
        isa = cfg.isa
        data_type = cfg.data_type

        if isa is not None and data_type is not None:
            isa_instance = isa()
            self.bytes_per_instruction = isa_instance.bytes_per_inst(data_type)
            self.ops_per_instruction = (
                # TODO: For simplicity, assume single-op instruction for now
                isa_instance.ops_per_inst(data_type, ArithmeticOperation.add) if data_type is not None else 1
            )
        else:
            self.bytes_per_instruction = data_type.bytes() if data_type is not None else 8
            self.ops_per_instruction = 1

        if data_type is DataType.f64:
            self.double_ratio = 1.0
        else:
            self.double_ratio = 0.0

        self._config = cfg

        debug(
            f"MetricContext initialized with ISA: {isa.__name__ if isa else 'None'}, "
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


# ---------------------------------------------------------------------------
# Computing roofline point data from raw counters
# ---------------------------------------------------------------------------


def compute_region_point(
    counters: dict[str, int],
    time_nsec: int,
    resolved: dict[MetricType, MetricDefinition],
    metric_ctx: MetricContext,
) -> dict[str, float]:
    """Compute (flops, bytes, time_s) for a single region from raw counters.

    Only metrics whose required events are all present in *counters* are
    computed; missing events cause the metric to be silently skipped.

    Args:
        counters: Raw event counter values keyed by event name.
        time_nsec: Wall-clock time in nanoseconds for the region.
        resolved: Resolved metric implementations from ``resolve_metrics()``.

    Returns:
        A dict with keys ``flops``, ``bytes``, and ``time_s``.
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

    return {
        "flops": flops,
        "bytes": bytes_val,
        "time_s": time_nsec / 1e9,
    }


def sum_roofline_points(points: list[dict[str, float]]) -> dict[str, float]:
    """Sum a list of (flops, bytes, time_s) dicts.

    Flops and bytes are summed; time_s uses ``max`` (all regions within a
    thread execute sequentially, so total wall time is the max).
    """
    flops = sum(p["flops"] for p in points)
    bytes_val = sum(p["bytes"] for p in points)
    time_s = max((p["time_s"] for p in points), default=0.0)
    return {"flops": flops, "bytes": bytes_val, "time_s": time_s}


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
