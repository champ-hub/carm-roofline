"""Software metric definitions for DynamoRIO instruction-count profiling."""

from __future__ import annotations

from .shared import MetricDefinition, MetricType

_WIDTH_LANES = {"scalar": 1, "128": 128, "256": 256, "512": 512}
_PRECISION_BITS = {"f32": 32, "f64": 64}
_OPERATIONS = ("add", "sub", "mul", "div", "fma")


def _flop_weight(event: str) -> int:
    _, operation, width, precision = event.split("_")
    lanes = _WIDTH_LANES[width] if width == "scalar" else _WIDTH_LANES[width] // _PRECISION_BITS[precision]
    return lanes * (2 if operation == "fma" else 1)


FLOP_EVENTS = frozenset(
    f"dr_{operation}_{width}_{precision}"
    for operation in _OPERATIONS
    for width in _WIDTH_LANES
    for precision in _PRECISION_BITS
)
BYTE_EVENTS = frozenset({"dr_read_bytes", "dr_write_bytes"})
AVAILABLE_EVENTS = FLOP_EVENTS | BYTE_EVENTS
FLOP_WEIGHTS = {event: _flop_weight(event) for event in FLOP_EVENTS}

METRICS = {
    MetricType.FLOPS: [
        MetricDefinition(
            type=MetricType.FLOPS,
            required_events=FLOP_EVENTS,
            compute=lambda counters, _ctx: sum(counters[event] * FLOP_WEIGHTS[event] for event in FLOP_EVENTS),
            priority=100,
            description="Software-derived FLOPs from DynamoRIO decoded instruction counts",
            warning="FLOPs are instruction-derived estimates; masked vectors use full encoded width.",
        )
    ],
    MetricType.BYTES: [
        MetricDefinition(
            type=MetricType.BYTES,
            required_events=BYTE_EVENTS,
            compute=lambda counters, _ctx: counters["dr_read_bytes"] + counters["dr_write_bytes"],
            priority=100,
            description="Executed data-memory reference bytes from DynamoRIO instrumentation",
        )
    ],
}


def resolve_dynamorio_metrics() -> dict[MetricType, MetricDefinition]:
    """Resolve DynamoRIO's fixed software counters without hardware discovery."""
    return {metric: definitions[0] for metric, definitions in METRICS.items()}
