"""Metric definitions, event discovery, and resolution for PAPI profiling.

This module separates three concerns:

1. **Roofline metrics** (``dp_flops``, ``sp_flops``, ``bytes``) — what we want to
   measure.
2. **Event sets** — which PAPI hardware events can provide those metrics.
3. **Resolution logic** — pick the best available implementation for each metric.

Architecture follows ``papi-metric-resolution.md``.
"""

from __future__ import annotations

import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from benchmark.generation import BaseISA, DataType
from benchmark.generation.code_gen.operation import ArithmeticOperation
from output_utils import debug, detail, warn


class MetricType(Enum):
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
    """Application/ISA-specific context for computing roofline metrics from PAPI events."""

    def __init__(self, config: MetricResolutionConfig | None = None):
        cfg = config or MetricResolutionConfig()
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
    """A concrete implementation of a roofline metric using specific PAPI events.

    Attributes:
        type: The type of the metric (e.g. ``MetricType.DP_FLOPS``, ``MetricType.SP_FLOPS``, ``MetricType.BYTES``).
        required_events: Set of PAPI event names this implementation requires.
        compute: Function that extracts the metric value from collected counter
            values. The dict key is the PAPI event name (as written in the CSV
            output column), the value is the counter reading.
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
# Metric definition factory
# ---------------------------------------------------------------------------

# FP_ARITH vector-width counter names (Intel) used by the BYTES definitions below
_FP128_DP = "FP_ARITH_INST_RETIRED:128B_PACKED_DOUBLE"
_FP256_DP = "FP_ARITH_INST_RETIRED:256B_PACKED_DOUBLE"
_FP512_DP = "FP_ARITH_INST_RETIRED:512B_PACKED_DOUBLE"
_FP_SCALAR_DP = "FP_ARITH_INST_RETIRED:SCALAR_DOUBLE"

_FP128_SP = "FP_ARITH_INST_RETIRED:128B_PACKED_SINGLE"
_FP256_SP = "FP_ARITH_INST_RETIRED:256B_PACKED_SINGLE"
_FP512_SP = "FP_ARITH_INST_RETIRED:512B_PACKED_SINGLE"
_FP_SCALAR_SP = "FP_ARITH_INST_RETIRED:SCALAR_SINGLE"


def make_intel_byte_metric_defs(
    arith_counters: dict[str, float], priority: int, prio_mod: Callable[[MetricResolutionConfig], int]
) -> MetricDefinition:
    def compute_fn(e: dict[str, float], ctx: MetricContext) -> float:
        byte_weight = 0.0
        total_arith_insts = sum(e.get(event, 0.0) for event in arith_counters)

        if total_arith_insts == 0:
            return 0.0

        # what's the proportion of each vector width in the total arithmetic instructions?
        for event, weight in arith_counters.items():
            byte_weight += (e[event] / total_arith_insts) * weight

        # scale by bytes per instruction to get total bytes, assuming load/store width matches arithmetic width
        return byte_weight * e["PAPI_LST_INS"]

    return MetricDefinition(
        type=MetricType.BYTES,
        required_events=frozenset(set(arith_counters.keys()) | {"PAPI_LST_INS"}),
        compute=compute_fn,
        priority=priority,
        priority_modifier=prio_mod,
        description="DP bytes from FP_ARITH vector-width counters (assumes arithmetic/store width match)",
    )


def _build_metric_definitions() -> dict[MetricType, list[MetricDefinition]]:
    """Build the full registry of metric implementations.

    This factory is called once at module load time.  Each logical metric
    (``dp_flops``, ``sp_flops``, ``bytes``) has multiple implementations
    at different base priorities.  Implementations can carry a
    ``priority_modifier`` that adjusts the effective priority at resolution
    time based on user preferences (``MetricResolutionConfig``).

    Returns:
        ``{MetricType: [MetricDefinition, ...]}``
    """

    def _data_type_match(dt: DataType) -> Callable[[MetricResolutionConfig], int]:
        """Return +15 when config.data_type matches *dt*, -15 otherwise."""
        return lambda cfg: 15 if cfg.data_type is dt else -15

    def _make_fmadd_warning(papi_event: str) -> str:
        return f"{papi_event} may count multiply-add instructions as 1 or 2 ops depending on the platform"

    return {
        MetricType.FLOPS: [
            MetricDefinition(
                type=MetricType.FLOPS,
                required_events=frozenset({"PAPI_FP_OPS"}),
                compute=lambda e, ctx: e["PAPI_FP_OPS"],
                priority=100,
                description="Exact value from PAPI_FP_OPS (both precisions)",
                warning=_make_fmadd_warning("PAPI_FP_OPS"),
            ),
            MetricDefinition(
                type=MetricType.FLOPS,
                required_events=frozenset({"PAPI_DP_OPS"}),
                compute=lambda e, ctx: e["PAPI_DP_OPS"],
                priority=90,
                priority_modifier=_data_type_match(DataType.f64),
                description="Exact value from PAPI_DP_OPS",
                warning=_make_fmadd_warning("PAPI_DP_OPS"),
            ),
            MetricDefinition(
                type=MetricType.FLOPS,
                required_events=frozenset({"PAPI_SP_OPS"}),
                compute=lambda e, ctx: e["PAPI_SP_OPS"],
                priority=90,
                priority_modifier=_data_type_match(DataType.f32),
                description="Exact value from PAPI_SP_OPS",
                warning=_make_fmadd_warning("PAPI_SP_OPS"),
            ),
            MetricDefinition(
                type=MetricType.FLOPS,
                required_events=frozenset({"PAPI_FP_INS"}),
                compute=lambda e, ctx: e["PAPI_FP_INS"] * ctx.ops_per_instruction,
                priority=10,
                description="Approximated from PAPI_FP_INS (FP instr count)",
                warning="PAPI_FP_INS counts fused multiply-add instructions as 1 operation",
            ),
        ],
        MetricType.BYTES: [
            # make arithmetic-weight-based defs for intel architectures
            # for architectures with AVX-512:
            make_intel_byte_metric_defs(
                {_FP_SCALAR_DP: 8, _FP128_DP: 16, _FP256_DP: 32, _FP512_DP: 64}, 100, _data_type_match(DataType.f64)
            ),
            make_intel_byte_metric_defs(
                {_FP_SCALAR_SP: 8, _FP128_SP: 16, _FP256_SP: 32, _FP512_SP: 64}, 100, _data_type_match(DataType.f32)
            ),
            # for architectures with AVX-2 only:
            make_intel_byte_metric_defs(
                {_FP_SCALAR_DP: 8, _FP128_DP: 16, _FP256_DP: 32}, 99, _data_type_match(DataType.f64)
            ),
            make_intel_byte_metric_defs(
                {_FP_SCALAR_SP: 8, _FP128_SP: 16, _FP256_SP: 32}, 99, _data_type_match(DataType.f32)
            ),
            # for architectures with SSE only (do those exist?):
            make_intel_byte_metric_defs({_FP_SCALAR_DP: 8, _FP128_DP: 16}, 98, _data_type_match(DataType.f64)),
            make_intel_byte_metric_defs({_FP_SCALAR_SP: 8, _FP128_SP: 16}, 98, _data_type_match(DataType.f32)),
            # basic definitions based on load/store counts and user-provided bytes per instruction
            MetricDefinition(
                type=MetricType.BYTES,
                required_events=frozenset({"PAPI_LST_INS"}),
                compute=lambda e, ctx: e["PAPI_LST_INS"] * ctx.bytes_per_instruction,
                priority=90,
                description="Approximated from PAPI_LST_INS and bytes per instruction",
            ),
            MetricDefinition(
                type=MetricType.BYTES,
                required_events=frozenset({"PAPI_LD_INS", "PAPI_SR_INS"}),
                compute=lambda e, ctx: (e["PAPI_LD_INS"] + e["PAPI_SR_INS"]) * ctx.bytes_per_instruction,
                priority=85,
                description="Approximated from PAPI_LD_INS + PAPI_SR_INS and bytes per instruction",
            ),
            # Relevant for Zen3 (no PAPI loads/store events, only DCA, native forwarding event for higher accuracy)
            MetricDefinition(
                type=MetricType.BYTES,
                required_events=frozenset({"PAPI_L1_DCA", "STORE_TO_LOAD_FORWARD"}),
                compute=lambda e, ctx: (e["PAPI_L1_DCA"] + e["STORE_TO_LOAD_FORWARD"]) * ctx.bytes_per_instruction,
                priority=40,
                description="Approximated from PAPI_L1_DCA + STORE_TO_LOAD_FORWARD and bytes per instruction",
                warning=(
                    "PAPI_L1_DCA counts cache accesses: if the microarchitecture coalesces requests into fewer cache "
                    "accesses, this metric may underestimate the number of bytes."
                ),
            ),
            MetricDefinition(
                type=MetricType.BYTES,
                required_events=frozenset({"PAPI_L1_DCA"}),
                compute=lambda e, ctx: e["PAPI_L1_DCA"] * ctx.bytes_per_instruction,
                priority=30,
                description="Approximated from PAPI_L1_DCA and bytes per instruction",
                warning=(
                    "PAPI_L1_DCA counts cache accesses: if the microarchitecture coalesces requests into fewer cache "
                    "accesses or forwards stores to loads, this metric may underestimate the number of bytes."
                ),
            ),
        ],
    }


# Build once at module load time
_METRICS = _build_metric_definitions()


# ---------------------------------------------------------------------------
# Available events discovery
# ---------------------------------------------------------------------------


def parse_available_events() -> frozenset[str]:
    """Run ``papi_xml_event_info`` and parse event names (base + modifiers) from its XML output.

    The command outputs an XML document containing both base event names
    (e.g. ``FP_ARITH_INST_RETIRED``) and their modifier names
    (e.g. ``FP_ARITH_INST_RETIRED:128B_PACKED_DOUBLE``) across all component
    event sets (NATIVE and PRESET).

    Returns:
        frozenset of available PAPI event name strings. Empty set if
        the ``papi_xml_event_info`` tool is not found or fails.
    """
    papi_xml = shutil.which("papi_xml_event_info")
    if papi_xml is None:
        warn("papi_xml_event_info not found - cannot determine available PAPI events")
        return frozenset()

    try:
        result = subprocess.run(
            [papi_xml],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        warn(f"Failed to run papi_xml_event_info: {exc}")
        return frozenset()

    if result.returncode != 0:
        warn(f"papi_xml_event_info exited with code {result.returncode}: {result.stderr.strip()}")
        return frozenset()

    return _parse_papi_xml_output(result.stdout)


def _parse_papi_xml_output(output: str) -> frozenset[str]:
    """Parse the XML output from ``papi_xml_event_info``.

    Collects both base event names and their modifier names from all
    ``<event>`` elements in the XML.

    Args:
        output: Raw XML stdout from ``papi_xml_event_info``.

    Returns:
        frozenset of available event names.
    """
    try:
        root = ET.fromstring(output)
    except ET.ParseError as exc:
        warn(f"Failed to parse papi_xml_event_info XML output: {exc}")
        return frozenset()

    events: set[str] = set()
    for event_elem in root.iter("event"):
        name = event_elem.get("name", "").strip()
        if not name:
            continue
        events.add(name)
        for modifier in event_elem.iter("modifier"):
            mod_name = modifier.get("name", "").strip()
            if mod_name:
                events.add(mod_name)

    return frozenset(events)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


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
        counters: Raw PAPI event counter values keyed by event name.
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
) -> dict[MetricType, MetricDefinition]:
    """Pick the best available implementation for each roofline metric.

    For each logical metric (``dp_flops``, ``sp_flops``, ``bytes``),
    selects the highest-priority ``MetricDefinition`` whose required events are
    all present in *available_events*.

    When *config* is provided, each definition's ``priority_modifier`` is
    called to adjust the base priority, allowing user preferences (e.g.
    data type) to influence which implementation is chosen.  Without a
    config the behaviour is identical to using base priorities only.

    Metrics with NO available implementation are silently omitted from the
    result dict.

    Args:
        available_events: Set of PAPI event names available on this system.
        config: Optional user preferences to bias resolution.

    Returns:
        Dict mapping metric type -> best ``MetricDefinition`` found.
    """
    resolved: dict[MetricType, MetricDefinition] = {}
    cfg = config or MetricResolutionConfig()

    for metric_type, implementations in _METRICS.items():
        valid = [impl for impl in implementations if impl.required_events <= available_events]
        if not valid:
            detail(f"No available implementation for metric '{metric_type}' with events: {sorted(available_events)}")
            continue

        def _effective_key(impl: MetricDefinition) -> tuple[int, int]:
            return (impl.priority + impl.priority_modifier(cfg), len(impl.required_events))

        best = max(valid, key=_effective_key)
        resolved[metric_type] = best

    return resolved
