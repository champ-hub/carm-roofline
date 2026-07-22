"""Metric definitions, event discovery, and resolution for PAPI profiling.

This module separates three concerns:

1. **Roofline metrics** (``dp_flops``, ``sp_flops``, ``bytes``) — what we want to
   measure.
2. **Event sets** — which PAPI hardware events can provide those metrics.
3. **Resolution logic** — pick the best available implementation for each metric.

Architecture follows ``papi-metric-resolution.md``.

Shared types (:class:`MetricType`, :class:`MetricDefinition`,
:class:`MetricResolutionConfig`, :class:`MetricContext`) and generic resolution
(:func:`resolve_metrics`) are defined in :mod:`.shared`.
"""

from __future__ import annotations

import shutil
import subprocess
import xml.etree.ElementTree as ET
from typing import Callable

from carm_roofline.core import DataType
from carm_roofline.isa import BaseISA
from carm_roofline.isa.x86 import X86AVX, X86AVX2, X86AVX512, X86SSE, X86Scalar
from carm_roofline.output_utils import warn

from .shared import (
    MetricContext,
    MetricDefinition,
    MetricResolutionConfig,
    MetricType,
    resolve_metrics as _resolve_metrics,
)

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

# Direct ISA class -> FP_ARITH counter name mapping.
# Each x86 vector ISA maps to exactly its width's packed counter;
# X86Scalar maps to SCALAR. Non-x86 ISAs are absent — they have no FP_ARITH counters.
_ISA_TO_COUNTER: dict[type, tuple[str, str]] = {
    # class -> (dp_counter, sp_counter)
    X86Scalar: ("FP_ARITH_INST_RETIRED:SCALAR_DOUBLE", "FP_ARITH_INST_RETIRED:SCALAR_SINGLE"),
    X86SSE: ("FP_ARITH_INST_RETIRED:128B_PACKED_DOUBLE", "FP_ARITH_INST_RETIRED:128B_PACKED_SINGLE"),
    X86AVX: ("FP_ARITH_INST_RETIRED:256B_PACKED_DOUBLE", "FP_ARITH_INST_RETIRED:256B_PACKED_SINGLE"),
    X86AVX2: ("FP_ARITH_INST_RETIRED:256B_PACKED_DOUBLE", "FP_ARITH_INST_RETIRED:256B_PACKED_SINGLE"),
    X86AVX512: ("FP_ARITH_INST_RETIRED:512B_PACKED_DOUBLE", "FP_ARITH_INST_RETIRED:512B_PACKED_SINGLE"),
}


def fp_arith_counters_for_isas(
    isa_classes: tuple[type[BaseISA], ...],
    data_type: DataType,
) -> set[str]:
    """Union of FP_ARITH counters directly needed by the given ISA classes.

    Each ISA class maps directly to its counter via _ISA_TO_COUNTER.
    Non-x86 ISAs not in the dict produce no counters (empty set).
    """
    suffix = "DP" if data_type is DataType.f64 else "SP"
    idx = 0 if suffix == "DP" else 1
    result: set[str] = set()
    for cls in isa_classes:
        entry = _ISA_TO_COUNTER.get(cls)
        if entry is not None:
            result.add(entry[idx])
    return result


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
        description="Bytes from FP_ARITH vector-width counters (assumes arithmetic/store width match)",
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
METRICS = _METRICS  # Public alias for tests and external access

# ---------------------------------------------------------------------------
# ISA-tailored custom metric factories
# ---------------------------------------------------------------------------


def _make_fp_arith_flops_metric(
    counters: set[str],
    data_type: DataType,
    isas: tuple[type[BaseISA], ...],
) -> MetricDefinition:
    """FLOPS directly from FP_ARITH counters: element_count x counter_value.

    Element counts: SCALAR=1, 128B=2, 256B=4, 512B=8.
    No PAPI_DP/SP_OPS dependency, avoids pulling in unrequested native counters.
    """
    _OP_COEFF: dict[str, int] = {"SCALAR": 1, "128B": 2, "256B": 4, "512B": 8}

    def compute_fn(e: dict[str, float], ctx: MetricContext) -> float:
        total = 0.0
        for c, coeff in _OP_COEFF.items():
            for ec in e:
                if c in ec:
                    total += e[ec] * coeff
        return total

    return MetricDefinition(
        type=MetricType.FLOPS,
        required_events=frozenset(counters),
        compute=compute_fn,
        priority=200,
        description=(
            f"Flops from FP_ARITH vector-width counters for "
            f"{', '.join(isa.__name__ for isa in isas)} ({data_type.name})"
        ),
    )


def _fp_arith_byte_compute(
    arith_counters: dict[str, float],
) -> Callable[[dict[str, float], MetricContext], float]:
    """Returns a compute closure: vector-width-weighted bytes x PAPI_LST_INS."""

    def compute_fn(e: dict[str, float], ctx: MetricContext) -> float:
        byte_weight = 0.0
        total_arith = sum(e.get(ev, 0.0) for ev in arith_counters)
        if total_arith == 0:
            return 0.0
        for ev, weight in arith_counters.items():
            byte_weight += (e[ev] / total_arith) * weight
        if "PAPI_LST_INS" not in e:
            return 0.0
        return byte_weight * e["PAPI_LST_INS"]

    return compute_fn


def _make_fp_arith_bytes_metric(
    counters: set[str],
    data_type: DataType,
    isas: tuple[type[BaseISA], ...],
) -> MetricDefinition:
    """BYTES from narrowed FP_ARITH counters + PAPI_LST_INS."""
    el_bytes = data_type.bytes()
    weight_map: dict[str, float] = {}
    _WIDTH_FACTOR = {"SCALAR": 1, "128B": 2, "256B": 4, "512B": 8}
    for c in counters:
        for prefix, factor in _WIDTH_FACTOR.items():
            if prefix in c:
                weight_map[c] = float(el_bytes * factor)
                break

    return MetricDefinition(
        type=MetricType.BYTES,
        required_events=frozenset(counters | {"PAPI_LST_INS"}),
        compute=_fp_arith_byte_compute(weight_map),
        priority=200,
        description=(
            f"Bytes from FP_ARITH vector-width counters for "
            f"{', '.join(isa.__name__ for isa in isas)} ({data_type.name})"
            f" (assumes arithmetic/store width match)"
        ),
    )


def _build_isa_custom_metrics(
    config: MetricResolutionConfig,
) -> dict[MetricType, list[MetricDefinition]] | None:
    """Return a registry fragment with custom FLOPS+BYTES for user-specified ISAs.

    Returns None when ISAs are empty, data_type is unset, or no FP_ARITH counters
    apply (ARM, RISC-V, etc.) -- caller can safely skip.
    """
    if not config.isas or config.data_type is None:
        return None
    counters = fp_arith_counters_for_isas(config.isas, config.data_type)
    if not counters:
        return None
    return {
        MetricType.FLOPS: [_make_fp_arith_flops_metric(counters, config.data_type, config.isas)],
        MetricType.BYTES: [_make_fp_arith_bytes_metric(counters, config.data_type, config.isas)],
    }


def build_isa_custom_metrics(
    isas: tuple[type[BaseISA], ...],
    data_type: DataType,
) -> dict[MetricType, list[MetricDefinition]] | None:
    """Public wrapper: build ISA-tailored metric definitions.

    Standalone entry point for tests and ad-hoc use (no registry needed).
    Delegates to _build_isa_custom_metrics after constructing a config.
    """
    return _build_isa_custom_metrics(MetricResolutionConfig(data_type=data_type, isas=isas))


# ---------------------------------------------------------------------------
# PAPIMetricRegistry -- stateful metric definition registry
# ---------------------------------------------------------------------------


class PAPIMetricRegistry:
    """Metric definition registry with configurable ISA-tailored metrics.

    Builds the full set of metric definitions at construction time, optionally
    including custom high-priority definitions for user-specified ISAs.
    Resolution is a single method call -- no manual registry merging needed.
    """

    def __init__(self, config: MetricResolutionConfig | None = None) -> None:
        self._config = config or MetricResolutionConfig()
        self.definitions: dict[MetricType, list[MetricDefinition]] = self._build()

    def _build(self) -> dict[MetricType, list[MetricDefinition]]:
        """Start from prebuilt standard definitions, add ISA-tailored if configured."""
        result = {mtype: list(defs) for mtype, defs in _METRICS.items()}
        custom = _build_isa_custom_metrics(self._config)
        if custom is not None:
            for mtype, defs in custom.items():
                result.setdefault(mtype, []).extend(defs)
        return result

    def resolve(
        self,
        available_events: frozenset[str],
    ) -> dict[MetricType, MetricDefinition]:
        """Resolve best available metrics for the given event set."""
        return _resolve_metrics(
            available_events,
            self._config,
            registry=self.definitions,
        )


# Default registry instance for backward-compat resolve_metrics wrapper
_DEFAULT_REGISTRY = PAPIMetricRegistry()


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
# PAPI-specific metric resolution wrapper (defaults to PAPI registry)
# ---------------------------------------------------------------------------


def resolve_metrics(
    available_events: frozenset[str],
    config: MetricResolutionConfig | None = None,
) -> dict[MetricType, MetricDefinition]:
    """Pick the best available PAPI metric implementation for each roofline metric.

    Wraps :func:`shared.resolve_metrics` with the PAPI metric definitions registry as the default. When config has ISAs,
    builds tailored definitions.

    Args:
        available_events: Set of PAPI event names available on this system.
        config: Optional user preferences to bias resolution.

    Returns:
        Dict mapping metric type -> best ``MetricDefinition`` found.
    """
    if config is None:
        return _DEFAULT_REGISTRY.resolve(available_events)
    return PAPIMetricRegistry(config).resolve(available_events)


# ---------------------------------------------------------------------------
# Pre-flight event set validation
# ---------------------------------------------------------------------------


def validate_event_set(events: frozenset[str]) -> bool:
    """Run ``papi_event_chooser`` to validate *events* can coexist.

    Returns True if ``papi_event_chooser`` reports the set is valid (events can be added simultaneously). Returns False
    if the tool is unavailable, errors, or reports incompatibility.

    ``papi_event_chooser`` accepts both PRESET and NATIVE events in the same invocation — PAPI resolves them internally
    to native codes. The incompatibility message is emitted on stderr with exit code 1; a valid set has exit code 0 with
    empty stderr.
    """
    if not events:
        return True
    papi_chooser = shutil.which("papi_event_chooser")
    if papi_chooser is None:
        warn("papi_event_chooser not found, cannot validate event set compatibility before profiling")
        return True  # Can't validate, proceed optimistically

    try:
        # papi_event_chooser accepts mixed preset+native sets, first arg PRESET just controls display formatting
        result = subprocess.run(
            [papi_chooser, "PRESET", *sorted(events)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return True

    # Non-zero exit code means incompatibility on this PAPI version
    if result.returncode != 0:
        return False

    # Defensive: also check stderr for the error message (some PAPI builds
    # may exit 0 but still report the conflict on stderr)
    return "can't be counted" not in result.stderr
