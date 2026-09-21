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

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from carm_roofline.architecture.identity import read_cpuinfo
from carm_roofline.core import DataType
from carm_roofline.isa import BaseISA
from carm_roofline.isa.x86 import X86AVX, X86AVX2, X86AVX512, X86SSE, X86Scalar
from carm_roofline.output_utils import debug, detail, warn
from carm_roofline.results_paths import user_cache_dir_for_carm

from .papi_lib import _load_papi_library, collectable_events
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


@dataclass(frozen=True)
class _MemoryInstructionSource:
    """A collectable event set that counts retired memory instructions."""

    required_events: frozenset[str]
    count: Callable[[dict[str, float]], float]
    description: str
    priority_offset: int


_MEMORY_INSTRUCTION_SOURCES = (
    _MemoryInstructionSource(
        required_events=frozenset({"PAPI_LST_INS"}),
        count=lambda e: e["PAPI_LST_INS"],
        description="PAPI_LST_INS",
        priority_offset=0,
    ),
    _MemoryInstructionSource(
        required_events=frozenset({"PAPI_LD_INS", "PAPI_SR_INS"}),
        count=lambda e: e["PAPI_LD_INS"] + e["PAPI_SR_INS"],
        description="PAPI_LD_INS + PAPI_SR_INS",
        priority_offset=-5,
    ),
    _MemoryInstructionSource(
        required_events=frozenset({"MEM_INST_RETIRED:ALL_LOADS", "MEM_INST_RETIRED:ALL_STORES"}),
        count=lambda e: e["MEM_INST_RETIRED:ALL_LOADS"] + e["MEM_INST_RETIRED:ALL_STORES"],
        description="MEM_INST_RETIRED:ALL_LOADS + MEM_INST_RETIRED:ALL_STORES",
        priority_offset=-5,
    ),
)

_FP_ARITH_WIDTH_BYTES = {"128B": 16, "256B": 32, "512B": 64}
_FP_ARITH_WIDTH_WARNING = (
    "This estimate assumes the average memory-instruction width matches the average FP arithmetic-instruction width."
)


def _fp_arith_byte_weights(counters: set[str], data_type: DataType) -> dict[str, float]:
    """Return the register width in bytes for each FP_ARITH counter."""
    return {
        counter: float(data_type.bytes() if prefix == "SCALAR" else width)
        for counter in counters
        for prefix, width in (("SCALAR", data_type.bytes()), *_FP_ARITH_WIDTH_BYTES.items())
        if prefix in counter
    }


def _fp_arith_byte_compute(
    weights: dict[str, float],
    source: _MemoryInstructionSource,
) -> Callable[[dict[str, float], MetricContext], float]:
    """Return an arithmetic-width byte estimate for one memory-instruction source."""

    def compute(events: dict[str, float], _ctx: MetricContext) -> float:
        total_arith = sum(events[event] for event in weights)
        if total_arith == 0:
            return 0.0
        average_width = sum(events[event] * width for event, width in weights.items()) / total_arith
        return average_width * source.count(events)

    return compute


def _fixed_width_byte_compute(
    source: _MemoryInstructionSource,
) -> Callable[[dict[str, float], MetricContext], float]:
    """Return a configured-width byte estimate for one memory-instruction source."""

    def compute(events: dict[str, float], ctx: MetricContext) -> float:
        return source.count(events) * ctx.bytes_per_instruction

    return compute


def _make_fp_arith_bytes_metrics(
    counters: set[str],
    data_type: DataType,
    priority: int,
    priority_modifier: Callable[[MetricResolutionConfig], int],
    description: str,
) -> list[MetricDefinition]:
    """Build arithmetic-width byte estimates for every memory-instruction source."""
    weights = _fp_arith_byte_weights(counters, data_type)
    return [
        MetricDefinition(
            type=MetricType.BYTES,
            required_events=frozenset(counters | source.required_events),
            compute=_fp_arith_byte_compute(weights, source),
            priority=priority + source.priority_offset,
            priority_modifier=priority_modifier,
            description=f"{description} and {source.description}",
            warning=_FP_ARITH_WIDTH_WARNING,
        )
        for source in _MEMORY_INSTRUCTION_SOURCES
    ]


def _make_fixed_width_bytes_metrics(priority: int) -> list[MetricDefinition]:
    """Build configured-width byte estimates for every memory-instruction source."""
    return [
        MetricDefinition(
            type=MetricType.BYTES,
            required_events=source.required_events,
            compute=_fixed_width_byte_compute(source),
            priority=priority + source.priority_offset,
            description=f"Approximated from {source.description} and bytes per instruction",
        )
        for source in _MEMORY_INSTRUCTION_SOURCES
    ]


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
            # Arithmetic-width estimates for AVX-512, AVX2, and SSE systems.
            *_make_fp_arith_bytes_metrics(
                {_FP_SCALAR_DP, _FP128_DP, _FP256_DP, _FP512_DP},
                DataType.f64,
                100,
                _data_type_match(DataType.f64),
                "Bytes from FP_ARITH vector-width counters",
            ),
            *_make_fp_arith_bytes_metrics(
                {_FP_SCALAR_SP, _FP128_SP, _FP256_SP, _FP512_SP},
                DataType.f32,
                100,
                _data_type_match(DataType.f32),
                "Bytes from FP_ARITH vector-width counters",
            ),
            *_make_fp_arith_bytes_metrics(
                {_FP_SCALAR_DP, _FP128_DP, _FP256_DP},
                DataType.f64,
                99,
                _data_type_match(DataType.f64),
                "Bytes from FP_ARITH vector-width counters",
            ),
            *_make_fp_arith_bytes_metrics(
                {_FP_SCALAR_SP, _FP128_SP, _FP256_SP},
                DataType.f32,
                99,
                _data_type_match(DataType.f32),
                "Bytes from FP_ARITH vector-width counters",
            ),
            *_make_fp_arith_bytes_metrics(
                {_FP_SCALAR_DP, _FP128_DP},
                DataType.f64,
                98,
                _data_type_match(DataType.f64),
                "Bytes from FP_ARITH vector-width counters",
            ),
            *_make_fp_arith_bytes_metrics(
                {_FP_SCALAR_SP, _FP128_SP},
                DataType.f32,
                98,
                _data_type_match(DataType.f32),
                "Bytes from FP_ARITH vector-width counters",
            ),
            *_make_fixed_width_bytes_metrics(90),
            # Relevant for Zen3 (no PAPI load/store events, only DCA and a native forwarding event).
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


def _make_fp_arith_bytes_metrics_for_isas(
    counters: set[str],
    data_type: DataType,
    isas: tuple[type[BaseISA], ...],
) -> list[MetricDefinition]:
    """Build ISA-tailored arithmetic-width byte estimates."""
    return _make_fp_arith_bytes_metrics(
        counters,
        data_type,
        200,
        lambda _: 0,
        f"Bytes from FP_ARITH vector-width counters for {', '.join(isa.__name__ for isa in isas)} ({data_type.name})",
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
        MetricType.BYTES: _make_fp_arith_bytes_metrics_for_isas(counters, config.data_type, config.isas),
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

_PAPI_EVENT_CACHE_PREFIX = "papi_events_"
_PAPI_EVENT_CACHE_VERSION = 2


def _papi_cache_dir() -> Path:
    """Cache directory for PAPI event catalogs (created on first write)."""
    return user_cache_dir_for_carm()


def _papi_cache_probes() -> dict[str, str]:
    """Cheap identity probes; every failure yields 'unknown' so the key always computes."""
    cpu = read_cpuinfo()
    probes = {
        "model": cpu.model_name or "unknown",
        "vendor": cpu.vendor or "unknown",
        "family": cpu.family or "unknown",
        "cpu_model": cpu.model or "unknown",
        "stepping": cpu.stepping or "unknown",
        "arch": platform.machine() or "unknown",
    }
    # papi_version: canonical library version string (fast, ~0.07s)
    papi_version_bin = shutil.which("papi_version")
    probes["papi_version"] = "unknown"
    if papi_version_bin is not None:
        try:
            result = subprocess.run([papi_version_bin], capture_output=True, text=True, timeout=5, check=False)
            if result.returncode == 0:
                probes["papi_version"] = result.stdout.strip() or "unknown"
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass
    # No papi_xml_event_info binary path/mtime/size probe: module systems can
    # re-link or move the wrapper binary between loads even when the underlying
    # PAPI library (and therefore the catalog) is unchanged. papi_version above
    # is the stable build identity.
    probes["kernel"] = platform.release() or "unknown"
    try:
        probes["perf_event_paranoid"] = Path("/proc/sys/kernel/perf_event_paranoid").read_text().strip() or "unknown"
    except OSError:
        probes["perf_event_paranoid"] = "unknown"
    return probes


def _papi_cache_key() -> str:
    """Full SHA-256 hex of sorted 'name=value' probe lines."""
    payload = "\n".join(f"{k}={v}" for k, v in sorted(_papi_cache_probes().items()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _papi_event_cache_path(key: str) -> Path:
    return _papi_cache_dir() / f"{_PAPI_EVENT_CACHE_PREFIX}{key[:16]}.json"


def _load_papi_event_cache(key: str) -> frozenset[str] | None:
    """Return cached events for *key*, or None; emits exactly one detail-status line."""
    path = _papi_event_cache_path(key)
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = None
        if (
            isinstance(data, dict)
            and data.get("version") == _PAPI_EVENT_CACHE_VERSION
            and data.get("key") == key
            and isinstance(data.get("events"), list)
        ):
            detail("PAPI event cache: hit, using cached event catalog")
            return frozenset(data["events"])
        detail("PAPI event cache: miss, cached events are for a different configuration")
        return None
    if any(_papi_cache_dir().glob(f"{_PAPI_EVENT_CACHE_PREFIX}*.json")):
        detail("PAPI event cache: miss, cached events are for a different configuration")
        return None
    detail("PAPI event cache: miss, no cached events")
    return None


def _store_papi_event_cache(key: str, events: frozenset[str]) -> None:
    """Atomically write events for *key*; any failure is debug-logged, never fatal."""
    try:
        directory = _papi_cache_dir()
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _PAPI_EVENT_CACHE_VERSION,
            "key": key,
            "probes": _papi_cache_probes(),
            "events": sorted(events),
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        fd, tmp = tempfile.mkstemp(prefix=".papi_events_tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            os.replace(tmp, _papi_event_cache_path(key))
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        detail(f"PAPI event cache: stored {len(events)} events for this configuration")
    except OSError as exc:
        debug(f"Failed to write PAPI event cache: {exc}")


def parse_available_events(use_cache: bool = True) -> frozenset[str]:
    """Run ``papi_xml_event_info`` and parse event names (base + modifiers) from its XML output.

    The command outputs an XML document containing both base event names
    (e.g. ``FP_ARITH_INST_RETIRED``) and their modifier names
    (e.g. ``FP_ARITH_INST_RETIRED:128B_PACKED_DOUBLE``) across all component
    event sets (NATIVE and PRESET).

    The catalog is cached per machine configuration under the XDG cache
    directory: a matching entry skips the (potentially slow) command, a
    stale or absent entry runs it and stores the result. ``use_cache=False``
    bypasses both the read and the write. Events that the installed library
    cannot add to an event set are filtered out before caching, so the
    catalog contains only collectable events.

    Returns:
        frozenset of available PAPI event name strings. Empty set if
        the ``papi_xml_event_info`` tool is not found or fails.
    """
    key = _papi_cache_key() if use_cache else None
    if key is not None:
        cached = _load_papi_event_cache(key)
        if cached is not None:
            return cached

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

    raw_events = _parse_papi_xml_output(result.stdout)
    library = _load_papi_library()
    if library is not None:
        events = collectable_events(raw_events, library)
        detail(
            f"PAPI event catalog: kept {len(events)} collectable events out of {len(raw_events)} reported by "
            "papi_xml_event_info"
        )
    else:
        debug("PAPI event catalog: collectability filter skipped (cannot load libpapi)")
        events = raw_events
    if key is not None:
        _store_papi_event_cache(key, events)
    return events


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
