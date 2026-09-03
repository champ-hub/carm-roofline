"""Optional roofline metrics beyond the core FLOPS/BYTES pair.

The CLI is metric-centric: FLOPS and BYTES are always selected (needed for
roofline plotting); other metrics are optional and selected by name (first
example: cache residency). Each optional metric maps to one or more
alternative event sets per backend; the first alternative whose events are
all available on the current system wins.

Event maps are keyed by :class:`.shared.BackendType`, defined there so this
module stays import-independent of :mod:`.config` (which imports this
module). Metric names are :class:`OptionalMetricName` members; the enum value
is the ``--metrics`` CLI string.

Cache-residency level model: serialized level keys are the canonical level
names from :data:`_CACHE_LEVEL_ORDER` (``l1`` < ``l2`` < ``l3`` < ``dram``).
A bucket is either an exact level or a ``"<level>plus"`` bucket meaning
"this level and everything beyond it" — e.g. ``l3plus`` groups L3 and DRAM,
``l2plus`` groups L2, L3 and DRAM. The bucket set a metric alternative emits
derives from the miss boundaries it can measure (its ``levels`` field): one
bucket per measured boundary plus a final everything-beyond bucket.

Most platforms expose no countable L3/LLC miss event — AMD Zen 3+ kernels
expose no ``amd_l3`` PMU, and ``perf::CACHE-MISSES`` is an L2 IC+DC
demand-miss event, not an L3 event — so "traffic that left L2" (demand L2
misses + L2 prefetch fills) is attributed to a single L3/DRAM bucket. Intel
perf, by contrast, exposes ``LLC-load-misses``/``LLC-store-misses``, so the
Intel perf alternative splits L3 and DRAM into separate buckets.

The L1 access count is scaled to line granularity before computing miss
rates: ``PAPI_L1_DCA`` counts per-instruction load dispatches while
``PAPI_L1_DCM`` counts per-64B-line misses, so
``total_lines = l1_accesses * bytes_per_instruction / 64``. L2 accounting is
prefetch-inclusive: demand L2 misses alone hide the L2 hardware prefetcher
filling L2 ahead of demand, so prefetch fills that leave the L2 count toward
the everything-beyond bucket too.

To add a new metric set, add an alternative role map declaring the
measurable ``levels`` and their events; the emitted bucket names and the
compute derive from it automatically.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Literal, TypedDict, Union

from carm_roofline.core import UserError
from carm_roofline.output_utils import warn

from .shared import BackendType


class OptionalMetricName(Enum):
    """Names of the available optional metrics (the ``--metrics`` values)."""

    CACHE_RESIDENCY = "cache-residency"
    CACHE_LINE_UTILIZATION = "cache-line-utilization"


# This is the existing fixed cache-line assumption for optional cache metrics.
CACHE_LINE_BYTES = 64.0


# Canonical cache hierarchy, smallest first. Serialized level keys are these
# names; a "<level>plus" bucket means "this level and everything beyond it".
_CACHE_LEVEL_ORDER: tuple[str, ...] = ("l1", "l2", "l3", "dram")


# The level set is dynamic: it derives from the miss boundaries a role map
# can measure (see ``_CACHE_LEVEL_ORDER`` and ``CacheRoles["levels"]``), so
# per-level resident bytes are a plain dict rather than a fixed TypedDict.
CacheLevelBytes = dict[str, float]


class DirectCacheRoles(TypedDict, total=False):
    """Role -> event map for the PAPI direct-counter shape.

    ``shape`` is a static type discriminator, never an event name.
    ``levels`` lists the canonical levels at which a MISS boundary is
    measurable — always a contiguous prefix of ``("l1", "l2", "l3")``.
    ``l1_accesses``, ``l1_misses`` and ``l2_misses`` are always present; the
    L2-prefetch roles may be absent per role map (PAPI exposes no Intel
    L2-prefetch presets, so the Intel alternative omits them), and
    ``l3_misses`` is present only when the platform exposes an L3/LLC miss
    counter.
    """

    shape: Literal["direct"]
    levels: tuple[str, ...]
    l1_accesses: str
    l1_misses: str
    l2_misses: str
    l2_pf_hit_l3: str
    l2_pf_l3: str
    l3_misses: str


class PairCacheRoles(TypedDict, total=False):
    """Role -> event map for the perf load/store-pair shape.

    The L2-prefetch roles may be absent per role map (Intel's
    ``l2_rqsts.miss`` already counts prefetch fills, so it needs none).
    ``l3_load_misses``/``l3_store_misses`` are present only when the platform
    exposes an L3/LLC miss counter (Intel perf's ``LLC-load-misses`` /
    ``LLC-store-misses``).
    """

    shape: Literal["pairs"]
    levels: tuple[str, ...]
    l1_loads: str
    l1_load_misses: str
    l1_stores: str
    l1_store_misses: str
    l2_misses: str
    l2_pf_hit_l3: str
    l2_pf_l3: str
    l3_load_misses: str
    l3_store_misses: str


CacheRoles = Union[DirectCacheRoles, PairCacheRoles]

# Computes return real dicts, but the declared type is the wider Mapping so
# shared.compute_region_point's existing ``cast(dict[str, float], levels)``
# (which narrows to the JSON-facing container type) stays non-redundant.
CacheCompute = Callable[[Mapping[str, float], float, CacheRoles, float], Mapping[str, float]]
OptionalMetricCompute = Callable[[Mapping[str, float], float, float], Mapping[str, float]]


def _event_names(roles: CacheRoles) -> frozenset[str]:
    """Event names a role map requires (excludes the ``shape`` discriminator)."""
    levels = roles["levels"]
    names: list[str]
    if roles["shape"] == "direct":
        names = [roles["l1_accesses"], roles["l1_misses"]]
        if "l2" in levels:
            names.append(roles["l2_misses"])
        if "l3" in levels:
            names.append(roles["l3_misses"])
    else:
        names = [
            roles["l1_loads"],
            roles["l1_load_misses"],
            roles["l1_stores"],
            roles["l1_store_misses"],
        ]
        if "l2" in levels:
            names.append(roles["l2_misses"])
        if "l3" in levels:
            names.append(roles["l3_load_misses"])
            names.append(roles["l3_store_misses"])
    # Optional L2-prefetch roles (part of the L2 boundary): include only the
    # keys present in the role map.
    if "l2" in levels:
        if "l2_pf_hit_l3" in roles:
            names.append(roles["l2_pf_hit_l3"])
        if "l2_pf_l3" in roles:
            names.append(roles["l2_pf_l3"])
    return frozenset(names)


@dataclass(frozen=True)
class OptionalMetricImplementation:
    """One event-set implementation of an optional metric."""

    required_events: frozenset[str]
    compute: OptionalMetricCompute


@dataclass(frozen=True)
class OptionalMetric:
    """An optional metric with ordered implementations for each backend."""

    description: str
    implementations: Mapping[BackendType, tuple[OptionalMetricImplementation, ...]]


@dataclass(frozen=True)
class ResolvedOptionalMetric:
    """An optional metric bound to one available implementation."""

    metric: OptionalMetric
    implementation: OptionalMetricImplementation

    @property
    def required_events(self) -> frozenset[str]:
        return self.implementation.required_events


def _last_bucket(levels: tuple[str, ...]) -> str:
    """Name of the everything-beyond bucket for a boundary level set.

    ``levels`` is a contiguous prefix of ``("l1", "l2", "l3")``. When the
    last boundary is L3 the beyond-bucket is the exact ``"dram"`` level;
    otherwise it is the next canonical level plus ``"plus"`` (``("l1",)`` ->
    ``l2plus``, ``("l1", "l2")`` -> ``l3plus``).
    """
    nxt = _CACHE_LEVEL_ORDER[_CACHE_LEVEL_ORDER.index(levels[-1]) + 1]
    return nxt if nxt == "dram" else f"{nxt}plus"


def _boundary_misses(counters: Mapping[str, float], roles: CacheRoles) -> dict[str, float]:
    """Miss count at each measurable boundary level (only ``roles["levels"]``).

    The L1 and L3 boundaries are shape-dependent (direct counters vs
    load+store pairs); the L2 boundary is the L2 miss count plus the
    prefetch-inclusive roles when the role map declares them. Missing
    counters default to 0.0 (e.g. a partitioned run that collected only part
    of the event set).
    """
    levels = roles["levels"]
    misses: dict[str, float] = {}
    if roles["shape"] == "direct":
        misses["l1"] = counters.get(roles["l1_misses"], 0.0)
    else:
        misses["l1"] = counters.get(roles["l1_load_misses"], 0.0) + counters.get(roles["l1_store_misses"], 0.0)
    if "l2" in levels:
        l2 = counters.get(roles["l2_misses"], 0.0)
        if "l2_pf_hit_l3" in roles:
            l2 += counters.get(roles["l2_pf_hit_l3"], 0.0)
        if "l2_pf_l3" in roles:
            l2 += counters.get(roles["l2_pf_l3"], 0.0)
        misses["l2"] = l2
    if "l3" in levels:
        if roles["shape"] == "direct":
            misses["l3"] = counters.get(roles["l3_misses"], 0.0)
        else:
            misses["l3"] = counters.get(roles["l3_load_misses"], 0.0) + counters.get(roles["l3_store_misses"], 0.0)
    return misses


def _cache_level_bytes(
    accesses: float,
    misses: Mapping[str, float],
    levels: tuple[str, ...],
    region_bytes: float,
    bytes_per_instruction: float,
) -> CacheLevelBytes:
    """Resident bytes per bucket from line-granularity miss fractions.

    Semantics: fraction of the region's total 64B-line traffic served at
    each measured boundary level, plus the everything-beyond bucket, scaled
    to resident bytes.

    The L1 access counter counts per-instruction load dispatches while the
    L1 miss counter counts per-64B-line misses, so accesses are scaled to
    line granularity first:
    ``total_lines = accesses * bytes_per_instruction / 64``.

    Miss fractions telescope and saturate so they always sum to exactly 1:
    ``prev`` tracks the fraction of line traffic not yet attributed; each
    boundary can claim at most what remains of ``prev``. A counter that
    overcounts its traffic class would otherwise make a fraction negative or
    push the sum past 1. For ``levels == ("l1", "l2")`` this reproduces
    ``f_l1 = 1 - min(m1/total, 1)``, ``f_l3plus = min(fills/total, 1 - f_l1)``,
    ``f_l2 = 1 - f_l1 - f_l3plus`` where ``fills`` is the L2 boundary count.
    """
    total_lines = accesses * bytes_per_instruction / CACHE_LINE_BYTES
    buckets = (*levels, _last_bucket(levels))
    if total_lines <= 0:
        return dict.fromkeys(buckets, 0.0)

    prev = 1.0  # fraction of line traffic not yet attributed
    out: dict[str, float] = {}
    for lvl in levels:
        miss_frac = min(misses[lvl] / total_lines, prev)  # can't attribute more than remains
        out[lvl] = (prev - miss_frac) * region_bytes
        prev = miss_frac
    out[_last_bucket(levels)] = prev * region_bytes
    return out


def _cache_residency_compute(
    counters: Mapping[str, float],
    region_bytes: float,
    roles: CacheRoles,
    bytes_per_instruction: float,
) -> CacheLevelBytes:
    """Compute per-level resident bytes for a region (see ``_cache_level_bytes``).

    The PAPI alternative reads accesses/misses directly; the perf alternative
    derives them from the load+store pairs. Load+store (data) semantics —
    matches ``PAPI_L1_DCA`` and the perf load+store pairs. The bucket set is
    derived from ``roles["levels"]``: the miss boundaries the role map can
    measure, plus the everything-beyond bucket. Missing counters default to
    0.0.
    """
    if roles["shape"] == "direct":
        accesses = counters.get(roles["l1_accesses"], 0.0)
    else:
        accesses = counters.get(roles["l1_loads"], 0.0) + counters.get(roles["l1_stores"], 0.0)
    return _cache_level_bytes(
        accesses,
        _boundary_misses(counters, roles),
        roles["levels"],
        region_bytes,
        bytes_per_instruction,
    )


def _cache_residency_implementation(roles: CacheRoles) -> OptionalMetricImplementation:
    """Bind cache-residency computation to one role map."""

    def compute(
        counters: Mapping[str, float], region_bytes: float, bytes_per_instruction: float
    ) -> Mapping[str, float]:
        return _cache_residency_compute(counters, region_bytes, roles, bytes_per_instruction)

    return OptionalMetricImplementation(required_events=_event_names(roles), compute=compute)


def _make_cache_residency() -> OptionalMetric:
    """Build the cache-residency optional metric.

    Levels derive from each role map's measurable miss boundaries (the
    ``levels`` field): one bucket per boundary plus an everything-beyond
    bucket. The AMD paths measure L1 and L2 only -> ``{l1, l2, l3plus}``
    (L3 and DRAM grouped, since AMD Zen 3+ exposes no countable L3/LLC miss
    event). The Intel perf path additionally measures L3
    (``LLC-load-misses``/``LLC-store-misses``) -> ``{l1, l2, l3, dram}``.

    PAPI alternatives (AMD first — on AMD both resolve, and the AMD map's
    prefetch-inclusive L2 accounting is the correct one): the AMD map uses
    the L1 presets plus the native L2 events. PAPI names the native events
    after the perf ones; on Zen 3+ ``l2_cache_misses_from_dc_misses`` is
    ``CORE_TO_L2_CACHEABLE_REQUEST_ACCESS_STATUS:LS_RD_BLK_C`` (event 0x64,
    umask 0x8 — verified identical to ``PAPI_L2_DCM`` on this machine) and
    ``l2_pf_miss_l2_hit_l3``/``l2_pf_miss_l2_l3`` are
    ``L2_PREFETCH_HIT_L3:L2_HW_PREFETCHER``/``L2_PREFETCH_MISS_L3:L2_HW_PREFETCHER``.
    The Intel map is demand-only: PAPI has no Intel L2-prefetch presets, so
    its ``l3plus`` undercounts when the L2 prefetcher runs; the accurate
    Intel path is the perf backend's ``l2_rqsts.miss`` (all L2 misses,
    including prefetch).

    NOTE: the Intel PAPI map deliberately stays 3-bucket (levels ``l1/l2``).
    Do NOT add ``PAPI_L3_TCM`` to it: on Intel the PAPI L2 preset is
    demand-only while ``PAPI_L3_TCM`` counts *all* L3 misses (including
    prefetch fills), so the boundary ordering (L2 misses >= L3 misses) would
    break and the telescoping would misattribute traffic. The correct
    4-bucket Intel path is the perf alternative below.

    perf alternatives (AMD first for the same reason): the AMD map adds the
    L2-prefetch fills to demand L2 misses; the Intel map relies on
    ``l2_rqsts.miss`` which already includes prefetch fills and adds the
    ``LLC-load-misses``/``LLC-store-misses`` L3 boundary to separate L3 from
    DRAM.
    """
    papi_amd: DirectCacheRoles = {
        "shape": "direct",
        "levels": ("l1", "l2"),
        "l1_accesses": "PAPI_L1_DCA",
        "l1_misses": "PAPI_L1_DCM",
        "l2_misses": "CORE_TO_L2_CACHEABLE_REQUEST_ACCESS_STATUS:LS_RD_BLK_C",
        "l2_pf_hit_l3": "L2_PREFETCH_HIT_L3:L2_HW_PREFETCHER",
        "l2_pf_l3": "L2_PREFETCH_MISS_L3:L2_HW_PREFETCHER",
    }
    papi_intel: DirectCacheRoles = {
        "shape": "direct",
        "levels": ("l1", "l2"),
        "l1_accesses": "PAPI_L1_DCA",
        "l1_misses": "PAPI_L1_DCM",
        "l2_misses": "PAPI_L2_DCM",
    }
    perf_amd: PairCacheRoles = {
        "shape": "pairs",
        "levels": ("l1", "l2"),
        "l1_loads": "L1-dcache-loads",
        "l1_load_misses": "L1-dcache-load-misses",
        "l1_stores": "L1-dcache-stores",
        "l1_store_misses": "L1-dcache-store-misses",
        "l2_misses": "l2_cache_misses_from_dc_misses",
        "l2_pf_hit_l3": "l2_pf_miss_l2_hit_l3",
        "l2_pf_l3": "l2_pf_miss_l2_l3",
    }
    perf_intel: PairCacheRoles = {
        "shape": "pairs",
        "levels": ("l1", "l2", "l3"),
        "l1_loads": "L1-dcache-loads",
        "l1_load_misses": "L1-dcache-load-misses",
        "l1_stores": "L1-dcache-stores",
        "l1_store_misses": "L1-dcache-store-misses",
        "l2_misses": "l2_rqsts.miss",
        "l3_load_misses": "LLC-load-misses",
        "l3_store_misses": "LLC-store-misses",
    }
    return OptionalMetric(
        description=(
            "Fraction of memory traffic served at each cache level per region "
            "(exact levels, or grouped '<level>plus' buckets when the platform "
            "lacks the counters), with per-level resident bytes"
        ),
        implementations={
            BackendType.PAPI: (
                _cache_residency_implementation(papi_amd),
                _cache_residency_implementation(papi_intel),
            ),
            BackendType.PERF: (
                _cache_residency_implementation(perf_amd),
                _cache_residency_implementation(perf_intel),
            ),
        },
    )


def _cache_line_utilization_implementation(*miss_events: str) -> OptionalMetricImplementation:
    """Build one implementation that records L1 miss traffic."""

    def compute(
        counters: Mapping[str, float], region_bytes: float, bytes_per_instruction: float
    ) -> Mapping[str, float]:
        return {"l1-miss": sum(counters[event] for event in miss_events) * CACHE_LINE_BYTES}

    return OptionalMetricImplementation(required_events=frozenset(miss_events), compute=compute)


def _make_cache_line_utilization() -> OptionalMetric:
    """Build the cache-line-utilization optional metric."""
    return OptionalMetric(
        description="Application bytes divided by L1 data-miss bytes",
        implementations={
            BackendType.PAPI: (_cache_line_utilization_implementation("PAPI_L1_DCM"),),
            BackendType.PERF: (
                _cache_line_utilization_implementation(
                    "L1-dcache-load-misses",
                    "L1-dcache-store-misses",
                ),
            ),
        },
    )


# Registry of available optional metrics (extend with new metrics here)
OPTIONAL_METRICS: dict[OptionalMetricName, OptionalMetric] = {
    OptionalMetricName.CACHE_RESIDENCY: _make_cache_residency(),
    OptionalMetricName.CACHE_LINE_UTILIZATION: _make_cache_line_utilization(),
}


def validate_metric_names(names: list[str] | None) -> tuple[OptionalMetricName, ...]:
    """Dedupe and validate metric names, preserving first-seen order.

    Args:
        names: Raw ``--metrics`` values (space-separated; may be None/empty).

    Returns:
        Deduplicated tuple of valid metric names.

    Raises:
        UserError: If a name is not in :data:`OPTIONAL_METRICS`.
    """
    if not names:
        return ()
    result: list[OptionalMetricName] = []
    for raw in names:
        try:
            name = OptionalMetricName(raw)
        except ValueError:
            raise UserError(
                f"Unknown optional metric '{raw}'. Available: {', '.join(sorted(m.value for m in OptionalMetricName))}"
            ) from None
        if name not in result:
            result.append(name)
    return tuple(result)


def resolve_optional_metrics(
    names: tuple[OptionalMetricName, ...],
    available_events: frozenset[str],
    backend: BackendType,
) -> dict[OptionalMetricName, ResolvedOptionalMetric]:
    """Resolve selected optional metrics against the backend's available events.

    Per name, the first alternative whose events are all available wins. A
    metric with no resolving alternative is skipped with a warning naming the
    events missing from its most-complete alternative; an unsupported backend
    skips the metric with a warning. All unavailable yields ``{}`` — a
    roofline-only run proceeds (not an error).

    Args:
        names: Validated optional metric names.
        available_events: Events the backend can collect on this system.
        backend: Profiler backend.

    Returns:
        Mapping metric name -> resolved metric (empty when none resolve).
    """
    resolved: dict[OptionalMetricName, ResolvedOptionalMetric] = {}
    for name in names:
        metric = OPTIONAL_METRICS[name]
        alternatives = metric.implementations.get(backend)
        if alternatives is None:
            warn(f"Optional metric '{name.value}' is not supported by the {backend.value} backend; skipping")
            continue
        best: OptionalMetricImplementation | None = None
        best_missing: frozenset[str] | None = None
        for implementation in alternatives:
            missing = implementation.required_events - available_events
            if best is None or len(missing) < len(best_missing or ()):
                best, best_missing = implementation, missing
            if not missing:
                resolved[name] = ResolvedOptionalMetric(metric=metric, implementation=implementation)
                break
        else:
            warn(
                f"Optional metric '{name.value}' is not available on this system; skipping. "
                f"Missing events: {sorted(best_missing or ())}"
            )
    return resolved
