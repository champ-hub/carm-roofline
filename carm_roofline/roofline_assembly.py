"""
Reusable roofline model assembly from stored benchmark results.

Provides a pure-function API to load JSONL benchmark records, filter them
by criteria (ISA, threads, data type, load-store ratio, operations), and
assemble a roofline model consisting of memory bandwidth per cache level and
peak performance per arithmetic operation.

Intended to be used by the GUI, CLI, and any future consumer of stored
benchmark results — no CARMContext or GUI dependency.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TypedDict

from carm_roofline.core import ArithmeticIntensity, Bandwidth, Performance
from carm_roofline.output_utils import debug, detail, info, warn


class RecordType(str, Enum):
    ARITHMETIC = "arithmetic"
    MEMORY = "memory"


class BenchmarkRecord(TypedDict, total=False):
    """A single JSONL benchmark record — arithmetic or memory.

    All fields are optional (``total=False``) to reflect that records come
    from external JSONL files and any field may be absent.  Consumers use
    ``.get()`` to handle missing fields defensively.
    """

    type: str
    name: str
    isa: str
    machine: str
    data_type: str
    num_threads: int
    timestamp: str
    # Arithmetic-specific
    operation: str
    performance_gops: float
    # Memory-specific
    load_store_ratio: str
    cache_level: str
    memory_level_name: str
    bandwidth_gbps: float
    actual_frequency_hz: int
    frequency_overridden: bool


class FilterOptions(TypedDict):
    """Sorted unique filter dimension values extracted from records."""

    machine: list[str]
    isa: list[str]
    num_threads: list[int]
    load_store_ratio: list[str]
    data_type: list[str]
    actual_frequency_hz: list[int]


RooflineTuple = tuple[str, str, int, str, str, int]
ALL_TUPLE_FIELDS = ("machine", "isa", "num_threads", "data_type", "load_store_ratio", "actual_frequency_hz")

# ── Filter & model data structures ────────────────────────────────────────────


@dataclass(frozen=True)
class RooflineFilter:
    """Filter criteria for selecting benchmarks to assemble into a roofline.

    Each field is an exact-match criterion except ``operations`` which is
    "any-of" (the record's ``operation`` must be in the set). A field set to
    ``None`` matches every record regardless of that field's value.
    """

    machine: str | None = None
    isa: str | None = None
    num_threads: int | None = None
    data_type: str | None = None
    operations: frozenset[str] | None = None
    load_store_ratio: str | None = None
    actual_frequency_hz: int | None = None


@dataclass(frozen=True)
class AssembledRoofline:
    """A roofline model assembled from filtered benchmark records.

    Contains the memory bandwidth for each cache level and the peak
    performance for each arithmetic operation found in the filtered data.
    """

    filter: RooflineFilter
    bandwidth_by_level: dict[str, Bandwidth] = field(default_factory=dict)
    peak_performance_by_op: dict[str, Performance] = field(default_factory=dict)
    source_timestamps: frozenset[str] = field(default_factory=frozenset)

    def ridge_points(self) -> dict[str, ArithmeticIntensity]:
        """Arithmetic intensity at which each bandwidth ceiling meets the
        highest compute ceiling.

        Returns empty dict when no bandwidth or no performance data is present.
        """
        if not self.peak_performance_by_op or not self.bandwidth_by_level:
            return {}
        peak_perf = max(p.value for p in self.peak_performance_by_op.values())
        return {level: ArithmeticIntensity(peak_perf / bw.value) for level, bw in self.bandwidth_by_level.items()}


@dataclass(frozen=True)
class ApplicationPoint:
    """A single application roofline point from an application run."""

    label: str
    total_flops: float
    total_bytes: float
    runtime_s: float
    num_ranks: int
    num_threads: int
    num_regions: int
    arithmetic_intensity: float
    flops_per_second: float
    bandwidth: float


@dataclass(frozen=True)
class ApplicationRecord:
    """A single application run record with metadata and one or more points."""

    id: str
    label: str
    aggregation: str
    metadata: dict[str, Any]
    points: list[ApplicationPoint]


# ── JSONL loading ─────────────────────────────────────────────────────────────


def load_benchmarks(path: Path) -> list[BenchmarkRecord]:
    """Read a JSONL file and return each non-empty line as a parsed dict.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of parsed JSON objects (one per line).

    Raises:
        FileNotFoundError: The path does not exist.
    """
    records: list[BenchmarkRecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                try:
                    records.append(json.loads(stripped))
                except json.JSONDecodeError as e:
                    warn(f"Skipping malformed JSON line in {path.name}: {e}")
                    continue
    if not records:
        warn(f"Empty benchmark file: {path}")
    else:
        info(f"Loaded {len(records)} benchmark records from {path.parent.name}/{path.name}")
    return records


def load_all_benchmarks(results_root: Path) -> list[BenchmarkRecord]:
    """Load benchmark records from all machine subdirectories under *results_root*.

    Each machine directory is expected to contain a ``benchmarks.jsonl``
    file.  All records are merged into a single list.

    Args:
        results_root: Root directory containing machine subdirectories.

    Returns:
        Merged list of benchmark records from every found file.
        Empty list when no data exists (no exception).
    """
    info(f"Loading benchmarks from {results_root}")
    all_records: list[BenchmarkRecord] = []
    found = 0
    if results_root.exists():
        for entry in sorted(results_root.iterdir()):
            if not entry.is_dir():
                continue
            jsonl_path = entry / "benchmarks.jsonl"
            if jsonl_path.exists():
                try:
                    all_records.extend(load_benchmarks(jsonl_path))
                    found += 1
                except Exception as e:
                    warn(f"Failed to load {jsonl_path}: {e}")
                    continue
    if not all_records:
        warn(f"No benchmark data found in {results_root}")
    else:
        info(f"Loaded {len(all_records)} total benchmark records from {found} file(s)")
    return all_records


def load_applications(path: Path) -> list[ApplicationRecord]:
    """Read a JSONL file of application runs and return one record per line.

    Each line must have ``format_version >= "2.0"``, a ``metadata`` dict with
    ``name``, ``date``, and ``command`` keys, an ``aggregation`` string, and a
    ``points`` list.  Lines that fail validation are logged with ``warn()``
    and skipped.

    Args:
        path: Path to ``applications.jsonl``.

    Returns:
        List of ``ApplicationRecord`` objects.
    """
    records: list[ApplicationRecord] = []
    if not path.exists():
        warn(f"Application file not found: {path}")
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as e:
                warn(f"Skipping malformed JSON line in {path.name}: {e}")
                continue
            fv = raw.get("format_version", "0.0")
            if not isinstance(fv, str) or fv < "2.0":
                warn(f"Skipping application record with format_version < 2.0: {fv}")
                continue
            meta: dict[str, Any] = raw.get("metadata", {})
            name = meta.get("name", "")
            date = meta.get("date", "")
            command = meta.get("command", "")
            aggregation = raw.get("aggregation", "")
            if not name and not date and not aggregation:
                warn(f"Skipping application record with empty metadata: {stripped[:120]}")
                continue
            id_ = f"{name}_{hashlib.sha1(f'{name}|{date}|{aggregation}|{command}'.encode()).hexdigest()[:8]}"
            label = f"{name} \u2014 {date} ({aggregation})"
            raw_points: list[dict[str, Any]] = raw.get("points", []) or []
            points: list[ApplicationPoint] = []
            for p in raw_points:
                points.append(
                    ApplicationPoint(
                        label=p.get("label", ""),
                        total_flops=float(p.get("total_flops", 0)),
                        total_bytes=float(p.get("total_bytes", 0)),
                        runtime_s=float(p.get("runtime_s", 0)),
                        num_ranks=int(p.get("num_ranks", 0)),
                        num_threads=int(p.get("num_threads", 0)),
                        num_regions=int(p.get("num_regions", 0)),
                        arithmetic_intensity=float(p.get("arithmetic_intensity", 0)),
                        flops_per_second=float(p.get("flops_per_second", 0)),
                        bandwidth=float(p.get("bandwidth", 0)),
                    )
                )
            records.append(
                ApplicationRecord(id=id_, label=label, aggregation=aggregation, metadata=meta, points=points)
            )
    if not records:
        warn(f"No valid application records in {path}")
    else:
        info(f"Loaded {len(records)} application record(s) from {path.parent.name}/{path.name}")
    return records


def load_all_applications(results_root: Path) -> list[ApplicationRecord]:
    """Load application records from all machine subdirectories under *results_root*.

    Each machine directory is expected to contain an ``applications.jsonl``
    file.  All records are merged into a single list.

    Args:
        results_root: Root directory containing machine subdirectories.

    Returns:
        Merged list of application records from every found file.
        Empty list when no data exists (no exception).
    """
    info(f"Loading applications from {results_root}")
    all_records: list[ApplicationRecord] = []
    found = 0
    if results_root.exists():
        for entry in sorted(results_root.iterdir()):
            if not entry.is_dir():
                continue
            jsonl_path = entry / "applications.jsonl"
            if jsonl_path.exists():
                try:
                    all_records.extend(load_applications(jsonl_path))
                    found += 1
                except Exception as e:
                    warn(f"Failed to load {jsonl_path}: {e}")
                    continue
    if not all_records:
        info(f"No application data found in {results_root}")
    else:
        info(f"Loaded {len(all_records)} total application record(s) from {found} file(s)")
    return all_records


# ── Filtering ─────────────────────────────────────────────────────────────────


def _is_roofline_memory_record(record: BenchmarkRecord) -> bool:
    """Return True if this memory record is a standard cache-level benchmark suitable for roofline ceiling construction.

    A memory record is roofline-eligible iff its ``memory_level_name`` agrees with its ``cache_level`` — both name the
    same target cache level. Sweep benchmarks deliberately differ (sweep index vs classified level) and are excluded.
    """
    if record.get("type") != RecordType.MEMORY:
        return False
    mem_level = record.get("memory_level_name")
    cache_lvl = record.get("cache_level")
    # Legacy records without memory_level_name are not sweeps.
    if mem_level is None:
        return cache_lvl is not None
    return mem_level == cache_lvl


def _matches_filter(record: BenchmarkRecord, flt: RooflineFilter) -> bool:
    """Return True when *record* matches all non-None fields of *flt*."""
    if flt.machine is not None and record.get("machine") != flt.machine:
        return False
    if flt.isa is not None and record.get("isa") != flt.isa:
        return False
    if flt.num_threads is not None and record.get("num_threads") != flt.num_threads:
        return False
    if flt.data_type is not None and record.get("data_type") != flt.data_type:
        return False
    if flt.actual_frequency_hz is not None and record.get("actual_frequency_hz") != flt.actual_frequency_hz:
        return False

    # load_store_ratio only applies to memory records
    if (
        flt.load_store_ratio is not None
        and record.get("type") == RecordType.MEMORY
        and record.get("load_store_ratio") != flt.load_store_ratio
    ):
        return False
    # operations only applies to arithmetic records
    return not (
        flt.operations is not None
        and record.get("type") == RecordType.ARITHMETIC
        and record.get("operation") not in flt.operations
    )


# ── Roofline assembly ─────────────────────────────────────────────────────────


def assemble_roofline(
    records: list[BenchmarkRecord],
    flt: RooflineFilter,
) -> AssembledRoofline:
    """Filter *records* by *flt* and assemble a roofline model.

    For each cache level the *latest* timestamped bandwidth measurement is
    used; for each arithmetic operation the *latest* timestamped performance
    measurement is used.  This allows aggregation across independent runs
    where, e.g., memory benchmarks were run separately from arithmetic ones.

    When no records match the filter the returned
    ``AssembledRoofline`` carries empty dicts — no exception is raised.
    """
    matched = [r for r in records if _matches_filter(r, flt)]
    if not matched:
        detail(
            f"No benchmark records match filter (isa={flt.isa}, threads={flt.num_threads}, data_type={flt.data_type})"
        )

    bandwidth_by_level: dict[str, Bandwidth] = {}
    peak_performance_by_op: dict[str, Performance] = {}
    timestamps: set[str] = set()

    mem_by_level: dict[str, BenchmarkRecord] = {}
    for rec in matched:
        if not _is_roofline_memory_record(rec):
            continue
        level = rec.get("cache_level")
        if not level:
            continue
        ts = rec.get("timestamp", "")
        if level not in mem_by_level or ts > mem_by_level[level].get("timestamp", ""):
            mem_by_level[level] = rec

    for level, rec in mem_by_level.items():
        gbps = rec.get("bandwidth_gbps")
        if gbps is not None:
            bandwidth_by_level[level] = Bandwidth(float(gbps) * 1e9)
            timestamps.add(rec.get("timestamp", ""))

    # Arithmetic: group by operation → latest timestamp
    arith_by_op: dict[str, BenchmarkRecord] = {}
    for rec in matched:
        if rec.get("type") != RecordType.ARITHMETIC:
            continue
        op = rec.get("operation")
        if not op:
            continue
        ts = rec.get("timestamp", "")
        if op not in arith_by_op or ts > arith_by_op[op].get("timestamp", ""):
            arith_by_op[op] = rec

    for op, rec in arith_by_op.items():
        gops = rec.get("performance_gops")
        if gops is not None:
            peak_performance_by_op[op] = Performance(float(gops) * 1e9)
            timestamps.add(rec.get("timestamp", ""))

    if bandwidth_by_level:
        levels_str = ", ".join(f"{k}={v}" for k, v in bandwidth_by_level.items())
        debug(f"Bandwidth per level: {levels_str}")
    if peak_performance_by_op:
        ops_str = ", ".join(f"{k}={v}" for k, v in peak_performance_by_op.items())
        debug(f"Peak performance per op: {ops_str}")
    debug(f"Source timestamps: {len(timestamps)} run(s)")

    return AssembledRoofline(
        filter=flt,
        bandwidth_by_level=bandwidth_by_level,
        peak_performance_by_op=peak_performance_by_op,
        source_timestamps=frozenset(timestamps),
    )


def assemble_roofline_from_file(
    path: Path,
    flt: RooflineFilter,
) -> AssembledRoofline:
    """Convenience: load benchmarks then assemble roofline in one call."""
    records = load_benchmarks(path)
    return assemble_roofline(records, flt)


def _compute_valid_tuples(records: list[BenchmarkRecord]) -> frozenset[RooflineTuple]:
    """Return frozenset of (machine, isa, num_threads, data_type, load_store_ratio, actual_frequency_hz) tuples where
    both arithmetic and roofline-eligible memory records exist for the same
    (machine, isa, num_threads, data_type, actual_frequency_hz) 5-tuple key — the precondition for assembling a
    complete roofline.  Records without ``actual_frequency_hz`` get frequency ``0`` so they pair up with each other.
    """
    arith_5tuples = {
        (r["machine"], r["isa"], r["num_threads"], r["data_type"], r.get("actual_frequency_hz", 0))
        for r in records
        if r.get("type") == RecordType.ARITHMETIC
        and r.get("machine")
        and r.get("isa")
        and r.get("num_threads")
        and r.get("data_type")
    }
    return frozenset(
        (
            r["machine"],
            r["isa"],
            r["num_threads"],
            r["data_type"],
            str(r["load_store_ratio"]),
            r.get("actual_frequency_hz", 0),
        )
        for r in records
        if _is_roofline_memory_record(r)
        and "load_store_ratio" in r
        and (r.get("machine"), r.get("isa"), r.get("num_threads"), r.get("data_type"), r.get("actual_frequency_hz", 0))
        in arith_5tuples
    )


def discover_filter_options(
    records: list[BenchmarkRecord],
    flt: RooflineFilter | None = None,
) -> FilterOptions:
    """Extract sorted unique filter dimension values from benchmark records.

    When *flt* is provided, each field's returned values are constrained to those appearing in valid roofline tuples
    that match ALL OTHER (non-None) fields of the filter. A field's own lock never constrains its own options — this
    lets every dropdown always show the full set of viable alternatives given the other selections.

    Every returned value comes from a tuple that has both arithmetic and memory records, so every option can form a
    complete roofline.
    """
    valid = _compute_valid_tuples(records)
    if flt is None:
        flt = RooflineFilter()
    sel = (flt.machine, flt.isa, flt.num_threads, flt.data_type, flt.load_store_ratio, flt.actual_frequency_hz)
    result: dict[str, list[Any]] = {}
    for i, f in enumerate(ALL_TUPLE_FIELDS):
        filtered = valid
        for j, _ in enumerate(ALL_TUPLE_FIELDS):
            if j != i and sel[j] is not None:
                filtered = frozenset(t for t in filtered if t[j] == sel[j])
        result[f] = sorted({t[i] for t in filtered})
    debug(
        f"Available options: {len(result['machine'])} machine(s), {len(result['isa'])} ISA(s), "
        f"{len(result['num_threads'])} thread count(s), {len(result['load_store_ratio'])} ratio(s), "
        f"{len(result['data_type'])} data type(s), "
        f"{len(result['actual_frequency_hz'])} frequency(ies)"
    )
    return FilterOptions(
        machine=result["machine"],
        isa=result["isa"],
        num_threads=result["num_threads"],
        data_type=result["data_type"],
        load_store_ratio=result["load_store_ratio"],
        actual_frequency_hz=result["actual_frequency_hz"],
    )


__all__ = [
    "ApplicationPoint",
    "ApplicationRecord",
    "AssembledRoofline",
    "BenchmarkRecord",
    "FilterOptions",
    "RecordType",
    "RooflineFilter",
    "assemble_roofline",
    "assemble_roofline_from_file",
    "discover_filter_options",
    "load_all_applications",
    "load_all_benchmarks",
    "load_applications",
    "load_benchmarks",
]
