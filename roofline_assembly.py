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

from output_utils import debug, detail, info, warn
from units import ArithmeticIntensity, Bandwidth, Performance


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
    bandwidth_gbps: float


class FilterOptions(TypedDict):
    """Sorted unique filter dimension values extracted from records."""

    machine: list[str]
    isa: list[str]
    threads: list[int]
    load_store_ratio: list[str]
    data_type: list[str]


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

    # Memory: group by cache_level → latest timestamp
    mem_by_level: dict[str, BenchmarkRecord] = {}
    for rec in matched:
        if rec.get("type") != RecordType.MEMORY:
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


# ── Discovery helpers ─────────────────────────────────────────────────────────


def discover_filter_options(
    records: list[BenchmarkRecord],
) -> FilterOptions:
    """Extract sorted unique filter dimensions from benchmark records.

    Returns a ``FilterOptions`` dict with keys ``machine``, ``isa``,
    ``threads``, ``load_store_ratio``, and ``data_type`` — each a sorted list
    suitable for populating UI dropdowns.  ``threads`` values are sorted
    numerically.
    """
    machines = sorted({r["machine"] for r in records if "machine" in r})
    isas = sorted({r["isa"] for r in records if "isa" in r})
    threads = sorted({r["num_threads"] for r in records if "num_threads" in r})
    load_store_ratios = sorted(
        {r["load_store_ratio"] for r in records if r.get("type") == RecordType.MEMORY and "load_store_ratio" in r}
    )
    data_types = sorted({r["data_type"] for r in records if "data_type" in r})
    result: FilterOptions = {
        "machine": machines,
        "isa": isas,
        "threads": threads,
        "load_store_ratio": load_store_ratios,
        "data_type": data_types,
    }
    detail(
        f"Available options: {len(machines)} machine(s), {len(isas)} ISA(s), "
        f"{len(threads)} thread count(s), {len(load_store_ratios)} ratio(s), "
        f"{len(data_types)} data type(s)"
    )
    return result


def discover_filter_options_for_selection(
    records: list[BenchmarkRecord],
    *,
    machine: str | None = None,
    isa: str | None = None,
    num_threads: int | None = None,
    data_type: str | None = None,
    load_store_ratio: str | None = None,
) -> FilterOptions:
    """For each filter field, return values present in records matching
    the current selections in ALL OTHER (non-None) fields.

    A None field means "don't filter on this field" — it widens the
    matching record set.  For each field X, records are filtered by every
    OTHER non-None field via ``_matches_filter``, then unique values of X
    are extracted from the matching subset.  ``threads`` values are sorted
    numerically; ``load_store_ratio`` values are extracted from memory
    records only (arithmetic records lack this field).
    Returns a ``FilterOptions`` dict.
    """
    # Machines: filter by isa, threads, data_type, load_store_ratio
    flt = RooflineFilter(isa=isa, num_threads=num_threads, data_type=data_type, load_store_ratio=load_store_ratio)
    machines = sorted({r["machine"] for r in records if "machine" in r and _matches_filter(r, flt)})

    # ISAs: filter by machine, threads, data_type, load_store_ratio
    flt = RooflineFilter(
        machine=machine, num_threads=num_threads, data_type=data_type, load_store_ratio=load_store_ratio
    )
    isas = sorted({r["isa"] for r in records if "isa" in r and _matches_filter(r, flt)})

    # Threads: filter by machine, isa, data_type, load_store_ratio
    flt = RooflineFilter(machine=machine, isa=isa, data_type=data_type, load_store_ratio=load_store_ratio)
    threads = sorted({r["num_threads"] for r in records if "num_threads" in r and _matches_filter(r, flt)})

    # Data types: filter by machine, isa, threads, load_store_ratio
    flt = RooflineFilter(machine=machine, isa=isa, num_threads=num_threads, load_store_ratio=load_store_ratio)
    data_types = sorted({r["data_type"] for r in records if "data_type" in r and _matches_filter(r, flt)})

    # Load-store ratios: filter by machine, isa, threads, data_type; only from memory records
    flt = RooflineFilter(machine=machine, isa=isa, num_threads=num_threads, data_type=data_type)
    ls_ratios = sorted(
        {
            r["load_store_ratio"]
            for r in records
            if r.get("type") == RecordType.MEMORY and "load_store_ratio" in r and _matches_filter(r, flt)
        }
    )

    return {
        "machine": machines,
        "isa": isas,
        "threads": threads,
        "load_store_ratio": ls_ratios,
        "data_type": data_types,
    }


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
    "discover_filter_options_for_selection",
    "load_all_applications",
    "load_all_benchmarks",
    "load_applications",
    "load_benchmarks",
]
