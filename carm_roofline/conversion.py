from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from carm_roofline.core import UserError

APPLICATION_SUFFIX = "_applications.csv"
ROOFLINE_SUFFIX = "_roofline.csv"
APPLICATION_HEADERS = ("Date", "Method", "Name", "ISA", "Precision", "Threads", "AI", "Gflops", "Bandwidth", "Time")
ROOFLINE_HEADERS = (
    "Date",
    "ISA",
    "Precision",
    "Threads",
    "Loads",
    "Stores",
    "Interleaved",
    "DRAM Bytes",
    "FP Inst.",
    "GB/s",
    "I/Cycle",
    "GB/s",
    "I/Cycle",
    "GB/s",
    "I/Cycle",
    "GB/s",
    "I/Cycle",
    "Gflop/s",
    "I/Cycle",
    "Gflop/s",
    "I/Cycle",
)
CACHE_LEVELS = ("L1", "L2", "L3", "DRAM")
ISA_NAMES = {"scalar": "x86", "sse": "x86_sse", "avx": "x86_avx", "avx2": "x86_avx2", "avx512": "x86_avx512"}
DATA_TYPES = {"sp": "f32", "dp": "f64"}


@dataclass(frozen=True)
class ConversionOptions:
    input_path: Path
    from_format: Literal["legacy-csv", "jsonl"]
    to_format: Literal["legacy-csv", "jsonl"]
    output_dir: Path
    overwrite: bool = False
    dry_run: bool = False
    allow_lossy: bool = False


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return slug or "legacy"


def _normalise_isa(value: str) -> str | None:
    stripped = value.strip()
    return ISA_NAMES.get(stripped.lower(), stripped.lower()) or None


def _normalise_data_type(value: str) -> str:
    stripped = value.strip()
    return DATA_TYPES.get(stripped.lower(), stripped.lower())


def _legacy_timestamp(value: str) -> str:
    stripped = value.strip()
    try:
        return datetime.strptime(stripped, "%Y-%m-%d %H:%M:%S").isoformat()
    except ValueError:
        return stripped


def _input_files(path: Path, suffixes: tuple[str, ...]) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise UserError(f"input path does not exist: {path}")
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)


def _read_legacy(path: Path) -> tuple[str, list[dict[str, str]], list[str], list[str] | None, list[list[str]]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        sample = source.read(4096)
        source.seek(0)
        first_line = sample.splitlines()[0] if sample.splitlines() else ""
        if first_line.startswith("Name:,") or ",L1 Size:," in first_line:
            rows = list(csv.reader(source))
            if len(rows) < 2:
                raise UserError(f"legacy roofline CSV needs metadata and header rows: {path}")
            return "benchmarks", [], [], rows[0], rows
        reader = csv.DictReader(source)
        headers = reader.fieldnames or []
        if not set(APPLICATION_HEADERS).issubset(headers):
            raise UserError(
                f"unrecognized legacy CSV headers in {path}; expected application or two-header roofline CSV"
            )
        return "applications", list(reader), list(headers), None, []


def _application_record(
    row: dict[str, str], headers: Sequence[str], row_index: int, source_name: str
) -> dict[str, Any]:
    try:
        ai = float(row["AI"])
        gflops = float(row["Gflops"])
        runtime = float(row["Time"])
        threads = int(row["Threads"])
    except (KeyError, ValueError) as error:
        raise UserError(f"invalid required application CSV value in {source_name} row {row_index}: {error}") from error
    if not all(math.isfinite(value) for value in (ai, gflops, runtime)):
        raise UserError(f"non-finite application metric in {source_name} row {row_index}")
    if ai <= 0:
        raise UserError(f"AI must be positive in {source_name} row {row_index}; cannot derive bandwidth")
    if runtime < 0 or gflops < 0 or threads < 0:
        raise UserError(f"negative runtime, performance, or thread count in {source_name} row {row_index}")
    name = row["Name"].strip()
    raw_row = [row.get(header) or "" for header in headers]
    return {
        "format_version": "3.0",
        "aggregation": "global",
        "optional_metrics": [],
        "metadata": {
            "name": name,
            "date": _legacy_timestamp(row["Date"]),
            "method": row["Method"].strip(),
            "isa": _normalise_isa(row["ISA"]),
            "precision": _normalise_data_type(row["Precision"]),
            "threads_per_rank": None,
            "command": "",
            "notes": "Converted from legacy CSV",
            "num_ranks": None,
            "total_threads": threads,
        },
        "points": [
            {
                "label": name,
                "total_flops": None,
                "total_bytes": None,
                "runtime_s": runtime,
                "num_ranks": None,
                "num_threads": threads,
                "num_regions": None,
                "arithmetic_intensity": ai,
                "flops_per_second": gflops * 1e9,
                "bandwidth": (gflops / ai) * 1e9,
                "optional_bytes": {},
                "optional_fractions": {},
            }
        ],
        "legacy_source": {
            "kind": "applications",
            "source_name": source_name,
            "row_index": row_index,
            "headers": list(headers),
            "row": raw_row,
        },
    }


def _metadata_value(row: Sequence[str], label: str) -> str:
    try:
        return row[row.index(label) + 1].strip()
    except (ValueError, IndexError) as error:
        raise UserError(f"missing legacy roofline metadata field {label}") from error


def _roofline_records(path: Path, source_name: str, machine_name: str) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.reader(source))
    if len(rows) < 2:
        raise UserError(f"legacy roofline CSV needs metadata and header rows: {path}")
    metadata, headers = rows[0], rows[1]
    _metadata_value(metadata, "Name:")

    def column(group: str, metric: str) -> int:
        matches = [
            i
            for i, (meta, header) in enumerate(zip(metadata, headers))
            if meta.strip() == group and header.strip() == metric
        ]
        if len(matches) != 1:
            raise UserError(f"expected exactly one {group} {metric} column in {path}")
        return matches[0]

    bandwidth_indexes = {level: column(level, "GB/s") for level in CACHE_LEVELS}
    fp_index = column("FP", "Gflop/s")
    fma_index = column("FP FMA", "Gflop/s")
    records: list[dict[str, Any]] = []
    for row_index, values in enumerate(rows[2:], 3):
        if not any(value.strip() for value in values):
            continue
        if len(values) != len(headers):
            raise UserError(f"legacy roofline row {row_index} has {len(values)} fields; expected {len(headers)}")
        fixed = dict(zip(headers[:9], values[:9]))
        try:
            threads = int(fixed["Threads"])
            loads = int(fixed["Loads"])
            stores = int(fixed["Stores"])
            bandwidths = {level: float(values[index]) for level, index in bandwidth_indexes.items()}
            performance = float(values[fp_index])
            fma_performance = float(values[fma_index])
        except (KeyError, ValueError) as error:
            raise UserError(f"invalid legacy roofline value in {path} row {row_index}: {error}") from error
        if not all(math.isfinite(value) for value in (*bandwidths.values(), performance, fma_performance)):
            raise UserError(f"non-finite legacy roofline metric in {path} row {row_index}")
        common: dict[str, Any] = {
            "isa": _normalise_isa(fixed["ISA"]),
            "data_type": _normalise_data_type(fixed["Precision"]),
            "num_threads": threads,
            "timestamp": _legacy_timestamp(fixed["Date"]),
            "machine": machine_name,
        }
        identity = {
            "kind": "roofline",
            "source_name": source_name,
            "row_index": row_index - 2,
            "metadata": metadata,
            "headers": headers,
            "row": values,
        }
        for level in CACHE_LEVELS:
            records.append(
                {
                    "type": "memory",
                    "name": f"legacy_{level.lower()}_{loads}ld_{stores}st",
                    **common,
                    "load_store_ratio": f"{loads}:{stores}",
                    "num_loads": loads,
                    "num_stores": stores,
                    "cache_level": level,
                    "memory_level_name": level,
                    "bandwidth_gbps": bandwidths[level],
                    "legacy_source": identity,
                }
            )
        operation = fixed["FP Inst."].strip().lower()
        for op, rate in ((operation, performance), ("fma", fma_performance)):
            records.append(
                {
                    "type": "arithmetic",
                    "name": f"legacy_{op}",
                    **common,
                    "operation": op,
                    "performance_gops": rate,
                    "legacy_source": identity,
                }
            )
    return records


def _json_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise UserError(f"invalid JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(record, dict):
                raise UserError(f"JSONL record at {path}:{line_number} must be an object")
            records.append(record)
    return records


def _legacy_isa(value: object) -> str:
    if value is None:
        return ""
    return {
        "x86": "scalar",
        "x86_sse": "sse",
        "x86_avx": "avx",
        "x86_avx2": "avx2",
        "x86_avx512": "avx512",
    }.get(str(value), str(value))


def _legacy_precision(value: object) -> str:
    if value is None:
        return ""
    return {"f32": "sp", "f64": "dp"}.get(str(value), str(value))


def _has_data(value: object) -> bool:
    return value not in (None, "", [], {})


def _app_to_rows(record: dict[str, Any], allow_lossy: bool) -> tuple[list[str], list[list[str]], list[str]]:
    provenance = record.get("legacy_source")
    if isinstance(provenance, dict) and provenance.get("kind") == "applications":
        headers, row = provenance.get("headers"), provenance.get("row")
        if isinstance(headers, list) and isinstance(row, list) and len(headers) == len(row):
            return [str(x) for x in headers], [[str(x) for x in row]], []
    has_unusable_provenance = "legacy_source" in record
    metadata = record.get("metadata")
    points = record.get("points")
    if not isinstance(metadata, dict) or not isinstance(points, list) or not points:
        raise UserError("native application JSONL requires metadata and at least one point")
    if any(not isinstance(point, dict) for point in points):
        raise UserError("application points must be JSON objects")
    losses = [
        key
        for key in ("command", "notes", "num_ranks", "threads_per_rank", "total_threads")
        if _has_data(metadata.get(key))
    ]
    if has_unusable_provenance:
        losses.append("legacy_source")
    if "aggregation" in record:
        losses.append("aggregation")
    if _has_data(record.get("optional_metrics")):
        losses.append("optional_metrics")
    if len(points) > 1:
        losses.append("multiple points")
    metadata_fields = {
        "name",
        "date",
        "method",
        "isa",
        "precision",
        "threads_per_rank",
        "command",
        "notes",
        "num_ranks",
        "total_threads",
    }
    losses.extend(
        f"metadata.{field}" for field, value in metadata.items() if field not in metadata_fields and _has_data(value)
    )
    record_fields = {"format_version", "aggregation", "optional_metrics", "metadata", "points", "legacy_source"}
    losses.extend(field for field, value in record.items() if field not in record_fields and _has_data(value))
    point_fields = {
        "label",
        "total_flops",
        "total_bytes",
        "runtime_s",
        "num_ranks",
        "num_threads",
        "num_regions",
        "arithmetic_intensity",
        "flops_per_second",
        "bandwidth",
        "optional_bytes",
        "optional_fractions",
        "time_s",
    }
    for point in points:
        losses.extend(
            f"point.{field}" for field, value in point.items() if field not in point_fields and _has_data(value)
        )
        losses.extend(
            f"point.{field}"
            for field in (
                "optional_bytes",
                "optional_fractions",
                "num_ranks",
                "num_regions",
                "total_flops",
                "total_bytes",
                "time_s",
            )
            if _has_data(point.get(field))
        )
        if _has_data(point.get("label")) and point.get("label") != metadata.get("name"):
            losses.append("point.label")
    for field in ("method", "isa", "precision"):
        if not metadata.get(field):
            losses.append(f"unknown {field}")
    if losses and not allow_lossy:
        raise UserError(
            f"application record has fields not representable in legacy CSV ({', '.join(sorted(set(losses)))}); "
            "use --allow-lossy"
        )
    name = metadata.get("name")
    date = metadata.get("date")
    if not isinstance(name, str) or not name or not isinstance(date, str) or not date:
        raise UserError("application record lacks a required name or date")
    rows: list[list[str]] = []
    for point in points:
        values = (
            point.get("arithmetic_intensity"),
            point.get("flops_per_second"),
            point.get("bandwidth"),
            point.get("runtime_s"),
        )
        if any(isinstance(value, bool) for value in values):
            raise UserError("application point metrics must be numeric")
        try:
            ai, flops_per_second, bandwidth, runtime = (float(value) for value in values)
        except (TypeError, ValueError) as error:
            raise UserError("application point lacks AI, performance, bandwidth, or runtime") from error
        threads = point.get("num_threads")
        if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
            raise UserError("application point lacks a valid positive thread count")
        if not all(math.isfinite(value) for value in (ai, flops_per_second, bandwidth, runtime)):
            raise UserError("application point contains a non-finite metric")
        values_out = [
            date,
            str(metadata.get("method") or ""),
            name,
            _legacy_isa(metadata.get("isa")),
            _legacy_precision(metadata.get("precision")),
            str(threads),
            str(ai),
            str(flops_per_second / 1e9),
            str(bandwidth / 1e9),
            str(runtime),
        ]
        rows.append(values_out)
    return list(APPLICATION_HEADERS), rows, losses


def _bench_to_rows(records: list[dict[str, Any]], allow_lossy: bool) -> tuple[list[list[str]], list[str]]:
    provenance_rows: dict[tuple[str, int], dict[str, Any]] = {}
    native: list[dict[str, Any]] = []
    for record in records:
        source = record.get("legacy_source")
        if "legacy_source" in record:
            if not isinstance(source, dict) or source.get("kind") != "roofline":
                raise UserError("unsupported benchmark legacy_source provenance")
            source_name = source.get("source_name")
            row_index = source.get("row_index")
            if not isinstance(source_name, str) or isinstance(row_index, bool) or not isinstance(row_index, int):
                raise UserError("legacy roofline provenance is incomplete")
            source_key = (source_name, row_index)
            existing = provenance_rows.get(source_key)
            if existing is not None and existing != source:
                raise UserError(f"conflicting legacy source data for {source_name} row {row_index}")
            provenance_rows[source_key] = source
            continue
        native.append(record)
    if provenance_rows and native:
        raise UserError("cannot mix provenance-backed and native benchmark records in one conversion")
    if provenance_rows:
        first = next(iter(provenance_rows.values()))
        metadata, headers = first.get("metadata"), first.get("headers")
        if not isinstance(metadata, list) or not isinstance(headers, list):
            raise UserError("legacy roofline provenance is incomplete")
        source_name = first.get("source_name")
        rows = [list(map(str, metadata)), list(map(str, headers))]
        for (current_name, row_index), source in sorted(
            provenance_rows.items(), key=lambda item: (item[0][0], item[0][1])
        ):
            if current_name != source_name or source.get("metadata") != metadata or source.get("headers") != headers:
                raise UserError("legacy roofline provenance contains inconsistent source headers or metadata")
            row = source.get("row")
            if not isinstance(row, list) or len(row) != len(headers):
                raise UserError(f"legacy roofline provenance row {row_index} is incomplete")
            rows.append(list(map(str, row)))
        return rows, []
    if not native:
        raise UserError("no benchmark records found")
    keys: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in native:
        if record.get("type") not in {"memory", "arithmetic"}:
            raise UserError("mixed records cannot be exported to legacy roofline CSV")
        affinity = record.get("thread_affinity")
        benchmark_key = (
            record.get("machine"),
            record.get("timestamp"),
            record.get("isa"),
            record.get("data_type"),
            record.get("num_threads"),
            record.get("frequency_hz"),
            record.get("actual_frequency_hz"),
            record.get("frequency_overridden"),
            tuple(affinity) if isinstance(affinity, list) else affinity,
        )
        keys[benchmark_key].append(record)
    rows = []
    losses: list[str] = []
    for benchmark_key, group in keys.items():
        mem = [r for r in group if r.get("type") == "memory"]
        arith = [r for r in group if r.get("type") == "arithmetic"]
        if len(mem) != 4 or {r.get("cache_level") for r in mem} != set(CACHE_LEVELS) or len(arith) != 2:
            raise UserError(
                f"incomplete or ambiguous benchmark set for {benchmark_key}; requires L1/L2/L3/DRAM "
                "and two arithmetic records"
            )
        required = {
            "machine": benchmark_key[0],
            "timestamp": benchmark_key[1],
            "isa": benchmark_key[2],
            "data_type": benchmark_key[3],
        }
        if any(not isinstance(value, str) or not value for value in required.values()):
            raise UserError(f"benchmark identity is incomplete for {benchmark_key}")
        if isinstance(benchmark_key[4], bool) or not isinstance(benchmark_key[4], int) or benchmark_key[4] <= 0:
            raise UserError(f"benchmark thread count is missing or invalid for {benchmark_key}")
        load_store = {(r.get("num_loads"), r.get("num_stores"), r.get("load_store_ratio")) for r in mem}
        if len(load_store) != 1:
            raise UserError(f"memory records do not share one load/store configuration for {benchmark_key}")
        if len({r.get("layout_mode") for r in mem}) > 1:
            raise UserError(f"memory records do not share one layout configuration for {benchmark_key}")
        operations = {str(r.get("operation")): r for r in arith}
        if len(operations) != 2 or "fma" not in operations or "None" in operations:
            raise UserError(f"ambiguous arithmetic operations for benchmark set {benchmark_key}")
        arithmetic_sizes = [record.get("num_ops") for record in arith]
        if any(size != arithmetic_sizes[0] for size in arithmetic_sizes[1:]):
            raise UserError(f"arithmetic records do not share one operation count for {benchmark_key}")
        if not allow_lossy:
            raise UserError(
                "native benchmark records cannot preserve DRAM Bytes, I/Cycle, cache capacities, timing/frequency, "
                "and extra fields; use --allow-lossy"
            )
        losses.extend(["DRAM Bytes", "I/Cycle", "cache capacities", "timing/frequency"])
        supported_fields = {
            "type",
            "machine",
            "timestamp",
            "isa",
            "data_type",
            "num_threads",
            "num_loads",
            "num_stores",
            "load_store_ratio",
            "cache_level",
            "bandwidth_gbps",
            "layout_mode",
            "operation",
            "performance_gops",
        }
        for record in group:
            losses.extend(
                field
                for field, value in record.items()
                if field not in supported_fields and field != "legacy_source" and value not in (None, "", [], {})
            )
        num_loads, num_stores, _ratio = next(iter(load_store))
        if (
            isinstance(num_loads, bool)
            or not isinstance(num_loads, int)
            or num_loads < 0
            or isinstance(num_stores, bool)
            or not isinstance(num_stores, int)
            or num_stores < 0
        ):
            raise UserError(f"missing or invalid load/store counts for benchmark set {benchmark_key}")
        rates = {str(r["cache_level"]): r.get("bandwidth_gbps") for r in mem}
        layout_mode = mem[0].get("layout_mode")
        interleaved = {"interleaved": "Yes", "sequential": "No"}.get(str(layout_mode), "")
        row = [
            str(benchmark_key[1]),
            _legacy_isa(benchmark_key[2]),
            _legacy_precision(benchmark_key[3]),
            str(benchmark_key[4]),
            str(num_loads),
            str(num_stores),
            interleaved,
            "",
            next(op for op in operations if op != "fma"),
        ]
        if not interleaved:
            losses.append("Interleaved")
        for level in CACHE_LEVELS:
            value = rates[level]
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise UserError(f"missing or invalid {level} bandwidth in benchmark set {benchmark_key}")
            row.extend([str(value), ""])
        for op in (next(op for op in operations if op != "fma"), "fma"):
            rate = operations[op].get("performance_gops")
            if not isinstance(rate, (int, float)) or not math.isfinite(float(rate)):
                raise UserError(f"missing or invalid {op} performance in benchmark set {benchmark_key}")
            row.extend([str(rate), ""])
        rows.append(row)
    machines = {str(key[0]) for key in keys}
    if len(machines) != 1:
        raise UserError("cannot combine native benchmark records from multiple machines")
    machine = str(next(iter(keys))[0] or "Unknown")
    metadata = [
        "Name:",
        machine,
        "L1 Size:",
        "",
        "L2 Size:",
        "",
        "L3 Size:",
        "",
        "",
        "L1",
        "L1",
        "L2",
        "L2",
        "L3",
        "L3",
        "DRAM",
        "DRAM",
        "FP",
        "FP",
        "FP FMA",
        "FP_FMA",
    ]
    headers = list(ROOFLINE_HEADERS)
    return [metadata, headers, *rows], sorted(set(losses))


def _legacy_files(path: Path) -> list[Path]:
    candidates = _input_files(path, (".csv",))
    result: list[Path] = []
    unsupported: list[str] = []
    for candidate in candidates:
        try:
            _read_legacy(candidate)
        except UserError:
            unsupported.append(str(candidate))
        else:
            result.append(candidate)
    if unsupported:
        raise UserError(f"unrecognized CSV input file(s): {', '.join(unsupported)}")
    if not result:
        raise UserError(f"no supported legacy CSV files found under {path}")
    return result


def _dataset_name(path: Path, kind: str) -> str:
    name = path.name
    suffix = APPLICATION_SUFFIX if kind == "applications" else ROOFLINE_SUFFIX
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    return _slug(name)


def _write_csv(path: Path, rows: list[list[str]], dry_run: bool) -> None:
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as output:
            csv.writer(output).writerows(rows)


def _provenance_source_name(records: Sequence[dict[str, Any]], kind: str) -> str | None:
    names = {
        source["source_name"]
        for record in records
        if isinstance((source := record.get("legacy_source")), dict)
        and source.get("kind") == kind
        and isinstance(source.get("source_name"), str)
    }
    if len(names) > 1:
        raise UserError("one JSONL file contains provenance from multiple legacy CSV source files")
    return next(iter(names), None)


def convert(options: ConversionOptions) -> list[str]:
    if options.from_format == options.to_format:
        raise UserError("--from and --to must specify different formats")
    if not options.input_path.exists():
        raise UserError(f"input path does not exist: {options.input_path}")
    if options.from_format == "legacy-csv":
        sources = _legacy_files(options.input_path)
        pending: dict[Path, list[dict[str, Any]]] = defaultdict(list)
        pending_sources: dict[Path, Path] = {}
        for source in sources:
            kind, app_rows, headers, _metadata, _rows = _read_legacy(source)
            dataset = _dataset_name(source, kind)
            filename = "applications.jsonl" if kind == "applications" else "benchmarks.jsonl"
            target = options.output_dir / dataset / filename
            if target in pending_sources and pending_sources[target] != source:
                raise UserError(f"multiple legacy CSV files map to {target}; refusing to merge distinct datasets")
            pending_sources[target] = source
            if kind == "applications":
                pending[target].extend(
                    _application_record(row, headers, index, source.name) for index, row in enumerate(app_rows, 2)
                )
            else:
                pending[target].extend(_roofline_records(source, source.name, dataset))
        empty = [str(target) for target, records in pending.items() if not records]
        if empty:
            raise UserError(f"legacy CSV contains no data rows: {', '.join(empty)}")
        existing = [str(path) for path in pending if path.exists()]
        if existing and not options.overwrite:
            raise UserError(f"output already exists: {', '.join(existing)} (use --overwrite)")
        serialized = {
            target: [json.dumps(record, sort_keys=True, allow_nan=False) for record in records]
            for target, records in pending.items()
        }
        forward_summary: list[str] = []
        for target, lines in serialized.items():
            if not options.dry_run:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
            forward_summary.append(f"{target}: {len(lines)} record(s)")
        return forward_summary

    sources = _input_files(options.input_path, (".jsonl",))
    if not sources:
        raise UserError(f"no JSONL files found under {options.input_path}")
    by_target: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    output_types: dict[Path, str] = {}
    losses: list[str] = []
    input_by_target: dict[Path, Path] = {}
    for source in sources:
        records = _json_records(source)
        if not records:
            continue
        if all("format_version" in record for record in records):
            csv_headers: list[str] | None = None
            converted_app_rows: list[list[str]] = []
            source_losses: list[str] = []
            for record in records:
                row_headers, record_rows, record_losses = _app_to_rows(record, options.allow_lossy)
                if csv_headers is None:
                    csv_headers = row_headers
                if row_headers != csv_headers:
                    raise UserError(f"inconsistent application CSV columns in {source}")
                converted_app_rows.extend(record_rows)
                source_losses.extend(record_losses)
            losses.extend(source_losses)
            original_name = _provenance_source_name(records, "applications")
            fallback = f"{_slug(source.parent.name)}{APPLICATION_SUFFIX}"
            target = options.output_dir / (Path(original_name).name if original_name else fallback)
            if target in input_by_target and input_by_target[target] != source:
                raise UserError(f"multiple JSONL files map to {target}; refusing to merge distinct datasets")
            input_by_target[target] = source
            by_target[target].append({"rows": converted_app_rows, "headers": csv_headers})
            output_types[target] = "applications"
        else:
            benchmark_rows, source_losses = _bench_to_rows(records, options.allow_lossy)
            losses.extend(source_losses)
            original_name = _provenance_source_name(records, "roofline")
            fallback = f"{_slug(source.parent.name)}{ROOFLINE_SUFFIX}"
            target = options.output_dir / (Path(original_name).name if original_name else fallback)
            if target in input_by_target and input_by_target[target] != source:
                raise UserError(f"multiple JSONL files map to {target}; refusing to merge distinct datasets")
            input_by_target[target] = source
            by_target[target].append({"rows": benchmark_rows})
            output_types[target] = "benchmarks"
    if not by_target:
        raise UserError("JSONL input contains no records")
    existing = [str(path) for path in by_target if path.exists()]
    if existing and not options.overwrite:
        raise UserError(f"output already exists: {', '.join(existing)} (use --overwrite)")
    prepared: dict[Path, list[list[str]]] = {}
    summary: list[str] = []
    for target, chunks in by_target.items():
        output_rows: list[list[str]] = []
        kind = output_types[target]
        if kind == "applications":
            csv_output_headers = chunks[0]["headers"]
            output_rows.append(csv_output_headers)
            for chunk in chunks:
                if chunk["headers"] != csv_output_headers:
                    raise UserError(f"inconsistent application CSV columns for {target}")
                output_rows.extend(chunk["rows"])
        else:
            first_rows = chunks[0]["rows"]
            output_rows.extend(first_rows)
            for chunk in chunks[1:]:
                chunk_rows = chunk["rows"]
                if chunk_rows[:2] != first_rows[:2]:
                    raise UserError(f"inconsistent roofline CSV headers for {target}")
                output_rows.extend(chunk_rows[2:])
        prepared[target] = output_rows
        summary.append(f"{target}: {len(output_rows) - (2 if kind == 'benchmarks' else 1)} row(s)")
    if losses:
        summary.append("Lossy fields: " + ", ".join(sorted(set(losses))))
    for target, output_rows in prepared.items():
        _write_csv(target, output_rows, options.dry_run)
    return summary
