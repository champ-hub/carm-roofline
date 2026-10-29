"""Parse DynamoRIO count and native timing JSONL into profiling metrics."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from carm_roofline.core import UserError

from .dynamorio_metrics import FLOP_EVENTS
from .model import RankMetrics, RegionMetrics, ThreadMetrics

_IDENTITY_FIELDS = ("process_index", "process_id", "thread_index", "thread_id", "name")
_INSTRUCTION_COUNTERS = frozenset(event.removeprefix("dr_") for event in FLOP_EVENTS)


@dataclass(frozen=True, order=True)
class RegionIdentity:
    """Stable process, thread, and region identity used to join profiler passes."""

    pidx: int
    pid: int
    tidx: int
    tid: int
    name: str


@dataclass(frozen=True)
class JsonRow:
    """One validated region record and its source line."""

    line_number: int
    values: dict[str, object]


@dataclass(frozen=True)
class CountRecord:
    """Instruction and memory-reference counters for one region."""

    counters: dict[str, int]


@dataclass(frozen=True)
class TimingRecord:
    """Native elapsed time for one region."""

    elapsed_ns: int


class IdentityRegistry:
    """Ensure local process/thread indexes map bijectively to stable IDs."""

    def __init__(self) -> None:
        self._process_id_by_index: dict[int, int] = {}
        self._process_index_by_id: dict[int, int] = {}
        self._thread_id_by_index: dict[tuple[int, int], int] = {}
        self._thread_index_by_id: dict[tuple[int, int], int] = {}

    def register(self, identity: RegionIdentity, path: Path, line_number: int) -> None:
        if (
            self._process_id_by_index.get(identity.pidx, identity.pid) != identity.pid
            or self._process_index_by_id.get(identity.pid, identity.pidx) != identity.pidx
        ):
            raise UserError(f"DynamoRIO output {path}:{line_number} maps a process index/ID inconsistently")
        self._process_id_by_index[identity.pidx] = identity.pid
        self._process_index_by_id[identity.pid] = identity.pidx

        thread_index_key = (identity.pid, identity.tidx)
        thread_id_key = (identity.pid, identity.tid)
        if (
            self._thread_id_by_index.get(thread_index_key, identity.tid) != identity.tid
            or self._thread_index_by_id.get(thread_id_key, identity.tidx) != identity.tidx
        ):
            raise UserError(f"DynamoRIO output {path}:{line_number} maps a thread index/ID inconsistently")
        self._thread_id_by_index[thread_index_key] = identity.tid
        self._thread_index_by_id[thread_id_key] = identity.tidx


def _as_json_object(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            return None
        result[key] = item
    return result


def _read_rows(path: Path, record_kind: Literal["count", "timing"]) -> Iterator[JsonRow]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise UserError(f"DynamoRIO {record_kind} output not found: {path}: {exc}") from exc

    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            payload: object = json.loads(line)
        except json.JSONDecodeError as exc:
            raise UserError(f"Invalid JSON in DynamoRIO output {path}:{line_number}: {exc}") from exc
        row = _as_json_object(payload)
        if row is None or row.get("type") != "region":
            raise UserError(f"Unexpected record in DynamoRIO output {path}:{line_number}; expected a region record")
        if row.get("thread_id_stable") is not True:
            raise UserError(
                f"DynamoRIO region {row.get('name', '<unknown>')!r} in {path} has no stable application thread ID; "
                "cannot safely match native timing to instruction counts. Set deterministic process/thread identity "
                "before region markers."
            )
        yield JsonRow(line_number, row)


def _coerce_identity_integer(value: object) -> int:
    if isinstance(value, (int, float, str)):
        return int(value)
    raise TypeError


def _parse_identity(row: dict[str, object], path: Path, line_number: int) -> RegionIdentity:
    if any(field not in row for field in _IDENTITY_FIELDS):
        raise UserError(f"DynamoRIO output {path}:{line_number} lacks required identity fields {_IDENTITY_FIELDS}")
    identity_values = (row["process_index"], row["process_id"], row["thread_index"], row["thread_id"])
    if any(isinstance(value, bool) for value in identity_values):
        raise UserError(f"DynamoRIO output {path}:{line_number} has non-integer process/thread identity")
    try:
        process_index, process_id, thread_index, thread_id = (
            _coerce_identity_integer(value) for value in identity_values
        )
    except (TypeError, ValueError) as exc:
        raise UserError(f"DynamoRIO output {path}:{line_number} has non-integer process/thread identity") from exc
    name = row["name"]
    if min(process_index, process_id, thread_index, thread_id) < 0 or not isinstance(name, str) or not name:
        raise UserError(f"DynamoRIO output {path}:{line_number} has invalid region identity")
    return RegionIdentity(process_index, process_id, thread_index, thread_id, name)


def _parse_count_record(row: dict[str, object], path: Path, line_number: int) -> CountRecord:
    if "elapsed_ns" not in row or row["elapsed_ns"] is not None:
        raise UserError(
            f"DynamoRIO count output {path}:{line_number} must have elapsed_ns:null; "
            "region timing comes only from the native pass."
        )
    instruction_counts = _as_json_object(row.get("instruction_counts"))
    if instruction_counts is None:
        raise UserError(f"DynamoRIO count output {path}:{line_number} lacks instruction_counts")
    unknown = set(instruction_counts) - _INSTRUCTION_COUNTERS
    if unknown:
        raise UserError(
            f"DynamoRIO count output {path}:{line_number} has unknown instruction counters: {sorted(unknown)}"
        )
    instruction_counts_by_name: dict[str, int] = {}
    for event, value in instruction_counts.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise UserError(f"DynamoRIO count output {path}:{line_number} has invalid instruction counts")
        instruction_counts_by_name[event] = value

    read_bytes, write_bytes = row.get("read_bytes"), row.get("write_bytes")
    if (
        isinstance(read_bytes, bool)
        or not isinstance(read_bytes, int)
        or read_bytes < 0
        or isinstance(write_bytes, bool)
        or not isinstance(write_bytes, int)
        or write_bytes < 0
    ):
        raise UserError(f"DynamoRIO count output {path}:{line_number} has invalid memory-byte totals")
    counters = {event: instruction_counts_by_name.get(event.removeprefix("dr_"), 0) for event in FLOP_EVENTS}
    counters["dr_read_bytes"] = read_bytes
    counters["dr_write_bytes"] = write_bytes
    return CountRecord(counters)


def _parse_timing_record(row: dict[str, object], path: Path, line_number: int) -> TimingRecord:
    elapsed_ns = row.get("elapsed_ns")
    if isinstance(elapsed_ns, bool) or not isinstance(elapsed_ns, int) or elapsed_ns < 0:
        raise UserError(f"DynamoRIO timing output {path}:{line_number} has invalid elapsed_ns")
    return TimingRecord(elapsed_ns)


def _read_count_records(path: Path) -> dict[RegionIdentity, CountRecord]:
    records: dict[RegionIdentity, CountRecord] = {}
    registry = IdentityRegistry()
    for row in _read_rows(path, "count"):
        identity = _parse_identity(row.values, path, row.line_number)
        registry.register(identity, path, row.line_number)
        record = _parse_count_record(row.values, path, row.line_number)
        previous = records.get(identity)
        if previous is not None:
            record = CountRecord({event: value + previous.counters[event] for event, value in record.counters.items()})
        records[identity] = record
    return records


def _read_timing_records(path: Path) -> dict[RegionIdentity, TimingRecord]:
    records: dict[RegionIdentity, TimingRecord] = {}
    registry = IdentityRegistry()
    for row in _read_rows(path, "timing"):
        identity = _parse_identity(row.values, path, row.line_number)
        registry.register(identity, path, row.line_number)
        record = _parse_timing_record(row.values, path, row.line_number)
        previous = records.get(identity)
        if previous is not None:
            record = TimingRecord(record.elapsed_ns + previous.elapsed_ns)
        records[identity] = record
    return records


def load_dynamorio_results(count_path: Path, timing_path: Path) -> list[RankMetrics]:
    """Load raw instruction categories and merge native timings on stable IDs."""
    counts = _read_count_records(count_path)
    timings = _read_timing_records(timing_path)
    if not counts:
        raise UserError(
            f"DynamoRIO produced no region counts in {count_path}; ensure the application uses CARM markers."
        )
    if counts.keys() != timings.keys():
        missing = sorted(counts.keys() - timings.keys())
        extra = sorted(timings.keys() - counts.keys())
        raise UserError(
            "DynamoRIO count and native timing region identities do not match "
            f"(missing timings: {missing[:5]}, unexpected timings: {extra[:5]}). "
            "Both runs must execute the same marked regions with stable application process/thread IDs."
        )

    by_rank: dict[int, dict[int, list[RegionMetrics]]] = defaultdict(lambda: defaultdict(list))
    for identity, count_record in counts.items():
        by_rank[identity.pidx][identity.tidx].append(
            RegionMetrics(
                name=identity.name,
                parent_region_id="-1",
                cycles=0,
                time_nsec=timings[identity].elapsed_ns,
                counters=count_record.counters,
            )
        )
    return [
        RankMetrics(
            rank_id=rank_id,
            event_definitions={"method": "DynamoRIO instruction counting", "counter_kind": "software"},
            threads=[
                ThreadMetrics(thread_id=thread_id, regions=regions) for thread_id, regions in sorted(threads.items())
            ],
        )
        for rank_id, threads in sorted(by_rank.items())
    ]
