from __future__ import annotations

import csv
import json
from pathlib import Path

from typing import Literal

import pytest

from carm_roofline.conversion import ConversionOptions, convert
from carm_roofline.core import UserError

pytestmark = pytest.mark.unit


def _app_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("Date", "Method", "Name", "ISA", "Precision", "Threads", "AI", "Gflops", "Bandwidth", "Time"))
        writer.writerow(("2026-02-21 08:27:47", "PMU", "gemm", "", "dp", 1, 0.5, 10.0, 20000.0, 2.0))


def _roof_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = ["Name:", "MN5", "L1 Size:", "32", "L2 Size:", "2048", "L3 Size:", "107520", "", "L1", "L1", "L2", "L2", "L3", "L3", "DRAM", "DRAM", "FP", "FP", "FP FMA", "FP_FMA"]
    headers = ["Date", "ISA", "Precision", "Threads", "Loads", "Stores", "Interleaved", "DRAM Bytes", "FP Inst.", "GB/s", "I/Cycle", "GB/s", "I/Cycle", "GB/s", "I/Cycle", "GB/s", "I/Cycle", "Gflop/s", "I/Cycle", "Gflop/s", "I/Cycle"]
    row = ["2026-02-21 08:27:47", "avx2", "dp", "4", "2", "1", "No", "800", "add", "100", "", "80", "", "60", "", "40", "", "20", "", "30", ""]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerows((metadata, headers, row))


def _run(
    source: Path, output: Path, source_format: Literal["legacy-csv", "jsonl"],
    target_format: Literal["legacy-csv", "jsonl"], **kwargs: bool,
) -> list[str]:
    return convert(ConversionOptions(source, source_format, target_format, output, **kwargs))


def test_legacy_application_conversion_preserves_unknowns_and_derives_bandwidth(tmp_path: Path) -> None:
    source = tmp_path / "legacy" / "apps" / "Machine_applications.csv"
    _app_csv(source)
    output = tmp_path / "json"

    _run(source, output, "legacy-csv", "jsonl")

    result = output / "Machine" / "applications.jsonl"
    record = json.loads(result.read_text(encoding="utf-8"))
    point = record["points"][0]
    assert record["metadata"]["date"] == "2026-02-21T08:27:47"
    assert record["metadata"]["isa"] is None
    assert record["metadata"]["num_ranks"] is None
    assert record["metadata"]["threads_per_rank"] is None
    assert record["metadata"]["total_threads"] == 1
    assert point["total_flops"] is None
    assert point["total_bytes"] is None
    assert point["num_ranks"] is None
    assert point["num_regions"] is None
    assert point["runtime_s"] == 2.0
    assert point["num_threads"] == 1
    assert point["flops_per_second"] == 10e9
    assert point["arithmetic_intensity"] == 0.5
    assert point["bandwidth"] == 20e9
    assert record["legacy_source"]["row"][-2:] == ["20000.0", "2.0"]


def test_legacy_roofline_conversion_does_not_fabricate_capacity_or_working_set(tmp_path: Path) -> None:
    source = tmp_path / "legacy" / "roofline" / "Machine_roofline.csv"
    _roof_csv(source)
    output = tmp_path / "json"

    _run(source, output, "legacy-csv", "jsonl")

    records = [json.loads(line) for line in (output / "Machine" / "benchmarks.jsonl").read_text().splitlines()]
    assert len(records) == 6
    assert all(record["timestamp"] == "2026-02-21T08:27:47" for record in records)
    memory = [record for record in records if record["type"] == "memory"]
    assert {record["cache_level"] for record in memory} == {"L1", "L2", "L3", "DRAM"}
    assert all("working_set_bytes" not in record and "size_per_thread_bytes" not in record for record in memory)
    assert all(record["machine"] == "Machine" for record in records)
    assert all("legacy_source" in record for record in records)


def test_provenance_round_trip_restores_application_csv_values(tmp_path: Path) -> None:
    source = tmp_path / "legacy" / "apps" / "Machine_applications.csv"
    _app_csv(source)
    json_dir = tmp_path / "json"
    _run(source, json_dir, "legacy-csv", "jsonl")
    csv_dir = tmp_path / "restored"

    _run(json_dir, csv_dir, "jsonl", "legacy-csv")

    with source.open(newline="", encoding="utf-8") as original:
        expected = list(csv.reader(original))
    restored = csv_dir / "Machine_applications.csv"
    with restored.open(newline="", encoding="utf-8") as actual:
        assert list(csv.reader(actual)) == expected


def test_provenance_round_trip_restores_roofline_csv_values(tmp_path: Path) -> None:
    source = tmp_path / "legacy" / "roofline" / "Machine_roofline.csv"
    _roof_csv(source)
    json_dir = tmp_path / "json"
    _run(source, json_dir, "legacy-csv", "jsonl")
    csv_dir = tmp_path / "restored"

    _run(json_dir, csv_dir, "jsonl", "legacy-csv")

    with source.open(newline="", encoding="utf-8") as original:
        expected = list(csv.reader(original))
    restored = csv_dir / "Machine_roofline.csv"
    with restored.open(newline="", encoding="utf-8") as actual:
        assert list(csv.reader(actual)) == expected


def test_dry_run_does_not_write_and_existing_output_requires_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "Machine_applications.csv"
    _app_csv(source)
    output = tmp_path / "out"

    _run(source, output, "legacy-csv", "jsonl", dry_run=True)
    assert not output.exists()
    _run(source, output, "legacy-csv", "jsonl")
    with pytest.raises(UserError):
        _run(source, output, "legacy-csv", "jsonl")
    _run(source, output, "legacy-csv", "jsonl", overwrite=True)


def test_native_application_reverse_requires_lossy_opt_in(tmp_path: Path) -> None:
    source = tmp_path / "native" / "applications.jsonl"
    source.parent.mkdir()
    record = {
        "format_version": "3.0",
        "aggregation": "global",
        "metadata": {
            "name": "x",
            "date": "d",
            "method": "PMU",
            "isa": "x86",
            "precision": "f64",
            "threads_per_rank": 1,
            "extra_metric": 42,
        },
        "points": [{
            "runtime_s": 2.0,
            "arithmetic_intensity": 0.5,
            "flops_per_second": 10e9,
            "bandwidth": 20e9,
            "num_threads": 1,
            "custom_counter": 7,
        }],
    }
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(UserError):
        _run(source, tmp_path / "strict", "jsonl", "legacy-csv")
    summary = _run(source, tmp_path / "lossy", "jsonl", "legacy-csv", allow_lossy=True)
    loss_report = next(item for item in summary if item.startswith("Lossy fields:"))
    assert "metadata.extra_metric" in loss_report
    assert "point.custom_counter" in loss_report
    with (tmp_path / "lossy" / "native_applications.csv").open(newline="", encoding="utf-8") as source_csv:
        row = list(csv.reader(source_csv))[1]
    assert row[3:5] == ["scalar", "dp"]
    assert row[7:9] == ["10.0", "20.0"]


def test_native_application_multiple_points_flatten_with_loss_report(tmp_path: Path) -> None:
    source = tmp_path / "native" / "applications.jsonl"
    source.parent.mkdir()
    record = {
        "format_version": "3.0",
        "metadata": {
            "name": "x",
            "date": "d",
            "method": "PMU",
            "isa": "x86",
            "precision": "f64",
            "threads_per_rank": 1,
        },
        "points": [
            {
                "label": "region-a",
                "runtime_s": 2.0,
                "arithmetic_intensity": 0.5,
                "flops_per_second": 10e9,
                "bandwidth": 20e9,
                "num_threads": 1,
            },
            {
                "label": "region-b",
                "runtime_s": 1.0,
                "arithmetic_intensity": 1.0,
                "flops_per_second": 12e9,
                "bandwidth": 12e9,
                "num_threads": 1,
            },
        ],
    }
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")

    summary = _run(source, tmp_path / "out", "jsonl", "legacy-csv", allow_lossy=True)

    with (tmp_path / "out" / "native_applications.csv").open(newline="", encoding="utf-8") as output:
        rows = list(csv.reader(output))
    assert len(rows) == 3
    assert rows[1][7:9] == ["10.0", "20.0"]
    assert rows[2][7:9] == ["12.0", "12.0"]
    assert "multiple points" in next(item for item in summary if item.startswith("Lossy fields:"))
    assert "point.label" in next(item for item in summary if item.startswith("Lossy fields:"))


def test_native_benchmark_reverse_requires_complete_groups_and_lossy_opt_in(tmp_path: Path) -> None:
    source = tmp_path / "native" / "benchmarks.jsonl"
    source.parent.mkdir()
    records = []
    common = {"machine": "m", "timestamp": "t", "isa": "x86", "data_type": "f64", "num_threads": 1}
    for level in ("L1", "L2", "L3", "DRAM"):
        records.append({"type": "memory", **common, "cache_level": level, "bandwidth_gbps": 1.0,
                        "num_loads": 1, "num_stores": 0, "layout_mode": "sequential"})
    records.extend([
        {"type": "arithmetic", **common, "operation": "add", "performance_gops": 2.0},
        {"type": "arithmetic", **common, "operation": "fma", "performance_gops": 3.0},
    ])
    source.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    with pytest.raises(UserError):
        _run(source, tmp_path / "strict", "jsonl", "legacy-csv")
    _run(source, tmp_path / "lossy", "jsonl", "legacy-csv", allow_lossy=True)
    output_row = list(csv.reader((tmp_path / "lossy" / "native_roofline.csv").open()))[2]
    assert output_row[6] == "No"
    assert output_row[8] == "add"
    assert output_row[7] == ""
    assert all(output_row[index] == "" for index in (10, 12, 14, 16, 18, 20))


def test_missing_jsonl_input_is_reported(tmp_path: Path) -> None:
    with pytest.raises(UserError):
        _run(tmp_path / "missing", tmp_path / "out", "legacy-csv", "jsonl")
