"""Unit tests for the reusable roofline assembly module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from roofline_assembly import (
    RooflineFilter,
    assemble_roofline,
    assemble_roofline_from_file,
    discover_filter_options,
    discover_filter_options_for_selection,
    load_all_benchmarks,
    load_benchmarks,
)
from units import ArithmeticIntensity, Bandwidth, Performance

pytestmark = pytest.mark.unit

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def jsonl_fixture(tmp_path: Path) -> Path:
    """Create a temp JSONL file with two timestamps of benchmark data.

    Timestamp A (older): arithmetic + memory for test_isa.
    Timestamp B (newer): arithmetic only for test_isa (fma updated, add same).
    Extra ISA records (should be excluded by filter).
    """
    records = [
        # ── Timestamp A: arithmetic + memory for test_isa ──
        {
            "type": "arithmetic",
            "name": "test_isa_arith_fma_f32",
            "isa": "test_isa",
            "machine": "test_machine",
            "data_type": "f32",
            "num_threads": 1,
            "operation": "fma",
            "performance_gops": 100.0,
            "timestamp": "2026-01-01T00:00:00",
        },
        {
            "type": "arithmetic",
            "name": "test_isa_arith_add_f32",
            "isa": "test_isa",
            "machine": "test_machine",
            "data_type": "f32",
            "num_threads": 1,
            "operation": "add",
            "performance_gops": 50.0,
            "timestamp": "2026-01-01T00:00:00",
        },
        {
            "type": "memory",
            "name": "test_isa_mem_2ld_1st_l1",
            "isa": "test_isa",
            "machine": "test_machine",
            "data_type": "f32",
            "num_threads": 1,
            "load_store_ratio": "2:1",
            "cache_level": "L1",
            "bandwidth_gbps": 400.0,
            "timestamp": "2026-01-01T00:00:00",
        },
        {
            "type": "memory",
            "name": "test_isa_mem_2ld_1st_l2",
            "isa": "test_isa",
            "machine": "test_machine",
            "data_type": "f32",
            "num_threads": 1,
            "load_store_ratio": "2:1",
            "cache_level": "L2",
            "bandwidth_gbps": 100.0,
            "timestamp": "2026-01-01T00:00:00",
        },
        {
            "type": "memory",
            "name": "test_isa_mem_2ld_1st_dram",
            "isa": "test_isa",
            "machine": "test_machine",
            "data_type": "f32",
            "num_threads": 1,
            "load_store_ratio": "2:1",
            "cache_level": "DRAM",
            "bandwidth_gbps": 30.0,
            "timestamp": "2026-01-01T00:00:00",
        },
        # ── Timestamp B (newer): arithmetic only, fma updated ──
        {
            "type": "arithmetic",
            "name": "test_isa_arith_fma_f32_v2",
            "isa": "test_isa",
            "machine": "test_machine",
            "data_type": "f32",
            "num_threads": 1,
            "operation": "fma",
            "performance_gops": 120.0,  # Updated (higher = newer)
            "timestamp": "2026-01-02T00:00:00",
        },
        {
            "type": "arithmetic",
            "name": "test_isa_arith_add_f32_v2",
            "isa": "test_isa",
            "machine": "test_machine",
            "data_type": "f32",
            "num_threads": 1,
            "operation": "add",
            "performance_gops": 55.0,  # Updated (higher = newer)
            "timestamp": "2026-01-02T00:00:00",
        },
        # ── Different ISA (should not match default filter) ──
        {
            "type": "arithmetic",
            "name": "other_isa_arith_fma_f32",
            "isa": "other_isa",
            "machine": "test_machine",
            "data_type": "f32",
            "num_threads": 1,
            "operation": "fma",
            "performance_gops": 200.0,
            "timestamp": "2026-01-01T00:00:00",
        },
    ]

    path = tmp_path / "benchmarks.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True))
            f.write("\n")
    return path


# ── load_benchmarks ────────────────────────────────────────────────────────────


def test_load_benchmarks(jsonl_fixture: Path) -> None:
    records = load_benchmarks(jsonl_fixture)
    assert len(records) == 8


def test_load_benchmarks_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent.jsonl"
    with pytest.raises(FileNotFoundError):
        load_benchmarks(missing)


def test_load_benchmarks_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    records = load_benchmarks(path)
    assert records == []


def test_load_benchmarks_skips_malformed_lines(tmp_path: Path) -> None:
    """Malformed JSON lines are skipped with a warning instead of raising."""
    path = tmp_path / "benchmarks.jsonl"
    path.write_text('{"machine": "valid1"}\nnot valid json\n{"machine": "valid2"}\n')
    records = load_benchmarks(path)
    assert len(records) == 2
    assert records[0]["machine"] == "valid1"
    assert records[1]["machine"] == "valid2"


def test_load_all_benchmarks_skips_corrupt_file(tmp_path: Path) -> None:
    """A corrupt benchmarks.jsonl in one machine dir does not block others."""
    # Valid machine dir
    valid_dir = tmp_path / "machine_a"
    valid_dir.mkdir()
    (valid_dir / "benchmarks.jsonl").write_text('{"machine": "a", "isa": "x86"}\n')
    # Corrupt machine dir (malformed JSONL)
    corrupt_dir = tmp_path / "machine_b"
    corrupt_dir.mkdir()
    (corrupt_dir / "benchmarks.jsonl").write_text("not valid json\n")
    records = load_all_benchmarks(tmp_path)
    assert len(records) == 1
    assert records[0]["machine"] == "a"


def test_discover_filter_options(jsonl_fixture: Path) -> None:
    """discover_filter_options returns all filter dimensions including data_type."""
    records = load_benchmarks(jsonl_fixture)
    result = discover_filter_options(records)
    assert "data_type" in result
    assert "f32" in result["data_type"]
    assert "machine" in result
    assert "isa" in result
    assert "threads" in result
    assert "load_store_ratio" in result


def test_discover_filter_options_for_selection(jsonl_fixture: Path) -> None:
    """Cross-field filtering: selecting test_isa narrows threads to [1]."""
    records = load_benchmarks(jsonl_fixture)
    # Filter by test_isa -> only threads=1 records match
    result = discover_filter_options_for_selection(records, isa="test_isa")
    assert result["threads"] == [1]
    # ISA options are unconstrained (isa filter not applied to isa field itself)
    assert "test_isa" in result["isa"]
    # Filtering by isa="other_isa" does NOT constrain isa options
    # (only OTHER non-None fields constrain each field's options)
    result2 = discover_filter_options_for_selection(records, isa="other_isa")
    assert "test_isa" in result2["isa"]
    assert "other_isa" in result2["isa"]
    # None fields widen: no filter -> all values
    result3 = discover_filter_options_for_selection(records)
    assert "test_isa" in result3["isa"]
    assert "other_isa" in result3["isa"]


# ── assemble_roofline ──────────────────────────────────────────────────────────


def test_assemble_roofline_cross_run(jsonl_fixture: Path) -> None:
    """Memory from timestamp A + arithmetic from timestamp B (cross-run assembly)."""
    records = load_benchmarks(jsonl_fixture)

    flt = RooflineFilter(isa="test_isa", num_threads=1, data_type="f32")
    model = assemble_roofline(records, flt)

    # bandwidth_by_level should have L1 + L2 + DRAM (no L3 in fixture)
    assert "L1" in model.bandwidth_by_level
    assert "L2" in model.bandwidth_by_level
    assert "DRAM" in model.bandwidth_by_level
    assert "L3" not in model.bandwidth_by_level

    # All bandwidth values should be from timestamp A (the only memory data)
    bw_l1 = model.bandwidth_by_level["L1"]
    assert isinstance(bw_l1, Bandwidth)
    assert bw_l1.value == pytest.approx(400.0 * 1e9)

    bw_dram = model.bandwidth_by_level["DRAM"]
    assert bw_dram.value == pytest.approx(30.0 * 1e9)

    # peak_performance should have fma + add from timestamp B (newer)
    assert "fma" in model.peak_performance_by_op
    assert "add" in model.peak_performance_by_op

    perf_fma = model.peak_performance_by_op["fma"]
    assert isinstance(perf_fma, Performance)
    # Should pick the newer (higher) timestamp B value = 120 GOPS
    assert perf_fma.value == pytest.approx(120.0 * 1e9)

    perf_add = model.peak_performance_by_op["add"]
    assert perf_add.value == pytest.approx(55.0 * 1e9)

    # source_timestamps should contain both timestamps
    assert "2026-01-01T00:00:00" in model.source_timestamps
    assert "2026-01-02T00:00:00" in model.source_timestamps


def test_assemble_roofline_empty_filter(jsonl_fixture: Path) -> None:
    """Filtering for a non-existent ISA returns empty model, not an error."""
    records = load_benchmarks(jsonl_fixture)

    flt = RooflineFilter(isa="nonexistent_isa")
    model = assemble_roofline(records, flt)

    assert model.bandwidth_by_level == {}
    assert model.peak_performance_by_op == {}
    assert model.source_timestamps == frozenset()


def test_assemble_roofline_filter_operations(jsonl_fixture: Path) -> None:
    """Only matching operations are included."""
    records = load_benchmarks(jsonl_fixture)

    flt = RooflineFilter(isa="test_isa", num_threads=1, data_type="f32", operations=frozenset({"fma"}))
    model = assemble_roofline(records, flt)

    assert "fma" in model.peak_performance_by_op
    assert "add" not in model.peak_performance_by_op
    # Memory should still pass through (operations filter doesn't apply to memory)
    assert "L1" in model.bandwidth_by_level


def test_assemble_roofline_filter_load_store_ratio(jsonl_fixture: Path) -> None:
    """Only memory records with matching load_store_ratio are included."""
    records = load_benchmarks(jsonl_fixture)

    # No memory records have "1:1" ratio in the fixture
    flt = RooflineFilter(isa="test_isa", num_threads=1, data_type="f32", load_store_ratio="1:1")
    model = assemble_roofline(records, flt)

    # Memory should be empty (no matching ratio)
    assert model.bandwidth_by_level == {}
    # Arithmetic should still be present (load_store_ratio doesn't apply)
    assert "fma" in model.peak_performance_by_op


def test_assemble_roofline_operations_none(jsonl_fixture: Path) -> None:
    """operations=None should include all operations."""
    records = load_benchmarks(jsonl_fixture)

    flt = RooflineFilter(isa="test_isa", num_threads=1, data_type="f32", operations=None)
    model = assemble_roofline(records, flt)

    assert "fma" in model.peak_performance_by_op
    assert "add" in model.peak_performance_by_op


# ── ridge_points ──────────────────────────────────────────────────────────────


def test_ridge_points(jsonl_fixture: Path) -> None:
    records = load_benchmarks(jsonl_fixture)
    flt = RooflineFilter(isa="test_isa", num_threads=1, data_type="f32")
    model = assemble_roofline(records, flt)

    rp = model.ridge_points()
    # peak_perf from timestamp B fma = 120 GOPS
    peak_perf = 120.0 * 1e9

    assert "L1" in rp
    assert isinstance(rp["L1"], ArithmeticIntensity)
    assert rp["L1"].value == pytest.approx(peak_perf / (400.0 * 1e9))

    assert "DRAM" in rp
    assert rp["DRAM"].value == pytest.approx(peak_perf / (30.0 * 1e9))

    # L3 is not in data
    assert "L3" not in rp


def test_ridge_points_empty_model() -> None:
    """No data → empty ridge points, not an error."""
    flt = RooflineFilter(isa="unknown")
    model = assemble_roofline([], flt)
    assert model.ridge_points() == {}


# ── assemble_roofline_from_file ────────────────────────────────────────────────


def test_assemble_from_file(jsonl_fixture: Path) -> None:
    flt = RooflineFilter(isa="test_isa", num_threads=1, data_type="f32")
    model = assemble_roofline_from_file(jsonl_fixture, flt)

    assert "L1" in model.bandwidth_by_level
    assert "fma" in model.peak_performance_by_op
    assert model.peak_performance_by_op["fma"].value == pytest.approx(120.0 * 1e9)
