"""Unit tests for the reusable roofline assembly module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carm_roofline.roofline_assembly import (
    RooflineFilter,
    assemble_roofline,
    assemble_roofline_from_file,
    discover_filter_options,
    load_all_benchmarks,
    load_benchmarks,
)

from carm_roofline.core import ArithmeticIntensity, Bandwidth, Performance

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
    result = discover_filter_options(records, RooflineFilter())

    assert "data_type" in result
    assert "f32" in result["data_type"]
    assert "machine" in result
    assert "isa" in result
    assert "num_threads" in result
    assert "load_store_ratio" in result


def test_discover_filter_options_cross_field(jsonl_fixture: Path) -> None:
    """Cross-field filtering: selecting test_isa narrows threads to [1]."""
    records = load_benchmarks(jsonl_fixture)
    # Filter by test_isa -> only threads=1 records match
    result = discover_filter_options(records, RooflineFilter(isa="test_isa"))
    assert result["num_threads"] == [1]
    # ISA options are unconstrained (isa filter not applied to isa field itself)
    assert "test_isa" in result["isa"]
    # other_isa has arithmetic but no matching memory -> should NOT appear in options.
    # (Old test asserted the opposite — the whole point of this refactor.)
    result2 = discover_filter_options(records, RooflineFilter(isa="other_isa"))
    assert "other_isa" not in result2["isa"]
    assert result2["load_store_ratio"] == []
    # No locks -> only isas that can form a complete roofline appear.
    result3 = discover_filter_options(records, RooflineFilter())
    assert "test_isa" in result3["isa"]
    assert "other_isa" not in result3["isa"]



def test_discover_filter_options_excludes_ratios_without_arithmetic(tmp_path: Path) -> None:
    """Load-store ratio values without matching arithmetic records are excluded."""
    records: list[dict[str, object]] = [
        # Arithmetic + memory pair for "2:1" on machine m1 with x86/f32/1 thread
        {
            "type": "arithmetic",
            "isa": "x86",
            "machine": "m1",
            "data_type": "f32",
            "num_threads": 1,
            "operation": "fma",
            "performance_gops": 100.0,
            "timestamp": "2026-01-01T00:00:00",
        },
        {
            "type": "memory",
            "isa": "x86",
            "machine": "m1",
            "data_type": "f32",
            "num_threads": 1,
            "load_store_ratio": "2:1",
            "cache_level": "L1",
            "bandwidth_gbps": 400.0,
            "timestamp": "2026-01-01T00:00:00",
        },
        # Memory record with "1:0" on a different machine (m2) — no arithmetic on m2
        {
            "type": "memory",
            "isa": "x86",
            "machine": "m2",
            "data_type": "f32",
            "num_threads": 1,
            "load_store_ratio": "1:0",
            "cache_level": "L1",
            "bandwidth_gbps": 400.0,
            "timestamp": "2026-01-01T00:00:00",
        },
    ]
    path = tmp_path / "benchmarks.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True))
            f.write("\n")
    loaded = load_benchmarks(path)
    result = discover_filter_options(loaded, RooflineFilter())

    assert "2:1" in result["load_store_ratio"], "ratio with matching arithmetic should be included"
    assert "1:0" not in result["load_store_ratio"], "ratio without matching arithmetic should be excluded"


def test_discover_filter_options_excludes_ratios_no_arithmetic_at_all(tmp_path: Path) -> None:
    """When no arithmetic records exist at all, all ratios are excluded."""
    records: list[dict[str, object]] = [
        {
            "type": "memory",
            "isa": "x86",
            "machine": "m1",
            "data_type": "f32",
            "num_threads": 1,
            "load_store_ratio": "1:0",
            "cache_level": "L1",
            "bandwidth_gbps": 400.0,
            "timestamp": "2026-01-01T00:00:00",
        },
    ]
    path = tmp_path / "benchmarks.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(records[0], sort_keys=True) + "\n")
    loaded = load_benchmarks(path)
    result = discover_filter_options(loaded, RooflineFilter())

    assert result["load_store_ratio"] == [], "no arithmetic records → ratios should be empty"


def test_discover_filter_options_includes_ratios_when_arithmetic_present(tmp_path: Path) -> None:
    """All ratios where matching arithmetic records exist should be included."""
    records: list[dict[str, object]] = [
        {
            "type": "arithmetic",
            "isa": "x86",
            "machine": "m1",
            "data_type": "f32",
            "num_threads": 1,
            "operation": "fma",
            "performance_gops": 100.0,
            "timestamp": "2026-01-01T00:00:00",
        },
        # Two ratios, both with matching arithmetic for same (m1, x86, f32, 1)
        {
            "type": "memory",
            "isa": "x86",
            "machine": "m1",
            "data_type": "f32",
            "num_threads": 1,
            "load_store_ratio": "2:1",
            "cache_level": "L1",
            "bandwidth_gbps": 400.0,
            "timestamp": "2026-01-01T00:00:00",
        },
        {
            "type": "memory",
            "isa": "x86",
            "machine": "m1",
            "data_type": "f32",
            "num_threads": 1,
            "load_store_ratio": "1:1",
            "cache_level": "L1",
            "bandwidth_gbps": 300.0,
            "timestamp": "2026-01-01T00:00:00",
        },
    ]
    path = tmp_path / "benchmarks.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True))
            f.write("\n")
    loaded = load_benchmarks(path)
    result = discover_filter_options(loaded, RooflineFilter())

    assert "2:1" in result["load_store_ratio"]
    assert "1:1" in result["load_store_ratio"]


def test_discover_filter_options_cross_machine_not_paired(tmp_path: Path) -> None:
    """Ratios must have arithmetic on the same machine — cross-machine pairings don't count."""
    records: list[dict[str, object]] = [
        # Arithmetic on machine A, memory with ratio "1:0" on machine B
        {
            "type": "arithmetic",
            "isa": "x86",
            "machine": "machine_a",
            "data_type": "f32",
            "num_threads": 1,
            "operation": "fma",
            "performance_gops": 100.0,
            "timestamp": "2026-01-01T00:00:00",
        },
        {
            "type": "memory",
            "isa": "x86",
            "machine": "machine_b",
            "data_type": "f32",
            "num_threads": 1,
            "load_store_ratio": "1:0",
            "cache_level": "L1",
            "bandwidth_gbps": 400.0,
            "timestamp": "2026-01-01T00:00:00",
        },
    ]
    path = tmp_path / "benchmarks.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True))
            f.write("\n")
    loaded = load_benchmarks(path)
    result = discover_filter_options(loaded, RooflineFilter())

    assert "1:0" not in result["load_store_ratio"], (
        "ratio should not appear — arithmetic and memory are on different machines"
    )


def test_discover_filter_options_excludes_ratios_without_arithmetic_on_selection(
    jsonl_fixture: Path,
) -> None:
    """Filtered discovery also excludes ratios without matching arithmetic."""
    records = load_benchmarks(jsonl_fixture)
    # The fixture has (test_machine, test_isa, f32, 1) for both arithmetic and "2:1" memory
    result = discover_filter_options(
        records,
        RooflineFilter(machine="test_machine", isa="test_isa", num_threads=1, data_type="f32"),
    )
    # "2:1" should appear (matched by arithmetic)
    assert "2:1" in result["load_store_ratio"]

    # Filter by a combo with memory but no arithmetic (only "other_isa" arithmetic exists in fixture)
    result2 = discover_filter_options(
        records,
        RooflineFilter(machine="test_machine", isa="nonexistent_isa"),
    )
    assert result2["load_store_ratio"] == [], (
        "arithmetic for nonexistent_isa doesn't exist, so ratios should be empty"
    )



def test_discover_filter_options_excludes_ratio_when_arithmetic_mismatches_tuple(tmp_path: Path) -> None:
    """Ratio excluded when arithmetic records exist but for a DIFFERENT (machine, isa, threads, data_type) tuple.

    This reproduces the exact user scenario: memory benchmarks at a specific
    (machine, isa, threads, data_type) with ratio "1:0", and arithmetic benchmarks
    that exist but at a DIFFERENT thread count or data type.  The ratio must not
    appear because no COMPLETE roofline can be assembled — the arithmetic records
    found by a loose ``_matches_filter`` with None fields would incorrectly pair with
    the memory records.  The fix uses per-tuple arithmetic-existence matching
    (same as ``discover_filter_options``), not ``_matches_filter``.
    """
    records: list[dict[str, object]] = [
        # Memory with "1:0" for (m1, x86, 1, f32)
        {
            "type": "memory",
            "isa": "x86",
            "machine": "m1",
            "data_type": "f32",
            "num_threads": 1,
            "load_store_ratio": "1:0",
            "cache_level": "L1",
            "memory_level_name": "L1",
            "bandwidth_gbps": 400.0,
            "timestamp": "2026-06-01T00:00:00",
        },
        # Arithmetic EXISTS but with (m1, x86, 4, f64) — DIFFERENT threads AND data_type
        {
            "type": "arithmetic",
            "isa": "x86",
            "machine": "m1",
            "data_type": "f64",
            "num_threads": 4,
            "operation": "fma",
            "performance_gops": 200.0,
            "timestamp": "2026-06-01T00:00:00",
        },
    ]
    path = tmp_path / "benchmarks.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    loaded = load_benchmarks(path)

    # Global discovery — no filters
    global_opts = discover_filter_options(loaded, RooflineFilter())

    assert "1:0" not in global_opts["load_store_ratio"], (
        "global: '1:0' excluded because memory tuple (m1, x86, 1, f32) has no matching arithmetic"
    )

    # Filtered discovery with all fields cleared (simulates "all other filter options cleared")
    all_cleared = discover_filter_options(loaded, RooflineFilter())

    assert "1:0" not in all_cleared["load_store_ratio"], (
        "all-cleared: '1:0' must not appear — arithmetic exists but for (4, f64) not (1, f32)"
    )

    # Partial filter: bound to machine but nothing else
    by_machine = discover_filter_options(loaded, RooflineFilter(machine="m1"))
    assert "1:0" not in by_machine["load_store_ratio"], (
        "machine filter alone: '1:0' still excluded — no arithmetic for (m1, x86, 1, f32)"
    )


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
