"""Unit tests for GUI data model serialization."""

from __future__ import annotations

import math

import pytest

from gui.data import RoofConfig, RoofStore, build_roofline_figure
from roofline_assembly import ApplicationPoint, ApplicationRecord, BenchmarkRecord

pytestmark = pytest.mark.unit


def test_roofstore_round_trip_preserves_all_fields() -> None:
    """to_dict -> from_dict preserves all RoofConfig fields including compute_insts."""
    roof = RoofConfig(
        label="Test Roof",
        machine="Machine X",
        isa="arm_neon",
        threads=4,
        data_type="f64",
        compute_insts=["mul", "div"],
        load_store_ratio="1:1",
    )
    store = RoofStore(roof_template=roof)
    data = store.to_dict()
    restored = RoofStore.from_dict(data)

    assert len(restored.roofs) == 1
    r = restored.roofs[0]
    assert r.label == "Test Roof"
    assert r.machine == "Machine X"
    assert r.isa == "arm_neon"
    assert r.threads == 4
    assert r.data_type == "f64"
    assert r.compute_insts == ["mul", "div"]  # the field that was silently lost
    assert r.load_store_ratio == "1:1"
    assert r.app_ids == []
    assert restored.normalize_by_threads is False


def test_roofstore_round_trip_with_none_fields() -> None:
    """None field values survive to_dict -> from_dict round trip."""
    roof = RoofConfig(
        label="Cleared Roof",
        machine=None,
        isa=None,
        threads=None,
        data_type=None,
        load_store_ratio=None,
    )
    store = RoofStore(roof_template=roof)
    restored = RoofStore.from_dict(store.to_dict())
    r = restored.roofs[0]
    assert r.machine is None
    assert r.isa is None
    assert r.threads is None
    assert r.data_type is None
    assert r.load_store_ratio is None


def test_roofstore_round_trip_app_ids() -> None:
    """app_ids survive to_dict -> from_dict round trip."""
    roof = RoofConfig(app_ids=["abc123"])
    store = RoofStore(roof_template=roof)
    restored = RoofStore.from_dict(store.to_dict())
    r = restored.roofs[0]
    assert r.app_ids == ["abc123"]
    # normalize_by_threads round-trips at the store level
    store.normalize_by_threads = True
    restored2 = RoofStore.from_dict(store.to_dict())
    assert restored2.normalize_by_threads is True


def test_build_roofline_figure_renders_application_points() -> None:
    """Enabled roof with selected app ids renders marker traces."""
    rec = ApplicationRecord(
        id="r1",
        label="run1 \u2014 2024-01-01 (global)",
        aggregation="global",
        metadata={},
        points=[
            ApplicationPoint(
                label="p1",
                total_flops=1e9,
                total_bytes=1e6,
                runtime_s=0.5,
                num_ranks=1,
                num_threads=1,
                num_regions=1,
                arithmetic_intensity=0.5,
                flops_per_second=2e9,
                bandwidth=1e9,
            ),
            ApplicationPoint(
                label="p2",
                total_flops=2e9,
                total_bytes=2e6,
                runtime_s=1.0,
                num_ranks=1,
                num_threads=2,
                num_regions=1,
                arithmetic_intensity=1.0,
                flops_per_second=4e9,
                bandwidth=2e9,
            ),
        ],
    )
    roof = RoofConfig(app_ids=["r1"])
    fig = build_roofline_figure([roof], [], {"r1": rec})
    markers = [t for t in fig.data if t.mode == "markers"]
    assert len(markers) == 1
    assert list(markers[0].x) == [0.5, 1.0]


def test_build_roofline_figure_normalize_by_threads() -> None:
    """Normalization halves performance when threads=2 and num_threads=2."""
    ts = "2026-01-01T00:00:00"
    records: list[BenchmarkRecord] = [
        {
            "type": "arithmetic",
            "name": "fma",
            "isa": "test_isa",
            "machine": "test_machine",
            "data_type": "f32",
            "num_threads": 2,
            "timestamp": ts,
            "operation": "fma",
            "performance_gops": 120.0,
        },
        {
            "type": "memory",
            "name": "L1 load",
            "isa": "test_isa",
            "machine": "test_machine",
            "data_type": "f32",
            "num_threads": 2,
            "timestamp": ts,
            "load_store_ratio": "2:1",
            "cache_level": "L1",
            "bandwidth_gbps": 400.0,
        },
    ]
    roof = RoofConfig(
        label="Test Roof",
        isa="test_isa",
        machine="test_machine",
        threads=2,
        data_type="f32",
        compute_insts=["fma"],
        load_store_ratio="2:1",
        app_ids=["r1"],
    )
    rec = ApplicationRecord(
        id="r1",
        label="run1",
        aggregation="global",
        metadata={},
        points=[
            ApplicationPoint(
                label="p1",
                total_flops=2e9,
                total_bytes=1e6,
                runtime_s=0.5,
                num_ranks=1,
                num_threads=2,
                num_regions=1,
                arithmetic_intensity=2.0,
                flops_per_second=4e9,
                bandwidth=2e9,
            ),
        ],
    )
    # Without normalization: app perf=4e9/1e9=4.0, peak ceiling=120/1e9=120
    fig_un = build_roofline_figure([roof], records, {"r1": rec})
    un_markers = [t for t in fig_un.data if t.mode == "markers" and t.showlegend]
    assert len(un_markers) == 1
    # The y-value is flops_per_second/1e9 = 4e9/1e9 = 4.0
    assert list(un_markers[0].y) == pytest.approx([4.0])
    # Ceiling line at perf=120 GOPS/s (the last ceiling trace)
    perf_ceilings = [t for t in fig_un.data if t.mode == "lines" and t.name.startswith("Test Roof")]
    assert any(t.y[0] == pytest.approx(120.0) for t in perf_ceilings)

    # With normalization: app perf=4e9/(2)/1e9=2.0, ceiling=120/2=60
    fig_norm = build_roofline_figure([roof], records, {"r1": rec}, normalize_by_threads=True)
    norm_markers = [t for t in fig_norm.data if t.mode == "markers" and t.showlegend]
    assert len(norm_markers) == 1
    assert list(norm_markers[0].y) == pytest.approx([2.0])
    perf_ceilings_norm = [t for t in fig_norm.data if t.mode == "lines" and t.name.startswith("Test Roof")]
    assert any(t.y[0] == pytest.approx(60.0) for t in perf_ceilings_norm)


def test_build_roofline_figure_dynamic_ranges() -> None:
    """Ranges and log2 ticks are computed from ridge points and peak perf."""
    ts = "2026-01-01T00:00:00"
    records: list[BenchmarkRecord] = [
        {
            "type": "arithmetic",
            "name": "fma",
            "isa": "test_isa",
            "machine": "test_machine",
            "data_type": "f32",
            "num_threads": 1,
            "timestamp": ts,
            "operation": "fma",
            "performance_gops": 120.0,
        },
        {
            "type": "memory",
            "name": "L1 load",
            "isa": "test_isa",
            "machine": "test_machine",
            "data_type": "f32",
            "num_threads": 1,
            "timestamp": ts,
            "load_store_ratio": "2:1",
            "cache_level": "L1",
            "bandwidth_gbps": 400.0,
        },
        {
            "type": "memory",
            "name": "L2 load",
            "isa": "test_isa",
            "machine": "test_machine",
            "data_type": "f32",
            "num_threads": 1,
            "timestamp": ts,
            "load_store_ratio": "2:1",
            "cache_level": "L2",
            "bandwidth_gbps": 100.0,
        },
        {
            "type": "memory",
            "name": "DRAM load",
            "isa": "test_isa",
            "machine": "test_machine",
            "data_type": "f32",
            "num_threads": 1,
            "timestamp": ts,
            "load_store_ratio": "2:1",
            "cache_level": "DRAM",
            "bandwidth_gbps": 30.0,
        },
    ]
    roof = RoofConfig(
        isa="test_isa",
        machine="test_machine",
        threads=1,
        data_type="f32",
        compute_insts=["fma"],
        load_store_ratio="2:1",
    )
    fig = build_roofline_figure([roof], records)
    assert list(fig.layout.xaxis.range) == pytest.approx([math.log10(0.03), math.log10(40.0)], rel=1e-6)
    assert list(fig.layout.yaxis.range) == pytest.approx([math.log10(12.0), math.log10(240.0)], rel=1e-6)
    assert fig.layout.xaxis.dtick == pytest.approx(math.log10(2))
    assert fig.layout.yaxis.dtick == pytest.approx(math.log10(2))
    assert fig.layout.xaxis.tick0 == 0
    assert fig.layout.yaxis.tick0 == 0
    assert fig.layout.xaxis.exponentformat == "none"


def test_build_roofline_figure_empty_fallback() -> None:
    """Empty roofs/records produce the fallback hardcoded ranges."""
    fig0 = build_roofline_figure([], [])
    assert list(fig0.layout.xaxis.range) == pytest.approx([-2.0, 2.0])
    assert list(fig0.layout.yaxis.range) == pytest.approx([0.0, 3.5])
