"""Unit tests for GUI data model serialization."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from carm_roofline.gui.data import (
    GUISettings,
    RoofConfig,
    RoofStore,
    build_paraver_figure,
    build_roofline_figure,
)
from carm_roofline.gui.providers import ParaverData
from carm_roofline.paraver import ParaverWindowMode
from carm_roofline.roofline_assembly import ApplicationPoint, ApplicationRecord, BenchmarkRecord

pytestmark = pytest.mark.unit


def test_roofstore_round_trip_preserves_all_fields() -> None:
    """to_dict -> from_dict preserves all RoofConfig fields including compute_insts."""
    roof = RoofConfig(
        label="Test Roof",
        machine="Machine X",
        isa="arm_neon",
        num_threads=4,
        data_type="f64",
        compute_insts=["mul", "div"],
        load_store_ratio="1:1",
        actual_frequency_hz=2500000000,
    )
    store = RoofStore(roof_template=roof)
    data = store.to_dict()
    restored = RoofStore.from_dict(data)

    assert len(restored.roofs) == 1
    r = restored.roofs[0]
    assert r.label == "Test Roof"
    assert r.machine == "Machine X"
    assert r.isa == "arm_neon"
    assert r.num_threads == 4
    assert r.data_type == "f64"
    assert r.compute_insts == ["mul", "div"]  # the field that was silently lost
    assert r.load_store_ratio == "1:1"
    assert r.actual_frequency_hz == 2500000000
    assert r.app_ids == []
    assert restored.settings.normalize_by_threads is False


def test_roofstore_round_trip_with_none_fields() -> None:
    """None field values survive to_dict -> from_dict round trip."""
    roof = RoofConfig(
        label="Cleared Roof",
        machine=None,
        isa=None,
        num_threads=None,
        data_type=None,
        load_store_ratio=None,
    )
    store = RoofStore(roof_template=roof)
    restored = RoofStore.from_dict(store.to_dict())
    r = restored.roofs[0]
    assert r.machine is None
    assert r.isa is None
    assert r.num_threads is None
    assert r.data_type is None
    assert r.actual_frequency_hz is None
    assert r.load_store_ratio is None


def test_roofstore_round_trip_app_ids() -> None:
    """app_ids survive to_dict -> from_dict round trip."""
    roof = RoofConfig(app_ids=["abc123"])
    store = RoofStore(roof_template=roof)
    restored = RoofStore.from_dict(store.to_dict())
    r = restored.roofs[0]
    assert r.app_ids == ["abc123"]
    # normalize_by_threads round-trips at the store level
    store.settings.normalize_by_threads = True
    restored2 = RoofStore.from_dict(store.to_dict())
    assert restored2.settings.normalize_by_threads is True


def test_roofstore_round_trip_preserves_paraver_state() -> None:
    """ParaverState survives to_dict -> from_dict round trip; default round-trips too."""
    store = RoofStore()
    store.paraver_state.time_window = (1.0, 5.5)
    restored = RoofStore.from_dict(store.to_dict())
    assert restored.paraver_state.time_window == (1.0, 5.5)

    default_store = RoofStore()
    assert default_store.paraver_state.time_window is None
    restored_default = RoofStore.from_dict(default_store.to_dict())
    assert restored_default.paraver_state.time_window is None

    # Old store JSON without the "paraver" key deserializes to defaults.
    old_data = default_store.to_dict()
    del old_data["paraver"]
    assert RoofStore.from_dict(old_data).paraver_state.time_window is None


def test_build_roofline_figure_renders_application_points() -> None:
    """Enabled roof with selected app ids renders marker traces."""
    rec = ApplicationRecord(
        id="r1",
        label="run1 — 2024-01-01 (global)",
        aggregation="global",
        metadata={},
        machine="test_machine",
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
    markers = [t for t in fig.data if t.mode == "markers+text"]
    assert len(markers) == 1
    assert list(markers[0].x) == [0.5, 1.0]
    # marker sizes are not uniform (different runtimes produce different sizes)
    assert markers[0].marker.size is not None
    sizes = list(markers[0].marker.size)
    assert len(sizes) == 2
    assert sizes == pytest.approx([50.0, 2550.0])
    # sizemode is 'area'
    assert markers[0].marker.sizemode == "area"
    # marker opacity is 0.6
    assert markers[0].marker.opacity == 0.6
    # customdata contains rich tooltip via _format_point_tooltip
    assert len(markers[0].customdata) == 2
    assert "<b>run1 \u2014 2024-01-01 (global)</b>" in markers[0].customdata[0]
    assert "<i>p1</i>" in markers[0].customdata[0]
    assert "Performance" in markers[0].customdata[0]
    assert "Execution" in markers[0].customdata[0]
    assert "Work" in markers[0].customdata[0]
    assert "  Arithmetic Intensity: 0.500 OPS/Byte" in markers[0].customdata[0]
    assert "  Duration:" in markers[0].customdata[0]
    assert markers[0].hovertemplate == "%{customdata}<extra></extra>"


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
        num_threads=2,
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
        machine="test_machine",
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
    un_markers = [t for t in fig_un.data if t.mode == "markers+text" and t.showlegend]
    assert len(un_markers) == 1
    # The y-value is flops_per_second/1e9 = 4e9/1e9 = 4.0
    assert list(un_markers[0].y) == pytest.approx([4.0])
    # Ceiling line at perf=120 GOPS/s (the last ceiling trace)
    perf_ceilings = [t for t in fig_un.data if t.mode == "lines" and t.name.startswith("Test Roof")]
    assert any(t.y[0] == pytest.approx(120.0) for t in perf_ceilings)

    # With normalization: app perf=4e9/(2)/1e9=2.0, ceiling=120/2=60
    fig_norm = build_roofline_figure([roof], records, {"r1": rec}, settings=GUISettings(normalize_by_threads=True))
    norm_markers = [t for t in fig_norm.data if t.mode == "markers+text" and t.showlegend]
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
        num_threads=1,
        data_type="f32",
        compute_insts=["fma"],
        load_store_ratio="2:1",
    )
    fig = build_roofline_figure([roof], records)
    assert list(fig.layout.xaxis.range) == pytest.approx([math.log10(0.01875), math.log10(16.0)], rel=1e-6)
    assert list(fig.layout.yaxis.range) == pytest.approx([math.log10(0.5625), math.log10(240.0)], rel=1e-6)
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


def _paraver_trace() -> pd.DataFrame:
    """Trace-shaped frame with two threads and codes 1/8/8."""
    return pd.DataFrame(
        {
            "thread_id": ["0", "1", "2"],
            "time_s": [0.0, 1.0, 2.0],
            "duration_s": [1.0, 1.0, 1.0],
            "state_code": [1.0, 8.0, 8.0],
            "flops": [100.0, 200.0, 300.0],
            "bytes": [10.0, 20.0, 30.0],
            "ai": [10.0, 10.0, 10.0],
            "perf": [100.0, 200.0, 300.0],
        }
    )


def test_build_paraver_figure_code_mode_groups_by_legend() -> None:
    """Code mode adds one marker trace per legend label, colored by the legend."""
    trace = _paraver_trace()
    trace["legend_label"] = ["Running", "Wait/WaitAll", "Wait/WaitAll"]
    trace["legend_color"] = ["rgb(0,0,255)", "rgb(235,0,0)", "rgb(235,0,0)"]
    legend = pd.DataFrame(
        {
            "code": [1.0, 8.0],
            "code_end": [1.0, 8.0],
            "label": ["Running", "Wait/WaitAll"],
            "r": [0, 235],
            "g": [0, 0],
            "b": [255, 0],
        }
    )
    paraver = ParaverData(
        trace=trace,
        label="t — w.csv",
        window_mode=ParaverWindowMode.CODE,
        time_unit="nanoseconds",
        prv_path="/p/t.prv",
        legend=legend,
    )
    fig = build_paraver_figure([RoofConfig()], [], paraver, trace)
    markers = [t for t in fig.data if t.mode == "markers"]
    assert [m.name for m in markers] == ["Running", "Wait/WaitAll"]
    assert [m.marker.color for m in markers] == ["rgb(0,0,255)", "rgb(235,0,0)"]
    # x/y values match the rows of each legend group (perf in GOPS).
    assert list(markers[0].x) == [10.0]
    assert list(markers[0].y) == [100.0 / 1e9]
    assert list(markers[1].x) == [10.0, 10.0]
    assert list(markers[1].y) == [200.0 / 1e9, 300.0 / 1e9]
    # Tooltip row: raw paraver value (1.0 renders as 1) with the semantic state label in parens.
    assert "<b>Paraver Value</b><br>  1 (Running)" in markers[0].customdata[0]


def test_build_paraver_figure_gradient_mode_single_trace() -> None:
    """Gradient mode adds a single Viridis-colored trace named by paraver.label."""
    trace = _paraver_trace()
    paraver = ParaverData(
        trace=trace,
        label="t — w.csv",
        window_mode=ParaverWindowMode.GRADIENT,
        time_unit="nanoseconds",
        prv_path="/p/t.prv",
        legend=None,
    )
    fig = build_paraver_figure([RoofConfig()], [], paraver, trace)
    markers = [t for t in fig.data if t.mode == "markers"]
    assert len(markers) == 1
    m = markers[0]
    assert m.name == "t — w.csv"
    assert list(m.marker.color) == [1.0, 8.0, 8.0]
    # Plotly resolves the "Viridis" name into its colorscale; pin its endpoints.
    assert m.marker.colorscale[0] == (0.0, "#440154")
    assert m.marker.colorscale[-1] == (1.0, "#fde725")
    assert m.marker.showscale is False
    # Gradient tooltips carry the raw value the color encodes (1.0 -> 1, 8.0 -> 8).
    assert "<b>Paraver Value</b><br>  1" in m.customdata[0]
    assert "<b>Paraver Value</b><br>  8" in m.customdata[1]


def test_build_paraver_figure_tooltip_omits_nan_value() -> None:
    """Rows with NaN state code omit the Paraver Value row from their tooltip."""
    trace = _paraver_trace()
    trace["state_code"] = [1.0, float("nan"), 8.0]
    paraver = ParaverData(
        trace=trace,
        label="t — w.csv",
        window_mode=ParaverWindowMode.GRADIENT,
        time_unit="nanoseconds",
        prv_path="/p/t.prv",
        legend=None,
    )
    fig = build_paraver_figure([RoofConfig()], [], paraver, trace)
    markers = [t for t in fig.data if t.mode == "markers"]
    assert len(markers) == 1
    assert "<b>Paraver Value</b>" in markers[0].customdata[0]
    assert "<b>Paraver Value</b>" not in markers[0].customdata[1]


def test_build_paraver_figure_skips_unmatched_rows() -> None:
    """Rows whose state code no legend range covers produce no trace or legend entry."""
    trace = _paraver_trace()
    trace["legend_label"] = ["Running", None, None]
    trace["legend_color"] = ["rgb(0,0,255)", None, None]
    paraver = ParaverData(
        trace=trace,
        label="t — w.csv",
        window_mode=ParaverWindowMode.CODE,
        time_unit="nanoseconds",
        prv_path="/p/t.prv",
        legend=None,
    )
    fig = build_paraver_figure([RoofConfig()], [], paraver, trace)
    markers = [t for t in fig.data if t.mode == "markers"]
    assert [m.name for m in markers] == ["Running"]
    assert list(markers[0].x) == [10.0]


def test_build_paraver_figure_extends_ranges_to_points() -> None:
    """Positive-metric extents widen the log10 axis ranges (0.5 decade margin)."""
    trace = _paraver_trace()
    trace["ai"] = [1e-9, 2e-9, 3e-9]
    trace["perf"] = [1.0, 2.0, 3.0]
    trace["legend_label"] = ["Running"] * 3
    trace["legend_color"] = ["rgb(0,0,255)"] * 3
    paraver = ParaverData(
        trace=trace,
        label="t — w.csv",
        window_mode=ParaverWindowMode.CODE,
        time_unit="nanoseconds",
        prv_path="/p/t.prv",
        legend=None,
    )
    fig = build_paraver_figure([RoofConfig()], [], paraver, trace)
    # Fallback ranges are [-2, 2] x [0, 3.5]; tiny positive metrics push lo below.
    assert fig.layout.xaxis.range[0] < -2
    assert fig.layout.yaxis.range[0] < 0


def test_build_paraver_figure_none_paraver_draws_ceilings_only() -> None:
    """paraver=None/trace=None (failed load) draws ceilings only, without exception."""
    fig = build_paraver_figure([RoofConfig()], [], None, None)
    assert all(t.mode != "markers" for t in fig.data)
