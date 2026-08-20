"""Unit tests for GUI data model serialization."""

from __future__ import annotations

import math
import re

import pandas as pd
import pytest

from carm_roofline.gui.config import GUISettings
from carm_roofline.gui.data import (
    _BW_FILL_OPACITIES,
    _BW_LINE_STYLES,
    _SELECTED_FILL_BASE_OPACITY,
    RoofConfig,
    RoofStore,
    _format_point_tooltip,
    _residency_alpha,
    _residency_to_level_fractions,
    _residency_width_mult,
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
    store.paraver_state.ai_threshold = 1e-3
    store.paraver_state.duration_threshold = 1e-3
    store.paraver_state.color_mode = "isa"
    restored = RoofStore.from_dict(store.to_dict())
    assert restored.paraver_state.time_window == (1.0, 5.5)
    assert restored.paraver_state.ai_threshold == pytest.approx(1e-3)
    assert restored.paraver_state.duration_threshold == pytest.approx(1e-3)
    assert restored.paraver_state.color_mode == "isa"

    # An explicit None (filter off) round-trips as None, not the default.
    store.paraver_state.ai_threshold = None
    store.paraver_state.duration_threshold = None
    assert RoofStore.from_dict(store.to_dict()).paraver_state.ai_threshold is None
    assert RoofStore.from_dict(store.to_dict()).paraver_state.duration_threshold is None

    default_store = RoofStore()
    assert default_store.paraver_state.time_window is None
    # Default is the minimum active filter (1e-5), not "no filtering".
    assert default_store.paraver_state.ai_threshold == pytest.approx(1e-5)
    # Default is the minimum active filter (100 us), not "no filtering".
    assert default_store.paraver_state.duration_threshold == pytest.approx(1e-4)
    restored_default = RoofStore.from_dict(default_store.to_dict())
    assert restored_default.paraver_state.time_window is None
    assert restored_default.paraver_state.ai_threshold == pytest.approx(1e-5)
    assert restored_default.paraver_state.duration_threshold == pytest.approx(1e-4)
    assert restored_default.paraver_state.color_mode == "paraver"

    # An empty dict (e.g. the `store_data or {}` guard) falls back to field defaults.
    empty = RoofStore.from_dict({})
    assert empty.paraver_state.time_window is None
    assert empty.paraver_state.ai_threshold == pytest.approx(1e-5)
    assert empty.paraver_state.duration_threshold == pytest.approx(1e-4)
    assert empty.paraver_state.color_mode == "paraver"


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
    roof = RoofConfig(roof_id="r1", app_ids=["r1"])
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
    # customdata[0] is the rich tooltip via _format_point_tooltip
    assert len(markers[0].customdata) == 2
    assert "<b>run1 \u2014 2024-01-01 (global)</b>" in markers[0].customdata[0][0]
    assert "<i>p1</i>" in markers[0].customdata[0][0]
    assert "Performance" in markers[0].customdata[0][0]
    assert "Execution" in markers[0].customdata[0][0]
    assert "Work" in markers[0].customdata[0][0]
    assert "  Arithmetic Intensity: 0.500 OPS/Byte" in markers[0].customdata[0][0]
    assert "  Duration:" in markers[0].customdata[0][0]
    # customdata[1] is the per-point cache-residency profile; customdata[2] the roof id
    assert markers[0].customdata[0][1] == {}
    assert markers[0].customdata[0][2] == "r1"
    assert markers[0].hovertemplate == "%{customdata[0]}<extra></extra>"


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
            "load_share": [2.0 / 3.0, 1.0 / 3.0, 1.0],
            "isa_scalar_pct": [100.0 / 3.0, 0.0, 0.0],
            "isa_sse_pct": [200.0 / 3.0, 0.0, 0.0],
            "isa_avx2_pct": [0.0, 100.0, 0.0],
            "isa_avx512_pct": [0.0, 0.0, float("nan")],
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


def test_paraver_figure_trace_only_single_legend_entry() -> None:
    """Trace-only ParaverData draws one gray scatter with the trace stem as its name."""
    trace = _paraver_trace()
    trace["state_code"] = [float("nan")] * len(trace)
    trace["legend_label"] = ["t"] * len(trace)
    trace["legend_color"] = ["rgb(128,128,128)"] * len(trace)
    paraver = ParaverData(
        trace=trace,
        label="t",
        window_mode=ParaverWindowMode.CODE,
        time_unit="Microseconds",
        prv_path="/p/t.prv",
        legend=None,
    )
    fig = build_paraver_figure([], [], paraver, trace)
    markers = [t for t in fig.data if t.mode == "markers"]
    assert len(markers) == 1
    assert markers[0].name == "t"
    assert markers[0].marker.color == "rgb(128,128,128)"


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


def test_build_paraver_figure_color_mode_age() -> None:
    """Age mode replaces the mode-specific traces with one light-to-dark trace."""
    trace = _paraver_trace()
    paraver = ParaverData(
        trace=trace,
        label="t — w.csv",
        window_mode=ParaverWindowMode.CODE,
        time_unit="nanoseconds",
        prv_path="/p/t.prv",
        legend=None,
    )
    fig = build_paraver_figure([RoofConfig()], [], paraver, trace, color_mode="age")
    markers = [t for t in fig.data if t.mode == "markers"]
    assert len(markers) == 1
    m = markers[0]
    assert m.name == "t — w.csv"
    assert list(m.marker.color) == ["#c6dbef", "#6785ad", "#08306b"]
    assert "<b>Paraver Value</b><br>  1" in m.customdata[0]


def test_build_paraver_figure_color_mode_thread() -> None:
    """Thread mode colors one trace deterministically per thread_id."""
    trace = _paraver_trace()
    paraver = ParaverData(
        trace=trace,
        label="t — w.csv",
        window_mode=ParaverWindowMode.GRADIENT,
        time_unit="nanoseconds",
        prv_path="/p/t.prv",
        legend=None,
    )
    fig = build_paraver_figure([RoofConfig()], [], paraver, trace, color_mode="thread")
    markers = [t for t in fig.data if t.mode == "markers"]
    assert len(markers) == 1
    assert list(markers[0].marker.color) == ["#2d5be5", "#b7e52d", "#2de56e"]


def test_build_paraver_figure_color_mode_ldst() -> None:
    """LD/ST mode colors from load_share; no-memory bursts gray."""
    trace = _paraver_trace()
    paraver = ParaverData(
        trace=trace,
        label="t — w.csv",
        window_mode=ParaverWindowMode.CODE,
        time_unit="nanoseconds",
        prv_path="/p/t.prv",
        legend=None,
    )
    fig = build_paraver_figure([RoofConfig()], [], paraver, trace, color_mode="ldst")
    markers = [t for t in fig.data if t.mode == "markers"]
    assert list(markers[0].marker.color) == ["#6464b9", "#b96464", "#0000ff"]
    trace["load_share"] = [2.0 / 3.0, math.nan, 1.0]
    fig = build_paraver_figure([RoofConfig()], [], paraver, trace, color_mode="ldst")
    markers = [t for t in fig.data if t.mode == "markers"]
    assert list(markers[0].marker.color) == ["#6464b9", "#999999", "#0000ff"]


def test_build_paraver_figure_color_mode_isa() -> None:
    """ISA mode blends per-ISA shares; a no-FP burst is gray."""
    trace = _paraver_trace()
    paraver = ParaverData(
        trace=trace,
        label="t — w.csv",
        window_mode=ParaverWindowMode.GRADIENT,
        time_unit="nanoseconds",
        prv_path="/p/t.prv",
        legend=None,
    )
    fig = build_paraver_figure([RoofConfig()], [], paraver, trace, color_mode="isa")
    markers = [t for t in fig.data if t.mode == "markers"]
    assert list(markers[0].marker.color) == ["#b47c45", "#2ca02c", "#999999"]


def test_build_paraver_figure_tooltip_load_store_percentages() -> None:
    """Tooltips derive load/store percentages from the load_share column."""
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
    assert "  Loads: 66.7% | Stores: 33.3%" in markers[0].customdata[0]
    assert "  Loads: 33.3% | Stores: 66.7%" in markers[0].customdata[1]
    assert "  Loads: 100.0% | Stores: 0.0%" in markers[0].customdata[2]
    # NaN load_share (no loads and no stores) renders placeholders, not 0%.
    trace["load_share"] = [2.0 / 3.0, float("nan"), 1.0]
    fig = build_paraver_figure([RoofConfig()], [], paraver, trace)
    markers = [t for t in fig.data if t.mode == "markers"]
    assert "  Loads: - | Stores: -" in markers[0].customdata[1]


def test_build_paraver_figure_tooltip_isa_percentages() -> None:
    """Tooltips list ISAs above 0.1% of operations, rounded to 1 dp, fixed order;
    a '-' placeholder renders when no ISA qualifies (no FP work in the burst)."""
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
    assert "  ISA: Scalar 33.3% | SSE 66.7%" in markers[0].customdata[0]
    assert "  ISA: AVX2 100.0%" in markers[0].customdata[1]
    assert "  ISA: -" in markers[0].customdata[2]  # NaN row (no FP work): placeholder
    # Sub-threshold shares (≤0.1%) are dropped so nothing renders as "0.0%".
    trace["isa_scalar_pct"] = [0.05, 0.0, 0.0]
    trace["isa_sse_pct"] = [0.09, 0.0, 0.0]
    trace["isa_avx2_pct"] = [0.0, 100.0, 0.0]
    trace["isa_avx512_pct"] = [0.0, float("nan"), float("nan")]
    fig = build_paraver_figure([RoofConfig()], [], paraver, trace)
    markers = [t for t in fig.data if t.mode == "markers"]
    assert "  ISA: -" in markers[0].customdata[0]  # sub-threshold only: placeholder
    assert "  ISA: AVX2 100.0%" in markers[0].customdata[1]


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


def test_build_paraver_figure_multi_roof_draws_points_once() -> None:
    """Points are drawn once regardless of roof count (regression: they were drawn per roof)."""
    trace = _paraver_trace()
    paraver = ParaverData(
        trace=trace,
        label="t — w.csv",
        window_mode=ParaverWindowMode.GRADIENT,
        time_unit="nanoseconds",
        prv_path="/p/t.prv",
        legend=None,
    )
    fig = build_paraver_figure([RoofConfig(), RoofConfig()], [], paraver, trace)
    markers = [t for t in fig.data if t.mode == "markers"]
    assert len(markers) == 1
    assert markers[0].name == "t — w.csv"
    assert list(markers[0].marker.color) == [1.0, 8.0, 8.0]


def test_build_paraver_figure_multi_roof_code_mode_legend_once() -> None:
    """Code mode adds each legend entry once even with several roofs."""
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
    fig = build_paraver_figure([RoofConfig(), RoofConfig()], [], paraver, trace)
    markers = [t for t in fig.data if t.mode == "markers"]
    assert [m.name for m in markers] == ["Running", "Wait/WaitAll"]


def test_build_paraver_figure_no_roofs_still_draws_points() -> None:
    """With all roofs removed the trace points are still drawn and widen the axes."""
    trace = _paraver_trace()
    paraver = ParaverData(
        trace=trace,
        label="t — w.csv",
        window_mode=ParaverWindowMode.GRADIENT,
        time_unit="nanoseconds",
        prv_path="/p/t.prv",
        legend=None,
    )
    fig = build_paraver_figure([], [], paraver, trace)
    markers = [t for t in fig.data if t.mode == "markers"]
    assert len(markers) == 1
    # perf in GOPS (100..300) pulls the y-axis bottom below the [0, 3.5] fallback.
    assert fig.layout.yaxis.range[0] < 0


def test_point_tooltip_includes_cache_residency() -> None:
    """Tooltip appends a Cache Residency section when the point carries fractions."""
    rec = ApplicationRecord(id="r1", label="run1", aggregation="global", metadata={}, machine="m", points=[])
    p = ApplicationPoint(
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
        optional_fractions={"cache-residency": {"l1": 0.6, "l2": 0.3, "l3": 0.08, "dram": 0.02}},
    )
    tooltip = _format_point_tooltip(rec, p)
    assert "<b>Cache Residency</b>" in tooltip
    assert "L1: 60.0% | L2: 30.0% | L3: 8.0% | DRAM: 2.0%" in tooltip


def test_point_tooltip_includes_3key_cache_residency() -> None:
    """Tooltip renders the merged L3+DRAM label for 3-bucket fractions."""
    rec = ApplicationRecord(id="r1", label="run1", aggregation="global", metadata={}, machine="m", points=[])
    p = ApplicationPoint(
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
        optional_fractions={"cache-residency": {"l1": 0.6, "l2": 0.3, "l3plus": 0.1}},
    )
    tooltip = _format_point_tooltip(rec, p)
    assert "<b>Cache Residency</b>" in tooltip
    assert "L1: 60.0% | L2: 30.0% | L3+DRAM: 10.0%" in tooltip


def test_point_tooltip_includes_l2plus_cache_residency() -> None:
    """Tooltip renders the merged L2+L3+DRAM label for an l2plus bucket."""
    rec = ApplicationRecord(id="r1", label="run1", aggregation="global", metadata={}, machine="m", points=[])
    p = ApplicationPoint(
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
        optional_fractions={"cache-residency": {"l1": 0.2, "l2plus": 0.8}},
    )
    tooltip = _format_point_tooltip(rec, p)
    assert "<b>Cache Residency</b>" in tooltip
    assert "L1: 20.0% | L2+L3+DRAM: 80.0%" in tooltip


def test_residency_to_level_fractions_filters_by_roof_levels() -> None:
    """Plus-key expansion targets only the canonical roof levels actually present."""
    assert _residency_to_level_fractions(
        {"l1": 0.2, "l3plus": 0.5}, roof_levels={"L1", "L2", "DRAM"}
    ) == {"L1": 0.2, "L2": 0.0, "L3": 0.0, "DRAM": 0.5}


def test_residency_to_level_fractions_default_expands_l3plus() -> None:
    """Without roof_levels, l3plus expands to L3 and DRAM in the canonical order."""
    assert _residency_to_level_fractions({"l3plus": 0.5}) == {"L1": 0.0, "L2": 0.0, "L3": 0.5, "DRAM": 0.5}


def test_residency_to_level_fractions_ignores_unknown_keys() -> None:
    """Unknown serialized keys are ignored; known levels keep their values."""
    assert _residency_to_level_fractions({"l1": 0.5, "bogus": 0.5}) == {"L1": 0.5, "L2": 0.0, "L3": 0.0, "DRAM": 0.0}


def test_residency_to_level_fractions_exact_legacy_keys_map_individually() -> None:
    """Legacy 4-key exact keys still map each level individually."""
    assert _residency_to_level_fractions({"l1": 0.6, "l2": 0.3, "l3": 0.08, "dram": 0.02}) == {
        "L1": 0.6,
        "L2": 0.3,
        "L3": 0.08,
        "DRAM": 0.02,
    }


def test_point_tooltip_omits_cache_residency_without_data() -> None:
    """Tooltip has no Cache Residency section when the point has no fractions."""
    rec = ApplicationRecord(id="r1", label="run1", aggregation="global", metadata={}, machine="m", points=[])
    p = ApplicationPoint(
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
    )
    tooltip = _format_point_tooltip(rec, p)
    assert "Cache Residency" not in tooltip


def _roofline_records_with_all_levels() -> list[BenchmarkRecord]:
    """Arithmetic plus L1/L2/L3/DRAM memory records for a single roof filter."""
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
        }
    ]
    for level, bw in (("L1", 400.0), ("L2", 100.0), ("L3", 60.0), ("DRAM", 30.0)):
        records.append(
            {
                "type": "memory",
                "name": f"{level} load",
                "isa": "test_isa",
                "machine": "test_machine",
                "data_type": "f32",
                "num_threads": 1,
                "timestamp": ts,
                "load_store_ratio": "2:1",
                "cache_level": level,
                "bandwidth_gbps": bw,
            }
        )
    return records


def _point_with_residency() -> ApplicationPoint:
    return ApplicationPoint(
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
        optional_fractions={"cache-residency": {"l1": 1.0, "l2": 0.5, "l3": 0.1, "dram": 0.0}},
    )


def _point_with_3key_residency() -> ApplicationPoint:
    return ApplicationPoint(
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
        optional_fractions={"cache-residency": {"l1": 0.2, "l2": 0.3, "l3plus": 0.5}},
    )


def _point_with_l2plus_residency() -> ApplicationPoint:
    return ApplicationPoint(
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
        optional_fractions={"cache-residency": {"l1": 0.2, "l2plus": 0.8}},
    )


def _memory_line(fig: Any, roof_id: str, level: str) -> Any:
    """The single memory ceiling trace of *roof_id* at *level* (located by dash)."""
    matches = [
        t
        for t in fig.data
        if t.mode == "lines" and t.legendgroup == roof_id and t.line.dash == _BW_LINE_STYLES[level]["dash"]
    ]
    assert len(matches) == 1, f"expected one {level} line for roof {roof_id}, got {len(matches)}"
    return matches[0]


def test_selection_emphasis_scales_selected_roof_lines() -> None:
    """Selection emphasis scales the selected roof's ceilings per level and dims other roofs."""
    records = _roofline_records_with_all_levels()
    roof1 = RoofConfig(
        roof_id="r1",
        label="Roof 1",
        isa="test_isa",
        machine="test_machine",
        num_threads=1,
        data_type="f32",
        compute_insts=["fma"],
        load_store_ratio="2:1",
        app_ids=["a1"],
    )
    roof2 = RoofConfig(
        roof_id="r2",
        label="Roof 2",
        isa="test_isa",
        machine="test_machine",
        num_threads=1,
        data_type="f32",
        compute_insts=["fma"],
        load_store_ratio="2:1",
        app_ids=["a2"],
    )
    apps = {
        "a1": ApplicationRecord(id="a1", label="app1", aggregation="global", metadata={}, machine="test_machine", points=[_point_with_residency()]),
        "a2": ApplicationRecord(id="a2", label="app2", aggregation="global", metadata={}, machine="test_machine", points=[_point_with_residency()]),
    }
    fig = build_roofline_figure(
        [roof1, roof2],
        records,
        apps,
        selected_roof_id="r1",
        selected_residency={"l1": 1.0, "l2": 0.5, "l3": 0.1, "dram": 0.0},
    )
    # Selected roof: each level scales with its own residency fraction.
    for level, frac in (("L1", 1.0), ("L2", 0.5), ("L3", 0.1), ("DRAM", 0.0)):
        line = _memory_line(fig, "r1", level)
        assert line.line.width == pytest.approx(1.5 * 1.5 * _residency_width_mult(frac))
        assert line.opacity == pytest.approx(_residency_alpha(frac))
    # Background roof: all levels at the fixed de-emphasis fraction.
    for level in ("L1", "L2", "L3", "DRAM"):
        line = _memory_line(fig, "r2", level)
        assert line.line.width == pytest.approx(2.25 * _residency_width_mult(0.1))
        assert line.opacity == pytest.approx(_residency_alpha(0.1))
    # Selected roof compute ceiling scales with the dominant level fraction (1.0).
    compute = [t for t in fig.data if t.mode == "lines" and t.legendgroup == "r1" and t.line.dash == "solid"]
    assert len(compute) == 1
    assert compute[0].line.width == pytest.approx(1.5 * 1.5 * 3.0)
    assert compute[0].opacity == pytest.approx(1.0)


def test_selection_emphasis_3key_l3plus_highlights_l3_and_dram() -> None:
    """3-bucket fractions: the l3plus value emphasizes the L3 AND DRAM ceilings equally."""
    records = _roofline_records_with_all_levels()
    roof1 = RoofConfig(
        roof_id="r1",
        label="Roof 1",
        isa="test_isa",
        machine="test_machine",
        num_threads=1,
        data_type="f32",
        compute_insts=["fma"],
        load_store_ratio="2:1",
        app_ids=["a1"],
    )
    roof2 = RoofConfig(
        roof_id="r2",
        label="Roof 2",
        isa="test_isa",
        machine="test_machine",
        num_threads=1,
        data_type="f32",
        compute_insts=["fma"],
        load_store_ratio="2:1",
        app_ids=["a2"],
    )
    apps = {
        "a1": ApplicationRecord(id="a1", label="app1", aggregation="global", metadata={}, machine="test_machine", points=[_point_with_3key_residency()]),
        "a2": ApplicationRecord(id="a2", label="app2", aggregation="global", metadata={}, machine="test_machine", points=[_point_with_3key_residency()]),
    }
    fig = build_roofline_figure(
        [roof1, roof2],
        records,
        apps,
        selected_roof_id="r1",
        selected_residency={"l1": 0.2, "l2": 0.3, "l3plus": 0.5},
    )
    # Selected roof: L1/L2 scale individually; L3 and DRAM share the l3plus fraction.
    for level, frac in (("L1", 0.2), ("L2", 0.3), ("L3", 0.5), ("DRAM", 0.5)):
        line = _memory_line(fig, "r1", level)
        assert line.line.width == pytest.approx(1.5 * 1.5 * _residency_width_mult(frac))
        assert line.opacity == pytest.approx(_residency_alpha(frac))
    # Background roof: all levels at the fixed de-emphasis fraction.
    for level in ("L1", "L2", "L3", "DRAM"):
        line = _memory_line(fig, "r2", level)
        assert line.line.width == pytest.approx(2.25 * _residency_width_mult(0.1))
        assert line.opacity == pytest.approx(_residency_alpha(0.1))
    # Selected roof compute ceiling scales with the dominant l3plus fraction (0.5).
    compute = [t for t in fig.data if t.mode == "lines" and t.legendgroup == "r1" and t.line.dash == "solid"]
    assert len(compute) == 1
    assert compute[0].line.width == pytest.approx(1.5 * 1.5 * _residency_width_mult(0.5))
    assert compute[0].opacity == pytest.approx(_residency_alpha(0.5))


def test_selection_emphasis_l2plus_expands_beyond_l2() -> None:
    """4-key schema with l2plus: L2, L3 AND DRAM ceilings share the l2plus fraction."""
    records = _roofline_records_with_all_levels()
    roof1 = RoofConfig(
        roof_id="r1",
        label="Roof 1",
        isa="test_isa",
        machine="test_machine",
        num_threads=1,
        data_type="f32",
        compute_insts=["fma"],
        load_store_ratio="2:1",
        app_ids=["a1"],
    )
    roof2 = RoofConfig(
        roof_id="r2",
        label="Roof 2",
        isa="test_isa",
        machine="test_machine",
        num_threads=1,
        data_type="f32",
        compute_insts=["fma"],
        load_store_ratio="2:1",
        app_ids=["a2"],
    )
    apps = {
        "a1": ApplicationRecord(id="a1", label="app1", aggregation="global", metadata={}, machine="test_machine", points=[_point_with_l2plus_residency()]),
        "a2": ApplicationRecord(id="a2", label="app2", aggregation="global", metadata={}, machine="test_machine", points=[_point_with_l2plus_residency()]),
    }
    fig = build_roofline_figure(
        [roof1, roof2],
        records,
        apps,
        selected_roof_id="r1",
        selected_residency={"l1": 0.2, "l2plus": 0.8},
    )
    # Selected roof: L1 scales individually; L2, L3 and DRAM share the l2plus fraction.
    for level, frac in (("L1", 0.2), ("L2", 0.8), ("L3", 0.8), ("DRAM", 0.8)):
        line = _memory_line(fig, "r1", level)
        assert line.line.width == pytest.approx(1.5 * 1.5 * _residency_width_mult(frac))
        assert line.opacity == pytest.approx(_residency_alpha(frac))
    # Background roof: all levels at the fixed de-emphasis fraction.
    for level in ("L1", "L2", "L3", "DRAM"):
        line = _memory_line(fig, "r2", level)
        assert line.line.width == pytest.approx(2.25 * _residency_width_mult(0.1))
        assert line.opacity == pytest.approx(_residency_alpha(0.1))
    # Selected roof compute ceiling scales with the dominant l2plus fraction (0.8).
    compute = [t for t in fig.data if t.mode == "lines" and t.legendgroup == "r1" and t.line.dash == "solid"]
    assert len(compute) == 1
    assert compute[0].line.width == pytest.approx(1.5 * 1.5 * _residency_width_mult(0.8))
    assert compute[0].opacity == pytest.approx(_residency_alpha(0.8))


def test_selection_emphasis_defaults_without_selection() -> None:
    """No selection info (or empty/stale residency) keeps default ceiling styles: no opacity, unscaled width."""
    records = _roofline_records_with_all_levels()
    roof = RoofConfig(
        roof_id="r1",
        label="Roof 1",
        isa="test_isa",
        machine="test_machine",
        num_threads=1,
        data_type="f32",
        compute_insts=["fma"],
        load_store_ratio="2:1",
    )
    fig = build_roofline_figure([roof], records)
    line = _memory_line(fig, "r1", "L1")
    assert line.line.width == pytest.approx(1.5 * 1.5)
    assert line.opacity is None
    compute = [t for t in fig.data if t.mode == "lines" and t.legendgroup == "r1" and t.line.dash == "solid"]
    assert len(compute) == 1
    assert compute[0].line.width == pytest.approx(1.5 * 1.5)
    assert compute[0].opacity is None
    # selected_roof_id=None with an empty residency dict also falls back to defaults.
    fig2 = build_roofline_figure([roof], records, selected_roof_id=None, selected_residency={})
    line2 = _memory_line(fig2, "r1", "L1")
    assert line2.line.width == pytest.approx(1.5 * 1.5)
    assert line2.opacity is None
    # A stale roof id (no longer in the roof list) also falls back to defaults.
    fig3 = build_roofline_figure(
        [roof],
        records,
        selected_roof_id="gone",
        selected_residency={"l1": 1.0, "l2": 0.5, "l3": 0.1, "dram": 0.0},
    )
    line3 = _memory_line(fig3, "r1", "L1")
    assert line3.line.width == pytest.approx(1.5 * 1.5)
    assert line3.opacity is None
    # A stale roof id with a 3-key residency dict also falls back to defaults.
    fig4 = build_roofline_figure(
        [roof],
        records,
        selected_roof_id="gone",
        selected_residency={"l1": 0.2, "l2": 0.3, "l3plus": 0.5},
    )
    line4 = _memory_line(fig4, "r1", "L1")
    assert line4.line.width == pytest.approx(1.5 * 1.5)
    assert line4.opacity is None


def _fill_alpha(fillcolor: str) -> float:
    """Extract the alpha channel from an ``rgba(r,g,b,a)`` fill color string."""
    match = re.search(r"rgba\(\s*\d+,\s*\d+,\s*\d+,\s*([\d.]+)\)", fillcolor)
    assert match, f"unexpected fillcolor {fillcolor!r}"
    return float(match.group(1))


def _fill_bands(fig: Any, roof_id: str) -> list[Any]:
    """The roof's level fill bands, ordered L1..DRAM by ridge x position."""
    fills = [t for t in fig.data if t.fill == "toself" and t.legendgroup == roof_id and t.mode == "lines"]
    fills.sort(key=lambda t: t.x[1])
    assert len(fills) == 4, f"expected 4 fill bands for roof {roof_id}, got {len(fills)}"
    return fills


def test_selection_fills_share_constant_base_opacity() -> None:
    """Selection: every roof's level bands use one constant base fill transparency,
    with the residency alpha applied on top; no selection keeps the default per-level
    opacities."""
    records = _roofline_records_with_all_levels()
    roof1 = RoofConfig(
        roof_id="r1",
        label="Roof 1",
        isa="test_isa",
        machine="test_machine",
        num_threads=1,
        data_type="f32",
        compute_insts=["fma"],
        load_store_ratio="2:1",
        app_ids=["a1"],
    )
    roof2 = RoofConfig(
        roof_id="r2",
        label="Roof 2",
        isa="test_isa",
        machine="test_machine",
        num_threads=1,
        data_type="f32",
        compute_insts=["fma"],
        load_store_ratio="2:1",
        app_ids=["a2"],
    )
    apps = {
        "a1": ApplicationRecord(id="a1", label="app1", aggregation="global", metadata={}, machine="test_machine", points=[_point_with_residency()]),
        "a2": ApplicationRecord(id="a2", label="app2", aggregation="global", metadata={}, machine="test_machine", points=[_point_with_residency()]),
    }
    fig = build_roofline_figure(
        [roof1, roof2],
        records,
        apps,
        selected_roof_id="r1",
        selected_residency={"l1": 1.0, "l2": 0.5, "l3": 0.1, "dram": 0.0},
    )
    # Selected roof: constant base * residency alpha per level (no per-level base).
    for band, frac in zip(_fill_bands(fig, "r1"), (1.0, 0.5, 0.1, 0.0)):
        assert _fill_alpha(band.fillcolor) == pytest.approx(_SELECTED_FILL_BASE_OPACITY * _residency_alpha(frac))
    # Background roof: same constant base, all levels at the de-emphasis fraction.
    for band in _fill_bands(fig, "r2"):
        assert _fill_alpha(band.fillcolor) == pytest.approx(_SELECTED_FILL_BASE_OPACITY * _residency_alpha(0.1))
    # Without a selection the default per-level opacities are kept.
    fig2 = build_roofline_figure([roof1], records)
    for band, level in zip(_fill_bands(fig2, "r1"), ("L1", "L2", "L3", "DRAM")):
        assert _fill_alpha(band.fillcolor) == pytest.approx(_BW_FILL_OPACITIES[level])
