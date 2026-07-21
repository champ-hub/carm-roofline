from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypedDict

import plotly.graph_objects as go

from core.units import Seconds
from output_utils import debug, warn
from roofline_assembly import (
    ApplicationRecord,
    AssembledRoofline,
    BenchmarkRecord,
    FilterOptions,
    RooflineFilter,
    assemble_roofline,
)


class DropdownOption(TypedDict):
    """A label-value pair for application dropdown options."""

    label: str
    value: str


class ActivePanel(str, Enum):
    CARM_VIEW = "carm_view"
    SETTINGS = "settings"


# Data-derived options (populated from loaded records via discover_filter_options)
# Fallback static defaults so component builders always have something.
DATA_TYPE_OPTIONS = ["f32", "f64", "i8", "i16", "i32", "i64"]
COMPUTE_INST_OPTIONS = ["fma", "add", "mul", "div"]


# Data models


@dataclass
class RoofConfig:
    id: str = ""
    label: str = ""
    machine: str | None = "Machine A"
    isa: str | None = None
    num_threads: int | None = None
    data_type: str | None = None
    compute_insts: list[str] = field(default_factory=lambda: ["fma", "add"])
    load_store_ratio: str | None = None
    collapsed: bool = False
    app_ids: list[str] = field(default_factory=list)

    def __init__(
        self,
        roof_id: str = "",
        label: str = "",
        machine: str | None = "Machine A",
        isa: str | None = None,
        num_threads: int | None = None,
        data_type: str | None = None,
        compute_insts: list[str] | None = None,
        load_store_ratio: str | None = None,
        app_ids: list[str] | None = None,
        collapsed: bool = False,
    ) -> None:
        self.id = roof_id or uuid.uuid4().hex
        self.label = label or f"Roof {self.id[:6]}"
        self.collapsed = collapsed
        self.machine = machine
        self.isa = isa
        self.num_threads = num_threads
        self.data_type = data_type
        self.compute_insts = compute_insts or ["fma", "add"]
        self.load_store_ratio = load_store_ratio
        self.app_ids = app_ids or []


def make_default_roof(opts: FilterOptions | None = None) -> RoofConfig:
    """Create a RoofConfig with defaults derived from available data options.

    Only *machine* is pinned to the first available option (or a fallback);
    all other user-facing filter fields start as ``None`` so the auto-resolution
    in ``_resolve_roof_data`` picks the broadest viable value from the data.
    """
    machine = opts["machine"][0] if opts and opts["machine"] else "Machine A"
    return RoofConfig(
        machine=machine,
        isa=None,
        num_threads=None,
        data_type=None,
        compute_insts=["fma", "add"],
        load_store_ratio=None,
    )


class RoofStore:
    """In-memory state store for the GUI."""

    def __init__(self, roof_template: RoofConfig | None = None) -> None:
        self.roofs: list[RoofConfig] = [roof_template or RoofConfig()]
        self.active_panel: ActivePanel = ActivePanel.CARM_VIEW
        self.normalize_by_threads: bool = False
        self.marker_scale_factor: float = 50.0

    # Roof CRUD
    def add_roof(self, roof_template: RoofConfig | None = None) -> RoofConfig:
        roof = roof_template or RoofConfig()
        self.roofs.append(roof)
        return roof

    def remove_roof(self, roof_id: str) -> None:
        before = len(self.roofs)
        self.roofs = [r for r in self.roofs if r.id != roof_id]
        if len(self.roofs) == before:
            msg = f"Roof {roof_id} not found"
            raise KeyError(msg)

    def update_roof(self, roof_id: str, **kwargs: object) -> None:
        for roof in self.roofs:
            if roof.id == roof_id:
                for key, val in kwargs.items():
                    if hasattr(roof, key):
                        setattr(roof, key, val)
                return
        msg = f"Roof {roof_id} not found"
        raise KeyError(msg)

    # Serialisation helpers

    def to_dict(self) -> dict[str, object]:
        return {
            "roofs": [
                {
                    "id": r.id,
                    "label": r.label,
                    "machine": r.machine,
                    "isa": r.isa,
                    "num_threads": r.num_threads,
                    "data_type": r.data_type,
                    "compute_insts": r.compute_insts,
                    "load_store_ratio": r.load_store_ratio,
                    "collapsed": r.collapsed,
                    "app_ids": r.app_ids,
                }
                for r in self.roofs
            ],
            "active_panel": self.active_panel,
            "normalize_by_threads": self.normalize_by_threads,
            "marker_scale_factor": self.marker_scale_factor,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RoofStore:
        store = cls.__new__(cls)
        store.roofs = [
            RoofConfig(
                roof_id=r["id"],
                label=r["label"],
                machine=r["machine"],
                isa=r["isa"],
                num_threads=r.get("num_threads", r.get("threads")),
                data_type=r.get("data_type", "f32"),
                compute_insts=r.get("compute_insts", ["fma", "add"]),
                load_store_ratio=r.get("load_store_ratio", "2:1"),
                collapsed=r.get("collapsed", False),
                app_ids=r.get("app_ids", []),
            )
            for r in data.get("roofs", [])
        ]
        store.active_panel = ActivePanel(data.get("active_panel", "carm_view"))
        store.normalize_by_threads = data.get("normalize_by_threads", False)
        store.marker_scale_factor = data.get("marker_scale_factor", 50.0)
        return store


# Plot data builder

_COLORS = [
    "#636efa",
    "#ef553b",
    "#00cc96",
    "#ab63fa",
    "#ffa15a",
    "#19d3f3",
    "#ff6692",
    "#b6e880",
    "#ff97ff",
    "#fecb52",
]

_BW_LINE_STYLES: dict[str, dict[str, object]] = {
    "L1": {"dash": "10px 0px", "width": 1.5},
    "L2": {"dash": "12px 4px", "width": 1.5},
    "L3": {"dash": "6px 2px", "width": 1.5},
    "DRAM": {"dash": "2px 2px", "width": 1.5},
}

_BW_FILL_OPACITIES: dict[str, float] = {
    "L1": 0.30,
    "L2": 0.20,
    "L3": 0.10,
    "DRAM": 0.05,
}


MIN_MARKER_SIZE: float = 50.0


def build_roofline_figure(
    roofs: list[RoofConfig],
    records: list[BenchmarkRecord],
    applications_by_id: dict[str, ApplicationRecord] | None = None,
    normalize_by_threads: bool = False,
    marker_scale_factor: float = 50.0,
) -> go.Figure:
    """Build a Plotly roofline figure from real benchmark records.

    For each roof config the records are filtered and a roofline model
    assembled; memory bandwidth ceilings, compute-performance ceilings,
    ridge-point markers, and optional application run points are drawn.
    Roofs with no matching data produce a warning annotation instead.
    """
    fig = go.Figure()
    LOG10_2 = math.log10(2)

    # Pre-pass: assemble every roofline model once and collect the data
    # extents that define the axis ranges.
    models: list[AssembledRoofline] = []
    ridge_pairs: list[tuple[float, float]] = []
    peak_perf_values: list[float] = []

    for roof in roofs:
        roof_divisor = roof.num_threads if (normalize_by_threads and roof.num_threads and roof.num_threads > 0) else 1
        flt = RooflineFilter(
            machine=roof.machine if roof.machine else None,
            isa=roof.isa if roof.isa else None,
            num_threads=roof.num_threads,
            data_type=roof.data_type if roof.data_type else None,
            operations=frozenset(roof.compute_insts) if roof.compute_insts else None,
            load_store_ratio=roof.load_store_ratio if roof.load_store_ratio else None,
        )
        model = assemble_roofline(records, flt)
        models.append(model)
        debug(
            f"Roof '{roof.label}': {len(model.bandwidth_by_level)} level(s), "
            f"{len(model.peak_performance_by_op)} operation(s) from "
            f"{len(model.source_timestamps)} run(s)"
        )
        if model.peak_performance_by_op:
            peak_perf_values.append(max(p.value for p in model.peak_performance_by_op.values()) / roof_divisor)
        rp = model.ridge_points()
        for level, ai_obj in rp.items():
            bw = model.bandwidth_by_level.get(level)
            if bw is not None:
                ridge_pairs.append((ai_obj.value, bw.value / roof_divisor))

    # Axis ranges, log10 coordinates (Plotly "log" axis convention).
    if ridge_pairs:
        ridge_ai_list = [ai for ai, _bw in ridge_pairs]
        x_min_data = min(ridge_ai_list) / 16.0
        x_max_data = max(ridge_ai_list) * 4.0
        max_ai = max(ridge_ai_list)
        rightmost_bw = max(bw for ai, bw in ridge_pairs if ai == max_ai)
        y_min_gops = rightmost_bw * x_min_data / 1e9
        x_range = [math.log10(x_min_data), math.log10(x_max_data)]
    else:
        x_range = [-2.0, 2.0]
        y_min_gops = 1e-3

    y_max_gops = max(peak_perf_values) / 1000000000.0 * 2.0 if peak_perf_values else 1000.0

    y_range = [math.log10(y_min_gops), math.log10(y_max_gops)] if ridge_pairs else [0.0, 3.5]

    # Compute runtime range for marker-size normalization
    runtime_min = float("inf")
    runtime_max = float("-inf")
    if applications_by_id:
        for roof in roofs:
            for app_id in roof.app_ids:
                rec = applications_by_id.get(app_id)
                if rec and rec.points:
                    for p in rec.points:
                        if p.runtime_s < runtime_min:
                            runtime_min = p.runtime_s
                        if p.runtime_s > runtime_max:
                            runtime_max = p.runtime_s
    if runtime_min == float("inf"):
        runtime_min = runtime_max = 0.0
    runtime_range = runtime_max - runtime_min

    for idx, (roof, model) in enumerate(zip(roofs, models)):
        color = _COLORS[idx % len(_COLORS)]
        roof_divisor = roof.num_threads if (normalize_by_threads and roof.num_threads and roof.num_threads > 0) else 1

        # Application points (drawn even when no ceiling data)
        if applications_by_id:
            for app_id in roof.app_ids:
                rec = applications_by_id.get(app_id)
                if rec is None or not rec.points:
                    continue
                marker_sizes = [
                    max(
                        MIN_MARKER_SIZE,
                        MIN_MARKER_SIZE
                        + ((p.runtime_s - runtime_min) / runtime_range) * MIN_MARKER_SIZE * marker_scale_factor,
                    )
                    if runtime_range > 0
                    else MIN_MARKER_SIZE
                    for p in rec.points
                ]
                duration_strings = [str(Seconds(p.runtime_s)) for p in rec.points]
                fig.add_trace(
                    go.Scatter(
                        x=[p.arithmetic_intensity for p in rec.points],
                        y=[
                            p.flops_per_second
                            / (p.num_threads if normalize_by_threads and p.num_threads and p.num_threads > 0 else 1)
                            / 1e9
                            for p in rec.points
                        ],
                        mode="markers",
                        name=rec.label,
                        legendgroup=roof.id,
                        showlegend=True,
                        marker={
                            "color": color,
                            "symbol": "circle",
                            "size": marker_sizes,
                            "sizemode": "area",
                            "opacity": 0.6,
                        },
                        text=[p.label for p in rec.points],
                        customdata=list(
                            zip(
                                [p.num_threads for p in rec.points],
                                duration_strings,
                            )
                        ),
                        hovertemplate=(
                            f"{rec.label}<br>%{{text}}<br>"
                            f"AI=%{{x:.3f}} OPS/Byte<br>"
                            f"Perf=%{{y:.1f}} GOPS/s<br>"
                            f"Threads=%{{customdata[0]}}<br>"
                            f"Duration=%{{customdata[1]}}"
                            "<extra></extra>"
                        ),
                    )
                )

        has_bw = bool(model.bandwidth_by_level)
        has_perf = bool(model.peak_performance_by_op)

        if not has_bw and not has_perf:
            warn(
                f"No matching data for roof '{roof.label}' "
                f"(machine={roof.machine}, isa={roof.isa}, num_threads={roof.num_threads}, "
                f"data_type={roof.data_type}, ratio={roof.load_store_ratio})"
            )
            fig.add_annotation(
                text=f"{roof.label}: no matching benchmark data",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font={"color": color, "size": 12},
            )
            continue
        if has_bw and not has_perf:
            warn(
                f"Incomplete roofline for '{roof.label}': memory data found but no matching "
                f"arithmetic benchmarks (machine={roof.machine}, isa={roof.isa}, "
                f"num_threads={roof.num_threads}, data_type={roof.data_type}, ratio={roof.load_store_ratio})"
            )
            fig.add_annotation(
                text=f"{roof.label}: no compute-performance data for these filters",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.35,
                showarrow=False,
                font={"color": color, "size": 12, "style": "italic"},
            )

        peak_perf_raw = max(p.value for p in model.peak_performance_by_op.values()) if has_perf else 0.0
        peak_perf = peak_perf_raw / roof_divisor

        # Pre-compute bandwidth line segments for levels that exist
        segments: list[tuple[str | None, float, float, float, float, Any, Any]] = []
        for level in ("L1", "L2", "L3", "DRAM"):
            bw = model.bandwidth_by_level.get(level)
            if bw is None:
                continue
            bw_norm = bw / roof_divisor
            ai_left = 1e-6
            y_left = bw_norm.value * ai_left / 1e9
            if peak_perf > 0:
                ridge_ai = peak_perf_raw / bw.value
                ai_right = ridge_ai
                y_right = peak_perf / 1e9
            else:
                ai_right = 1e6
                y_right = bw_norm.value * ai_right / 1e9
            style = _BW_LINE_STYLES.get(level, {"dash": "solid", "width": 1})
            segments.append((level, ai_left, ai_right, y_left, y_right, bw_norm, style))
        # Append synthetic extension anchor so every real segment has a
        # "next" to pair with in a single fill loop.
        segments.append((None, 1e6, 1e6, y_min_gops, y_min_gops, None, None))

        # Shaded fills: each level draws a band toward the next level
        for i, seg in enumerate(segments):
            _level, c_al, c_ar, c_yl, c_yr = seg[:5]
            if _level is None:  # reached the synthetic anchor
                break
            level = _level
            next_seg = segments[i + 1]
            is_ext = next_seg[0] is None

            if is_ext:
                # Lowest level: fill from DRAM bandwidth left, along roof to ridge,
                # then right, down, and back to bandwidth left
                x_pts = [c_al, c_ar, 1e6, 1e6]
                y_pts = [c_yl, c_yr, c_yr, c_yl]
            else:
                _, n_al, n_ar, n_yl, n_yr = next_seg[:5]
                x_pts = [c_al, c_ar, n_ar, n_al]
                y_pts = [c_yl, c_yr, n_yr, n_yl]

            opacity = _BW_FILL_OPACITIES.get(level, 0.1)
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            fill_color = f"rgba({r},{g},{b},{opacity})"

            fig.add_trace(
                go.Scatter(
                    x=x_pts,
                    y=y_pts,
                    mode="lines",
                    fill="toself",
                    fillcolor=fill_color,
                    line={"width": 0},
                    showlegend=False,
                    hoverinfo="skip",
                    name="",
                )
            )

        _first = True
        for seg in segments:
            _level, ai_left, ai_right, y_left, y_right, _bw_norm, line_style = seg
            if _level is None:  # skip synthetic anchor
                continue
            level = _level
            fig.add_trace(
                go.Scatter(
                    x=[ai_left, ai_right],
                    y=[y_left, y_right],
                    mode="lines",
                    name=roof.label if _first else f"{roof.label} {level} ({_bw_norm!s})",
                    legendgroup=roof.id,
                    line={"color": color, **line_style},
                    hoverinfo="skip",
                    showlegend=_first,
                )
            )
            _first = False

        # Compute-performance ceilings (single segment: left-most ridge → right edge)
        for op_name, perf in model.peak_performance_by_op.items():
            perf_norm = perf / roof_divisor
            gops = perf_norm.value / 1e9
            if model.bandwidth_by_level:
                op_ridge_ai = min(perf.value / bw.value for bw in model.bandwidth_by_level.values())
                compute_x_start = op_ridge_ai
            else:
                compute_x_start = 1e-6
            is_top = perf.value == peak_perf_raw
            line_style = {"dash": "solid", "width": 1.5} if is_top else {"dash": "dot", "width": 2}
            fig.add_trace(
                go.Scatter(
                    x=[compute_x_start, 1e6],
                    y=[gops, gops],
                    mode="lines",
                    name=f"{roof.label} {op_name} ({perf_norm!s})",
                    legendgroup=roof.id,
                    line={"color": color, **line_style},
                    hoverinfo="skip",
                    showlegend=_first,
                )
            )
            _first = False

        # Invisible ridge-point hover markers (one per op x cache level)
        if model.bandwidth_by_level and model.peak_performance_by_op:
            for op_name, perf in model.peak_performance_by_op.items():
                gops = (perf / roof_divisor).value / 1e9
                for level, bw in model.bandwidth_by_level.items():
                    ridge_ai = perf.value / bw.value
                    fig.add_trace(
                        go.Scatter(
                            x=[ridge_ai],
                            y=[gops],
                            mode="markers",
                            marker={"opacity": 0, "color": color},
                            hoverinfo="text",
                            hovertext=(
                                f"{roof.label} {op_name} x {level}<br>"
                                f"AI = {ridge_ai:.2f} OPS/Byte<br>"
                                f"Performance = {gops:.1f} GOPS/s<br>"
                                f"BW = {bw / roof_divisor!s}"
                            ),
                            showlegend=False,
                        )
                    )
    fig.update_layout(
        template="plotly_white",
        xaxis={
            "title": "Arithmetic Intensity (OPS/Byte)",
            "type": "log",
            "dtick": LOG10_2,
            "tick0": 0,
            "exponentformat": "none",
            "gridcolor": "lightgray",
            "range": x_range,
        },
        yaxis={
            "title": "Performance (GOPS/s)",
            "type": "log",
            "dtick": LOG10_2,
            "tick0": 0,
            "exponentformat": "none",
            "gridcolor": "lightgray",
            "range": y_range,
        },
        uirevision="roofline-plot",
        hovermode="closest",
        dragmode="zoom",
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 0.95,
            "xanchor": "left",
            "x": 1.02,
            "font": {"size": 10},
        },
    )

    return fig
