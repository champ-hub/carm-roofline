from __future__ import annotations

import math
import re
import uuid
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any, TypedDict

import plotly.graph_objects as go

from carm_roofline.core.units import Bandwidth, Bytes, Frequency, Operations, Performance, Seconds
from carm_roofline.output_utils import debug, warn
from carm_roofline.roofline_assembly import (
    ApplicationPoint,
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
FREQUENCY_OPTIONS = [(str(Frequency(hz)), str(hz)) for hz in [2500000000, 3000000000, 3200000000, 4000000000]]


# Data models


def _machine_display_name(machine: str) -> str:
    """Strip the hash suffix from a machine name for display.

    Machine names are generated as ``"<short_model>_<config_hash>"`` (e.g.
    ``"Ryzen-7-7735HS_59486dd1"``).  This extracts the short model name
    without the 8-hex-digit hash suffix.
    """
    return re.sub(r"_[0-9a-f]{8}$", "", machine)


def format_roof_label(
    machine: str | None,
    isa: str | None,
    num_threads: int | None,
    data_type: str | None,
) -> str:
    """Generate a descriptive label for a roof configuration.

    Format: ``"<machine> (<isa>, <N> threads, <data_type>)"``
    with each optional component omitted when ``None``.
    """
    display_machine = _machine_display_name(machine) if machine else "Unknown"
    details: list[str] = []
    if isa:
        details.append(isa)
    if num_threads is not None:
        details.append(f"{num_threads} threads")
    if data_type:
        details.append(data_type)
    if details:
        return f"{display_machine} ({', '.join(details)})"
    return display_machine


@dataclass
class RoofConfig:
    id: str = ""
    label: str = ""
    machine: str | None = "Machine A"
    isa: str | None = None
    num_threads: int | None = None
    actual_frequency_hz: int | None = None
    compute_insts: list[str] = field(default_factory=lambda: ["fma", "add"])
    load_store_ratio: str | None = None
    collapsed: bool = False
    advanced_collapsed: bool = True
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
        actual_frequency_hz: int | None = None,
        load_store_ratio: str | None = None,
        app_ids: list[str] | None = None,
        collapsed: bool = False,
        advanced_collapsed: bool = True,
    ) -> None:
        self.id = roof_id or uuid.uuid4().hex
        self.label = label or format_roof_label(machine, isa, num_threads, data_type)
        self.collapsed = collapsed
        self.advanced_collapsed = advanced_collapsed
        self.machine = machine
        self.isa = isa
        self.num_threads = num_threads
        self.actual_frequency_hz = actual_frequency_hz
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
        self.settings: GUISettings = GUISettings()

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
                    "actual_frequency_hz": r.actual_frequency_hz,
                    "data_type": r.data_type,
                    "compute_insts": r.compute_insts,
                    "load_store_ratio": r.load_store_ratio,
                    "collapsed": r.collapsed,
                    "advanced_collapsed": r.advanced_collapsed,
                    "app_ids": r.app_ids,
                }
                for r in self.roofs
            ],
            "active_panel": self.active_panel,
            "settings": asdict(self.settings),
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
                actual_frequency_hz=r.get("actual_frequency_hz"),
                data_type=r.get("data_type", "f32"),
                compute_insts=r.get("compute_insts", ["fma", "add"]),
                load_store_ratio=r.get("load_store_ratio", "2:1"),
                advanced_collapsed=r.get("advanced_collapsed", True),
                collapsed=r.get("collapsed", False),
                app_ids=r.get("app_ids", []),
            )
            for r in data.get("roofs", [])
        ]
        store.active_panel = ActivePanel(data.get("active_panel", "carm_view"))
        store.settings = GUISettings.from_dict(data.get("settings", {}))
        return store


@dataclass
class GUISettings:
    """Persistent user preferences for the GUI."""

    normalize_by_threads: bool = False
    marker_scale_factor: float = 50.0
    power2_ticks: bool = False
    show_roof_fills: bool = True
    line_width: float = 1.5
    axis_label_font_size: int = 14
    axis_tick_font_size: int = 12
    tooltip_font_size: int = 12
    legend_font_size: int = 10

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GUISettings:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


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

_BW_LINE_STYLES: dict[str, dict[str, Any]] = {
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


# Distinct marker symbols assigned to each application under the same roof
_APP_MARKER_SYMBOLS: tuple[str, ...] = (
    "circle",
    "diamond",
    "square",
    "cross",
    "x",
    "star",
    "pentagon",
    "hexagram",
    "triangle-up",
    "triangle-down",
    "hourglass",
    "bowtie",
    "asterisk",
    "hash",
)

MIN_MARKER_SIZE: float = 50.0


def _format_point_tooltip(rec: ApplicationRecord, p: ApplicationPoint) -> str:
    """Format a rich tooltip string for an application roofline point."""
    parts = [
        f"<b>{rec.label}</b>",
        f"<i>{p.label}</i>",
        "<b>Performance</b>",
        f"  Arithmetic Intensity: {p.arithmetic_intensity:.3f} OPS/Byte",
        f"  Performance: {Performance(p.flops_per_second)!s}",
        f"  Bandwidth: {Bandwidth(p.bandwidth)!s}",
        "<b>Execution</b>",
        f"  {p.num_threads} thread(s), {p.num_ranks} rank(s)",
        f"  Duration: {Seconds(p.runtime_s)!s}",
        "<b>Work</b>",
        f"  Total FLOPs: {Operations(int(p.total_flops))!s}",
        f"  Total Bytes: {Bytes(int(p.total_bytes))!s}",
        f"  Regions: {p.num_regions}",
    ]
    return "<br>".join(parts)


def build_roofline_figure(
    roofs: list[RoofConfig],
    records: list[BenchmarkRecord],
    applications_by_id: dict[str, ApplicationRecord] | None = None,
    settings: GUISettings | None = None,
) -> go.Figure:
    """Build a Plotly roofline figure from real benchmark records.

    For each roof config the records are filtered and a roofline model
    assembled; memory bandwidth ceilings, compute-performance ceilings,
    ridge-point markers, and optional application run points are drawn.
    Roofs with no matching data produce a warning annotation instead.
    """
    s = settings or GUISettings()
    normalize_by_threads = s.normalize_by_threads
    marker_scale_factor = s.marker_scale_factor
    power2_ticks = s.power2_ticks
    line_width = s.line_width
    axis_label_font_size = s.axis_label_font_size
    axis_tick_font_size = s.axis_tick_font_size
    tooltip_font_size = s.tooltip_font_size
    legend_font_size = s.legend_font_size
    show_roof_fills = s.show_roof_fills
    fig = go.Figure()
    LOG10_2 = math.log10(2)

    def _power2_tick_config(
        log10_range: list[float],
    ) -> dict[str, object] | None:
        """Build array tick config for 2^N formatting, or None if the range is empty."""
        lo, hi = log10_range
        min_exp = math.ceil(lo / LOG10_2)
        max_exp = math.floor(hi / LOG10_2)
        if min_exp > max_exp:
            return None
        exponents = list(range(min_exp, max_exp + 1))
        tickvals = [10.0 ** (e * LOG10_2) for e in exponents]
        ticktext = [f"2<sup>{e}</sup>" for e in exponents]
        return {
            "tickmode": "array",
            "tickvals": tickvals,
            "ticktext": ticktext,
        }

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
            actual_frequency_hz=roof.actual_frequency_hz,
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

    # Assign a distinct marker symbol per application, consistent across roofs.
    app_symbol: dict[str, str] = {}
    if applications_by_id:
        for roof in roofs:
            for app_id in roof.app_ids:
                if app_id not in app_symbol:
                    app_symbol[app_id] = _APP_MARKER_SYMBOLS[len(app_symbol) % len(_APP_MARKER_SYMBOLS)]

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
                            "symbol": app_symbol[app_id],
                            "size": marker_sizes,
                            "sizemode": "area",
                            "opacity": 0.6,
                        },
                        text=[_format_point_tooltip(rec, p) for p in rec.points],
                        hovertemplate="%{text}<extra></extra>",
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
            style = dict(_BW_LINE_STYLES.get(level, {"dash": "solid", "width": 1}))
            style["width"] = style["width"] * line_width
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

            if show_roof_fills:
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
                        legendgroup=roof.id,
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
            line_style = (
                {"dash": "solid", "width": 1.5 * line_width} if is_top else {"dash": "dot", "width": 2 * line_width}
            )
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
                                f"<b>{roof.label}</b><br>"
                                f"{op_name} x {level}<br>"
                                f"<b>Ridge Point</b><br>"
                                f"  Arithmetic Intensity: {ridge_ai:.2f} OPS/Byte<br>"
                                f"  Performance: {gops:.1f} GOPS/s<br>"
                                f"  Bandwidth: {bw / roof_divisor!s}"
                            ),
                            showlegend=False,
                        )
                    )
    # Apply 2^N tick formatting when enabled
    xaxis: dict[str, object] = {
        "title": "Arithmetic Intensity (OPS/Byte)",
        "type": "log",
        "dtick": LOG10_2,
        "tick0": 0,
        "exponentformat": "none",
        "gridcolor": "lightgray",
        "range": x_range,
    }
    yaxis: dict[str, object] = {
        "title": "Performance (GOPS/s)",
        "type": "log",
        "dtick": LOG10_2,
        "tick0": 0,
        "exponentformat": "none",
        "gridcolor": "lightgray",
        "range": y_range,
    }
    if power2_ticks:
        xticks = _power2_tick_config(x_range)
        yticks = _power2_tick_config(y_range)
        if xticks is not None:
            xaxis.update(xticks)
        if yticks is not None:
            yaxis.update(yticks)

    xaxis["title_font"] = {"size": axis_label_font_size}
    yaxis["title_font"] = {"size": axis_label_font_size}
    xaxis["tickfont"] = {"size": axis_tick_font_size}
    yaxis["tickfont"] = {"size": axis_tick_font_size}

    fig.update_layout(
        template="plotly_white",
        xaxis=xaxis,
        yaxis=yaxis,
        uirevision="roofline-plot",
        hovermode="closest",
        dragmode="zoom",
        hoverlabel={"font": {"size": tooltip_font_size}},
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 0.95,
            "xanchor": "left",
            "x": 1.02,
            "font": {"size": legend_font_size},
        },
    )

    return fig
