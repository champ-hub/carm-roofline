from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import dash
import plotly.graph_objects as go
from dash import ALL, Input, Output, Patch, State, callback_context

from carm_roofline.core.units import Performance
from carm_roofline.gui.callbacks.helpers import filter_app_options_for_roofs, selection_payload
from carm_roofline.gui.callbacks.resolution import resolve_roofs
from carm_roofline.gui.callbacks.state import CallbackState
from carm_roofline.gui.components import (
    _build_guide_panels,
    build_carm_view_panel,
    build_export_panel,
    build_settings_panel,
)
from carm_roofline.gui.data import build_paraver_figure, build_roofline_figure
from carm_roofline.gui.ids import BenchmarkPanelID, PlotAreaID, ProfilePanelID, RoofCardID, SidebarID, StoreID
from carm_roofline.gui.providers import BenchmarkAppsProvider, filter_trace
from carm_roofline.roofline_assembly import discover_filter_options, load_all_benchmarks

_GUIDE_TRACE_ROLE = "point-guide"
_GUIDE_TRACE_KEYS: tuple[tuple[str, str], ...] = (
    ("hover", "line"),
    ("hover", "intersections"),
    ("selected", "line"),
    ("selected", "intersections"),
)


def _guide_point(figure: dict[str, Any], identity: dict[str, Any] | None) -> tuple[float, float] | None:
    if not identity:
        return None
    for trace in figure.get("data", []):
        meta = trace.get("meta")
        if (
            not isinstance(meta, dict)
            or meta.get("carm_role") != "application"
            or meta.get("roof_id") != identity.get("roof_id")
            or meta.get("app_id") != identity.get("app_id")
            or trace.get("visible") is False
            or trace.get("visible") == "legendonly"
        ):
            continue
        index = identity.get("point_index")
        xs, ys = trace.get("x", []), trace.get("y", [])
        if not isinstance(index, int) or isinstance(index, bool) or index < 0 or index >= min(len(xs), len(ys)):
            return None
        try:
            x, y = float(xs[index]), float(ys[index])
        except (TypeError, ValueError):
            return None
        return (x, y) if math.isfinite(x) and math.isfinite(y) and x > 0 and y > 0 else None
    return None


def _roof_intersections(figure: dict[str, Any], intensity: float) -> list[float]:
    """Return unique visible roof performances intersecting the guide intensity."""
    log_x = math.log10(intensity)
    intersections: dict[float, float] = {}
    for trace in figure.get("data", []):
        meta = trace.get("meta")
        if (
            not isinstance(meta, dict)
            or meta.get("carm_role") != "roof-ceiling"
            or trace.get("visible") is False
            or trace.get("visible") == "legendonly"
        ):
            continue
        xs, ys = trace.get("x", []), trace.get("y", [])
        for index in range(min(len(xs), len(ys)) - 1):
            try:
                x1, x2 = float(xs[index]), float(xs[index + 1])
                y1, y2 = float(ys[index]), float(ys[index + 1])
            except (TypeError, ValueError):
                continue
            if min(x1, x2, y1, y2) <= 0 or not all(map(math.isfinite, (x1, x2, y1, y2))):
                continue
            log_x1, log_x2 = math.log10(x1), math.log10(x2)
            if not min(log_x1, log_x2) <= log_x <= max(log_x1, log_x2):
                continue
            if log_x1 == log_x2:
                if log_x != log_x1:
                    continue
                performance = max(y1, y2)
            else:
                fraction = (log_x - log_x1) / (log_x2 - log_x1)
                performance = 10 ** (math.log10(y1) + fraction * (math.log10(y2) - math.log10(y1)))
            if not math.isfinite(performance):
                continue
            intersections[round(math.log10(performance), 10)] = performance
    return sorted(intersections.values(), reverse=True)


def _point_guide_traces(figure: dict[str, Any], hover_data: dict[str, Any] | None) -> list[go.Scatter]:
    """Build fixed hover and selection overlay traces from the current figure."""
    layout = figure.get("layout", {})
    meta = layout.get("meta")
    selected_identity = meta.get("carm_selected_point") if isinstance(meta, dict) else None
    hover_identity = None
    points = hover_data.get("points", []) if hover_data else []
    if points:
        customdata = points[0].get("customdata")
        if isinstance(customdata, list) and len(customdata) >= 5:
            hover_identity = {
                "roof_id": customdata[2],
                "app_id": customdata[3],
                "point_index": customdata[4],
            }

    font = layout.get("font")
    color_value = font.get("color") if isinstance(font, dict) else None
    color = color_value if isinstance(color_value, str) else "#666666"
    guide_traces: dict[tuple[str, str], go.Scatter] = {}
    for guide, kind in _GUIDE_TRACE_KEYS:
        trace_meta = {"carm_role": _GUIDE_TRACE_ROLE, "guide": guide, "kind": kind}
        if kind == "line":
            guide_traces[(guide, kind)] = go.Scatter(
                x=[],
                y=[],
                mode="lines",
                name="",
                line={"color": color, "width": 3},
                opacity=0.5,
                showlegend=False,
                hoverinfo="skip",
                meta=trace_meta,
            )
        else:
            guide_traces[(guide, kind)] = go.Scatter(
                x=[],
                y=[],
                mode="markers",
                name="",
                marker={"color": color, "symbol": "circle", "size": 10, "opacity": 0.5},
                hovertext=[],
                hovertemplate="%{hovertext}<extra></extra>",
                showlegend=False,
                meta=trace_meta,
            )

    yaxis = layout.get("yaxis", {})
    y_range = yaxis.get("range") if isinstance(yaxis, dict) else None
    xaxis = layout.get("xaxis", {})
    x_range = xaxis.get("range") if isinstance(xaxis, dict) else None
    if not isinstance(y_range, list) or len(y_range) != 2:
        return [guide_traces[key] for key in _GUIDE_TRACE_KEYS]
    try:
        low, high = map(float, y_range)
    except (TypeError, ValueError):
        return [guide_traces[key] for key in _GUIDE_TRACE_KEYS]
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        return [guide_traces[key] for key in _GUIDE_TRACE_KEYS]

    x_limits: tuple[float, float] | None = None
    if isinstance(x_range, list) and len(x_range) == 2:
        try:
            x_low, x_high = map(float, x_range)
        except (TypeError, ValueError):
            pass
        else:
            if math.isfinite(x_low) and math.isfinite(x_high) and x_high > x_low:
                x_limits = (x_low, x_high)

    guides: list[tuple[str, dict[str, Any] | None]] = [
        ("hover", hover_identity),
        ("selected", selected_identity),
    ]
    resolved = [(guide, _guide_point(figure, identity)) for guide, identity in guides]
    if resolved[0][1] and resolved[1][1] and resolved[0][1][0] == resolved[1][1][0]:
        resolved[0] = ("hover", None)
    for guide, point in resolved:
        if point is None:
            continue
        log_x = math.log10(point[0])
        if x_limits is not None and not x_limits[0] <= log_x <= x_limits[1]:
            continue
        intersections = _roof_intersections(figure, point[0])
        if not intersections:
            continue
        roof = intersections[0]
        if math.log10(roof) < low:
            continue

        line_trace = guide_traces[(guide, "line")]
        line_trace.x = [point[0], point[0]]
        line_trace.y = [10**low, roof]
        markers = guide_traces[(guide, "intersections")]
        marker_x: list[float] = []
        marker_y: list[float] = []
        labels: list[str] = []
        for performance in intersections:
            log_y = math.log10(performance)
            if math.log10(performance / point[1]) <= 1e-9 or not low <= log_y <= high:
                continue
            marker_x.append(point[0])
            marker_y.append(performance)
            labels.append(f"{Performance(performance * 1e9)!s} · {performance / point[1]:.2f}x")
        markers.x = marker_x
        markers.y = marker_y
        markers.hovertext = labels
    return [guide_traces[key] for key in _GUIDE_TRACE_KEYS]


def _guide_trace_key(trace: dict[str, Any]) -> tuple[str, ...] | None:
    meta = trace.get("meta")
    if not isinstance(meta, dict):
        return None
    role = meta.get("carm_role")
    if role == "application":
        return role, str(meta.get("roof_id")), str(meta.get("app_id"))
    if role == "roof-ceiling":
        return role, str(trace.get("legendgroup")), str(trace.get("name"))
    return None


def _preserve_guide_view(figure: Any, current_figure: dict[str, Any] | None) -> None:
    if not current_figure:
        return
    old_traces = {
        key: trace for trace in current_figure.get("data", []) if (key := _guide_trace_key(trace)) is not None
    }
    for trace in figure.data:
        key = _guide_trace_key(trace.to_plotly_json())
        previous = old_traces.get(key) if key is not None else None
        if previous and previous.get("visible") in (False, "legendonly"):
            trace.visible = previous["visible"]
    layout = current_figure.get("layout", {})
    current_meta = layout.get("meta")
    manual_axes = current_meta.get("carm_manual_guide_axes", []) if isinstance(current_meta, dict) else []
    figure_meta = dict(figure.layout.meta) if isinstance(figure.layout.meta, dict) else {}
    figure_meta["carm_manual_guide_axes"] = list(manual_axes)
    figure.update_layout(meta=figure_meta)
    for axis in manual_axes:
        axis_data = layout.get(axis, {})
        if axis_data.get("range"):
            figure.update_layout(**{axis: {"range": axis_data["range"]}})


def register_roofline_callbacks(app: dash.Dash, state: CallbackState) -> None:
    """Register roof selection, figure refresh, and sidebar callbacks."""
    config = state.config
    records = state.records
    app_by_id = state.apps_by_id
    app_dropdown_options = state.app_options
    paraver_data = state.paraver_data
    mode = state.mode

    if mode.show_app_dropdown:

        @app.callback(
            Output(StoreID.SELECTED_POINT, "data"),
            Output(PlotAreaID.ROOFLINE_PLOT, "clickData"),
            Input(PlotAreaID.ROOFLINE_PLOT, "clickData"),
            Input(PlotAreaID.BG_CLICK_CLEAR, "value"),
            State(StoreID.SELECTED_POINT, "data"),
            prevent_initial_call=True,
        )
        def _update_selected_point(
            click_data: dict[str, Any] | None,
            _bg_click: str | None,
            current_selection: dict[str, Any] | None,
        ) -> tuple[dict[str, Any] | None, None]:
            state.trace("_update_selected_point enter")
            ctx = callback_context
            prop_id = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
            return selection_payload(click_data, prop_id, current_selection), None

    # Plot switches should not rebuild the figure.
    plot_inputs: list[Any] = [
        Input(StoreID.ROOF_STORE, "data"),
        Input({"type": RoofCardID.DROPDOWN_MACHINE, "index": ALL}, "value"),
        Input({"type": RoofCardID.DROPDOWN_ISA, "index": ALL}, "value"),
        Input({"type": RoofCardID.DROPDOWN_THREADS, "index": ALL}, "value"),
        Input({"type": RoofCardID.DROPDOWN_COMPUTE, "index": ALL}, "value"),
        Input({"type": RoofCardID.DROPDOWN_DATA_TYPE, "index": ALL}, "value"),
        Input({"type": RoofCardID.DROPDOWN_LS_RATIO, "index": ALL}, "value"),
        Input({"type": RoofCardID.DROPDOWN_FREQUENCY, "index": ALL}, "value"),
        Input({"type": RoofCardID.DROPDOWN_APPS, "index": ALL}, "value"),
    ]
    plot_inputs.append(Input(BenchmarkPanelID.JOB, "data"))
    plot_inputs.append(Input(ProfilePanelID.JOB, "data"))
    plot_states: list[Any] = []
    if mode.show_app_dropdown:  # CARM mode only; paraver has no application points
        plot_inputs.extend(
            [
                Input(StoreID.SELECTED_POINT, "data"),
                Input(PlotAreaID.ROOFLINE_PLOT, "hoverData"),
                Input(PlotAreaID.ROOFLINE_PLOT, "relayoutData"),
                Input(PlotAreaID.ROOFLINE_PLOT, "restyleData"),
            ]
        )
        plot_states.append(State(PlotAreaID.ROOFLINE_PLOT, "figure"))
    last_refreshed_job: dict[str, Any] | None = None

    @app.callback(Output(PlotAreaID.ROOFLINE_PLOT, "figure"), *plot_inputs, *plot_states)
    def _update_plot(
        store_data: dict[str, Any] | None,
        machine_vals: list[str | None],
        isa_vals: list[str | None],
        threads_vals: list[str | None],
        compute_vals: list[list[str] | None],
        data_type_vals: list[str | None],
        ls_ratio_vals: list[str | None],
        freq_vals: list[str | None],
        app_ids_vals: list[list[str] | None],
        job_data: dict[str, Any] | None,
        _profile_data: dict[str, Any] | None,
        selection_data: dict[str, Any] | None = None,
        hover_data: dict[str, Any] | None = None,
        relayout_data: dict[str, Any] | None = None,
        restyle_data: list[Any] | None = None,
        current_figure: dict[str, Any] | None = None,
    ) -> dict[str, Any] | Patch:
        nonlocal last_refreshed_job
        if mode.show_app_dropdown:
            interactions = {
                f"{PlotAreaID.ROOFLINE_PLOT}.hoverData",
                f"{PlotAreaID.ROOFLINE_PLOT}.relayoutData",
                f"{PlotAreaID.ROOFLINE_PLOT}.restyleData",
            }
            triggered = set(callback_context.triggered_prop_ids)
            if triggered and triggered <= interactions:
                figure_state = current_figure or {}
                figure_data = figure_state.get("data", [])
                slots = figure_data[-len(_GUIDE_TRACE_KEYS) :] if len(figure_data) >= len(_GUIDE_TRACE_KEYS) else []
                slots_ready = len(slots) == len(_GUIDE_TRACE_KEYS) and all(
                    isinstance(trace, dict)
                    and isinstance(trace.get("meta"), dict)
                    and trace["meta"].get("carm_role") == _GUIDE_TRACE_ROLE
                    and (trace["meta"].get("guide"), trace["meta"].get("kind")) == key
                    for trace, key in zip(slots, _GUIDE_TRACE_KEYS)
                )
                if slots_ready:
                    patch = Patch()
                    guide_traces = _point_guide_traces(figure_state, hover_data)
                    first_guide_index = len(figure_data) - len(_GUIDE_TRACE_KEYS)
                    for index, trace in enumerate(guide_traces, start=first_guide_index):
                        patch["data"][index] = trace.to_plotly_json()
                    if f"{PlotAreaID.ROOFLINE_PLOT}.relayoutData" in triggered:
                        layout = figure_state.get("layout", {})
                        meta = layout.get("meta")
                        manual_axes = set(meta.get("carm_manual_guide_axes", [])) if isinstance(meta, dict) else set()
                        for axis in ("xaxis", "yaxis"):
                            has_range = f"{axis}.range" in (relayout_data or {}) or (
                                f"{axis}.range[0]" in (relayout_data or {})
                                and f"{axis}.range[1]" in (relayout_data or {})
                            )
                            if has_range:
                                manual_axes.add(axis)
                            elif (relayout_data or {}).get(f"{axis}.autorange") is True:
                                manual_axes.discard(axis)
                        patch["layout"]["meta"]["carm_manual_guide_axes"] = sorted(manual_axes)
                    return patch
        state.trace("_update_plot enter")
        if (
            job_data
            and job_data.get("finished")
            and job_data.get("return_code") == 0
            and not job_data.get("dry_run")
            and job_data != last_refreshed_job
        ):
            output_root = Path(str(job_data.get("output_dir") or config.results_dir))
            output_records = load_all_benchmarks(output_root)
            if output_root.resolve() != config.results_dir.resolve():
                combined_records = load_all_benchmarks(config.results_dir) + output_records
                records[:] = list({json.dumps(record, sort_keys=True): record for record in combined_records}.values())
            else:
                records[:] = output_records
            if mode.show_app_dropdown:
                app_by_id.clear()
                app_by_id.update(BenchmarkAppsProvider(config.results_dir).load())
                app_dropdown_options[:] = [{"label": item.label, "value": item.id} for item in app_by_id.values()]
            state.options = discover_filter_options(records) if records else None
            last_refreshed_job = job_data.copy()
        store, resolved_roofs, _per_roof_opts = resolve_roofs(
            state,
            store_data,
            machine_vals,
            isa_vals,
            threads_vals,
            compute_vals,
            data_type_vals,
            ls_ratio_vals,
            freq_vals,
            app_ids_vals,
        )
        plot_records = records or []
        plot_apps = app_by_id or {}
        if mode.show_time_slider:
            window = store.paraver_state.time_window
            filtered_trace = (
                filter_trace(
                    paraver_data.trace,
                    window,
                    store.paraver_state.ai_threshold,
                    store.paraver_state.duration_threshold,
                )
                if paraver_data is not None
                else None
            )
            figure = build_paraver_figure(
                resolved_roofs,
                plot_records,
                paraver_data,
                filtered_trace,
                settings=store.settings,
                color_mode=store.paraver_state.color_mode,
            )
        else:
            selected = selection_data or {}
            figure = build_roofline_figure(
                resolved_roofs,
                plot_records,
                plot_apps,
                settings=store.settings,
                selected_roof_id=selected.get("roof_id"),
                selected_residency=selected.get("fractions"),
                selected_point=selected,
            )
        if mode.show_app_dropdown:
            _preserve_guide_view(figure, current_figure)
            figure_state = cast("dict[str, Any]", figure.to_dict())
            for trace in _point_guide_traces(figure_state, hover_data):
                figure.add_trace(trace)
            figure_dict = cast("dict[str, Any]", figure.to_dict())
        else:
            figure_dict = cast("dict[str, Any]", figure.to_dict())
        state.trace(f"_update_plot exit traces={len(figure_dict.get('data', []))}")
        return figure_dict

    # Sidebar visibility follows the active panel.
    @app.callback(
        Output(SidebarID.SIDEBAR_CONTENT, "children"),
        Input(StoreID.ROOF_STORE, "data"),
        Input(StoreID.ACTIVE_PANEL, "data"),
        Input({"type": RoofCardID.DROPDOWN_MACHINE, "index": ALL}, "value"),
        Input({"type": RoofCardID.DROPDOWN_ISA, "index": ALL}, "value"),
        Input({"type": RoofCardID.DROPDOWN_THREADS, "index": ALL}, "value"),
        Input({"type": RoofCardID.DROPDOWN_COMPUTE, "index": ALL}, "value"),
        Input({"type": RoofCardID.DROPDOWN_DATA_TYPE, "index": ALL}, "value"),
        Input({"type": RoofCardID.DROPDOWN_LS_RATIO, "index": ALL}, "value"),
        Input({"type": RoofCardID.DROPDOWN_FREQUENCY, "index": ALL}, "value"),
        Input({"type": RoofCardID.DROPDOWN_APPS, "index": ALL}, "value"),
        Input(ProfilePanelID.JOB, "data"),
    )
    def _update_sidebar(
        store_data: dict[str, Any] | None,
        active_panel: str | None,
        machine_vals: list[str | None],
        isa_vals: list[str | None],
        threads_vals: list[str | None],
        compute_vals: list[list[str] | None],
        data_type_vals: list[str | None],
        ls_ratio_vals: list[str | None],
        freq_vals: list[str | None],
        app_ids_vals: list[list[str] | None],
        _profile_job_data: dict[str, Any] | None,
    ) -> list[Any]:
        state.trace("_update_sidebar enter")
        store, resolved_roofs, per_roof_opts = resolve_roofs(
            state,
            store_data,
            machine_vals,
            isa_vals,
            threads_vals,
            compute_vals,
            data_type_vals,
            ls_ratio_vals,
            freq_vals,
            app_ids_vals,
            active_panel=active_panel,
        )
        carm_view_panel = build_carm_view_panel(
            store,
            per_roof_opts,
            resolved_roofs,
            per_roof_app_options=filter_app_options_for_roofs(resolved_roofs, state.app_options, state.apps_by_id),
            include_apps_section=state.mode.show_app_dropdown,
        )
        settings_panel = build_settings_panel(store, None)
        children = [carm_view_panel, settings_panel, *_build_guide_panels(store.active_panel)]
        if mode.has_export_tab:
            children.append(build_export_panel(store))
        state.trace(f"_update_sidebar exit panel={store.active_panel}")
        return children
