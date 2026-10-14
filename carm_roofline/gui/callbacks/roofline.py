from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import dash
from dash import ALL, Input, Output, Patch, State, callback_context, html

from carm_roofline.gui.callbacks.helpers import filter_app_options_for_roofs, selection_payload
from carm_roofline.gui.callbacks.resolution import resolve_roofs
from carm_roofline.gui.callbacks.state import CallbackState
from carm_roofline.gui.components import build_carm_view_panel, build_export_panel, build_settings_panel
from carm_roofline.gui.data import build_paraver_figure, build_roofline_figure
from carm_roofline.gui.ids import BenchmarkPanelID, PlotAreaID, ProfilePanelID, RoofCardID, SidebarID, StoreID
from carm_roofline.gui.providers import BenchmarkAppsProvider, filter_trace
from carm_roofline.roofline_assembly import discover_filter_options, load_all_benchmarks

_HOVER_GUIDE = "carm-hover-guide"
_SELECTED_GUIDE = "carm-selected-guide"
_GUIDE_NAMES = {_HOVER_GUIDE, _SELECTED_GUIDE}


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


def _roof_height(figure: dict[str, Any], intensity: float) -> float | None:
    log_x = math.log10(intensity)
    maximum: float | None = None
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
            fraction = (log_x - log_x1) / (log_x2 - log_x1) if log_x1 != log_x2 else 0.0
            y = 10 ** (math.log10(y1) + fraction * (math.log10(y2) - math.log10(y1)))
            if math.isfinite(y) and (maximum is None or y > maximum):
                maximum = y
    return maximum


def _point_guide_shapes(figure: dict[str, Any], hover_data: dict[str, Any] | None) -> list[dict[str, Any]]:
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

    shapes = [shape for shape in layout.get("shapes", []) if shape.get("name") not in _GUIDE_NAMES]
    yaxis = layout.get("yaxis", {})
    axis_range = yaxis.get("range") if isinstance(yaxis, dict) else None
    if not isinstance(axis_range, list) or len(axis_range) != 2:
        return shapes
    try:
        low, high = map(float, axis_range)
    except (TypeError, ValueError):
        return shapes
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        return shapes

    font = layout.get("font")
    color = font.get("color") if isinstance(font, dict) else None
    guides: list[tuple[str, dict[str, Any] | None]] = [
        (_HOVER_GUIDE, hover_identity),
        (_SELECTED_GUIDE, selected_identity),
    ]
    resolved = [(name, _guide_point(figure, identity)) for name, identity in guides]
    if resolved[0][1] and resolved[1][1] and resolved[0][1][0] == resolved[1][1][0]:
        resolved[0] = (_HOVER_GUIDE, None)
    for name, point in resolved:
        if point is None:
            continue
        roof = _roof_height(figure, point[0])
        if roof is None or math.log10(roof) < low:
            continue
        shapes.append(
            {
                "type": "line",
                "name": name,
                "xref": "x",
                "x0": point[0],
                "x1": point[0],
                "yref": "paper",
                "y0": 0,
                "y1": min(1, (math.log10(roof) - low) / (high - low)),
                "line": {"color": color or "#666666", "width": 1, "dash": "solid"},
                "layer": "above",
                "editable": False,
                "showlegend": False,
                "captureevents": False,
            }
        )
    return shapes


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
                patch = Patch()
                figure_state = current_figure or {}
                patch["layout"]["shapes"] = _point_guide_shapes(figure_state, hover_data)
                if f"{PlotAreaID.ROOFLINE_PLOT}.relayoutData" in triggered:
                    layout = figure_state.get("layout", {})
                    meta = layout.get("meta")
                    manual_axes = set(meta.get("carm_manual_guide_axes", [])) if isinstance(meta, dict) else set()
                    for axis in ("xaxis", "yaxis"):
                        has_range = f"{axis}.range" in (relayout_data or {}) or (
                            f"{axis}.range[0]" in (relayout_data or {}) and f"{axis}.range[1]" in (relayout_data or {})
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
            figure_dict = cast("dict[str, Any]", figure.to_dict())
            figure_dict["layout"]["shapes"] = _point_guide_shapes(figure_dict, hover_data)
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
    ) -> list[html.Div]:
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
        children = [carm_view_panel, settings_panel]
        if mode.has_export_tab:
            children.append(build_export_panel(store))
        state.trace(f"_update_sidebar exit panel={store.active_panel}")
        return children
