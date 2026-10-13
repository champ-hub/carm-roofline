from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import dash
from dash import ALL, Input, Output, State, callback_context, html

from carm_roofline.gui.callbacks.helpers import filter_app_options_for_roofs, selection_payload
from carm_roofline.gui.callbacks.resolution import resolve_roofs
from carm_roofline.gui.callbacks.state import CallbackState
from carm_roofline.gui.components import build_carm_view_panel, build_export_panel, build_settings_panel
from carm_roofline.gui.data import build_paraver_figure, build_roofline_figure
from carm_roofline.gui.ids import BenchmarkPanelID, PlotAreaID, RoofCardID, SidebarID, StoreID
from carm_roofline.gui.providers import BenchmarkAppsProvider, filter_trace
from carm_roofline.roofline_assembly import discover_filter_options, load_all_benchmarks


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
    if mode.show_app_dropdown:  # CARM mode only; paraver has no residency data
        plot_inputs.append(Input(StoreID.SELECTED_POINT, "data"))
    last_refreshed_job: dict[str, Any] | None = None

    @app.callback(Output(PlotAreaID.ROOFLINE_PLOT, "figure"), *plot_inputs)
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
        selection_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        nonlocal last_refreshed_job
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
            )
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
