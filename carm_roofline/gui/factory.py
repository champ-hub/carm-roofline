from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Any, cast

import dash
import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback_context, html
from dash.exceptions import PreventUpdate

from carm_roofline.gui.components import build_carm_view_panel, build_layout, build_settings_panel
from carm_roofline.gui.config import GUIConfig, gui_settings_path, load_gui_settings, save_gui_settings
from carm_roofline.gui.data import (
    ActivePanel,
    DropdownOption,
    RoofConfig,
    RoofStore,
    build_roofline_figure,
    make_default_roof,
)
from carm_roofline.gui.ids import (
    CarmViewPanelID,
    NavbarID,
    PlotAreaID,
    RoofCardID,
    SettingsPanelID,
    SidebarID,
    StoreID,
)
from carm_roofline.output_utils import debug
from carm_roofline.roofline_assembly import (
    ApplicationRecord,
    BenchmarkRecord,
    FilterOptions,
    RooflineFilter,
    discover_filter_options,
    load_all_applications,
    load_all_benchmarks,
)

# Helpers


def _parse_trigger_id() -> dict[str, Any]:
    """Parse the pattern-matching ID dict from the triggered input's prop_id."""
    ctx = callback_context
    if not ctx.triggered:
        return {}
    prop_id = ctx.triggered[0].get("prop_id", "")
    id_str, _prop = prop_id.rsplit(".", 1)
    try:
        return cast(dict[str, Any], json.loads(id_str))
    except (json.JSONDecodeError, ValueError):
        return {}


def _get_trigger_index() -> int:
    """Extract the ``index`` field from the triggered input's pattern-matching ID."""
    trigger_id = _parse_trigger_id()
    try:
        return int(trigger_id.get("index", -1))
    except (ValueError, TypeError):
        return -1


def _first_or_none(vals: list[str]) -> str | None:
    """First value from a filtered-options list, or None if empty."""
    return vals[0] if vals else None


def _first_int_or_none(vals: list[int]) -> int | None:
    """First value as int from a filtered-options list, or None if empty."""
    return vals[0] if vals else None


# App factory


def create_app(config: GUIConfig) -> dash.Dash:
    """Create and return a configured Dash application."""
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        assets_folder="assets",
        suppress_callback_exceptions=True,
        title="CARM Roofline",
    )

    # Suppress Flask/Werkzeug HTTP request logging — only errors and above
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    # set dash logging to show all callbacks
    logging.getLogger("dash").setLevel(logging.DEBUG)
    # Load all benchmark records from results directory
    records: list[BenchmarkRecord] = []
    applications: list[ApplicationRecord] = []
    if config.results_dir.exists():
        records = load_all_benchmarks(config.results_dir)
        applications = load_all_applications(config.results_dir)

    # Build application lookup maps
    app_by_id: dict[str, ApplicationRecord] = {a.id: a for a in applications}
    app_dropdown_options: list[DropdownOption] = [{"label": a.label, "value": a.id} for a in applications]

    # Discover available filter / dropdown options from data
    opts: FilterOptions | None = None
    if records:
        opts = discover_filter_options(records)
        debug(f"Available machines: {opts['machine']}")
        debug(f"Available ISAs: {opts['isa']}")
        debug(f"Available threads: {opts['num_threads']}")
        debug(f"Available load-store ratios: {opts['load_store_ratio']}")

    # Build a sensible default roof config from the first available options
    initial_roof = make_default_roof(opts)
    initial_store = RoofStore(roof_template=initial_roof)
    initial_store.roofs = [initial_roof]
    # Override defaults with persisted settings
    saved = load_gui_settings(gui_settings_path())
    initial_store.settings = saved
    debug(f"Loaded GUI settings: {saved}")

    app.layout = build_layout(initial_store, opts, app_dropdown_options)

    _register_callbacks(app, config, records, opts, app_by_id, app_dropdown_options)
    return app


def _register_callbacks(
    app: dash.Dash,
    config: GUIConfig,
    records: list[BenchmarkRecord] | None = None,
    opts: FilterOptions | None = None,
    app_by_id: dict[str, ApplicationRecord] | None = None,
    app_dropdown_options: list[DropdownOption] | None = None,
) -> None:
    """Register all application callbacks."""
    _cb_seq: int = 0

    def _tr(msg: str) -> None:
        nonlocal _cb_seq
        _cb_seq += 1
        if config.gui_debug:
            debug(f"[CB#{_cb_seq}] {msg}")

    # 1. Toggle active panel
    @app.callback(
        Output(StoreID.ACTIVE_PANEL, "data"),
        Input(NavbarID.BTN_CARM_VIEW, "n_clicks"),
        Input(NavbarID.BTN_SETTINGS, "n_clicks"),
    )
    def _toggle_panel(carm_view_clicks: int | None, settings_clicks: int | None) -> ActivePanel:
        ctx = callback_context
        if not ctx.triggered:
            return ActivePanel.CARM_VIEW
        trigger = ctx.triggered[0]["prop_id"]
        if NavbarID.BTN_SETTINGS in trigger:
            return ActivePanel.SETTINGS
        return ActivePanel.CARM_VIEW

    # 2. Add roof
    @app.callback(
        Output(StoreID.ROOF_STORE, "data"),
        Input(CarmViewPanelID.BTN_ADD_ROOF, "n_clicks"),
        State(StoreID.ROOF_STORE, "data"),
    )
    def _add_roof(
        n_clicks: int | None,
        store_data: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not n_clicks:
            raise PreventUpdate
        store = RoofStore.from_dict(store_data or {})
        _tr(f"_add_roof enter roofs={len(store.roofs)}")
        store.add_roof(roof_template=make_default_roof(opts))
        _tr(f"_add_roof exit  roofs={len(store.roofs)}")
        return store.to_dict()

    # 3. Remove roof
    @app.callback(
        Output(StoreID.ROOF_STORE, "data", allow_duplicate=True),
        Input({"type": RoofCardID.BTN_REMOVE_ROOF, "index": ALL}, "n_clicks"),
        State(StoreID.ROOF_STORE, "data"),
        prevent_initial_call=True,
    )
    def _remove_roof(
        n_clicks_list: list[int | None],
        store_data: dict[str, Any] | None,
    ) -> dict[str, Any]:
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        val = ctx.triggered[0].get("value", 0)
        if not val:  # component recreated during sidebar rebuild, not a real click
            raise PreventUpdate
        store = RoofStore.from_dict(store_data or {})
        _tr(f"_remove_roof enter trigger={ctx.triggered[0].get('prop_id', '?')} roofs={len(store.roofs)}")
        index = _get_trigger_index()
        if 0 <= index < len(store.roofs):
            store.remove_roof(store.roofs[index].id)
        _tr(f"_remove_roof exit  roofs={len(store.roofs)}")
        return store.to_dict()

    # 8a. Shared: merge dropdown values into store and resolve roofs
    def _resolve_roof_data(
        store_data: dict[str, Any] | None,
        machine_vals: list[str | None],
        isa_vals: list[str | None],
        threads_vals: list[str | None],
        compute_vals: list[list[str] | None],
        data_type_vals: list[str | None],
        ls_ratio_vals: list[str | None],
        freq_vals: list[str | None],
        app_ids_vals: list[list[str] | None],
        active_panel: str | None = None,
    ) -> tuple[RoofStore, list[RoofConfig], list[FilterOptions | None]]:
        store = RoofStore.from_dict(store_data or {})
        if active_panel is not None:
            store.active_panel = ActivePanel(active_panel) if active_panel else ActivePanel.CARM_VIEW
        for i, roof in enumerate(store.roofs):
            if i < len(machine_vals):
                roof.machine = machine_vals[i]
            if i < len(isa_vals):
                roof.isa = isa_vals[i]
            if i < len(threads_vals):
                thr = threads_vals[i]
                if thr is not None:
                    roof.num_threads = int(thr)
                else:
                    roof.num_threads = None
            if i < len(compute_vals):
                cv = compute_vals[i]
                if cv is not None:
                    roof.compute_insts = cv
            if i < len(data_type_vals):
                roof.data_type = data_type_vals[i]
            if i < len(ls_ratio_vals):
                roof.load_store_ratio = ls_ratio_vals[i]
            if i < len(freq_vals):
                fv = freq_vals[i]
                if fv is not None:
                    roof.actual_frequency_hz = int(fv)
                else:
                    roof.actual_frequency_hz = None
            if i < len(app_ids_vals):
                roof.app_ids = list(app_ids_vals[i] or [])
        debug(f"_resolve_roof_data: {len(store.roofs)} roof(s), panel={store.active_panel}")
        recs = records or []
        per_roof_opts: list[FilterOptions | None] = []
        resolved_roofs: list[RoofConfig] = []

        for roof in store.roofs:
            if not recs:
                per_roof_opts.append(None)
                resolved_roofs.append(roof)
                continue

            # Stabilization: single pass — clears stale values incompatible with current locks
            base = RooflineFilter(
                machine=roof.machine,
                isa=roof.isa,
                num_threads=roof.num_threads,
                data_type=roof.data_type,
                actual_frequency_hz=roof.actual_frequency_hz,
                load_store_ratio=roof.load_store_ratio,
            )
            fo = discover_filter_options(recs, base)
            if roof.machine is not None and roof.machine not in fo["machine"]:
                debug(f"_resolve_roof_data[{roof.id}]: machine '{roof.machine}' not in options -> None")
                roof.machine = None
            if roof.isa is not None and roof.isa not in fo["isa"]:
                debug(f"_resolve_roof_data[{roof.id}]: isa '{roof.isa}' not in options -> None")
                roof.isa = None
            if roof.num_threads is not None and roof.num_threads not in fo["num_threads"]:
                debug(f"_resolve_roof_data[{roof.id}]: threads {roof.num_threads} not in options -> None")
                roof.num_threads = None
            if roof.data_type is not None and roof.data_type not in fo["data_type"]:
                debug(f"_resolve_roof_data[{roof.id}]: data_type '{roof.data_type}' not in options -> None")
                roof.data_type = None
            if roof.load_store_ratio is not None and roof.load_store_ratio not in fo["load_store_ratio"]:
                debug(f"_resolve_roof_data[{roof.id}]: ratio '{roof.load_store_ratio}' not in options -> None")
                roof.load_store_ratio = None
            if roof.actual_frequency_hz is not None and roof.actual_frequency_hz not in fo["actual_frequency_hz"]:
                debug(f"_resolve_roof_data[{roof.id}]: freq {roof.actual_frequency_hz} not in options -> None")
                roof.actual_frequency_hz = None

            # Build user-locks filter from stabilized (user-set) values only.
            # Used for per-roof dropdown options so auto-resolved values dont't constrain dropdown menus.
            user_locks = RooflineFilter(
                machine=roof.machine,
                isa=roof.isa,
                num_threads=roof.num_threads,
                data_type=roof.data_type,
                load_store_ratio=roof.load_store_ratio,
                actual_frequency_hz=roof.actual_frequency_hz,
            )

            # Auto-resolution: pick first valid for any field the user did not set.
            # `acc` is seeded with ALL user locks so every discover_filter_options call
            # respects every user constraint. Uses "modify filter, call again" pattern.
            acc = RooflineFilter(
                machine=roof.machine,
                isa=roof.isa,
                num_threads=roof.num_threads,
                data_type=roof.data_type,
                load_store_ratio=roof.load_store_ratio,
                actual_frequency_hz=roof.actual_frequency_hz,
            )
            cur_machine = (
                roof.machine
                if roof.machine is not None
                else _first_or_none(discover_filter_options(recs, acc)["machine"])
            )
            acc = replace(acc, machine=cur_machine)
            cur_isa = roof.isa if roof.isa is not None else _first_or_none(discover_filter_options(recs, acc)["isa"])
            acc = replace(acc, isa=cur_isa)
            cur_threads = (
                roof.num_threads
                if roof.num_threads is not None
                else _first_int_or_none(discover_filter_options(recs, acc)["num_threads"])
            )
            acc = replace(acc, num_threads=cur_threads)
            cur_data_type = (
                roof.data_type
                if roof.data_type is not None
                else _first_or_none(discover_filter_options(recs, acc)["data_type"])
            )
            acc = replace(acc, data_type=cur_data_type)
            cur_ls_ratio = (
                roof.load_store_ratio
                if roof.load_store_ratio is not None
                else _first_or_none(discover_filter_options(recs, acc)["load_store_ratio"])
            )
            acc = replace(acc, load_store_ratio=cur_ls_ratio)
            cur_freq = (
                roof.actual_frequency_hz
                if roof.actual_frequency_hz is not None
                else _first_int_or_none(discover_filter_options(recs, acc)["actual_frequency_hz"])
            )
            acc = replace(acc, actual_frequency_hz=cur_freq)
            resolved_roofs.append(
                RoofConfig(
                    roof_id=roof.id,
                    label=roof.label,
                    machine=cur_machine,
                    isa=cur_isa,
                    num_threads=cur_threads,
                    data_type=cur_data_type,
                    compute_insts=roof.compute_insts,
                    load_store_ratio=cur_ls_ratio,
                    actual_frequency_hz=cur_freq,
                    app_ids=roof.app_ids,
                )
            )
            per_roof_opts.append(discover_filter_options(recs, user_locks))

        return store, resolved_roofs, per_roof_opts

    # 8b. Update plot (no ACTIVE_PANEL, panel switches shouldn't rebuild the figure)
    @app.callback(
        Output(PlotAreaID.ROOFLINE_PLOT, "figure"),
        Input(StoreID.ROOF_STORE, "data"),
        Input({"type": RoofCardID.DROPDOWN_MACHINE, "index": ALL}, "value"),
        Input({"type": RoofCardID.DROPDOWN_ISA, "index": ALL}, "value"),
        Input({"type": RoofCardID.DROPDOWN_THREADS, "index": ALL}, "value"),
        Input({"type": RoofCardID.DROPDOWN_COMPUTE, "index": ALL}, "value"),
        Input({"type": RoofCardID.DROPDOWN_DATA_TYPE, "index": ALL}, "value"),
        Input({"type": RoofCardID.DROPDOWN_LS_RATIO, "index": ALL}, "value"),
        Input({"type": RoofCardID.DROPDOWN_FREQUENCY, "index": ALL}, "value"),
        Input({"type": RoofCardID.DROPDOWN_APPS, "index": ALL}, "value"),
    )
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
    ) -> dict[str, Any]:
        _tr("_update_plot enter")
        store, resolved_roofs, _per_roof_opts = _resolve_roof_data(
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
        figure = build_roofline_figure(
            resolved_roofs,
            records or [],
            app_by_id,
            settings=store.settings,
        )
        figure_dict = cast("dict[str, Any]", figure.to_dict())
        _tr(f"_update_plot exit traces={len(figure_dict.get('data', []))}")
        return figure_dict

    # 8c. Update sidebar (includes ACTIVE_PANEL, needs to toggle panel visibility)
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
        _tr("_update_sidebar enter")
        store, resolved_roofs, per_roof_opts = _resolve_roof_data(
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
        carm_view_panel = build_carm_view_panel(store, per_roof_opts, resolved_roofs, app_dropdown_options)
        settings_panel = build_settings_panel(store, None)
        _tr(f"_update_sidebar exit panel={store.active_panel}")
        return [carm_view_panel, settings_panel]

    # 9. Normalize by threads toggle
    @app.callback(
        Output(StoreID.ROOF_STORE, "data", allow_duplicate=True),
        Input(SettingsPanelID.SWITCH_NORMALIZE, "value"),
        State(StoreID.ROOF_STORE, "data"),
        prevent_initial_call=True,
    )
    def _toggle_normalize(
        normalize: bool | None,
        store_data: dict[str, Any] | None,
    ) -> dict[str, Any]:
        store = RoofStore.from_dict(store_data or {})
        store.settings = replace(store.settings, normalize_by_threads=bool(normalize))
        save_gui_settings(gui_settings_path(), store.settings)
        return store.to_dict()

    # 10. Marker scale slider
    @app.callback(
        Output(StoreID.ROOF_STORE, "data", allow_duplicate=True),
        Input(SettingsPanelID.SLIDER_MARKER_SIZE, "value"),
        State(StoreID.ROOF_STORE, "data"),
        prevent_initial_call=True,
    )
    def _update_marker_scale(
        scale: float | None,
        store_data: dict[str, Any] | None,
    ) -> dict[str, Any]:
        store = RoofStore.from_dict(store_data or {})
        store.settings = replace(store.settings, marker_scale_factor=float(scale if scale is not None else 50.0))
        save_gui_settings(gui_settings_path(), store.settings)
        return store.to_dict()

    # 11. Power2 ticks toggle
    @app.callback(
        Output(StoreID.ROOF_STORE, "data", allow_duplicate=True),
        Input(SettingsPanelID.SWITCH_POWER2_TICKS, "value"),
        State(StoreID.ROOF_STORE, "data"),
        prevent_initial_call=True,
    )
    def _toggle_power2_ticks(
        power2: bool | None,
        store_data: dict[str, Any] | None,
    ) -> dict[str, Any]:
        store = RoofStore.from_dict(store_data or {})
        store.settings = replace(store.settings, power2_ticks=bool(power2))
        save_gui_settings(gui_settings_path(), store.settings)
        return store.to_dict()

    # 12. Line width slider
    @app.callback(
        Output(StoreID.ROOF_STORE, "data", allow_duplicate=True),
        Input(SettingsPanelID.SLIDER_LINE_WIDTH, "value"),
        State(StoreID.ROOF_STORE, "data"),
        prevent_initial_call=True,
    )
    def _update_line_width(
        width: float | None,
        store_data: dict[str, Any] | None,
    ) -> dict[str, Any]:
        store = RoofStore.from_dict(store_data or {})
        store.settings = replace(store.settings, line_width=float(width if width is not None else 1.5))
        save_gui_settings(gui_settings_path(), store.settings)
        return store.to_dict()

    # 13. Axis label font size slider
    @app.callback(
        Output(StoreID.ROOF_STORE, "data", allow_duplicate=True),
        Input(SettingsPanelID.SLIDER_FONT_SIZE_AXIS_LABEL, "value"),
        State(StoreID.ROOF_STORE, "data"),
        prevent_initial_call=True,
    )
    def _update_axis_label_font_size(
        size: int | None,
        store_data: dict[str, Any] | None,
    ) -> dict[str, Any]:
        store = RoofStore.from_dict(store_data or {})
        store.settings = replace(store.settings, axis_label_font_size=int(size if size is not None else 14))
        save_gui_settings(gui_settings_path(), store.settings)
        return store.to_dict()

    # 14. Axis tick font size slider
    @app.callback(
        Output(StoreID.ROOF_STORE, "data", allow_duplicate=True),
        Input(SettingsPanelID.SLIDER_FONT_SIZE_AXIS_TICK, "value"),
        State(StoreID.ROOF_STORE, "data"),
        prevent_initial_call=True,
    )
    def _update_axis_tick_font_size(
        size: int | None,
        store_data: dict[str, Any] | None,
    ) -> dict[str, Any]:
        store = RoofStore.from_dict(store_data or {})
        store.settings = replace(store.settings, axis_tick_font_size=int(size if size is not None else 12))
        save_gui_settings(gui_settings_path(), store.settings)
        return store.to_dict()

    # 15. Tooltip font size slider
    @app.callback(
        Output(StoreID.ROOF_STORE, "data", allow_duplicate=True),
        Input(SettingsPanelID.SLIDER_FONT_SIZE_TOOLTIP, "value"),
        State(StoreID.ROOF_STORE, "data"),
        prevent_initial_call=True,
    )
    def _update_tooltip_font_size(
        size: int | None,
        store_data: dict[str, Any] | None,
    ) -> dict[str, Any]:
        store = RoofStore.from_dict(store_data or {})
        store.settings = replace(store.settings, tooltip_font_size=int(size if size is not None else 12))
        save_gui_settings(gui_settings_path(), store.settings)
        return store.to_dict()

    # 16. Legend font size slider
    @app.callback(
        Output(StoreID.ROOF_STORE, "data", allow_duplicate=True),
        Input(SettingsPanelID.SLIDER_FONT_SIZE_LEGEND, "value"),
        State(StoreID.ROOF_STORE, "data"),
        prevent_initial_call=True,
    )
    def _update_legend_font_size(
        size: int | None,
        store_data: dict[str, Any] | None,
    ) -> dict[str, Any]:
        store = RoofStore.from_dict(store_data or {})
        store.settings = replace(store.settings, legend_font_size=int(size if size is not None else 10))
        save_gui_settings(gui_settings_path(), store.settings)
        return store.to_dict()

    # 17. Sync button styles with active panel
    @app.callback(
        Output(NavbarID.BTN_CARM_VIEW, "className"),
        Output(NavbarID.BTN_SETTINGS, "className"),
        Input(StoreID.ACTIVE_PANEL, "data"),
    )
    def _update_button_styles(active_panel: str | None) -> tuple[str, str]:
        is_carm_view = active_panel == ActivePanel.CARM_VIEW
        carm_view_cls = f"navbar-btn{' navbar-btn--active' if is_carm_view else ''}"
        settings_cls = f"navbar-btn{' navbar-btn--active' if not is_carm_view else ''}"
        return carm_view_cls, settings_cls

    # 14. Toggle roof card collapse
    @app.callback(
        Output(StoreID.ROOF_STORE, "data", allow_duplicate=True),
        Input({"type": RoofCardID.BTN_COLLAPSE_ROOF, "index": ALL}, "n_clicks"),
        State(StoreID.ROOF_STORE, "data"),
        prevent_initial_call=True,
    )
    def _toggle_collapse_roof(
        n_clicks: list[int | None],
        store_data: dict[str, Any] | None,
    ) -> dict[str, Any]:
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        val = ctx.triggered[0].get("value", 0)
        if not val:
            raise PreventUpdate
        store = RoofStore.from_dict(store_data or {})
        _tr(f"_toggle_collapse_roof trigger={ctx.triggered[0].get('prop_id', '?')} roofs={len(store.roofs)}")
        index = _get_trigger_index()
        if 0 <= index < len(store.roofs):
            store.roofs[index].collapsed = not store.roofs[index].collapsed
        _tr(f"_toggle_collapse_roof exit  roofs={len(store.roofs)}")
        return store.to_dict()

    # 15. Toggle advanced section collapse
    @app.callback(
        Output(StoreID.ROOF_STORE, "data", allow_duplicate=True),
        Input({"type": RoofCardID.BTN_ADVANCED_COLLAPSE, "index": ALL}, "n_clicks"),
        State(StoreID.ROOF_STORE, "data"),
        prevent_initial_call=True,
    )
    def _toggle_advanced_collapse(
        n_clicks: list[int | None],
        store_data: dict[str, Any] | None,
    ) -> dict[str, Any]:
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        val = ctx.triggered[0].get("value", 0)
        if not val:
            raise PreventUpdate
        store = RoofStore.from_dict(store_data or {})
        _tr(f"_toggle_advanced_collapse trigger={ctx.triggered[0].get('prop_id', '?')} roofs={len(store.roofs)}")
        index = _get_trigger_index()
        if 0 <= index < len(store.roofs):
            store.roofs[index].advanced_collapsed = not store.roofs[index].advanced_collapsed
        _tr(f"_toggle_advanced_collapse exit  roofs={len(store.roofs)}")
        return store.to_dict()
