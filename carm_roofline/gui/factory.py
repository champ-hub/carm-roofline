from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, cast

import dash
import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback_context, html
from dash.exceptions import PreventUpdate

from carm_roofline.core.error import UserError
from carm_roofline.gui.colors import COLOR_MODE_PARAVER
from carm_roofline.gui.components import (
    AI_FILTER_LOG_MIN,
    DURATION_FILTER_LOG_MIN,
    build_carm_view_panel,
    build_export_panel,
    build_layout,
    build_settings_panel,
)
from carm_roofline.gui.config import (
    GUIConfig,
    GUIMode,
    gui_settings_path,
    load_gui_settings,
    save_gui_settings,
)
from carm_roofline.gui.data import (
    ActivePanel,
    DropdownOption,
    RoofConfig,
    RoofStore,
    build_paraver_figure,
    build_roofline_figure,
    format_roof_label,
    make_default_roof,
    roof_divisor,
    roof_to_filter,
)
from carm_roofline.gui.export import (
    ExportModeExporter,
    export_ai,
    export_ldst_percent,
    export_performance,
    export_proximity,
    export_region,
    export_roof_labels,
    write_export_files,
)
from carm_roofline.gui.ids import (
    CarmViewPanelID,
    ExportPanelID,
    NavbarID,
    ParaverID,
    PlotAreaID,
    RoofCardID,
    SettingsPanelID,
    SidebarID,
    StoreID,
)
from carm_roofline.gui.providers import (
    BenchmarkAppsProvider,
    ParaverData,
    ParaverProvider,
    filter_trace,
    trace_time_range,
)
from carm_roofline.output_utils import debug, warn
from carm_roofline.roofline_assembly import (
    ApplicationRecord,
    AssembledRoofline,
    BenchmarkRecord,
    FilterOptions,
    RooflineFilter,
    assemble_roofline,
    discover_filter_options,
    load_all_benchmarks,
)


@dataclass(frozen=True)
class ExportModeSpec:
    """Wiring for one Paraver export mode: trigger button, status readout, exporter, roof requirement."""

    button: ExportPanelID
    status: ExportPanelID
    exporter: ExportModeExporter
    needs_roof: bool


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


# (field_name, filter_key, first_fn) — first_fn is _first_or_none or _first_int_or_none
_FILTER_FIELD_SPECS: list[tuple[str, str, Callable[..., Any]]] = [
    ("machine", "machine", _first_or_none),
    ("isa", "isa", _first_or_none),
    ("num_threads", "num_threads", _first_int_or_none),
    ("data_type", "data_type", _first_or_none),
    ("load_store_ratio", "load_store_ratio", _first_or_none),
    ("actual_frequency_hz", "actual_frequency_hz", _first_int_or_none),
]


def _filter_app_options_for_roofs(
    roofs: list[RoofConfig],
    app_dropdown_options: list[DropdownOption] | None,
    app_by_id: dict[str, ApplicationRecord] | None,
) -> list[list[DropdownOption]] | None:
    """Filter app dropdown options per roof by machine."""
    if not app_dropdown_options or not app_by_id:
        return None
    return [[o for o in app_dropdown_options if app_by_id[o["value"]].machine == roof.machine] for roof in roofs]


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
    mode = GUIMode.PARAVER if config.paraver_trace is not None else GUIMode.CARM

    records: list[BenchmarkRecord] = []
    if config.results_dir.exists():
        records = load_all_benchmarks(config.results_dir)

    app_by_id: dict[str, ApplicationRecord] = {}
    app_dropdown_options: list[DropdownOption] = []
    trace_bounds: tuple[float, float] | None = None
    initial_window: tuple[float, float] | None = None
    paraver_data: ParaverData | None = None

    if mode.show_time_slider:
        # Paraver mode: application points come from an external trace.
        if config.paraver_trace is not None and config.paraver_window_csv is not None:
            provider = ParaverProvider(config.paraver_trace, config.paraver_window_csv)
            try:
                paraver_data = provider.load()
            except UserError as exc:
                warn(str(exc))
            if paraver_data is not None:
                trace_bounds = trace_time_range(paraver_data.trace)
                if config.paraver_use_semantic_window:
                    initial_window = provider.window_extent
        else:
            warn("Paraver mode needs --paraver-trace and --paraver-window-csv; running without application points.")
    elif config.results_dir.exists():
        app_by_id = BenchmarkAppsProvider(config.results_dir).load()
        app_dropdown_options = [{"label": a.label, "value": a.id} for a in app_by_id.values()]

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

    # Callable layout: Dash evaluates this on EVERY page request, so the
    # dcc.Store(ROOF_STORE) initial data reflects the latest saved settings.
    # This prevents callbacks 9-16 from overwriting user changes on reload.
    def serve_layout() -> html.Div:
        saved = load_gui_settings(gui_settings_path())
        fresh_store = RoofStore(roof_template=initial_roof)
        fresh_store.roofs = [initial_roof]
        fresh_store.settings = saved
        if initial_window is not None:
            fresh_store.paraver_state.time_window = initial_window
        return build_layout(
            fresh_store,
            opts,
            _filter_app_options_for_roofs([initial_roof], app_dropdown_options, app_by_id),
            mode,
            trace_bounds,
        )

    app.layout = serve_layout

    _register_callbacks(app, config, records, opts, app_by_id, app_dropdown_options, paraver_data, mode)
    return app


def _register_callbacks(
    app: dash.Dash,
    config: GUIConfig,
    records: list[BenchmarkRecord] | None = None,
    opts: FilterOptions | None = None,
    app_by_id: dict[str, ApplicationRecord] | None = None,
    app_dropdown_options: list[DropdownOption] | None = None,
    paraver_data: ParaverData | None = None,
    mode: GUIMode = GUIMode.CARM,
) -> None:
    """Register all application callbacks."""
    _cb_seq: int = 0

    def _tr(msg: str) -> None:
        nonlocal _cb_seq
        _cb_seq += 1
        if config.gui_debug:
            debug(f"[CB#{_cb_seq}] {msg}")

    def _register_setting_callback(
        input_id: str,
        field_name: str,
        converter: Callable[[Any], Any],
        default: Any,
    ) -> None:
        @app.callback(
            Output(StoreID.ROOF_STORE, "data", allow_duplicate=True),
            Input(input_id, "value"),
            State(StoreID.ROOF_STORE, "data"),
            prevent_initial_call=True,
        )
        def _update_setting(
            val: Any,
            store_data: dict[str, Any] | None,
        ) -> dict[str, Any]:
            store = RoofStore.from_dict(store_data or {})
            value = converter(val) if val is not None else default
            store.settings = replace(store.settings, **{field_name: value})
            save_gui_settings(gui_settings_path(), store.settings)
            return store.to_dict()

    def _register_collapse_callback(
        input_type_id: str,
        field_name: str,
        label: str,
    ) -> None:
        @app.callback(
            Output(StoreID.ROOF_STORE, "data", allow_duplicate=True),
            Input({"type": input_type_id, "index": ALL}, "n_clicks"),
            State(StoreID.ROOF_STORE, "data"),
            prevent_initial_call=True,
        )
        def _toggle_collapse(
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
            _tr(f"_{label} trigger={ctx.triggered[0].get('prop_id', '?')} roofs={len(store.roofs)}")
            index = _get_trigger_index()
            if 0 <= index < len(store.roofs):
                old = getattr(store.roofs[index], field_name)
                setattr(store.roofs[index], field_name, not old)
            _tr(f"_{label} exit roofs={len(store.roofs)}")
            return store.to_dict()

    # 1. Toggle active panel
    @app.callback(
        Output(StoreID.ACTIVE_PANEL, "data"),
        Input(NavbarID.BTN_CARM_VIEW, "n_clicks"),
        Input(NavbarID.BTN_SETTINGS, "n_clicks"),
        # allow_optional: the Export button only exists in paraver mode; Dash 4's client
        # renderer hard-errors on a missing Input dependency otherwise.
        Input(NavbarID.BTN_EXPORT, "n_clicks", allow_optional=True),
    )
    def _toggle_panel(
        carm_view_clicks: int | None,
        settings_clicks: int | None,
        export_clicks: int | None,
    ) -> ActivePanel:
        ctx = callback_context
        if not ctx.triggered:
            return ActivePanel.CARM_VIEW
        trigger = ctx.triggered[0]["prop_id"]
        if NavbarID.BTN_EXPORT in trigger:
            return ActivePanel.EXPORT
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
                old_machine = roof.machine
                roof.machine = machine_vals[i]
                if roof.machine != old_machine:
                    roof.app_ids = []
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
        # Refresh labels from current field values
        for roof in store.roofs:
            roof.label = format_roof_label(roof.machine, roof.isa, roof.num_threads, roof.data_type)
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
            fo_dict = cast(dict[str, list[Any]], fo)
            # Stabilization: single pass — clears stale values incompatible with current locks
            for field_name, filter_key, _ in _FILTER_FIELD_SPECS:
                val = getattr(roof, field_name)
                if val is not None and val not in fo_dict[filter_key]:
                    debug(f"_resolve_roof_data[{roof.id}]: {field_name} '{val}' not in options -> None")
                    setattr(roof, field_name, None)

            # Build user-locks filter from stabilized (user-set) values only.
            # Used for per-roof dropdown options so auto-resolved values dont't constrain dropdown menus.
            user_locks = RooflineFilter(**{name: getattr(roof, name) for name, _, _ in _FILTER_FIELD_SPECS})

            acc_dict: dict[str, Any] = {name: getattr(roof, name) for name, _, _ in _FILTER_FIELD_SPECS}

            resolved_kwargs: dict[str, Any] = {
                "roof_id": roof.id,
                "compute_insts": roof.compute_insts,
                "app_ids": roof.app_ids,
            }
            for field_name, filter_key, first_fn in _FILTER_FIELD_SPECS:
                val = getattr(roof, field_name)
                if val is not None:
                    resolved_kwargs[field_name] = val
                else:
                    r_opts = discover_filter_options(recs, RooflineFilter(**acc_dict))
                    resolved_kwargs[field_name] = first_fn(cast(dict[str, list[Any]], r_opts)[filter_key])
                acc_dict[field_name] = resolved_kwargs[field_name]

            resolved_roofs.append(
                RoofConfig(
                    roof_id=roof.id,
                    label=roof.label,
                    machine=resolved_kwargs["machine"],
                    isa=resolved_kwargs["isa"],
                    num_threads=resolved_kwargs["num_threads"],
                    data_type=resolved_kwargs["data_type"],
                    compute_insts=roof.compute_insts,
                    load_store_ratio=resolved_kwargs["load_store_ratio"],
                    actual_frequency_hz=resolved_kwargs["actual_frequency_hz"],
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
                records or [],
                paraver_data,
                filtered_trace,
                settings=store.settings,
                color_mode=store.paraver_state.color_mode,
            )
        else:
            figure = build_roofline_figure(
                resolved_roofs,
                records or [],
                app_by_id or {},
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
        carm_view_panel = build_carm_view_panel(
            store,
            per_roof_opts,
            resolved_roofs,
            per_roof_app_options=_filter_app_options_for_roofs(resolved_roofs, app_dropdown_options, app_by_id),
            include_apps_section=mode.show_app_dropdown,
        )
        settings_panel = build_settings_panel(store, None)
        children = [carm_view_panel, settings_panel]
        if mode.has_export_tab:
            children.append(build_export_panel(store))
        _tr(f"_update_sidebar exit panel={store.active_panel}")
        return children

    # 9-16. Settings controls (factory-generated)
    _register_setting_callback(SettingsPanelID.SWITCH_NORMALIZE, "normalize_by_threads", bool, False)
    _register_setting_callback(SettingsPanelID.SLIDER_MARKER_SIZE, "marker_scale_factor", float, 50.0)
    _register_setting_callback(SettingsPanelID.SWITCH_POWER2_TICKS, "power2_ticks", bool, False)
    _register_setting_callback(SettingsPanelID.SWITCH_SHOW_ROOF_FILLS, "show_roof_fills", bool, True)
    _register_setting_callback(SettingsPanelID.SLIDER_LINE_WIDTH, "line_width", float, 1.5)
    _register_setting_callback(SettingsPanelID.SLIDER_FONT_SIZE_AXIS_LABEL, "axis_label_font_size", int, 14)
    _register_setting_callback(SettingsPanelID.SLIDER_FONT_SIZE_AXIS_TICK, "axis_tick_font_size", int, 12)
    _register_setting_callback(SettingsPanelID.SLIDER_FONT_SIZE_TOOLTIP, "tooltip_font_size", int, 12)
    _register_setting_callback(SettingsPanelID.SLIDER_FONT_SIZE_LEGEND, "legend_font_size", int, 10)

    # 17. Sync button styles with active panel
    @app.callback(
        Output(NavbarID.BTN_CARM_VIEW, "className"),
        Output(NavbarID.BTN_SETTINGS, "className"),
        Input(StoreID.ACTIVE_PANEL, "data"),
    )
    def _update_panel_button_styles(active_panel: str | None) -> tuple[str, str]:
        active_panel = active_panel or ActivePanel.CARM_VIEW
        carm_view_cls = f"navbar-btn{' navbar-btn--active' if active_panel == ActivePanel.CARM_VIEW else ''}"
        settings_cls = f"navbar-btn{' navbar-btn--active' if active_panel == ActivePanel.SETTINGS else ''}"
        return carm_view_cls, settings_cls

    # Export button style — registered only in paraver mode: Dash 4's Output has no
    # allow_optional, and a missing Output target hard-errors the client renderer.
    if mode.has_export_tab:

        @app.callback(
            Output(NavbarID.BTN_EXPORT, "className"),
            Input(StoreID.ACTIVE_PANEL, "data"),
        )
        def _update_export_button_style(active_panel: str | None) -> str:
            active_panel = active_panel or ActivePanel.CARM_VIEW
            return f"navbar-btn{' navbar-btn--active' if active_panel == ActivePanel.EXPORT else ''}"

    # 18. Paraver time-window slider -> per-session window state in ROOF_STORE
    if mode.show_time_slider:

        @app.callback(
            Output(StoreID.ROOF_STORE, "data", allow_duplicate=True),
            Input(ParaverID.SLIDER_TIME_WINDOW, "value"),
            State(StoreID.ROOF_STORE, "data"),
            prevent_initial_call=True,
        )
        def _update_time_window(
            value: list[float] | None,
            store_data: dict[str, Any] | None,
        ) -> dict[str, Any]:
            store = RoofStore.from_dict(store_data or {})
            store.paraver_state.time_window = (value[0], value[1]) if value else None
            return store.to_dict()

    # 19. Export modes (paraver mode only)
    if mode.has_export_tab:
        # 19b. Threshold-filter sliders (AI, duration) -> per-session state in ROOF_STORE
        def _register_threshold_filter_callback(slider_id: str, field_name: str, log_min: float) -> None:
            @app.callback(
                Output(StoreID.ROOF_STORE, "data", allow_duplicate=True),
                Input(slider_id, "value"),
                State(StoreID.ROOF_STORE, "data"),
                prevent_initial_call=True,
            )
            def _update_threshold_filter(
                value: float | None,
                store_data: dict[str, Any] | None,
            ) -> dict[str, Any]:
                store = RoofStore.from_dict(store_data or {})
                # Leftmost slider position (log10 == log_min) means "off".
                threshold: float | None = None if value is None or value <= log_min else 10.0**value
                setattr(store.paraver_state, field_name, threshold)
                return store.to_dict()

        _register_threshold_filter_callback(ExportPanelID.SLIDER_AI_THRESHOLD, "ai_threshold", AI_FILTER_LOG_MIN)
        _register_threshold_filter_callback(
            ExportPanelID.SLIDER_DURATION_THRESHOLD, "duration_threshold", DURATION_FILTER_LOG_MIN
        )

        # 19c. Point-color mode radio -> per-session state in ROOF_STORE
        @app.callback(
            Output(StoreID.ROOF_STORE, "data", allow_duplicate=True),
            Input(ExportPanelID.RADIO_COLOR_MODE, "value"),
            State(StoreID.ROOF_STORE, "data"),
            prevent_initial_call=True,
        )
        def _update_color_mode(value: str | None, store_data: dict[str, Any] | None) -> dict[str, Any]:
            store = RoofStore.from_dict(store_data or {})
            store.paraver_state.color_mode = value or COLOR_MODE_PARAVER
            return store.to_dict()

        # Exports are written next to the .prv trace (legacy os.path.dirname(prv_trace_path)).
        if config.paraver_trace is not None:
            output_dir = Path(config.paraver_trace).parent
        elif config.paraver_window_csv is not None:
            output_dir = Path(config.paraver_window_csv).parent
        else:  # pragma: no cover — paraver mode requires a trace
            output_dir = Path.cwd()

        _EXPORT_MODE_SPECS: tuple[ExportModeSpec, ...] = (
            ExportModeSpec(
                ExportPanelID.BTN_EXPORT_PERFORMANCE, ExportPanelID.STATUS_PERFORMANCE, export_performance, False
            ),
            ExportModeSpec(ExportPanelID.BTN_EXPORT_AI, ExportPanelID.STATUS_AI, export_ai, False),
            ExportModeSpec(
                ExportPanelID.BTN_EXPORT_LDST_PERCENT,
                ExportPanelID.STATUS_LDST_PERCENT,
                export_ldst_percent,
                False,
            ),
            ExportModeSpec(
                ExportPanelID.BTN_EXPORT_ROOF_LABELS, ExportPanelID.STATUS_ROOF_LABELS, export_roof_labels, True
            ),
            ExportModeSpec(ExportPanelID.BTN_EXPORT_REGION, ExportPanelID.STATUS_REGION, export_region, True),
            ExportModeSpec(ExportPanelID.BTN_EXPORT_PROXIMITY, ExportPanelID.STATUS_PROXIMITY, export_proximity, True),
        )
        for _spec in _EXPORT_MODE_SPECS:

            @app.callback(
                Output(_spec.status, "children"),
                Input(_spec.button, "n_clicks"),
                State(StoreID.ROOF_STORE, "data"),
                prevent_initial_call=True,
            )
            def _export_mode(
                n_clicks: int | None,
                store_data: dict[str, Any] | None,
                _spec: ExportModeSpec = _spec,
            ) -> str:
                if not n_clicks:
                    raise PreventUpdate
                if paraver_data is None:
                    return "No paraver trace loaded."
                store = RoofStore.from_dict(store_data or {})
                # Exports cover the whole trace, not the filtered view (paraver
                # needs every timestamp of the loaded window).
                trace = paraver_data.trace
                if trace.empty:
                    return "Nothing to export."
                model: AssembledRoofline | None = None
                divisor = 1
                if _spec.needs_roof:
                    if not store.roofs:
                        return "No roof configured."
                    divisor = roof_divisor(store.roofs[0], store.settings)
                    model = assemble_roofline(records or [], roof_to_filter(store.roofs[0]))
                files = _spec.exporter(trace, paraver_data, model, divisor)
                if not files:
                    return "Nothing to export (no roof data for this mode)."
                try:
                    written = write_export_files(files, output_dir)
                except OSError as exc:
                    return f"Failed to write export to {output_dir}: {exc}"
                for path in written:
                    # Paraver loads the window CSV and finds the .legend.csv file next to it automatically, don't print
                    if path.name.endswith(".legend.csv"):
                        continue
                    print(path, flush=True)  # noqa: T201 ; print to stdout for paraver to capture
                return ""

    # Collapse toggles (factory-generated)
    _register_collapse_callback(RoofCardID.BTN_COLLAPSE_ROOF, "collapsed", "toggle_collapse_roof")
    _register_collapse_callback(RoofCardID.BTN_ADVANCED_COLLAPSE, "advanced_collapsed", "toggle_advanced_collapse")
