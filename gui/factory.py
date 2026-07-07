from __future__ import annotations

import json
import logging
from typing import Any, cast

import dash
import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback_context, html
from dash.exceptions import PreventUpdate

from gui.components import build_carm_view_panel, build_layout, build_settings_panel
from gui.config import GUIConfig
from gui.data import ActivePanel, DropdownOption, RoofConfig, RoofStore, build_roofline_figure, make_default_roof
from gui.ids import (
    CarmViewPanelID,
    NavbarID,
    PlotAreaID,
    RoofCardID,
    SettingsPanelID,
    SidebarID,
    StoreID,
)
from output_utils import debug
from roofline_assembly import (
    ApplicationRecord,
    BenchmarkRecord,
    FilterOptions,
    discover_filter_options,
    discover_filter_options_for_selection,
    load_all_applications,
    load_all_benchmarks,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


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


# ── App factory ────────────────────────────────────────────────────────────────


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
        debug(f"Available threads: {opts['threads']}")
        debug(f"Available load-store ratios: {opts['load_store_ratio']}")

    # Build a sensible default roof config from the first available options
    initial_roof = make_default_roof(opts)
    initial_store = RoofStore(roof_template=initial_roof)
    initial_store.roofs = [initial_roof]

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

    # ── 1. Toggle active panel ─────────────────────────────────────────────────
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

    # ── 2. Add roof ────────────────────────────────────────────────────────────
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

    # ── 3. Remove roof ─────────────────────────────────────────────────────────
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

    # ── 8. Regenerate plot + sidebar on store, panel, or UI change ────────
    @app.callback(
        Output(PlotAreaID.ROOFLINE_PLOT, "figure"),
        Output(SidebarID.SIDEBAR_CONTENT, "children"),
        Input(StoreID.ROOF_STORE, "data"),
        Input(StoreID.ACTIVE_PANEL, "data"),
        Input({"type": RoofCardID.DROPDOWN_MACHINE, "index": ALL}, "value"),
        Input({"type": RoofCardID.DROPDOWN_ISA, "index": ALL}, "value"),
        Input({"type": RoofCardID.DROPDOWN_THREADS, "index": ALL}, "value"),
        Input({"type": RoofCardID.DROPDOWN_COMPUTE, "index": ALL}, "value"),
        Input({"type": RoofCardID.DROPDOWN_DATA_TYPE, "index": ALL}, "value"),
        Input({"type": RoofCardID.DROPDOWN_LS_RATIO, "index": ALL}, "value"),
        Input({"type": RoofCardID.DROPDOWN_APPS, "index": ALL}, "value"),
    )
    def _update_plot_and_sidebar(
        store_data: dict[str, Any] | None,
        active_panel: str | None,
        machine_vals: list[str | None],
        isa_vals: list[str | None],
        threads_vals: list[str | None],
        compute_vals: list[list[str] | None],
        data_type_vals: list[str | None],
        ls_ratio_vals: list[str | None],
        app_ids_vals: list[list[str] | None],
    ) -> tuple[dict[str, Any], list[html.Div]]:
        store = RoofStore.from_dict(store_data or {})
        store.active_panel = ActivePanel(active_panel) if active_panel else ActivePanel.CARM_VIEW
        _tr(
            f"_update_plot_and_sidebar enter "
            f"roofs={len(store.roofs)} "
            f"mach={len(machine_vals)} isa={len(isa_vals)} "
            f"thr={len(threads_vals)} comp={len(compute_vals)} "
            f"dt={len(data_type_vals)} ls={len(ls_ratio_vals)} "
            f"panel={store.active_panel}"
        )
        # Merge live UI values into the store model so the figure and sidebar reflect the latest user choices regardless
        # of which Input triggered.
        for i, roof in enumerate(store.roofs):
            if i < len(machine_vals):
                roof.machine = machine_vals[i]
            if i < len(isa_vals):
                roof.isa = isa_vals[i]
            if i < len(threads_vals):
                thr = threads_vals[i]
                if thr is not None:
                    roof.threads = int(thr)
                else:
                    roof.threads = None
            if i < len(compute_vals):
                cv = compute_vals[i]
                if cv is not None:
                    roof.compute_insts = cv
            if i < len(data_type_vals):
                roof.data_type = data_type_vals[i]
            if i < len(ls_ratio_vals):
                roof.load_store_ratio = ls_ratio_vals[i]
            if i < len(app_ids_vals):
                roof.app_ids = list(app_ids_vals[i] or [])
        debug(f"_update_plot_and_sidebar: {len(store.roofs)} roof(s), panel={store.active_panel}")
        recs = records or []
        per_roof_opts: list[FilterOptions | None] = []
        resolved_roofs: list[RoofConfig] = []

        for roof in store.roofs:
            if not recs:
                per_roof_opts.append(None)
                resolved_roofs.append(roof)
                continue

            # Stabilize: clear fields whose value isn't in the cross-filtered options.
            changed = True
            while changed:
                changed = False
                fo = discover_filter_options_for_selection(
                    recs,
                    machine=roof.machine,
                    isa=roof.isa,
                    num_threads=roof.threads,
                    data_type=roof.data_type,
                    load_store_ratio=roof.load_store_ratio,
                )
                if roof.machine is not None and roof.machine not in fo["machine"]:
                    roof.machine = None
                    changed = True
                if roof.isa is not None and roof.isa not in fo["isa"]:
                    roof.isa = None
                    changed = True
                if roof.threads is not None and roof.threads not in fo["threads"]:
                    roof.threads = None
                    changed = True
                if roof.data_type is not None and roof.data_type not in fo["data_type"]:
                    roof.data_type = None
                    changed = True
                if roof.load_store_ratio is not None and roof.load_store_ratio not in fo["load_store_ratio"]:
                    roof.load_store_ratio = None
                    changed = True

            # Final filtered options for display (after stabilization)
            per_roof_opts.append(fo)

            # Resolve None -> first available value for the plot
            resolved_roofs.append(
                RoofConfig(
                    roof_id=roof.id,
                    label=roof.label,
                    machine=roof.machine if roof.machine is not None else _first_or_none(fo["machine"]),
                    isa=roof.isa if roof.isa is not None else _first_or_none(fo["isa"]),
                    threads=roof.threads if roof.threads is not None else _first_int_or_none(fo["threads"]),
                    data_type=roof.data_type if roof.data_type is not None else _first_or_none(fo["data_type"]),
                    compute_insts=roof.compute_insts,
                    load_store_ratio=roof.load_store_ratio
                    if roof.load_store_ratio is not None
                    else _first_or_none(fo["load_store_ratio"]),
                    app_ids=roof.app_ids,
                )
            )

        figure = build_roofline_figure(resolved_roofs, recs, app_by_id, normalize_by_threads=store.normalize_by_threads)
        figure_dict = cast("dict[str, Any]", figure.to_dict())
        carm_view_panel = build_carm_view_panel(store, per_roof_opts, resolved_roofs, app_dropdown_options)
        settings_panel = build_settings_panel(store, None)
        _tr(f"_update_plot_and_sidebar exit traces={len(figure_dict.get('data', []))}")
        return figure_dict, [carm_view_panel, settings_panel]

    # ── 9. Normalize by threads toggle ─────────────────────────────────────────
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
        store.normalize_by_threads = bool(normalize)
        return store.to_dict()

    # ── 12. Sync button styles with active panel ──────────────────────────────
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

    # ── 14. Toggle roof card collapse ──────────────────────────────────────────
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
