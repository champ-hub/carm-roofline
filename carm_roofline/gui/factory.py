from __future__ import annotations

import logging

import dash
import dash_bootstrap_components as dbc
from dash import html

from carm_roofline.core.error import UserError
from carm_roofline.gui.callbacks.benchmark import register_benchmark_callbacks
from carm_roofline.gui.callbacks.helpers import filter_app_options_for_roofs
from carm_roofline.gui.callbacks.navigation import register_navigation_callbacks
from carm_roofline.gui.callbacks.paraver import register_paraver_callbacks
from carm_roofline.gui.callbacks.profile import register_profile_callbacks
from carm_roofline.gui.callbacks.roofline import register_roofline_callbacks
from carm_roofline.gui.callbacks.roofs import register_roof_callbacks
from carm_roofline.gui.callbacks.settings import register_settings_callbacks
from carm_roofline.gui.callbacks.state import CallbackState
from carm_roofline.gui.callbacks.welcome import register_welcome_callbacks
from carm_roofline.gui.components import build_layout
from carm_roofline.gui.config import (
    GUIConfig,
    GUIMode,
    gui_settings_path,
    load_gui_settings,
)
from carm_roofline.gui.data import DropdownOption, RoofStore, make_default_roof
from carm_roofline.gui.providers import BenchmarkAppsProvider, ParaverData, ParaverProvider, trace_time_range
from carm_roofline.output_utils import debug, warn
from carm_roofline.roofline_assembly import ApplicationRecord, discover_filter_options, load_all_benchmarks


def create_app(config: GUIConfig) -> dash.Dash:
    """Create and return a configured Dash application."""
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        assets_folder="assets",
        suppress_callback_exceptions=True,
        title="CARM Tool",
    )

    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("dash").setLevel(logging.DEBUG)
    mode = GUIMode.PARAVER if config.paraver_trace is not None else GUIMode.CARM

    records = load_all_benchmarks(config.results_dir) if config.results_dir.exists() else []
    apps_by_id: dict[str, ApplicationRecord] = {}
    app_options: list[DropdownOption] = []
    trace_bounds: tuple[float, float] | None = None
    initial_window: tuple[float, float] | None = None
    paraver_data: ParaverData | None = None

    if mode.show_time_slider and config.paraver_trace is not None:
        provider = ParaverProvider(config.paraver_trace, config.paraver_window_csv)
        try:
            paraver_data = provider.load()
        except UserError as exc:
            warn(str(exc))
        if paraver_data is not None:
            trace_bounds = trace_time_range(paraver_data.trace)
            if config.paraver_use_semantic_window:
                initial_window = provider.window_extent
    elif config.results_dir.exists():
        apps_by_id = BenchmarkAppsProvider(config.results_dir).load()
        app_options = [{"label": app.label, "value": app.id} for app in apps_by_id.values()]

    options = discover_filter_options(records) if records else None
    if options is not None:
        debug(f"Available machines: {options['machine']}")
        debug(f"Available ISAs: {options['isa']}")
        debug(f"Available threads: {options['num_threads']}")
        debug(f"Available load-store ratios: {options['load_store_ratio']}")
    state = CallbackState(config, records, options, apps_by_id, app_options, paraver_data, mode)
    initial_roof = make_default_roof(options)

    def serve_layout() -> html.Div:
        saved = load_gui_settings(gui_settings_path())
        fresh_store = RoofStore(roof_template=initial_roof)
        fresh_store.roofs = []
        fresh_store.settings = saved
        if initial_window is not None:
            fresh_store.paraver_state.time_window = initial_window
        return build_layout(
            fresh_store,
            options,
            filter_app_options_for_roofs([initial_roof], app_options, apps_by_id),
            mode,
            trace_bounds,
            str(config.results_dir),
        )

    app.layout = serve_layout
    _register_callbacks(app, state)
    return app


def _register_callbacks(app: dash.Dash, state: CallbackState) -> None:
    """Register GUI callbacks by domain."""
    register_navigation_callbacks(app, state)
    register_benchmark_callbacks(app, state)
    register_profile_callbacks(app, state)
    register_roof_callbacks(app, state)
    register_roofline_callbacks(app, state)
    register_settings_callbacks(app)
    register_welcome_callbacks(app)
    register_paraver_callbacks(app, state)
