from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

import dash
from dash import Input, Output, State

from carm_roofline.gui.config import gui_settings_path, load_gui_settings, save_gui_settings
from carm_roofline.gui.data import RoofStore
from carm_roofline.gui.ids import SettingsPanelID, StoreID


def register_settings_callbacks(app: dash.Dash) -> None:
    """Register persistent plotting-setting controls."""
    settings = (
        (SettingsPanelID.SWITCH_NORMALIZE, "normalize_by_threads", bool, False),
        (SettingsPanelID.SLIDER_MARKER_SIZE, "marker_scale_factor", float, 50.0),
        (SettingsPanelID.SLIDER_MARKER_BASE_SIZE, "marker_base_size", float, 50.0),
        (SettingsPanelID.SWITCH_POWER2_TICKS, "power2_ticks", bool, False),
        (SettingsPanelID.DROPDOWN_ROOF_FILL_MODE, "roof_fill_mode", str, "between"),
        (SettingsPanelID.SWITCH_SHOW_MIXED_BENCHMARKS, "show_mixed_benchmarks", bool, False),
        (SettingsPanelID.SLIDER_LINE_WIDTH, "line_width", float, 1.5),
        (SettingsPanelID.SLIDER_FONT_SIZE_AXIS_LABEL, "axis_label_font_size", int, 14),
        (SettingsPanelID.SLIDER_FONT_SIZE_AXIS_TICK, "axis_tick_font_size", int, 12),
        (SettingsPanelID.SLIDER_FONT_SIZE_TOOLTIP, "tooltip_font_size", int, 12),
        (SettingsPanelID.SLIDER_FONT_SIZE_LEGEND, "legend_font_size", int, 10),
        (SettingsPanelID.SLIDER_LEGEND_LABEL_LENGTH, "legend_label_length", int, 32),
    )
    for input_id, field_name, converter, default in settings:
        _register_setting_callback(app, input_id, field_name, converter, default)


def _register_setting_callback(
    app: dash.Dash,
    input_id: SettingsPanelID,
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
    def _update_setting(val: Any, store_data: dict[str, Any] | None) -> dict[str, Any]:
        store = RoofStore.from_dict(store_data or {})
        value = converter(val) if val is not None else default
        store.settings = replace(store.settings, **{field_name: value})
        path = gui_settings_path()
        persisted = load_gui_settings(path)
        save_gui_settings(path, replace(persisted, **{field_name: value}))
        return store.to_dict()
