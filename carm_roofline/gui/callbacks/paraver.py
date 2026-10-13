from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dash
from dash import Input, Output, State
from dash.exceptions import PreventUpdate

from carm_roofline.gui.callbacks.state import CallbackState
from carm_roofline.gui.colors import COLOR_MODE_PARAVER
from carm_roofline.gui.components import AI_FILTER_LOG_MIN, DURATION_FILTER_LOG_MIN
from carm_roofline.gui.data import RoofStore, roof_divisor, roof_to_filter
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
from carm_roofline.gui.ids import ExportPanelID, ParaverID, StoreID
from carm_roofline.roofline_assembly import AssembledRoofline, assemble_roofline


@dataclass(frozen=True)
class ExportModeSpec:
    """Wiring for one Paraver export mode."""

    button: ExportPanelID
    status: ExportPanelID
    exporter: ExportModeExporter
    needs_roof: bool


def register_paraver_callbacks(app: dash.Dash, state: CallbackState) -> None:
    """Register Paraver filters, color selection, and exports."""
    config = state.config
    records = state.records
    paraver_data = state.paraver_data
    mode = state.mode
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

    # Export controls are available only in Paraver mode.
    if mode.has_export_tab:
        # Threshold filters update the current session state.
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

        # The color radio updates per-session state.
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
                    # Paraver loads the window CSV and finds the .legend.csv file next to it automatically.
                    if path.name.endswith(".legend.csv"):
                        continue
                    print(f"PARAVER_PROTOCOL open_csv {path}", flush=True)  # noqa: T201 ; print for Paraver to capture
                    print(path, flush=True)  # noqa: T201 ; print for Paraver to capture
                return ""
