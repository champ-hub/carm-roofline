from __future__ import annotations

from typing import Any

import dash
from dash import ALL, MATCH, Input, Output, State, callback_context
from dash.exceptions import PreventUpdate

from carm_roofline.gui.callbacks.helpers import apply_roof_controls, decode_roof_control_patches
from carm_roofline.gui.callbacks.resolution import normalize_selected_roof_controls
from carm_roofline.gui.callbacks.state import CallbackState
from carm_roofline.gui.data import RoofStore, make_default_roof
from carm_roofline.gui.ids import CarmViewPanelID, RoofCardID, StoreID


def register_roof_callbacks(app: dash.Dash, state: CallbackState) -> None:
    """Register add, remove, and collapse actions for roof cards."""

    @app.callback(
        Output(StoreID.ROOF_STORE, "data"),
        Input(CarmViewPanelID.BTN_ADD_ROOF, "n_clicks"),
        State(StoreID.ROOF_STORE, "data"),
    )
    def _add_roof(n_clicks: int | None, store_data: dict[str, Any] | None) -> dict[str, Any]:
        if not n_clicks:
            raise PreventUpdate
        store = RoofStore.from_dict(store_data or {})
        state.trace(f"_add_roof enter roofs={len(store.roofs)}")
        store.add_roof(roof_template=make_default_roof(state.options))
        state.trace(f"_add_roof exit  roofs={len(store.roofs)}")
        return store.to_dict()

    @app.callback(
        Output(StoreID.ROOF_STORE, "data", allow_duplicate=True),
        Input({"type": RoofCardID.BTN_REMOVE_ROOF, "roof_id": MATCH}, "n_clicks"),
        State(StoreID.ROOF_STORE, "data"),
        State({"type": RoofCardID.DROPDOWN_MACHINE, "roof_id": ALL}, "value"),
        State({"type": RoofCardID.DROPDOWN_ISA, "roof_id": ALL}, "value"),
        State({"type": RoofCardID.DROPDOWN_THREADS, "roof_id": ALL}, "value"),
        State({"type": RoofCardID.DROPDOWN_COMPUTE, "roof_id": ALL}, "value"),
        State({"type": RoofCardID.DROPDOWN_DATA_TYPE, "roof_id": ALL}, "value"),
        State({"type": RoofCardID.DROPDOWN_LS_RATIO, "roof_id": ALL}, "value"),
        State({"type": RoofCardID.DROPDOWN_FREQUENCY, "roof_id": ALL}, "value"),
        State({"type": RoofCardID.DROPDOWN_APPS, "roof_id": ALL}, "value"),
        prevent_initial_call=True,
    )
    def _remove_roof(
        n_clicks: int | None,
        store_data: dict[str, Any] | None,
        *_control_state: Any,
    ) -> dict[str, Any]:
        context = callback_context
        if not n_clicks:
            raise PreventUpdate
        trigger_id = context.triggered_id
        if not isinstance(trigger_id, dict):
            raise PreventUpdate
        roof_id = trigger_id.get("roof_id")
        if not isinstance(roof_id, str):
            raise PreventUpdate
        store = RoofStore.from_dict(store_data or {})
        apply_roof_controls(store, decode_roof_control_patches(context.states_list))
        try:
            store.remove_roof(roof_id)
        except KeyError:
            raise PreventUpdate from None
        normalize_selected_roof_controls(state, store)
        state.trace(f"_remove_roof exit roofs={len(store.roofs)}")
        return store.to_dict()

    _register_collapse_callback(app, state, RoofCardID.BTN_COLLAPSE_ROOF, "collapsed", "toggle_collapse_roof")
    _register_collapse_callback(
        app, state, RoofCardID.BTN_ADVANCED_COLLAPSE, "advanced_collapsed", "toggle_advanced_collapse"
    )


def _register_collapse_callback(
    app: dash.Dash,
    state: CallbackState,
    input_type_id: str,
    field_name: str,
    label: str,
) -> None:
    @app.callback(
        Output(StoreID.ROOF_STORE, "data", allow_duplicate=True),
        Input({"type": input_type_id, "roof_id": MATCH}, "n_clicks"),
        State(StoreID.ROOF_STORE, "data"),
        prevent_initial_call=True,
    )
    def _toggle_collapse(
        n_clicks: int | None,
        store_data: dict[str, Any] | None,
    ) -> dict[str, Any]:
        context = callback_context
        if not n_clicks:
            raise PreventUpdate
        trigger_id = context.triggered_id
        if not isinstance(trigger_id, dict):
            raise PreventUpdate
        roof_id = trigger_id.get("roof_id")
        if not isinstance(roof_id, str):
            raise PreventUpdate
        store = RoofStore.from_dict(store_data or {})
        roof = next((item for item in store.roofs if item.id == roof_id), None)
        if roof is None:
            raise PreventUpdate
        state.trace(f"_{label} trigger={trigger_id} roofs={len(store.roofs)}")
        old_value = getattr(roof, field_name)
        setattr(roof, field_name, not old_value)
        state.trace(f"_{label} exit roofs={len(store.roofs)}")
        return store.to_dict()
