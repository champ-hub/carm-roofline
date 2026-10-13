from __future__ import annotations

from typing import Any

import dash
from dash import ALL, Input, Output, State, callback_context
from dash.exceptions import PreventUpdate

from carm_roofline.gui.callbacks.helpers import get_trigger_index
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
        Input({"type": RoofCardID.BTN_REMOVE_ROOF, "index": ALL}, "n_clicks"),
        State(StoreID.ROOF_STORE, "data"),
        prevent_initial_call=True,
    )
    def _remove_roof(
        n_clicks_list: list[int | None],
        store_data: dict[str, Any] | None,
    ) -> dict[str, Any]:
        context = callback_context
        if not context.triggered or not context.triggered[0].get("value", 0):
            raise PreventUpdate
        store = RoofStore.from_dict(store_data or {})
        state.trace(f"_remove_roof enter trigger={context.triggered[0].get('prop_id', '?')} roofs={len(store.roofs)}")
        index = get_trigger_index()
        if 0 <= index < len(store.roofs):
            store.remove_roof(store.roofs[index].id)
        state.trace(f"_remove_roof exit  roofs={len(store.roofs)}")
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
        Input({"type": input_type_id, "index": ALL}, "n_clicks"),
        State(StoreID.ROOF_STORE, "data"),
        prevent_initial_call=True,
    )
    def _toggle_collapse(
        n_clicks: list[int | None],
        store_data: dict[str, Any] | None,
    ) -> dict[str, Any]:
        context = callback_context
        if not context.triggered or not context.triggered[0].get("value", 0):
            raise PreventUpdate
        store = RoofStore.from_dict(store_data or {})
        state.trace(f"_{label} trigger={context.triggered[0].get('prop_id', '?')} roofs={len(store.roofs)}")
        index = get_trigger_index()
        if 0 <= index < len(store.roofs):
            old_value = getattr(store.roofs[index], field_name)
            setattr(store.roofs[index], field_name, not old_value)
        state.trace(f"_{label} exit roofs={len(store.roofs)}")
        return store.to_dict()
