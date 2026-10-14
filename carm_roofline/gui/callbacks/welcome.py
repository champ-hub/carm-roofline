from __future__ import annotations

from dataclasses import replace

import dash
from dash import Input, Output

from carm_roofline.gui.config import gui_settings_path, load_gui_settings, save_gui_settings
from carm_roofline.gui.ids import WelcomePanelID


def register_welcome_callbacks(app: dash.Dash) -> None:
    """Register dismissal and persistent opt-out for the welcome panel."""

    @app.callback(
        Output(WelcomePanelID.CONTAINER, "style"),
        Input(WelcomePanelID.GOT_IT, "n_clicks"),
        Input(WelcomePanelID.DO_NOT_SHOW_AGAIN, "value"),
        prevent_initial_call=True,
    )
    def _dismiss_welcome(clicks: int | None, dont_show_again: list[str] | None) -> dict[str, str]:
        opted_out = "hide" in (dont_show_again or [])
        if opted_out:
            path = gui_settings_path()
            settings = load_gui_settings(path)
            save_gui_settings(path, replace(settings, show_welcome_panel=False))
        if opted_out or clicks:
            return {"display": "none"}
        return {"display": "block"}
