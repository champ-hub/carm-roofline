from __future__ import annotations

import dash
from dash import Input, Output, callback_context

from carm_roofline.gui.callbacks.state import CallbackState
from carm_roofline.gui.data import ActivePanel
from carm_roofline.gui.ids import BenchmarkPanelID, NavbarID, PlotAreaID, ProfilePanelID, StoreID


def register_navigation_callbacks(app: dash.Dash, state: CallbackState) -> None:
    """Register navigation controls and active-panel presentation callbacks."""

    @app.callback(
        Output(StoreID.ACTIVE_PANEL, "data"),
        Input(NavbarID.BTN_CARM_VIEW, "n_clicks"),
        Input(NavbarID.BTN_BENCHMARK, "n_clicks"),
        Input(NavbarID.BTN_PROFILE, "n_clicks"),
        Input(NavbarID.BTN_SETTINGS, "n_clicks"),
        Input(NavbarID.BTN_EXPORT, "n_clicks", allow_optional=True),
    )
    def _toggle_panel(
        carm_view_clicks: int | None,
        benchmark_clicks: int | None,
        profile_clicks: int | None,
        settings_clicks: int | None,
        export_clicks: int | None,
    ) -> ActivePanel:
        context = callback_context
        if not context.triggered:
            return ActivePanel.CARM_VIEW
        trigger = context.triggered[0]["prop_id"]
        if NavbarID.BTN_EXPORT in trigger:
            return ActivePanel.EXPORT
        if NavbarID.BTN_BENCHMARK in trigger:
            return ActivePanel.BENCHMARK
        if NavbarID.BTN_PROFILE in trigger:
            return ActivePanel.PROFILE
        if NavbarID.BTN_SETTINGS in trigger:
            return ActivePanel.SETTINGS
        return ActivePanel.CARM_VIEW

    @app.callback(
        Output(NavbarID.BTN_CARM_VIEW, "className"),
        Output(NavbarID.BTN_BENCHMARK, "className"),
        Output(NavbarID.BTN_PROFILE, "className"),
        Output(NavbarID.BTN_SETTINGS, "className"),
        Output(BenchmarkPanelID.PAGE, "style"),
        Output(ProfilePanelID.PAGE, "style"),
        Output(PlotAreaID.CONTAINER, "style"),
        Input(StoreID.ACTIVE_PANEL, "data"),
    )
    def _update_panel_button_styles(
        active_panel: str | None,
    ) -> tuple[str, str, str, str, dict[str, str], dict[str, str], dict[str, str]]:
        active_panel = active_panel or ActivePanel.CARM_VIEW
        carm_view_class = f"navbar-btn{' navbar-btn--active' if active_panel == ActivePanel.CARM_VIEW else ''}"
        benchmark_class = f"navbar-btn{' navbar-btn--active' if active_panel == ActivePanel.BENCHMARK else ''}"
        profile_class = f"navbar-btn{' navbar-btn--active' if active_panel == ActivePanel.PROFILE else ''}"
        settings_class = f"navbar-btn{' navbar-btn--active' if active_panel == ActivePanel.SETTINGS else ''}"
        benchmark_style = {"display": "block"} if active_panel == ActivePanel.BENCHMARK else {"display": "none"}
        profile_style = {"display": "block"} if active_panel == ActivePanel.PROFILE else {"display": "none"}
        plot_style = (
            {"display": "none"} if active_panel in (ActivePanel.BENCHMARK, ActivePanel.PROFILE) else {"display": "flex"}
        )
        return (
            carm_view_class,
            benchmark_class,
            profile_class,
            settings_class,
            benchmark_style,
            profile_style,
            plot_style,
        )

    if state.mode.has_export_tab:

        @app.callback(
            Output(NavbarID.BTN_EXPORT, "className"),
            Input(StoreID.ACTIVE_PANEL, "data"),
        )
        def _update_export_button_style(active_panel: str | None) -> str:
            active_panel = active_panel or ActivePanel.CARM_VIEW
            return f"navbar-btn{' navbar-btn--active' if active_panel == ActivePanel.EXPORT else ''}"
