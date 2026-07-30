from __future__ import annotations

from typing import Any, Callable

import dash_bootstrap_components as dbc
from dash import dcc, html

from carm_roofline.core.units import Frequency
from carm_roofline.gui.data import (
    COMPUTE_INST_OPTIONS,
    DATA_TYPE_OPTIONS,
    FREQUENCY_OPTIONS,
    ActivePanel,
    DropdownOption,
    RoofConfig,
    RoofStore,
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
from carm_roofline.roofline_assembly import FilterOptions

# Static fallback options for dropdowns (overridden at runtime when data is loaded).
# The actual available values depend on the loaded JSONL benchmark records.
MACHINE_OPTIONS = ["Machine A", "Machine B", "Machine C"]
ISA_OPTIONS = ["x86_avx512", "x86_avx2", "x86_avx", "x86_sse", "arm_neon", "riscv_rvv"]
LOAD_STORE_RATIO_OPTIONS = ["2:1", "1:0", "0:1", "1:1", "3:1", "4:1"]
THREADS_OPTIONS = ["1", "2", "4", "8", "16", "32", "64", "128"]

# Maps placeholder var suffix -> (roof_config_field, display_fn)
_PLACEHOLDER_SPEC: dict[str, tuple[str, Callable[..., Any]]] = {
    "machine": ("machine", str),
    "frequency": ("actual_frequency_hz", lambda v: str(Frequency(v))),
    "isa": ("isa", str),
    "threads": ("num_threads", str),
    "data_type": ("data_type", str),
    "ls_ratio": ("load_store_ratio", str),
}


def _make_id(type_: str, **parts: int) -> dict[str, str | int]:
    """Build a pattern-matching ID dict."""
    d: dict[str, str | int] = {"type": type_}
    d.update(parts)
    return d


def _slider_marks(*values: float | int) -> dict[float | int, str]:
    """Generate marks dict for dcc.Slider from explicit label values."""
    return {v: str(v) for v in values}


_SWITCH_CONTROLS: list[tuple[str, str, str]] = [
    ("Normalize performance by threads", SettingsPanelID.SWITCH_NORMALIZE, "normalize_by_threads"),
    ("2^N axis tick labels", SettingsPanelID.SWITCH_POWER2_TICKS, "power2_ticks"),
]

_SLIDER_CONTROLS: list[tuple[str, str, str, dict[str, Any]]] = [
    (
        "Point size multiplier",
        SettingsPanelID.SLIDER_MARKER_SIZE,
        "marker_scale_factor",
        {
            "min": 0,
            "max": 200,
            "step": 1,
            "marks": _slider_marks(0, 50, 100, 150, 200),
        },
    ),
    (
        "Line width",
        SettingsPanelID.SLIDER_LINE_WIDTH,
        "line_width",
        {
            "min": 0.5,
            "max": 5.0,
            "step": 0.25,
            "marks": _slider_marks(0.5, 1, 2, 3, 4, 5),
        },
    ),
    (
        "Axis label font size",
        SettingsPanelID.SLIDER_FONT_SIZE_AXIS_LABEL,
        "axis_label_font_size",
        {
            "min": 8,
            "max": 24,
            "step": 1,
            "marks": _slider_marks(8, 12, 16, 20, 24),
        },
    ),
    (
        "Axis tick font size",
        SettingsPanelID.SLIDER_FONT_SIZE_AXIS_TICK,
        "axis_tick_font_size",
        {
            "min": 8,
            "max": 24,
            "step": 1,
            "marks": _slider_marks(8, 12, 16, 20, 24),
        },
    ),
    (
        "Tooltip font size",
        SettingsPanelID.SLIDER_FONT_SIZE_TOOLTIP,
        "tooltip_font_size",
        {
            "min": 8,
            "max": 24,
            "step": 1,
            "marks": _slider_marks(8, 12, 16, 20, 24),
        },
    ),
    (
        "Legend font size",
        SettingsPanelID.SLIDER_FONT_SIZE_LEGEND,
        "legend_font_size",
        {
            "min": 8,
            "max": 24,
            "step": 1,
            "marks": _slider_marks(8, 12, 16, 20, 24),
        },
    ),
]


def _build_settings_switch(label: str, id_: str, value: bool) -> html.Div:
    return html.Div(
        className="settings-toggle-row",
        children=[
            html.Span(label, className="settings-toggle-label"),
            dbc.Switch(id=id_, value=value, className="normalize-toggle"),
        ],
    )


def _build_settings_slider(label: str, id_: str, value: float, **kwargs: Any) -> html.Div:
    return html.Div(
        className="settings-slider-row",
        children=[
            html.Span(label, className="settings-toggle-label"),
            dcc.Slider(
                id=id_,
                value=value,
                **kwargs,
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ],
    )


def build_navbar(active_panel: ActivePanel) -> html.Div:
    """Black top navbar with logo and panel-toggle buttons."""
    carm_view_active = active_panel == ActivePanel.CARM_VIEW
    settings_active = active_panel == ActivePanel.SETTINGS

    return html.Div(
        className="navbar",
        children=[
            # Left: logo + title
            html.Div(
                className="navbar-left",
                children=[
                    html.Img(
                        src="/assets/CARM_icon.svg",
                        className="navbar-logo",
                    ),
                    html.Span("CARM Roofline", className="navbar-title"),
                ],
            ),
            # Right: toggle buttons
            html.Div(
                className="navbar-right",
                children=[
                    html.Button(
                        "CARM View",
                        id=NavbarID.BTN_CARM_VIEW,
                        className=f"navbar-btn{' navbar-btn--active' if carm_view_active else ''}",
                        n_clicks=0,
                    ),
                    html.Button(
                        "Settings",
                        id=NavbarID.BTN_SETTINGS,
                        className=f"navbar-btn{' navbar-btn--active' if settings_active else ''}",
                        n_clicks=0,
                    ),
                ],
            ),
        ],
    )


def build_sidebar(
    store: RoofStore, roof_options: FilterOptions | None = None, app_options: list[DropdownOption] | None = None
) -> html.Div:
    """Left sidebar containing settings or data-selection panel."""
    return html.Div(
        className="sidebar",
        children=[
            # Hidden store for active panel
            html.Div(
                id=SidebarID.SIDEBAR_CONTENT,
                children=[
                    build_carm_view_panel(
                        store, [roof_options] * len(store.roofs) if roof_options else None, app_options=app_options
                    ),
                    build_settings_panel(store, roof_options),
                ],
            ),
        ],
    )


def build_carm_view_panel(
    store: RoofStore,
    per_roof_options: list[FilterOptions | None] | None = None,
    resolved_roofs: list[RoofConfig] | None = None,
    app_options: list[DropdownOption] | None = None,
) -> html.Div:
    """CARM View panel listing all roof configuration cards."""
    return html.Div(
        className="carm-view-panel",
        style={"display": "block"} if store.active_panel == ActivePanel.CARM_VIEW else {"display": "none"},
        children=[
            html.H5("Roof Configurations", className="panel-header"),
            *[
                build_roof_card(
                    roof,
                    i,
                    per_roof_options[i] if per_roof_options and i < len(per_roof_options) else None,
                    resolved_roofs[i] if resolved_roofs and i < len(resolved_roofs) else None,
                    app_options,
                )
                for i, roof in enumerate(store.roofs)
            ],
            html.Button(
                "+ Add Roof",
                id=CarmViewPanelID.BTN_ADD_ROOF,
                className="btn-add-roof",
                n_clicks=0,
            ),
        ],
    )


def build_roof_card(
    roof: RoofConfig,
    index: int,
    options: FilterOptions | None = None,
    resolved_roof: RoofConfig | None = None,
    app_options: list[DropdownOption] | None = None,
) -> dbc.Card:
    # Placeholder text for cleared dropdowns that have an auto-resolved value
    placeholders = {}
    if resolved_roof is not None:
        for var_suffix, (field, display_fn) in _PLACEHOLDER_SPEC.items():
            val = getattr(roof, field, None)
            if val is None:
                resolved_val = getattr(resolved_roof, field, None)
                if resolved_val is not None:
                    placeholders[var_suffix] = f"{display_fn(resolved_val)} (auto)"

    ph_machine = placeholders.get("machine")
    ph_frequency = placeholders.get("frequency")
    ph_isa = placeholders.get("isa")
    ph_threads = placeholders.get("threads")
    ph_data_type = placeholders.get("data_type")
    ph_ls_ratio = placeholders.get("ls_ratio")
    return dbc.Card(
        className="roof-card" + (" roof-card--collapsed" if roof.collapsed else ""),
        children=[
            # Header row
            html.Div(
                className="roof-card-header",
                children=[
                    html.Button(
                        "\u25b6" if roof.collapsed else "\u25bc",
                        id=_make_id(RoofCardID.BTN_COLLAPSE_ROOF, index=index),
                        className="btn-collapse-roof",
                        n_clicks=0,
                    ),
                    html.Span(roof.label, className="roof-card-title"),
                    html.Button(
                        "\u00d7",
                        id=_make_id(RoofCardID.BTN_REMOVE_ROOF, index=index),
                        className="btn-remove-roof",
                        n_clicks=0,
                    ),
                ],
            ),
            # Collapsible fields
            dbc.Collapse(
                is_open=not roof.collapsed,
                children=html.Div(
                    className="roof-card-body",
                    children=[
                        _field_row(
                            "Machine",
                            dcc_dropdown(
                                _make_id(RoofCardID.DROPDOWN_MACHINE, index=index),
                                options["machine"] if options else MACHINE_OPTIONS,
                                roof.machine,
                                placeholder=ph_machine,
                            ),
                        ),
                        _field_row_pair(
                            _field_row(
                                "ISA",
                                dcc_dropdown(
                                    _make_id(RoofCardID.DROPDOWN_ISA, index=index),
                                    options["isa"] if options else ISA_OPTIONS,
                                    roof.isa,
                                    placeholder=ph_isa,
                                ),
                            ),
                            _field_row(
                                "Threads",
                                dcc_dropdown(
                                    _make_id(RoofCardID.DROPDOWN_THREADS, index=index),
                                    [str(t) for t in options["num_threads"]] if options else THREADS_OPTIONS,
                                    str(roof.num_threads) if roof.num_threads is not None else None,
                                    placeholder=ph_threads,
                                ),
                            ),
                        ),
                        _field_row(
                            "Data Type",
                            dcc_dropdown(
                                _make_id(RoofCardID.DROPDOWN_DATA_TYPE, index=index),
                                options["data_type"] if options else DATA_TYPE_OPTIONS,
                                roof.data_type,
                                placeholder=ph_data_type,
                            ),
                        ),
                        _field_row(
                            "Compute Inst",
                            _multi_dropdown(
                                _make_id(RoofCardID.DROPDOWN_COMPUTE, index=index),
                                [{"label": o, "value": o} for o in COMPUTE_INST_OPTIONS],
                                roof.compute_insts,
                            ),
                        ),
                        # Advanced subsection (collapsible, collapsed by default)
                        html.Div(
                            className="advanced-section",
                            children=[
                                html.Div(
                                    className="advanced-section-header",
                                    children=[
                                        html.Span("Advanced", className="advanced-section-title"),
                                        html.Button(
                                            "\u25b6" if roof.advanced_collapsed else "\u25bc",
                                            id=_make_id(RoofCardID.BTN_ADVANCED_COLLAPSE, index=index),
                                            className="btn-advanced-collapse",
                                            n_clicks=0,
                                        ),
                                    ],
                                ),
                                dbc.Collapse(
                                    is_open=not roof.advanced_collapsed,
                                    children=[
                                        _field_row(
                                            "Frequency",
                                            dcc_dropdown(
                                                _make_id(RoofCardID.DROPDOWN_FREQUENCY, index=index),
                                                [(str(Frequency(hz)), str(hz)) for hz in options["actual_frequency_hz"]]
                                                if options
                                                else FREQUENCY_OPTIONS,
                                                str(roof.actual_frequency_hz)
                                                if roof.actual_frequency_hz is not None
                                                else None,
                                                placeholder=ph_frequency,
                                            ),
                                        ),
                                        _field_row(
                                            "Load-Store Ratio",
                                            dcc_dropdown(
                                                _make_id(RoofCardID.DROPDOWN_LS_RATIO, index=index),
                                                options["load_store_ratio"] if options else LOAD_STORE_RATIO_OPTIONS,
                                                roof.load_store_ratio,
                                                placeholder=ph_ls_ratio,
                                            ),
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        # Apps subsection
                        html.Div(
                            className="apps-section",
                            children=[
                                html.Div(
                                    className="apps-section-header",
                                    children=[
                                        html.Span("Applications", className="apps-section-title"),
                                    ],
                                ),
                                _multi_dropdown(
                                    _make_id(RoofCardID.DROPDOWN_APPS, index=index),
                                    app_options or [],
                                    roof.app_ids,
                                    clearable=True,
                                    placeholder="Search applications\u2026"
                                    if (app_options or [])
                                    else "No application data",
                                ),
                            ],
                        ),
                    ],
                ),
            ),
        ],
    )


def build_settings_panel(store: RoofStore, options: FilterOptions | None = None) -> html.Div:
    """Settings panel with various plotting and appearance options."""
    s = store.settings
    switch_children = [_build_settings_switch(label, id_, getattr(s, field)) for label, id_, field in _SWITCH_CONTROLS]
    slider_children = [
        _build_settings_slider(label, id_, getattr(s, field), **kwargs)
        for label, id_, field, kwargs in _SLIDER_CONTROLS
    ]

    return html.Div(
        className="settings-panel",
        style={"display": "block"} if store.active_panel == ActivePanel.SETTINGS else {"display": "none"},
        children=[
            html.H5("Settings", className="panel-header"),
            *switch_children,
            *slider_children,
        ],
    )


def build_plot_area() -> html.Div:
    """The main plotting area with a Graph component."""
    from carm_roofline.gui.data import build_roofline_figure

    return html.Div(
        className="plot-area",
        children=[
            dcc_graph(
                PlotAreaID.ROOFLINE_PLOT,
                figure=build_roofline_figure([], []),
                config={"responsive": True, "toImageButtonOptions": {"format": "svg", "filename": "roofline"}},
            ),
        ],
    )


def build_layout(
    store: RoofStore, roof_options: FilterOptions | None = None, app_options: list[DropdownOption] | None = None
) -> html.Div:
    """Top-level application layout."""

    return html.Div(
        className="app-container",
        children=[
            build_navbar(store.active_panel),
            html.Div(
                className="content-row",
                children=[
                    build_sidebar(store, roof_options, app_options),
                    html.Div(
                        className="main-area",
                        children=[
                            build_plot_area(),
                        ],
                    ),
                ],
            ),
            # Hidden stores
            dcc.Store(id=StoreID.ROOF_STORE, data=store.to_dict()),
            dcc.Store(id=StoreID.ACTIVE_PANEL, data=store.active_panel),
        ],
    )


def dcc_dropdown(
    id_: Any,
    options: list[str] | list[tuple[str, str]],
    value: str | None,
    placeholder: str | None = None,
) -> html.Div:
    dd_options: list[dict[str, str]] = []
    for o in options or []:
        if isinstance(o, tuple):
            dd_options.append({"label": o[0], "value": o[1]})
        else:
            dd_options.append({"label": o, "value": o})
    return html.Div(
        dcc.Dropdown(
            id=id_,
            options=dd_options,
            value=value,
            placeholder=placeholder,
            searchable=True,
            clearable=True,
        ),
    )


def _multi_dropdown(
    id_: Any,
    options: list[DropdownOption],
    value: list[str],
    clearable: bool = False,
    placeholder: str | None = None,
) -> html.Div:
    """A searchable multi-select dcc.Dropdown with label-value options."""

    return html.Div(
        dcc.Dropdown(
            id=id_,
            options=options,
            value=value,
            searchable=True,
            clearable=clearable,
            multi=True,
            placeholder=placeholder or "",
        ),
    )


def dcc_input(
    id_: Any,
    type_: str,
    value: int | float,
    min_val: int | float | None = None,
    max_val: int | float | None = None,
    step_val: int | float | None = None,
) -> html.Div:
    """A dcc.Input wrapped in a div."""

    kwargs: dict[str, Any] = {
        "id": id_,
        "type": type_,
        "value": value,
    }
    if min_val is not None:
        kwargs["min"] = min_val
    if max_val is not None:
        kwargs["max"] = max_val
    if step_val is not None:
        kwargs["step"] = step_val

    return html.Div(dcc.Input(**kwargs))


def dcc_graph(id_: str, figure: Any, config: Any = None) -> html.Div:
    """A dcc.Graph wrapped in a div."""

    return html.Div(
        dcc.Graph(id=id_, figure=figure, style={"height": "100%"}, config=config),
        style={"flex": "1", "height": "100%"},
    )


def _field_row(label: str, component: html.Div) -> html.Div:
    """A label + component row for roof card forms."""
    return html.Div(
        className="field-row",
        children=[
            html.Div(label, className="field-label"),
            component,
        ],
    )


def _field_row_pair(left: html.Div, right: html.Div) -> html.Div:
    """Two field rows placed side by side."""
    return html.Div(
        className="field-row-pair",
        children=[left, right],
    )
