from __future__ import annotations

import math
from typing import Any, Callable

import dash_bootstrap_components as dbc
from dash import dcc, html

from carm_roofline.core.units import Frequency, Seconds
from carm_roofline.gui.colors import COLOR_MODE_OPTIONS
from carm_roofline.gui.config import GUIMode
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
    BenchmarkPanelID,
    CarmViewPanelID,
    ExportPanelID,
    FolderImportID,
    NavbarID,
    ParaverID,
    PlotAreaID,
    ProfilePanelID,
    RoofCardID,
    SettingsPanelID,
    SidebarID,
    StoreID,
    WelcomePanelID,
)
from carm_roofline.gui.providers import (
    AI_FILTER_LOG_STEP,
    AI_FILTER_MAX_AI,
    AI_FILTER_OFF_AI,
    DURATION_FILTER_LOG_STEP,
    DURATION_FILTER_MAX_S,
    DURATION_FILTER_OFF_S,
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


def _slider_marks(*values: float | int) -> dict[float | int, str]:
    """Generate marks dict for dcc.Slider from explicit label values."""
    return {v: str(v) for v in values}


def _time_window_marks(lo: float, hi: float, max_ticks: int = 5) -> dict[float, str]:
    """Readable tick marks for the time-window slider.

    Dash 4 auto-generates marks at fixed 0/25/50/75/100% positions when none are
    given, labeling them with raw float values (e.g. ``0.06614603825000001``).
    Instead, cover the trace extent with a "nice" 1/2/5 x 10**n step and format
    each tick with just enough decimals for that step.
    """
    span = hi - lo
    if not math.isfinite(span) or span <= 0:
        return {lo: f"{lo:.3g}"}
    raw_step = span / (max_ticks - 1)
    exponent = math.floor(math.log10(raw_step))
    mantissa = raw_step / (10.0**exponent)
    # Round the raw step to a "nice" 1/2/5 x 10**n value; the 10 case is
    # renormalized to 1 x 10**(n+1) so the coefficient stays in {1, 2, 5}.
    if mantissa < 1.5:
        coefficient = 1.0
    elif mantissa < 3.5:
        coefficient = 2.0
    elif mantissa < 7.5:
        coefficient = 5.0
    else:
        coefficient = 1.0
        exponent += 1
    step = coefficient * (10.0**exponent)
    decimals = max(0, -exponent)
    first = max(math.ceil(lo / step) * step, lo)
    eps = step * 1e-9
    ticks: list[float] = []
    tick = first
    while tick <= hi + eps:
        ticks.append(round(tick, decimals))
        tick += step
    if not ticks:
        ticks = [lo]
    # Rounding can nudge the last tick a hair past hi; keep every mark key in range.
    return {min(t, hi): _format_tick_label(t, decimals) for t in ticks}


def _format_tick_label(value: float, decimals: int) -> str:
    """Format a slider tick value, trimming trailing zeros after the point."""
    label = f"{value:.{decimals}f}"
    if "." in label:
        label = label.rstrip("0").rstrip(".")
    return label


def _format_scientific(value: float) -> str:
    """Scientific notation with one significant digit and no zero-padded exponent (1e-5, not 1e-05)."""
    mantissa, exponent = f"{value:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def _format_duration_s(seconds: float) -> str:
    """Human duration label via core Seconds formatting (1e-4 -> "100 us", 1e-1 -> "100 ms")."""
    return str(Seconds(seconds))


def _log_threshold_marks(log_min: float, log_max: float, formatter: Callable[[float], str]) -> dict[float, str]:
    """Marks for a log-scale threshold filter slider: one label per decade from min
    through max ("Off" at min), with the max itself always labeled."""
    marks: dict[float, str] = {}
    decade = math.ceil(log_min)
    while decade <= math.floor(log_max) + 1e-9:
        marks[float(decade)] = formatter(10.0**decade)
        decade += 1
    marks[log_min] = "Off"
    if log_max not in marks:
        marks[log_max] = formatter(10.0**log_max)
    return marks


_SWITCH_CONTROLS: list[tuple[str, str, str]] = [
    ("Normalize performance by threads", SettingsPanelID.SWITCH_NORMALIZE, "normalize_by_threads"),
    ("2^N axis tick labels", SettingsPanelID.SWITCH_POWER2_TICKS, "power2_ticks"),
]

_SLIDER_CONTROLS: list[tuple[str, str, str, dict[str, Any]]] = [
    (
        "Base point size",
        SettingsPanelID.SLIDER_MARKER_BASE_SIZE,
        "marker_base_size",
        {
            "min": 1,
            "max": 200,
            "step": 1,
            "marks": _slider_marks(1, 50, 100, 150, 200),
        },
    ),
    (
        "Duration size multiplier",
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
    (
        "Legend label length",
        SettingsPanelID.SLIDER_LEGEND_LABEL_LENGTH,
        "legend_label_length",
        {
            "min": 8,
            "max": 64,
            "step": 1,
            "marks": _slider_marks(8, 16, 32, 48, 64),
        },
    ),
]

# Paraver AI-filter slider. Positions are log10(ai threshold in FLOPS/Byte),
# derived from the canonical filter constants in providers.py so the slider
# geometry can never drift from the filter semantics.
AI_FILTER_LOG_MIN = math.log10(AI_FILTER_OFF_AI)  # leftmost position: filtering off
AI_FILTER_LOG_MAX = math.log10(AI_FILTER_MAX_AI)  # rightmost position
AI_FILTER_MARKS = _log_threshold_marks(AI_FILTER_LOG_MIN, AI_FILTER_LOG_MAX, _format_scientific)

# Paraver duration-filter slider, same geometry as the AI filter: positions are
# log10(minimum duration in seconds); leftmost position is filtering off.
DURATION_FILTER_LOG_MIN = math.log10(DURATION_FILTER_OFF_S)  # leftmost position: filtering off
DURATION_FILTER_LOG_MAX = math.log10(DURATION_FILTER_MAX_S)  # rightmost position
DURATION_FILTER_MARKS = _log_threshold_marks(DURATION_FILTER_LOG_MIN, DURATION_FILTER_LOG_MAX, _format_duration_s)


def _build_settings_switch(label: str, id_: str, value: bool) -> html.Div:
    return html.Div(
        className="settings-toggle-row",
        children=[
            html.Span(label, className="settings-toggle-label"),
            dbc.Switch(id=id_, value=value, className="normalize-toggle"),
        ],
    )


def _build_settings_slider(label: str, id_: str, value: float, **kwargs: Any) -> html.Div:
    tooltip = kwargs.pop("tooltip", {"placement": "bottom", "always_visible": False})
    return html.Div(
        className="settings-slider-row",
        children=[
            html.Span(label, className="settings-toggle-label"),
            dcc.Slider(
                id=id_,
                value=value,
                **kwargs,
                tooltip=tooltip,
            ),
        ],
    )


def _build_threshold_filter_slider(
    label: str,
    slider_id: str,
    log_value: float,
    log_min: float,
    log_max: float,
    log_step: float,
    marks: dict[float, str],
    tooltip_transform: str,
) -> html.Div:
    """Log-scale threshold filter slider (AI, duration): min position = filtering off."""
    return _build_settings_slider(
        label,
        slider_id,
        log_value,
        min=log_min,
        max=log_max,
        step=log_step,
        marks=marks,
        tooltip={"placement": "bottom", "transform": tooltip_transform},
    )


def build_navbar(active_panel: ActivePanel, mode: GUIMode = GUIMode.CARM) -> html.Div:
    """Black top navbar with logo and panel-toggle buttons."""
    carm_view_active = active_panel == ActivePanel.CARM_VIEW
    benchmark_active = active_panel == ActivePanel.BENCHMARK
    profile_active = active_panel == ActivePanel.PROFILE
    settings_active = active_panel == ActivePanel.SETTINGS
    export_active = active_panel == ActivePanel.EXPORT

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
                    html.Span("CARM Tool", className="navbar-title"),
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
                    dbc.Tooltip(
                        "Explore the roofline and compare the performance of applications and architectures.",
                        id=NavbarID.TOOLTIP_CARM_VIEW,
                        target=NavbarID.BTN_CARM_VIEW,
                        trigger="hover focus",
                    ),
                    *(
                        [
                            html.Button(
                                "Paraver",
                                id=NavbarID.BTN_EXPORT,
                                className=f"navbar-btn{' navbar-btn--active' if export_active else ''}",
                                n_clicks=0,
                            ),
                        ]
                        if mode.has_export_tab
                        else []
                    ),
                    html.Button(
                        "Benchmark",
                        id=NavbarID.BTN_BENCHMARK,
                        className=f"navbar-btn{' navbar-btn--active' if benchmark_active else ''}",
                        n_clicks=0,
                    ),
                    dbc.Tooltip(
                        "Configure and run a new benchmark (on the same machine the GUI is running).",
                        id=NavbarID.TOOLTIP_BENCHMARK,
                        target=NavbarID.BTN_BENCHMARK,
                        trigger="hover focus",
                    ),
                    html.Button(
                        "Profile",
                        id=NavbarID.BTN_PROFILE,
                        className=f"navbar-btn{' navbar-btn--active' if profile_active else ''}",
                        n_clicks=0,
                    ),
                    dbc.Tooltip(
                        "Configure and run PAPI or perf profiling for a local application.",
                        id=NavbarID.TOOLTIP_PROFILE,
                        target=NavbarID.BTN_PROFILE,
                        trigger="hover focus",
                    ),
                    html.Button(
                        "Settings",
                        id=NavbarID.BTN_SETTINGS,
                        className=f"navbar-btn{' navbar-btn--active' if settings_active else ''}",
                        n_clicks=0,
                    ),
                    dbc.Tooltip(
                        "Adjust plot appearance and how the performance is displayed.",
                        id=NavbarID.TOOLTIP_SETTINGS,
                        target=NavbarID.BTN_SETTINGS,
                        trigger="hover focus",
                    ),
                ],
            ),
        ],
    )


def build_sidebar(
    store: RoofStore,
    roof_options: FilterOptions | None = None,
    per_roof_app_options: list[list[DropdownOption]] | None = None,
    mode: GUIMode = GUIMode.CARM,
) -> html.Div:
    """Left sidebar containing settings or data-selection panel."""
    children = [
        build_carm_view_panel(
            store,
            [roof_options] * len(store.roofs) if roof_options else None,
            per_roof_app_options=per_roof_app_options,
            include_apps_section=mode.show_app_dropdown,
        ),
        build_settings_panel(store, roof_options),
        *_build_guide_panels(store.active_panel),
    ]
    if mode.has_export_tab:
        children.append(build_export_panel(store))
    return html.Div(
        className="sidebar",
        children=[
            # Hidden store for active panel
            html.Div(
                id=SidebarID.SIDEBAR_CONTENT,
                children=children,
            ),
        ],
    )


def _build_guide_panels(active_panel: ActivePanel) -> list[dbc.Card]:
    """Build sidebar guides, showing the one for the active task panel."""
    from carm_roofline.gui.guides import BENCHMARK_GUIDE, build_guide_panel
    from carm_roofline.gui.profile_guide import PROFILE_GUIDE

    return [
        build_guide_panel("Benchmark guide", BENCHMARK_GUIDE, visible=active_panel == ActivePanel.BENCHMARK),
        build_guide_panel("Profile guide", PROFILE_GUIDE, visible=active_panel == ActivePanel.PROFILE),
    ]


def build_carm_view_panel(
    store: RoofStore,
    per_roof_options: list[FilterOptions | None] | None = None,
    resolved_roofs: list[RoofConfig] | None = None,
    per_roof_app_options: list[list[DropdownOption]] | None = None,
    include_apps_section: bool = True,
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
                    per_roof_options[i] if per_roof_options and i < len(per_roof_options) else None,
                    resolved_roofs[i] if resolved_roofs and i < len(resolved_roofs) else None,
                    per_roof_app_options[i] if per_roof_app_options and i < len(per_roof_app_options) else None,
                    include_apps_section=include_apps_section,
                )
                for i, roof in enumerate(store.roofs)
            ],
            html.Button(
                "+ Add Roof",
                id=CarmViewPanelID.BTN_ADD_ROOF,
                className="btn-add-roof",
                n_clicks=0,
            ),
            dbc.Tooltip(
                "Add another roof to visualize and compare.",
                id=CarmViewPanelID.TOOLTIP_ADD_ROOF,
                target=CarmViewPanelID.BTN_ADD_ROOF,
            ),
        ],
    )


def _build_apps_section(
    roof_id: str,
    app_ids: list[str],
    app_options: list[DropdownOption] | None,
    include: bool,
) -> list[html.Div]:
    """The applications multi-select subsection of a roof card.

    Returns an empty list when *include* is False so callers can splat the
    result into the roof-card body children.
    """
    if not include:
        return []
    return [
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
                    {"type": RoofCardID.DROPDOWN_APPS, "roof_id": roof_id},
                    app_options or [],
                    app_ids,
                    clearable=True,
                    placeholder="Search applications\u2026" if (app_options or []) else "No application data",
                ),
            ],
        ),
    ]


def build_roof_card(
    roof: RoofConfig,
    options: FilterOptions | None = None,
    resolved_roof: RoofConfig | None = None,
    app_options: list[DropdownOption] | None = None,
    include_apps_section: bool = True,
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
                        id={"type": RoofCardID.BTN_COLLAPSE_ROOF, "roof_id": roof.id},
                        className="btn-collapse-roof",
                        n_clicks=0,
                    ),
                    html.Span(roof.label, className="roof-card-title"),
                    html.Button(
                        "\u00d7",
                        id={"type": RoofCardID.BTN_REMOVE_ROOF, "roof_id": roof.id},
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
                                {"type": RoofCardID.DROPDOWN_MACHINE, "roof_id": roof.id},
                                options["machine"] if options else MACHINE_OPTIONS,
                                roof.machine,
                                placeholder=ph_machine,
                            ),
                        ),
                        _field_row_pair(
                            _field_row(
                                "ISA",
                                dcc_dropdown(
                                    {"type": RoofCardID.DROPDOWN_ISA, "roof_id": roof.id},
                                    options["isa"] if options else ISA_OPTIONS,
                                    roof.isa,
                                    placeholder=ph_isa,
                                ),
                            ),
                            _field_row(
                                "Threads",
                                dcc_dropdown(
                                    {"type": RoofCardID.DROPDOWN_THREADS, "roof_id": roof.id},
                                    [str(t) for t in options["num_threads"]] if options else THREADS_OPTIONS,
                                    str(roof.num_threads) if roof.num_threads is not None else None,
                                    placeholder=ph_threads,
                                ),
                            ),
                        ),
                        _field_row(
                            "Data Type",
                            dcc_dropdown(
                                {"type": RoofCardID.DROPDOWN_DATA_TYPE, "roof_id": roof.id},
                                options["data_type"] if options else DATA_TYPE_OPTIONS,
                                roof.data_type,
                                placeholder=ph_data_type,
                            ),
                        ),
                        _field_row(
                            "Compute Inst",
                            _multi_dropdown(
                                {"type": RoofCardID.DROPDOWN_COMPUTE, "roof_id": roof.id},
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
                                            id={"type": RoofCardID.BTN_ADVANCED_COLLAPSE, "roof_id": roof.id},
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
                                                {"type": RoofCardID.DROPDOWN_FREQUENCY, "roof_id": roof.id},
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
                                                {"type": RoofCardID.DROPDOWN_LS_RATIO, "roof_id": roof.id},
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
                        *_build_apps_section(roof.id, roof.app_ids, app_options, include_apps_section),
                    ],
                ),
            ),
        ],
    )


def build_settings_panel(store: RoofStore, options: FilterOptions | None = None) -> html.Div:
    """Settings panel with various plotting and appearance options, grouped into accordion sections."""
    s = store.settings

    # Plotting settings — switches that affect the displayed roofline data
    plotting_switches = [
        _build_settings_switch(
            "Normalize performance by threads",
            SettingsPanelID.SWITCH_NORMALIZE,
            s.normalize_by_threads,
        ),
        _build_settings_switch(
            "Show mixed benchmarks",
            SettingsPanelID.SWITCH_SHOW_MIXED_BENCHMARKS,
            s.show_mixed_benchmarks,
        ),
    ]

    # Plot style — everything else
    plot_style_controls = [
        _build_settings_switch(
            "2^N axis tick labels",
            SettingsPanelID.SWITCH_POWER2_TICKS,
            s.power2_ticks,
        ),
        _build_settings_switch(
            "Show roof labels",
            SettingsPanelID.SWITCH_SHOW_ROOF_LABELS,
            s.show_roof_labels,
        ),
        _field_row(
            "Roof fill",
            dcc_dropdown(
                SettingsPanelID.DROPDOWN_ROOF_FILL_MODE,
                [("Between-roof fills", "between"), ("No fill", "none"), ("Bound regions", "bound")],
                s.roof_fill_mode,
            ),
        ),
    ]
    plot_style_sliders = [
        _build_settings_slider(label, id_, getattr(s, field), **kwargs)
        for label, id_, field, kwargs in _SLIDER_CONTROLS
    ]

    return html.Div(
        className="settings-panel",
        style={"display": "block"} if store.active_panel == ActivePanel.SETTINGS else {"display": "none"},
        children=[
            html.H5("Settings", className="panel-header"),
            dbc.Accordion(
                children=[
                    dbc.AccordionItem(
                        title="Plotting settings",
                        item_id="plotting-settings",
                        children=plotting_switches,
                    ),
                    dbc.AccordionItem(
                        title="Plot style",
                        item_id="plot-style",
                        children=[*plot_style_controls, *plot_style_sliders],
                    ),
                ],
                always_open=True,
                active_item=["plotting-settings", "plot-style"],
                start_collapsed=False,
            ),
        ],
    )


def _export_row(label: str, btn_id: str, status_id: str) -> html.Div:
    """Button + per-row status Div (no Download — exports are written to disk)."""
    return html.Div(
        className="export-row",
        children=[
            html.Button(label, id=btn_id, className="btn-export-mode", n_clicks=0),
            html.Div(id=status_id, className="export-panel-status"),
        ],
    )


def _build_export_section() -> dbc.AccordionItem:
    return dbc.AccordionItem(
        title="Export",
        item_id="export",
        children=[
            _export_row(
                "Performance (GFLOPS/s)",
                ExportPanelID.BTN_EXPORT_PERFORMANCE,
                ExportPanelID.STATUS_PERFORMANCE,
            ),
            _export_row("Arithmetic intensity", ExportPanelID.BTN_EXPORT_AI, ExportPanelID.STATUS_AI),
            _export_row(
                "Load percentage",
                ExportPanelID.BTN_EXPORT_LDST_PERCENT,
                ExportPanelID.STATUS_LDST_PERCENT,
            ),
            _export_row("Roof labels", ExportPanelID.BTN_EXPORT_ROOF_LABELS, ExportPanelID.STATUS_ROOF_LABELS),
            _export_row("Roofline region", ExportPanelID.BTN_EXPORT_REGION, ExportPanelID.STATUS_REGION),
            _export_row("Roof proximity", ExportPanelID.BTN_EXPORT_PROXIMITY, ExportPanelID.STATUS_PROXIMITY),
        ],
    )


def _build_semantic_window_switch(store: RoofStore) -> html.Div:
    return html.Div(
        className="settings-toggle-row",
        children=[
            html.Span("Use semantic window", className="settings-toggle-label"),
            dbc.Switch(
                id=ExportPanelID.SWITCH_SEMANTIC_WINDOW,
                value=store.paraver_state.use_semantic_window,
                className="normalize-toggle",
            ),
            dbc.Tooltip(
                "When enabled, shows only points visible in Parever (within the semantic"
                " window). When disabled, shows all points. Also affects exports.",
                id=ExportPanelID.TOOLTIP_SEMANTIC_WINDOW,
                target=ExportPanelID.SWITCH_SEMANTIC_WINDOW,
                trigger="hover focus",
            ),
        ],
    )


def _build_export_filter_section(store: RoofStore) -> dbc.AccordionItem:
    ai_log = (
        round(math.log10(store.paraver_state.ai_threshold), 2)
        if store.paraver_state.ai_threshold is not None
        else AI_FILTER_LOG_MIN
    )
    duration_log = (
        round(math.log10(store.paraver_state.duration_threshold), 2)
        if store.paraver_state.duration_threshold is not None
        else DURATION_FILTER_LOG_MIN
    )
    return dbc.AccordionItem(
        title="Filtering",
        item_id="filtering",
        children=[
            _build_semantic_window_switch(store),
            _build_threshold_filter_slider(
                "Minimum arithmetic intensity",
                ExportPanelID.SLIDER_AI_THRESHOLD,
                ai_log,
                AI_FILTER_LOG_MIN,
                AI_FILTER_LOG_MAX,
                AI_FILTER_LOG_STEP,
                AI_FILTER_MARKS,
                "aiThreshold",
            ),
            _build_threshold_filter_slider(
                "Minimum duration",
                ExportPanelID.SLIDER_DURATION_THRESHOLD,
                duration_log,
                DURATION_FILTER_LOG_MIN,
                DURATION_FILTER_LOG_MAX,
                DURATION_FILTER_LOG_STEP,
                DURATION_FILTER_MARKS,
                "paraverDuration",
            ),
        ],
    )


def _build_export_color_section(store: RoofStore) -> dbc.AccordionItem:
    return dbc.AccordionItem(
        title="Color",
        item_id="color",
        children=[
            html.Div(
                className="settings-slider-row",
                children=[
                    html.Span("Color points based on", className="settings-toggle-label"),
                    dbc.RadioItems(
                        id=ExportPanelID.RADIO_COLOR_MODE,
                        options=[{"label": label, "value": value} for label, value in COLOR_MODE_OPTIONS],
                        value=store.paraver_state.color_mode,
                    ),
                ],
            ),
        ],
    )


def build_export_panel(store: RoofStore) -> html.Div:
    """Paraver panel: export, filtering, and color controls grouped into accordion sections."""
    return html.Div(
        className="export-panel",
        style={"display": "block"} if store.active_panel == ActivePanel.EXPORT else {"display": "none"},
        children=[
            html.H5("Paraver", className="panel-header"),
            dbc.Accordion(
                children=[
                    _build_export_section(),
                    _build_export_filter_section(store),
                    _build_export_color_section(store),
                ],
                always_open=True,
                active_item=["export", "filtering", "color"],
                start_collapsed=False,
            ),
        ],
    )


def _benchmark_field(label: str, control: Any, help_text: str) -> html.Div:
    target_name = label.casefold().replace(" ", "-").replace("(", "").replace(")", "")
    target_id = f"benchmark-help-{target_name}"
    return html.Div(
        className="benchmark-field",
        children=[
            html.Label(label, id=target_id, className="benchmark-label"),
            control,
            dbc.Tooltip(
                help_text,
                target=target_id,
                placement="top",
                class_name="benchmark-help-tooltip",
            ),
        ],
    )


_PROFILE_FIELD_HELP_IDS: dict[ProfilePanelID, ProfilePanelID] = {
    ProfilePanelID.BACKEND: ProfilePanelID.TOOLTIP_BACKEND,
    ProfilePanelID.COMMAND: ProfilePanelID.TOOLTIP_COMMAND,
    ProfilePanelID.AGGREGATION: ProfilePanelID.TOOLTIP_AGGREGATION,
    ProfilePanelID.MACHINE_NAME: ProfilePanelID.TOOLTIP_MACHINE_NAME,
    ProfilePanelID.APP_NAME: ProfilePanelID.TOOLTIP_APP_NAME,
    ProfilePanelID.OUTPUT_DIR: ProfilePanelID.TOOLTIP_OUTPUT_DIR,
    ProfilePanelID.METRICS: ProfilePanelID.TOOLTIP_METRICS,
    ProfilePanelID.ISA: ProfilePanelID.TOOLTIP_ISA,
    ProfilePanelID.DATA_TYPE: ProfilePanelID.TOOLTIP_DATA_TYPE,
    ProfilePanelID.MERGE_RUNS: ProfilePanelID.TOOLTIP_MERGE_RUNS,
    ProfilePanelID.NO_PAPI_CACHE: ProfilePanelID.TOOLTIP_NO_PAPI_CACHE,
    ProfilePanelID.PERF_INTERVAL: ProfilePanelID.TOOLTIP_PERF_INTERVAL,
    ProfilePanelID.KEEP_ARTIFACTS: ProfilePanelID.TOOLTIP_KEEP_ARTIFACTS,
}


def _profile_field(label: str, control: Any, help_text: str) -> html.Div:
    help_id = _PROFILE_FIELD_HELP_IDS[ProfilePanelID(control.id)]
    return html.Div(
        className="benchmark-field",
        children=[
            html.Label(label, id=help_id, className="benchmark-label"),
            control,
            dbc.Tooltip(
                help_text,
                target=help_id,
                placement="top",
                class_name="benchmark-help-tooltip",
            ),
        ],
    )


def build_benchmark_panel(results_dir: str, visible: bool = True) -> html.Div:
    """Build the CLI benchmark configurator and its live run status."""
    from carm_roofline.core import ArithmeticOperation, DataType
    from carm_roofline.gui.ids import BenchmarkPanelID
    from carm_roofline.isa import BaseISA

    test_options = [
        {"label": "Roofline", "value": "roofline"},
        {"label": "Arithmetic", "value": "arithmetic"},
        {"label": "Memory", "value": "memory"},
        {"label": "Mixed", "value": "mixed"},
        {"label": "Memory sweep", "value": "memory_sweep"},
    ]
    basic_fields = [
        _benchmark_field(
            "Test",
            dcc.Dropdown(id=BenchmarkPanelID.TEST, options=test_options, value="roofline", clearable=False),
            "Roofline combines arithmetic and memory measurements. Other test types measure one workload.",
        ),
        _benchmark_field(
            "ISA",
            dcc.Dropdown(
                id=BenchmarkPanelID.ISA,
                options=[{"label": "Automatic detection", "value": "auto"}]
                + [{"label": isa, "value": isa} for isa in BaseISA.names()],
                value=["auto"],
                multi=True,
            ),
            (
                "Automatic detection selects compatible instruction sets on this machine. "
                "Manual choices can fail if the CPU cannot run them."
            ),
        ),
        _benchmark_field(
            "Data types",
            dcc.Dropdown(
                id=BenchmarkPanelID.DATA_TYPES,
                options=[{"label": dtype.name, "value": dtype.name} for dtype in DataType],
                value=["f32"],
                multi=True,
            ),
            "Each selected type adds measurements. Supported instructions depend on the ISA and data type.",
        ),
        _benchmark_field(
            "Threads",
            dcc.Input(id=BenchmarkPanelID.THREADS, type="text", value="1", placeholder="1 2 4"),
            "Enter one or more thread counts, separated by spaces or commas. Each count adds measurements.",
        ),
        _benchmark_field(
            "Run time (s)",
            dcc.Input(id=BenchmarkPanelID.TEST_TIME, type="number", value=25.0, min=0.1, step=0.1),
            (
                "Target runtime for each individual microbenchmark. The total run time grows with the number "
                "of measurements. Values below 10 seconds can reduce accuracy."
            ),
        ),
        _benchmark_field(
            "Run name",
            dcc.Input(id=BenchmarkPanelID.NAME, type="text", placeholder="Auto-generated"),
            (
                "Optional name for this result set. The tool creates a name from the detected processor when "
                "this field is empty."
            ),
        ),
    ]
    advanced_fields = [
        _benchmark_field(
            "Memory levels",
            dcc.Dropdown(
                id=BenchmarkPanelID.MEMORY_TARGET,
                options=[{"label": level, "value": level} for level in ("L1", "L2", "L3", "DRAM", "all")],
                value=["all"],
                multi=True,
            ),
            "Memory levels used for memory measurements. The default measures all levels.",
        ),
        _benchmark_field(
            "Arithmetic operations",
            dcc.Input(id=BenchmarkPanelID.NUM_OPS, type="number", value=32768, min=1, step=1),
            "Number of arithmetic operations in each arithmetic benchmark.",
        ),
        _benchmark_field(
            "Instructions",
            dcc.Dropdown(
                id=BenchmarkPanelID.INSTRUCTIONS,
                options=[{"label": op.name, "value": op.name} for op in ArithmeticOperation],
                value=["add", "fma"],
                multi=True,
            ),
            "Arithmetic instructions to measure. Instruction support depends on the selected ISA and data type.",
        ),
        _benchmark_field(
            "Load:store ratios",
            dcc.Input(id=BenchmarkPanelID.LD_ST_RATIO, type="text", value="2:1"),
            "One or more load:store ratios separated by spaces or commas. Use 1:0 for load-only or 0:1 for store-only.",
        ),
        _benchmark_field(
            "Arithmetic:memory ratio",
            dcc.Input(id=BenchmarkPanelID.ARITH_MEM_RATIO, type="text", value="2:3"),
            "Ratio of arithmetic instructions to memory instructions in each mixed-test block.",
        ),
        _benchmark_field(
            "Intensity range",
            html.Div(
                className="benchmark-field-pair",
                children=[
                    dcc.Input(id=BenchmarkPanelID.AI_MIN, type="number", value=2**-4, min=0, step="any"),
                    dcc.Input(id=BenchmarkPanelID.AI_MAX, type="number", value=2**5, min=0, step="any"),
                ],
            ),
            "Minimum and maximum arithmetic intensity in FLOP per byte for mixed tests.",
        ),
        _benchmark_field(
            "Intensity points",
            dcc.Input(id=BenchmarkPanelID.AI_POINTS, type="number", value=10, min=1, step=1),
            "Number of logarithmically spaced arithmetic-intensity values for mixed tests.",
        ),
        _benchmark_field(
            "Memory sizes",
            dcc.Input(id=BenchmarkPanelID.MEMORY_SIZES, type="text", placeholder="auto auto auto auto"),
            (
                "Array size for L1, L2, L3, and DRAM in that order. Use binary units such as 16KiB or use auto. "
                "Empty means automatic sizing for all levels."
            ),
        ),
        _benchmark_field(
            "Topology file",
            dcc.Input(id=BenchmarkPanelID.TOPOLOGY, type="text", placeholder="Auto-detect"),
            "Optional TOML file that defines the cache hierarchy. By default the tool detects it from system files.",
        ),
        _benchmark_field(
            "Vector length (bytes)",
            dcc.Input(id=BenchmarkPanelID.VECTOR_LENGTH, type="number", min=1, step=1),
            "Override detected vector register length. Use a power-of-two value supported by the selected ISA.",
        ),
        _benchmark_field(
            "RISC-V vector grouping",
            dcc.Dropdown(
                id=BenchmarkPanelID.VECTOR_LMUL,
                options=[{"label": str(value), "value": value} for value in (1, 2, 4, 8)],
                placeholder="Auto",
                clearable=True,
            ),
            "RISC-V vector length multiplier. This setting applies only to RISC-V vector ISAs.",
        ),
        _benchmark_field(
            "Frequency",
            dcc.Input(id=BenchmarkPanelID.FREQUENCY, type="text", placeholder="Auto-detect"),
            "Optional processor frequency such as 2GHz. The tool detects frequency when possible.",
        ),
        _benchmark_field(
            "Compiler",
            dcc.Input(id=BenchmarkPanelID.COMPILER, type="text", value="gcc"),
            "C compiler executable used to build the benchmark. Cross-compilers are supported.",
        ),
        _benchmark_field(
            "Simulator",
            dcc.Input(id=BenchmarkPanelID.SIMULATOR, type="text", placeholder="Native execution"),
            "Optional simulator command. Include {binary} where the generated benchmark executable must appear.",
        ),
        _benchmark_field(
            "Output directory",
            dcc.Input(id=BenchmarkPanelID.OUTPUT_DIR, type="text", value=results_dir),
            "Directory on the machine that runs the GUI server. Result files are written here.",
        ),
        _benchmark_field(
            "Output formats",
            dcc.Dropdown(
                id=BenchmarkPanelID.OUTPUT_FORMATS,
                options=[{"label": value.upper(), "value": value} for value in ("table", "plot", "jsonl", "csv")],
                value=["table", "plot", "jsonl", "csv"],
                multi=True,
            ),
            "Select result formats to write. JSONL files feed the GUI roofline view.",
        ),
        _benchmark_field(
            "Interleaved affinity",
            dcc.Checklist(
                id=BenchmarkPanelID.INTERLEAVED,
                options=[{"label": "Use interleaved NUMA thread mapping", "value": "on"}],
                value=[],
            ),
            "Use interleaved thread mapping on systems with interleaved NUMA core domains.",
        ),
        _benchmark_field(
            "Set processor frequency",
            dcc.Checklist(
                id=BenchmarkPanelID.SET_FREQUENCY,
                options=[{"label": "Change system frequency", "value": "on"}],
                value=[],
            ),
            "Change processor frequency before the run. This can require administrator permissions.",
        ),
        _benchmark_field(
            "Dry run",
            dcc.Checklist(
                id=BenchmarkPanelID.DRY_RUN,
                options=[{"label": "Generate code only", "value": "on"}],
                value=[],
            ),
            "Generate benchmark code without compiling or running measurements. No result data is produced.",
        ),
        _benchmark_field(
            "Keep artifacts",
            dcc.Checklist(
                id=BenchmarkPanelID.KEEP_ARTIFACTS,
                options=[{"label": "Keep generated source and binary", "value": "on"}],
                value=[],
            ),
            "Keep generated benchmark source and binary files after execution.",
        ),
    ]
    return html.Div(
        id=BenchmarkPanelID.PAGE,
        className="benchmark-page",
        style={"display": "block"} if visible else {"display": "none"},
        children=[
            html.H2("Benchmark"),
            html.P("Configure and run a local CARM benchmark. The GUI server runs the benchmark on its host."),
            html.Div(className="benchmark-grid", children=basic_fields),
            html.Details(
                className="benchmark-advanced",
                children=[
                    html.Summary("Advanced settings"),
                    html.Div(className="benchmark-grid", children=advanced_fields),
                ],
            ),
            html.Button("Start benchmark", id=BenchmarkPanelID.START, n_clicks=0, className="benchmark-start"),
            html.Pre(
                id=BenchmarkPanelID.COMMAND_PREVIEW,
                className="benchmark-command-preview",
                children="The CLI command appears here after you start a run.",
            ),
            html.Div(id=BenchmarkPanelID.STATUS, className="benchmark-status", children="Ready"),
            html.Div(id=BenchmarkPanelID.ELAPSED, className="benchmark-elapsed", children="Elapsed: 0 s"),
            html.Progress(id=BenchmarkPanelID.PROGRESS, value="0", max="1", className="benchmark-progress"),
            html.Pre(id=BenchmarkPanelID.LOG, className="benchmark-log"),
            dcc.Interval(id=BenchmarkPanelID.INTERVAL, interval=1000, n_intervals=0, disabled=True),
        ],
    )


def build_profile_panel(results_dir: str, visible: bool = False) -> html.Div:
    """Build the CLI profile configurator and its live run status."""
    from carm_roofline.gui.ids import ProfilePanelID
    from carm_roofline.isa import BaseISA

    return html.Div(
        id=ProfilePanelID.PAGE,
        className="benchmark-page",
        style={"display": "block"} if visible else {"display": "none"},
        children=[
            html.H2("Profile"),
            html.P("Configure and run PAPI or Linux perf profiling on the GUI server host."),
            _profile_field(
                "Profiler backend",
                dcc.Dropdown(
                    id=ProfilePanelID.BACKEND,
                    options=[{"label": "PAPI", "value": "papi"}, {"label": "perf", "value": "perf"}],
                    value="papi",
                    clearable=False,
                ),
                "Choose PAPI for annotated region data or Linux perf for system-wide application profiling.",
            ),
            _profile_field(
                "Application command",
                dcc.Textarea(
                    id=ProfilePanelID.COMMAND,
                    placeholder="./my_app --input 'input file'",
                    className="profile-command",
                ),
                (
                    "Enter the executable and arguments. Shell quoting is parsed into argv; commands are "
                    "never run through a shell."
                ),
            ),
            html.Div(
                className="benchmark-grid",
                children=[
                    _profile_field(
                        "Aggregation",
                        dcc.Dropdown(
                            id=ProfilePanelID.AGGREGATION,
                            options=[
                                {"label": value.replace("_", " ").title(), "value": value}
                                for value in ("global", "rank", "thread", "region_merged", "region_per_thread")
                            ],
                            value="global",
                            clearable=False,
                        ),
                        "Choose how ranks, threads, and regions are combined into roofline points.",
                    ),
                    _profile_field(
                        "ISA",
                        dcc.Dropdown(
                            id=ProfilePanelID.ISA,
                            options=[{"label": isa, "value": isa} for isa in BaseISA.names()],
                            multi=True,
                            value=[],
                            placeholder="Unspecified",
                        ),
                        "Optional ISAs used by the workload, used to select the appropriate counters.",
                    ),
                    _profile_field(
                        "Data type",
                        dcc.Dropdown(
                            id=ProfilePanelID.DATA_TYPE,
                            options=[
                                {"label": value, "value": value}
                                for value in ("i8", "i16", "i32", "i64", "f32", "f64", "bf16")
                            ],
                            clearable=True,
                            placeholder="Unspecified",
                        ),
                        "Optional data precision for better estimating operation and byte counts.",
                    ),
                ],
            ),
            html.Details(
                className="benchmark-advanced",
                children=[
                    html.Summary("Advanced settings"),
                    html.Div(
                        className="benchmark-grid",
                        children=[
                            _profile_field(
                                "Machine name",
                                dcc.Input(id=ProfilePanelID.MACHINE_NAME, type="text", placeholder="Auto-detected"),
                                "Optional machine name used to match benchmark roof data.",
                            ),
                            _profile_field(
                                "Application name",
                                dcc.Input(id=ProfilePanelID.APP_NAME, type="text", placeholder="Derived from command"),
                                "Optional label for the profiled application.",
                            ),
                            _profile_field(
                                "Output directory",
                                dcc.Input(
                                    id=ProfilePanelID.OUTPUT_DIR,
                                    type="text",
                                    value="",
                                    placeholder=results_dir,
                                ),
                                "Directory on the GUI server where profile results are written.",
                            ),
                            _profile_field(
                                "Optional metrics (experimental)",
                                dcc.Dropdown(
                                    id=ProfilePanelID.METRICS,
                                    options=[
                                        {"label": "Cache residency", "value": "cache-residency"},
                                        {"label": "Cache-line utilization", "value": "cache-line-utilization"},
                                    ],
                                    multi=True,
                                    value=[],
                                ),
                                (
                                    "Optional hardware-counter metrics. These metrics are experimental "
                                    "and may not be accurate."
                                ),
                            ),
                            _profile_field(
                                "Counter budget",
                                dcc.Checklist(
                                    id=ProfilePanelID.MERGE_RUNS,
                                    options=[{"label": "Run multiple times to collect all events", "value": "on"}],
                                    value=[],
                                ),
                                "Partition events that exceed the hardware counter budget and merge the runs.",
                            ),
                            html.Div(
                                id=ProfilePanelID.BACKEND_OPTIONS,
                                children=[
                                    html.Div(
                                        _profile_field(
                                            "PAPI event cache",
                                            dcc.Checklist(
                                                id=ProfilePanelID.NO_PAPI_CACHE,
                                                options=[{"label": "Disable PAPI event catalog cache", "value": "on"}],
                                                value=[],
                                            ),
                                            "Disable the cached PAPI event catalog for this run.",
                                        ),
                                        id=ProfilePanelID.PAPI_OPTIONS,
                                    ),
                                    html.Div(
                                        _profile_field(
                                            "perf interval (ms)",
                                            dcc.Input(id=ProfilePanelID.PERF_INTERVAL, type="number", min=1, step=1),
                                            "Sample counters at this interval; empty means full-run mode.",
                                        ),
                                        id=ProfilePanelID.PERF_OPTIONS,
                                        style={"display": "none"},
                                    ),
                                ],
                            ),
                            _profile_field(
                                "Keep artifacts",
                                dcc.Checklist(
                                    id=ProfilePanelID.KEEP_ARTIFACTS,
                                    options=[{"label": "Keep raw profiling files", "value": "on"}],
                                    value=[],
                                ),
                                "Preserve raw profiler output files for debugging.",
                            ),
                        ],
                    ),
                ],
            ),
            html.Button("Start profile", id=ProfilePanelID.START, n_clicks=0, className="benchmark-start"),
            html.Pre(
                id=ProfilePanelID.COMMAND_PREVIEW,
                className="benchmark-command-preview",
                children="The CLI command appears here after you start a run.",
            ),
            html.Div(id=ProfilePanelID.STATUS, className="benchmark-status", children="Ready"),
            html.Div(id=ProfilePanelID.ELAPSED, className="benchmark-elapsed", children="Elapsed: 0 s"),
            html.Progress(id=ProfilePanelID.PROGRESS, value="0", max="1", className="benchmark-progress"),
            html.Pre(id=ProfilePanelID.LOG, className="benchmark-log"),
            dcc.Interval(id=ProfilePanelID.INTERVAL, interval=1000, n_intervals=0, disabled=True),
        ],
    )


def build_plot_area(
    mode: GUIMode = GUIMode.CARM,
    trace_bounds: tuple[float, float] | None = None,
    visible: bool = True,
) -> html.Div:
    """The main plotting area with a Graph component."""
    from carm_roofline.gui.data import build_roofline_figure

    children: list[Any] = [
        dcc_graph(
            PlotAreaID.ROOFLINE_PLOT,
            figure=build_roofline_figure([], []),
            config={"responsive": True, "toImageButtonOptions": {"format": "svg", "filename": "roofline"}},
            clear_on_unhover=True,
        ),
    ]
    if not mode.show_time_slider:
        # Background-click channel for click-driven roof emphasis (see
        # assets/roofline_bg_click.js): dcc.Graph only reports point clicks.
        children.append(dcc.Input(id=PlotAreaID.BG_CLICK_CLEAR, type="text", value="0", style={"display": "none"}))
    if mode.show_time_slider:
        lo, hi = trace_bounds if trace_bounds else (0.0, 1.0)
        # Dash 4.1.0's step=None is bugged (collapses both handles) so use an explicit fine-grained step
        span = hi - lo
        step = span / 1000 if span > 0 else 1.0
        children.insert(
            0,
            html.Div(
                className="time-window-control",
                children=[
                    html.Span("Time window", className="time-window-label"),
                    dcc.RangeSlider(
                        id=ParaverID.SLIDER_TIME_WINDOW,
                        min=lo,
                        max=hi,
                        value=[lo, hi],
                        step=step,
                        marks=_time_window_marks(lo, hi),
                        allow_direct_input=False,
                        disabled=trace_bounds is None,
                        tooltip={"placement": "bottom", "transform": "paraverTime"},
                        className="time-window-slider",
                    ),
                ],
            ),
        )
    return html.Div(
        id=PlotAreaID.CONTAINER,
        className="plot-area",
        style={"display": "flex"} if visible else {"display": "none"},
        children=children,
    )


def build_welcome_panel(visible: bool) -> html.Div:
    """Build the non-blocking first-run help panel."""
    return html.Div(
        id=WelcomePanelID.CONTAINER,
        className="welcome-panel",
        style={"display": "block"} if visible else {"display": "none"},
        children=[
            html.H4("Welcome to the CARM Tool"),
            html.P("Hover over each UI element for help."),
            html.P('If you haven\'t benchmarked your system yet, start with the "Benchmark" panel.'),
            dbc.Checklist(
                id=WelcomePanelID.DO_NOT_SHOW_AGAIN,
                options=[{"label": "Do not show again", "value": "hide"}],
                value=[],
                className="welcome-panel-checkbox",
            ),
            html.Button(
                "Got it",
                id=WelcomePanelID.GOT_IT,
                className="btn btn-primary",
                n_clicks=0,
            ),
        ],
    )


def build_layout(
    store: RoofStore,
    roof_options: FilterOptions | None = None,
    per_roof_app_options: list[list[DropdownOption]] | None = None,
    mode: GUIMode = GUIMode.CARM,
    trace_bounds: tuple[float, float] | None = None,
    results_dir: str = "",
) -> html.Div:
    is_benchmark = store.active_panel == ActivePanel.BENCHMARK
    is_profile = store.active_panel == ActivePanel.PROFILE
    benchmark_page = build_benchmark_panel(results_dir, is_benchmark)
    profile_page = build_profile_panel(results_dir, is_profile)
    plot_area = build_plot_area(mode, trace_bounds, not (is_benchmark or is_profile))

    return html.Div(
        className="app-container",
        children=[
            build_navbar(store.active_panel, mode),
            build_welcome_panel(store.settings.show_welcome_panel),
            html.Div(
                id=FolderImportID.OVERLAY,
                className="folder-import-overlay",
                children=html.Div("Drop a CARM results folder to import", className="folder-import-message"),
            ),
            html.Div(
                id=FolderImportID.STATUS,
                className="folder-import-status",
                role="status",
            ),
            html.Div(
                className="content-row",
                children=[
                    build_sidebar(store, roof_options, per_roof_app_options, mode),
                    html.Div(
                        className="main-area",
                        children=[
                            plot_area,
                            benchmark_page,
                            profile_page,
                        ],
                    ),
                ],
            ),
            # Hidden stores
            dcc.Store(id=StoreID.ROOF_STORE, data=store.to_dict()),
            dcc.Store(id=StoreID.ACTIVE_PANEL, data=store.active_panel),
            dcc.Store(id=StoreID.SELECTED_POINT, data=None),
            dcc.Store(id=StoreID.FOLDER_IMPORT_PAYLOAD, data=None),
            dcc.Store(id=StoreID.FOLDER_IMPORT_COMPLETE, data=None),
            dcc.Store(id=BenchmarkPanelID.JOB, data=None),
            dcc.Store(id=ProfilePanelID.JOB, data=None),
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


def dcc_graph(id_: str, figure: Any, config: Any = None, clear_on_unhover: bool = False) -> html.Div:
    """A dcc.Graph wrapped in a div."""

    return html.Div(
        dcc.Graph(
            id=id_,
            figure=figure,
            style={"height": "100%"},
            config=config,
            clear_on_unhover=clear_on_unhover,
        ),
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
