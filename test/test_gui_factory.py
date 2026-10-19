from __future__ import annotations
from typing import Any

import pytest

from carm_roofline.core.units import Performance
from carm_roofline.gui.callbacks.helpers import selection_payload
from carm_roofline.gui.callbacks.roofline import _point_guide_traces


pytestmark = pytest.mark.unit


def _find_component(node: Any, comp_id: str) -> Any | None:
    if getattr(node, "id", None) == comp_id:
        return node
    children = getattr(node, "children", None)
    if isinstance(children, list):
        for child in children:
            found = _find_component(child, comp_id)
            if found is not None:
                return found
    return None


def test_paraver_time_slider_uses_explicit_step() -> None:
    """The time-window slider must use an explicit step, not None.

    Dash 4.1.0 corrupts RangeSlider's internal value (both handles jump to the
    min) when step=None is passed explicitly, so the slider must carry a
    positive numeric step; direct numeric inputs stay hidden.
    """
    from carm_roofline.gui.components import build_plot_area
    from carm_roofline.gui.config import GUIMode
    from carm_roofline.gui.ids import ParaverID

    div = build_plot_area(GUIMode.PARAVER, (0.0, 2.0))
    slider = _find_component(div, ParaverID.SLIDER_TIME_WINDOW)
    assert slider is not None
    assert isinstance(slider.step, float)
    assert slider.step > 0
    assert slider.step == pytest.approx(2.0 / 1000)
    assert slider.allow_direct_input is False
    assert slider.value == [0.0, 2.0]
    # Tooltip values are formatted via window.dccFunctions.paraverTime (assets/paraver_time_slider.js).
    assert slider.tooltip == {"placement": "bottom", "transform": "paraverTime"}


def test_paraver_time_slider_degenerate_range_keeps_positive_step() -> None:
    """A zero-length trace range still yields a positive step (no div-by-zero)."""
    from carm_roofline.gui.components import build_plot_area
    from carm_roofline.gui.config import GUIMode
    from carm_roofline.gui.ids import ParaverID

    div = build_plot_area(GUIMode.PARAVER, (1.5, 1.5))
    slider = _find_component(div, ParaverID.SLIDER_TIME_WINDOW)
    assert slider is not None
    assert slider.step == 1.0


def test_paraver_time_slider_marks_are_readable() -> None:
    """Slider marks use concise labels and remain within the trace extent."""
    from carm_roofline.gui.components import build_plot_area
    from carm_roofline.gui.config import GUIMode
    from carm_roofline.gui.ids import ParaverID

    slider = _find_component(build_plot_area(GUIMode.PARAVER, (0.0, 0.264584153)), ParaverID.SLIDER_TIME_WINDOW)
    assert slider is not None
    marks = slider.marks
    assert isinstance(marks, dict) and marks
    assert all(0.0 <= key <= 0.264584153 for key in marks)
    assert all(len(label) < 10 for label in marks.values())
    assert "0.066146038" not in marks.values()


def test_paraver_semantic_window_switch_defaults_off_with_explanation() -> None:
    """Paraver's filtering menu exposes the opt-in switch and a clear tooltip."""
    from carm_roofline.gui.components import build_export_panel
    from carm_roofline.gui.data import RoofStore
    from carm_roofline.gui.ids import ExportPanelID

    panel = build_export_panel(RoofStore())
    switch = _find_component(panel, ExportPanelID.SWITCH_SEMANTIC_WINDOW)
    tooltip = _find_component(panel, ExportPanelID.TOOLTIP_SEMANTIC_WINDOW)

    assert switch is not None and switch.value is False
    assert tooltip is not None
    assert tooltip.target == ExportPanelID.SWITCH_SEMANTIC_WINDOW


def test_selection_payload_parses_application_identity_and_clears_background() -> None:
    """Application clicks preserve identity without requiring residency data."""
    customdata = ["<b>tooltip</b>", {}, "r1", "app-1", 0]
    valid_click = {"points": [{"customdata": customdata}]}
    assert selection_payload(valid_click, "roofline-plot.clickData", None) == {
        "roof_id": "r1",
        "fractions": {},
        "app_id": "app-1",
        "point_index": 0,
    }
    assert selection_payload(valid_click, "roofline-bg-click.value", {"roof_id": "r1"}) is None
    assert selection_payload(None, "roofline-bg-click.value", None) is None
    assert selection_payload(None, "roofline-plot.clickData", {"roof_id": "r1"}) == {"roof_id": "r1"}
    assert selection_payload({"points": [{"customdata": ["t", {}, "r1", "", 0]}]}, "plot.clickData", None) is None
    assert selection_payload({"points": [{"customdata": ["t", {}, "r1", "app-1", -1]}]}, "plot.clickData", None) is None
    assert selection_payload({"points": [{"customdata": ["t", {}, "r1"]}]}, "plot.clickData", None) is None


def test_point_guide_traces_show_unique_visible_roof_intersections_and_labels() -> None:
    figure = {
        "data": [
            {
                "meta": {"carm_role": "application", "roof_id": "r1", "app_id": "app-1"},
                "x": [2.0, 4.0],
                "y": [1.5, 2.5],
            },
            {
                "meta": {"carm_role": "roof-ceiling", "carm_roof_label": "DRAM"},
                "legendgroup": "r1",
                "name": "visible roof",
                "x": [1.0, 5.0],
                "y": [10.0, 10.0],
            },
            {
                "meta": {"carm_role": "roof-ceiling", "carm_roof_label": "L1"},
                "legendgroup": "r1",
                "name": "coincident roof",
                "x": [1.0, 5.0],
                "y": [10.0, 10.0],
            },
            {
                "meta": {"carm_role": "roof-ceiling", "carm_roof_label": "L1"},
                "legendgroup": "r1",
                "name": "at and below application points",
                "x": [1.0, 5.0],
                "y": [1.5, 1.5],
            },
            {
                "meta": {"carm_role": "roof-ceiling", "carm_roof_label": "add"},
                "legendgroup": "r1",
                "name": "sloped ceiling",
                "x": [1.0, 5.0],
                "y": [3.0, 15.0],
            },
            {
                "meta": {"carm_role": "roof-ceiling"},
                "legendgroup": "r2",
                "name": "hidden roof",
                "visible": "legendonly",
                "x": [1.0, 5.0],
                "y": [100.0, 100.0],
            },
        ],
        "layout": {
            "meta": {"carm_selected_point": {"roof_id": "r1", "app_id": "app-1", "point_index": 1}},
            "yaxis": {"range": [0.0, 3.0]},
        },
    }
    hover = {"points": [{"customdata": ["tip", {}, "r1", "app-1", 0]}]}
    traces = _point_guide_traces(figure, hover)
    guide_traces = {
        (trace.meta["guide"], trace.meta["kind"]): trace
        for trace in traces
        if trace.meta["carm_role"] == "point-guide"
    }
    hover_line = guide_traces[("hover", "line")]
    selected_line = guide_traces[("selected", "line")]
    hover_markers = guide_traces[("hover", "intersections")]
    selected_markers = guide_traces[("selected", "intersections")]

    assert len(guide_traces) == 4
    assert list(hover_line.x) == [2.0, 2.0]
    assert list(hover_line.y) == [1.0, 10.0]
    assert list(selected_line.x) == [4.0, 4.0]
    assert list(selected_line.y) == pytest.approx([1.0, 12.0])
    assert hover_line.mode == selected_line.mode == "lines"
    assert hover_line.line.width == selected_line.line.width == 3
    assert hover_line.opacity == selected_line.opacity == 0.5
    assert list(hover_markers.x) == [2.0, 2.0]
    assert list(hover_markers.y) == pytest.approx([10.0, 6.0])
    assert list(hover_markers.hovertext) == [f"{Performance(10e9)} · 6.67x @DRAM/L1", f"{Performance(6e9)} · 4.00x @add"]
    assert list(selected_markers.x) == [4.0, 4.0]
    assert list(selected_markers.y) == pytest.approx([12.0, 10.0])
    assert list(selected_markers.hovertext) == [f"{Performance(12e9)} · 4.80x @add", f"{Performance(10e9)} · 4.00x @DRAM/L1"]
    assert hover_markers.mode == selected_markers.mode == "markers"
    assert hover_markers.hovertemplate == selected_markers.hovertemplate == "%{hovertext}<extra></extra>"
    assert hover_markers.marker.size == selected_markers.marker.size == 10
    assert hover_markers.marker.opacity == selected_markers.marker.opacity == 0.5
    assert hover_markers.opacity is selected_markers.opacity is None
    assert all(trace.showlegend is False for trace in traces)
    assert hover_line.hoverinfo == selected_line.hoverinfo == "skip"


def test_point_guide_traces_clear_hover_and_selection_without_touching_layout() -> None:
    figure = {
        "data": [
            {
                "meta": {"carm_role": "application", "roof_id": "r1", "app_id": "app-1"},
                "x": [2.0, 3.0],
                "y": [1.5, 1.0],
            },
            {
                "meta": {"carm_role": "roof-ceiling"},
                "legendgroup": "r1",
                "name": "roof",
                "x": [1.0, 5.0],
                "y": [10.0, 10.0],
            },
        ],
        "layout": {
            "meta": {"carm_selected_point": {"roof_id": "r1", "app_id": "app-1", "point_index": 0}},
            "yaxis": {"range": [0.0, 3.0]},
            "shapes": [{"name": "unrelated-shape"}],
            "annotations": [{"name": "warning", "text": "warning"}],
        },
    }
    hover = {"points": [{"customdata": ["tip", {}, "r1", "app-1", 1]}]}
    with_hover = _point_guide_traces(figure, hover)
    without_hover = _point_guide_traces(figure, None)
    with_hover_by_key = {
        (trace.meta["guide"], trace.meta["kind"]): trace
        for trace in with_hover
        if trace.meta["carm_role"] == "point-guide"
    }
    traces_by_key = {
        (trace.meta["guide"], trace.meta["kind"]): trace
        for trace in without_hover
        if trace.meta["carm_role"] == "point-guide"
    }

    assert list(with_hover_by_key[("hover", "line")].x) == [3.0, 3.0]
    assert list(with_hover_by_key[("selected", "line")].x) == [2.0, 2.0]
    assert list(traces_by_key[("hover", "line")].x) == []
    assert list(traces_by_key[("hover", "intersections")].hovertext) == []
    assert list(traces_by_key[("selected", "line")].x) == [2.0, 2.0]
    assert list(traces_by_key[("selected", "intersections")].hovertext) == [f"{Performance(10e9)} · 6.67x @roof"]
    marker_hover = _point_guide_traces(figure, {"points": [{"customdata": None}]})
    marker_hover_by_key = {
        (trace.meta["guide"], trace.meta["kind"]): trace
        for trace in marker_hover
        if trace.meta["carm_role"] == "point-guide"
    }
    assert list(marker_hover_by_key[("selected", "line")].x) == [2.0, 2.0]
    assert list(marker_hover_by_key[("selected", "intersections")].hovertext) == [
        f"{Performance(10e9)} · 6.67x @roof"
    ]
    figure["layout"]["meta"]["carm_selected_point"] = None
    cleared = _point_guide_traces(figure, None)
    assert all(not trace.x and not trace.y for trace in cleared)
    assert figure["layout"]["shapes"] == [{"name": "unrelated-shape"}]
    assert figure["layout"]["annotations"] == [{"name": "warning", "text": "warning"}]
