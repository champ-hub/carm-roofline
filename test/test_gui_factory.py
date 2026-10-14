from __future__ import annotations
import math
from typing import Any

import pytest

from carm_roofline.gui.callbacks.helpers import selection_payload
from carm_roofline.gui.callbacks.roofline import _point_guide_shapes


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


def test_point_guide_shapes_follow_stable_identity_and_visible_roof_ceiling() -> None:
    figure = {
        "data": [
            {
                "meta": {"carm_role": "application", "roof_id": "r1", "app_id": "app-1"},
                "x": [2.0, 4.0],
                "y": [1.5, 1.0],
            },
            {
                "meta": {"carm_role": "roof-ceiling"},
                "legendgroup": "r1",
                "name": "visible roof",
                "x": [1.0, 5.0],
                "y": [10.0, 10.0],
            },
            {
                "meta": {"carm_role": "roof-ceiling"},
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
            "shapes": [],
        },
    }
    hover = {"points": [{"curveNumber": 2, "pointNumber": 0, "customdata": ["tip", {}, "r1", "app-1", 0]}]}

    shapes = _point_guide_shapes(figure, hover)

    assert [(shape["name"], shape["x0"]) for shape in shapes] == [
        ("carm-hover-guide", 2.0),
        ("carm-selected-guide", 4.0),
    ]
    assert shapes[0]["y1"] == pytest.approx(1 / 3)
    assert shapes[1]["y1"] == pytest.approx(math.log10(12) / 3)

def test_point_guide_shapes_clear_hover_but_keep_selected_and_drop_stale_points() -> None:
    figure = {
        "data": [
            {
                "meta": {"carm_role": "application", "roof_id": "r1", "app_id": "app-1"},
                "x": [2.0],
                "y": [1.5],
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
            "shapes": [],
        },
    }

    assert [shape["name"] for shape in _point_guide_shapes(figure, None)] == ["carm-selected-guide"]
    stale_hover = {"points": [{"customdata": ["tip", {}, "removed-roof", "app-1", 0]}]}
    assert [shape["name"] for shape in _point_guide_shapes(figure, stale_hover)] == ["carm-selected-guide"]
