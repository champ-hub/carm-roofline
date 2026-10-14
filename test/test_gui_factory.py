from __future__ import annotations

from typing import Any

import pytest

from carm_roofline.gui.callbacks.helpers import clicked_point_residency, selection_payload

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


def test_clicked_point_residency_parses_customdata() -> None:
    """clickData payloads map to (roof_id, fractions); invalid payloads return None."""
    assert clicked_point_residency(None) is None
    assert clicked_point_residency({}) is None
    assert clicked_point_residency({"points": []}) is None
    valid = {
        "points": [
            {
                "customdata": [
                    "<b>tooltip</b>",
                    {"l1": 0.6, "l2": 0.3, "l3": 0.08, "dram": 0.02},
                    "r1",
                ]
            }
        ]
    }
    assert clicked_point_residency(valid) == ("r1", {"l1": 0.6, "l2": 0.3, "l3": 0.08, "dram": 0.02})
    # 3-bucket schema: keys pass through untouched (no shape assumption in parsing).
    valid_3key = {
        "points": [
            {
                "customdata": [
                    "<b>tooltip</b>",
                    {"l1": 0.6, "l2": 0.3, "l3plus": 0.1},
                    "r1",
                ]
            }
        ]
    }
    assert clicked_point_residency(valid_3key) == ("r1", {"l1": 0.6, "l2": 0.3, "l3plus": 0.1})
    # missing element 2 -> None
    assert clicked_point_residency({"points": [{"customdata": ["t", {}]}]}) is None
    # non-str roof id -> None
    assert clicked_point_residency({"points": [{"customdata": ["t", {"l1": 0.6}, 42]}]}) is None
    # empty fractions dict -> None
    assert clicked_point_residency({"points": [{"customdata": ["t", {}, "r1"]}]}) is None
    # non-dict fractions -> None
    assert clicked_point_residency({"points": [{"customdata": ["t", "residency", "r1"]}]}) is None
    # customdata not a list -> None
    assert clicked_point_residency({"points": [{"customdata": "nope"}]}) is None


def test_selection_payload_clears_on_background_click() -> None:
    """A background-click trigger clears the selection even when clickData holds an old point."""
    valid_click = {
        "points": [
            {
                "customdata": [
                    "<b>tooltip</b>",
                    {"l1": 0.6, "l2": 0.3, "l3": 0.08, "dram": 0.02},
                    "r1",
                ]
            }
        ]
    }
    assert selection_payload(valid_click, "roofline-bg-click.value", {"roof_id": "r1"}) is None
    assert selection_payload(None, "roofline-bg-click.value", None) is None
    assert selection_payload(valid_click, "roofline-plot.clickData", None) == {
        "roof_id": "r1",
        "fractions": {"l1": 0.6, "l2": 0.3, "l3": 0.08, "dram": 0.02},
    }
    # 3-bucket schema: fractions pass through untouched.
    valid_click_3key = {
        "points": [
            {
                "customdata": [
                    "<b>tooltip</b>",
                    {"l1": 0.6, "l2": 0.3, "l3plus": 0.1},
                    "r1",
                ]
            }
        ]
    }
    assert selection_payload(valid_click_3key, "roofline-plot.clickData", None) == {
        "roof_id": "r1",
        "fractions": {"l1": 0.6, "l2": 0.3, "l3plus": 0.1},
    }
    # Echo of the callback's own clickData reset keeps the current selection.
    assert selection_payload(None, "roofline-plot.clickData", {"roof_id": "r1", "fractions": {"l1": 0.6}}) == {
        "roof_id": "r1",
        "fractions": {"l1": 0.6},
    }
    # non-app point (no customdata) -> None
    assert selection_payload({"points": [{"x": 1, "y": 2}]}, "roofline-plot.clickData", None) is None
