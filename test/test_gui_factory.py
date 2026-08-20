"""Unit tests for GUI factory wiring of the Paraver launch contract."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from carm_roofline.gui.config import GUIConfig
from carm_roofline.gui.factory import _clicked_point_residency, _selection_payload, create_app
from carm_roofline.gui.ids import StoreID
from carm_roofline.gui.providers import ParaverData
from carm_roofline.paraver import ParaverWindowMode

pytestmark = pytest.mark.unit


def _paraver_namespace(trace: Path, window: Path | None, use_semantic_window: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        verbose=0,
        results_dir=Path("/tmp/carm"),
        gui_host="0.0.0.0",
        gui_port=8050,
        gui_debug=False,
        paraver_trace=trace,
        paraver_window_csv=window,
        paraver_use_semantic_window=use_semantic_window,
    )


def _fake_paraver_data() -> ParaverData:
    return ParaverData(
        trace=pd.DataFrame(
            {
                "thread_id": ["1.1.1"],
                "time_s": [0.5],
                "duration_s": [1.0],
                "state_code": [1.0],
                "flops": [16.0],
                "bytes": [16.0],
                "ai": [1.0],
                "perf": [16.0],
            }
        ),
        label="t — w.csv",
        window_mode=ParaverWindowMode.CODE,
        time_unit="nanoseconds",
        prv_path="/p/t.prv",
        legend=None,
    )


class _FakeProvider:
    """Provider double capturing constructor args; returns one fixed ParaverData."""

    def __init__(self, trace_path: Path, window_csv_path: Path, **kwargs: Any) -> None:
        self.captured_trace = trace_path
        self.captured_window = window_csv_path
        self.captured_kwargs = kwargs

    def load(self) -> ParaverData:
        return _fake_paraver_data()

    @property
    def window_extent(self) -> tuple[float, float] | None:
        return (0.0, 2.0)


def _find_component(node: Any, comp_id: str) -> Any | None:
    """Walk a Dash layout tree for the component with the given id."""
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
    """The time-window slider must label ticks with bounded-precision values.

    Dash 4 auto-generates marks at fixed 0/25/50/75/100% positions labeled with
    raw floats (e.g. ``0.06614603825000001``); explicit marks must replace those
    with nice-step labels inside the trace extent.
    """
    from carm_roofline.gui.components import build_plot_area
    from carm_roofline.gui.config import GUIMode
    from carm_roofline.gui.ids import ParaverID

    div = build_plot_area(GUIMode.PARAVER, (0.0, 0.26458415300000004))
    slider = _find_component(div, ParaverID.SLIDER_TIME_WINDOW)
    assert slider is not None
    marks = slider.marks
    assert isinstance(marks, dict) and marks
    assert all(0.0 <= key <= 0.26458415300000004 for key in marks)
    assert "0.06614603825000001" not in marks.values()
    assert set(marks.values()) == {"0", "0.05", "0.1", "0.15", "0.2", "0.25"}


def test_time_window_marks_adapt_precision_to_trace_scale() -> None:
    """Tick labels must carry just enough decimals for the trace's time scale."""
    from carm_roofline.gui.components import _time_window_marks

    # Sub-second trace: 0.05 s step, two decimals at most.
    assert _time_window_marks(0.0, 0.26458415300000004) == {
        0.0: "0",
        0.05: "0.05",
        0.1: "0.1",
        0.15: "0.15",
        0.2: "0.2",
        0.25: "0.25",
    }
    # Multi-second trace: 0.5 s step, whole-second labels stay clean.
    assert _time_window_marks(0.0, 2.0) == {0.0: "0", 0.5: "0.5", 1.0: "1", 1.5: "1.5", 2.0: "2"}
    # Microsecond-scale trace: five decimals are required to stay meaningful.
    marks = _time_window_marks(0.0, 0.00026458415300000004)
    assert "0.00005" in marks.values()
    assert all(len(label.split(".")[1]) <= 5 for label in marks.values() if "." in label)
    # Off-origin ranges: first tick is the first step multiple >= lo, all keys in range.
    marks = _time_window_marks(0.91, 1.0)
    assert all(0.91 <= key <= 1.0 for key in marks)
    # Degenerate range: a single mark at the trace time.
    assert _time_window_marks(1.5, 1.5) == {1.5: "1.5"}


def _find_store_data(node: Any, store_id: str) -> dict[str, Any] | None:
    """Walk a Dash layout tree for a dcc.Store with the given id."""
    if node is None:
        return None
    if getattr(node, "id", None) == store_id:
        return node.data
    children = getattr(node, "children", None)
    if isinstance(children, list):
        for child in children:
            found = _find_store_data(child, store_id)
            if found is not None:
                return found
    return None


def test_create_app_paraver_wires_config_and_leaves_app_ids_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Paraver config reaches the provider; the initial roof preselects no records."""
    trace = tmp_path / "t.prv"
    trace.write_text("#dummy\n", encoding="utf-8")
    window = tmp_path / "w.csv"
    window.write_text(
        "#ts:CSV:RUNAPP:/p/t.prv:nanoseconds:window_in_code_mode\n1.1.1\t0.0\t1000000000.0\t1.0\n",
        encoding="utf-8",
    )
    (tmp_path / "w.legend.csv").write_text('1.000000 "Running" 0,0,255\n', encoding="utf-8")

    monkeypatch.setattr("carm_roofline.gui.factory.ParaverProvider", _FakeProvider)

    app = create_app(GUIConfig(_paraver_namespace(trace, window)))
    store_data = _find_store_data(app.layout(), StoreID.ROOF_STORE)
    assert store_data is not None

    # Paraver points are plotted directly from the trace; no record ids are preselected.
    assert store_data["roofs"][0]["app_ids"] == []

    # Default launch: no semantic window -> time window unset (full range);
    # the AI filter defaults to its minimum active threshold (1e-5), not "off".
    assert store_data["paraver"]["time_window"] is None
    assert store_data["paraver"]["ai_threshold"] == pytest.approx(1e-5)
    # The duration filter defaults to its minimum active threshold (100 us), not "off".
    assert store_data["paraver"]["duration_threshold"] == pytest.approx(1e-4)


def test_create_app_paraver_trace_only_loads_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A trace without a window CSV still loads the provider (no warn-and-skip)."""
    trace = tmp_path / "t.prv"
    trace.write_text("#dummy\n", encoding="utf-8")

    calls: list[tuple[Path, Path | None]] = []

    class _RecordingProvider(_FakeProvider):
        def __init__(self, trace_path: Path, window_csv_path: Path | None, **kwargs: Any) -> None:
            calls.append((trace_path, window_csv_path))
            super().__init__(trace_path, window_csv_path, **kwargs)

    monkeypatch.setattr("carm_roofline.gui.factory.ParaverProvider", _RecordingProvider)

    app = create_app(GUIConfig(_paraver_namespace(trace, None)))
    store_data = _find_store_data(app.layout(), StoreID.ROOF_STORE)
    assert store_data is not None

    # The provider was constructed with the trace and a None window CSV, and
    # its points are plotted directly (no record ids preselected).
    assert calls == [(trace, None)]
    assert store_data["roofs"][0]["app_ids"] == []


def test_create_app_paraver_semantic_window_sets_initial_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--paraver-use-semantic-window seeds the initial time window from the provider extent."""
    trace = tmp_path / "t.prv"
    trace.write_text("#dummy\n", encoding="utf-8")
    window = tmp_path / "w.csv"
    window.write_text(
        "#ts:CSV:RUNAPP:/p/t.prv:nanoseconds:window_in_code_mode\n1.1.1\t0.0\t1000000000.0\t1.0\n",
        encoding="utf-8",
    )
    (tmp_path / "w.legend.csv").write_text('1.000000 "Running" 0,0,255\n', encoding="utf-8")

    monkeypatch.setattr("carm_roofline.gui.factory.ParaverProvider", _FakeProvider)

    app = create_app(GUIConfig(_paraver_namespace(trace, window, use_semantic_window=True)))
    store_data = _find_store_data(app.layout(), StoreID.ROOF_STORE)
    assert store_data is not None
    assert store_data["paraver"]["time_window"] == [0.0, 2.0]


def test_create_app_without_trace_uses_carm_layout() -> None:
    """No --paraver-trace means CARM mode: the Paraver time-window slider is absent."""
    ns = argparse.Namespace(
        verbose=0,
        results_dir=Path("/tmp/carm"),
        gui_host="0.0.0.0",
        gui_port=8050,
        gui_debug=False,
        paraver_trace=None,
        paraver_window_csv=None,
        paraver_use_semantic_window=False,
    )
    app = create_app(GUIConfig(ns))
    from carm_roofline.gui.ids import ParaverID

    assert _find_component(app.layout(), ParaverID.SLIDER_TIME_WINDOW) is None


def test_navbar_paraver_tab_label_uses_short_name() -> None:
    """The paraver navbar button is labeled "Paraver", not "Paraver Export"."""
    from carm_roofline.gui.components import build_navbar
    from carm_roofline.gui.config import GUIMode
    from carm_roofline.gui.data import ActivePanel
    from carm_roofline.gui.ids import NavbarID

    navbar = build_navbar(ActivePanel.CARM_VIEW, GUIMode.PARAVER)
    button = _find_component(navbar, NavbarID.BTN_EXPORT)
    assert button is not None
    assert button.children == "Paraver"


def test_export_panel_accordion_groups_export_filter_and_color() -> None:
    """The paraver panel shows Export, Filtering, and Color accordion items with the AI slider.

    The slider spans log10(AI) from the "Off" limit (1e-6, a decade below 1e-5)
    to 1e-2 with decade marks; the 1e-5 position is the default (minimum active
    filter).
    """
    import dash_bootstrap_components as dbc
    from dash import dcc

    from carm_roofline.gui.components import build_export_panel
    from carm_roofline.gui.data import RoofStore
    from carm_roofline.gui.ids import ExportPanelID

    panel = build_export_panel(RoofStore())
    accordion = next(child for child in panel.children if isinstance(child, dbc.Accordion))
    assert isinstance(accordion.children, list) and len(accordion.children) == 3
    assert [item.title for item in accordion.children] == ["Export", "Filtering", "Color"]

    export_item, filtering_item, color_item = accordion.children
    for btn_id, status_id in (
        (ExportPanelID.BTN_EXPORT_PERFORMANCE, ExportPanelID.STATUS_PERFORMANCE),
        (ExportPanelID.BTN_EXPORT_AI, ExportPanelID.STATUS_AI),
        (ExportPanelID.BTN_EXPORT_LDST_PERCENT, ExportPanelID.STATUS_LDST_PERCENT),
        (ExportPanelID.BTN_EXPORT_ROOF_LABELS, ExportPanelID.STATUS_ROOF_LABELS),
        (ExportPanelID.BTN_EXPORT_REGION, ExportPanelID.STATUS_REGION),
        (ExportPanelID.BTN_EXPORT_PROXIMITY, ExportPanelID.STATUS_PROXIMITY),
    ):
        assert _find_component(export_item, btn_id) is not None
        assert _find_component(export_item, status_id) is not None

    radio = _find_component(color_item, ExportPanelID.RADIO_COLOR_MODE)
    assert radio is not None
    assert radio.value == "paraver"
    assert [option["label"] for option in radio.options] == [
        "Paraver colors",
        "Age",
        "Thread ID",
        "Load/store ratio",
        "ISA",
    ]
    assert [option["value"] for option in radio.options] == ["paraver", "age", "thread", "ldst", "isa"]

    # Exports are written to disk: the Export item holds exactly the six
    # button+status rows, with no dcc.Download component left over.
    def _contains_download(node: Any) -> bool:
        if isinstance(node, dcc.Download):
            return True
        children = getattr(node, "children", None)
        return isinstance(children, list) and any(_contains_download(c) for c in children)

    assert not _contains_download(export_item)
    assert len(export_item.children) == 6  # six export rows

    slider = _find_component(filtering_item, ExportPanelID.SLIDER_AI_THRESHOLD)
    assert slider is not None
    assert slider.min == -6.0
    assert slider.max == -2.0
    assert slider.step == 0.2
    # Default store state (filter at 1e-5) puts the slider on the 1e-5 position.
    assert slider.value == -5.0
    assert slider.marks == {-6.0: "Off", -5.0: "1e-5", -4.0: "1e-4", -3.0: "1e-3", -2.0: "1e-2"}

    duration_slider = _find_component(filtering_item, ExportPanelID.SLIDER_DURATION_THRESHOLD)
    assert duration_slider is not None
    assert duration_slider.min == -6.0
    assert duration_slider.max == -1.0
    assert duration_slider.step == 0.2
    assert duration_slider.value == -4.0  # default store state: 100 us threshold
    assert duration_slider.marks == {
        -6.0: "Off",
        -5.0: "10 us",
        -4.0: "100 us",
        -3.0: "1 ms",
        -2.0: "10 ms",
        -1.0: "100 ms",
    }
    assert duration_slider.tooltip == {"placement": "bottom", "transform": "paraverDuration"}


def test_clicked_point_residency_parses_customdata() -> None:
    """clickData payloads map to (roof_id, fractions); invalid payloads return None."""
    assert _clicked_point_residency(None) is None
    assert _clicked_point_residency({}) is None
    assert _clicked_point_residency({"points": []}) is None
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
    assert _clicked_point_residency(valid) == ("r1", {"l1": 0.6, "l2": 0.3, "l3": 0.08, "dram": 0.02})
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
    assert _clicked_point_residency(valid_3key) == ("r1", {"l1": 0.6, "l2": 0.3, "l3plus": 0.1})
    # missing element 2 -> None
    assert _clicked_point_residency({"points": [{"customdata": ["t", {}]}]}) is None
    # non-str roof id -> None
    assert _clicked_point_residency({"points": [{"customdata": ["t", {"l1": 0.6}, 42]}]}) is None
    # empty fractions dict -> None
    assert _clicked_point_residency({"points": [{"customdata": ["t", {}, "r1"]}]}) is None
    # non-dict fractions -> None
    assert _clicked_point_residency({"points": [{"customdata": ["t", "residency", "r1"]}]}) is None
    # customdata not a list -> None
    assert _clicked_point_residency({"points": [{"customdata": "nope"}]}) is None


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
    assert _selection_payload(valid_click, "roofline-bg-click.value", {"roof_id": "r1"}) is None
    assert _selection_payload(None, "roofline-bg-click.value", None) is None
    assert _selection_payload(valid_click, "roofline-plot.clickData", None) == {
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
    assert _selection_payload(valid_click_3key, "roofline-plot.clickData", None) == {
        "roof_id": "r1",
        "fractions": {"l1": 0.6, "l2": 0.3, "l3plus": 0.1},
    }
    # Echo of the callback's own clickData reset keeps the current selection.
    assert _selection_payload(None, "roofline-plot.clickData", {"roof_id": "r1", "fractions": {"l1": 0.6}}) == {
        "roof_id": "r1",
        "fractions": {"l1": 0.6},
    }
    # non-app point (no customdata) -> None
    assert _selection_payload({"points": [{"x": 1, "y": 2}]}, "roofline-plot.clickData", None) is None
