"""Unit tests for GUI factory wiring of the Paraver launch contract."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from carm_roofline.gui.config import GUIConfig
from carm_roofline.gui.factory import create_app
from carm_roofline.gui.ids import StoreID
from carm_roofline.gui.providers import ParaverData
from carm_roofline.paraver import ParaverWindowMode

pytestmark = pytest.mark.unit


def _paraver_namespace(trace: Path, window: Path, use_semantic_window: bool = False) -> argparse.Namespace:
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

    # Default launch: no semantic window -> time window unset (full range).
    assert store_data["paraver"]["time_window"] is None


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
