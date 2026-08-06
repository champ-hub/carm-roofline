"""Unit tests for GUI configuration (GUIMode and GUIConfig)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from carm_roofline.core.error import UserError
from carm_roofline.gui.config import GUIConfig, GUIMode

pytestmark = pytest.mark.unit


def test_guimode_carm_returns_all_default_flags() -> None:
    """CARM mode keeps every builder call site unchanged."""
    mode = GUIMode.CARM
    assert mode.show_app_dropdown is True
    assert mode.show_time_slider is False
    assert mode.has_export_tab is False


def test_guimode_paraver_flips_all_flags() -> None:
    """PARAVER hides the apps dropdown, shows the slider and the export tab."""
    mode = GUIMode.PARAVER
    assert (mode.show_app_dropdown, mode.show_time_slider, mode.has_export_tab) == (False, True, True)


def test_guimode_covers_exactly_the_valid_flag_combos() -> None:
    """Every member maps to a distinct valid flag combo (no invalid states)."""
    assert {(m.show_app_dropdown, m.show_time_slider, m.has_export_tab) for m in GUIMode} == {
        (True, False, False),
        (False, True, True),
    }


# -- GUIConfig argument parsing and validation ---------------------------------


def _carm_namespace(**overrides: object) -> argparse.Namespace:
    """Build a minimal Namespace for GUI config tests."""
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
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def test_guiconfig_carm_defaults_do_not_validate_paraver() -> None:
    """CARM mode skips Paraver validation even with missing paraver args."""
    config = GUIConfig(_carm_namespace())
    assert config.paraver_trace is None
    assert config.paraver_window_csv is None


def test_guiconfig_paraver_args_default_to_none() -> None:
    """New Paraver args default to None/False."""
    config = GUIConfig(_carm_namespace())
    assert config.paraver_window_csv is None
    assert config.paraver_use_semantic_window is False


def test_guiconfig_trace_enables_paraver_validation() -> None:
    """--paraver-trace alone flips Paraver validation on (window CSV then required)."""
    with pytest.raises(UserError, match="--paraver-window-csv is required"):
        GUIConfig(_carm_namespace(paraver_trace=Path("/tmp/t.prv")))


def test_guiconfig_paraver_missing_window_csv_raises_user_error() -> None:
    """Paraver mode with a trace but no window CSV raises UserError."""
    with pytest.raises(UserError, match="--paraver-window-csv is required"):
        GUIConfig(_carm_namespace(paraver_trace=Path("/tmp/t.prv")))


def test_guiconfig_paraver_trace_not_file_raises(tmp_path: Path) -> None:
    """Paraver mode with a non-existent trace raises UserError with path."""
    missing = tmp_path / "nope.prv"
    with pytest.raises(UserError, match=str(missing)):
        GUIConfig(
            _carm_namespace(
                paraver_trace=missing,
                paraver_window_csv=missing,
            )
        )


def test_guiconfig_paraver_window_csv_not_file_raises(tmp_path: Path) -> None:
    """Paraver mode with non-existent window CSV but valid trace raises."""
    trace = tmp_path / "t.prv"
    trace.write_text("#dummy\n")
    window = tmp_path / "nope.csv"
    with pytest.raises(UserError, match=str(window)):
        GUIConfig(
            _carm_namespace(
                paraver_trace=trace,
                paraver_window_csv=window,
            )
        )


def test_guiconfig_paraver_code_mode_requires_legend(tmp_path: Path) -> None:
    """Code-mode windows require the derived legend CSV (no mode token → code)."""
    trace = tmp_path / "t.prv"
    window = tmp_path / "window.csv"
    trace.write_text("#dummy\n")
    window.write_text("#ts:CSV:RUNAPP:/p/t.prv:microseconds:\n")

    # No legend file → error.
    with pytest.raises(UserError, match="legend CSV not found"):
        GUIConfig(
            _carm_namespace(
                paraver_trace=trace,
                paraver_window_csv=window,
            )
        )

    # Create the derived legend — now it passes.
    legend = tmp_path / "window.legend.csv"
    legend.write_text('-1 "idle" 128,128,128\n')
    config = GUIConfig(
        _carm_namespace(
            paraver_trace=trace,
            paraver_window_csv=window,
        )
    )
    assert config.paraver_window_csv == window


def test_guiconfig_paraver_gradient_mode_skips_legend(tmp_path: Path) -> None:
    """Gradient-mode windows need no legend CSV at all."""
    trace = tmp_path / "t.prv"
    window = tmp_path / "window.csv"
    trace.write_text("#dummy\n")
    window.write_text("#ts:CSV:RUNAPP:/p/t.prv:microseconds:window_in_null_gradient_mode:\n")
    config = GUIConfig(
        _carm_namespace(
            paraver_trace=trace,
            paraver_window_csv=window,
        )
    )
    assert config.paraver_window_csv == window


def test_guiconfig_paraver_valid_combo_passes(tmp_path: Path) -> None:
    """A valid Paraver trace + window CSV combo passes validation."""
    trace = tmp_path / "t.prv"
    window = tmp_path / "window.csv"
    trace.write_text("#dummy\n")
    window.write_text("#ts:CSV:RUNAPP:/p/t.prv:microseconds:\n")
    (tmp_path / "window.legend.csv").write_text('-1 "idle" 128,128,128\n')
    config = GUIConfig(
        _carm_namespace(
            paraver_trace=trace,
            paraver_window_csv=window,
            paraver_use_semantic_window=True,
        )
    )
    assert config.paraver_trace == trace
    assert config.paraver_window_csv == window
    assert config.paraver_use_semantic_window is True
