"""Unit tests for GUI configuration (GUIMode and GUIConfig)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from carm_roofline.core.error import UserError
from carm_roofline.gui.config import GUIConfig, GUIMode

pytestmark = pytest.mark.unit


def test_guimode_carm_returns_all_default_flags() -> None:
    """GUIMode.from_name('carm') keeps every builder call site unchanged."""
    mode = GUIMode.from_name("carm")
    assert mode.show_app_dropdown is True
    assert mode.show_time_slider is False
    assert mode.has_export_tab is False


def test_guimode_paraver_flips_all_flags() -> None:
    """GUIMode.from_name('paraver') hides the apps dropdown, shows the slider and the export tab."""
    mode = GUIMode.from_name("paraver")
    assert (mode.show_app_dropdown, mode.show_time_slider, mode.has_export_tab) == (False, True, True)


def test_guimode_default_matches_carm() -> None:
    """The default GUIMode() is identical to the CARM mode."""
    assert GUIMode() == GUIMode.from_name("carm")


# -- GUIConfig argument parsing and validation ---------------------------------


def _carm_namespace(**overrides: object) -> argparse.Namespace:
    """Build a minimal CARM-mode Namespace for testing."""
    ns = argparse.Namespace(
        verbose=0,
        results_dir=Path("/tmp/carm"),
        gui_host="0.0.0.0",
        gui_port=8050,
        gui_debug=False,
        gui_mode="carm",
        paraver_trace=None,
        paraver_window_csv=None,
        paraver_legend_csv=None,
        paraver_use_colors=False,
        paraver_use_semantic_window=False,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def test_guiconfig_carm_defaults_do_not_validate_paraver() -> None:
    """CARM mode skips Paraver validation even with missing paraver args."""
    config = GUIConfig(_carm_namespace())
    assert config.gui_mode == "carm"
    assert config.paraver_trace is None
    assert config.paraver_window_csv is None
    assert config.paraver_use_colors is False


def test_guiconfig_paraver_args_default_to_none() -> None:
    """New Paraver args default to None/False."""
    config = GUIConfig(_carm_namespace())
    assert config.paraver_window_csv is None
    assert config.paraver_legend_csv is None
    assert config.paraver_use_colors is False
    assert config.paraver_use_semantic_window is False


def test_guiconfig_paraver_missing_trace_raises_user_error() -> None:
    """Paraver mode with no --paraver-trace raises UserError."""
    with pytest.raises(UserError, match="--paraver-trace is required"):
        GUIConfig(_carm_namespace(gui_mode="paraver"))


def test_guiconfig_paraver_missing_window_csv_raises_user_error() -> None:
    """Paraver mode with a trace but no window CSV raises UserError."""
    with pytest.raises(UserError, match="--paraver-window-csv is required"):
        GUIConfig(_carm_namespace(gui_mode="paraver", paraver_trace=Path("/tmp/t.prv")))


def test_guiconfig_paraver_trace_not_file_raises(tmp_path: Path) -> None:
    """Paraver mode with a non-existent trace raises UserError with path."""
    missing = tmp_path / "nope.prv"
    with pytest.raises(UserError, match=str(missing)):
        GUIConfig(
            _carm_namespace(
                gui_mode="paraver",
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
                gui_mode="paraver",
                paraver_trace=trace,
                paraver_window_csv=window,
            )
        )


def test_guiconfig_paraver_use_colors_no_legend_derives(tmp_path: Path) -> None:
    """--paraver-use-colors derives legend path from window CSV stem."""
    trace = tmp_path / "t.prv"
    window = tmp_path / "window.csv"
    trace.write_text("#dummy\n")
    window.write_text("#ts:CSV:RUNAPP:/p/t.prv:microseconds:\n")

    # No legend file → error.
    with pytest.raises(UserError, match="legend CSV not found"):
        GUIConfig(
            _carm_namespace(
                gui_mode="paraver",
                paraver_trace=trace,
                paraver_window_csv=window,
                paraver_use_colors=True,
            )
        )

    # Create the derived legend — now it passes.
    legend = tmp_path / "window.legend.csv"
    legend.write_text('-1 "idle" 128,128,128\n')
    config = GUIConfig(
        _carm_namespace(
            gui_mode="paraver",
            paraver_trace=trace,
            paraver_window_csv=window,
            paraver_use_colors=True,
        )
    )
    assert config.paraver_legend_csv == legend


def test_guiconfig_paraver_use_colors_explicit_legend(tmp_path: Path) -> None:
    """--paraver-use-colors with explicit --paraver-legend-csv validates it."""
    trace = tmp_path / "t.prv"
    window = tmp_path / "window.csv"
    legend = tmp_path / "explicit.csv"
    trace.write_text("#dummy\n")
    window.write_text("#ts:CSV:RUNAPP:/p/t.prv:microseconds:\n")

    # Explicit legend missing → error.
    with pytest.raises(UserError, match=str(legend)):
        GUIConfig(
            _carm_namespace(
                gui_mode="paraver",
                paraver_trace=trace,
                paraver_window_csv=window,
                paraver_legend_csv=legend,
                paraver_use_colors=True,
            )
        )

    # Create the explicit legend → passes.
    legend.write_text('-1 "idle" 128,128,128\n')
    config = GUIConfig(
        _carm_namespace(
            gui_mode="paraver",
            paraver_trace=trace,
            paraver_window_csv=window,
            paraver_legend_csv=legend,
            paraver_use_colors=True,
        )
    )
    assert config.paraver_legend_csv == legend


def test_guiconfig_paraver_valid_combo_passes(tmp_path: Path) -> None:
    """A valid Paraver trace + window CSV combos passes validation."""
    trace = tmp_path / "t.prv"
    window = tmp_path / "window.csv"
    trace.write_text("#dummy\n")
    window.write_text("#ts:CSV:RUNAPP:/p/t.prv:microseconds:\n")
    config = GUIConfig(
        _carm_namespace(
            gui_mode="paraver",
            paraver_trace=trace,
            paraver_window_csv=window,
            paraver_use_semantic_window=True,
        )
    )
    assert config.paraver_trace == trace
    assert config.paraver_window_csv == window
    assert config.paraver_use_semantic_window is True
    assert config.paraver_use_colors is False
