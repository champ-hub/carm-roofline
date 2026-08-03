"""Unit tests for GUI configuration (GUIMode)."""

from __future__ import annotations

import pytest

from carm_roofline.gui.config import GUIMode

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
