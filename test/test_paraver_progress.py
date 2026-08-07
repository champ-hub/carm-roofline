"""Unit tests for the legacy-compatible Paraver progress bar (paraver/progress.py)."""

from __future__ import annotations

import pytest

from carm_roofline.paraver.progress import ProgressBar

pytestmark = pytest.mark.unit

# Byte-exact lines of the Paraver progress protocol (porting-reference
# 08-environment-and-deployment.md §4.2): '#' × segments, spaces to width 30,
# "] {progress*100:.1f}%", '\r'-terminated; the completed line adds '\n'.
ZERO_LINE = "[                              ] 0.0%\r"
MID_LINE = "[###############               ] 50.0%\r"
CAP_LINE = "[############################# ] 99.0%\r"
FULL_LINE = "[##############################] 100.0%"


def test_update_zero_is_byte_exact(capsys: pytest.CaptureFixture[str]) -> None:
    ProgressBar(total=100).update(0)
    assert capsys.readouterr().out == ZERO_LINE


def test_update_mid_progress_segments(capsys: pytest.CaptureFixture[str]) -> None:
    ProgressBar(total=100).update(50)
    assert capsys.readouterr().out == MID_LINE  # ceil(30 * 0.5) = 15 segments


def test_update_caps_visible_percent_below_final(capsys: pytest.CaptureFixture[str]) -> None:
    ProgressBar(total=100).update(99)
    assert capsys.readouterr().out == CAP_LINE  # min(99/100, .99) → ceil(29.7) capped at 29


def test_full_bar_printed_exactly_once(capsys: pytest.CaptureFixture[str]) -> None:
    bar = ProgressBar(total=100)
    bar.update(100)
    bar.update(100)
    bar.update(250)  # after completion, updates are no-ops
    out = capsys.readouterr().out
    assert out.count(FULL_LINE) == 1
    assert out.endswith(FULL_LINE + "\r\n")


def test_zero_total_is_noop(capsys: pytest.CaptureFixture[str]) -> None:
    ProgressBar(total=0).update(0)
    ProgressBar(total=0).update(5)
    assert capsys.readouterr().out == ""


def test_total_fixed_after_construction(capsys: pytest.CaptureFixture[str]) -> None:
    # Provider builds the bar before paramedir (total unknown); build_trace_table
    # fixes total to the merged burst count afterwards.
    bar = ProgressBar(total=1)
    bar.update(0)
    bar.total = 150
    bar.update(75)
    assert capsys.readouterr().out == ZERO_LINE + MID_LINE
