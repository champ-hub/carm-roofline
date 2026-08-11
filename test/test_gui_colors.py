"""Unit tests for the per-point Paraver color modes (carm_roofline/gui/colors.py)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from carm_roofline.gui.colors import (
    COLOR_MODE_AGE,
    COLOR_MODE_ISA,
    COLOR_MODE_LDST,
    COLOR_MODE_PARAVER,
    COLOR_MODE_THREAD,
    point_colors,
)

pytestmark = pytest.mark.unit


def _trace(n: int = 3, **overrides: object) -> pd.DataFrame:
    data: dict[str, object] = {
        "thread_id": [str(i) for i in range(n)],
        "load_share": [0.5] * n,
        "isa_scalar_pct": [0.0] * n,
        "isa_sse_pct": [0.0] * n,
        "isa_avx2_pct": [0.0] * n,
        "isa_avx512_pct": [0.0] * n,
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_point_colors_age_spans_light_to_dark() -> None:
    """Age colors the first point lightest and the last darkest, spanning the
    full gradient (the legacy index/n formula stopped short of the dark end)."""
    assert point_colors(_trace(n=3), COLOR_MODE_AGE) == ["#c6dbef", "#6785ad", "#08306b"]


def test_point_colors_age_single_point_uses_start_color() -> None:
    assert point_colors(_trace(n=1), COLOR_MODE_AGE) == ["#c6dbef"]


def test_point_colors_thread_deterministic_per_thread() -> None:
    """Same thread id always yields the same color; different ids differ."""
    trace = _trace(thread_id=["0", "1", "0"])
    colors = point_colors(trace, COLOR_MODE_THREAD)
    assert colors == ["#2d5be5", "#b7e52d", "#2d5be5"]
    assert len(set(colors)) == 2


def test_point_colors_ldst_ratio_endpoints_mid_and_no_data() -> None:
    """100% loads blue, 0% red, 50/50 gray; no memory ops (NaN) also gray."""
    trace = _trace(n=4, load_share=[1.0, 0.0, 0.5, math.nan])
    assert point_colors(trace, COLOR_MODE_LDST) == ["#0000ff", "#ff0000", "#969696", "#999999"]


def test_point_colors_ldst_intermediate_factors() -> None:
    """Mid-gradient shares interpolate both branches toward the gray midpoint."""
    trace = _trace(n=3, load_share=[0.25, 0.75, math.nan])
    assert point_colors(trace, COLOR_MODE_LDST) == ["#ca4b4b", "#4b4bca", "#999999"]


def test_point_colors_isa_nan_excluded_and_normalized() -> None:
    """NaN shares are ignored (not a weight); partial totals normalize per row."""
    trace = _trace(
        n=2,
        isa_scalar_pct=[50.0, 0.0],
        isa_sse_pct=[math.nan, 100.0 / 3.0],
        isa_avx2_pct=[50.0, 100.0 / 3.0],
        isa_avx512_pct=[0.0, 100.0 / 3.0],
    )
    assert point_colors(trace, COLOR_MODE_ISA) == ["#268c70", "#ab7020"]


def test_point_colors_isa_blend_and_no_data() -> None:
    """Weights blend the active ISAs; no FP work (all shares 0/NaN) -> gray."""
    trace = _trace(
        isa_scalar_pct=[100.0 / 3.0, 0.0, 0.0],
        isa_sse_pct=[200.0 / 3.0, 0.0, 0.0],
        isa_avx2_pct=[0.0, 100.0, 0.0],
        isa_avx512_pct=[0.0, 0.0, math.nan],
    )
    assert point_colors(trace, COLOR_MODE_ISA) == ["#b47c45", "#2ca02c", "#999999"]


def test_point_colors_paraver_mode_raises() -> None:
    """The 'paraver' mode is the existing legend/colorscale path, not computed here."""
    with pytest.raises(ValueError, match="unknown color mode"):
        point_colors(_trace(), COLOR_MODE_PARAVER)


def test_point_colors_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="unknown color mode"):
        point_colors(_trace(), "precision")
