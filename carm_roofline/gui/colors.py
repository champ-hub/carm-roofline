"""Per-point marker colors for the Paraver scatter plot.

Computes the per-point color modes (Age, Thread ID, Load/store ratio, ISA); each mode's palette is chosen for
readability — see the per-mode comments. The "paraver" mode (legend/colorscale coloring) is the existing path in
gui/data.py; only its wire value is defined here.
"""

from __future__ import annotations

import colorsys
import hashlib

import numpy as np
import pandas as pd

from carm_roofline.paraver import trace_metric

# Color-mode wire values: stored in ParaverState.color_mode (dcc.Store JSON) and used as RadioItems values.
COLOR_MODE_PARAVER = "paraver"
COLOR_MODE_AGE = "age"
COLOR_MODE_THREAD = "thread"
COLOR_MODE_LDST = "ldst"
COLOR_MODE_ISA = "isa"

# (label, value) pairs for the Color accordion's radio group, display order.
COLOR_MODE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("Paraver colors", COLOR_MODE_PARAVER),
    ("Age", COLOR_MODE_AGE),
    ("Thread ID", COLOR_MODE_THREAD),
    ("Load/store ratio", COLOR_MODE_LDST),
    ("ISA", COLOR_MODE_ISA),
)

# A burst with no data for the selected mode (no memory ops / no FP work).
_NO_DATA_COLOR = "#999999"

# Age gradient: ColorBrewer "Blues" 9-scale endpoints (#c6dbef -> #08306b). Light->dark age semantics, with a gradient
# strong enough on the white plot background.
_AGE_START = (198, 219, 239)
_AGE_END = (8, 48, 107)

# Load/store ratio: diverging scale load-blue <-> mid-gray <-> store-red. The gray midpoint gives 50/50 an explicit
# neutral and keeps both extremes readable.
_LDST_LOAD = (0, 0, 255)
_LDST_STORE = (255, 0, 0)
_LDST_MID = (150, 150, 150)

# ISA palette: four distinct qualitative colors, each recognizable when one ISA dominates a burst.
_ISA_COLORS = {
    "scalar": (31, 119, 180),  # #1f77b4
    "sse": (255, 127, 14),  # #ff7f0e
    "avx2": (44, 160, 44),  # #2ca02c
    "avx512": (215, 48, 39),  # #d73027
}


def _interpolate_hex(start: tuple[int, int, int], end: tuple[int, int, int], factor: float) -> str:
    """Per-channel linear RGB interpolation between two colors, as #rrggbb."""

    r, g, b = (int(start[i] + factor * (end[i] - start[i])) for i in range(3))
    return f"#{r:02x}{g:02x}{b:02x}"


def _hash_to_color(thread_id: str) -> str:
    """Deterministic per-thread color: sha256 -> hue, saturation 0.8 / value 0.9, converted with stdlib colorsys.
    `% 360` keeps the hue in [0, 1), so it always falls inside the six HSV sectors."""
    digest = int(hashlib.sha256(thread_id.encode("utf-8")).hexdigest(), 16)
    r, g, b = colorsys.hsv_to_rgb((digest % 360) / 360.0, 0.8, 0.9)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def point_colors(trace: pd.DataFrame, color_mode: str) -> list[str]:
    """Per-row marker color for *trace* under *color_mode*.

    Only the point-color modes are computed here; COLOR_MODE_PARAVER is the existing legend/colorscale path in
    gui/data.py and raises here. Unknown modes raise ValueError.
    """
    if color_mode == COLOR_MODE_AGE:
        return _age_colors(len(trace))
    if color_mode == COLOR_MODE_THREAD:
        by_thread: dict[str, str] = {
            str(thread_id): _hash_to_color(str(thread_id)) for thread_id in trace["thread_id"].unique()
        }
        return [by_thread[str(thread_id)] for thread_id in trace["thread_id"]]
    if color_mode == COLOR_MODE_LDST:
        return _ldst_colors(trace_metric(trace, "load_share"))
    if color_mode == COLOR_MODE_ISA:
        return _isa_colors(trace)
    raise ValueError(f"unknown color mode: {color_mode!r}")


def _age_colors(n_points: int) -> list[str]:
    """Light -> dark over trace row order (first plotted point lightest). Spans the full gradient."""
    if n_points == 0:
        return []
    if n_points == 1:
        return [_interpolate_hex(_AGE_START, _AGE_END, 0.0)]
    return [_interpolate_hex(_AGE_START, _AGE_END, i / (n_points - 1)) for i in range(n_points)]


def _ldst_colors(load_share: pd.Series[float]) -> list[str]:
    """Blue at 100% loads, red at 0%, gray at 50/50; NaN (no memory ops) gray."""
    shares = np.asarray(load_share, dtype=float)
    no_data = np.isnan(shares)
    # NaN rows take the <= 0.5 branch (factor 1.0) and are overwritten below.
    shares = np.where(no_data, 0.5, shares)
    factor = np.where(shares <= 0.5, shares * 2.0, (shares - 0.5) * 2.0)
    lo = shares <= 0.5
    red = np.where(
        lo,
        _LDST_STORE[0] + factor * (_LDST_MID[0] - _LDST_STORE[0]),
        _LDST_MID[0] + factor * (_LDST_LOAD[0] - _LDST_MID[0]),
    )
    green = np.where(
        lo,
        _LDST_STORE[1] + factor * (_LDST_MID[1] - _LDST_STORE[1]),
        _LDST_MID[1] + factor * (_LDST_LOAD[1] - _LDST_MID[1]),
    )
    blue = np.where(
        lo,
        _LDST_STORE[2] + factor * (_LDST_MID[2] - _LDST_STORE[2]),
        _LDST_MID[2] + factor * (_LDST_LOAD[2] - _LDST_MID[2]),
    )
    rgb = np.stack([red, green, blue], axis=1).astype(int)
    colors = [f"#{int(r):02x}{int(g):02x}{int(b):02x}" for r, g, b in rgb]
    for i in np.nonzero(no_data)[0]:
        colors[i] = _NO_DATA_COLOR
    return colors


def _isa_colors(trace: pd.DataFrame) -> list[str]:
    """Weighted blend of the active ISAs' colors by per-ISA op share; no FP work (all shares 0 or NaN) gray."""
    pcts = np.stack(
        [
            np.asarray(trace_metric(trace, "isa_scalar_pct"), dtype=float),
            np.asarray(trace_metric(trace, "isa_sse_pct"), dtype=float),
            np.asarray(trace_metric(trace, "isa_avx2_pct"), dtype=float),
            np.asarray(trace_metric(trace, "isa_avx512_pct"), dtype=float),
        ],
        axis=1,
    )
    # Per-row weights: each ISA's share where not NaN and > 0, zero otherwise (matches the old active-dict filter).
    weights = np.where(~np.isnan(pcts) & (pcts > 0.0), pcts, 0.0)
    totals = weights.sum(axis=1)
    no_data = totals == 0.0
    isa_rgb = np.asarray([_ISA_COLORS[name] for name in _ISA_COLORS], dtype=float)
    # Per-channel weighted average, left-folded in the same order as the old sum(); rows with no active ISA divide by 1
    # and are overwritten with the no-data color below.
    blended = (
        weights[:, 0:1] * isa_rgb[0]
        + weights[:, 1:2] * isa_rgb[1]
        + weights[:, 2:3] * isa_rgb[2]
        + weights[:, 3:4] * isa_rgb[3]
    ) / np.where(no_data, 1.0, totals)[:, None]
    colors = [f"#{int(r):02x}{int(g):02x}{int(b):02x}" for r, g, b in np.rint(blended).astype(int)]
    for i in np.nonzero(no_data)[0]:
        colors[i] = _NO_DATA_COLOR
    return colors
