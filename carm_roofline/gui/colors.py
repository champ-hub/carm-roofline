"""Per-point marker colors for the Paraver scatter plot.

Computes the per-point color modes (Age, Thread ID, Load/store ratio, ISA); each mode's palette is chosen for
readability — see the per-mode comments. The "paraver" mode (legend/colorscale coloring) is the existing path in
gui/data.py; only its wire value is defined here.
"""

from __future__ import annotations

import colorsys
import hashlib
import math

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


def _blend_hex(weights: dict[str, float], colors: dict[str, tuple[int, int, int]]) -> str:
    """Weighted per-channel RGB average, as #rrggbb; empty weights -> gray."""
    if not weights:
        return _NO_DATA_COLOR

    total = sum(weights.values())
    r, g, b = (round(sum(w * colors[key][i] for key, w in weights.items()) / total) for i in range(3))
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
        return [_hash_to_color(str(thread_id)) for thread_id in trace["thread_id"]]
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
    colors = []
    for share in load_share:
        if math.isnan(share):
            colors.append(_NO_DATA_COLOR)
        elif share <= 0.5:
            colors.append(_interpolate_hex(_LDST_STORE, _LDST_MID, share * 2.0))
        else:
            colors.append(_interpolate_hex(_LDST_MID, _LDST_LOAD, (share - 0.5) * 2.0))
    return colors


def _isa_colors(trace: pd.DataFrame) -> list[str]:
    """Weighted blend of the active ISAs' colors by per-ISA op share; no FP work (all shares 0 or NaN) gray."""
    scalar = trace_metric(trace, "isa_scalar_pct")
    sse = trace_metric(trace, "isa_sse_pct")
    avx2 = trace_metric(trace, "isa_avx2_pct")
    avx512 = trace_metric(trace, "isa_avx512_pct")
    colors = []
    for row in zip(scalar, sse, avx2, avx512):
        active = {
            key: float(value)
            for key, value in zip(("scalar", "sse", "avx2", "avx512"), row)
            if not math.isnan(value) and value > 0
        }
        colors.append(_blend_hex(active, _ISA_COLORS))
    return colors
