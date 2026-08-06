"""Export of the displayed roofline data back to Paraver.

Each export writes Paraver CSV trace files (a ``#`` metadata header line plus
tab-separated ``ThreadID/Timestamp/Duration/value`` rows) that Paraver imports
as event windows. Files are written next to the ``.prv`` trace and silently
overwritten on repeat clicks; the absolute path of each written file is
printed to the GUI's terminal.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from carm_roofline.gui.providers import ParaverData
from carm_roofline.paraver import (
    CODE_WINDOW_MODE,
    DEFAULT_CSV_PRECISION,
    GRADIENT_WINDOW_MODE,
    time_unit_to_seconds,
    trace_metric,
)
from carm_roofline.roofline_assembly import AssembledRoofline


@dataclass(frozen=True)
class ExportFile:
    """One Paraver CSV file to write: its file name and full text content."""

    name: str
    content: str


@dataclass(frozen=True)
class LegendRow:
    """One Paraver legend entry: numeric code, quoted label, and RGB color."""

    code: int
    label: str
    r: int
    g: int
    b: int


PERFORMANCE_FILENAME = "carm_gflops.csv"
AI_FILENAME = "carm_ai.csv"
ROOF_LABELS_FILENAME = "carm_roofs.csv"
ROOF_LABELS_LEGEND_FILENAME = "carm_roofs.legend.csv"
REGION_FILENAME = "carm_roofline_region.csv"
REGION_LEGEND_FILENAME = "carm_roofline_region.legend.csv"
PROXIMITY_SUFFIXES = {"L1": "l1", "L2": "l2", "L3": "l3", "DRAM": "dram"}
ROOF_LEVEL_CODES = {"L1": 1, "L2": 2, "L3": 3, "DRAM": 4}
ROOF_LABEL_LEGEND = (  # six rows, incl. 5 which data never emits
    LegendRow(1, "L1", 0, 255, 0),
    LegendRow(2, "L2", 0, 0, 255),
    LegendRow(3, "L3", 255, 165, 0),
    LegendRow(4, "DRAM", 255, 0, 0),
    LegendRow(5, "No Floating Point Operations Found", 75, 0, 130),
    LegendRow(6, "Above L1", 255, 192, 203),
)
REGION_LEGEND = (
    LegendRow(1, "Memory Bound", 0, 0, 255),
    LegendRow(2, "Mixed", 128, 0, 128),
    LegendRow(3, "Compute Bound", 255, 0, 0),
)


def build_csv_metadata_line(
    prv_path: str,
    time_unit: str,
    window_mode: str,
    vmin: float | int,
    vmax: float | int,
    timestamp: str | None = None,
    vmin_vmax_precision: int = DEFAULT_CSV_PRECISION.header,
) -> str:
    """'#<YYYYMMDDHHMMSS>:CSV:RUNAPP:<prv_path>:<time_unit>:<window_mode>:<vmin>:<vmax>'.

    timestamp defaults to datetime.now().strftime("%Y%m%d%H%M%S"); the parameter
    exists so tests pin it. time_unit '' → 'Unknown' (legacy default). vmin/vmax
    are formatted with *vmin_vmax_precision* decimals (6 by default, mirroring
    the window CSV header's own vmin/vmax fields).
    """
    ts = timestamp or datetime.now().strftime("%Y%m%d%H%M%S")
    unit = time_unit if time_unit else "Unknown"
    return (
        f"#{ts}:CSV:RUNAPP:{prv_path}:{unit}:{window_mode}:"
        f"{vmin:.{vmin_vmax_precision}f}:{vmax:.{vmin_vmax_precision}f}"
    )


def natural_sort_key(thread_id: str) -> tuple[int, ...]:
    """Dot-segment numeric key ('2.1' → (2,1)) so '2.1' < '10.0'."""
    return tuple(int(part) for part in str(thread_id).split("."))


def serialize_legend(rows: Sequence[LegendRow]) -> str:
    """'<code> "<label>",<r>,<g>,<b>\\n' per row, trailing newline on every line."""
    return "".join(f'{row.code} "{row.label}",{row.r},{row.g},{row.b}\n' for row in rows)


def write_export_files(files: Sequence[ExportFile], output_dir: str | Path) -> list[Path]:
    """
    Write each ExportFile into *output_dir*; return the sorted list of written
    absolute paths (for terminal/status display).
    """
    written = []
    for file in files:
        path = Path(output_dir) / file.name
        path.write_text(file.content, encoding="utf-8")
        written.append(path.resolve())
    return sorted(written)


def _csv_content(
    trace: pd.DataFrame,
    paraver: ParaverData,
    values: pd.Series[Any],
    window_mode: str,
    vmin: float | int,
    vmax: float | int,
) -> str:
    """Serialize trace rows to one Paraver CSV: metadata header + 4-column tab rows.

    time_s/duration_s are scaled back from seconds to the header's declared unit;
    rows are ordered by natural thread_id then time_s, ascending.
    Every numeric cell — header vmin/vmax, time, duration, value — is formatted
    with ``paraver.precision`` (the input window CSV's decimal places), so
    Paraver re-imports the file exactly like its own round-trip exports.
    """
    p = paraver.precision
    header = build_csv_metadata_line(
        paraver.prv_path, paraver.time_unit, window_mode, vmin, vmax, vmin_vmax_precision=p.header
    )
    scale = 1.0 / time_unit_to_seconds(paraver.time_unit)
    out = pd.DataFrame(
        {
            "thread_id": trace["thread_id"].astype(str),
            "time_s": trace_metric(trace, "time_s") * scale,
            "duration_s": trace_metric(trace, "duration_s") * scale,
            "value": values,
        }
    )
    # Two stable sorts: time first, then natural thread key — primary thread_id,
    # secondary time_s, without sorting tuple-typed object columns.
    # Format AFTER sorting: the sort must stay numeric (formatted strings would
    # sort lexicographically, mis-ordering "1000000.00" before "500000.00").
    out["_key"] = out["thread_id"].map(natural_sort_key)
    out = out.sort_values("time_s", kind="stable").sort_values("_key", kind="stable")
    out = out.drop(columns=["_key"])
    out["time_s"] = out["time_s"].map(lambda x: f"{x:.{p.time}f}")
    out["duration_s"] = out["duration_s"].map(lambda x: f"{x:.{p.duration}f}")
    out["value"] = out["value"].map(lambda v: f"{v:.{p.value}f}")
    return header + "\n" + out.to_csv(sep="\t", header=False, index=False)


def _csv_export(
    name: str,
    trace: pd.DataFrame,
    paraver: ParaverData,
    values: pd.Series[Any],
    window_mode: str,
    vmin: float | int,
    vmax: float | int,
) -> ExportFile:
    """One ExportFile whose content is the full Paraver CSV for *values*.

    *values* is ``pd.Series[Any]`` because it is either float (perf, ai,
    proximity ratios) or int (roof/region codes).
    """
    return ExportFile(name=name, content=_csv_content(trace, paraver, values, window_mode, vmin, vmax))


# Roof math helpers


def _roof_peak_gflops(model: AssembledRoofline, divisor: int) -> float:
    """max over peak_performance_by_op values / divisor / 1e9."""
    return max(p.value for p in model.peak_performance_by_op.values()) / divisor / 1e9


def roof_value_gflops(ai: pd.Series[float], level: str, model: AssembledRoofline, divisor: int) -> pd.Series[float]:
    """min(bw_level.value / divisor * ai, peak_gflops), all in GFLOPS."""
    bw = model.bandwidth_by_level[level]
    return (bw.value / divisor * ai / 1e9).clip(upper=_roof_peak_gflops(model, divisor))


def roof_label_codes(
    ai: pd.Series[float], perf: pd.Series[float], model: AssembledRoofline, divisor: int
) -> pd.Series[Any]:
    """Code per row (int64):
    - 0 when ai <= 0 or perf <= 0;
    - else walk ("DRAM", "L3", "L2", "L1") in that order, skipping levels absent from
      model.bandwidth_by_level; first level with perf*1e-9 < roof_value_gflops(ai, level)
      wins (codes from ROOF_LEVEL_CODES); rows below no roof → 6.
    """
    codes = np.full(len(ai), 6, dtype=np.int64)
    codes[(ai.to_numpy() <= 0) | (perf.to_numpy() <= 0)] = 0
    perf_gflops = (perf * 1e-9).to_numpy()
    for level in ("DRAM", "L3", "L2", "L1"):
        if level not in model.bandwidth_by_level:
            continue
        below = perf_gflops < roof_value_gflops(ai, level, model, divisor).to_numpy()
        codes[below & (codes == 6)] = ROOF_LEVEL_CODES[level]
    return pd.Series(codes, index=ai.index)


def region_label_codes(ai: pd.Series[float], l1_ridge_x: float, dram_ridge_x: float) -> pd.Series[Any]:
    """1 when ai < l1_ridge_x; 2 when l1_ridge_x <= ai <= dram_ridge_x; 3 when ai > dram_ridge_x.
    No positivity guard.
    """
    return pd.Series(np.where(ai < l1_ridge_x, 1, np.where(ai <= dram_ridge_x, 2, 3)), index=ai.index)


def proximity_ratio(
    ai: pd.Series[float], perf: pd.Series[float], level: str, model: AssembledRoofline, divisor: int
) -> pd.Series[Any]:
    """np.where((ai > 0) & (perf > 0) & (roof > 0), np.minimum((perf*1e-9) / roof, 1.0), 0.0)."""
    roof = roof_value_gflops(ai, level, model, divisor)
    return pd.Series(
        np.where((ai > 0) & (perf > 0) & (roof > 0), np.minimum((perf * 1e-9) / roof, 1.0), 0.0),
        index=ai.index,
    )


# Per-mode facades — uniform signature and return, so the factory callback is one
# generic body. Each returns tuple[ExportFile, ...] (the mode's complete file set);
# () when the mode cannot export (no trace rows, or roof data missing).

#: Common signature of the per-mode exporters, as consumed by the GUI factory.
#: ``Optional`` because this alias is evaluated at import time on Python 3.9.
ExportModeExporter = Callable[[pd.DataFrame, ParaverData, Optional[AssembledRoofline], int], tuple[ExportFile, ...]]


def export_performance(
    trace: pd.DataFrame, paraver: ParaverData, model: AssembledRoofline | None = None, divisor: int = 1
) -> tuple[ExportFile, ...]:
    """GFLOPS per row (perf in FLOP/s / 1e9), GRADIENT window, vmin:vmax from the values."""
    if trace.empty:
        return ()
    values = trace_metric(trace, "perf") * 1e-9
    return (
        _csv_export(
            PERFORMANCE_FILENAME, trace, paraver, values, GRADIENT_WINDOW_MODE, float(values.min()), float(values.max())
        ),
    )


def export_ai(
    trace: pd.DataFrame, paraver: ParaverData, model: AssembledRoofline | None = None, divisor: int = 1
) -> tuple[ExportFile, ...]:
    """Arithmetic intensity per row, GRADIENT window, vmin:vmax from the values."""
    if trace.empty:
        return ()
    values = trace_metric(trace, "ai")
    return (
        _csv_export(
            AI_FILENAME, trace, paraver, values, GRADIENT_WINDOW_MODE, float(values.min()), float(values.max())
        ),
    )


def export_roof_labels(
    trace: pd.DataFrame, paraver: ParaverData, model: AssembledRoofline | None, divisor: int
) -> tuple[ExportFile, ...]:
    """Roof-label codes (CODE window, 1:6) plus the six-row legend."""
    if trace.empty:
        return ()
    if model is None or not model.bandwidth_by_level or not model.peak_performance_by_op:
        return ()
    codes = roof_label_codes(trace_metric(trace, "ai"), trace_metric(trace, "perf"), model, divisor)
    return (
        _csv_export(ROOF_LABELS_FILENAME, trace, paraver, codes, CODE_WINDOW_MODE, 1, 6),
        ExportFile(name=ROOF_LABELS_LEGEND_FILENAME, content=serialize_legend(ROOF_LABEL_LEGEND)),
    )


def export_region(
    trace: pd.DataFrame, paraver: ParaverData, model: AssembledRoofline | None, divisor: int
) -> tuple[ExportFile, ...]:
    """Roofline-region codes (CODE window, 1:3) plus the three-row legend.
    Ridge points are divisor-independent since ridge_x = peak/bw.
    """
    if trace.empty:
        return ()
    if model is None:
        return ()
    ridge = model.ridge_points()
    if "L1" not in ridge or "DRAM" not in ridge:
        return ()
    codes = region_label_codes(trace_metric(trace, "ai"), float(ridge["L1"].value), float(ridge["DRAM"].value))
    return (
        _csv_export(REGION_FILENAME, trace, paraver, codes, CODE_WINDOW_MODE, 1, 3),
        ExportFile(name=REGION_LEGEND_FILENAME, content=serialize_legend(REGION_LEGEND)),
    )


def export_proximity(
    trace: pd.DataFrame, paraver: ParaverData, model: AssembledRoofline | None, divisor: int
) -> tuple[ExportFile, ...]:
    """One GRADIENT CSV per roof level present in the model (absent levels silently
    skipped); ratio clamped to 1.0, 0.0 on non-positive operands, vmin:vmax 0.0:1.0.
    """
    if trace.empty:
        return ()
    if model is None or not model.bandwidth_by_level:
        return ()
    ai = trace_metric(trace, "ai")
    perf = trace_metric(trace, "perf")
    files: list[ExportFile] = []
    for level in ("L1", "L2", "L3", "DRAM"):
        if level not in model.bandwidth_by_level:
            continue
        values = proximity_ratio(ai, perf, level, model, divisor)
        files.append(
            _csv_export(
                f"carm_rel_{PROXIMITY_SUFFIXES[level]}.csv",
                trace,
                paraver,
                values,
                GRADIENT_WINDOW_MODE,
                0.0,
                1.0,
            )
        )
    return tuple(files)
