"""Application-point sources for the roofline plot.

CARM mode sources points from benchmarked applications (``BenchmarkAppsProvider``,
records keyed by id); paraver mode loads an external Paraver trace into a
:class:`ParaverData` trace table via the paramedir counter pipeline.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from carm_roofline.core.error import UserError
from carm_roofline.paraver import (
    ParaverWindowMode,
    build_trace_table,
    default_legend_path,
    load_legend_csv,
    load_window_csv,
    parse_paraver_header,
    run_paramedir,
    trace_metric,
    trace_state_code,
)
from carm_roofline.roofline_assembly import ApplicationRecord, load_all_applications


class BenchmarkAppsProvider:
    """Application points from CARM-benchmarked applications (the current source)."""

    def __init__(self, results_dir: Path) -> None:
        self._results_dir = results_dir

    def load(self) -> dict[str, ApplicationRecord]:
        applications = load_all_applications(self._results_dir)
        return {a.id: a for a in applications}


@dataclass(frozen=True)
class ParaverData:
    """Loaded paraver trace ready for direct plotting.

    ``trace`` carries TRACE_COLUMNS; in code mode it additionally carries
    ``legend_label`` (str) and ``legend_color`` (str "rgb(r,g,b)") per row, NaN
    for rows whose state_code no legend range covers. ``window_mode`` is a
    :class:`ParaverWindowMode` member (``CODE`` | ``GRADIENT``). ``label`` is
    the short display name: the window name with its app suffix stripped
    (gradient mode, where the legend entry names the counter) or the trace stem
    (code mode, where the legend already names the states).
    """

    trace: pd.DataFrame
    label: str
    window_mode: ParaverWindowMode  # CODE | GRADIENT
    time_unit: str
    prv_path: str
    legend: pd.DataFrame | None  # code/code_end/label/r/g/b; None in gradient mode


class ParaverProvider:
    """Paraver trace table via the paramedir pipeline.

    The provider runs ``paramedir`` over the ``.prv`` trace, loads counter CSVs,
    builds a trace table, and (in code mode) maps each state code to its legend
    entry. Returns a :class:`ParaverData` ready for direct plotting.
    """

    def __init__(
        self,
        trace_path: Path,
        window_csv_path: Path,
        legend_csv_path: Path | None = None,
    ) -> None:
        self._trace_path = trace_path.resolve()
        self._window_csv_path = window_csv_path.resolve()
        self._legend_csv_path = legend_csv_path.resolve() if legend_csv_path else None
        self._window_extent: tuple[float, float] | None = None

    @property
    def window_extent(self) -> tuple[float, float] | None:
        """Loaded window CSV interval (min start, max end) in seconds, or None.

        Set by :meth:`load`; used to initialize the semantic-window startup filter.
        """
        return self._window_extent

    def load(self) -> ParaverData:
        # Parse the window CSV header to learn the mode, time unit, and prv path.
        with open(self._window_csv_path, encoding="utf-8") as fh:
            header = parse_paraver_header(fh.readline().strip())
        window_mode = ParaverWindowMode.from_header(header.window_mode)
        is_gradient = window_mode == ParaverWindowMode.GRADIENT

        # Fail fast on a missing legend (code mode) before the paramedir run.
        legend = None
        if not is_gradient:
            legend_path = self._legend_csv_path or default_legend_path(self._window_csv_path)
            if not legend_path.is_file():
                raise UserError(f"paraver legend CSV not found: {legend_path}")
            legend = load_legend_csv(legend_path)

        # Create a temporary working directory for paramedir counter outputs.
        work_dir = Path(tempfile.mkdtemp(prefix="carm-paraver-"))
        try:
            if not shutil.which("paramedir"):
                raise UserError("paramedir not found on PATH; install it to load Paraver traces")

            run_paramedir(self._trace_path, work_dir, header.time_unit)

            trace = build_trace_table(self._window_csv_path, work_dir, header.time_unit)
        except Exception:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise

        if trace.empty:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise UserError(
                f"trace table is empty after processing {self._trace_path} with window {self._window_csv_path}"
            )

        # Loaded window extent (min start, max end) in seconds for the semantic window.
        window_frame = load_window_csv(self._window_csv_path)
        if not window_frame.empty:
            starts = window_frame["time_s"]
            ends = starts + window_frame["duration_s"]
            self._window_extent = (float(starts.min()), float(ends.max()))

        if legend is not None:
            # Old-tool semantics: backward merge_asof on the legend's lower bound,
            # then inclusive [code, code_end] containment. Unmatched rows stay in the
            # trace with NaN legend_label/legend_color (never plotted, but they keep
            # the slider bounds covering the whole trace).
            codes = trace_state_code(trace).astype(float)
            left = trace.assign(_code=codes).sort_values("_code")
            matched = pd.merge_asof(
                left,
                legend,
                left_on="_code",
                right_on="code",
                direction="backward",
            )
            in_range = (matched["_code"] >= matched["code"]) & (matched["_code"] <= matched["code_end"])
            matched = matched.loc[in_range]
            # merge_asof returns a fresh RangeIndex; restore the original row
            # identity so the assignments below align by index, not position.
            matched.index = left.index[in_range.to_numpy()]
            trace = trace.copy()
            trace["legend_label"] = matched["label"]
            trace["legend_color"] = (
                "rgb("
                + matched["r"].astype(str)
                + ","
                + matched["g"].astype(str)
                + ","
                + matched["b"].astype(str)
                + ")"
            )

        trace_stem = self._trace_path.stem
        if window_mode == ParaverWindowMode.GRADIENT:
            # The legend entry names the gradient window's counter; the app is known.
            label = _window_display_name(self._window_csv_path.name, trace_stem)
        else:
            # Legend entries are the state names; the label names the trace in tooltips.
            label = trace_stem
        return ParaverData(
            trace=trace,
            label=label,
            window_mode=window_mode,
            time_unit=header.time_unit,
            prv_path=header.prv_path,
            legend=legend,
        )


def _window_display_name(window_csv_name: str, trace_stem: str) -> str:
    """Short window name with wxparaver's app suffix stripped.

    Exported window CSVs are named ``<window name>_<trace stem>.csv``; dropping
    the suffix keeps the display name free of the app name the user already
    knows. Falls back to the bare CSV stem when the suffix is absent.
    """
    suffix = f"_{trace_stem}.csv"
    if window_csv_name.endswith(suffix):
        return window_csv_name[: -len(suffix)]
    return window_csv_name.removesuffix(".csv")


def filter_trace_by_window(trace: pd.DataFrame, window: tuple[float, float] | None) -> pd.DataFrame:
    """Keep rows with time_s inside *window* (inclusive); None returns *trace* unchanged."""
    if window is None:
        return trace
    lo, hi = window
    time_s = trace_metric(trace, "time_s")
    return trace[(time_s >= lo) & (time_s <= hi)]


def trace_time_range(trace: pd.DataFrame) -> tuple[float, float] | None:
    """Full timestamp extent of the trace, or None when it is empty."""
    if trace.empty:
        return None
    time_s = trace_metric(trace, "time_s")
    return (float(time_s.min()), float(time_s.max()))
