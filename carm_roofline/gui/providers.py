"""Application-point sources for the roofline plot.

CARM mode sources points from benchmarked applications (``BenchmarkAppsProvider``,
records keyed by id); paraver mode loads an external Paraver trace into a
:class:`ParaverData` trace table via the paramedir counter pipeline.
"""

from __future__ import annotations

import math
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd

from carm_roofline.core.error import UserError
from carm_roofline.core.units import Bandwidth, Bytes, Operations, Performance, Seconds
from carm_roofline.paraver import (
    DEFAULT_CSV_PRECISION,
    CsvPrecision,
    MetricColumn,
    ParaverWindowMode,
    ProgressBar,
    TraceRow,
    build_trace_table,
    default_legend_path,
    load_legend_csv,
    load_window_csv,
    parse_paraver_header,
    run_paramedir,
    trace_metric,
    trace_state_code,
    trace_text,
    window_csv_precision,
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
    for rows whose state_code no legend range covers. Every row carries
    ``_tooltip`` (str): hover HTML formatted once at load so figure builds slice
    the column instead of rebuilding it per row. ``window_mode`` is a
    :class:`ParaverWindowMode` member (``CODE`` | ``GRADIENT``). ``label`` is
    the short display name: the window name with its app suffix stripped
    (gradient mode, where the legend entry names the counter) or the trace stem
    (code mode, where the legend already names the states). ``precision``
    carries the decimal places detected from the window CSV; every export
    formats its numeric cells with them.
    """

    trace: pd.DataFrame
    label: str
    window_mode: ParaverWindowMode  # CODE | GRADIENT
    time_unit: str
    prv_path: str
    legend: pd.DataFrame | None  # code/code_end/label/r/g/b; None in gradient mode
    precision: CsvPrecision = DEFAULT_CSV_PRECISION


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

            # Progress popup: open at 0% before the paramedir run; build_trace_table then updates and closes it at 100%.
            progress = ProgressBar(total=1)
            progress.update(0)

            run_paramedir(self._trace_path, work_dir, header.time_unit)

            trace = build_trace_table(self._window_csv_path, work_dir, header.time_unit, progress=progress)
        except (ValueError, RuntimeError) as exc:
            # Pipeline failures (no counter CSVs, paramedir non-zero exit, empty
            # legend merge — MergeError is a ValueError) surface as UserError so
            # the GUI degrades gracefully instead of crashing startup.
            shutil.rmtree(work_dir, ignore_errors=True)
            raise UserError(str(exc)) from exc
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
            # Bursts whose start falls outside the window's state timeline carry
            # NaN state_code (a normal case, not corrupted input); merge_asof
            # rejects null keys, so merge only the coded rows and leave the rest
            # NaN in the assignments below.
            left = trace.assign(_code=codes).loc[codes.notna()].sort_values("_code")
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
        # Precompute per-row tooltip HTML once here: figure callbacks slice the
        # column instead of rebuilding ~10 f-strings per row on every slider tick.
        trace["_tooltip"] = paraver_tooltips(trace, label)
        return ParaverData(
            trace=trace,
            label=label,
            window_mode=window_mode,
            time_unit=header.time_unit,
            prv_path=header.prv_path,
            legend=legend,
            precision=window_csv_precision(self._window_csv_path),
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


# The GUI slider positions are log10(ai threshold in OPS/Byte); components.py
# derives its slider geometry from these constants.
AI_FILTER_OFF_AI = 1e-6  # thresholds at or below this disable filtering (slider leftmost)
AI_FILTER_DEFAULT_AI = 1e-5  # default filter threshold (slider default position)
AI_FILTER_MAX_AI = 1e-2  # slider rightmost position
AI_FILTER_LOG_STEP = 0.2  # slider step, in log10 decades


def _filter_trace_by_threshold(
    trace: pd.DataFrame, metric: MetricColumn, threshold: float | None, off_boundary: float
) -> pd.DataFrame:
    """Keep rows with *metric* >= *threshold*; None or a threshold <= *off_boundary*
    disables filtering and returns *trace* unchanged (same object)."""
    if threshold is None or threshold <= off_boundary:
        return trace
    return trace[trace_metric(trace, metric) >= threshold]


def filter_trace_by_ai(trace: pd.DataFrame, ai_threshold: float | None) -> pd.DataFrame:
    """Keep rows with ai >= *ai_threshold*; None or a threshold <= AI_FILTER_OFF_AI
    disables filtering and returns *trace* unchanged (same object)."""
    return _filter_trace_by_threshold(trace, "ai", ai_threshold, AI_FILTER_OFF_AI)


# Duration-filter slider (mirrors the AI filter above): slider positions are
# log10(minimum duration in seconds). Off at the leftmost position.
DURATION_FILTER_OFF_S = 1e-6  # 1 us; thresholds at or below this disable filtering (slider leftmost)
DURATION_FILTER_DEFAULT_S = 1e-4  # 100 us; default filter threshold (slider default position)
DURATION_FILTER_MAX_S = 1e-1  # 100 ms; slider rightmost position
DURATION_FILTER_LOG_STEP = 0.2  # slider step, in log10 decades


def filter_trace_by_duration(trace: pd.DataFrame, min_duration_s: float | None) -> pd.DataFrame:
    """Keep rows with duration_s >= *min_duration_s*; None or a threshold <=
    DURATION_FILTER_OFF_S disables filtering and returns *trace* unchanged (same object)."""
    return _filter_trace_by_threshold(trace, "duration_s", min_duration_s, DURATION_FILTER_OFF_S)


def filter_trace(
    trace: pd.DataFrame,
    window: tuple[float, float] | None,
    ai_threshold: float | None,
    duration_threshold: float | None,
) -> pd.DataFrame:
    """Filter *trace* by time window, ai and duration through a single boolean mask.

    Applies the same boundary rules as the individual helpers: the window is
    inclusive [lo, hi]; ai/duration keep rows with metric >= threshold; a threshold
    of None or <= its OFF constant disables that term; a None window disables the
    time term. The frame is indexed exactly once; when no term is active, *trace*
    is returned unchanged (same object).
    """
    terms: list[pd.Series[bool]] = []
    if window is not None:
        lo, hi = window
        time_s = trace_metric(trace, "time_s")
        terms.append((time_s >= lo) & (time_s <= hi))
    if ai_threshold is not None and ai_threshold > AI_FILTER_OFF_AI:
        terms.append(trace_metric(trace, "ai") >= ai_threshold)
    if duration_threshold is not None and duration_threshold > DURATION_FILTER_OFF_S:
        terms.append(trace_metric(trace, "duration_s") >= duration_threshold)
    if not terms:
        return trace
    mask = terms[0]
    for term in terms[1:]:
        mask = mask & term
    return trace[mask]


def trace_time_range(trace: pd.DataFrame) -> tuple[float, float] | None:
    """Full timestamp extent of the trace, or None when it is empty."""
    if trace.empty:
        return None
    time_s = trace_metric(trace, "time_s")
    return (float(time_s.min()), float(time_s.max()))


# Per-row tooltip formatting. These run once at load (``paraver_tooltips``) so
# per-callback figure builds only slice the precomputed ``_tooltip`` column
# instead of rebuilding ~10 f-strings per row for every slider/filter change.


def _load_store_pct_line(load_share: float) -> str:
    """One tooltip line with load/store percentages derived from the load_share fraction."""
    if math.isnan(load_share):
        return "  Loads: - | Stores: -"
    loads_pct = 100.0 * load_share
    return f"  Loads: {loads_pct:.1f}% | Stores: {100.0 - loads_pct:.1f}%"


def _isa_pct_line(row: TraceRow) -> str:
    """Per-ISA operation-share line: only ISAs above 0.1% (legacy display filter),
    rounded to 1 dp; a '-' placeholder when no ISA qualifies (no FP work in the
    burst)."""
    entries = [
        f"{label} {round(pct, 1):.1f}%"
        for label, pct in (
            ("Scalar", row.isa_scalar_pct),
            ("SSE", row.isa_sse_pct),
            ("AVX2", row.isa_avx2_pct),
            ("AVX512", row.isa_avx512_pct),
        )
        if pct > 0.1
    ]
    return f"  ISA: {' | '.join(entries)}" if entries else "  ISA: -"


def _format_paraver_tooltip(label: str, row: TraceRow, state_label: str | None) -> str:
    """Rich HTML tooltip for a trace-table row; row is one TraceRow from ``itertuples()``."""
    dur = float(row.duration_s)
    parts = [f"<b>{label}</b>", f"<i>{row.thread_id}</i>"]
    value = float(row.state_code)
    if not math.isnan(value):
        shown = f"{value:g}" if value.is_integer() else f"{value}"
        if state_label is not None:
            shown = f"{shown} ({state_label})"
        parts += ["<b>Paraver Value</b>", f"  {shown}"]
    parts += [
        "<b>Performance</b>",
        f"  Arithmetic Intensity: {float(row.ai):.3f} OPS/Byte",
        f"  Performance: {Performance(float(row.perf))!s}",
        f"  Bandwidth: {Bandwidth(float(row.bytes) / dur if dur > 0 else 0.0)!s}",
        "<b>Execution</b>",
        f"  Duration: {Seconds(dur)!s}",
        "<b>Work</b>",
        f"  Total FLOPs: {Operations(int(row.flops))!s}",
        f"  Total Bytes: {Bytes(int(row.bytes))!s}",
        _load_store_pct_line(float(row.load_share)),
        _isa_pct_line(row),
    ]
    return "<br>".join(parts)


def paraver_tooltips(trace: pd.DataFrame, label: str) -> list[str]:
    """Per-row tooltip HTML for a whole trace, formatted once instead of per figure build.

    state_label mirrors the per-call behavior: the row's own ``legend_label``
    when the column exists and the value is a str (code mode), else None
    (gradient mode, or rows no legend range covers).
    """
    if "legend_label" in trace.columns:
        state_labels: list[str | None] = [
            lbl if isinstance(lbl, str) else None for lbl in trace_text(trace, "legend_label")
        ]
    else:
        state_labels = [None] * len(trace)
    rows = cast(Iterable[TraceRow], trace.itertuples())
    return [_format_paraver_tooltip(label, row, state_label) for row, state_label in zip(rows, state_labels)]
