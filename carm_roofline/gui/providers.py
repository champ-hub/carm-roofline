"""Application-point sources for the roofline plot.

Each provider implements ``load() -> dict[str, ApplicationRecord]`` (records keyed
by id). CARM mode sources points from benchmarked applications; paraver mode sources
them from an external Paraver trace via the paramedir counter pipeline.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from carm_roofline.core.error import UserError
from carm_roofline.paraver import (
    build_trace_table,
    load_legend_csv,
    load_window_csv,
    parse_paraver_header,
    run_paramedir,
)
from carm_roofline.roofline_assembly import ApplicationPoint, ApplicationRecord, load_all_applications


class BenchmarkAppsProvider:
    """Application points from CARM-benchmarked applications (the current source)."""

    def __init__(self, results_dir: Path) -> None:
        self._results_dir = results_dir

    def load(self) -> dict[str, ApplicationRecord]:
        applications = load_all_applications(self._results_dir)
        return {a.id: a for a in applications}


class ParaverProvider:
    """Application points from an external Paraver trace via the paramedir pipeline.

    The provider runs ``paramedir`` over the ``.prv`` trace, loads counter CSVs,
    builds a trace table, and converts each row into an :class:`ApplicationPoint`
    grouped in one :class:`ApplicationRecord`.
    """

    def __init__(
        self,
        trace_path: Path,
        window_csv_path: Path,
        legend_csv_path: Path | None = None,
        use_colors: bool = False,
    ) -> None:
        self._trace_path = trace_path.resolve()
        self._window_csv_path = window_csv_path.resolve()
        self._legend_csv_path = legend_csv_path.resolve() if legend_csv_path else None
        self._use_colors = use_colors
        self._window_extent: tuple[float, float] | None = None

    @property
    def window_extent(self) -> tuple[float, float] | None:
        """Loaded window CSV interval (min start, max end) in seconds, or None.

        Set by :meth:`load`; used to initialize the semantic-window startup filter.
        """
        return self._window_extent

    def load(self) -> dict[str, ApplicationRecord]:
        # Parse the window CSV header to get the time unit for paramedir.
        with open(self._window_csv_path, encoding="utf-8") as fh:
            header = parse_paraver_header(fh.readline().strip())

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

        # Convert trace rows to ApplicationPoint objects.
        points: list[ApplicationPoint] = []
        for _, row in trace.iterrows():
            dur: float = float(row["duration_s"])
            flops: float = float(row["flops"])
            byt: float = float(row["bytes"])
            bandwidth = byt / dur if dur > 0 else 0.0
            points.append(
                ApplicationPoint(
                    label=str(row["thread_id"]),
                    total_flops=flops,
                    total_bytes=byt,
                    runtime_s=dur,
                    num_ranks=1,
                    num_threads=1,
                    num_regions=1,
                    arithmetic_intensity=float(row["ai"]),
                    flops_per_second=float(row["perf"]),
                    bandwidth=bandwidth,
                    time_s=float(row["time_s"]),
                )
            )

        # Stable record id derived from trace and window paths (not Python hash).
        record_id = hashlib.sha256(f"{self._trace_path}:{self._window_csv_path}".encode()).hexdigest()[:16]
        trace_stem = self._trace_path.stem
        window_name = self._window_csv_path.name

        metadata: dict[str, Any] = {
            "prv_path": str(self._trace_path),
            "window_csv": str(self._window_csv_path),
            "time_unit": header.time_unit,
            "window_mode": header.window_mode,
        }

        # Load legend if colors are requested.
        legend = None
        if self._use_colors and self._legend_csv_path is not None:
            legend = load_legend_csv(self._legend_csv_path)
            metadata["legend"] = legend.to_dict(orient="records")

        record = ApplicationRecord(
            id=record_id,
            label=f"{trace_stem} — {window_name}",
            aggregation="paraver",
            metadata=metadata,
            points=points,
            machine=trace_stem,
        )

        return {record_id: record}


def filter_points_by_window(
    app_by_id: dict[str, ApplicationRecord],
    window: tuple[float, float] | None,
) -> dict[str, ApplicationRecord]:
    """Keep points with time_s inside *window* (inclusive); drop records left empty.

    A ``None`` window returns the input dict unchanged. Points without a timestamp
    are excluded whenever a window is applied.
    """
    if window is None:
        return app_by_id
    lo, hi = window
    filtered: dict[str, ApplicationRecord] = {}
    for rec_id, rec in app_by_id.items():
        kept = [p for p in rec.points if p.time_s is not None and lo <= p.time_s <= hi]
        if kept:
            filtered[rec_id] = replace(rec, points=kept)
    return filtered


def trace_time_range(app_by_id: dict[str, ApplicationRecord]) -> tuple[float, float] | None:
    """Full timestamp extent of the loaded points, or None when no point has a timestamp."""
    lo = min((p.time_s for rec in app_by_id.values() for p in rec.points if p.time_s is not None), default=None)
    hi = max((p.time_s for rec in app_by_id.values() for p in rec.points if p.time_s is not None), default=None)
    return (lo, hi) if lo is not None and hi is not None else None
