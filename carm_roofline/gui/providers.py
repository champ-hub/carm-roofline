"""Application-point sources for the roofline plot.

Each provider implements ``load() -> dict[str, ApplicationRecord]`` (records keyed
by id). CARM mode sources points from benchmarked applications; paraver mode sources
them from an external trace whose format is an external dependency not yet defined.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from carm_roofline.roofline_assembly import ApplicationRecord, load_all_applications


class BenchmarkAppsProvider:
    """Application points from CARM-benchmarked applications (the current source)."""

    def __init__(self, results_dir: Path) -> None:
        self._results_dir = results_dir

    def load(self) -> dict[str, ApplicationRecord]:
        applications = load_all_applications(self._results_dir)
        return {a.id: a for a in applications}


class ParaverProvider:
    """Application points from an external paraver trace.

    The .prv/.pcf trace format is an external dependency that has not been defined
    yet, so loading is not implemented. This class is the single plug point where
    the trace parser will land; nothing else in the GUI may depend on the format.
    """

    def __init__(self, trace_path: Path) -> None:
        self._trace_path = trace_path

    def load(self) -> dict[str, ApplicationRecord]:
        raise NotImplementedError(f"paraver trace format not defined yet; cannot load {self._trace_path}")


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
