"""Unit tests for application-point providers and window filtering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carm_roofline.gui.providers import (
    BenchmarkAppsProvider,
    ParaverProvider,
    filter_points_by_window,
    trace_time_range,
)
from carm_roofline.roofline_assembly import ApplicationPoint, ApplicationRecord

pytestmark = pytest.mark.unit


def _point(label: str, time_s: float | None = None) -> ApplicationPoint:
    return ApplicationPoint(
        label=label,
        total_flops=1e9,
        total_bytes=1e6,
        runtime_s=1.0,
        num_ranks=1,
        num_threads=1,
        num_regions=1,
        arithmetic_intensity=1000.0,
        flops_per_second=1e9,
        bandwidth=1e6,
        time_s=time_s,
    )


def _record(rec_id: str, points: list[ApplicationPoint]) -> ApplicationRecord:
    return ApplicationRecord(
        id=rec_id,
        label=f"{rec_id} — 2026-01-01 (avg)",
        aggregation="avg",
        metadata={"name": rec_id, "date": "2026-01-01", "command": "run"},
        points=points,
        machine="machine-a",
    )


def test_filter_points_by_window_keeps_only_points_in_window() -> None:
    """Points at t=0/5/10 with window (2, 8) keep only the t=5 point."""
    rec = _record("r1", [_point("p0", 0.0), _point("p5", 5.0), _point("p10", 10.0)])
    result = filter_points_by_window({"r1": rec}, (2.0, 8.0))
    assert list(result["r1"].points) == [_point("p5", 5.0)]


def test_filter_points_by_window_drops_empty_records() -> None:
    """A record with no surviving points is dropped from the dict."""
    rec_outside = _record("r1", [_point("p0", 0.0)])
    rec_inside = _record("r2", [_point("p5", 5.0)])
    result = filter_points_by_window({"r1": rec_outside, "r2": rec_inside}, (2.0, 8.0))
    assert list(result.keys()) == ["r2"]


def test_filter_points_by_window_none_returns_same_dict() -> None:
    """A None window returns the input dict unchanged (same object)."""
    rec = _record("r1", [_point("p0", 0.0)])
    app_by_id = {"r1": rec}
    assert filter_points_by_window(app_by_id, None) is app_by_id


def test_filter_points_by_window_excludes_untimestamped_points() -> None:
    """Points with time_s=None are excluded whenever a window is applied."""
    rec = _record("r1", [_point("p_no_ts"), _point("p5", 5.0)])
    result = filter_points_by_window({"r1": rec}, (2.0, 8.0))
    assert [p.label for p in result["r1"].points] == ["p5"]


def test_filter_points_by_window_excluding_everything_returns_empty() -> None:
    """A window covering no point yields an empty dict."""
    rec = _record("r1", [_point("p0", 0.0)])
    assert filter_points_by_window({"r1": rec}, (2.0, 8.0)) == {}


def test_trace_time_range_across_records() -> None:
    """trace_time_range spans the min and max timestamps across records."""
    app_by_id = {
        "r1": _record("r1", [_point("p0", 0.0), _point("p5", 5.0)]),
        "r2": _record("r2", [_point("p10", 10.0)]),
    }
    assert trace_time_range(app_by_id) == (0.0, 10.0)


def test_trace_time_range_none_without_timestamps() -> None:
    """trace_time_range returns None when no point has a timestamp."""
    app_by_id = {"r1": _record("r1", [_point("p_no_ts")])}
    assert trace_time_range(app_by_id) is None


def test_paraver_provider_load_not_implemented() -> None:
    """ParaverProvider.load raises NotImplementedError until the trace format is defined."""
    with pytest.raises(NotImplementedError, match="not defined yet"):
        ParaverProvider(Path("trace.prv")).load()


def test_benchmark_apps_provider_loads_applications_jsonl(tmp_path: Path) -> None:
    """BenchmarkAppsProvider loads records keyed by id from applications.jsonl."""
    machine_dir = tmp_path / "machine-a"
    machine_dir.mkdir()
    (machine_dir / "applications.jsonl").write_text(
        json.dumps(
            {
                "format_version": "2.0",
                "metadata": {"name": "app", "date": "2026-01-01", "command": "run"},
                "aggregation": "avg",
                "points": [
                    {
                        "label": "p",
                        "total_flops": 1,
                        "total_bytes": 1,
                        "runtime_s": 1,
                        "num_ranks": 1,
                        "num_threads": 1,
                        "num_regions": 1,
                        "arithmetic_intensity": 1,
                        "flops_per_second": 1,
                        "bandwidth": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    app_by_id = BenchmarkAppsProvider(tmp_path).load()
    assert len(app_by_id) == 1
    rec = next(iter(app_by_id.values()))
    assert rec.id.startswith("app_")
    assert rec.machine == "machine-a"
