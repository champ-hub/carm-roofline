"""Unit tests for application-point providers and window filtering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carm_roofline.core.error import UserError
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


def test_paraver_provider_converts_trace_rows_to_application_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ParaverProvider converts trace table rows to ApplicationPoints grouped in one ApplicationRecord."""
    # Create a window CSV with header so parse_paraver_header works and
    # build_trace_table can attach state codes.
    window_csv = tmp_path / "window.csv"
    window_csv.write_text(
        "#20260803:CSV:RUNAPP:/p/t.prv:nanoseconds:window_in_code_mode\n1.1.1\t0.0\t1000000000.0\t1.0\n"
    )

    # Dummy .prv file — only needs to exist on disk for the provider check.
    trace = tmp_path / "t.prv"
    trace.write_text("#Paraver dummy\n")

    # Capture the args run_paramedir receives so we can assert the provider
    # forwards the trace path and the window header's time unit.
    captured: list[tuple[str | Path, str | Path, str]] = []

    def _fake_run_paramedir(trace_path: str | Path, output_dir: str | Path, time_unit: str) -> None:
        captured.append((trace_path, output_dir, time_unit))
        out = Path(output_dir)
        # One FP counter row: thread 1.1.1, 0-1 s, value=4 instructions
        (out / "fp-avx2-dp.csv").write_text(
            f"#ts:CSV:RUNAPP:/p/t.prv:{time_unit}:window_in_code_mode\n1.1.1\t0.0\t1000000000.0\t4\n",
            encoding="utf-8",
        )
        # One memory counter row: same burst, value=2 loads
        (out / "mem-loads.csv").write_text(
            f"#ts:CSV:RUNAPP:/p/t.prv:{time_unit}:window_in_code_mode\n1.1.1\t0.0\t1000000000.0\t2\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "carm_roofline.gui.providers.run_paramedir",
        _fake_run_paramedir,
    )
    # Also skip the which("paramedir") check in the provider.
    monkeypatch.setattr("shutil.which", lambda _x: "/usr/bin/paramedir")

    provider = ParaverProvider(trace, window_csv)
    app_by_id = provider.load()

    # The trace path (resolved) and the window header's time unit reach run_paramedir.
    assert len(captured) == 1
    called_trace, _out_dir, called_unit = captured[0]
    assert Path(called_trace) == trace.resolve()
    assert called_unit == "nanoseconds"

    # The loaded window CSV interval (min start, max end) is exposed in seconds.
    assert provider.window_extent == (0.0, 1.0)

    assert len(app_by_id) == 1
    rec_id, rec = next(iter(app_by_id.items()))
    assert rec_id == rec.id
    assert len(rec_id) == 16  # sha256 hex prefix

    # Label and machine derive from file stems.
    assert rec.machine == "t"
    assert "t" in rec.label
    assert "window.csv" in rec.label

    # Aggregation is "paraver" (not benchmark avg/min/max).
    assert rec.aggregation == "paraver"

    # Metadata carries header details.
    assert rec.metadata["prv_path"] == str(trace.resolve())
    assert rec.metadata["window_csv"] == str(window_csv.resolve())
    assert rec.metadata["time_unit"] == "nanoseconds"
    assert rec.metadata["window_mode"] == "window_in_code_mode"

    # One point per burst row.
    assert len(rec.points) == 1
    p = rec.points[0]

    # Label is the Paraver thread_id.
    assert p.label == "1.1.1"

    # Timestamp and runtime (1e9 ns → 1.0 s).
    assert p.time_s == 0.0
    assert p.runtime_s == pytest.approx(1.0)

    # num_threads/num_ranks are 1 per-row because the table is per-thread.
    assert p.num_threads == 1
    assert p.num_ranks == 1
    assert p.num_regions == 1

    # Computed metrics (exact values depend on counter weights, but invariants hold).
    assert p.total_flops > 0
    assert p.total_bytes > 0
    assert p.arithmetic_intensity == pytest.approx(p.total_flops / p.total_bytes)
    assert p.flops_per_second == pytest.approx(p.total_flops / p.runtime_s)
    assert p.bandwidth == pytest.approx(p.total_bytes / p.runtime_s)


def test_paraver_provider_missing_paramedir_raises_user_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ParaverProvider raises UserError when paramedir is not on PATH."""
    window_csv = tmp_path / "window.csv"
    window_csv.write_text(
        "#20260803:CSV:RUNAPP:/p/t.prv:microseconds:window_in_code_mode\n1.1.1\t0.0\t1000000.0\t1.0\n"
    )
    trace = tmp_path / "t.prv"
    trace.write_text("#dummy\n")

    monkeypatch.setattr("shutil.which", lambda _x: None)

    with pytest.raises(UserError, match="paramedir not found"):
        ParaverProvider(trace, window_csv).load()


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
