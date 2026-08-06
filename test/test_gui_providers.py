"""Unit tests for application-point providers and trace-table filtering."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from carm_roofline.core.error import UserError
from carm_roofline.gui.providers import (
    AI_FILTER_DEFAULT_AI,
    AI_FILTER_OFF_AI,
    BenchmarkAppsProvider,
    ParaverProvider,
    filter_trace_by_ai,
    filter_trace_by_window,
    trace_time_range,
)
from carm_roofline.paraver import ParaverWindowMode

pytestmark = pytest.mark.unit


def _trace_frame() -> pd.DataFrame:
    """Trace-shaped frame with timestamps t=0/5/10 and zero metrics."""
    return pd.DataFrame(
        {
            "thread_id": ["0", "0", "0"],
            "time_s": [0.0, 5.0, 10.0],
            "duration_s": [1.0, 1.0, 1.0],
            "state_code": [1.0, 1.0, 1.0],
            "flops": [0.0, 0.0, 0.0],
            "bytes": [0.0, 0.0, 0.0],
            "ai": [0.0, 0.0, 0.0],
            "perf": [0.0, 0.0, 0.0],
        }
    )


def test_filter_trace_by_window_keeps_only_rows_in_window() -> None:
    """Rows at t=0/5/10 with window (2, 8) keep only the t=5 row."""
    result = filter_trace_by_window(_trace_frame(), (2.0, 8.0))
    assert result["time_s"].tolist() == [5.0]


def test_filter_trace_by_window_none_returns_same_trace() -> None:
    """A None window returns the input frame unchanged (same object)."""
    trace = _trace_frame()
    assert filter_trace_by_window(trace, None) is trace


def test_filter_trace_by_window_empty_window_returns_empty() -> None:
    """A window covering no timestamp yields an empty frame."""
    result = filter_trace_by_window(_trace_frame(), (20.0, 30.0))
    assert result.empty


def test_filter_trace_by_ai_keeps_rows_above_threshold() -> None:
    """Threshold 1e-3 keeps only rows with ai >= 1e-3."""
    trace = _trace_frame()
    trace["ai"] = [1e-4, 1e-3, 1e-2]
    result = filter_trace_by_ai(trace, 1e-3)
    assert result["ai"].tolist() == [1e-3, 1e-2]


def test_filter_trace_by_ai_none_returns_same_trace() -> None:
    """A None threshold returns the input frame unchanged (same object)."""
    trace = _trace_frame()
    assert filter_trace_by_ai(trace, None) is trace


def test_filter_trace_by_ai_default_threshold_is_active() -> None:
    """Threshold == AI_FILTER_DEFAULT_AI (the 1e-5 default) is a real filter, not "off"."""
    trace = _trace_frame()
    trace["ai"] = [1e-6, 1e-5, 1e-4]
    result = filter_trace_by_ai(trace, AI_FILTER_DEFAULT_AI)
    assert result["ai"].tolist() == [1e-5, 1e-4]


def test_filter_trace_by_ai_between_off_and_default_is_active() -> None:
    """A threshold between the off boundary and the default (e.g. 5e-6) still filters.

    Guards against the off boundary drifting back up to the default: every slider
    position strictly right of the leftmost one must filter at its own threshold.
    """
    trace = _trace_frame()
    trace["ai"] = [1e-6, 5e-6, 1e-5]
    result = filter_trace_by_ai(trace, 5e-6)
    assert result["ai"].tolist() == [5e-6, 1e-5]


def test_filter_trace_by_ai_off_threshold_disables() -> None:
    """Threshold <= AI_FILTER_OFF_AI (slider leftmost, 1e-6) disables filtering.

    Pins "slider = 1e-6 => filtering off": the input frame is returned unchanged.
    """
    trace = _trace_frame()
    assert filter_trace_by_ai(trace, AI_FILTER_OFF_AI) is trace
    assert filter_trace_by_ai(trace, 5e-7) is trace


def test_trace_time_range_spans_timestamps() -> None:
    """trace_time_range spans the min and max timestamps of the frame."""
    assert trace_time_range(_trace_frame()) == (0.0, 10.0)


def test_trace_time_range_empty_returns_none() -> None:
    """trace_time_range returns None for an empty frame."""
    assert trace_time_range(_trace_frame().iloc[0:0]) is None


def test_paraver_provider_loads_code_mode_trace_with_legend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ParaverProvider maps code-mode rows to legend entries on a trace table."""
    # Dummy .prv file — only needs to exist on disk for the provider check.
    trace = tmp_path / "t.prv"
    trace.write_text("#Paraver dummy\n")

    # Create a window CSV with header so parse_paraver_header works and
    # build_trace_table can attach state codes.
    window_csv = tmp_path / "window.csv"
    window_csv.write_text(
        f"#20260803:CSV:RUNAPP:{trace.resolve()}:nanoseconds:window_in_code_mode\n1.1.1\t0.0\t1000000000.0\t1.0\n",
        encoding="utf-8",
    )
    (tmp_path / "window.legend.csv").write_text('1.000000 "Running" 0,0,255\n', encoding="utf-8")

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
    data = provider.load()

    # The trace path (resolved) and the window header's time unit reach run_paramedir.
    assert len(captured) == 1
    called_trace, _out_dir, called_unit = captured[0]
    assert Path(called_trace) == trace.resolve()
    assert called_unit == "nanoseconds"

    # The loaded window CSV interval (min start, max end) is exposed in seconds.
    assert provider.window_extent == (0.0, 1.0)

    # Mode, unit, and prv path come from the window header.
    assert data.window_mode == ParaverWindowMode.CODE
    assert data.time_unit == "nanoseconds"
    assert data.prv_path == str(trace.resolve())

    # Code-mode label names the trace; the window name stays out (legend entries
    # already carry the state names, so the label only appears in tooltips).
    assert data.label == "t"
    assert "window" not in data.label

    # The legend was loaded and matched the single trace row.
    assert len(data.legend) == 1
    row = data.trace.iloc[0]
    assert row["legend_label"] == "Running"
    assert row["legend_color"] == "rgb(0,0,255)"

    # Computed metrics (exact values depend on counter weights, but invariants hold).
    assert row["flops"] > 0
    assert row["bytes"] > 0
    assert row["ai"] == pytest.approx(row["flops"] / row["bytes"])
    assert row["perf"] == pytest.approx(row["flops"] / row["duration_s"])


def test_paraver_provider_code_mode_missing_legend_raises_user_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Code-mode windows without a legend fail fast, before paramedir runs."""
    window_csv = tmp_path / "window.csv"
    window_csv.write_text(
        "#20260803:CSV:RUNAPP:/p/t.prv:microseconds:\n1.1.1\t0.0\t1000000.0\t1.0\n",
        encoding="utf-8",
    )
    trace = tmp_path / "t.prv"
    trace.write_text("#dummy\n")

    captured: list[object] = []
    monkeypatch.setattr("carm_roofline.gui.providers.run_paramedir", lambda *args: captured.append(args))

    with pytest.raises(UserError, match="legend CSV not found"):
        ParaverProvider(trace, window_csv).load()
    assert captured == []


def test_paraver_provider_legend_labels_align_to_state_not_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows keep their own legend entry even when the trace is not code-sorted.

    Regression: merge_asof returns a fresh RangeIndex, so the label/color
    assignment used to pair positionally with the code-sorted merge output,
    mislabeling every row whose state differs from the code order.
    """
    trace = tmp_path / "t.prv"
    trace.write_text("#Paraver dummy\n")
    window_csv = tmp_path / "window.csv"
    # State 8.0 active first, state 1.0 second — reverse of the legend's code order.
    window_csv.write_text(
        f"#20260803:CSV:RUNAPP:{trace.resolve()}:nanoseconds:window_in_code_mode\n"
        "1.1.1\t0.0\t1000000.0\t8.0\n"
        "1.1.1\t1000000.0\t1000000.0\t1.0\n",
        encoding="utf-8",
    )
    (tmp_path / "window.legend.csv").write_text(
        '1.000000 "Running" 0,0,255\n8.000000 "Wait/WaitAll" 235,0,0\n', encoding="utf-8"
    )

    def _fake_run_paramedir(trace_path: str | Path, output_dir: str | Path, time_unit: str) -> None:
        out = Path(output_dir)
        for name in ("fp-avx2-dp.csv", "mem-loads.csv"):
            (out / name).write_text(
                f"#ts:CSV:RUNAPP:/p/t.prv:{time_unit}:window_in_code_mode\n"
                "1.1.1\t0.0\t1000000.0\t4\n"
                "1.1.1\t1000000.0\t1000000.0\t4\n",
                encoding="utf-8",
            )

    monkeypatch.setattr("carm_roofline.gui.providers.run_paramedir", _fake_run_paramedir)
    monkeypatch.setattr("shutil.which", lambda _x: "/usr/bin/paramedir")

    data = ParaverProvider(trace, window_csv).load()
    assert data.trace["state_code"].astype(float).tolist() == [8.0, 1.0]
    assert data.trace["legend_label"].tolist() == ["Wait/WaitAll", "Running"]
    assert data.trace["legend_color"].tolist() == ["rgb(235,0,0)", "rgb(0,0,255)"]


def test_paraver_provider_gradient_mode_loads_without_legend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Gradient-mode windows need no legend and carry no legend columns."""
    window_csv = tmp_path / "window.csv"
    window_csv.write_text(
        "#20260803:CSV:RUNAPP:/p/t.prv:microseconds:window_in_null_gradient_mode\n1.1.1\t0.0\t1000000.0\t1.0\n",
        encoding="utf-8",
    )
    trace = tmp_path / "t.prv"
    trace.write_text("#dummy\n")

    def _fake_run_paramedir(trace_path: str | Path, output_dir: str | Path, time_unit: str) -> None:
        out = Path(output_dir)
        (out / "fp-avx2-dp.csv").write_text(
            f"#ts:CSV:RUNAPP:/p/t.prv:{time_unit}:window_in_null_gradient_mode\n1.1.1\t0.0\t1000000.0\t4\n",
            encoding="utf-8",
        )
        (out / "mem-loads.csv").write_text(
            f"#ts:CSV:RUNAPP:/p/t.prv:{time_unit}:window_in_null_gradient_mode\n1.1.1\t0.0\t1000000.0\t2\n",
            encoding="utf-8",
        )

    monkeypatch.setattr("carm_roofline.gui.providers.run_paramedir", _fake_run_paramedir)
    monkeypatch.setattr("shutil.which", lambda _x: "/usr/bin/paramedir")

    data = ParaverProvider(trace, window_csv).load()
    assert data.window_mode == ParaverWindowMode.GRADIENT
    assert data.legend is None
    assert "legend_label" not in data.trace.columns
    # Gradient label falls back to the bare CSV stem (no app suffix to strip).
    assert data.label == "window"


def test_paraver_provider_gradient_label_strips_app_suffix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Gradient legend entry drops the app stem wxparaver appends to window CSVs."""
    trace = tmp_path / "t.prv"
    trace.write_text("#Paraver dummy\n")
    window_csv = tmp_path / "Instructions_per_cycle_t.csv"
    window_csv.write_text(
        f"#20260803:CSV:RUNAPP:{trace.resolve()}:microseconds:window_in_null_gradient_mode\n"
        "1.1.1\t0.0\t1000000.0\t1.0\n",
        encoding="utf-8",
    )

    def _fake_run_paramedir(trace_path: str | Path, output_dir: str | Path, time_unit: str) -> None:
        out = Path(output_dir)
        (out / "fp-avx2-dp.csv").write_text(
            f"#ts:CSV:RUNAPP:/p/t.prv:{time_unit}:window_in_null_gradient_mode\n1.1.1\t0.0\t1000000.0\t4\n",
            encoding="utf-8",
        )
        (out / "mem-loads.csv").write_text(
            f"#ts:CSV:RUNAPP:/p/t.prv:{time_unit}:window_in_null_gradient_mode\n1.1.1\t0.0\t1000000.0\t2\n",
            encoding="utf-8",
        )

    monkeypatch.setattr("carm_roofline.gui.providers.run_paramedir", _fake_run_paramedir)
    monkeypatch.setattr("shutil.which", lambda _x: "/usr/bin/paramedir")

    data = ParaverProvider(trace, window_csv).load()
    assert data.label == "Instructions_per_cycle"


def test_paraver_provider_missing_paramedir_raises_user_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ParaverProvider raises UserError when paramedir is not on PATH."""
    window_csv = tmp_path / "window.csv"
    window_csv.write_text(
        "#20260803:CSV:RUNAPP:/p/t.prv:microseconds:window_in_null_gradient_mode\n1.1.1\t0.0\t1000000.0\t1.0\n",
        encoding="utf-8",
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
