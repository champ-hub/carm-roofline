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
    DURATION_FILTER_DEFAULT_S,
    DURATION_FILTER_OFF_S,
    BenchmarkAppsProvider,
    ParaverProvider,
    filter_trace,
    filter_trace_by_ai,
    filter_trace_by_duration,
    filter_trace_by_window,
    trace_time_range,
)
from carm_roofline.paraver import CsvPrecision, ParaverWindowMode
from carm_roofline.roofline_assembly import load_applications

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


def test_filter_trace_by_duration_keeps_rows_above_threshold() -> None:
    """Threshold 1e-3 s keeps only rows with duration_s >= 1e-3."""
    trace = _trace_frame()
    trace["duration_s"] = [1e-5, 1e-3, 1e-2]
    result = filter_trace_by_duration(trace, 1e-3)
    assert result["duration_s"].tolist() == [1e-3, 1e-2]


def test_filter_trace_by_duration_none_returns_same_trace() -> None:
    """A None threshold returns the input frame unchanged (same object)."""
    trace = _trace_frame()
    assert filter_trace_by_duration(trace, None) is trace


def test_filter_trace_by_duration_default_threshold_is_active() -> None:
    """Threshold == DURATION_FILTER_DEFAULT_S (the 100 us default) is a real filter, not "off"."""
    trace = _trace_frame()
    trace["duration_s"] = [1e-6, 1e-4, 1e-2]
    result = filter_trace_by_duration(trace, DURATION_FILTER_DEFAULT_S)
    assert result["duration_s"].tolist() == [1e-4, 1e-2]


def test_filter_trace_by_duration_off_threshold_disables() -> None:
    """Threshold <= DURATION_FILTER_OFF_S (slider leftmost, 1e-6) disables filtering.

    Pins "slider = 1e-6 => filtering off": the input frame is returned unchanged.
    """
    trace = _trace_frame()
    assert filter_trace_by_duration(trace, DURATION_FILTER_OFF_S) is trace
    assert filter_trace_by_duration(trace, 5e-7) is trace


def test_filter_trace_all_disabled_returns_same_trace() -> None:
    """No active term (None window and thresholds) returns the input frame unchanged."""
    trace = _trace_frame()
    assert filter_trace(trace, None, None, None) is trace


def test_filter_trace_window_only() -> None:
    """A window with no thresholds keeps rows with time_s inside [lo, hi]."""
    result = filter_trace(_trace_frame(), (2.0, 8.0), None, None)
    assert result["time_s"].tolist() == [5.0]


def test_filter_trace_ai_only() -> None:
    """An ai threshold with no window keeps rows with ai >= threshold."""
    trace = _trace_frame()
    trace["ai"] = [1e-4, 1e-3, 1e-2]
    result = filter_trace(trace, None, 1e-3, None)
    assert result["ai"].tolist() == [1e-3, 1e-2]


def test_filter_trace_duration_only() -> None:
    """A duration threshold with no window keeps rows with duration_s >= threshold."""
    trace = _trace_frame()
    trace["duration_s"] = [1e-5, 1e-3, 1e-2]
    result = filter_trace(trace, None, None, 1e-3)
    assert result["duration_s"].tolist() == [1e-3, 1e-2]


def test_filter_trace_combined_keeps_rows_passing_all_terms() -> None:
    """A row failing ANY active term is dropped; only rows passing all are kept."""
    trace = pd.DataFrame(
        {
            "thread_id": ["0", "0", "0", "0"],
            "time_s": [1.0, 5.0, 5.0, 5.0],
            "duration_s": [1e-2, 1e-2, 1e-5, 1e-2],
            "state_code": [1.0, 1.0, 1.0, 1.0],
            "flops": [0.0, 0.0, 0.0, 0.0],
            "bytes": [0.0, 0.0, 0.0, 0.0],
            "ai": [1e-2, 1e-4, 1e-2, 1e-2],
            "perf": [0.0, 0.0, 0.0, 0.0],
        }
    )
    result = filter_trace(trace, (2.0, 8.0), 1e-3, 1e-3)
    assert result["time_s"].tolist() == [5.0]
    assert result["ai"].tolist() == [1e-2]
    assert result["duration_s"].tolist() == [1e-2]


def test_filter_trace_off_boundary_threshold_disables_term() -> None:
    """A threshold <= its OFF constant disables that term, like a None threshold."""
    trace = _trace_frame()
    trace["ai"] = [1e-6, 1e-3, 1e-2]
    result = filter_trace(trace, (2.0, 8.0), AI_FILTER_OFF_AI, DURATION_FILTER_OFF_S)
    assert result["time_s"].tolist() == [5.0]
    assert result["ai"].tolist() == [1e-3]
    below = filter_trace(trace, (2.0, 8.0), 5e-7, 5e-7)
    assert below["time_s"].tolist() == [5.0]
    assert below["ai"].tolist() == [1e-3]


def test_trace_time_range_spans_timestamps() -> None:
    """trace_time_range spans the min and max timestamps of the frame."""
    assert trace_time_range(_trace_frame()) == (0.0, 10.0)


def test_trace_time_range_empty_returns_none() -> None:
    """trace_time_range returns None for an empty frame."""
    assert trace_time_range(_trace_frame().iloc[0:0]) is None


def test_paraver_provider_loads_code_mode_trace_with_legend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
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

    # Precision is detected from the window CSV: row '1.1.1\t0.0\t1000000000.0\t1.0'
    # has 1 dp per data column; the header has 6 ':'-fields so header dp falls
    # back to 6.
    assert data.precision == CsvPrecision(1, 1, 1, 6)

    # Code-mode label names the trace; the window name stays out (legend entries
    # already carry the state names, so the label only appears in tooltips).
    assert data.label == "t"
    assert "window" not in data.label

    # The legend was loaded and matched the single trace row.
    assert len(data.legend) == 1
    row = data.trace.iloc[0]
    assert row["legend_label"] == "Running"
    assert row["legend_color"] == "rgb(0,0,255)"

    # Progress popup protocol: the 0% line opens it before paramedir; exactly one
    # full 100% line closes it, and nothing is printed after it.
    out = capsys.readouterr().out
    assert out.startswith("[                              ] 0.0%\r")
    assert out.count("[##############################] 100.0%") == 1
    assert out.endswith("[##############################] 100.0%\r\n")

    # Computed metrics (exact values depend on counter weights, but invariants hold).
    assert row["flops"] > 0
    assert row["bytes"] > 0
    assert row["ai"] == pytest.approx(row["flops"] / row["bytes"])
    assert row["perf"] == pytest.approx(row["flops"] / row["duration_s"])
    assert row["load_share"] == 1.0  # 2 loads, no mem-stores.csv → stores zero-filled


def test_paraver_provider_trace_only_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Trace without window CSV: CODE mode, single constant legend entry, µs unit."""
    trace = tmp_path / "t.prv"
    trace.write_text("#Paraver dummy\n")
    captured: list[str] = []

    def _fake_run_paramedir(trace_path: str | Path, output_dir: str | Path, time_unit: str) -> None:
        captured.append(time_unit)
        out = Path(output_dir)
        # One FP counter row: thread 1.1.1, 0-1 s, value=4 instructions
        (out / "fp-avx2-dp.csv").write_text(
            f"#ts:CSV:RUNAPP:/p/t.prv:{time_unit}:window_in_code_mode\n1.1.1\t0.0\t1000000.0\t4\n",
            encoding="utf-8",
        )
        # One memory counter row: same burst, value=2 loads
        (out / "mem-loads.csv").write_text(
            f"#ts:CSV:RUNAPP:/p/t.prv:{time_unit}:window_in_code_mode\n1.1.1\t0.0\t1000000.0\t2\n",
            encoding="utf-8",
        )

    monkeypatch.setattr("carm_roofline.gui.providers.run_paramedir", _fake_run_paramedir)
    monkeypatch.setattr("shutil.which", lambda _x: "/usr/bin/paramedir")

    provider = ParaverProvider(trace, None)
    data = provider.load()

    assert captured == ["Microseconds"]
    assert provider.window_extent is None
    assert data.window_mode == ParaverWindowMode.CODE
    assert data.time_unit == "Microseconds"
    assert data.label == "t"
    assert data.prv_path == str(trace.resolve())
    assert data.legend is None
    assert data.precision == CsvPrecision(2, 2, 2, 6)  # DEFAULT_CSV_PRECISION
    assert data.trace["legend_label"].tolist() == ["t"] * len(data.trace)
    assert data.trace["legend_color"].tolist() == ["rgb(128,128,128)"] * len(data.trace)
    assert data.trace["state_code"].astype(float).isna().all()


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


def test_paraver_provider_no_counter_csvs_raises_user_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A paramedir output dir with no counter CSVs raises UserError, not ValueError.

    Traces without the registered Intel counters (e.g. ARM/GPU) make paramedir
    write no counter files; build_trace_table's ValueError must surface as
    UserError so the GUI can warn and continue instead of crashing startup.
    """
    window_csv = tmp_path / "window.csv"
    window_csv.write_text(
        "#20260803:CSV:RUNAPP:/p/t.prv:microseconds:window_in_null_gradient_mode\n1.1.1\t0.0\t1000000.0\t1.0\n",
        encoding="utf-8",
    )
    trace = tmp_path / "t.prv"
    trace.write_text("#dummy\n")

    # paramedir "succeeds" but writes no counter CSVs into its output dir.
    monkeypatch.setattr("carm_roofline.gui.providers.run_paramedir", lambda *args: None)
    monkeypatch.setattr("shutil.which", lambda _x: "/usr/bin/paramedir")

    with pytest.raises(UserError, match="counter CSVs"):
        ParaverProvider(trace, window_csv).load()


def test_paraver_provider_paramedir_failure_raises_user_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-zero paramedir exit raises UserError, not RuntimeError.

    Corrupt/unsupported .prv traces make paramedir exit non-zero; the
    RuntimeError must surface as UserError so the GUI can warn and continue
    instead of crashing startup.
    """
    window_csv = tmp_path / "window.csv"
    window_csv.write_text(
        "#20260803:CSV:RUNAPP:/p/t.prv:microseconds:window_in_null_gradient_mode\n1.1.1\t0.0\t1000000.0\t1.0\n",
        encoding="utf-8",
    )
    trace = tmp_path / "t.prv"
    trace.write_text("#dummy\n")

    def _failing_paramedir(trace_path: str | Path, output_dir: str | Path, time_unit: str) -> None:
        raise RuntimeError("paramedir failed (exit 1): corrupt trace")

    monkeypatch.setattr("carm_roofline.gui.providers.run_paramedir", _failing_paramedir)
    monkeypatch.setattr("shutil.which", lambda _x: "/usr/bin/paramedir")

    with pytest.raises(UserError, match="paramedir failed"):
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


def test_load_applications_round_trips_time_s(tmp_path: Path) -> None:
    """A point's time_s survives the JSONL round trip; absent time_s loads as None."""
    path = tmp_path / "applications.jsonl"
    path.write_text(
        json.dumps(
            {
                "format_version": "2.0",
                "metadata": {"name": "app", "date": "2026-01-01", "command": "run"},
                "aggregation": "avg",
                "points": [
                    {
                        "label": "p_traced",
                        "total_flops": 1,
                        "total_bytes": 1,
                        "runtime_s": 1,
                        "num_ranks": 1,
                        "num_threads": 1,
                        "num_regions": 1,
                        "arithmetic_intensity": 1,
                        "flops_per_second": 1,
                        "bandwidth": 1,
                        "time_s": 12.5,
                    },
                    {
                        "label": "p_carm",
                        "total_flops": 1,
                        "total_bytes": 1,
                        "runtime_s": 1,
                        "num_ranks": 1,
                        "num_threads": 1,
                        "num_regions": 1,
                        "arithmetic_intensity": 1,
                        "flops_per_second": 1,
                        "bandwidth": 1,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    records = load_applications(path)
    assert len(records) == 1
    traced, carm = records[0].points
    assert traced.label == "p_traced"
    assert traced.time_s == pytest.approx(12.5)
    assert carm.label == "p_carm"
    assert carm.time_s is None


def test_load_applications_mixed_format_versions_preserves_optional_fractions(tmp_path: Path) -> None:
    """A file mixing 2.0 (4-key) and 3.0 (3-key) records loads both shapes verbatim."""
    path = tmp_path / "applications.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(rec, sort_keys=True)
            for rec in [
                {
                    "format_version": "2.0",
                    "metadata": {"name": "legacy", "date": "2026-01-01", "command": "run"},
                    "aggregation": "avg",
                    "points": [
                        {
                            "label": "p_legacy",
                            "total_flops": 1,
                            "total_bytes": 1,
                            "runtime_s": 1,
                            "num_ranks": 1,
                            "num_threads": 1,
                            "num_regions": 1,
                            "arithmetic_intensity": 1,
                            "flops_per_second": 1,
                            "bandwidth": 1,
                            "optional_fractions": {
                                "cache-residency": {"l1": 0.6, "l2": 0.3, "l3": 0.08, "dram": 0.02}
                            },
                        }
                    ],
                },
                {
                    "format_version": "3.0",
                    "metadata": {"name": "modern", "date": "2026-01-02", "command": "run"},
                    "aggregation": "avg",
                    "points": [
                        {
                            "label": "p_modern",
                            "total_flops": 1,
                            "total_bytes": 1,
                            "runtime_s": 1,
                            "num_ranks": 1,
                            "num_threads": 1,
                            "num_regions": 1,
                            "arithmetic_intensity": 1,
                            "flops_per_second": 1,
                            "bandwidth": 1,
                            "optional_fractions": {"cache-residency": {"l1": 0.6, "l2": 0.3, "l3plus": 0.1}},
                        }
                    ],
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_applications(path)
    assert len(records) == 2
    legacy, modern = records
    assert legacy.points[0].optional_fractions == {
        "cache-residency": {"l1": 0.6, "l2": 0.3, "l3": 0.08, "dram": 0.02}
    }
    assert modern.points[0].optional_fractions == {"cache-residency": {"l1": 0.6, "l2": 0.3, "l3plus": 0.1}}


def test_paraver_provider_legend_keeps_nan_state_code_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bursts outside the window's state timeline load and keep NaN legend columns.

    attach_state_codes yields NaN state_code for bursts whose start falls past the
    window's state timeline (counter CSVs span the whole trace while the window CSV
    covers only the exported subset). Regression: the code-mode legend merge fed
    merge_asof those NaN keys, aborting load() with "Merge keys contain null
    values on left side"; the unmatched row must stay in the trace with NaN
    legend_label/legend_color (never plotted, keeping the slider bounds).
    """
    trace = tmp_path / "t.prv"
    trace.write_text("#Paraver dummy\n")
    window_csv = tmp_path / "window.csv"
    # One state active over [0, 1 s); the second burst starts at t=2 s, past the
    # state's end, so attach_state_codes gives it NaN state_code.
    window_csv.write_text(
        f"#20260803:CSV:RUNAPP:{trace.resolve()}:nanoseconds:window_in_code_mode\n"
        "1.1.1\t0.0\t1000000000.0\t1.0\n",
        encoding="utf-8",
    )
    (tmp_path / "window.legend.csv").write_text('1.000000 "Running" 0,0,255\n', encoding="utf-8")

    def _fake_run_paramedir(trace_path: str | Path, output_dir: str | Path, time_unit: str) -> None:
        out = Path(output_dir)
        for name in ("fp-avx2-dp.csv", "mem-loads.csv"):
            (out / name).write_text(
                f"#ts:CSV:RUNAPP:/p/t.prv:{time_unit}:window_in_code_mode\n"
                "1.1.1\t0.0\t1000000000.0\t4\n"
                "1.1.1\t2000000000.0\t1000000000.0\t4\n",
                encoding="utf-8",
            )

    monkeypatch.setattr("carm_roofline.gui.providers.run_paramedir", _fake_run_paramedir)
    monkeypatch.setattr("shutil.which", lambda _x: "/usr/bin/paramedir")

    data = ParaverProvider(trace, window_csv).load()

    # Both bursts survive: the in-window row maps to its legend entry, the
    # out-of-window row keeps NaN state_code and NaN legend columns.
    assert len(data.trace) == 2
    codes = data.trace["state_code"].astype(float).tolist()
    assert codes[0] == 1.0
    assert pd.isna(codes[1])
    assert data.trace["legend_label"].iloc[0] == "Running"
    assert pd.isna(data.trace["legend_label"].iloc[1])
    assert data.trace["legend_color"].iloc[0] == "rgb(0,0,255)"
    assert pd.isna(data.trace["legend_color"].iloc[1])


def test_paraver_provider_precomputes_tooltips_that_survive_filters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load() stores per-row tooltip HTML in a _tooltip column the filters keep aligned."""
    trace = tmp_path / "t.prv"
    trace.write_text("#Paraver dummy\n")
    window_csv = tmp_path / "window.csv"
    window_csv.write_text(
        f"#20260803:CSV:RUNAPP:{trace.resolve()}:nanoseconds:window_in_code_mode\n"
        "1.1.1\t0.0\t1000000000.0\t8.0\n"
        "1.1.1\t1000000000.0\t1000000000.0\t1.0\n",
        encoding="utf-8",
    )
    (tmp_path / "window.legend.csv").write_text(
        '1.000000 "Running" 0,0,255\n8.000000 "Wait/WaitAll" 235,0,0\n', encoding="utf-8"
    )

    def _fake_run_paramedir(trace_path: str | Path, output_dir: str | Path, time_unit: str) -> None:
        out = Path(output_dir)
        # Two bursts with different FP counts (so ai differs) for the AI filter split.
        (out / "fp-avx2-dp.csv").write_text(
            f"#ts:CSV:RUNAPP:/p/t.prv:{time_unit}:window_in_code_mode\n"
            "1.1.1\t0.0\t1000000000.0\t4\n"
            "1.1.1\t1000000000.0\t1000000000.0\t8\n",
            encoding="utf-8",
        )
        (out / "mem-loads.csv").write_text(
            f"#ts:CSV:RUNAPP:/p/t.prv:{time_unit}:window_in_code_mode\n"
            "1.1.1\t0.0\t1000000000.0\t2\n"
            "1.1.1\t1000000000.0\t1000000000.0\t2\n",
            encoding="utf-8",
        )

    monkeypatch.setattr("carm_roofline.gui.providers.run_paramedir", _fake_run_paramedir)
    monkeypatch.setattr("shutil.which", lambda _x: "/usr/bin/paramedir")

    data = ParaverProvider(trace, window_csv).load()
    loaded = data.trace
    assert "_tooltip" in loaded.columns
    assert len(loaded) == 2
    tooltips = loaded["_tooltip"].tolist()
    assert all(isinstance(t, str) for t in tooltips)
    # Each tooltip names the trace and pairs the state code with its own legend label.
    assert tooltips[0].startswith(f"<b>{data.label}</b>")
    assert "<b>Paraver Value</b>" in tooltips[0]
    assert " 8 (Wait/WaitAll)" in tooltips[0]
    assert " 1 (Running)" in tooltips[1]

    # The three filter helpers preserve the column and keep it aligned to rows.
    by_window = filter_trace_by_window(loaded, (0.0, 0.5))
    assert by_window["_tooltip"].tolist() == [tooltips[0]]
    mid_ai = (float(loaded["ai"].min()) + float(loaded["ai"].max())) / 2.0
    by_ai = filter_trace_by_ai(loaded, mid_ai)
    assert len(by_ai) == 1
    assert by_ai["_tooltip"].tolist() == [tooltips[by_ai.index[0]]]
    assert filter_trace_by_duration(loaded, 1.5)["_tooltip"].tolist() == []
    assert filter_trace_by_duration(loaded, None) is loaded
