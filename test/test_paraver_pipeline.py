"""Unit tests for carm_roofline.paraver.pipeline and carm_roofline.paraver.counters."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import pytest

from carm_roofline.paraver import pipeline
from carm_roofline.paraver.counters import (
    INTEL_COUNTERS,
    CounterSpec,
    counter_config_template,
)
from carm_roofline.paraver.pipeline import (
    TRACE_COLUMNS,
    attach_state_codes,
    build_trace_table,
    compute_trace_metrics,
    load_counter_data,
    merge_counter_frames,
    parse_counter_csv,
    render_counter_config,
    run_paramedir,
    write_counter_configs,
)

pytestmark = pytest.mark.unit

EXPECTED_NAMES = [
    "fp-scalar-dp",
    "fp-scalar-sp",
    "fp-sse-dp",
    "fp-sse-sp",
    "fp-avx2-dp",
    "fp-avx2-sp",
    "fp-avx512-dp",
    "fp-avx512-sp",
    "mem-loads",
    "mem-stores",
]

EXPECTED_EVT_TYPES = [
    44548973,
    42001053,
    44561246,
    42001056,
    44995982,
    42001055,
    44021956,
    42001054,
    44723342,
    44604811,
]

EXPECTED_FLOPS_MULTIPLIERS = [1, 1, 2, 4, 4, 8, 8, 16, 0, 0]
EXPECTED_BYTES_PER_INST = [8.0, 4.0, 16.0, 16.0, 32.0, 32.0, 64.0, 64.0, 0.0, 0.0]
EXPECTED_IS_MEMORY = [False] * 8 + [True] * 2


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def _counter_csv(unit: str | None, rows: str) -> str:
    header = f"#20260803170420:CSV:RUNAPP:/p/x.prv:{unit}:window_in_code_mode\n" if unit else ""
    return header + rows


def _registry_index(name: str) -> int:
    return EXPECTED_NAMES.index(name)


def test_intel_counters_registry_entries() -> None:
    assert [spec.name for spec in INTEL_COUNTERS] == EXPECTED_NAMES
    for spec, evt_type, flops, bytes_per_inst, is_memory in zip(
        INTEL_COUNTERS,
        EXPECTED_EVT_TYPES,
        EXPECTED_FLOPS_MULTIPLIERS,
        EXPECTED_BYTES_PER_INST,
        EXPECTED_IS_MEMORY,
    ):
        assert spec.evt_type == evt_type
        assert spec.evt_type > 0
        if is_memory:
            assert spec.flops_multiplier == 0
            assert spec.bytes_per_inst == 0.0
            assert spec.is_memory is True
        else:
            assert spec.flops_multiplier > 0
            assert spec.bytes_per_inst > 0.0
            assert spec.is_memory is False


def test_counter_config_template_placeholders() -> None:
    template = counter_config_template()
    assert template.is_file()
    text = template.read_text(encoding="utf-8")
    assert "window_name $COUNTER" in text
    assert "window_units $TIME_UNIT" in text
    assert "window_filter_module evt_type 1 $EVT_TYPE" in text
    assert 'window_filter_module evt_type_label' not in text


def test_render_counter_config_substitutes() -> None:
    rendered = render_counter_config(INTEL_COUNTERS[_registry_index("fp-avx2-dp")], "Nanoseconds")
    assert "window_name fp-avx2-dp" in rendered
    assert "window_units Nanoseconds" in rendered
    assert "window_filter_module evt_type 1 44995982" in rendered
    assert "window_filter_module evt_type_label" not in rendered
    assert "$" not in rendered


def test_render_counter_config_rejects_leftover_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pipeline,
        "_load_template",
        lambda: "window_name $COUNTER\nwindow_units $UNKNOWN\n",
    )
    with pytest.raises(ValueError, match="unsubstituted placeholder"):
        render_counter_config(INTEL_COUNTERS[0], "Microseconds")


def test_write_counter_configs_writes_all(tmp_path: Path) -> None:
    output_dir = tmp_path / "cfgs"
    written = write_counter_configs(output_dir, "Nanoseconds")
    assert len(written) == 10
    assert [path.name for path in written] == [f"{name}.cfg" for name in EXPECTED_NAMES]
    for spec, path in zip(INTEL_COUNTERS, written):
        assert path.is_file()
        text = path.read_text(encoding="utf-8")
        assert f"window_name {spec.name}" in text
        assert "window_units Nanoseconds" in text
        assert f"window_filter_module evt_type 1 {spec.evt_type}" in text
        assert "window_filter_module evt_type_label" not in text


def test_write_counter_configs_defaults_time_unit(tmp_path: Path) -> None:
    written = write_counter_configs(tmp_path, "")
    text = written[0].read_text(encoding="utf-8")
    assert "window_units Microseconds" in text
    assert "$" not in text


@pytest.mark.parametrize("unit", ["Unknown", "unknown", "  Unknown "])
def test_write_counter_configs_normalizes_unknown_unit(tmp_path: Path, unit: str) -> None:
    written = write_counter_configs(tmp_path, unit)
    assert "window_units Microseconds" in written[0].read_text(encoding="utf-8")


def test_write_counter_configs_canonicalizes_unit_case(tmp_path: Path) -> None:
    written = write_counter_configs(tmp_path, "milliseconds")
    assert "window_units Milliseconds" in written[0].read_text(encoding="utf-8")


def test_write_counter_configs_rejects_unknown_unit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown time unit"):
        write_counter_configs(tmp_path, "fortnights")


def test_parse_counter_csv_scales_to_seconds(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "counter.csv",
        _counter_csv("Microseconds", "1.1.1\t0.00\t1000000.00\t42\n1.1.1\t1000000.00\t2000000.00\t7\n"),
    )
    frame = parse_counter_csv(path)
    assert frame["thread_id"].dtype.name == "category"
    assert frame["time_s"].tolist() == pytest.approx([0.0, 1.0])
    assert frame["duration_s"].tolist() == pytest.approx([1.0, 2.0])
    assert frame["value"].tolist() == [42, 7]


def test_parse_counter_csv_without_header_defaults_microseconds(tmp_path: Path) -> None:
    path = _write(tmp_path, "counter.csv", "1.1.1\t0.00\t1000000.00\t42\n")
    frame = parse_counter_csv(path)
    assert len(frame) == 1
    assert frame["time_s"].tolist() == pytest.approx([0.0])
    assert frame["duration_s"].tolist() == pytest.approx([1.0])
    assert frame["value"].tolist() == [42]


def test_parse_counter_csv_explicit_unit_overrides_header(tmp_path: Path) -> None:
    path = _write(tmp_path, "counter.csv", _counter_csv("Microseconds", "1.1.1\t0.00\t1000000.00\t42\n"))
    frame = parse_counter_csv(path, time_unit="nanoseconds")
    assert frame["duration_s"].tolist() == pytest.approx([1e-3])


def test_parse_counter_csv_explicit_unit_for_headerless(tmp_path: Path) -> None:
    path = _write(tmp_path, "counter.csv", "1.1.1\t0.00\t1000000000.00\t42\n")
    frame = parse_counter_csv(path, time_unit="Nanoseconds")
    assert frame["time_s"].tolist() == pytest.approx([0.0])
    assert frame["duration_s"].tolist() == pytest.approx([1.0])
    assert frame["value"].tolist() == [42]


def test_load_counter_data_finds_present_files(tmp_path: Path) -> None:
    _write(tmp_path, "mem-loads.csv", _counter_csv("Microseconds", "1.1.1\t0.00\t1000000.00\t42\n"))
    _write(tmp_path, "mem-stores.csv", _counter_csv("Microseconds", "1.1.1\t0.00\t1000000.00\t7\n"))
    data = load_counter_data(tmp_path)
    assert list(data) == ["mem-loads", "mem-stores"]
    assert "mem-loads" in data["mem-loads"].columns
    assert "value" not in data["mem-loads"].columns
    assert data["mem-loads"]["mem-loads"].tolist() == [42]


def test_load_counter_data_forwards_time_unit(tmp_path: Path) -> None:
    _write(tmp_path, "mem-loads.csv", "1.1.1\t0.00\t1000000000.00\t42\n")
    data = load_counter_data(tmp_path, "Nanoseconds")
    assert data["mem-loads"]["duration_s"].tolist() == pytest.approx([1.0])


def _aligned_frame(counter: str, values: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "thread_id": pd.Categorical(["1.1.1", "1.1.1"]),
            "time_s": [0.0, 1.0],
            "duration_s": [0.5, 0.5],
            counter: values,
        }
    )


def test_merge_counter_frames_aligned_column_stacks() -> None:
    merged = merge_counter_frames(
        {"mem-loads": _aligned_frame("mem-loads", [10, 20]), "mem-stores": _aligned_frame("mem-stores", [3, 4])}
    )
    assert list(merged.columns) == ["thread_id", "time_s", "duration_s", *EXPECTED_NAMES]
    assert len(merged) == 2
    assert merged["mem-loads"].tolist() == [10, 20]
    assert merged["mem-stores"].tolist() == [3, 4]
    assert not merged.isna().any().any()


def test_merge_counter_frames_misaligned_outer_joins() -> None:
    loads = _aligned_frame("mem-loads", [10, 20])
    stores = _aligned_frame("mem-stores", [100, 200])
    stores["time_s"] = [0.0, 2.0]  # misalign the second row
    merged = merge_counter_frames({"mem-loads": loads, "mem-stores": stores})
    assert len(merged) == 3
    by_time = merged.set_index("time_s")
    assert by_time.loc[0.0, "mem-loads"] == 10
    assert by_time.loc[0.0, "mem-stores"] == 100
    assert by_time.loc[1.0, "mem-loads"] == 20
    assert by_time.loc[1.0, "mem-stores"] == 0.0  # fillna(0)
    assert by_time.loc[2.0, "mem-loads"] == 0.0
    assert by_time.loc[2.0, "mem-stores"] == 200


def test_merge_counter_frames_empty_raises() -> None:
    with pytest.raises(ValueError, match="no counter frames"):
        merge_counter_frames({})


def test_merge_counter_frames_zero_fills_missing_registry_counters() -> None:
    merged = merge_counter_frames({"mem-loads": _aligned_frame("mem-loads", [10, 20])})
    assert list(merged.columns) == ["thread_id", "time_s", "duration_s", *EXPECTED_NAMES]
    for name in EXPECTED_NAMES:
        if name != "mem-loads":
            assert merged[name].tolist() == [0.0, 0.0]
    assert merged["mem-loads"].tolist() == [10, 20]


def test_merge_counter_frames_differing_thread_categories_keeps_categorical() -> None:
    # A counter firing on a subset of threads gives frames with different
    # categorical thread_id sets; the outer merge must keep thread_id categorical
    # (not coerce it to object), or the later merge_asof(by="thread_id") raises.
    loads = pd.DataFrame(
        {
            "thread_id": pd.Categorical(["1.1.1", "1.1.2"]),
            "time_s": [0.0, 1.0],
            "duration_s": [0.5, 0.5],
            "mem-loads": [10, 20],
        }
    )
    stores = pd.DataFrame(
        {
            "thread_id": pd.Categorical(["1.1.1"]),
            "time_s": [0.0],
            "duration_s": [0.5],
            "mem-stores": [3],
        }
    )
    merged = merge_counter_frames({"mem-loads": loads, "mem-stores": stores})
    assert merged["thread_id"].dtype.name == "category"
    window = pd.DataFrame(
        {
            "thread_id": pd.Categorical(["1.1.1", "1.1.2"]),
            "time_s": [0.0, 0.0],
            "duration_s": [10.0, 10.0],
            "state_code": [1.0, 8.0],
        }
    )
    # Must complete without raising MergeError/incompatible merge keys.
    attach_state_codes(merged, window)


def test_compute_trace_metrics_hand_computed() -> None:
    bursts = pd.DataFrame(
        {
            "thread_id": pd.Categorical(["1.1.1"]),
            "time_s": [0.0],
            "duration_s": [1e-6],
            **{name: 0 for name in EXPECTED_NAMES},
            "fp-avx2-dp": 10,
            "fp-scalar-sp": 4,
            "mem-loads": 100,
            "mem-stores": 50,
        }
    )
    metrics = compute_trace_metrics(bursts)
    row = metrics.iloc[0]
    assert row["flops"] == pytest.approx(44.0)  # 10×4 + 4×1
    assert row["bytes"] == pytest.approx(3600.0)  # 150 × (10×32 + 4×4)/14
    assert row["ai"] == pytest.approx(44.0 / 3600.0)
    assert row["perf"] == pytest.approx(4.4e7)


def test_compute_trace_metrics_drops_raw_counters() -> None:
    bursts = pd.DataFrame(
        {
            "thread_id": pd.Categorical(["1.1.1"]),
            "time_s": [0.0],
            "duration_s": [1e-6],
            **{name: 1 for name in EXPECTED_NAMES},
        }
    )
    metrics = compute_trace_metrics(bursts)
    assert list(metrics.columns) == ["thread_id", "time_s", "duration_s", "flops", "bytes", "ai", "perf"]


def test_compute_trace_metrics_zero_fp_row() -> None:
    bursts = pd.DataFrame(
        {
            "thread_id": pd.Categorical(["1.1.1"]),
            "time_s": [0.0],
            "duration_s": [0.0],
            **{name: 0 for name in EXPECTED_NAMES},
        }
    )
    metrics = compute_trace_metrics(bursts)
    row = metrics.iloc[0]
    assert row["flops"] == 0.0
    assert row["bytes"] == 0.0
    assert row["ai"] == 0.0
    assert row["perf"] == 0.0
    assert row["ai"] == row["ai"]  # not NaN
    assert row["perf"] == row["perf"]  # not NaN


def test_attach_state_codes_backward_and_end_check() -> None:
    window = pd.DataFrame(
        {
            "thread_id": pd.Categorical(["1.1.1", "1.1.1"]),
            "time_s": [0.0, 10.0],
            "duration_s": [10.0, 10.0],
            "state_code": [1.0, 8.0],
        }
    )
    bursts = pd.DataFrame(
        {
            "thread_id": pd.Categorical(["1.1.1", "1.1.1", "1.1.1"]),
            "time_s": [5.0, 15.0, 25.0],
            "duration_s": [1.0, 1.0, 1.0],
        }
    )
    result = attach_state_codes(bursts, window)
    assert result["state_code"].tolist() == pytest.approx([1.0, 8.0, float("nan")], nan_ok=True)
    assert result["state_code"].dtype == "float64"


def test_attach_state_codes_missing_thread() -> None:
    window = pd.DataFrame(
        {
            "thread_id": pd.Categorical(["1.1.1"]),
            "time_s": [0.0],
            "duration_s": [10.0],
            "state_code": [1.0],
        }
    )
    bursts = pd.DataFrame(
        {
            "thread_id": pd.Categorical(["1.2.1"]),
            "time_s": [5.0],
            "duration_s": [1.0],
        }
    )
    result = attach_state_codes(bursts, window)
    assert len(result) == 1
    assert result["state_code"].iloc[0] != result["state_code"].iloc[0]  # NaN


def test_build_trace_table_end_to_end(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "window.csv",
        "#20260803170420:CSV:RUNAPP:/p/x.prv:Nanoseconds:window_in_code_mode\n"
        "1.1.1\t0\t1000\t0.00\n"
        "1.1.1\t1000\t1000\t1.00\n",
    )
    counters = tmp_path / "counters"
    counters.mkdir()
    _write(
        counters,
        "fp-avx2-dp.csv",
        _counter_csv("Nanoseconds", "1.1.1\t0\t500\t2\n1.1.1\t1000\t500\t4\n"),
    )
    _write(
        counters,
        "mem-loads.csv",
        _counter_csv("Nanoseconds", "1.1.1\t0\t500\t100\n1.1.1\t1000\t500\t200\n"),
    )
    trace = build_trace_table(tmp_path / "window.csv", counters)
    assert list(trace.columns) == list(TRACE_COLUMNS)
    assert trace["state_code"].dtype.name == "category"
    assert trace["thread_id"].dtype.name == "category"
    assert trace["state_code"].cat.categories.tolist() == [0.0, 1.0]
    first = trace.iloc[0]
    assert first["time_s"] == pytest.approx(0.0)
    assert first["flops"] == pytest.approx(8.0)  # 2×4
    assert first["bytes"] == pytest.approx(3200.0)  # 100 × (2×32)/2
    assert first["ai"] == pytest.approx(8.0 / 3200.0)
    assert first["perf"] == pytest.approx(8.0 / 500e-9)
    assert first["state_code"] == 0.0
    second = trace.iloc[1]
    assert second["flops"] == pytest.approx(16.0)
    assert second["state_code"] == 1.0


def test_build_trace_table_headerless_counters_use_window_unit(tmp_path: Path) -> None:
    window = _write(
        tmp_path,
        "window.csv",
        "#20260803170420:CSV:RUNAPP:/p/x.prv:Nanoseconds:window_in_code_mode\n"
        "1.1.1\t0\t1000000000\t0.00\n",
    )
    counters = tmp_path / "counters"
    counters.mkdir()
    _write(counters, "fp-avx2-dp.csv", "1.1.1\t0\t1000000000\t1\n")
    trace = build_trace_table(window, counters)  # no explicit unit → window header unit
    assert trace.iloc[0]["duration_s"] == pytest.approx(1.0)
    assert trace.iloc[0]["perf"] == pytest.approx(4.0)  # 1 inst × 4 flops / 1 s


def test_build_trace_table_explicit_time_unit_wins(tmp_path: Path) -> None:
    window = _write(
        tmp_path,
        "window.csv",
        "#20260803170420:CSV:RUNAPP:/p/x.prv:Nanoseconds:window_in_code_mode\n"
        "1.1.1\t0\t1000000000\t0.00\n",
    )
    counters = tmp_path / "counters"
    counters.mkdir()
    _write(counters, "fp-avx2-dp.csv", "1.1.1\t0\t1000000\t1\n")  # µs durations
    trace = build_trace_table(window, counters, time_unit="microseconds")
    assert trace.iloc[0]["duration_s"] == pytest.approx(1.0)
    assert trace.iloc[0]["perf"] == pytest.approx(4.0 / 1.0)


def test_build_trace_table_raises_without_counters(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "window.csv",
        "#20260803170420:CSV:RUNAPP:/p/x.prv:Nanoseconds:window_in_code_mode\n1.1.1\t0\t1000\t0.00\n",
    )
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="no counter CSVs"):
        build_trace_table(tmp_path / "window.csv", empty_dir)


def test_run_paramedir_writes_cfgs_and_invokes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trace = _write(tmp_path, "trace.prv", "#Paraver trace\n")
    out = tmp_path / "out"
    calls: list[tuple[list[str], dict]] = []

    def fake_run(argv: list[str], **kwargs: dict) -> subprocess.CompletedProcess:
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(pipeline.shutil, "which", lambda _: "/bin/paramedir")
    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    run_paramedir(trace, out, "Nanoseconds")

    (argv, kwargs), = calls
    assert argv[:2] == ["paramedir", str(trace)]
    assert len(argv) == 2 + 2 * len(EXPECTED_NAMES)
    assert list(zip(argv[2::2], argv[3::2])) == [
        (str(out.resolve() / f"{name}.cfg"), f"{name}.csv") for name in EXPECTED_NAMES
    ]
    assert kwargs == {"cwd": str(out.resolve()), "check": True, "capture_output": True, "text": True}
    assert len(list(out.glob("*.cfg"))) == 10
    for path in out.glob("*.cfg"):
        assert "window_units Nanoseconds" in path.read_text(encoding="utf-8")


def test_run_paramedir_missing_trace_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="trace file not found"):
        run_paramedir(tmp_path / "nope.prv", tmp_path / "out", "Microseconds")


def test_run_paramedir_missing_binary_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trace = _write(tmp_path, "trace.prv", "#Paraver trace\n")
    monkeypatch.setattr(pipeline.shutil, "which", lambda _: None)
    with pytest.raises(FileNotFoundError, match="paramedir not found"):
        run_paramedir(trace, tmp_path / "out", "Microseconds")


def test_run_paramedir_nonzero_exit_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trace = _write(tmp_path, "trace.prv", "#Paraver trace\n")
    monkeypatch.setattr(pipeline.shutil, "which", lambda _: "/bin/paramedir")

    def boom(argv: list[str], **kwargs: dict) -> None:
        raise subprocess.CalledProcessError(1, argv, stderr="no such event")

    monkeypatch.setattr(pipeline.subprocess, "run", boom)
    with pytest.raises(RuntimeError, match="no such event"):
        run_paramedir(trace, tmp_path / "out", "Microseconds")
