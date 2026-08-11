"""Unit tests for carm_roofline.paraver.loading."""

from __future__ import annotations

from pathlib import Path
from typing import get_args

import pandas as pd
import pytest

from carm_roofline.paraver.loading import (
    CsvPrecision,
    DEFAULT_CSV_PRECISION,
    METRIC_COLUMNS,
    TIME_SCALE_FACTORS,
    TRACE_COLUMNS,
    MetricColumn,
    ParaverHeader,
    ParaverWindowMode,
    TraceRow,
    WINDOW_CSV_COLUMNS,
    load_legend_csv,
    load_window_csv,
    parse_paraver_header,
    time_unit_to_seconds,
    window_csv_precision,
)

pytestmark = pytest.mark.unit

REAL_HEADER = (
    "#20260803170420:CSV:RUNAPP:/home/alexandre/Desktop/carm-paraver/"
    "carm_traces_lulesh/carm/lulesh2.0_64p_carm_DP.chop2.prv:Nanoseconds:window_in_code_mode:1.000000:15.000000"
)


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_trace_schema_single_source_of_truth() -> None:
    """TraceRow is the schema source; the runtime tuples and MetricColumn agree."""
    assert TRACE_COLUMNS == TraceRow._fields
    assert METRIC_COLUMNS == tuple(name for name in TRACE_COLUMNS if name not in WINDOW_CSV_COLUMNS)
    assert tuple(get_args(MetricColumn)) == tuple(
        name for name in TRACE_COLUMNS if name not in ("thread_id", "state_code")
    )


def test_time_unit_to_seconds_table() -> None:
    assert {unit: time_unit_to_seconds(unit) for unit in TIME_SCALE_FACTORS} == {
        "seconds": 1.0,
        "milliseconds": 1e-3,
        "microseconds": 1e-6,
        "nanoseconds": 1e-9,
    }


def test_time_unit_to_seconds_empty_and_missing_default_microseconds() -> None:
    assert time_unit_to_seconds(None) == 1e-6
    assert time_unit_to_seconds("") == 1e-6


def test_time_unit_to_seconds_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown time unit"):
        time_unit_to_seconds("foo")


def test_parse_paraver_header_full() -> None:
    header = parse_paraver_header(REAL_HEADER)
    assert header == ParaverHeader(
        timestamp="20260803170420",
        prv_path="/home/alexandre/Desktop/carm-paraver/carm_traces_lulesh/carm/lulesh2.0_64p_carm_DP.chop2.prv",
        time_unit="Nanoseconds",
        window_mode="window_in_code_mode",
    )


def test_parse_paraver_header_short_defaults() -> None:
    header = parse_paraver_header("#ts:CSV:RUNAPP:/p/x.prv")
    assert header == ParaverHeader(
        timestamp="ts",
        prv_path="/p/x.prv",
        time_unit="",
        window_mode="window_in_code_mode",
    )


@pytest.mark.parametrize("line", ["1.1.1\t0.00\t1294.00\t0.00", "#abc"])
def test_parse_paraver_header_rejects_non_header(line: str) -> None:
    with pytest.raises(ValueError):
        parse_paraver_header(line)


@pytest.mark.parametrize(
    ("header_mode", "expected"),
    [
        ("window_in_code_mode", ParaverWindowMode.CODE),
        ("window_in_null_gradient_mode", ParaverWindowMode.GRADIENT),
    ],
)
def test_paraver_window_mode_from_header(header_mode: str, expected: ParaverWindowMode) -> None:
    assert ParaverWindowMode.from_header(header_mode) == expected


@pytest.mark.parametrize("header_mode", ["", "window_in_code", "some_other_mode"])
def test_paraver_window_mode_from_header_unknown_raises(header_mode: str) -> None:
    with pytest.raises(ValueError, match="unknown Paraver window mode"):
        ParaverWindowMode.from_header(header_mode)


def test_load_window_csv_schema_and_scaling(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "window.csv",
        "#20260803170420:CSV:RUNAPP:/p/x.prv:Nanoseconds:window_in_code_mode\n"
        "1.1.1\t0.00\t1294.00\t0.00\n"
        "1.1.1\t1294.00\t100.00\t1.00\n"
        "1.2.1\t0.00\t500.00\t8.00\n",
    )
    frame = load_window_csv(path)
    assert list(frame.columns) == list(TRACE_COLUMNS)
    assert frame["thread_id"].dtype.name == "category"
    assert frame["state_code"].dtype.name == "category"
    assert frame["time_s"].dtype == "float64"
    assert frame["duration_s"].dtype == "float64"
    for column in METRIC_COLUMNS:
        assert frame[column].dtype == "float64"
        assert frame[column].isna().all()
    assert frame["time_s"].tolist() == pytest.approx([0.0, 1294e-9, 0.0])
    assert frame["duration_s"].tolist() == pytest.approx([1294e-9, 100e-9, 500e-9])


def test_load_window_csv_requires_header(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "headerless.csv",
        "1.1.1\t0.00\t1294.00\t0.00\n1.1.1\t1294.00\t100.00\t1.00\n",
    )
    with pytest.raises(ValueError, match="header"):
        load_window_csv(path)


def test_load_legend_csv_basic(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "legend.csv",
        '15.000000 "Others" 192,224,0\n'
        '0.000000 "Idle" 0,0,0\n'
        '1.000000 "Running" 0,0,255\n'
        '8.000000 "Wait/WaitAll" 235,0,0\n'
        '10.000000 "Immediate Send" 255,0,255\n'
        '11.000000 "Immediate Receive" 100,100,177\n'
        '13.000000 "Group Communication" 255,144,26\n',
    )
    legend = load_legend_csv(path)
    assert legend["code"].tolist() == [0.0, 1.0, 8.0, 10.0, 11.0, 13.0, 15.0]
    assert legend["code_end"].tolist() == legend["code"].tolist()
    assert legend["label"].tolist() == [
        "Idle",
        "Running",
        "Wait/WaitAll",
        "Immediate Send",
        "Immediate Receive",
        "Group Communication",
        "Others",
    ]
    assert legend["r"].tolist() == [0, 0, 235, 255, 100, 255, 192]
    assert legend["g"].tolist() == [0, 0, 0, 0, 100, 144, 224]
    assert legend["b"].tolist() == [0, 255, 0, 255, 177, 26, 0]
    assert legend["r"].dtype == "int64"
    assert legend["g"].dtype == "int64"
    assert legend["b"].dtype == "int64"


def test_load_legend_csv_ranges(tmp_path: Path) -> None:
    path = _write(tmp_path, "legend.csv", '10-15 "x" 1,2,3\n')
    legend = load_legend_csv(path)
    assert legend["code"].tolist() == [10.0]
    assert legend["code_end"].tolist() == [15.0]
    assert legend["label"].tolist() == ["x"]


def test_load_legend_csv_negative_code(tmp_path: Path) -> None:
    path = _write(tmp_path, "legend.csv", '-5 "x" 1,2,3\n')
    legend = load_legend_csv(path)
    assert legend["code"].tolist() == [-5.0]
    assert legend["label"].tolist() == ["x"]


@pytest.mark.parametrize("line", ['garbage\n', '5 "no colors"\n', '5- "x" 1,2,3\n'])
def test_load_legend_csv_malformed_line_raises(tmp_path: Path, line: str) -> None:
    path = _write(tmp_path, "legend.csv", line)
    with pytest.raises(ValueError, match="malformed legend line 1"):
        load_legend_csv(path)


def test_window_csv_precision_reads_paraver_format(tmp_path: Path) -> None:
    """Real Paraver window: 2 dp data columns, 6 dp header vmin/vmax."""
    path = _write(
        tmp_path,
        "window.csv",
        "#20260805180157:CSV:RUNAPP:/p/x.prv:Microseconds:window_in_null_gradient_mode:0.055993:4.599730\n"
        "1.1.1\t0.00\t1.29\t0.00\n"
        "1.1.1\t134056.45\t11.72\t0.52\n",
    )
    assert window_csv_precision(path) == CsvPrecision(2, 2, 2, 6)


def test_window_csv_precision_per_column_max(tmp_path: Path) -> None:
    """Per-column max over the data rows; header dp from vmin/vmax."""
    path = _write(
        tmp_path,
        "window.csv",
        "#20260805180157:CSV:RUNAPP:/p/x.prv:Microseconds:window_in_null_gradient_mode:0.1:1.5\n"
        "1.1.1\t0.0\t1.25\t0.0001\n"
        "1.1.1\t10.5\t1.2\t0.1\n",
    )
    assert window_csv_precision(path) == CsvPrecision(1, 2, 4, 1)


def test_window_csv_precision_no_data_rows_falls_back(tmp_path: Path) -> None:
    """A header-only file (no data rows) falls back to the defaults."""
    path = _write(
        tmp_path,
        "window.csv",
        "#20260805180157:CSV:RUNAPP:/p/x.prv:Microseconds:window_in_null_gradient_mode:0.055993:4.599730\n",
    )
    assert window_csv_precision(path) == DEFAULT_CSV_PRECISION


def test_window_csv_precision_short_header_defaults(tmp_path: Path) -> None:
    """A header with <8 ':'-fields leaves header dp at the default 6; data dp from the rows."""
    path = _write(
        tmp_path,
        "window.csv",
        "#20260805180157:CSV:RUNAPP:/p/x.prv:Microseconds:window_in_null_gradient_mode\n"
        "1.1.1\t0.00\t1.29\t0.00\n",
    )
    assert window_csv_precision(path) == CsvPrecision(2, 2, 2, 6)


def test_time_unit_to_seconds_unknown_maps_to_microseconds() -> None:
    """'Unknown' (any case, the exported form of an empty unit) is the µs default,
    exactly like the empty string."""
    assert time_unit_to_seconds("Unknown") == 1e-6
    assert time_unit_to_seconds("unknown") == 1e-6
