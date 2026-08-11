"""Unit tests for the Paraver export serializers (GUI export panel).

The serializers are pure functions: exact file bytes (header line, tab layout,
precision-mirrored values, µs scaling, natural ordering, legend lines) are
asserted here, plus the disk write-back (``write_export_files`` into a
``tmp_path``).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from carm_roofline.gui.export import (
    AI_FILENAME,
    LDST_PERCENT_FILENAME,
    PERFORMANCE_FILENAME,
    REGION_FILENAME,
    REGION_LEGEND_FILENAME,
    ROOF_LABEL_LEGEND,
    ROOF_LABELS_FILENAME,
    ROOF_LABELS_LEGEND_FILENAME,
    ExportFile,
    build_csv_metadata_line,
    export_ai,
    export_ldst_percent,
    export_performance,
    export_proximity,
    export_region,
    export_roof_labels,
    natural_sort_key,
    region_label_codes,
    roof_label_codes,
    serialize_legend,
    write_export_files,
)
from carm_roofline.gui.providers import ParaverData
from carm_roofline.paraver import (
    CODE_WINDOW_MODE,
    DEFAULT_CSV_PRECISION,
    GRADIENT_WINDOW_MODE,
    CsvPrecision,
    ParaverWindowMode,
    load_window_csv,
    parse_paraver_header,
)
from carm_roofline.roofline_assembly import AssembledRoofline, BenchmarkRecord, RooflineFilter, assemble_roofline

pytestmark = pytest.mark.unit


def _by_name(files: tuple[ExportFile, ...]) -> dict[str, str]:
    """{file name: content} lookup over a facade's ExportFile tuple."""
    return {file.name: file.content for file in files}


def _metric_trace() -> pd.DataFrame:
    """Two rows in reversed natural order (10.0 first), times ascending with it."""
    return pd.DataFrame(
        {
            "thread_id": pd.Categorical(["10.0", "2.1"]),
            "time_s": [0.5, 1.0],
            "duration_s": [0.2, 0.1],
            "state_code": [2.0, 1.0],
            "flops": [32.0, 16.0],
            "bytes": [16.0, 8.0],
            "ai": [2.0, 2.0],
            "perf": [32.0, 16.0],
            "load_share": [0.5, 0.0],  # 10.0 half-loads; 2.1 store-only
        }
    )


def _paraver_data(trace: pd.DataFrame, precision: CsvPrecision = DEFAULT_CSV_PRECISION) -> ParaverData:
    return ParaverData(
        trace=trace,
        label="t",
        window_mode=ParaverWindowMode.CODE,
        time_unit="Microseconds",
        prv_path="/p/x.prv",
        legend=None,
        precision=precision,
    )


def _roof_records() -> list[BenchmarkRecord]:
    """Bandwidths L1=100/L2=50/L3=25/DRAM=12.5 GB/s, fma peak 100 GOPS.

    With divisor 1 the GFLOPS roof at AI *a* is min(GB/s * a, 100) per level and
    the ridge points are L1=1.0, L2=2.0, L3=4.0, DRAM=8.0.
    """
    return [
        {
            "type": "memory",
            "machine": "m",
            "isa": "x86",
            "data_type": "f32",
            "num_threads": 1,
            "cache_level": "L1",
            "memory_level_name": "L1",
            "bandwidth_gbps": 100.0,
            "load_store_ratio": "2:1",
            "timestamp": "2026-01-01T00:00:00",
        },
        {
            "type": "memory",
            "machine": "m",
            "isa": "x86",
            "data_type": "f32",
            "num_threads": 1,
            "cache_level": "L2",
            "memory_level_name": "L2",
            "bandwidth_gbps": 50.0,
            "load_store_ratio": "2:1",
            "timestamp": "2026-01-01T00:00:00",
        },
        {
            "type": "memory",
            "machine": "m",
            "isa": "x86",
            "data_type": "f32",
            "num_threads": 1,
            "cache_level": "L3",
            "memory_level_name": "L3",
            "bandwidth_gbps": 25.0,
            "load_store_ratio": "2:1",
            "timestamp": "2026-01-01T00:00:00",
        },
        {
            "type": "memory",
            "machine": "m",
            "isa": "x86",
            "data_type": "f32",
            "num_threads": 1,
            "cache_level": "DRAM",
            "memory_level_name": "DRAM",
            "bandwidth_gbps": 12.5,
            "load_store_ratio": "2:1",
            "timestamp": "2026-01-01T00:00:00",
        },
        {
            "type": "arithmetic",
            "machine": "m",
            "isa": "x86",
            "data_type": "f32",
            "num_threads": 1,
            "operation": "fma",
            "performance_gops": 100.0,
            "timestamp": "2026-01-01T00:00:00",
        },
    ]


def _assembled_roofline() -> AssembledRoofline:
    return assemble_roofline(_roof_records(), RooflineFilter(isa="x86", num_threads=1, data_type="f32"))


def _assembled_roofline_without_l1() -> AssembledRoofline:
    records = [r for r in _roof_records() if r.get("cache_level") != "L1"]
    return assemble_roofline(records, RooflineFilter(isa="x86", num_threads=1, data_type="f32"))


# ── Header, sort key, legend ──────────────────────────────────────────────────


def test_build_csv_metadata_line_exact() -> None:
    """Exact metadata template, incl. empty-unit → Unknown, CODE mode variant,
    and a custom vmin/vmax precision."""
    line = build_csv_metadata_line(
        "/p/x.prv", "Microseconds", GRADIENT_WINDOW_MODE, 0.0, 16.0, timestamp="20260803170420"
    )
    assert line == "#20260803170420:CSV:RUNAPP:/p/x.prv:Microseconds:window_in_null_gradient_mode:0.000000:16.000000"
    code_line = build_csv_metadata_line("/p/x.prv", "", CODE_WINDOW_MODE, 1, 6, timestamp="20260803170420")
    assert code_line == "#20260803170420:CSV:RUNAPP:/p/x.prv:Unknown:window_in_code_mode:1.000000:6.000000"
    custom = build_csv_metadata_line(
        "/p/x.prv", "Microseconds", GRADIENT_WINDOW_MODE, 0.0, 16.0, timestamp="20260803170420", vmin_vmax_precision=2
    )
    assert custom == "#20260803170420:CSV:RUNAPP:/p/x.prv:Microseconds:window_in_null_gradient_mode:0.00:16.00"


def test_natural_sort_key() -> None:
    """Dot-segment numeric key so '2.1' sorts before '10.0' (legacy §1.3)."""
    assert natural_sort_key("2.1") < natural_sort_key("10.0")
    assert natural_sort_key("1.1.1") == (1, 1, 1)
    assert natural_sort_key("7") == (7,)


def test_legend_serialization() -> None:
    """Exact §3.3.1 legend: quoted labels, comma colors, trailing newline per row."""
    expected = (
        '1 "L1",0,255,0\n'
        '2 "L2",0,0,255\n'
        '3 "L3",255,165,0\n'
        '4 "DRAM",255,0,0\n'
        '5 "No Floating Point Operations Found",75,0,130\n'
        '6 "Above L1",255,192,203\n'
    )
    assert serialize_legend(ROOF_LABEL_LEGEND) == expected


# ── Metric exports ────────────────────────────────────────────────────────────


def test_metric_export_contract() -> None:
    """Full export_performance file: header, 4 tab columns, µs scaling, 2 dp cells
    (default precision), natural thread order (2.1 before 10.0) even when the
    trace order is reversed."""
    paraver = _paraver_data(_metric_trace())
    content = _by_name(export_performance(paraver.trace, paraver))[PERFORMANCE_FILENAME]
    lines = content.split("\n")
    assert re.fullmatch(
        r"#\d{14}:CSV:RUNAPP:/p/x\.prv:Microseconds:window_in_null_gradient_mode:0\.000000:0\.000000", lines[0]
    )
    rows = [ln for ln in lines[1:] if ln]
    assert len(rows) == 2
    assert all(len(r.split("\t")) == 4 for r in rows)
    # Natural thread order first, then time; time/duration scaled to µs.
    assert rows[0] == "2.1\t1000000.00\t100000.00\t0.00"
    assert rows[1] == "10.0\t500000.00\t200000.00\t0.00"


def test_performance_values_are_gflops() -> None:
    """perf (FLOP/s) → GFLOPS in the value column; export_ai passes AI through."""
    trace = _metric_trace()
    trace["perf"] = [32e9, 16e9]  # GFLOPS stay visible at 2 dp (16.00 / 32.00)
    paraver = _paraver_data(trace)
    perf_content = _by_name(export_performance(trace, paraver))[PERFORMANCE_FILENAME]
    perf_rows = [ln for ln in perf_content.split("\n")[1:] if ln]
    assert perf_rows[0].split("\t")[3] == "16.00"  # thread 2.1 → 16 GFLOPS
    assert perf_rows[1].split("\t")[3] == "32.00"  # thread 10.0 → 32 GFLOPS
    ai_content = _by_name(export_ai(trace, paraver))[AI_FILENAME]
    ai_rows = [ln for ln in ai_content.split("\n")[1:] if ln]
    assert ai_rows[0].split("\t")[3] == "2.00"


def test_export_uses_paraver_precision() -> None:
    """The writer follows ParaverData.precision, not a hardcoded 2 dp: with a
    3/3/10 precision every numeric cell is formatted at exactly those decimals."""
    paraver = _paraver_data(_metric_trace(), precision=CsvPrecision(time=3, duration=3, value=10, header=6))
    content = _by_name(export_performance(paraver.trace, paraver))[PERFORMANCE_FILENAME]
    rows = [ln for ln in content.split("\n")[1:] if ln]
    assert rows[0] == "2.1\t1000000.000\t100000.000\t0.0000000160"


def test_empty_trace_returns_no_files() -> None:
    """Every facade returns () on an empty trace (nothing to export)."""
    empty = pd.DataFrame(columns=["thread_id", "time_s", "duration_s", "state_code", "flops", "bytes", "ai", "perf"])
    paraver = _paraver_data(empty)
    model = _assembled_roofline()
    assert export_performance(empty, paraver) == ()
    assert export_ai(empty, paraver) == ()
    assert export_ldst_percent(empty, paraver) == ()
    assert export_roof_labels(empty, paraver, model, 1) == ()
    assert export_region(empty, paraver, model, 1) == ()
    assert export_proximity(empty, paraver, model, 1) == ()


def test_ldst_percent_export_contract() -> None:
    """Full export_ldst_percent file: GRADIENT header with vmin:vmax from the
    values, exactly one file, 2 dp cells (default precision), natural thread
    order; store-only burst floored to 0.1, share 0.5 → 50%."""
    paraver = _paraver_data(_metric_trace())
    files = export_ldst_percent(paraver.trace, paraver)
    assert [f.name for f in files] == [LDST_PERCENT_FILENAME]
    content = files[0].content
    assert content.split("\n")[0].endswith(":window_in_null_gradient_mode:0.010000:50.000000")
    rows = [ln for ln in content.split("\n")[1:] if ln]
    assert len(rows) == 2
    assert rows[0].split("\t")[3] == "0.01"  # thread 2.1, store-only → floored to 0.01
    assert rows[1].split("\t")[3] == "50.00"  # thread 10.0, share 0.5 → 50%


def test_ldst_percent_zero_rows() -> None:
    """Exactly 0 (no 0.1 floor, vmin 0.0) when the burst has no memory ops or
    no FP activity — legacy two-frame masking semantics."""
    trace = _metric_trace()
    trace["load_share"] = [0.5, float("nan")]  # thread 2.1 has no memory ops
    paraver = _paraver_data(trace)
    files = export_ldst_percent(trace, paraver)
    content = files[0].content
    assert content.split("\n")[0].endswith(":window_in_null_gradient_mode:0.000000:50.000000")
    rows = [ln for ln in content.split("\n")[1:] if ln]
    assert rows[0].split("\t")[3] == "0.00"  # no memory ops → exactly 0, no floor

    trace2 = _metric_trace()
    trace2["flops"] = [32.0, 0.0]  # thread 2.1 has no FP activity
    paraver2 = _paraver_data(trace2)
    files2 = export_ldst_percent(trace2, paraver2)
    rows2 = [ln for ln in files2[0].content.split("\n")[1:] if ln]
    assert rows2[0].split("\t")[3] == "0.00"  # no FP → masked to 0 despite the store-only share


# ── Roof-label exports ────────────────────────────────────────────────────────


def test_roof_label_codes() -> None:
    """Code per row: 4/3/2/1 by the DRAM→L1 walk, 6 above L1, 0 on non-positive;
    code 5 is never emitted."""
    model = _assembled_roofline()
    ai = pd.Series([1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 2.0])
    perf = pd.Series([80e9, 30e9, 20e9, 10e9, 60e9, 50e9, 0.0])
    codes = roof_label_codes(ai, perf, model, 1)
    assert list(codes) == [1, 2, 3, 4, 6, 0, 0]
    assert 5 not in set(codes)


def test_roof_labels_export_files() -> None:
    """Both files returned: CSV with CODE header 1.000000:6.000000 and 2-dp codes;
    exact legend."""
    paraver = _paraver_data(_metric_trace())
    files = export_roof_labels(paraver.trace, paraver, _assembled_roofline(), 1)
    by_name = _by_name(files)
    assert {f.name for f in files} == {ROOF_LABELS_FILENAME, ROOF_LABELS_LEGEND_FILENAME}
    csv = by_name[ROOF_LABELS_FILENAME]
    assert csv.split("\n")[0].endswith(":window_in_code_mode:1.000000:6.000000")
    for ln in csv.split("\n")[1:]:
        if not ln:
            continue
        code = ln.split("\t")[3]
        assert re.fullmatch(r"\d+\.\d\d", code)
        assert float(code) in {0.0, 1.0, 2.0, 3.0, 4.0, 6.0}
    assert by_name[ROOF_LABELS_LEGEND_FILENAME] == serialize_legend(ROOF_LABEL_LEGEND)


# ── Region + proximity exports ────────────────────────────────────────────────


def test_region_codes_and_guard() -> None:
    """1 below the L1 ridge, 2 between the ridges (inclusive), 3 above DRAM;
    a model without L1 bandwidth refuses the export."""
    ai = pd.Series([0.5, 2.0, 10.0])
    assert list(region_label_codes(ai, 1.0, 8.0)) == [1, 2, 3]
    assert list(region_label_codes(pd.Series([1.0, 8.0]), 1.0, 8.0)) == [2, 2]

    paraver = _paraver_data(_metric_trace())
    files = export_region(paraver.trace, paraver, _assembled_roofline(), 1)
    by_name = _by_name(files)
    assert {f.name for f in files} == {REGION_FILENAME, REGION_LEGEND_FILENAME}
    assert by_name[REGION_FILENAME].split("\n")[0].endswith(":window_in_code_mode:1.000000:3.000000")
    # L1 missing → refused (required level, legacy §3.3.2).
    assert export_region(paraver.trace, paraver, _assembled_roofline_without_l1(), 1) == ()


def test_proximity_export() -> None:
    """One file per present level (absent skipped), ratio min(perf/roof, 1.0),
    0.0 on non-positive rows, header 0.0:1.0."""
    trace = pd.DataFrame(
        {
            "thread_id": pd.Categorical(["1.1", "2.1", "3.1"]),
            "time_s": [0.0, 1.0, 2.0],
            "duration_s": [1.0, 1.0, 1.0],
            "state_code": [1.0, 1.0, 1.0],
            "flops": [0.0, 50e9, 50e9],
            "bytes": [0.0, 50e9, 0.0],
            "ai": [0.0, 1.0, 1.0],
            "perf": [50e9, 50e9, 0.0],
        }
    )
    paraver = _paraver_data(trace)
    files = export_proximity(trace, paraver, _assembled_roofline(), 1)
    by_name = _by_name(files)
    assert {f.name for f in files} == {"carm_rel_l1.csv", "carm_rel_l2.csv", "carm_rel_l3.csv", "carm_rel_dram.csv"}
    assert by_name["carm_rel_l1.csv"].split("\n")[0].endswith(":window_in_null_gradient_mode:0.000000:1.000000")
    l1_rows = [ln.split("\t") for ln in by_name["carm_rel_l1.csv"].split("\n")[1:] if ln]
    # ai=0 → 0.0; ai=1, perf=50 GFLOPS vs L1 roof 100 → 0.5; perf=0 → 0.0
    assert [r[3] for r in l1_rows] == ["0.00", "0.50", "0.00"]
    # L2 roof at ai=1 is 50 GFLOPS → ratio clamps to 1.0
    l2_rows = [ln.split("\t") for ln in by_name["carm_rel_l2.csv"].split("\n")[1:] if ln]
    assert l2_rows[1][3] == "1.00"
    # Absent level skipped: without L1 only l2/l3/dram files are written.
    without_l1 = export_proximity(trace, paraver, _assembled_roofline_without_l1(), 1)
    assert {f.name for f in without_l1} == {"carm_rel_l2.csv", "carm_rel_l3.csv", "carm_rel_dram.csv"}


# ── Disk write-back ───────────────────────────────────────────────────────────


def test_write_export_files(tmp_path: Path) -> None:
    """Writes each ExportFile with exact content, returns sorted absolute paths,
    and a repeat call with the same names silently overwrites."""
    written = write_export_files(
        [ExportFile(name="a.csv", content="one"), ExportFile(name="b.csv", content="two")], tmp_path
    )
    assert written == sorted([(tmp_path / "a.csv").resolve(), (tmp_path / "b.csv").resolve()])
    assert (tmp_path / "a.csv").read_text(encoding="utf-8") == "one"
    assert (tmp_path / "b.csv").read_text(encoding="utf-8") == "two"
    write_export_files([ExportFile(name="a.csv", content="ONE")], tmp_path)
    assert (tmp_path / "a.csv").read_text(encoding="utf-8") == "ONE"
    assert (tmp_path / "b.csv").read_text(encoding="utf-8") == "two"


def test_export_unknown_unit_header_round_trips(tmp_path: Path) -> None:
    """An exported metadata line for an empty-unit window writes ':Unknown:'; the
    resulting file must re-import (parse_paraver_header + load_window_csv) with
    the µs legacy default instead of raising."""
    header = build_csv_metadata_line(
        "/p/x.prv", "", GRADIENT_WINDOW_MODE, 0.0, 16.0, timestamp="20260803170420"
    )
    assert header == "#20260803170420:CSV:RUNAPP:/p/x.prv:Unknown:window_in_null_gradient_mode:0.000000:16.000000"
    assert parse_paraver_header(header).time_unit == "Unknown"
    path = tmp_path / "exported.csv"
    path.write_text(f"{header}\n1.1.1\t0.00\t1294.00\t0.00\n1.1.1\t1294.00\t100.00\t1.00\n", encoding="utf-8")
    frame = load_window_csv(path)
    assert frame["time_s"].tolist() == pytest.approx([0.0, 1294e-6])
    assert frame["duration_s"].tolist() == pytest.approx([1294e-6, 100e-6])


def test_proximity_guard_missing_compute_peaks() -> None:
    """Memory-only model (bandwidth present, no arithmetic records) refuses the
    proximity export with () instead of raising on the empty peak sequence."""
    records = [r for r in _roof_records() if r.get("type") == "memory"]
    model = assemble_roofline(records, RooflineFilter(isa="x86", num_threads=1, data_type="f32"))
    assert model.bandwidth_by_level  # bandwidths present...
    assert not model.peak_performance_by_op  # ...but no compute peaks
    paraver = _paraver_data(_metric_trace())
    assert export_proximity(paraver.trace, paraver, model, 1) == ()
