"""Loading of Paraver window CSVs, legend CSVs, and their ``#`` header lines.

Time values in Paraver CSVs are expressed in the trace's time unit, named in the
header line; every loader normalizes to seconds at load time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final, Literal, NamedTuple

import pandas as pd

from carm_roofline.core.error import UserError


class TraceRow(NamedTuple):
    """One trace-table row (TRACE_COLUMNS), as yielded by ``DataFrame.itertuples()``."""

    thread_id: object
    time_s: float
    duration_s: float
    state_code: float
    flops: float
    bytes: float
    ai: float
    perf: float
    load_share: float
    isa_scalar_pct: float
    isa_sse_pct: float
    isa_avx2_pct: float
    isa_avx512_pct: float


# Single source of truth: TraceRow's field order and names ARE the trace-table
# schema.  The runtime tuples below derive from it, so DataFrame columns and
# NamedTuple attributes cannot drift apart.
TRACE_COLUMNS: Final[tuple[str, ...]] = TraceRow._fields

# Physical column layout of Paraver window CSVs (external contract); it is exactly
# the leading TRACE_COLUMNS.
WINDOW_CSV_COLUMNS: Final[tuple[str, ...]] = ("thread_id", "time_s", "duration_s", "state_code")

# Columns the window CSV does not provide (appended as float64 NaN at load):
# trace columns minus window columns, by construction.
METRIC_COLUMNS: Final[tuple[str, ...]] = tuple(name for name in TRACE_COLUMNS if name not in WINDOW_CSV_COLUMNS)

# Type-safe column accessors for trace tables.  pandas-stubs cannot encode a column
# schema on DataFrame, so every column read funnels through these Literal-keyed
# accessors: the key is checked statically and the return type declares the Series
# dtype (column-name typos become static errors instead of runtime KeyErrors).
# A Literal cannot derive from a runtime tuple before Python 3.11 (PEP 646), so
# this static duplicate of TRACE_COLUMNS minus {thread_id, state_code} is pinned
# by test_paraver_loading.
MetricColumn = Literal[
    "time_s",
    "duration_s",
    "flops",
    "bytes",
    "ai",
    "perf",
    "load_share",
    "isa_scalar_pct",
    "isa_sse_pct",
    "isa_avx2_pct",
    "isa_avx512_pct",
]
TextColumn = Literal["legend_label", "legend_color"]


def trace_metric(trace: pd.DataFrame, column: MetricColumn) -> pd.Series[float]:
    """Typed read of one float64 trace column (timestamps + metrics)."""
    return trace[column]


def trace_text(trace: pd.DataFrame, column: TextColumn) -> pd.Series[str]:
    """Typed read of one text trace column (legend label/color; NaN when unmapped)."""
    return trace[column]


def trace_state_code(trace: pd.DataFrame) -> pd.Series[float]:
    """Typed read of the state_code column (float64; category-backed in final traces)."""
    return trace["state_code"]


# Header time-unit strings → seconds multiplier.
TIME_SCALE_FACTORS = {"seconds": 1.0, "milliseconds": 1e-3, "microseconds": 1e-6, "nanoseconds": 1e-9}

_LEGEND_LINE_RE = re.compile(r'^(-?[\d\.]+)(?:-(-?[\d\.]+))?\s+"([^"]+)"\s+(\d+),(\d+),(\d+)$')


def time_unit_to_seconds(unit: str | None) -> float:
    """Map a Paraver header time-unit string to seconds; missing/empty/'unknown'
    (any case, the exported form of an empty unit) → 1e-6 (µs, legacy default);
    any other unit not in :data:`TIME_SCALE_FACTORS` raises ValueError.
    """
    if unit is None or not unit.strip():
        return 1e-6
    key = unit.strip().lower()
    if key == "unknown":
        return 1e-6
    if key not in TIME_SCALE_FACTORS:
        raise ValueError(f"unknown time unit {unit!r}; expected one of {sorted(TIME_SCALE_FACTORS)}")
    return TIME_SCALE_FACTORS[key]


@dataclass(frozen=True)
class ParaverHeader:
    """Parsed Paraver CSV ``#`` header line."""

    timestamp: str  # leading '#' stripped
    prv_path: str  # header parts[3]
    time_unit: str  # header parts[4]; "" when absent
    window_mode: str  # header parts[5]; "window_in_code_mode" when absent


def parse_paraver_header(line: str) -> ParaverHeader:
    """Parse a Paraver CSV '#' header line.

    Raise ValueError unless the line starts with '#' and splits (':') into ≥4 fields.
    parts[0] = '#<timestamp>', parts[1] = 'CSV', parts[2] = 'RUNAPP',
    parts[3] = prv path, parts[4] = time unit, parts[5] = window mode.
    Missing/empty parts[4] default to ""; missing/empty parts[5] default to the
    legacy code-mode string; extra fields (vmin/vmax) are ignored.
    """
    if not line.startswith("#"):
        raise ValueError(f"not a Paraver header line (must start with '#'): {line!r}")
    parts = line.split(":")
    if len(parts) < 4:
        raise ValueError(f"malformed Paraver header line (expected ≥4 ':'-separated fields): {line!r}")
    return ParaverHeader(
        timestamp=parts[0][1:],
        prv_path=parts[3],
        time_unit=parts[4] if len(parts) > 4 else "",
        window_mode=parts[5] if len(parts) > 5 and parts[5] else CODE_WINDOW_MODE,
    )


def _read_header(path: str | Path) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.readline()


def load_window_csv(path: str | Path) -> pd.DataFrame:
    """Window CSV → trace-shaped frame.

    Requires the '#' header (raises ValueError when missing, since the unit comes
    from it). Scales time_s/duration_s to seconds and casts state_code to category;
    the metric columns are appended as float64 NaN.
    """
    first_line = _read_header(path)
    if not first_line.startswith("#"):
        raise ValueError(f"window CSV is missing its '#' header line: {path}")
    scale = time_unit_to_seconds(parse_paraver_header(first_line).time_unit)
    frame = pd.read_csv(
        path,
        sep="\t",
        skiprows=1,
        header=None,
        names=WINDOW_CSV_COLUMNS,
        dtype={"thread_id": "category"},
    )
    frame["time_s"] = frame["time_s"] * scale
    frame["duration_s"] = frame["duration_s"] * scale
    frame["state_code"] = frame["state_code"].astype("category")
    for column in METRIC_COLUMNS:
        frame[column] = float("nan")
    return frame[list(TRACE_COLUMNS)]


@dataclass(frozen=True)
class CsvPrecision:
    """Fixed decimal places mirrored from the input window CSV into exported CSVs.

    Paraver writes its window exports with a fixed number of decimals per column
    (observed: 2 for every data column, 6 for the header vmin/vmax); exported
    CARM files reproduce the input window's own precision so Paraver re-imports
    them exactly like its own round-trip files.
    """

    time: int = 2
    duration: int = 2
    value: int = 2
    header: int = 6


DEFAULT_CSV_PRECISION = CsvPrecision()


def _decimal_places(cell: str) -> int:
    """Decimal digits after the last '.' in a numeric CSV cell ('0.00'→2, '15'→0).

    Cells without a '.' (incl. scientific notation, which Paraver never writes)
    count as 0.
    """
    cell = cell.strip()
    if "." not in cell:
        return 0
    return len(cell.rsplit(".", 1)[1])


def window_csv_precision(path: str | Path) -> CsvPrecision:
    """Per-column decimal places of a window CSV.

    Data columns (Timestamp, Duration, value) get the max decimal count over the
    data rows, per column; the header precision comes from the vmin/vmax fields
    of the '#' header line (parts[6]/parts[7] after ':'-split). '#'-prefixed
    lines are never data. A file without data rows falls back to
    DEFAULT_CSV_PRECISION.
    """
    header_dp = DEFAULT_CSV_PRECISION.header
    time_dp = duration_dp = value_dp = 0
    saw_data = False
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                parts = stripped.split(":")
                if len(parts) >= 8:
                    header_dp = max(_decimal_places(parts[6]), _decimal_places(parts[7]))
                continue
            cells = stripped.split("\t")
            if len(cells) < 4:
                continue  # malformed line: ignore (load_window_csv would have failed already)
            saw_data = True
            time_dp = max(time_dp, _decimal_places(cells[1]))
            duration_dp = max(duration_dp, _decimal_places(cells[2]))
            value_dp = max(value_dp, _decimal_places(cells[3]))
    if not saw_data:
        return DEFAULT_CSV_PRECISION
    return CsvPrecision(time=time_dp, duration=duration_dp, value=value_dp, header=header_dp)


def load_legend_csv(path: str | Path) -> pd.DataFrame:
    """Legend CSV → DataFrame[code: float64, label: str, r/g/b: int64] sorted by code.

    One row per line; a range 'start-end "label" r,g,b' yields code=start,
    code_end=end (code_end == code when absent). Raises ValueError on the first
    malformed line (with line number) — an incomplete legend means an incomplete
    state mapping.
    """
    rows = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            match = _LEGEND_LINE_RE.match(stripped)
            if match is None:
                raise ValueError(f"malformed legend line {lineno}: {stripped!r}")
            code = float(match.group(1))
            rows.append(
                {
                    "code": code,
                    "code_end": float(match.group(2)) if match.group(2) else code,
                    "label": match.group(3),
                    "r": int(match.group(4)),
                    "g": int(match.group(5)),
                    "b": int(match.group(6)),
                }
            )
    legend = pd.DataFrame(rows, columns=["code", "code_end", "label", "r", "g", "b"])
    return legend.sort_values("code").reset_index(drop=True)


GRADIENT_WINDOW_MODE = "window_in_null_gradient_mode"
CODE_WINDOW_MODE = "window_in_code_mode"


class ParaverWindowMode(Enum):
    """Normalized Paraver window mode for trace-table plotting.

    ``CODE`` (semantic, legend-colored) is the default; ``GRADIENT`` is the
    null-gradient variant colored continuously over ``state_code``.
    """

    CODE = "code"
    GRADIENT = "gradient"

    @classmethod
    def from_header(cls, window_mode: str) -> ParaverWindowMode:
        """Map a Paraver header mode string; unknown modes raise ValueError."""
        if window_mode == CODE_WINDOW_MODE:
            return cls.CODE
        if window_mode == GRADIENT_WINDOW_MODE:
            return cls.GRADIENT
        raise ValueError(
            f"unknown Paraver window mode {window_mode!r}; expected {CODE_WINDOW_MODE!r} or {GRADIENT_WINDOW_MODE!r}"
        )


def default_legend_path(window_csv: Path) -> Path:
    """Derive the legend path for a window CSV (foo.csv -> foo.legend.csv)."""
    if window_csv.suffix != ".csv":
        raise UserError(f"cannot derive a legend CSV from {window_csv}: window CSV must end in '.csv'")
    return window_csv.with_suffix(".legend.csv")
