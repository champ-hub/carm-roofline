"""Loading of Paraver window CSVs, legend CSVs, and their ``#`` header lines.

Time values in Paraver CSVs are expressed in the trace's time unit, named in the
header line; every loader normalizes to seconds at load time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, NamedTuple

import pandas as pd

from carm_roofline.core.error import UserError

# Trace-shaped column schema shared by the window frame and the final trace table.
TRACE_COLUMNS = ("thread_id", "time_s", "duration_s", "state_code", "flops", "bytes", "ai", "perf")
METRIC_COLUMNS = ("flops", "bytes", "ai", "perf")
WINDOW_CSV_COLUMNS = ("thread_id", "time_s", "duration_s", "state_code")

# Type-safe column accessors for trace tables.  pandas-stubs cannot encode a column
# schema on DataFrame, so every column read funnels through these Literal-keyed
# accessors: the key is checked statically and the return type declares the Series
# dtype (column-name typos become static errors instead of runtime KeyErrors).
MetricColumn = Literal["time_s", "duration_s", "flops", "bytes", "ai", "perf"]
TextColumn = Literal["legend_label", "legend_color"]


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
    """Map a Paraver header time-unit string to seconds; missing/empty → 1e-6 (µs,
    legacy default); a non-empty unit not in :data:`TIME_SCALE_FACTORS` raises
    ValueError.
    """
    if unit is None or not unit.strip():
        return 1e-6
    key = unit.strip().lower()
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
