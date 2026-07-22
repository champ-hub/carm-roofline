"""Parse ``perf stat -x,`` CSV output into ``RegionMetrics`` for the profiling pipeline.

Supports two output formats produced by ``perf stat``:

- **Full-run mode** (``perf stat -x, -e <events> -o <out_csv>``): a single timing
  interval with each row containing ``<count>,<unit>,<event>,...``.
- **Interval mode** (``perf stat -x, -I <ms> -e <events> -o <out_csv>``): multiple
  timing intervals with rows of ``<timestamp_s>,<count>,<unit>,<event>,...``.

In both modes, each row's event name maps to a counter value.  Interval-mode
counters are raw per-interval deltas (perf ``-I`` already outputs interval-only
values).  Full-run counters are totals for the entire run.

All rows are synthesized onto **rank 0, thread 0** - perf has no MPI rank concept.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from carm_roofline.output_utils import debug, detail, warn

from .model import RegionMetrics, ThreadMetrics

# Perf stat -x, CSV column indices for interval mode (with -I)
# Format: <timestamp_s>,<count>,<unit>,<event>,<scaled_count>,<percentage>,...
_COL_TIMESTAMP = 0
_COL_COUNT_INTERVAL = 1
_COL_EVENT_INTERVAL = 3

# Column indices for full-run mode (without -I)
# Format: <count>,<unit>,<event>,<scaled_count>,<percentage>,...
_COL_COUNT_FULLRUN = 0
_COL_EVENT_FULLRUN = 2


def _is_interval_format(rows: list[list[str]]) -> bool:
    """Detect whether the CSV is in interval or full-run format.

    Interval format has a float timestamp in the first column; full-run
    has an integer (counter value) in the first column.
    """
    if not rows:
        return False
    first_col = rows[0][_COL_TIMESTAMP].strip()
    if not first_col:
        return False
    # If the first field can be parsed as a float >= 0 with decimals, it's a timestamp
    try:
        val = float(first_col)
        return val >= 0 and "." in first_col
    except ValueError:
        return False


def parse_perf_csv(filepath: str | Path, interval_ms: int | None = None) -> list[RegionMetrics]:
    """Parse a perf stat CSV output file into a list of ``RegionMetrics``.

    The function auto-detects interval vs. full-run format unless
    *interval_ms* is explicitly provided.

    Args:
        filepath: Path to the CSV file written by ``perf stat -x,``.
        interval_ms: Known interval in ms.  When ``None``, auto-detects
            format from the first column content.

    Returns:
        A list of ``RegionMetrics`` - one per timestamp interval (interval mode)
        or a single entry (full-run mode).
    """
    path = Path(filepath)
    if not path.is_file():
        warn(f"Perf output file not found: {path}")
        return []

    try:
        text = path.read_text()
    except OSError as exc:
        warn(f"Failed to read perf output file {path}: {exc}")
        return []

    return _parse_perf_csv_text(text, interval_ms)


def _parse_perf_csv_text(text: str, interval_ms: int | None = None) -> list[RegionMetrics]:
    """Parse perf CSV text content.

    Args:
        text: Raw CSV content from a ``perf stat -x,`` output file.
        interval_ms: Known interval in ms.  When ``None``, auto-detects.

    Returns:
        List of ``RegionMetrics``.
    """
    # Skip comment lines (starting with #)
    lines = [line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    if not lines:
        warn("Perf output file is empty or contains only comments")
        return []

    reader = csv.reader(io.StringIO("\n".join(lines)))
    rows = [row for row in reader if row and row[0].strip()]

    if not rows:
        return []

    is_interval = _is_interval_format(rows)

    if is_interval:
        return _parse_interval_mode(rows, interval_ms)
    else:
        return _parse_full_run_mode(rows)


def _parse_full_run_mode(rows: list[list[str]]) -> list[RegionMetrics]:
    """Parse full-run mode CSV (no -I flag).

    All rows belong to a single "total" region.  Each row contributes one
    counter entry.
    """
    counters: dict[str, int] = {}
    for row in rows:
        if len(row) <= _COL_EVENT_FULLRUN:
            continue
        count_str = row[_COL_COUNT_FULLRUN].strip()
        event = row[_COL_EVENT_FULLRUN].strip()
        if not count_str or not event:
            continue
        try:
            counters[event] = int(count_str)
        except ValueError:
            try:
                # Some events (e.g., task-clock) may have fractional counts
                counters[event] = int(float(count_str))
            except ValueError:
                detail(f"Skipping non-integer count for event '{event}': {count_str}")
                continue
    # Extract wall-clock time from duration_time (nanoseconds, always integer).
    time_nsec = counters.pop("duration_time", 0)

    if not counters:
        warn("No usable counters found in full-run perf output")
        return []

    debug(f"Parsed {len(counters)} perf events from full-run output")
    return [
        RegionMetrics(
            name="total",
            parent_region_id="-1",
            cycles=0,
            time_nsec=time_nsec,
            counters=counters,
        )
    ]


def _parse_interval_mode(rows: list[list[str]], interval_ms: int | None) -> list[RegionMetrics]:
    """Parse interval-mode CSV (with -I flag).

    Groups rows by timestamp - each timestamp becomes one ``RegionMetrics``
    with a synthetic name ``"sample_{n}"``.

    Interval mode gives per-interval (delta) counter values, so we use
    the value directly without computing diffs.
    """
    # Group rows by timestamp
    groups: dict[str, dict[str, int]] = {}
    timestamps: list[str] = []  # preserve order

    for row in rows:
        if len(row) <= _COL_EVENT_INTERVAL:
            continue
        ts = row[_COL_TIMESTAMP].strip()
        count_str = row[_COL_COUNT_INTERVAL].strip()
        event = row[_COL_EVENT_INTERVAL].strip()
        if not ts or not count_str or not event:
            continue
        try:
            count = int(count_str)
        except ValueError:
            try:
                count = int(float(count_str))
            except ValueError:
                detail(f"Skipping non-integer count for event '{event}': {count_str}")
                continue

        if ts not in groups:
            groups[ts] = {}
            timestamps.append(ts)
        groups[ts][event] = count

    if not timestamps:
        warn("No usable data found in interval-mode perf output")
        return []

    # Compute interval duration in nanoseconds
    # If we know interval_ms explicitly, use that. Otherwise derive from timestamp diffs.
    if interval_ms is not None:
        interval_ns = int(interval_ms * 1_000_000)
        fixed_interval = True
    else:
        interval_ns = _derive_interval_ns(timestamps)
        fixed_interval = False

    regions: list[RegionMetrics] = []
    for i, ts in enumerate(timestamps):
        # Time_nsec: use interval duration.  For the first sample, use the timestamp
        # value directly (time since start).  For subsequent samples, use the interval.
        if i == 0:
            try:
                time_ns = int(float(ts) * 1_000_000_000)
            except ValueError:
                time_ns = interval_ns
        else:
            try:
                prev_t = float(timestamps[i - 1])
                cur_t = float(ts)
                time_ns = int((cur_t - prev_t) * 1_000_000_000)
            except ValueError:
                time_ns = interval_ns

        # Pop timing events - not performance counters
        counters_sample = groups[ts]
        counters_sample.pop("duration_time", None)
        counters_sample.pop("task-clock", None)

        regions.append(
            RegionMetrics(
                name=f"sample_{i}",
                parent_region_id="-1",
                cycles=0,
                time_nsec=time_ns,
                counters=counters_sample,
            )
        )

    detail(
        f"Parsed {len(regions)} interval samples from perf output ({'fixed' if fixed_interval else 'derived'} interval)"
    )
    return regions


def _derive_interval_ns(timestamps: list[str]) -> int:
    """Derive the sampling interval in nanoseconds from consecutive timestamps.

    Falls back to 1 ms (1_000_000 ns) if timestamps can't be parsed.
    """
    if len(timestamps) < 2:
        return 1_000_000  # default 1ms
    try:
        t0 = float(timestamps[0])
        t1 = float(timestamps[1])
        diff_s = t1 - t0
        if diff_s > 0:
            return int(diff_s * 1_000_000_000)
    except (ValueError, IndexError):
        pass
    return 1_000_000


def perf_csv_to_thread_metrics(filepath: str | Path, interval_ms: int | None = None) -> ThreadMetrics | None:
    """Parse perf output and wrap it as a single-thread ``ThreadMetrics``.

    Convenience function for integration with the pipeline.
    """
    regions = parse_perf_csv(filepath, interval_ms)
    debug(f"regions: {regions}")
    if not regions:
        return None
    return ThreadMetrics(thread_id=0, regions=regions)
