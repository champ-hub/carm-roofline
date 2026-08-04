"""Counter CSV pipeline: render paramedir configs, parse counter CSVs, merge them
into bursts, and compute the trace metrics.

``run_paramedir`` executes the installed paramedir binary over a .prv trace with
the rendered counter configs; counter CSVs may alternatively be produced externally
and loaded with :func:`load_counter_data`.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from carm_roofline.paraver.counters import (
    INTEL_COUNTERS,
    CounterSpec,
    bytes_weights,
    counter_config_template,
    flops_weights,
    fp_names,
    memory_names,
)
from carm_roofline.paraver.loading import (
    TIME_SCALE_FACTORS,
    TRACE_COLUMNS,
    load_window_csv,
    parse_paraver_header,
    time_unit_to_seconds,
)

COUNTER_CSV_COLUMNS = ("thread_id", "time_s", "duration_s", "value")

_MERGE_KEYS = ("thread_id", "time_s", "duration_s")


def render_counter_config(spec: CounterSpec, time_unit: str) -> str:
    """Render the template config for one counter.

    Substitutes ``$COUNTER`` → spec.name, ``$TIME_UNIT`` → time_unit,
    ``$EVT_TYPE`` → str(spec.evt_type) (plain str.replace, in that order; the
    template body is brace-heavy so ``$`` placeholders only). After substitution,
    if any ``$`` remains in the rendered text, raise ValueError — catches template
    drift (unknown/new placeholder) instead of silently emitting a broken config.
    """
    rendered = (
        _load_template()
        .replace("$COUNTER", spec.name)
        .replace("$TIME_UNIT", time_unit)
        .replace("$EVT_TYPE", str(spec.evt_type))
    )
    if "$" in rendered:
        raise ValueError(
            "counter config template contains an unsubstituted placeholder; expected only "
            "$COUNTER, $TIME_UNIT, $EVT_TYPE — the template may have drifted"
        )
    return rendered


def _load_template() -> str:
    """Read the counter config template (the seam tests monkeypatch)."""
    path = counter_config_template()
    if not path.is_file():
        raise FileNotFoundError(f"counter config template not found: {path}")
    return path.read_text(encoding="utf-8")


def write_counter_configs(output_dir: str | Path, time_unit: str) -> list[Path]:
    """Render one config per :data:`INTEL_COUNTERS` entry into ``output_dir`` as
    '<name>.cfg', in registry order. ``time_unit`` ''/None/'Unknown' (any case) →
    "Microseconds" (legacy fallback); anything not in
    :data:`TIME_SCALE_FACTORS` raises ValueError (the write side is strict: a bogus
    ``window_units`` silently breaks the paramedir run). Returns the written paths.
    Strict: missing template or unwritable dir raises.
    """
    if not counter_config_template().is_file():
        raise FileNotFoundError(f"counter config template not found: {counter_config_template()}")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    unit = time_unit.strip()
    if unit.lower() in ("", "unknown"):
        unit = "Microseconds"
    elif unit.lower() not in TIME_SCALE_FACTORS:
        raise ValueError(
            f"unknown time unit {time_unit!r}; expected one of {sorted(TIME_SCALE_FACTORS)} "
            "or ''/'Unknown' for the Microseconds default"
        )
    else:
        unit = unit.lower().capitalize()  # e.g. milliseconds → Milliseconds (cfg enum casing)
    written = []
    for spec in INTEL_COUNTERS:
        path = directory / f"{spec.name}.cfg"
        path.write_text(render_counter_config(spec, unit), encoding="utf-8")
        written.append(path)
    return written


def run_paramedir(trace_path: str | Path, output_dir: str | Path, time_unit: str) -> None:
    """Run paramedir over the .prv trace with the rendered counter configs.

    Writes one '<name>.cfg' per registered counter into output_dir (via
    write_counter_configs), then invokes
    ``paramedir <prv> <cfg> <out.csv> ...`` with cwd=output_dir so each
    '<name>.csv' lands there (the explicit output file per cfg is required:
    paramedir's default output name is '<cfg>.mcr', not '<name>.csv').
    paramedir writes times in time_unit (the cfg window_units). Raises
    FileNotFoundError when the trace is missing or paramedir is not on PATH;
    RuntimeError when paramedir exits non-zero, with stderr included in the
    message. Returns None; missing per-counter outputs are legitimate (a
    counter absent from the trace yields no file) and are handled by
    load_counter_data / build_trace_table.
    """
    trace = Path(trace_path)
    if not trace.is_file():
        raise FileNotFoundError(f"trace file not found: {trace}")
    if shutil.which("paramedir") is None:
        raise FileNotFoundError(
            "paramedir not found on PATH; add Paraver's bin/ directory (e.g. export PATH=/path/to/paraver/bin:$PATH)"
        )
    directory = Path(output_dir).resolve()
    cfgs = write_counter_configs(directory, time_unit)
    argv = ["paramedir", str(trace)]
    for cfg in cfgs:
        argv.extend([str(cfg), cfg.with_suffix(".csv").name])
    try:
        subprocess.run(argv, cwd=str(directory), check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() if exc.stderr else ""
        raise RuntimeError(f"paramedir failed (exit {exc.returncode}): {detail}") from exc


def parse_counter_csv(path: str | Path, time_unit: str | None = None) -> pd.DataFrame:
    """Counter CSV (same 4-column tab format as the window CSV) → frame with columns
    ['thread_id' (category), 'time_s', 'duration_s', 'value'] in SECONDS.

    ``time_unit`` (case-insensitive) is the unit the file's times are expressed in;
    when None, the unit is taken from the '#' header line when present, else
    microseconds (the pinned legacy default). A first line starting with '#' is
    always treated as a header line, regardless of the unit source.
    """
    with open(path, encoding="utf-8") as fh:
        first_line = fh.readline()
    has_header = first_line.startswith("#")
    if time_unit is None and has_header:
        time_unit = parse_paraver_header(first_line).time_unit
    scale = time_unit_to_seconds(time_unit)
    frame = pd.read_csv(
        path,
        sep="\t",
        skiprows=1 if has_header else 0,
        header=None,
        names=COUNTER_CSV_COLUMNS,
        dtype={"thread_id": "category"},
    )
    frame["time_s"] = frame["time_s"] * scale
    frame["duration_s"] = frame["duration_s"] * scale
    return frame


def load_counter_data(counter_csv_dir: str | Path, time_unit: str | None = None) -> dict[str, pd.DataFrame]:
    """For each :data:`INTEL_COUNTERS` name, parse '<name>.csv' when present and
    rename 'value' → name. ``time_unit`` is forwarded to
    :func:`parse_counter_csv` (None → per-file header unit / µs default). Returns
    only the found counters (missing ones are the merge step's business).
    """
    directory = Path(counter_csv_dir)
    result: dict[str, pd.DataFrame] = {}
    for spec in INTEL_COUNTERS:
        path = directory / f"{spec.name}.csv"
        if path.is_file():
            result[spec.name] = parse_counter_csv(path, time_unit).rename(columns={"value": spec.name})
    return result


def merge_counter_frames(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine counter frames into one burst frame [thread_id, time_s, duration_s,
    <each registry counter>] with every registered counter as a column (registry
    order; zero-filled 0.0 when absent from *frames*). Fast path: if all present
    frames have identical key columns (df[key_cols].equals), column-stack via
    pd.concat(axis=1); else outer-merge on the 3 keys then fillna(0) (legacy
    behavior). Raise ValueError when *frames* is empty. Sort result by
    ('time_s', 'thread_id').
    """
    if not frames:
        raise ValueError("merge_counter_frames: no counter frames to merge")
    names = [spec.name for spec in INTEL_COUNTERS]
    present = {name: frames[name] for name in names if name in frames}
    if not present:
        raise ValueError("merge_counter_frames: no registered counter frames to merge")
    keys = list(_MERGE_KEYS)
    # Union categorical thread_id categories across all present frames so a counter
    # firing on a subset of threads doesn't drop the combined key to object dtype
    # (which would break the later merge_asof(by="thread_id")).
    cat_names = [n for n in present if isinstance(present[n]["thread_id"].dtype, pd.CategoricalDtype)]
    if cat_names:
        categories = present[cat_names[0]]["thread_id"].cat.categories
        for n in cat_names[1:]:
            categories = categories.union(present[n]["thread_id"].cat.categories)
        for n in cat_names:
            present[n] = present[n].assign(thread_id=present[n]["thread_id"].cat.set_categories(categories))
    first = present[next(iter(present))]
    if all(frame[keys].equals(first[keys]) for frame in present.values()):
        combined = pd.concat([first[keys]] + [present[name][[name]] for name in present], axis=1)
    else:
        combined = first[keys].copy()
        for _, frame in present.items():
            combined = pd.merge(combined, frame, on=keys, how="outer")
        present_cols = [name for name in names if name in combined.columns]
        combined[present_cols] = combined[present_cols].fillna(0.0)
    for name in names:
        if name not in combined.columns:
            combined[name] = 0.0
    return combined[keys + names].sort_values(["time_s", "thread_id"]).reset_index(drop=True)


def compute_trace_metrics(bursts: pd.DataFrame) -> pd.DataFrame:
    """Vectorized metric computation; drops raw counters. Input: the
    :func:`merge_counter_frames` output.

    Formulas (mirroring the reference per-row loop Paraver_CARM.py:852-880):
        fp_inst   = Σ bursts[fp_names]                          (per row)
        flops     = Σ bursts[fp_names] x flops_weights          (per row)
        mem_ops   = bursts[memory_names].sum(axis=1)            (loads + stores)
        bytes_mod = (Σ bursts[fp_names] x bytes_weights) / fp_inst, 0.0 where fp_inst == 0
        bytes     = mem_ops x bytes_mod
        ai        = flops / bytes,   0.0 where bytes == 0
        perf      = flops / duration_s, 0.0 where duration_s == 0
    Rows with fp_inst == 0 get flops=0, bytes=0, ai=0, perf=0 (legacy zero-default).
    Returns columns ('thread_id', 'time_s', 'duration_s', 'flops', 'bytes', 'ai',
    'perf') — state_code is attached by :func:`attach_state_codes` next.
    """
    fp_cols = list(fp_names)
    fp_inst = bursts[fp_cols].sum(axis=1)
    flops = (bursts[fp_cols] * pd.Series(flops_weights, index=fp_names)).sum(axis=1)
    mem_ops = bursts[list(memory_names)].sum(axis=1)
    bytes_mod = (bursts[fp_cols] * pd.Series(bytes_weights, index=fp_names)).sum(axis=1) / fp_inst.where(fp_inst != 0)
    bytes_ = mem_ops * bytes_mod.fillna(0.0)
    ai = (flops / bytes_.where(bytes_ != 0)).fillna(0.0)
    perf = (flops / bursts["duration_s"].where(bursts["duration_s"] != 0)).fillna(0.0)
    return pd.DataFrame(
        {
            "thread_id": bursts["thread_id"],
            "time_s": bursts["time_s"],
            "duration_s": bursts["duration_s"],
            "flops": flops,
            "bytes": bytes_,
            "ai": ai,
            "perf": perf,
        }
    )


def attach_state_codes(bursts: pd.DataFrame, window: pd.DataFrame) -> pd.DataFrame:
    """Attach the state active at each burst's start: per-thread backward merge_asof
    on time_s. Both frames are sorted by time_s (merge_asof requires the on-key
    sorted globally); categorical thread keys are given union categories so threads
    absent from one side still merge to NaN. Bursts whose start exceeds the matched
    state's end (time_s > state_time_s + state_duration_s) get state_code NaN.
    state_code comes out float64 (NaN-safe). Returns bursts plus the state_code
    column.
    """
    left = bursts.sort_values(["time_s", "thread_id"])
    right = (
        window[["thread_id", "time_s", "duration_s", "state_code"]]
        .astype({"state_code": float})
        .rename(columns={"time_s": "state_time_s", "duration_s": "state_duration_s"})
        .sort_values("state_time_s")
    )
    if isinstance(left["thread_id"].dtype, pd.CategoricalDtype) and isinstance(
        right["thread_id"].dtype, pd.CategoricalDtype
    ):
        categories = left["thread_id"].cat.categories.union(right["thread_id"].cat.categories)
        left = left.assign(thread_id=left["thread_id"].cat.set_categories(categories))
        right = right.assign(thread_id=right["thread_id"].cat.set_categories(categories))
    merged = pd.merge_asof(
        left,
        right,
        left_on="time_s",
        right_on="state_time_s",
        by="thread_id",
        direction="backward",
    )
    out_of_range = merged["time_s"] > merged["state_time_s"] + merged["state_duration_s"]
    merged.loc[out_of_range, "state_code"] = float("nan")
    return merged.drop(columns=["state_time_s", "state_duration_s"])


def _window_header_unit(window_csv: str | Path) -> str:
    """Parse the window CSV's '#' header line and return its time unit."""
    with open(window_csv, encoding="utf-8") as fh:
        return parse_paraver_header(fh.readline()).time_unit


def build_trace_table(
    window_csv: str | Path, counter_csv_dir: str | Path, time_unit: str | None = None
) -> pd.DataFrame:
    """End-to-end pipeline: load_window_csv → load_counter_data (ValueError when the
    dir yields no counter CSVs) → merge_counter_frames → compute_trace_metrics →
    attach_state_codes → cast state_code to category → return df[TRACE_COLUMNS].
    Counter CSVs are interpreted in ``time_unit`` when given; otherwise in the
    window CSV's header unit (the unit paramedir was asked to write).
    """
    window = load_window_csv(window_csv)
    if time_unit is None:
        time_unit = _window_header_unit(window_csv)
    frames = load_counter_data(counter_csv_dir, time_unit)
    if not frames:
        raise ValueError(f"no counter CSVs found in {counter_csv_dir}")
    bursts = merge_counter_frames(frames)
    trace = attach_state_codes(compute_trace_metrics(bursts), window)
    trace["state_code"] = trace["state_code"].astype("category")
    return trace[list(TRACE_COLUMNS)]
