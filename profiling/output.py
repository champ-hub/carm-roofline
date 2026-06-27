"""Output writers for profiling results.

Produces both CSV (GUI-compatible applications format) and JSON (full
hierarchy preserving ranks/threads) output.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from output_utils import detail, info

from .aggregation import AggregatedPoint
from .config import AggregationMode
from .model import RunResults


def write_profile_results(
    run: RunResults,
    name: str,
    output_dir: Path,
    aggregation: AggregationMode,
    points: list[AggregatedPoint],
) -> None:
    """Write profiling results in both CSV and JSON formats.

    CSV is written to ``<output_dir>/<name>/applications.csv``
    (GUI-compatible legacy format).  JSON is written to
    ``<output_dir>/<name>/profile.json`` (full rank/thread hierarchy plus
    aggregated view).
    """
    write_applications_csv(points, name, output_dir, run)
    write_profile_json(run, name, output_dir, aggregation, points)


def write_applications_csv(
    points: list[AggregatedPoint],
    name: str,
    output_dir: Path,
    run: RunResults,
) -> None:
    """Write aggregated profiling results as GUI-compatible applications CSV.

    Each ``AggregatedPoint`` becomes one CSV row.

    CSV columns (matching ``gui_utils.read_application_csv_file``):
        Date, Method, Name, ISA, Precision, Threads,
        AI, GFLOPS, Bandwidth, Time
    """
    out_dir = output_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / "applications.csv"

    date_str = run.metadata.date or datetime.now().isoformat(timespec="seconds")
    method = run.metadata.method or "PAPI_HL"
    isa = run.metadata.isa or ""
    precision = run.metadata.precision or ""

    header = ["Date", "Method", "Name", "ISA", "Precision", "Threads", "AI", "GFLOPS", "Bandwidth", "Time"]

    file_exists = filepath.exists()
    with filepath.open("a" if file_exists else "w", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)

        for pt in points:
            ai = pt.arithmetic_intensity
            gflops = pt.flops_per_second / 1e9
            bw = pt.bandwidth
            writer.writerow(
                [
                    date_str,
                    method,
                    pt.label,
                    isa,
                    precision,
                    str(pt.num_threads),
                    f"{ai:.6f}",
                    f"{gflops:.6f}",
                    f"{bw:.6f}",
                    f"{pt.runtime_s:.6f}",
                ]
            )

    info(f"Applications CSV written: {filepath}")
    detail(f"Aggregation={points[0].label if points else ''}, {len(points)} point(s)")


def write_profile_json(
    run: RunResults,
    name: str,
    output_dir: Path,
    aggregation: AggregationMode,
    points: list[AggregatedPoint],
) -> None:
    """Write full profiling results as JSON.

    The JSON includes the original rank/thread/region hierarchy *and* an
    aggregated view using the selected aggregation strategy.
    """
    out_dir = output_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / "profile.json"

    output = {
        "format_version": "1.0",
        "aggregation": aggregation.value,
        "original": run.to_dict(),
        "aggregated": {
            "name": run.metadata.name,
            "points": [
                {
                    "label": pt.label,
                    "total_flops": pt.total_flops,
                    "total_bytes": pt.total_bytes,
                    "runtime_s": pt.runtime_s,
                    "num_ranks": pt.num_ranks,
                    "num_threads": pt.num_threads,
                    "num_regions": pt.num_regions,
                }
                for pt in points
            ],
        },
    }

    filepath.write_text(json.dumps(output, indent=2))
    info(f"Profile JSON written: {filepath}")
