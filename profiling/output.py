"""Output writers for profiling results.

Produces CSV (GUI-compatible applications format) and JSONL (one appended
line per run, embedding metadata and the aggregated points) output.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime

from output_utils import detail, info

from .aggregation import AggregatedPoint
from .config import ProfileConfig
from .model import RunResults


def write_profile_results(
    run: RunResults,
    config: ProfileConfig,
    points: list[AggregatedPoint],
) -> None:
    """Write profiling results in both CSV and JSONL formats.

    CSV is written to ``<output_dir>/<machine_name>/applications.csv``
    (GUI-compatible legacy format).  JSONL is written to
    ``<output_dir>/<machine_name>/applications.jsonl`` (one appended line per run,
    embedding run metadata and the aggregated points).
    """
    write_applications_csv(points, config, run)
    write_profile_jsonl(run, config, points)


def write_applications_csv(
    points: list[AggregatedPoint],
    config: ProfileConfig,
    run: RunResults,
) -> None:
    """Write aggregated profiling results as GUI-compatible applications CSV.

    The CSV is appended to ``<output_dir>/<machine_name>/applications.csv``.
    """
    out_dir = config.output_dir / config.machine_name
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
            writer.writerow(
                (
                    date_str,
                    method,
                    pt.label,
                    isa,
                    precision,
                    pt.num_threads,
                    f"{pt.arithmetic_intensity:.3f}",
                    f"{pt.flops_per_second / 1e9:.3f}",
                    f"{pt.bandwidth / 1e9:.3f}",
                    f"{pt.runtime_s:.3f}",
                )
            )

    info(f"Applications CSV written: {filepath}")
    detail(f"Aggregation={points[0].label if points else ''}, {len(points)} point(s)")


def write_profile_jsonl(
    run: RunResults,
    config: ProfileConfig,
    points: list[AggregatedPoint],
) -> None:
    """Write profiling results as JSONL.

    Produces one appended JSON line per run, embedding run metadata and
    the aggregated roofline points.  Written to
    ``<output_dir>/<machine_name>/applications.jsonl``.
    """
    out_dir = config.output_dir / config.machine_name
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / "applications.jsonl"

    record = {
        "format_version": "2.0",
        "aggregation": config.aggregation.value,
        "metadata": {
            "name": run.metadata.name,
            "date": run.metadata.date,
            "method": run.metadata.method,
            "isa": run.metadata.isa,
            "precision": run.metadata.precision,
            "threads_per_rank": run.metadata.threads_per_rank,
            "command": run.metadata.command,
            "notes": run.metadata.notes,
            "num_ranks": run.num_ranks,
            "total_threads": run.total_threads,
        },
        "points": [
            {
                "label": pt.label,
                "total_flops": pt.total_flops,
                "total_bytes": pt.total_bytes,
                "runtime_s": pt.runtime_s,
                "num_ranks": pt.num_ranks,
                "num_threads": pt.num_threads,
                "num_regions": pt.num_regions,
                "arithmetic_intensity": pt.arithmetic_intensity,
                "flops_per_second": pt.flops_per_second,
                "bandwidth": pt.bandwidth,
            }
            for pt in points
        ],
    }

    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True))
        f.write("\n")

    info(f"Profile JSONL written: {filepath}")
    detail(f"Aggregation={config.aggregation.value}, {len(points)} point(s)")
