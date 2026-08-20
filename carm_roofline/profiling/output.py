"""Output writers for profiling results.

Produces JSONL (one appended line per run, embedding metadata and the
aggregated points) and a machine-signature debug file.
"""

from __future__ import annotations

import json

from carm_roofline.architecture import write_machine_json
from carm_roofline.output_utils import detail, info

from .aggregation import AggregatedPoint
from .config import ProfileConfig
from .model import RunResults


def write_profile_results(
    run: RunResults,
    config: ProfileConfig,
    points: list[AggregatedPoint],
) -> None:
    """Write profiling results in JSONL format.

    JSONL is written to ``<output_dir>/<machine_name>/applications.jsonl``
    (one appended line per run, embedding run metadata and the aggregated
    points). A ``machine.json`` debugging file is also written to
    ``<output_dir>/<machine_name>/machine.json`` on the first run.
    """
    machine_dir = config.output_dir / config.machine_name
    write_machine_json(config.machine_signature, machine_dir)
    write_profile_jsonl(run, config, points)


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
        "format_version": "3.0",
        "aggregation": config.aggregation.value,
        "optional_metrics": sorted({name for pt in points for name in pt.optional_bytes}),
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
                "optional_bytes": pt.optional_bytes,
                "optional_fractions": pt.optional_fractions,
            }
            for pt in points
        ],
    }

    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True))
        f.write("\n")

    info(f"Profile JSONL written: {filepath}")
    detail(f"Aggregation={config.aggregation.value}, {len(points)} point(s)")
