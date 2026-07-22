"""File discovery and parsing for PAPI HL profiling result files.

Expected file naming: ``rank_{NNNNN}.json`` (zero-padded).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from carm_roofline.output_utils import debug, detail, warn

from .model import RankMetrics, RegionMetrics, ThreadMetrics

_RANK_FILE_RE = re.compile(r"^rank_(\d+)\.json$")


def discover_rank_files(results_dir: Path) -> dict[int, Path]:
    """Scan *results_dir* for rank-specific profiling files.

    Returns a dict mapping rank ID to file path.  Files are matched by the
    pattern ``rank_{N}.json`` (e.g. ``rank_0.json``).
    """
    rank_files: dict[int, Path] = {}

    if not results_dir.is_dir():
        detail(f"Profile results directory does not exist: {results_dir}")
        return rank_files

    for child in sorted(results_dir.iterdir()):
        if not child.is_file():
            continue
        m = _RANK_FILE_RE.match(child.name)
        if m is None:
            continue
        rank_id = int(m.group(1))
        rank_files[rank_id] = child
        debug(f"Discovered file for rank {rank_id}: {child}")

    detail(f"Found {len(rank_files)} rank file(s) in {results_dir}")
    return rank_files


def _extract_rank_id(filepath: Path) -> int:
    """Extract rank ID from a file path using the regex.

    Returns 0 by default if the path does not match (should not happen in practice).
    """
    m = _RANK_FILE_RE.match(filepath.name)
    return int(m.group(1)) if m else 0


def parse_rank_file(filepath: Path) -> RankMetrics | None:
    """Parse a single PAPI HL JSON rank file into a ``RankMetrics``.

    Expected JSON structure (from ``PAPI_hl_output``)::

        {
          "papi_version": "...",
          "cpu_info": "...",
          "max_cpu_rate_mhz": "...",
          "min_cpu_rate_mhz": "...",
          "event_definitions": {
            "PAPI_FP_OPS": {"component": "perf_event", "type": "delta"},
            ...
          },
          "threads": {
            "0": {
              "regions": {
                "0": {
                  "name": "...",
                  "parent_region_id": <parent region id or "-1">,
                  "cycles": "...",
                  "real_time_nsec": "...",
                  "PAPI_FP_OPS": "20971520",
                  "PAPI_L1_DCA": "38637435"
                },
                ...
              }
            },
            ...
          }
        }

    Args:
        filepath: Path to the JSON rank file.

    Returns:
        A ``RankMetrics`` instance, or ``None`` if parsing fails.
    """
    try:
        data: dict[str, Any] = json.loads(filepath.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        warn(f"Failed to parse rank file {filepath}: {exc}")
        return None

    rank_id = _extract_rank_id(filepath)
    event_defs: dict[str, Any] = data.get("event_definitions", {})
    threads_data: dict[str, Any] = data.get("threads", {})
    threads: list[ThreadMetrics] = []

    for thread_key in sorted(threads_data, key=int):
        try:
            thread_entry = threads_data[thread_key]
            tid = int(thread_key)
            regions_data: dict[str, Any] = thread_entry.get("regions", {})
            regions: list[RegionMetrics] = []

            for region_key in sorted(regions_data, key=int):
                reg = regions_data[region_key]
                name: str = reg.get("name", f"region_{region_key}")
                parent: str = reg.get("parent_region_id", "-1")
                cycles: int = int(reg.get("cycles", 0))
                time_ns: int = int(reg.get("real_time_nsec", 0))

                # All remaining numeric fields are raw PAPI counters
                counters: dict[str, int] = {}
                for k, v in reg.items():
                    if k in ("name", "parent_region_id", "cycles", "real_time_nsec"):
                        continue
                    try:
                        counters[k] = int(v)
                    except ValueError:
                        warn(f"Non-integer counter value for {k}, region {name}, thread {tid} in {filepath}: {v}")
                        counters[k] = 0

                regions.append(
                    RegionMetrics(
                        name=name,
                        parent_region_id=parent,
                        cycles=cycles,
                        time_nsec=time_ns,
                        counters=counters,
                    )
                )

            if regions:
                threads.append(ThreadMetrics(thread_id=tid, regions=regions))

        except (ValueError, KeyError) as exc:
            warn(f"Skipping malformed thread entry {thread_key} in {filepath}: {exc}")
            continue

    if not threads:
        warn(f"No valid thread data found in {filepath}")
        return None

    return RankMetrics(
        rank_id=rank_id,
        event_definitions=event_defs,
        threads=threads,
    )


def load_all_ranks(results_dir: Path) -> list[RankMetrics]:
    """Discover and parse all PAPI HL rank files in *results_dir*.

    Returns a list of :class:`RankMetrics` objects sorted by rank ID.
    """
    rank_paths = discover_rank_files(results_dir)
    ranks: list[RankMetrics] = []

    for rank_id in sorted(rank_paths):
        path = rank_paths[rank_id]
        rank = parse_rank_file(path)
        if rank is not None:
            ranks.append(rank)
            debug(f"Loaded rank {rank_id}: {len(rank.threads)} thread(s) from {path.name}")

    return ranks
