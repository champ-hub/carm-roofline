"""Run merging machinery for partitioned profiling runs.

When a required event pool cannot be counted in a single run, the pool is
partitioned into disjoint feasible chunks (one app run per chunk) and the
per-run results merge at the raw region-counter level before metric
computation. This module provides:

- :func:`partition_events` — capability-driven, deterministic partitioning.
- :func:`merge_runs` — structural merge of per-run results into one run.
- :func:`missing_required_events` — coverage check helper.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any, Callable

from carm_roofline.output_utils import warn

from .model import RankMetrics, RegionMetrics, RunMetadata, RunResults, ThreadMetrics


def partition_events(
    events: Iterable[str],
    validator: Callable[[frozenset[str]], bool],
) -> list[list[str]]:
    """Partition *events* into disjoint feasible runs.

    The whole pool is checked first — when everything fits, a single run
    results (the common case, one validator call). Otherwise a greedy pass
    adds events one at a time in sorted order to a run while the validator
    accepts the candidate; each run is a maximal feasible subset. An event
    the validator rejects even alone is collected in its own run with a
    warning; termination is guaranteed because every iteration removes at
    least one event.

    Args:
        events: Event pool to partition (deduplicated; order ignored).
        validator: Returns True when the given event set can be collected
            together in one run.

    Returns:
        List of disjoint, sorted event lists; each list is one app run.
    """
    remaining = set(events)
    if not remaining:
        return []

    if validator(frozenset(remaining)):
        return [sorted(remaining)]

    chunks: list[list[str]] = []
    while remaining:
        run: set[str] = set()
        for e in sorted(remaining):
            if validator(frozenset(run | {e})):
                run.add(e)
        if not run:
            alone = sorted(remaining)[0]
            warn(f"Event {alone} cannot be counted together with any other event; collecting it alone")
            run = {alone}
        chunks.append(sorted(run))
        remaining -= run
    return chunks


def _check_region_structure(runs: list[RunResults]) -> None:
    """Verify every run has the same rank/thread/region structure as run 0.

    Ranks pair by position (rank count must match; ``rank_id`` is an
    execution label — PAPI HL names rank files per-process, so ids differ
    across runs of the same app). Per rank the thread_id sequence must
    match; per thread the region ``(name, parent_region_id)`` sequences are
    compared by position.

    Raises:
        RuntimeError: On any structural mismatch (rank count, thread_id
            sequence, or per-thread region sequence).
    """
    base = runs[0]
    for i, run in enumerate(runs[1:], start=1):
        if len(run.ranks) != len(base.ranks):
            r = min(len(run.ranks), len(base.ranks))
            raise RuntimeError(f"Run {i} region structure differs from run 0 at rank {r} thread 0")
        for _r, (base_rank, rank) in enumerate(zip(base.ranks, run.ranks)):
            if len(rank.threads) != len(base_rank.threads):
                t = min(len(rank.threads), len(base_rank.threads))
                raise RuntimeError(
                    f"Run {i} region structure differs from run 0 at rank {base_rank.rank_id} thread {t}"
                )
            for t, (base_thread, thread) in enumerate(zip(base_rank.threads, rank.threads)):
                if thread.thread_id != base_thread.thread_id:
                    raise RuntimeError(
                        f"Run {i} region structure differs from run 0 at rank {base_rank.rank_id} thread {t}"
                    )
                if len(thread.regions) != len(base_thread.regions):
                    raise RuntimeError(
                        f"Run {i} region structure differs from run 0 at rank {base_rank.rank_id} "
                        f"thread {base_thread.thread_id}"
                    )
                for base_region, region in zip(base_thread.regions, thread.regions):
                    if (region.name, region.parent_region_id) != (base_region.name, base_region.parent_region_id):
                        raise RuntimeError(
                            f"Run {i} region structure differs from run 0 at rank {base_rank.rank_id} "
                            f"thread {base_thread.thread_id}"
                        )


def merge_runs(runs: list[RunResults]) -> RunResults:
    """Merge per-run results into one run at the raw region-counter level.

    Structural contract (deterministic app): every run has the same number
    of ranks, paired by position (``rank_id`` is an execution label and may
    differ — PAPI HL names rank files per-process); per rank the same
    thread_id sequence; per thread the region ``(name, parent_region_id)``
    sequences are identical (paired by position). Any mismatch raises a
    :class:`RuntimeError`.

    Merged region counters are the union of per-run counters; an event key
    already present warns and keeps the first value (runs are disjoint by
    construction, so this branch should not trigger). ``cycles`` and
    ``time_nsec`` come from run 0 (identical executions; time is never
    summed). ``RankMetrics.event_definitions`` is the dict union across runs.

    Args:
        runs: Per-run results to merge (at least 2).

    Returns:
        Merged :class:`RunResults` with fresh metadata (new date, notes
        ``"merged from N runs"``).
    """
    if len(runs) < 2:
        raise RuntimeError("merge_runs requires at least 2 runs")
    _check_region_structure(runs)
    base = runs[0]

    merged_ranks: list[RankMetrics] = []
    for r, base_rank in enumerate(base.ranks):
        threads: list[ThreadMetrics] = []
        for t, base_thread in enumerate(base_rank.threads):
            regions: list[RegionMetrics] = []
            for rg, base_region in enumerate(base_thread.regions):
                counters = dict(base_region.counters)
                for other in runs[1:]:
                    other_region = other.ranks[r].threads[t].regions[rg]
                    for event, value in other_region.counters.items():
                        if event in counters:
                            warn(f"Event {event} collected in multiple runs; keeping first value")
                        else:
                            counters[event] = value
                regions.append(
                    RegionMetrics(
                        name=base_region.name,
                        parent_region_id=base_region.parent_region_id,
                        cycles=base_region.cycles,
                        time_nsec=base_region.time_nsec,
                        counters=counters,
                    )
                )
            threads.append(ThreadMetrics(thread_id=base_thread.thread_id, regions=regions))
        event_definitions: dict[str, Any] = {}
        for run in runs:
            event_definitions.update(run.ranks[r].event_definitions)
        merged_ranks.append(
            RankMetrics(rank_id=base_rank.rank_id, event_definitions=event_definitions, threads=threads)
        )

    metadata = RunMetadata(
        name=base.metadata.name,
        date=datetime.now().isoformat(timespec="seconds"),
        method=base.metadata.method,
        isa=base.metadata.isa,
        precision=base.metadata.precision,
        threads_per_rank=base.metadata.threads_per_rank,
        command=base.metadata.command,
        notes=f"merged from {len(runs)} runs",
    )
    return RunResults(metadata=metadata, ranks=merged_ranks)


def missing_required_events(run: RunResults, required: set[str]) -> set[str]:
    """Return *required* events absent from every region counter in *run*.

    The union of all region counters across ranks/threads/regions is used, so
    an event collected in any region counts as collected.

    Args:
        run: Merged (or single-run) results.
        required: Events the resolved metrics need.

    Returns:
        Set of required events that were not collected anywhere.
    """
    collected: set[str] = set()
    for rank in run.ranks:
        for thread in rank.threads:
            for region in thread.regions:
                collected |= set(region.counters.keys())
    return set(required) - collected
