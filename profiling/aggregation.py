"""Aggregation strategies for multi-rank profiling results.

Aggregation transforms a ``RunResults`` (rank → thread → region hierarchy
with raw counters) into one or more ``AggregatedPoint`` instances.

The four modes match the hierarchy:

- **GLOBAL**:   one point for the entire MPI job (all regions everywhere).
- **PER_RANK**:   one point per MPI rank.
- **PER_THREAD**:  one point per (rank, thread).
- **PER_REGION_MERGED**:  one point per unique region name across all ranks/threads.
- **PER_REGION_PER_THREAD**:  one point per (rank, thread, region); no cross-thread aggregation.

At aggregation time, raw PAPI counters are converted to roofline metrics
(flops, bytes, time_s) using the resolved ``MetricDefinition``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import AggregationMode
from .model import RegionMetrics, RunResults
from .shared import (
    MetricContext,
    MetricDefinition,
    MetricType,
    compute_region_point,
    sum_roofline_points,
)


@dataclass
class AggregatedPoint:
    """A single roofline point produced by aggregation."""

    label: str
    total_flops: float
    total_bytes: float
    runtime_s: float
    num_ranks: int
    num_threads: int
    num_regions: int = 0

    @property
    def arithmetic_intensity(self) -> float:
        return self.total_flops / self.total_bytes if self.total_bytes > 0 else 0.0

    @property
    def flops_per_second(self) -> float:
        return self.total_flops / self.runtime_s if self.runtime_s > 0 else 0.0

    @property
    def bandwidth(self) -> float:
        return self.total_bytes / self.runtime_s if self.runtime_s > 0 else 0.0


def _region_points(
    regions: list[RegionMetrics],
    resolved: dict[MetricType, MetricDefinition],
    metric_ctx: MetricContext,
) -> list[dict[str, float]]:
    """Compute (flops, bytes, time_s) for each region in a list."""
    return [compute_region_point(r.counters, r.time_nsec, resolved, metric_ctx) for r in regions]


def aggregate_global(
    run: RunResults,
    resolved: dict[MetricType, MetricDefinition],
    metric_ctx: MetricContext,
) -> AggregatedPoint:
    """Aggregate all regions from all ranks/threads into a single point.

    Flops/bytes are summed across all regions; runtime is the *max* time
    across all regions (regions within a thread execute sequentially,
    threads within a rank run in parallel).
    """
    all_points: list[dict[str, float]] = []
    for rank in run.ranks:
        for thread in rank.threads:
            all_points.extend(_region_points(thread.regions, resolved, metric_ctx))

    total = sum_roofline_points(all_points)
    num_threads = sum(len(r.threads) for r in run.ranks)
    return AggregatedPoint(
        label=run.metadata.name or "global",
        total_flops=total["flops"],
        total_bytes=total["bytes"],
        runtime_s=total["time_s"],
        num_ranks=run.num_ranks,
        num_threads=num_threads,
        num_regions=len(all_points),
    )


def aggregate_per_rank(
    run: RunResults,
    resolved: dict[MetricType, MetricDefinition],
    metric_ctx: MetricContext,
) -> list[AggregatedPoint]:
    """Aggregate each rank independently into its own roofline point.

    Within each rank, all thread regions are combined into a single point.
    """
    points: list[AggregatedPoint] = []
    for rank in run.ranks:
        rank_points: list[dict[str, float]] = []
        for thread in rank.threads:
            rank_points.extend(_region_points(thread.regions, resolved, metric_ctx))

        total = sum_roofline_points(rank_points)
        label = f"{run.metadata.name}_rank{rank.rank_id}" if run.metadata.name else f"rank{rank.rank_id}"
        points.append(
            AggregatedPoint(
                label=label,
                total_flops=total["flops"],
                total_bytes=total["bytes"],
                runtime_s=total["time_s"],
                num_ranks=1,
                num_threads=len(rank.threads),
                num_regions=len(rank_points),
            )
        )
    return points


def aggregate_per_thread(
    run: RunResults,
    resolved: dict[MetricType, MetricDefinition],
    metric_ctx: MetricContext,
) -> list[AggregatedPoint]:
    """Aggregate each (rank, thread) pair into its own roofline point."""
    points: list[AggregatedPoint] = []
    for rank in run.ranks:
        for thread in rank.threads:
            thread_points = _region_points(thread.regions, resolved, metric_ctx)
            total = sum_roofline_points(thread_points)
            label = (
                f"{run.metadata.name}_rank{rank.rank_id}_thread{thread.thread_id}"
                if run.metadata.name
                else f"rank{rank.rank_id}_thread{thread.thread_id}"
            )
            points.append(
                AggregatedPoint(
                    label=label,
                    total_flops=total["flops"],
                    total_bytes=total["bytes"],
                    runtime_s=total["time_s"],
                    num_ranks=1,
                    num_threads=1,
                    num_regions=len(thread_points),
                )
            )
    return points


def aggregate_per_region_merged(
    run: RunResults,
    resolved: dict[MetricType, MetricDefinition],
    metric_ctx: MetricContext,
) -> list[AggregatedPoint]:
    """Aggregate by region name across all ranks/threads.

    All occurrences of the same region name across all ranks and threads are summed into a single point.
    """
    by_name: dict[str, list[dict[str, float]]] = {}
    rank_ids: dict[str, set[int]] = {}
    thread_ids: dict[str, set[tuple[int, int]]] = {}

    for rank in run.ranks:
        for thread in rank.threads:
            for region in thread.regions:
                region_point = compute_region_point(region.counters, region.time_nsec, resolved, metric_ctx)
                by_name.setdefault(region.name, []).append(region_point)
                rank_ids.setdefault(region.name, set()).add(rank.rank_id)
                thread_ids.setdefault(region.name, set()).add((rank.rank_id, thread.thread_id))

    points: list[AggregatedPoint] = []
    for name, region_points in sorted(by_name.items()):
        total = sum_roofline_points(region_points)
        label = f"{run.metadata.name}_{name}" if run.metadata.name else name
        points.append(
            AggregatedPoint(
                label=label,
                total_flops=total["flops"],
                total_bytes=total["bytes"],
                runtime_s=total["time_s"],
                num_ranks=len(rank_ids.get(name, set())),
                num_threads=len(thread_ids.get(name, set())),
                num_regions=len(region_points),
            )
        )
    return points


def aggregate_per_region_per_thread(
    run: RunResults,
    resolved: dict[MetricType, MetricDefinition],
    metric_ctx: MetricContext,
) -> list[AggregatedPoint]:
    """One roofline point per (rank, thread, region); no cross-thread aggregation.

    Each region measurement of each thread becomes its own point, so duplicates of the same region name across
    threads/ranks stay distinct rather than being summed.
    """
    points: list[AggregatedPoint] = []
    for rank in run.ranks:
        for thread in rank.threads:
            for region in thread.regions:
                region_point = compute_region_point(region.counters, region.time_nsec, resolved, metric_ctx)
                label = (
                    f"{run.metadata.name}_rank{rank.rank_id}_thread{thread.thread_id}_{region.name}"
                    if run.metadata.name
                    else f"rank{rank.rank_id}_thread{thread.thread_id}_{region.name}"
                )
                points.append(
                    AggregatedPoint(
                        label=label,
                        total_flops=region_point["flops"],
                        total_bytes=region_point["bytes"],
                        runtime_s=region_point["time_s"],
                        num_ranks=1,
                        num_threads=1,
                        num_regions=1,
                    )
                )
    return points


def aggregate(
    run: RunResults,
    mode: AggregationMode,
    resolved: dict[MetricType, MetricDefinition],
    metric_ctx: MetricContext,
) -> list[AggregatedPoint]:
    """Dispatch to the appropriate aggregation strategy.

    Args:
        run: Full profiling results with raw region counters.
        mode: Aggregation strategy.
        resolved: Resolved metric implementations for computing flops/bytes.

    Returns:
        A list of one or more :class:`AggregatedPoint` instances.
    """
    if mode == AggregationMode.GLOBAL:
        return [aggregate_global(run, resolved, metric_ctx)]
    if mode == AggregationMode.RANK:
        return aggregate_per_rank(run, resolved, metric_ctx)
    if mode == AggregationMode.THREAD:
        return aggregate_per_thread(run, resolved, metric_ctx)
    if mode == AggregationMode.REGION_MERGED:
        return aggregate_per_region_merged(run, resolved, metric_ctx)
    if mode == AggregationMode.REGION_PER_THREAD:
        return aggregate_per_region_per_thread(run, resolved, metric_ctx)
    raise ValueError(f"Unknown aggregation mode: {mode}")
