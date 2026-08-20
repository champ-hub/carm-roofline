"""Profile package for instrumented application profiling with roofline metrics.

Provides a pipeline for collecting, discovering, parsing, aggregating, and
outputting profiling results from PAPI-instrumented (or other backend)
applications, with first-class MPI rank and thread awareness.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from carm_roofline.core import UserError
from carm_roofline.output_utils import detail, info, warn
from carm_roofline.workspace import workspace_context

from .aggregation import AggregatedPoint, aggregate
from .backends import ProfilerBackend, RunResult, RunSpec, create_backend
from .config import AggregationMode, ProfileConfig
from .merge import merge_runs, missing_required_events, partition_events
from .model import RankMetrics, RegionMetrics, RunMetadata, RunResults, ThreadMetrics
from .optional_metrics import OPTIONAL_METRICS, resolve_optional_metrics
from .output import write_profile_results
from .papi_loader import load_all_ranks
from .papi_metrics import resolve_metrics
from .perf_loader import parse_perf_csv
from .perf_metrics import parse_perf_available_events, resolve_perf_metrics
from .shared import (
    BackendType,
    MetricContext,
    MetricDefinition,
    MetricResolutionConfig,
    MetricType,
    RooflinePoint,
    compute_region_point,
    sum_roofline_points,
)

__all__ = [
    "OPTIONAL_METRICS",
    "AggregatedPoint",
    "AggregationMode",
    "BackendType",
    "MetricDefinition",
    "MetricResolutionConfig",
    "MetricType",
    "ProfileConfig",
    "ProfilerBackend",
    "RankMetrics",
    "RegionMetrics",
    "RooflinePoint",
    "RunMetadata",
    "RunResult",
    "RunResults",
    "RunSpec",
    "ThreadMetrics",
    "aggregate",
    "compute_region_point",
    "create_backend",
    "load_all_ranks",
    "merge_runs",
    "missing_required_events",
    "parse_perf_available_events",
    "parse_perf_csv",
    "partition_events",
    "profile_main",
    "resolve_metrics",
    "resolve_optional_metrics",
    "resolve_perf_metrics",
    "sum_roofline_points",
    "write_profile_results",
]


def profile_main(config: ProfileConfig) -> int:
    """Run the profiling pipeline given a resolved configuration.

    Pipeline:
        1. List optional metrics and exit when ``--list-metrics`` is passed.
        2. Select the backend via factory and resolve metrics for the current system (one-time session probe).
        3. Resolve optional metrics and build the required event pool.
        4. Partition the pool into feasible runs (`partition_events`) only when `--merge-runs` is passed;
           otherwise run the whole pool as one chunk.
        5. Run each chunk via the same backend with a per-run `RunSpec` → output files.
        6. Parse each run (inside `profile`) and merge at the raw region-counter level (only when multiple runs
           were performed).
        7. Check all required events were collected; warn on gaps.
        8. Aggregate according to the chosen strategy (computing flops/bytes + optional metrics).
        9. Write output files to the final output directory.

    Args:
        config: Resolved profile configuration.

    Returns:
        Exit code (0 on success, 1 on failure).
    """
    if config.list_metrics:
        for name in sorted(OPTIONAL_METRICS, key=lambda metric: metric.value):
            info(f"{name.value}: {OPTIONAL_METRICS[name].description}")
        return 0

    # Validate command
    if not config.command:
        raise UserError("No command specified for profiling. Use 'carm profile -- <command>'.")

    # Build user-preference config for metric resolution
    resolution_cfg = MetricResolutionConfig(data_type=config.data_type, isas=config.isas)

    with workspace_context(keep=config.keep_artifacts, prefix="carm-profile-") as workspace_dir:
        workspace = Path(workspace_dir)

        if config.keep_artifacts:
            detail(f"Profiling artifacts will be kept in: {workspace}")

        # Backend: resolve metrics and check availability (one-time session probe)
        backend = create_backend(config, resolution_cfg)
        any_non_ideal = backend.check_prerequisites()
        if any_non_ideal and (not config.isas or config.data_type is None):
            warn(
                "Some of the available metrics don't provide ops/bytes directly. To improve the accuracy of derived "
                "metrics, specify the dominant ISA and data type of your application using --isa and --data-type."
            )

        # Read resolved metrics from the backend after check_prerequisites
        resolved = backend.resolved_metrics
        if not resolved:
            warn("No metric implementations could be resolved. Ops/bytes will be zero.")

        # Resolve optional metrics against the backend's available events
        optional_defs = resolve_optional_metrics(
            config.optional_metrics,
            backend.available_events,
            config.backend,
        )

        # Event pool: union of core + optional required events (deduped);
        # partitioned into disjoint chunks so each event is collected exactly once.
        pool: set[str] = set()
        for impl in resolved.values():
            pool |= impl.required_events
        for ro in optional_defs.values():
            pool |= ro.required_events
        if pool:
            if config.merge_runs:
                chunks: Sequence[list[str] | None] = partition_events(pool, backend.can_collect)
            else:
                chunks = [sorted(pool)]
        else:
            chunks = [None]

        # Create a metric context for computing flops/bytes based on user preferences
        metric_ctx = MetricContext(resolution_cfg)

        # One app run per chunk; each run gets its own workspace subdir so output files never overwrite.
        runs: list[RunResults] = []
        for i, chunk in enumerate(chunks):
            run_dir = workspace / f"run-{i}"
            result = backend.profile(
                RunSpec(output_dir=run_dir, events=",".join(chunk) if chunk else None),
                config.command,
                cwd=Path.cwd(),
            )
            if result.exit_code != 0:
                warn(f"Profiled command exited with code {result.exit_code}. Processing any available results.")

            runs.append(
                RunResults(
                    metadata=RunMetadata(
                        name=config.app_name,
                        date=datetime.now().isoformat(timespec="seconds"),
                        method=backend.run_method_name,
                        command=" ".join(config.command),
                        threads_per_rank=max((len(r.threads) for r in result.ranks), default=1),
                    ),
                    ranks=result.ranks,
                )
            )

        # Merge per-run results at the raw region-counter level (single run when no partitioning)
        run = merge_runs(runs) if len(runs) > 1 else runs[0]
        detail(f"Run has {run.num_ranks} rank(s), {run.total_threads} total thread(s)")

        # Coverage check: every required event must appear in the merged counters
        required: set[str] = set()
        for impl in resolved.values():
            required |= impl.required_events
        for ro in optional_defs.values():
            required |= ro.required_events
        missing = missing_required_events(run, required)
        if missing:
            warn(
                "The following required events were NOT collected: "
                f"{sorted(missing)}. Flops/bytes/optional metrics may be undercounted or zero."
            )

        # Aggregate (computes flops/bytes + optional metrics from raw counters)
        points = aggregate(run, config.aggregation, resolved, metric_ctx, optional_defs)

        # Write outputs to the final output directory
        write_profile_results(run, config, points)

        # Brief summary
        for pt in points:
            info(
                f"  {pt.label}: "
                f"AI={pt.arithmetic_intensity:.3f} FLOP/Byte, "
                f"{pt.flops_per_second / 1e9:.3f} GFLOP/s, "
                f"{pt.bandwidth / 1e9:.3f} GB/s, "
                f"{pt.runtime_s:.3f}s"
            )

    return 0
