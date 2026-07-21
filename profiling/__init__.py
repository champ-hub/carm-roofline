"""Profile package for instrumented application profiling with roofline metrics.

Provides a pipeline for collecting, discovering, parsing, aggregating, and
outputting profiling results from PAPI-instrumented (or other backend)
applications, with first-class MPI rank and thread awareness.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from core import UserError
from output_utils import detail, info, warn
from workspace import workspace_context

from .aggregation import AggregatedPoint, aggregate
from .backends import ProfilerBackend, create_backend
from .config import AggregationMode, BackendType, ProfileConfig
from .model import RankMetrics, RegionMetrics, RunMetadata, RunResults, ThreadMetrics
from .output import write_profile_results
from .papi_loader import load_all_ranks
from .papi_metrics import resolve_metrics
from .perf_loader import parse_perf_csv
from .perf_metrics import parse_perf_available_events, resolve_perf_metrics
from .shared import (
    MetricContext,
    MetricDefinition,
    MetricResolutionConfig,
    MetricType,
    RooflinePoint,
    compute_region_point,
    sum_roofline_points,
)

__all__ = [
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
    "RunResults",
    "ThreadMetrics",
    "aggregate",
    "compute_region_point",
    "create_backend",
    "load_all_ranks",
    "parse_perf_available_events",
    "parse_perf_csv",
    "profile_main",
    "resolve_metrics",
    "resolve_perf_metrics",
    "sum_roofline_points",
    "write_profile_results",
]


def profile_main(config: ProfileConfig) -> int:
    """Run the profiling pipeline given a resolved configuration.

    Pipeline:
        1. Select backend via factory and resolve metrics for the current system.
        2. Create a temporary workspace for raw profiling output.
        3. Run the profiled command (via the selected backend) → output files.
        4. Parse output files via the backend into the rank/thread/region model.
        5. Aggregate according to the chosen strategy (computing flops/bytes).
        6. Write output files to the final output directory.

    Args:
        config: Resolved profile configuration.

    Returns:
        Exit code (0 on success, 1 on failure).
    """
    # Validate command
    if not config.command:
        raise UserError("No command specified for profiling. Use 'carm profile -- <command>'.")

    # Build user-preference config for metric resolution
    resolution_cfg = MetricResolutionConfig(data_type=config.data_type, isas=config.isas)

    with workspace_context(keep=config.keep_artifacts, prefix="carm-profile-") as workspace_dir:
        workspace = Path(workspace_dir)

        if config.keep_artifacts:
            detail(f"Profiling artifacts will be kept in: {workspace}")

        # Factory: create the appropriate backend
        backend = create_backend(config, workspace, resolution_cfg)

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

        # Create a metric context for computing flops/bytes based on user preferences
        metric_ctx = MetricContext(resolution_cfg)

        exit_code = backend.run(config.command, cwd=Path.cwd())
        if exit_code != 0:
            warn(f"Profiled command exited with code {exit_code}. Processing any available results.")

        # Parse output via the backend (no per-backend branching)
        ranks = backend.parse_output()

        # Build metadata
        metadata = RunMetadata(
            name=config.app_name,
            date=datetime.now().isoformat(timespec="seconds"),
            method=backend.run_method_name,
            command=" ".join(config.command),
            threads_per_rank=max((len(r.threads) for r in ranks), default=1),
        )

        run = RunResults(metadata=metadata, ranks=ranks)
        detail(f"Run has {run.num_ranks} rank(s), {run.total_threads} total thread(s)")

        # Aggregate (computes flops/bytes from raw counters)
        points = aggregate(run, config.aggregation, resolved, metric_ctx)

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
