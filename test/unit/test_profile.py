"""Unit tests for the profile package: model, aggregation, loaders, output, config."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

import carm_roofline.profiling.papi_backend as papi_backend
import carm_roofline.profiling.papi_metrics as papi_metrics
import carm_roofline.profiling.perf_backend as perf_backend
import carm_roofline.profiling.shared as shared
from carm_roofline.core import DataType, UserError
from carm_roofline.profiling import RunResult, RunSpec, profile_main
from carm_roofline.profiling.aggregation import (
    AggregatedPoint,
    aggregate,
    aggregate_global,
    aggregate_per_rank,
    aggregate_per_region_merged,
    aggregate_per_region_per_thread,
    aggregate_per_thread,
)
from carm_roofline.profiling.config import AggregationMode, ProfileConfig
from carm_roofline.profiling.merge import merge_runs, missing_required_events, partition_events
from carm_roofline.profiling.model import RegionMetrics, RunMetadata, RunResults, ThreadMetrics
from carm_roofline.profiling.optional_metrics import (
    OPTIONAL_METRICS,
    OptionalMetric,
    OptionalMetricImplementation,
    OptionalMetricName,
    _cache_level_bytes,
    _last_bucket,
    resolve_optional_metrics,
    validate_metric_names,
)
from carm_roofline.profiling.papi_lib import collectable_events
from carm_roofline.profiling.papi_loader import (
    RankMetrics,
    discover_rank_files,
    load_all_ranks,
    parse_rank_file,
)
from carm_roofline.profiling.papi_metrics import (
    METRICS,
    PAPIMetricRegistry,
    _papi_cache_key,
    _papi_event_cache_path,
    _parse_papi_xml_output,
    _store_papi_event_cache,
    build_isa_custom_metrics,
    fp_arith_counters_for_isas,
    parse_available_events,
    resolve_metrics,
)
from carm_roofline.profiling.perf_backend import PerfBackend
from carm_roofline.profiling.perf_loader import multiplexed_events
from carm_roofline.profiling.shared import (
    BackendType,
    MetricContext,
    MetricDefinition,
    MetricResolutionConfig,
    MetricType,
    RooflinePoint,
    compute_region_point,
    sum_roofline_points,
)

pytestmark = pytest.mark.unit

# Default MetricContext for tests (8 bytes/inst, 1 op/inst)
DEFAULT_CTX = MetricContext(MetricResolutionConfig())

# ---------------------------------------------------------------------------
# PAPI HL sample data (matches the actual format from rank_000003.json)
# ---------------------------------------------------------------------------

SAMPLE_PAPI_COUNTERS = {"PAPI_FP_OPS": 20971520, "PAPI_L1_DCA": 38637435}

SAMPLE_REGION_RAW: dict = {
    "name": "daxpy",
    "parent_region_id": "-1",
    "cycles": "1364391136",
    "real_time_nsec": "427162799",
    "PAPI_FP_OPS": "20971520",
    "PAPI_L1_DCA": "38637435",
}

SAMPLE_RANK_FILE: dict = {
    "papi_version": "7.2.0.0",
    "cpu_info": "AMD Ryzen 7 7735HS with Radeon Graphics",
    "max_cpu_rate_mhz": "4829",
    "min_cpu_rate_mhz": "400",
    "event_definitions": {
        "PAPI_FP_OPS": {"component": "perf_event", "type": "delta"},
        "PAPI_L1_DCA": {"component": "perf_event", "type": "delta"},
    },
    "threads": {
        "0": {
            "regions": {
                "0": dict(SAMPLE_REGION_RAW),
            }
        },
        "1": {
            "regions": {
                "0": dict(SAMPLE_REGION_RAW),
            }
        },
    },
}

# Resolved metrics matching the PAPI events in the sample
SAMPLE_RESOLVED: dict[MetricType, MetricDefinition] = resolve_metrics(frozenset({"PAPI_FP_OPS", "PAPI_L1_DCA"}))


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


def test_region_metrics() -> None:
    reg = RegionMetrics(
        name="daxpy",
        parent_region_id="-1",
        cycles=1364391136,
        time_nsec=427162799,
        counters=dict(SAMPLE_PAPI_COUNTERS),
    )
    assert reg.name == "daxpy"
    assert reg.counters["PAPI_FP_OPS"] == 20971520


def test_thread_metrics_with_regions() -> None:
    reg = RegionMetrics(
        name="daxpy",
        parent_region_id="-1",
        cycles=1364391136,
        time_nsec=427162799,
        counters=dict(SAMPLE_PAPI_COUNTERS),
    )
    tm = ThreadMetrics(thread_id=0, regions=[reg])
    assert tm.thread_id == 0
    assert len(tm.regions) == 1
    assert tm.regions[0].name == "daxpy"


def test_rank_metrics() -> None:
    reg = RegionMetrics(
        name="daxpy",
        parent_region_id="-1",
        cycles=1000,
        time_nsec=1000000,
        counters={"PAPI_FP_OPS": 100},
    )
    threads = [ThreadMetrics(thread_id=0, regions=[reg])]
    rank = RankMetrics(rank_id=0, event_definitions={"PAPI_FP_OPS": {"type": "delta"}}, threads=threads)
    assert rank.rank_id == 0
    assert len(rank.threads) == 1
    assert "PAPI_FP_OPS" in rank.event_definitions


def test_rank_metrics_empty() -> None:
    rank = RankMetrics(rank_id=0)
    assert rank.rank_id == 0
    assert rank.threads == []
    assert rank.event_definitions == {}


def test_run_results_hierarchy() -> None:
    reg = RegionMetrics(name="daxpy", parent_region_id="-1", cycles=1000, time_nsec=1000000, counters={})
    threads = [ThreadMetrics(thread_id=0, regions=[reg])]
    ranks = [RankMetrics(rank_id=0, threads=threads), RankMetrics(rank_id=1, threads=threads)]
    run = RunResults(metadata=RunMetadata(name="test"), ranks=ranks)
    assert run.num_ranks == 2
    assert run.total_threads == 2


def test_run_results_to_dict() -> None:
    reg = RegionMetrics(
        name="daxpy",
        parent_region_id="-1",
        cycles=1000,
        time_nsec=1000000,
        counters={"PAPI_FP_OPS": 100},
    )
    tm = ThreadMetrics(thread_id=0, regions=[reg])
    rank = RankMetrics(rank_id=0, threads=[tm])
    run = RunResults(metadata=RunMetadata(name="test", date="2024-01-01"), ranks=[rank])
    d = run.to_dict()
    assert d["metadata"]["name"] == "test"
    assert d["metadata"]["num_ranks"] == 1
    assert d["metadata"]["total_threads"] == 1
    assert len(d["ranks"]) == 1
    assert d["ranks"][0]["rank_id"] == 0
    assert len(d["ranks"][0]["threads"]) == 1
    assert d["ranks"][0]["threads"][0]["thread_id"] == 0
    assert d["ranks"][0]["threads"][0]["regions"][0]["name"] == "daxpy"
    assert d["ranks"][0]["threads"][0]["regions"][0]["counters"]["PAPI_FP_OPS"] == 100


# ---------------------------------------------------------------------------
# Metrics compute helpers tests
# ---------------------------------------------------------------------------


def test_sum_roofline_points() -> None:
    pts = [
        RooflinePoint(flops=100.0, bytes=50.0, time_s=1.0),
        RooflinePoint(flops=200.0, bytes=30.0, time_s=2.0),
    ]
    total = sum_roofline_points(pts)
    assert total.flops == 300.0
    assert total.bytes == 80.0
    assert total.time_s == 3.0  # sum (sequential execution)


def test_sum_roofline_points_empty() -> None:
    total = sum_roofline_points([])
    assert total.flops == 0.0
    assert total.bytes == 0.0
    assert total.time_s == 0.0


def test_compute_region_point() -> None:
    counters = {"PAPI_FP_OPS": 1000, "PAPI_L1_DCA": 500}
    resolved = resolve_metrics(frozenset({"PAPI_FP_OPS", "PAPI_L1_DCA"}))
    pt = compute_region_point(counters, 1_000_000_000, resolved, DEFAULT_CTX)
    # With DEFAULT_CTX (no data_type): double_ratio=0.0, single_ratio=1.0
    # DP_FLOPS via PAPI_FP_OPS = 1000 * 0.0 = 0.0
    # SP_FLOPS via PAPI_FP_OPS = 1000 * 1.0 = 1000.0
    assert pt.flops == 1000.0
    assert pt.bytes == 500 * DEFAULT_CTX.bytes_per_instruction  # bytes via PAPI_L1_DCA
    assert pt.time_s == 1.0


def test_compute_region_point_no_events() -> None:
    pt = compute_region_point({"PAPI_TOT_CYC": 100}, 1_000_000_000, {}, DEFAULT_CTX)
    assert pt.flops == 0.0
    assert pt.bytes == 0.0
    assert pt.time_s == 1.0


# ---------------------------------------------------------------------------
# Aggregation tests
# ---------------------------------------------------------------------------


def _make_sample_run() -> RunResults:
    """Create a simple run with 2 ranks, 1 thread each, 1 region."""
    reg = RegionMetrics(
        name="daxpy",
        parent_region_id="-1",
        cycles=1364391136,
        time_nsec=427162799,
        counters={"PAPI_FP_OPS": 20971520, "PAPI_L1_DCA": 38637435},
    )
    th = ThreadMetrics(thread_id=0, regions=[reg])
    ranks = [
        RankMetrics(rank_id=0, threads=[th]),
        RankMetrics(rank_id=1, threads=[th]),
    ]
    return RunResults(metadata=RunMetadata(name="test"), ranks=ranks)


def test_aggregate_global() -> None:
    run = _make_sample_run()
    pt = aggregate_global(run, SAMPLE_RESOLVED, DEFAULT_CTX)
    assert pt.num_ranks == 2
    assert pt.num_threads == 2
    assert pt.num_regions == 2
    assert pt.total_flops > 0
    assert pt.total_bytes > 0
    assert pt.runtime_s > 0


def test_aggregate_per_rank() -> None:
    run = _make_sample_run()
    pts = aggregate_per_rank(run, SAMPLE_RESOLVED, DEFAULT_CTX)
    assert len(pts) == 2
    for pt in pts:
        assert pt.num_ranks == 1
        assert pt.num_threads == 1
        assert pt.total_flops > 0


def test_aggregate_per_thread() -> None:
    run = _make_sample_run()
    pts = aggregate_per_thread(run, SAMPLE_RESOLVED, DEFAULT_CTX)
    assert len(pts) == 2  # 2 ranks x 1 thread each
    for pt in pts:
        assert pt.num_ranks == 1
        assert pt.num_threads == 1


def test_aggregate_per_region_merged() -> None:
    # Add a second region type to one thread only (avoid _make_sample_run's
    # shared ThreadMetrics reference across ranks)
    reg_a = RegionMetrics(
        name="daxpy",
        parent_region_id="-1",
        cycles=1364391136,
        time_nsec=427162799,
        counters={"PAPI_FP_OPS": 20971520, "PAPI_L1_DCA": 38637435},
    )
    reg_b = RegionMetrics(
        name="saxpy",
        parent_region_id="-1",
        cycles=1000,
        time_nsec=500000,
        counters={"PAPI_FP_OPS": 1000},
    )
    th0 = ThreadMetrics(thread_id=0, regions=[reg_a, reg_b])
    th1 = ThreadMetrics(thread_id=0, regions=[reg_a])
    ranks = [
        RankMetrics(rank_id=0, threads=[th0]),
        RankMetrics(rank_id=1, threads=[th1]),
    ]
    run = RunResults(metadata=RunMetadata(name="test"), ranks=ranks)

    pts = aggregate_per_region_merged(run, SAMPLE_RESOLVED, DEFAULT_CTX)
    names = {pt.label for pt in pts}
    assert "test_daxpy" in names
    assert "test_saxpy" in names
    pts_by_name = {pt.label: pt for pt in pts}
    assert pts_by_name["test_daxpy"].num_ranks == 2  # both ranks
    assert pts_by_name["test_daxpy"].num_threads == 2  # both threads
    assert pts_by_name["test_saxpy"].num_ranks == 1  # only rank 0
    assert pts_by_name["test_saxpy"].num_threads == 1  # only thread 0


def test_aggregate_dispatch_global() -> None:
    run = _make_sample_run()
    pts = aggregate(run, AggregationMode.GLOBAL, SAMPLE_RESOLVED, DEFAULT_CTX)
    assert len(pts) == 1


def test_aggregate_dispatch_per_rank() -> None:
    run = _make_sample_run()
    pts = aggregate(run, AggregationMode.RANK, SAMPLE_RESOLVED, DEFAULT_CTX)
    assert len(pts) == 2


def test_aggregate_dispatch_per_thread() -> None:
    run = _make_sample_run()
    pts = aggregate(run, AggregationMode.THREAD, SAMPLE_RESOLVED, DEFAULT_CTX)
    assert len(pts) == 2


def test_aggregate_dispatch_per_region_merged() -> None:
    run = _make_sample_run()
    pts = aggregate(run, AggregationMode.REGION_MERGED, SAMPLE_RESOLVED, DEFAULT_CTX)
    assert len(pts) == 1  # only "daxpy" exists


def test_aggregate_per_region_per_thread() -> None:
    # Reuse the reg_a / reg_b two-region pattern from test_aggregate_per_region_merged,
    # but expect one point per (rank, thread, region) with no merging.
    reg_a = RegionMetrics(
        name="daxpy",
        parent_region_id="-1",
        cycles=1364391136,
        time_nsec=427162799,
        counters={"PAPI_FP_OPS": 20971520, "PAPI_L1_DCA": 38637435},
    )
    reg_b = RegionMetrics(
        name="saxpy",
        parent_region_id="-1",
        cycles=1000,
        time_nsec=500000,
        counters={"PAPI_FP_OPS": 1000},
    )
    th0 = ThreadMetrics(thread_id=0, regions=[reg_a, reg_b])
    th1 = ThreadMetrics(thread_id=0, regions=[reg_a])
    ranks = [
        RankMetrics(rank_id=0, threads=[th0]),
        RankMetrics(rank_id=1, threads=[th1]),
    ]
    run = RunResults(metadata=RunMetadata(name="test"), ranks=ranks)

    pts = aggregate_per_region_per_thread(run, SAMPLE_RESOLVED, DEFAULT_CTX)
    assert len(pts) == 3  # 2 regions in th0 + 1 region in th1
    # Labels distinguish (rank, thread, region), so duplicate "daxpy" stays separate.
    labels = [pt.label for pt in pts]
    assert labels == [
        "test_rank0_thread0_daxpy",
        "test_rank0_thread0_saxpy",
        "test_rank1_thread0_daxpy",
    ]
    assert all(pt.num_ranks == 1 and pt.num_threads == 1 and pt.num_regions == 1 for pt in pts)


def test_aggregate_dispatch_per_region_per_thread() -> None:
    run = _make_sample_run()
    pts = aggregate(run, AggregationMode.REGION_PER_THREAD, SAMPLE_RESOLVED, DEFAULT_CTX)
    # _make_sample_run: 2 ranks, 1 thread each, 1 "daxpy" region each -> 2 points.
    assert len(pts) == 2


def test_aggregate_unknown_mode() -> None:
    run = _make_sample_run()
    with pytest.raises(ValueError, match="Unknown aggregation mode"):
        aggregate(run, "invalid", SAMPLE_RESOLVED, DEFAULT_CTX)  # type: ignore[arg-type]


def test_aggregated_point_properties() -> None:
    pt = AggregatedPoint(
        label="test",
        total_flops=1e10,
        total_bytes=1e8,
        runtime_s=10.0,
        num_ranks=1,
        num_threads=1,
        num_regions=2,
    )
    assert pt.arithmetic_intensity == 100.0
    assert pt.flops_per_second == 1e9
    assert pt.bandwidth == 1e7
    assert pt.num_regions == 2


def test_aggregated_point_zero_runtime() -> None:
    pt = AggregatedPoint(
        label="test",
        total_flops=0.0,
        total_bytes=0.0,
        runtime_s=0.0,
        num_ranks=1,
        num_threads=1,
    )
    assert pt.arithmetic_intensity == 0.0
    assert pt.flops_per_second == 0.0
    assert pt.bandwidth == 0.0


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_aggregation_mode_enum() -> None:
    assert AggregationMode.GLOBAL.value == "global"
    assert AggregationMode.RANK.value == "rank"
    assert AggregationMode.THREAD.value == "thread"
    assert AggregationMode.REGION_MERGED.value == "region_merged"
    assert AggregationMode.REGION_PER_THREAD.value == "region_per_thread"


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------


def test_discover_rank_files(tmp_path: Path) -> None:
    (tmp_path / "rank_000000.json").write_text(json.dumps(SAMPLE_RANK_FILE))
    (tmp_path / "rank_000001.json").write_text(json.dumps(SAMPLE_RANK_FILE))
    (tmp_path / "other.txt").write_text("not a rank file\n")
    (tmp_path / "nope.json").write_text("{}\n")

    files = discover_rank_files(tmp_path)
    assert len(files) == 2
    assert 0 in files
    assert 1 in files


def test_discover_rank_files_empty_dir(tmp_path: Path) -> None:
    assert discover_rank_files(tmp_path) == {}


def test_discover_rank_files_nonexistent(tmp_path: Path) -> None:
    assert discover_rank_files(tmp_path / "nonexistent") == {}


def test_parse_rank_file_json(tmp_path: Path) -> None:
    path = tmp_path / "rank_000003.json"
    path.write_text(json.dumps(SAMPLE_RANK_FILE))
    rank = parse_rank_file(path)
    assert rank is not None
    assert rank.rank_id == 3
    assert len(rank.threads) == 2
    assert rank.threads[0].thread_id == 0
    assert len(rank.threads[0].regions) == 1
    assert rank.threads[0].regions[0].name == "daxpy"
    assert rank.threads[0].regions[0].counters["PAPI_FP_OPS"] == 20971520
    assert "PAPI_FP_OPS" in rank.event_definitions


def test_parse_rank_file_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "rank_000000.json"
    path.write_text("not valid json")
    rank = parse_rank_file(path)
    assert rank is None


def test_parse_rank_file_empty_json(tmp_path: Path) -> None:
    path = tmp_path / "rank_000000.json"
    path.write_text("{}")
    rank = parse_rank_file(path)
    assert rank is None  # no threads


def test_parse_rank_file_no_threads(tmp_path: Path) -> None:
    path = tmp_path / "rank_000000.json"
    path.write_text('{"papi_version": "7.2", "threads": {}}')
    rank = parse_rank_file(path)
    assert rank is None


def test_parse_rank_file_malformed_region_no_counters(tmp_path: Path) -> None:
    """Region with only a name yields empty counters, not a failure."""
    data = dict(SAMPLE_RANK_FILE)
    data["threads"]["2"] = {"regions": {"0": {"name": "broken"}}}
    path = tmp_path / "rank_000000.json"
    path.write_text(json.dumps(data))
    rank = parse_rank_file(path)
    assert rank is not None
    assert len(rank.threads) == 3  # bad region just has empty counters
    assert rank.threads[2].regions[0].counters == {}


def test_load_all_ranks(tmp_path: Path) -> None:
    for rank_id in range(3):
        path = tmp_path / f"rank_{rank_id:06d}.json"
        path.write_text(json.dumps(SAMPLE_RANK_FILE))
    ranks = load_all_ranks(tmp_path)
    assert len(ranks) == 3
    assert ranks[0].rank_id == 0
    assert ranks[1].rank_id == 1
    assert ranks[2].rank_id == 2


def test_load_all_ranks_empty_dir(tmp_path: Path) -> None:
    assert load_all_ranks(tmp_path) == []


def test_default_app_name() -> None:
    from carm_roofline.profiling.config import _default_app_name

    assert _default_app_name(["./build/myapp", "arg1"]) == "myapp"
    assert _default_app_name(["mpirun", "-np", "4", "./myapp"]) == "myapp"
    assert _default_app_name(["srun", "-n", "4", "/path/to/app"]) == "app"
    assert _default_app_name(["myapp"]) == "myapp"
    assert _default_app_name(["myapp", "--input", "foo"]) == "myapp"
    assert _default_app_name([]) == "app"
    assert _default_app_name(["-flag", "--opt"]) == "app"


# ---------------------------------------------------------------------------
# Output tests
# ---------------------------------------------------------------------------


def test_write_profile_jsonl(tmp_path: Path) -> None:
    from unittest.mock import MagicMock

    from carm_roofline.profiling.output import write_profile_jsonl

    cfg = MagicMock(machine_name="test_run", output_dir=tmp_path, aggregation=AggregationMode.GLOBAL)
    run = _make_sample_run()
    pts = [aggregate_global(run, SAMPLE_RESOLVED, DEFAULT_CTX)]
    write_profile_jsonl(run, cfg, pts)

    jsonl_path = tmp_path / "test_run" / "applications.jsonl"
    assert jsonl_path.exists()
    lines = jsonl_path.read_text().strip().split("\n")
    assert len(lines) == 1  # single line per run

    record = json.loads(lines[0])
    assert record["format_version"] == "3.0"
    assert record["aggregation"] == "global"
    assert record["metadata"]["name"] == "test"
    assert "num_ranks" in record["metadata"]
    assert "total_threads" in record["metadata"]

    points = record["points"]
    assert isinstance(points, list)
    assert len(points) == 1
    point = points[0]
    assert "type" not in point
    assert point["label"] == "test"
    assert point["total_flops"] >= 0
    assert point["total_bytes"] >= 0
    assert point["runtime_s"] >= 0
    assert point["arithmetic_intensity"] >= 0
    assert point["flops_per_second"] >= 0
    assert point["bandwidth"] >= 0
    assert point["num_ranks"] == 2
    assert point["num_threads"] == 2
    assert point["num_regions"] == 2


def test_write_profile_jsonl_per_rank(tmp_path: Path) -> None:
    from unittest.mock import MagicMock

    from carm_roofline.profiling.output import write_profile_jsonl

    cfg = MagicMock(machine_name="test_run", output_dir=tmp_path, aggregation=AggregationMode.RANK)
    run = _make_sample_run()
    pts = aggregate_per_rank(run, SAMPLE_RESOLVED, DEFAULT_CTX)
    write_profile_jsonl(run, cfg, pts)

    jsonl_path = tmp_path / "test_run" / "applications.jsonl"
    lines = jsonl_path.read_text().strip().split("\n")
    assert len(lines) == 1  # single line per run

    record = json.loads(lines[0])
    assert record["aggregation"] == "rank"

    points = record["points"]
    assert isinstance(points, list)
    assert len(points) == 2
    for point in points:
        assert "type" not in point
        assert point["label"] in ("test_rank0", "test_rank1")


def test_write_profile_jsonl_per_region_merged(tmp_path: Path) -> None:
    from unittest.mock import MagicMock

    from carm_roofline.profiling.output import write_profile_jsonl

    reg_a = RegionMetrics(
        name="daxpy",
        parent_region_id="-1",
        cycles=1364391136,
        time_nsec=427162799,
        counters={"PAPI_FP_OPS": 20971520, "PAPI_L1_DCA": 38637435},
    )
    reg_b = RegionMetrics(
        name="saxpy",
        parent_region_id="-1",
        cycles=1000,
        time_nsec=500000,
        counters={"PAPI_FP_OPS": 1000},
    )
    th0 = ThreadMetrics(thread_id=0, regions=[reg_a, reg_b])
    th1 = ThreadMetrics(thread_id=0, regions=[reg_a])
    ranks = [
        RankMetrics(rank_id=0, threads=[th0]),
        RankMetrics(rank_id=1, threads=[th1]),
    ]
    run = RunResults(metadata=RunMetadata(name="test"), ranks=ranks)

    cfg = MagicMock(machine_name="test_run", output_dir=tmp_path, aggregation=AggregationMode.REGION_MERGED)
    pts = aggregate_per_region_merged(run, SAMPLE_RESOLVED, DEFAULT_CTX)
    write_profile_jsonl(run, cfg, pts)

    jsonl_path = tmp_path / "test_run" / "applications.jsonl"
    lines = jsonl_path.read_text().strip().split("\n")
    assert len(lines) == 1  # single line per run

    record = json.loads(lines[0])
    assert record["aggregation"] == "region_merged"

    points = record["points"]
    assert isinstance(points, list)
    assert len(points) == 2

    points_by_label = {}
    for point in points:
        assert "type" not in point
        points_by_label[point["label"]] = point

    assert points_by_label["test_daxpy"]["num_ranks"] == 2
    assert points_by_label["test_daxpy"]["num_threads"] == 2
    assert points_by_label["test_saxpy"]["num_ranks"] == 1
    assert points_by_label["test_saxpy"]["num_threads"] == 1


# ---------------------------------------------------------------------------
# Metrics registry tests
# ---------------------------------------------------------------------------


def test_metric_definition_frozen() -> None:
    md = MetricDefinition(
        type=MetricType.FLOPS,
        required_events=frozenset({"PAPI_DP_OPS"}),
        compute=lambda e, ctx: e["PAPI_DP_OPS"],
        priority=100,
        description="test",
    )
    assert md.type == MetricType.FLOPS
    assert md.required_events == frozenset({"PAPI_DP_OPS"})
    assert md.priority == 100
    assert md.description == "test"


def test_metric_registry_has_expected_metrics() -> None:
    assert MetricType.FLOPS in METRICS
    assert MetricType.BYTES in METRICS


def test_metric_registry_has_implementations_for_each_type() -> None:
    for mtype, impls in METRICS.items():
        assert len(impls) >= 3, f"Metric '{mtype}' should have at least 3 implementations, got {len(impls)}"


def test_resolve_metrics_flops_via_dp_ops() -> None:
    available = frozenset({"PAPI_DP_OPS", "PAPI_TOT_CYC"})
    resolved = resolve_metrics(available)
    assert MetricType.FLOPS in resolved


def test_resolve_metrics_flops_via_fp_ops() -> None:
    available = frozenset({"PAPI_FP_OPS", "PAPI_TOT_CYC"})
    resolved = resolve_metrics(available)
    assert MetricType.FLOPS in resolved
    assert "PAPI_FP_OPS" in resolved[MetricType.FLOPS].description


def test_resolve_metrics_flops_none_when_nothing_available() -> None:
    available = frozenset({"PAPI_TOT_CYC", "PAPI_L1_DCM"})
    resolved = resolve_metrics(available)
    assert MetricType.FLOPS not in resolved


def test_resolve_metrics_flops_via_fp_and_dp_ops() -> None:
    available = frozenset({"PAPI_FP_OPS", "PAPI_DP_OPS"})
    resolved = resolve_metrics(available)
    # PAPI_FP_OPS wins (priority 100 > 90)
    assert MetricType.FLOPS in resolved
    value = resolved[MetricType.FLOPS].compute({"PAPI_FP_OPS": 1000.0, "PAPI_DP_OPS": 400.0}, DEFAULT_CTX)
    assert value == 1000.0


def test_resolve_metrics_bytes_from_load_store() -> None:
    available = frozenset({"PAPI_LD_INS", "PAPI_SR_INS"})
    resolved = resolve_metrics(available)
    impl = resolved.get(MetricType.BYTES)
    assert impl is not None
    value = impl.compute({"PAPI_LD_INS": 1000.0, "PAPI_SR_INS": 500.0}, DEFAULT_CTX)
    assert value == (1000 + 500) * DEFAULT_CTX.bytes_per_instruction


def test_resolve_metrics_bytes_from_l1_accesses() -> None:
    available = frozenset({"PAPI_L1_DCA"})
    resolved = resolve_metrics(available)
    impl = resolved.get(MetricType.BYTES)
    assert impl is not None
    value = impl.compute({"PAPI_L1_DCA": 100.0}, DEFAULT_CTX)
    assert value == 100 * DEFAULT_CTX.bytes_per_instruction


def test_resolve_metrics_bytes_no_events() -> None:
    resolved = resolve_metrics(frozenset({"PAPI_TOT_CYC"}))
    assert MetricType.BYTES not in resolved


def test_resolve_metrics_empty_available() -> None:
    resolved = resolve_metrics(frozenset())
    assert resolved == {}


def test_resolve_metrics_all_metrics_resolved() -> None:
    all_papi = frozenset({"PAPI_DP_OPS", "PAPI_SP_OPS", "PAPI_LD_INS", "PAPI_SR_INS"})
    resolved = resolve_metrics(all_papi)
    assert MetricType.FLOPS in resolved
    assert MetricType.BYTES in resolved


# ---------------------------------------------------------------------------
# papi_decode parsing
# ---------------------------------------------------------------------------


def test_parse_papi_xml_output() -> None:
    sample = """<?xml version="1.0" encoding="UTF-8"?>
<eventinfo>
<component index="0" type="CPU" id="perf_event">
  <eventset type="NATIVE">
    <event index="0" name="PAPI_L1_DCM" desc="L1D cache misses"></event>
    <event index="1" name="PAPI_L2_DCM" desc="L2D cache misses"></event>
    <event index="2" name="PAPI_TOT_CYC" desc="Total cycles"></event>
  </eventset>
</component>
</eventinfo>"""
    events = _parse_papi_xml_output(sample)
    assert events == frozenset({"PAPI_L1_DCM", "PAPI_L2_DCM", "PAPI_TOT_CYC"})


def test_parse_papi_xml_output_empty() -> None:
    sample = """<?xml version="1.0" encoding="UTF-8"?>
<eventinfo>
<component index="0" type="CPU" id="perf_event">
  <eventset type="NATIVE">
  </eventset>
</component>
</eventinfo>"""
    assert _parse_papi_xml_output(sample) == frozenset()


def test_parse_papi_xml_output_malformed() -> None:
    events = _parse_papi_xml_output("not xml at all")
    assert events == frozenset()


# ---------------------------------------------------------------------------
# PAPI event catalog cache tests
# ---------------------------------------------------------------------------

SAMPLE_PAPI_EVENTS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<eventinfo>
<component index="0" type="CPU" id="perf_event">
  <eventset type="NATIVE">
    <event index="0" name="PAPI_L1_DCM" desc="L1D cache misses"></event>
    <event index="1" name="PAPI_TOT_CYC" desc="Total cycles"></event>
  </eventset>
</component>
</eventinfo>"""


def _patch_cache_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    """Point the PAPI event cache at *tmp_path* with a deterministic key."""
    monkeypatch.setattr(papi_metrics, "_papi_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(papi_metrics, "_papi_cache_key", lambda: "k" * 64)
    return "k" * 64


def _patch_xml_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make papi_xml_event_info discovery succeed and return the sample XML."""
    monkeypatch.setattr(papi_metrics.shutil, "which", lambda name: f"/fake/bin/{name}")
    result = SimpleNamespace(stdout=SAMPLE_PAPI_EVENTS_XML, stderr="", returncode=0)
    monkeypatch.setattr(papi_metrics.subprocess, "run", lambda *args, **kwargs: result)


def test_parse_available_events_cache_hit_skips_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key = _patch_cache_env(monkeypatch, tmp_path)
    monkeypatch.setattr(papi_metrics, "_load_papi_library", lambda: None)
    _store_papi_event_cache(key, frozenset({"PAPI_L1_DCM", "PAPI_TOT_CYC"}))

    def _must_not_run(*args: object, **kwargs: object) -> object:
        raise AssertionError("papi_xml_event_info must not run on cache hit")

    monkeypatch.setattr(papi_metrics.subprocess, "run", _must_not_run)

    events = parse_available_events()
    assert events == frozenset({"PAPI_L1_DCM", "PAPI_TOT_CYC"})


def test_parse_available_events_cache_miss_runs_and_stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key = _patch_cache_env(monkeypatch, tmp_path)
    _patch_xml_command(monkeypatch)
    monkeypatch.setattr(papi_metrics, "_load_papi_library", lambda: None)

    events = parse_available_events()
    assert events == frozenset({"PAPI_L1_DCM", "PAPI_TOT_CYC"})

    cache_file = _papi_event_cache_path(key)
    assert cache_file.is_file()
    data = json.loads(cache_file.read_text(encoding="utf-8"))
    assert data["key"] == key
    assert data["events"] == sorted({"PAPI_L1_DCM", "PAPI_TOT_CYC"})


def test_parse_available_events_cache_stale_key_runs_and_stores_new(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = _patch_cache_env(monkeypatch, tmp_path)
    _store_papi_event_cache("a" * 64, frozenset({"PAPI_TOT_CYC"}))
    _patch_xml_command(monkeypatch)
    monkeypatch.setattr(papi_metrics, "_load_papi_library", lambda: None)

    events = parse_available_events()
    assert events == frozenset({"PAPI_L1_DCM", "PAPI_TOT_CYC"})
    assert _papi_event_cache_path(key).is_file()


def test_parse_available_events_use_cache_false_ignores_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key = _patch_cache_env(monkeypatch, tmp_path)
    monkeypatch.setattr(papi_metrics, "_load_papi_library", lambda: None)
    _store_papi_event_cache(key, frozenset({"PAPI_L1_DCM", "PAPI_TOT_CYC"}))

    calls: list[list[str]] = []
    result = SimpleNamespace(stdout=SAMPLE_PAPI_EVENTS_XML, stderr="", returncode=0)

    def _recording_run(args: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(args)
        return result

    monkeypatch.setattr(papi_metrics.subprocess, "run", _recording_run)

    events = parse_available_events(use_cache=False)
    assert events == frozenset({"PAPI_L1_DCM", "PAPI_TOT_CYC"})
    assert calls, "papi_xml_event_info must run when the cache is disabled"


def test_parse_available_events_cache_corrupt_file_runs_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = _patch_cache_env(monkeypatch, tmp_path)
    _papi_event_cache_path(key).write_bytes(b"not json at all")
    _patch_xml_command(monkeypatch)
    monkeypatch.setattr(papi_metrics, "_load_papi_library", lambda: None)

    events = parse_available_events()
    assert events == frozenset({"PAPI_L1_DCM", "PAPI_TOT_CYC"})


def test_parse_available_events_filters_uncollectable_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only events kept by the collectability filter reach the catalog and its cache."""
    key = _patch_cache_env(monkeypatch, tmp_path)
    _patch_xml_command(monkeypatch)
    monkeypatch.setattr(papi_metrics, "_load_papi_library", lambda: object())
    monkeypatch.setattr(
        papi_metrics,
        "collectable_events",
        lambda events, library: frozenset(e for e in events if e == "PAPI_L1_DCM"),
    )

    events = parse_available_events()
    assert events == frozenset({"PAPI_L1_DCM"})

    payload = json.loads(_papi_event_cache_path(key).read_text(encoding="utf-8"))
    assert payload["version"] == papi_metrics._PAPI_EVENT_CACHE_VERSION
    assert payload["events"] == ["PAPI_L1_DCM"]


def test_collectable_events_keeps_only_addable_events() -> None:
    """Per-event add test drops unresolvable/unaddable names and always cleans up."""
    from unittest.mock import Mock

    lib = Mock()
    lib.PAPI_version.return_value = 0x07020000
    lib.PAPI_library_init.return_value = 0x07020000

    def fake_name_to_code(name: bytes, code_ptr: object) -> int:
        text = name.decode()
        if text == "gone":
            return -7
        code_ptr[0] = 5 if text == "good" else 6
        return 0

    lib.PAPI_event_name_to_code.side_effect = fake_name_to_code
    lib.PAPI_create_eventset.return_value = 0
    lib.PAPI_add_event.side_effect = lambda eventset, code: 0 if code == 5 else -1
    lib.PAPI_cleanup_eventset.return_value = 0
    lib.PAPI_destroy_eventset.return_value = 0

    result = collectable_events({"good", "bad", "gone"}, lib)
    assert result == frozenset({"good"})
    assert lib.PAPI_add_event.call_count == 2
    assert lib.PAPI_cleanup_eventset.call_count == 2
    assert lib.PAPI_destroy_eventset.call_count == 2


def test_papi_cache_key_changes_when_probe_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(papi_metrics, "_papi_cache_probes", lambda: {"a": "1", "b": "2"})
    key1 = _papi_cache_key()
    monkeypatch.setattr(papi_metrics, "_papi_cache_probes", lambda: {"a": "1", "b": "3"})
    key2 = _papi_cache_key()
    assert key1 != key2
    assert len(key1) == 64
    assert len(key2) == 64


# ---------------------------------------------------------------------------
# ISA -> FP_ARITH counter mapping tests
# ---------------------------------------------------------------------------


def test_fp_arith_counters_for_isas_scalar() -> None:
    from carm_roofline.isa.x86 import X86Scalar

    counters = fp_arith_counters_for_isas((X86Scalar,), DataType.f64)
    assert counters == {"FP_ARITH_INST_RETIRED:SCALAR_DOUBLE"}

    counters_f32 = fp_arith_counters_for_isas((X86Scalar,), DataType.f32)
    assert counters_f32 == {"FP_ARITH_INST_RETIRED:SCALAR_SINGLE"}


def test_fp_arith_counters_for_isas_sse() -> None:
    from carm_roofline.isa.x86 import X86SSE

    counters = fp_arith_counters_for_isas((X86SSE,), DataType.f64)
    assert counters == {"FP_ARITH_INST_RETIRED:128B_PACKED_DOUBLE"}


def test_fp_arith_counters_for_isas_avx2() -> None:
    from carm_roofline.isa.x86 import X86AVX2

    counters = fp_arith_counters_for_isas((X86AVX2,), DataType.f64)
    assert counters == {"FP_ARITH_INST_RETIRED:256B_PACKED_DOUBLE"}


def test_fp_arith_counters_for_isas_multiple() -> None:
    from carm_roofline.isa.x86 import X86AVX2, X86SSE, X86Scalar

    counters = fp_arith_counters_for_isas((X86AVX2, X86SSE, X86Scalar), DataType.f64)
    assert counters == {
        "FP_ARITH_INST_RETIRED:256B_PACKED_DOUBLE",
        "FP_ARITH_INST_RETIRED:128B_PACKED_DOUBLE",
        "FP_ARITH_INST_RETIRED:SCALAR_DOUBLE",
    }


def test_fp_arith_counters_for_isas_non_x86_returns_empty() -> None:
    from carm_roofline.isa.arm import ArmNeon

    counters = fp_arith_counters_for_isas((ArmNeon,), DataType.f64)
    assert counters == set()


# ---------------------------------------------------------------------------
# Custom metric factory tests
# ---------------------------------------------------------------------------


def test_build_isa_custom_metrics_returns_correct_events() -> None:
    from carm_roofline.isa.x86 import X86AVX2, X86SSE

    registry = build_isa_custom_metrics((X86AVX2, X86SSE), DataType.f64)
    assert registry is not None

    # FLOPS metric
    assert MetricType.FLOPS in registry
    flops_def = registry[MetricType.FLOPS][0]
    assert flops_def.priority == 200
    assert "PAPI_DP_OPS" not in flops_def.required_events  # NO derived preset!
    assert "FP_ARITH_INST_RETIRED:256B_PACKED_DOUBLE" in flops_def.required_events
    assert "FP_ARITH_INST_RETIRED:128B_PACKED_DOUBLE" in flops_def.required_events
    assert "FP_ARITH_INST_RETIRED:SCALAR_DOUBLE" not in flops_def.required_events  # not specified

    # BYTES metric
    assert MetricType.BYTES in registry
    bytes_def = registry[MetricType.BYTES][0]
    assert bytes_def.priority == 200
    assert "PAPI_LST_INS" in bytes_def.required_events
    assert "FP_ARITH_INST_RETIRED:256B_PACKED_DOUBLE" in bytes_def.required_events
    assert "FP_ARITH_INST_RETIRED:128B_PACKED_DOUBLE" in bytes_def.required_events
    assert "FP_ARITH_INST_RETIRED:SCALAR_DOUBLE" not in bytes_def.required_events  # not specified


def test_build_isa_custom_metrics_non_x86_returns_none() -> None:
    from carm_roofline.isa.arm import ArmNeon

    result = build_isa_custom_metrics((ArmNeon,), DataType.f64)
    assert result is None


def test_build_isa_custom_metrics_empty_isas_returns_none() -> None:
    result = build_isa_custom_metrics((), DataType.f64)
    assert result is None


# ---------------------------------------------------------------------------
# End-to-end resolution with custom metrics
# ---------------------------------------------------------------------------


def test_resolve_metrics_with_custom_isa_outranks_default() -> None:
    from carm_roofline.isa.x86 import X86AVX2

    available = frozenset(
        {
            "FP_ARITH_INST_RETIRED:256B_PACKED_DOUBLE",
            "PAPI_LST_INS",
        }
    )

    cfg = MetricResolutionConfig(data_type=DataType.f64, isas=(X86AVX2,))
    registry = PAPIMetricRegistry(cfg)
    resolved = registry.resolve(available)

    # FLOPS: custom wins (priority 200), no PAPI_DP_OPS
    assert MetricType.FLOPS in resolved
    flops_impl = resolved[MetricType.FLOPS]
    assert flops_impl.priority == 200
    assert "PAPI_DP_OPS" not in flops_impl.required_events
    assert "X86AVX2" in str(flops_impl.description)

    # BYTES: custom wins (priority 200)
    assert MetricType.BYTES in resolved
    bytes_impl = resolved[MetricType.BYTES]
    assert bytes_impl.priority == 200
    assert "X86AVX2" in str(bytes_impl.description)


# ---------------------------------------------------------------------------
# Run partitioning and merging (merge.py)
# ---------------------------------------------------------------------------


def test_partition_events_empty() -> None:
    assert partition_events([], lambda s: True) == []


def test_partition_events_all_fit_single_run_short_circuit() -> None:
    calls: list[frozenset[str]] = []

    def validator(events: frozenset[str]) -> bool:
        calls.append(events)
        return True

    chunks = partition_events({"b", "a", "c"}, validator)
    assert chunks == [["a", "b", "c"]]
    assert len(calls) == 1  # short-circuit: whole pool checked once


def test_partition_events_pairs_when_capacity_two() -> None:
    chunks = partition_events(list("abcdefghij"), lambda s: len(s) <= 2)
    assert chunks == [["a", "b"], ["c", "d"], ["e", "f"], ["g", "h"], ["i", "j"]]


def test_partition_events_all_rejected_isolates_with_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    warns: list[str] = []
    monkeypatch.setattr("carm_roofline.profiling.merge.warn", lambda *args, **kwargs: warns.append(str(args[0])))
    chunks = partition_events(["b", "a", "c"], lambda s: False)
    assert chunks == [["a"], ["b"], ["c"]]
    assert len(warns) == 3
    assert all("cannot be counted together" in w for w in warns)


def test_partition_events_partial_fit() -> None:
    # Pool of 4; capacity 2 (but the whole pool must be rejected first).
    def validator(events: frozenset[str]) -> bool:
        return len(events) != 4 and len(events) <= 2

    chunks = partition_events({"PAPI_DP_OPS", "PAPI_FP_OPS", "PAPI_L1_DCA", "PAPI_LST_INS"}, validator)
    assert chunks == [["PAPI_DP_OPS", "PAPI_FP_OPS"], ["PAPI_L1_DCA", "PAPI_LST_INS"]]


def _run_with_counters(counters: dict[str, int], name: str = "app") -> RunResults:
    region = RegionMetrics(name="total", parent_region_id="-1", cycles=100, time_nsec=1_000_000, counters=counters)
    thread = ThreadMetrics(thread_id=0, regions=[region])
    return RunResults(
        metadata=RunMetadata(name=name, date="2024-01-01T00:00:00"),
        ranks=[RankMetrics(rank_id=0, threads=[thread])],
    )


def test_merge_runs_requires_at_least_two() -> None:
    with pytest.raises(RuntimeError, match="requires at least 2 runs"):
        merge_runs([_run_with_counters({"PAPI_FP_OPS": 1000})])


def test_merge_runs_unions_disjoint_counters() -> None:
    merged = merge_runs([_run_with_counters({"PAPI_FP_OPS": 1000}), _run_with_counters({"PAPI_L1_DCA": 500})])
    region = merged.ranks[0].threads[0].regions[0]
    assert region.counters == {"PAPI_FP_OPS": 1000, "PAPI_L1_DCA": 500}
    assert region.cycles == 100  # from run 0 (identical executions)
    assert region.time_nsec == 1_000_000  # from run 0


def test_merge_runs_metadata_fresh_date_and_notes() -> None:
    run0 = _run_with_counters({"PAPI_FP_OPS": 1000})
    run1 = _run_with_counters({"PAPI_L1_DCA": 500})
    merged = merge_runs([run0, run1])
    assert merged.metadata.name == "app"
    assert merged.metadata.method == "PAPI_HL"  # from run 0
    assert merged.metadata.notes == "merged from 2 runs"
    assert merged.metadata.date != "2024-01-01T00:00:00"  # fresh date


def test_merge_runs_structural_mismatch_raises() -> None:
    run0 = _run_with_counters({"PAPI_FP_OPS": 1000})
    run1 = _run_with_counters({"PAPI_L1_DCA": 500})
    run1.ranks[0].threads[0].regions.append(
        RegionMetrics(name="second", parent_region_id="-1", cycles=1, time_nsec=1, counters={})
    )
    with pytest.raises(RuntimeError, match="region structure differs"):
        merge_runs([run0, run1])


def test_merge_runs_pairs_ranks_by_position_not_id() -> None:
    """PAPI HL names rank files per-process, so rank_ids differ across runs."""
    region0 = RegionMetrics(name="total", parent_region_id="-1", cycles=1, time_nsec=1, counters={"PAPI_FP_OPS": 1})
    region1 = RegionMetrics(name="total", parent_region_id="-1", cycles=1, time_nsec=1, counters={"PAPI_L1_DCA": 2})
    thread = ThreadMetrics(thread_id=0, regions=[region0])
    run0 = RunResults(metadata=RunMetadata(name="app"), ranks=[RankMetrics(rank_id=207867, threads=[thread])])
    run1 = RunResults(
        metadata=RunMetadata(name="app"),
        ranks=[RankMetrics(rank_id=760823, threads=[ThreadMetrics(thread_id=0, regions=[region1])])],
    )
    merged = merge_runs([run0, run1])
    assert merged.ranks[0].rank_id == 207867  # run-0 label kept
    assert merged.ranks[0].threads[0].regions[0].counters == {"PAPI_FP_OPS": 1, "PAPI_L1_DCA": 2}


def test_merge_runs_overlapping_event_warns_keeps_first(monkeypatch: pytest.MonkeyPatch) -> None:
    warns: list[str] = []
    monkeypatch.setattr("carm_roofline.profiling.merge.warn", lambda *args, **kwargs: warns.append(str(args[0])))
    merged = merge_runs(
        [
            _run_with_counters({"PAPI_FP_OPS": 1000, "PAPI_L1_DCA": 500}),
            _run_with_counters({"PAPI_L1_DCA": 999, "PAPI_LST_INS": 10}),
        ]
    )
    region = merged.ranks[0].threads[0].regions[0]
    assert region.counters["PAPI_L1_DCA"] == 500  # run-0 value kept
    assert region.counters["PAPI_LST_INS"] == 10
    assert any("PAPI_L1_DCA" in w and "multiple runs" in w for w in warns)


def test_merge_runs_event_definitions_union() -> None:
    def _rank_with_defs(defs: dict[str, object]) -> RankMetrics:
        region = RegionMetrics(name="total", parent_region_id="-1", cycles=1, time_nsec=1, counters={"PAPI_FP_OPS": 1})
        return RankMetrics(rank_id=0, event_definitions=defs, threads=[ThreadMetrics(thread_id=0, regions=[region])])

    run0 = RunResults(metadata=RunMetadata(name="app"), ranks=[_rank_with_defs({"PAPI_FP_OPS": {"type": "delta"}})])
    run1 = RunResults(metadata=RunMetadata(name="app"), ranks=[_rank_with_defs({"PAPI_L1_DCA": {"type": "delta"}})])
    merged = merge_runs([run0, run1])
    assert merged.ranks[0].event_definitions == {
        "PAPI_FP_OPS": {"type": "delta"},
        "PAPI_L1_DCA": {"type": "delta"},
    }


def test_missing_required_events_partial_collection() -> None:
    run = _run_with_counters({"PAPI_FP_OPS": 1000})
    assert missing_required_events(run, {"PAPI_FP_OPS", "PAPI_L1_DCA"}) == {"PAPI_L1_DCA"}
    assert missing_required_events(run, {"PAPI_FP_OPS"}) == set()
    assert missing_required_events(run, set()) == set()


# ---------------------------------------------------------------------------
# Metric-centric CLI (config.py)
# ---------------------------------------------------------------------------


def _parse_profile_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    ProfileConfig.insert_arguments(parser)
    return parser.parse_args(argv)


@pytest.mark.parametrize("flag", ["--papi-events", "--perf-events"])
def test_cli_removed_event_overrides_rejected(flag: str) -> None:
    with pytest.raises(SystemExit):
        _parse_profile_args([flag, "PAPI_FP_OPS", "--", "./app"])


def test_cli_metrics_space_separated_list_and_deduped() -> None:
    args = _parse_profile_args(
        ["--metrics", "cache-residency", "cache-line-utilization", "cache-residency", "--", "./app"]
    )
    config = ProfileConfig(args)
    assert config.optional_metrics == (
        OptionalMetricName.CACHE_RESIDENCY,
        OptionalMetricName.CACHE_LINE_UTILIZATION,
    )
    assert config.list_metrics is False


def test_cli_no_metric_defaults_empty() -> None:
    config = ProfileConfig(_parse_profile_args(["--", "./app"]))
    assert config.optional_metrics == ()


def test_cli_unknown_metric_rejected_at_parse_time() -> None:
    with pytest.raises(SystemExit):
        _parse_profile_args(["--metrics", "bogus", "--", "./app"])


def test_cli_list_metrics_flag() -> None:
    config = ProfileConfig(_parse_profile_args(["--list-metrics"]))
    assert config.list_metrics is True


def test_cli_merge_runs_flag_defaults_false() -> None:
    config = ProfileConfig(_parse_profile_args(["--", "./app"]))
    assert config.merge_runs is False


def test_profile_main_list_metrics_exits_without_command(monkeypatch: pytest.MonkeyPatch) -> None:
    messages: list[str] = []
    monkeypatch.setattr("carm_roofline.profiling.info", lambda *args, **kwargs: messages.append(str(args[0])))
    config = ProfileConfig(_parse_profile_args(["--list-metrics"]))
    assert profile_main(config) == 0
    assert any("cache-residency" in m for m in messages)
    assert any("cache-line-utilization" in m for m in messages)


# ---------------------------------------------------------------------------
# Cache-residency optional metric (optional_metrics.py)
# ---------------------------------------------------------------------------

PAPI_CACHE_IMPLEMENTATION = OPTIONAL_METRICS[OptionalMetricName.CACHE_RESIDENCY].implementations[BackendType.PAPI][0]
PERF_CACHE_IMPLEMENTATION = OPTIONAL_METRICS[OptionalMetricName.CACHE_RESIDENCY].implementations[BackendType.PERF][0]
PERF_INTEL_CACHE_IMPLEMENTATION = OPTIONAL_METRICS[OptionalMetricName.CACHE_RESIDENCY].implementations[BackendType.PERF][1]
L1_ONLY_CACHE_IMPLEMENTATION = OptionalMetricImplementation(
    required_events=frozenset({"PAPI_L1_DCA", "PAPI_L1_DCM"}),
    compute=lambda counters, region_bytes, bytes_per_instruction: _cache_level_bytes(
        counters.get("PAPI_L1_DCA", 0.0),
        {"l1": counters.get("PAPI_L1_DCM", 0.0)},
        ("l1",),
        region_bytes,
        bytes_per_instruction,
    ),
)


def _cache_compute(
    counters: dict[str, float],
    implementation: OptionalMetricImplementation,
    region_bytes: float = 800.0,
    bytes_per_instruction: float = 8.0,
) -> dict[str, float]:
    return dict(implementation.compute(counters, region_bytes, bytes_per_instruction))


@pytest.mark.parametrize(
    "counters, expected_fractions",
    [
        # pure L1: m1 = 0 -> all traffic served at L1
        (
            {"PAPI_L1_DCA": 800, "PAPI_L1_DCM": 0, "CORE_TO_L2_CACHEABLE_REQUEST_ACCESS_STATUS:LS_RD_BLK_C": 0},
            {"l1": 1.0, "l2": 0.0, "l3plus": 0.0},
        ),
        # pure L2: m1 = 1 (every line misses L1), nothing leaves L2
        (
            {"PAPI_L1_DCA": 800, "PAPI_L1_DCM": 100, "CORE_TO_L2_CACHEABLE_REQUEST_ACCESS_STATUS:LS_RD_BLK_C": 0},
            {"l1": 0.0, "l2": 1.0, "l3plus": 0.0},
        ),
        # pure L3plus: m1 = 1 and all L2-missing traffic leaves L2
        (
            {
                "PAPI_L1_DCA": 800,
                "PAPI_L1_DCM": 100,
                "CORE_TO_L2_CACHEABLE_REQUEST_ACCESS_STATUS:LS_RD_BLK_C": 100,
            },
            {"l1": 0.0, "l2": 0.0, "l3plus": 1.0},
        ),
        # mixed: 800 accesses * 8 B/inst / 64 = 100 lines; 40 miss L1 (m1=0.4);
        # 16 demand + 8 + 8 prefetch fills leave L2 -> f_l3plus = 0.32
        (
            {
                "PAPI_L1_DCA": 800,
                "PAPI_L1_DCM": 40,
                "CORE_TO_L2_CACHEABLE_REQUEST_ACCESS_STATUS:LS_RD_BLK_C": 16,
                "L2_PREFETCH_HIT_L3:L2_HW_PREFETCHER": 8,
                "L2_PREFETCH_MISS_L3:L2_HW_PREFETCHER": 8,
            },
            {"l1": 0.6, "l2": 0.08, "l3plus": 0.32},
        ),
        # saturation: fills exceed total_lines -> f_l2 == 0, sum still 1
        (
            {
                "PAPI_L1_DCA": 800,
                "PAPI_L1_DCM": 100,
                "CORE_TO_L2_CACHEABLE_REQUEST_ACCESS_STATUS:LS_RD_BLK_C": 500,
            },
            {"l1": 0.0, "l2": 0.0, "l3plus": 1.0},
        ),
        # saturation past the L1-miss remainder: fills = 100/100 but only 0.4
        # of traffic missed L1 -> f_l3plus caps at 0.4, f_l2 == 0
        (
            {
                "PAPI_L1_DCA": 800,
                "PAPI_L1_DCM": 40,
                "CORE_TO_L2_CACHEABLE_REQUEST_ACCESS_STATUS:LS_RD_BLK_C": 100,
            },
            {"l1": 0.6, "l2": 0.0, "l3plus": 0.4},
        ),
        # l1_misses > total_lines -> m1 saturates at 1
        (
            {"PAPI_L1_DCA": 800, "PAPI_L1_DCM": 500, "CORE_TO_L2_CACHEABLE_REQUEST_ACCESS_STATUS:LS_RD_BLK_C": 0},
            {"l1": 0.0, "l2": 1.0, "l3plus": 0.0},
        ),
    ],
)
def test_cache_residency_fractions_sum_to_one(counters: dict[str, float], expected_fractions: dict[str, float]) -> None:
    result = _cache_compute(counters, PAPI_CACHE_IMPLEMENTATION)
    assert sum(result.values()) == pytest.approx(800.0)  # saturated fractions sum to 1
    for level, frac in expected_fractions.items():
        assert result[level] == pytest.approx(frac * 800.0)


def test_cache_residency_zero_l1_accesses_all_zero() -> None:
    result = _cache_compute(
        {
            "PAPI_L1_DCA": 0,
            "PAPI_L1_DCM": 40,
            "CORE_TO_L2_CACHEABLE_REQUEST_ACCESS_STATUS:LS_RD_BLK_C": 10,
        },
        PAPI_CACHE_IMPLEMENTATION,
    )
    assert result == {"l1": 0.0, "l2": 0.0, "l3plus": 0.0}


def test_cache_residency_perf_derives_load_store_pairs() -> None:
    counters = {
        "L1-dcache-loads": 480,
        "L1-dcache-stores": 320,
        "L1-dcache-load-misses": 24,
        "L1-dcache-store-misses": 16,
        "l2_cache_misses_from_dc_misses": 16,
        "l2_pf_miss_l2_hit_l3": 8,
        "l2_pf_miss_l2_l3": 8,
    }
    # accesses = 480 + 320 = 800 -> 100 lines; misses = 40 -> m1 = 0.4;
    # fills = 16 + 8 + 8 = 32 -> f_l3plus = 0.32, f_l2 = 0.08
    result = _cache_compute(counters, PERF_CACHE_IMPLEMENTATION)
    assert result == {
        "l1": pytest.approx(480.0),
        "l2": pytest.approx(64.0),
        "l3plus": pytest.approx(256.0),
    }


def test_cache_residency_perf_missing_store_and_pf_roles_default_zero() -> None:
    # Partitioned runs can lack the store/pf counters: absent roles -> 0.0.
    counters = {
        "L1-dcache-loads": 800,
        "L1-dcache-load-misses": 40,
        "l2_cache_misses_from_dc_misses": 32,
    }
    result = _cache_compute(counters, PERF_CACHE_IMPLEMENTATION)
    assert result == {
        "l1": pytest.approx(480.0),
        "l2": pytest.approx(64.0),
        "l3plus": pytest.approx(256.0),
    }


def test_cache_residency_perf_zero_misses_all_l1() -> None:
    counters = {
        "L1-dcache-loads": 400,
        "L1-dcache-stores": 400,
        "L1-dcache-load-misses": 0,
        "L1-dcache-store-misses": 0,
        "l2_cache_misses_from_dc_misses": 0,
        "l2_pf_miss_l2_hit_l3": 0,
        "l2_pf_miss_l2_l3": 0,
    }
    result = _cache_compute(counters, PERF_CACHE_IMPLEMENTATION)
    assert result == {"l1": 800.0, "l2": 0.0, "l3plus": 0.0}

def test_validate_metric_names_empty_and_dedupe() -> None:
    assert validate_metric_names(None) == ()
    assert validate_metric_names([]) == ()
    assert tuple(
        m.value for m in validate_metric_names(["cache-residency", "cache-line-utilization", "cache-residency"])
    ) == ("cache-residency", "cache-line-utilization")


def test_validate_metric_names_unknown_raises_user_error() -> None:
    with pytest.raises(UserError) as exc_info:
        validate_metric_names(["bogus"])
    assert str(exc_info.value) == (
        "Unknown optional metric 'bogus'. Available: cache-line-utilization, cache-residency"
    )


AMD_PAPI_EVENTS = frozenset(
    {
        "PAPI_L1_DCA",
        "PAPI_L1_DCM",
        "CORE_TO_L2_CACHEABLE_REQUEST_ACCESS_STATUS:LS_RD_BLK_C",
        "L2_PREFETCH_HIT_L3:L2_HW_PREFETCHER",
        "L2_PREFETCH_MISS_L3:L2_HW_PREFETCHER",
    }
)


def test_resolve_optional_metrics_papi_prefers_amd_prefetch_inclusive() -> None:
    """The AMD implementation wins when all AMD events are available."""
    resolved = resolve_optional_metrics((OptionalMetricName.CACHE_RESIDENCY,), AMD_PAPI_EVENTS, BackendType.PAPI)
    implementation = resolved[OptionalMetricName.CACHE_RESIDENCY].implementation
    assert implementation is PAPI_CACHE_IMPLEMENTATION
    assert implementation.required_events == AMD_PAPI_EVENTS


def test_resolve_optional_metrics_papi_falls_back_to_intel_demand_only() -> None:
    """The Intel PAPI implementation resolves when AMD events are absent."""
    available = frozenset({"PAPI_L1_DCA", "PAPI_L1_DCM", "PAPI_L2_DCM"})
    resolved = resolve_optional_metrics((OptionalMetricName.CACHE_RESIDENCY,), available, BackendType.PAPI)
    implementation = resolved[OptionalMetricName.CACHE_RESIDENCY].implementation
    assert implementation is OPTIONAL_METRICS[OptionalMetricName.CACHE_RESIDENCY].implementations[BackendType.PAPI][1]
    assert implementation.required_events == available


def test_resolve_optional_metrics_perf_prefers_amd_prefetch_inclusive() -> None:
    available = PERF_CACHE_IMPLEMENTATION.required_events
    resolved = resolve_optional_metrics((OptionalMetricName.CACHE_RESIDENCY,), available, BackendType.PERF)
    assert resolved[OptionalMetricName.CACHE_RESIDENCY].implementation is PERF_CACHE_IMPLEMENTATION


def test_resolve_optional_metrics_perf_falls_back_to_intel_l2_rqsts() -> None:
    """The Intel perf implementation resolves when AMD events are absent."""
    available = PERF_INTEL_CACHE_IMPLEMENTATION.required_events
    resolved = resolve_optional_metrics((OptionalMetricName.CACHE_RESIDENCY,), available, BackendType.PERF)
    implementation = resolved[OptionalMetricName.CACHE_RESIDENCY].implementation
    assert implementation is PERF_INTEL_CACHE_IMPLEMENTATION
    assert {"LLC-load-misses", "LLC-store-misses"} <= implementation.required_events


def test_resolve_optional_metrics_missing_events_warns_and_omits(monkeypatch: pytest.MonkeyPatch) -> None:
    warns: list[str] = []
    monkeypatch.setattr(
        "carm_roofline.profiling.optional_metrics.warn", lambda *args, **kwargs: warns.append(str(args[0]))
    )
    available = frozenset({"PAPI_L1_DCA", "PAPI_L2_DCM"})  # no PAPI_L1_DCM
    resolved = resolve_optional_metrics((OptionalMetricName.CACHE_RESIDENCY,), available, BackendType.PAPI)
    assert resolved == {}
    assert warns
    assert "cache-residency" in warns[0]
    assert "Missing events" in warns[0]


def test_resolve_optional_metrics_all_unavailable_empty() -> None:
    assert resolve_optional_metrics((OptionalMetricName.CACHE_RESIDENCY,), frozenset(), BackendType.PAPI) == {}


def test_resolve_optional_metrics_unsupported_backend_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    warns: list[str] = []
    monkeypatch.setattr(
        "carm_roofline.profiling.optional_metrics.warn", lambda *args, **kwargs: warns.append(str(args[0]))
    )
    papi_only = OptionalMetric(
        description="papi-only",
        implementations={BackendType.PAPI: (PAPI_CACHE_IMPLEMENTATION,)},
    )
    monkeypatch.setattr(
        "carm_roofline.profiling.optional_metrics.OPTIONAL_METRICS",
        {OptionalMetricName.CACHE_RESIDENCY: papi_only},
    )
    resolved = resolve_optional_metrics(
        (OptionalMetricName.CACHE_RESIDENCY,), frozenset({"PAPI_L1_DCA"}), BackendType.PERF
    )
    assert resolved == {}
    assert warns and "not supported by the perf backend" in warns[0]


def test_compute_region_point_optional_bytes() -> None:
    counters = {
        "PAPI_FP_OPS": 1000,
        "PAPI_L1_DCA": 800,
        "PAPI_L1_DCM": 40,
        "CORE_TO_L2_CACHEABLE_REQUEST_ACCESS_STATUS:LS_RD_BLK_C": 16,
        "L2_PREFETCH_HIT_L3:L2_HW_PREFETCHER": 8,
        "L2_PREFETCH_MISS_L3:L2_HW_PREFETCHER": 8,
    }
    resolved = resolve_metrics(frozenset({"PAPI_FP_OPS", "PAPI_L1_DCA"}))
    resolved_optional = resolve_optional_metrics(
        (OptionalMetricName.CACHE_RESIDENCY,), frozenset(counters), BackendType.PAPI
    )
    pt = compute_region_point(counters, 1_000_000_000, resolved, DEFAULT_CTX, resolved_optional)
    assert pt.flops == 1000.0
    assert pt.bytes == 800 * DEFAULT_CTX.bytes_per_instruction  # 800 accesses x 8 B/inst = 6400
    assert pt.optional_bytes == {
        "cache-residency": {
            "l1": pytest.approx(0.6 * 6400.0),
            "l2": pytest.approx(0.08 * 6400.0),
            "l3plus": pytest.approx(0.32 * 6400.0),
        }
    }


def test_cache_level_bytes_scales_accesses_to_lines_and_saturates() -> None:
    """L1 accesses are scaled to 64B lines; fills saturating past the L1-miss
    remainder must not push the sum past 1.

    accesses=800 * 8 B/inst / 64 = 100 lines; m1 = 40/100 = 0.4 -> f_l1 = 0.6;
    l2 boundary (misses + pf fills) = 500/100 = 5 -> saturated to the 0.4
    remainder: f_l3plus = 0.4, f_l2 = 0.
    """
    levels = _cache_level_bytes(
        accesses=800,
        misses={"l1": 40.0, "l2": 200.0 + 150.0 + 150.0},
        levels=("l1", "l2"),
        region_bytes=800.0,
        bytes_per_instruction=8.0,
    )
    assert levels == {
        "l1": pytest.approx(0.6 * 800.0),
        "l2": pytest.approx(0.0),
        "l3plus": pytest.approx(0.4 * 800.0),
    }
    assert sum(levels.values()) == pytest.approx(800.0)


def test_last_bucket_names() -> None:
    """The everything-beyond bucket: dram when L3 is the last boundary, else next-plus."""
    assert _last_bucket(("l1",)) == "l2plus"
    assert _last_bucket(("l1", "l2")) == "l3plus"
    assert _last_bucket(("l1", "l2", "l3")) == "dram"


@pytest.mark.parametrize(
    "counters, expected_fractions",


    [
        # pure L1: no misses anywhere
        (
            {
                "L1-dcache-loads": 480,
                "L1-dcache-stores": 320,
                "L1-dcache-load-misses": 0,
                "L1-dcache-store-misses": 0,
                "l2_rqsts.miss": 0,
                "LLC-load-misses": 0,
                "LLC-store-misses": 0,
            },
            {"l1": 1.0, "l2": 0.0, "l3": 0.0, "dram": 0.0},
        ),
        # telescoping: 800 accesses * 8 B/inst / 64 = 100 lines;
        # l1=40 (0.4), l2=12 (0.12), l3=7 (0.07) -> f = 0.6 / 0.28 / 0.05 / 0.07
        (
            {
                "L1-dcache-loads": 480,
                "L1-dcache-stores": 320,
                "L1-dcache-load-misses": 24,
                "L1-dcache-store-misses": 16,
                "l2_rqsts.miss": 12,
                "LLC-load-misses": 4,
                "LLC-store-misses": 3,
            },
            {"l1": 0.6, "l2": 0.28, "l3": 0.05, "dram": 0.07},
        ),
        # saturation across three boundaries: l3 misses > l2 misses clamps
        # f_l3 to 0 and keeps the sum at 1
        (
            {
                "L1-dcache-loads": 480,
                "L1-dcache-stores": 320,
                "L1-dcache-load-misses": 24,
                "L1-dcache-store-misses": 16,
                "l2_rqsts.miss": 12,
                "LLC-load-misses": 30,
                "LLC-store-misses": 20,
            },
            {"l1": 0.6, "l2": 0.28, "l3": 0.0, "dram": 0.12},
        ),
        # l1_misses > total_lines saturates m1 at 1: all traffic leaves L1,
        # then telescopes through l2/l3
        (
            {
                "L1-dcache-loads": 480,
                "L1-dcache-stores": 320,
                "L1-dcache-load-misses": 90,
                "L1-dcache-store-misses": 60,
                "l2_rqsts.miss": 12,
                "LLC-load-misses": 4,
                "LLC-store-misses": 3,
            },
            {"l1": 0.0, "l2": 0.88, "l3": 0.05, "dram": 0.07},
        ),
    ],
)
def test_cache_residency_four_boundaries_sum_to_one(
    counters: dict[str, float], expected_fractions: dict[str, float]
) -> None:
    """4-bucket shape (levels l1/l2/l3): exact {l1, l2, l3, dram} buckets."""
    result = _cache_compute(counters, PERF_INTEL_CACHE_IMPLEMENTATION)
    assert sum(result.values()) == pytest.approx(800.0)  # saturated fractions sum to 1
    assert set(result) == {"l1", "l2", "l3", "dram"}
    for level, frac in expected_fractions.items():
        assert result[level] == pytest.approx(frac * 800.0)


def test_cache_line_utilization_papi_resolution_and_formula() -> None:
    counters = {"PAPI_L1_DCM": 1}
    resolved = resolve_optional_metrics(
        (OptionalMetricName.CACHE_LINE_UTILIZATION,), frozenset(counters), BackendType.PAPI
    )
    point = compute_region_point(counters, 1, {}, DEFAULT_CTX, resolved)
    aggregate_point = AggregatedPoint(
        label="test",
        total_flops=0.0,
        total_bytes=128.0,
        runtime_s=0.0,
        num_ranks=1,
        num_threads=1,
        optional_bytes=point.optional_bytes,
    )
    assert point.optional_bytes == {"cache-line-utilization": {"l1-miss": 64.0}}
    assert aggregate_point.optional_fractions == {"cache-line-utilization": {"value": 2.0}}


def test_cache_line_utilization_perf_resolution_and_zero_misses() -> None:
    events = frozenset({"L1-dcache-load-misses", "L1-dcache-store-misses"})
    resolved = resolve_optional_metrics((OptionalMetricName.CACHE_LINE_UTILIZATION,), events, BackendType.PERF)
    point = compute_region_point(
        {"L1-dcache-load-misses": 0, "L1-dcache-store-misses": 0}, 1, {}, DEFAULT_CTX, resolved
    )
    aggregate_point = AggregatedPoint(
        label="test",
        total_flops=0.0,
        total_bytes=128.0,
        runtime_s=0.0,
        num_ranks=1,
        num_threads=1,
        optional_bytes=point.optional_bytes,
    )
    assert point.optional_bytes == {"cache-line-utilization": {"l1-miss": 0.0}}
    assert aggregate_point.optional_fractions == {}


def test_cache_residency_four_boundaries_zero_accesses_all_zero() -> None:
    result = _cache_compute(
        {
            "L1-dcache-loads": 0,
            "L1-dcache-stores": 0,
            "L1-dcache-load-misses": 40,
            "L1-dcache-store-misses": 10,
            "l2_rqsts.miss": 10,
            "LLC-load-misses": 5,
            "LLC-store-misses": 5,
        },
        PERF_INTEL_CACHE_IMPLEMENTATION,
    )
    assert result == {"l1": 0.0, "l2": 0.0, "l3": 0.0, "dram": 0.0}


@pytest.mark.parametrize(
    "counters, expected_fractions",
    [
        # pure L1: m1 = 0 -> everything served at L1
        ({"PAPI_L1_DCA": 800, "PAPI_L1_DCM": 0}, {"l1": 1.0, "l2plus": 0.0}),
        # all-miss: m1 = 1 -> everything beyond L1
        ({"PAPI_L1_DCA": 800, "PAPI_L1_DCM": 100}, {"l1": 0.0, "l2plus": 1.0}),
        # mixed: 40/100 lines miss L1
        ({"PAPI_L1_DCA": 800, "PAPI_L1_DCM": 40}, {"l1": 0.6, "l2plus": 0.4}),
        # saturation: m1 saturates at 1
        ({"PAPI_L1_DCA": 800, "PAPI_L1_DCM": 500}, {"l1": 0.0, "l2plus": 1.0}),
    ],
)
def test_cache_residency_single_boundary_sum_to_one(
    counters: dict[str, float], expected_fractions: dict[str, float]
) -> None:
    """1-bucket-shape (levels ("l1",)): {l1, l2plus} with everything beyond L1 grouped."""
    result = _cache_compute(counters, L1_ONLY_CACHE_IMPLEMENTATION)
    assert sum(result.values()) == pytest.approx(800.0)
    assert set(result) == {"l1", "l2plus"}
    for level, frac in expected_fractions.items():
        assert result[level] == pytest.approx(frac * 800.0)


# ---------------------------------------------------------------------------
# Optional bytes through aggregation and output
# ---------------------------------------------------------------------------

# cache-residency counters (same ratios as the end-to-end test): per region
# bytes = PAPI_L1_DCA * 8 (DEFAULT_CTX) = 6400; accesses=800 -> 100 lines;
# m1=0.4, fills=32 -> fractions {l1:0.6, l2:0.08, l3plus:0.32}.
_CACHE_COUNTERS = {
    "PAPI_FP_OPS": 1000,
    "PAPI_L1_DCA": 800,
    "PAPI_L1_DCM": 40,
    "CORE_TO_L2_CACHEABLE_REQUEST_ACCESS_STATUS:LS_RD_BLK_C": 16,
    "L2_PREFETCH_HIT_L3:L2_HW_PREFETCHER": 8,
    "L2_PREFETCH_MISS_L3:L2_HW_PREFETCHER": 8,
}
_CACHE_OPTIONAL = resolve_optional_metrics(
    (OptionalMetricName.CACHE_RESIDENCY,), frozenset(_CACHE_COUNTERS), BackendType.PAPI
)


def _make_cache_run() -> RunResults:
    """Run with cache-residency counters mirroring _make_sample_run (2 ranks x 1 thread)."""
    reg = RegionMetrics(
        name="daxpy",
        parent_region_id="-1",
        cycles=1364391136,
        time_nsec=427162799,
        counters=dict(_CACHE_COUNTERS),
    )
    th = ThreadMetrics(thread_id=0, regions=[reg])
    ranks = [
        RankMetrics(rank_id=0, threads=[th]),
        RankMetrics(rank_id=1, threads=[th]),
    ]
    return RunResults(metadata=RunMetadata(name="test"), ranks=ranks)


@pytest.mark.parametrize(
    "mode, expected_per_point",
    [
        (AggregationMode.GLOBAL, [{"l1": 7680.0, "l2": 1024.0, "l3plus": 4096.0}]),
        (AggregationMode.RANK, [{"l1": 3840.0, "l2": 512.0, "l3plus": 2048.0}] * 2),
        (AggregationMode.THREAD, [{"l1": 3840.0, "l2": 512.0, "l3plus": 2048.0}] * 2),
        (AggregationMode.REGION_MERGED, [{"l1": 7680.0, "l2": 1024.0, "l3plus": 4096.0}]),
        (AggregationMode.REGION_PER_THREAD, [{"l1": 3840.0, "l2": 512.0, "l3plus": 2048.0}] * 2),
    ],
)
def test_aggregate_modes_carry_optional_bytes(
    mode: AggregationMode, expected_per_point: list[dict[str, float]]
) -> None:
    run = _make_cache_run()
    resolved = resolve_metrics(frozenset({"PAPI_FP_OPS", "PAPI_L1_DCA"}))
    points = aggregate(run, mode, resolved, DEFAULT_CTX, _CACHE_OPTIONAL)
    assert len(points) == len(expected_per_point)
    for pt, expected in zip(points, expected_per_point):
        assert set(pt.optional_bytes) == {"cache-residency"}
        levels = pt.optional_bytes["cache-residency"]
        for level, value in expected.items():
            assert levels[level] == pytest.approx(value)
        level_total = sum(expected.values())
        assert pt.total_bytes == pytest.approx(level_total)
        fractions = pt.optional_fractions["cache-residency"]
        for level, value in expected.items():
            assert fractions[level] == pytest.approx(value / level_total)
        assert sum(fractions.values()) == pytest.approx(1.0)


def test_aggregate_without_optional_is_empty() -> None:
    run = _make_cache_run()
    resolved = resolve_metrics(frozenset({"PAPI_FP_OPS", "PAPI_L1_DCA"}))
    pt = aggregate_global(run, resolved, DEFAULT_CTX)
    assert pt.optional_bytes == {}
    assert pt.optional_fractions == {}


def test_write_profile_jsonl_optional_metrics(tmp_path: Path) -> None:
    from unittest.mock import MagicMock

    from carm_roofline.profiling.output import write_profile_jsonl

    run = _make_cache_run()
    resolved = resolve_metrics(frozenset({"PAPI_FP_OPS", "PAPI_L1_DCA"}))
    pts = aggregate(run, AggregationMode.GLOBAL, resolved, DEFAULT_CTX, _CACHE_OPTIONAL)
    cfg = MagicMock(machine_name="test_run", output_dir=tmp_path, aggregation=AggregationMode.GLOBAL)
    write_profile_jsonl(run, cfg, pts)

    jsonl_path = tmp_path / "test_run" / "applications.jsonl"
    record = json.loads(jsonl_path.read_text().strip().split("\n")[0])
    assert record["format_version"] == "3.0"
    assert record["optional_metrics"] == ["cache-residency"]

    point = record["points"][0]
    assert set(point["optional_bytes"]) == {"cache-residency"}
    for level, value in point["optional_bytes"]["cache-residency"].items():
        assert value == pytest.approx({"l1": 7680.0, "l2": 1024.0, "l3plus": 4096.0}[level])
    fractions = point["optional_fractions"]["cache-residency"]
    assert fractions == {
        "l1": pytest.approx(0.6),
        "l2": pytest.approx(0.08),
        "l3plus": pytest.approx(0.32),
    }


def test_load_applications_carries_optional_fractions(tmp_path: Path) -> None:
    from carm_roofline.roofline_assembly import load_applications

    record = {
        "format_version": "3.0",
        "aggregation": "global",
        "optional_metrics": ["cache-residency"],
        "metadata": {"name": "app", "date": "2026-01-01T00:00:00", "command": "./app"},
        "points": [
            {
                "label": "app",
                "total_flops": 1000.0,
                "total_bytes": 6400.0,
                "runtime_s": 1.0,
                "num_ranks": 1,
                "num_threads": 1,
                "num_regions": 1,
                "arithmetic_intensity": 0.15625,
                "flops_per_second": 1000.0,
                "bandwidth": 6400.0,
                "optional_bytes": {"cache-residency": {"l1": 3840.0, "l2": 512.0, "l3plus": 2048.0}},
                "optional_fractions": {"cache-residency": {"l1": 0.6, "l2": 0.08, "l3plus": 0.32}},
            }
        ],
    }
    path = tmp_path / "applications.jsonl"
    path.write_text(json.dumps(record, sort_keys=True) + "\n")

    records = load_applications(path)
    assert len(records) == 1
    pt = records[0].points[0]
    # Known fields survive the round trip; optional fractions are carried too.
    assert pt.label == "app"
    assert pt.total_flops == 1000.0
    assert pt.total_bytes == 6400.0
    assert pt.runtime_s == 1.0
    assert pt.arithmetic_intensity == 0.15625
    assert pt.bandwidth == 6400.0
    assert pt.optional_fractions == {"cache-residency": {"l1": 0.6, "l2": 0.08, "l3plus": 0.32}}


# ---------------------------------------------------------------------------
# End-to-end profile_main with mocked PAPI backend
# ---------------------------------------------------------------------------

_CACHE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<eventinfo>
<component index="0" type="CPU" id="perf_event">
  <eventset type="NATIVE">
    <event index="0" name="PAPI_FP_OPS" desc="floating point ops"></event>
    <event index="1" name="PAPI_L1_DCA" desc="l1 data accesses"></event>
    <event index="2" name="PAPI_L1_DCM" desc="l1 data misses"></event>
    <event index="3" name="CORE_TO_L2_CACHEABLE_REQUEST_ACCESS_STATUS:LS_RD_BLK_C" desc="l2 demand data misses"></event>
    <event index="4" name="L2_PREFETCH_HIT_L3:L2_HW_PREFETCHER" desc="l2 prefetch fills hitting l3"></event>
    <event index="5" name="L2_PREFETCH_MISS_L3:L2_HW_PREFETCHER" desc="l2 prefetch fills missing l3"></event>
  </eventset>
</component>
</eventinfo>"""


def _patch_papi_discovery(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Make PAPI discovery hermetic: fake lib, XML catalog, isolated cache."""
    monkeypatch.setattr(papi_backend, "_find_papi_library_path", lambda: Path("/fake/libpapi.so"))
    monkeypatch.setattr(papi_backend.shutil, "which", lambda name: f"/fake/bin/{name}")
    monkeypatch.setattr(papi_metrics, "_papi_cache_dir", lambda: tmp_path / "papi-cache")
    monkeypatch.setattr(papi_metrics, "_papi_cache_key", lambda: "k" * 64)
    monkeypatch.setattr(papi_metrics, "_load_papi_library", lambda: None)

    def fake_run(args: list[str], **kwargs: object) -> SimpleNamespace:
        if args[0] == "nm":
            return SimpleNamespace(stdout="PAPI_hl_region_begin\n", stderr="", returncode=0)
        return SimpleNamespace(stdout=_CACHE_XML, stderr="", returncode=0)

    monkeypatch.setattr(papi_backend.subprocess, "run", fake_run)


def test_profile_main_papi_optional_metric_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Single-run profile: cache-residency resolves against the AMD prefetch-inclusive alternative.

    Counters: PAPI_FP_OPS=1000, PAPI_L1_DCA=800, PAPI_L1_DCM=40,
    CORE_TO_L2_CACHEABLE_REQUEST_ACCESS_STATUS:LS_RD_BLK_C=16,
    L2_PREFETCH_HIT_L3:L2_HW_PREFETCHER=8, L2_PREFETCH_MISS_L3:L2_HW_PREFETCHER=8
    -> 800 accesses * 8 B/inst / 64 = 100 lines; m1=0.4; fills=32 ->
    fractions {l1:0.6, l2:0.08, l3plus:0.32} and bytes {3840, 512, 2048}
    of 6400 (PAPI_L1_DCA x 8 B/inst).
    """
    _patch_papi_discovery(monkeypatch, tmp_path)
    monkeypatch.setattr(shared, "perf_event_paranoid", lambda: 2)
    monkeypatch.setattr(papi_backend, "validate_event_set", lambda events: True)

    counters = {
        "PAPI_FP_OPS": 1000,
        "PAPI_L1_DCA": 800,
        "PAPI_L1_DCM": 40,
        "CORE_TO_L2_CACHEABLE_REQUEST_ACCESS_STATUS:LS_RD_BLK_C": 16,
        "L2_PREFETCH_HIT_L3:L2_HW_PREFETCHER": 8,
        "L2_PREFETCH_MISS_L3:L2_HW_PREFETCHER": 8,
    }

    def fake_profile(
        self: papi_backend.PAPIHLBackend, run_spec: RunSpec, command: list[str], cwd: Path
    ) -> RunResult:
        region = RegionMetrics(
            name="total", parent_region_id="-1", cycles=0, time_nsec=1_000_000_000, counters=counters
        )
        ranks = [RankMetrics(rank_id=0, threads=[ThreadMetrics(thread_id=0, regions=[region])])]
        return RunResult(exit_code=0, ranks=ranks)

    monkeypatch.setattr(papi_backend.PAPIHLBackend, "profile", fake_profile)

    args = _parse_profile_args(
        [
            "--backend",
            "papi",
            "--data-type",
            "f64",
            "--metrics",
            "cache-residency",
            "--aggregation",
            "global",
            "--output-dir",
            str(tmp_path),
            "--",
            "./app",
        ]
    )
    config = ProfileConfig(args)
    assert profile_main(config) == 0

    jsonl_path = tmp_path / config.machine_name / "applications.jsonl"
    lines = jsonl_path.read_text().strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[-1])
    assert record["format_version"] == "3.0"
    assert record["optional_metrics"] == ["cache-residency"]

    point = record["points"][0]
    assert point["total_flops"] == pytest.approx(1000.0)
    assert point["total_bytes"] == pytest.approx(6400.0)
    assert set(point["optional_bytes"]) == {"cache-residency"}
    for level, value in point["optional_bytes"]["cache-residency"].items():
        assert value == pytest.approx({"l1": 3840.0, "l2": 512.0, "l3plus": 2048.0}[level])
    assert point["optional_fractions"]["cache-residency"] == {
        "l1": pytest.approx(0.6),
        "l2": pytest.approx(0.08),
        "l3plus": pytest.approx(0.32),
    }


def _make_fake_backend_class() -> type:
    """Fake PAPIHLBackend: 4-event pool with capacity 2 -> two disjoint runs.

    FLOPS needs {PAPI_DP_OPS, PAPI_FP_OPS}; BYTES needs {PAPI_L1_DCA, PAPI_LST_INS}.
    Each run returns only its own events' counters, so merged counters carry both.
    """
    flops_impl = MetricDefinition(
        type=MetricType.FLOPS,
        required_events=frozenset({"PAPI_DP_OPS", "PAPI_FP_OPS"}),
        compute=lambda e, ctx: e["PAPI_FP_OPS"],
        priority=100,
        description="FLOPS",
    )
    bytes_impl = MetricDefinition(
        type=MetricType.BYTES,
        required_events=frozenset({"PAPI_L1_DCA", "PAPI_LST_INS"}),
        compute=lambda e, ctx: e["PAPI_L1_DCA"] * 8,
        priority=100,
        description="BYTES",
    )
    resolved = {MetricType.FLOPS: flops_impl, MetricType.BYTES: bytes_impl}

    class FakePAPIHLBackend:
        instances: ClassVar[list[FakePAPIHLBackend]] = []
        can_collect_calls: ClassVar[int] = 0
        check_prerequisites_calls: ClassVar[int] = 0

        def __init__(
            self,
            resolution_config: MetricResolutionConfig,
            *,
            use_cache: bool = True,
        ) -> None:
            self._resolution_config = resolution_config
            self._use_cache = use_cache
            self.profile_calls = 0
            self.received_specs: list[RunSpec] = []
            FakePAPIHLBackend.instances.append(self)

        @property
        def resolved_metrics(self) -> dict[MetricType, MetricDefinition]:
            return dict(resolved)

        @property
        def available_events(self) -> frozenset[str]:
            return frozenset({"PAPI_DP_OPS", "PAPI_FP_OPS", "PAPI_L1_DCA", "PAPI_LST_INS"})

        def can_collect(self, events: frozenset[str]) -> bool:
            FakePAPIHLBackend.can_collect_calls += 1
            return len(events) <= 2

        def check_prerequisites(self) -> bool:
            FakePAPIHLBackend.check_prerequisites_calls += 1
            return False

        def profile(self, run_spec: RunSpec, command: list[str], cwd: Path) -> RunResult:
            self.profile_calls += 1
            self.received_specs.append(run_spec)
            counters: dict[str, int] = {}
            if run_spec.events is not None and "PAPI_FP_OPS" in run_spec.events:
                counters["PAPI_DP_OPS"] = 1000
                counters["PAPI_FP_OPS"] = 1000
            if run_spec.events is not None and "PAPI_L1_DCA" in run_spec.events:
                counters["PAPI_L1_DCA"] = 100
                counters["PAPI_LST_INS"] = 200
            region = RegionMetrics(
                name="total", parent_region_id="-1", cycles=0, time_nsec=1_000_000_000, counters=counters
            )
            ranks = [RankMetrics(rank_id=0, threads=[ThreadMetrics(thread_id=0, regions=[region])])]
            return RunResult(exit_code=0, ranks=ranks)

        @property
        def run_method_name(self) -> str:
            return "fake"

    return FakePAPIHLBackend


def test_profile_main_merges_partitioned_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pool of 4 events with capacity 2 -> 2 app runs merged into one JSONL record."""
    fake_cls = _make_fake_backend_class()
    monkeypatch.setattr(papi_backend, "PAPIHLBackend", fake_cls)
    fake_cls.instances.clear()
    fake_cls.can_collect_calls = 0
    fake_cls.check_prerequisites_calls = 0

    args = _parse_profile_args(
        [
            "--backend",
            "papi",
            "--merge-runs",
            "--aggregation",
            "global",
            "--output-dir",
            str(tmp_path),
            "--",
            "./app",
        ]
    )
    config = ProfileConfig(args)
    assert profile_main(config) == 0

    # One session-scoped backend; each chunk is one profile() call with its own event set.
    assert len(fake_cls.instances) == 1
    backend = fake_cls.instances[0]
    assert [s.events for s in backend.received_specs] == [
        "PAPI_DP_OPS,PAPI_FP_OPS",
        "PAPI_L1_DCA,PAPI_LST_INS",
    ]
    assert backend.profile_calls == 2

    # Resolution is a one-time session probe: check_prerequisites runs once.
    assert fake_cls.check_prerequisites_calls == 1

    # Each run gets its own output directory so files never overwrite.
    assert len({s.output_dir for s in backend.received_specs}) == 2

    jsonl_path = tmp_path / config.machine_name / "applications.jsonl"
    lines = jsonl_path.read_text().strip().split("\n")
    assert len(lines) == 1  # one JSONL record despite two app runs

    record = json.loads(lines[0])
    assert record["metadata"]["notes"] == "merged from 2 runs"
    assert record["metadata"]["date"]  # fresh merged date
    assert record["optional_metrics"] == []

    point = record["points"][0]
    assert point["total_flops"] == pytest.approx(1000.0)  # from run 0's counters
    assert point["total_bytes"] == pytest.approx(800.0)  # from run 1's counters
    assert point["total_flops"] > 0
    assert point["total_bytes"] > 0


def test_profile_main_single_run_without_merge_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without --merge-runs the whole 4-event pool is one chunk: one run, no partitioning, no merge."""
    fake_cls = _make_fake_backend_class()
    monkeypatch.setattr(papi_backend, "PAPIHLBackend", fake_cls)
    fake_cls.instances.clear()
    fake_cls.can_collect_calls = 0
    fake_cls.check_prerequisites_calls = 0

    args = _parse_profile_args(
        ["--backend", "papi", "--aggregation", "global", "--output-dir", str(tmp_path), "--", "./app"]
    )
    config = ProfileConfig(args)
    assert profile_main(config) == 0

    # One session-scoped backend; the whole 4-event pool is one chunk in sorted order (D < F < L1 < LST).
    assert len(fake_cls.instances) == 1
    backend = fake_cls.instances[0]
    assert [s.events for s in backend.received_specs] == [
        "PAPI_DP_OPS,PAPI_FP_OPS,PAPI_L1_DCA,PAPI_LST_INS"
    ]
    assert backend.profile_calls == 1
    assert fake_cls.check_prerequisites_calls == 1
    assert fake_cls.can_collect_calls == 0  # partition logic never consulted

    jsonl_path = tmp_path / config.machine_name / "applications.jsonl"
    lines = jsonl_path.read_text().strip().split("\n")
    assert len(lines) == 1  # one JSONL record from the single run

    record = json.loads(lines[0])
    assert record["metadata"]["notes"] == ""
    assert record["optional_metrics"] == []

    point = record["points"][0]
    assert point["total_flops"] == pytest.approx(1000.0)
    assert point["total_bytes"] == pytest.approx(800.0)


# ---------------------------------------------------------------------------
# Perf backend multiplexing checks (perf_backend.py, perf_loader.py)
# ---------------------------------------------------------------------------


def test_multiplexed_events_full_run_fit() -> None:
    text = (
        "# started on Wed Aug 19 16:55:32 2026\n"
        "\n"
        "1552836,ns,duration_time,1552836,100,00,,\n"
        "260989,,ls_dispatch.ld_dispatch,550492,100,00,,\n"
        "10781,,ls_dispatch.ld_st_dispatch,550492,100,00,,\n"
    )
    assert multiplexed_events(text) == []


def test_multiplexed_events_full_run_scaled() -> None:
    text = (
        "850123,,cycles,73949,16,00,,\n"
        "888426,,instructions,439544,100,00,1,insn per cycle\n"
        "185958,,branches,439544,100,00,,\n"
    )
    assert multiplexed_events(text) == ["cycles"]


def test_multiplexed_events_full_run_not_counted() -> None:
    text = (
        "850123,,cycles,73949,100,00,,\n"
        "<not counted>,,stalled-cycles-frontend,0,0,00,,\n"
        "<not supported>,,ls_dispatch.ld_dispatch,0,100,00,,\n"
    )
    assert multiplexed_events(text) == ["ls_dispatch.ld_dispatch", "stalled-cycles-frontend"]


def test_multiplexed_events_interval_format() -> None:
    text = (
        "0.100118315,432081371,,cycles,40357201,40,00,,\n"
        "0.100118315,1185147799,,instructions,41396681,41,00,2,insn per cycle\n"
        "0.100118315,,,,,,0,stalled cycles per insn\n"
        "0.100118315,<not counted>,,cache-misses,0,0,00,,\n"
        "0.100118315,168534787,,ls_dispatch.store_dispatch,31978881,100,00,,\n"
    )
    assert multiplexed_events(text) == ["cache-misses", "cycles", "instructions"]


def test_multiplexed_events_empty_or_comment_only() -> None:
    assert multiplexed_events("") == []
    assert multiplexed_events("# only comments\n\n") == []


def _perf_backend_with_probe(monkeypatch: pytest.MonkeyPatch, fake_run: object) -> PerfBackend:
    """PerfBackend whose can_collect probe is replaced by *fake_run*."""
    monkeypatch.setattr(perf_backend.shutil, "which", lambda name: "/usr/bin/perf")
    monkeypatch.setattr(perf_backend.subprocess, "run", fake_run)
    return PerfBackend(MetricResolutionConfig())


def test_perf_can_collect_probe_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="1000,,cycles,550492,100,00,,\n", stderr="")

    backend = _perf_backend_with_probe(monkeypatch, fake_run)  # type: ignore[arg-type]
    assert backend.can_collect(frozenset({"cycles"})) is True
    assert calls == [
        ["/usr/bin/perf", "stat", "-x,", "-e", "duration_time,cycles", "--", "sleep", "0.05"]
    ]


def test_perf_can_collect_probe_rejects_multiplexed(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(args: list[str], **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "850123,,cycles,73949,16,00,,\n"
                "888426,,instructions,439544,100,00,1,insn per cycle\n"
            ),
            stderr="",
        )

    backend = _perf_backend_with_probe(monkeypatch, fake_run)  # type: ignore[arg-type]
    assert backend.can_collect(frozenset({"cycles", "instructions"})) is False


def test_perf_can_collect_probe_rejects_not_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(args: list[str], **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            returncode=0,
            stdout="<not counted>,,stalled-cycles-frontend,0,0,00,,\n",
            stderr="",
        )

    backend = _perf_backend_with_probe(monkeypatch, fake_run)  # type: ignore[arg-type]
    assert backend.can_collect(frozenset({"stalled-cycles-frontend"})) is False


def test_perf_can_collect_probe_nonzero_exit_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(args: list[str], **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=129, stdout="", stderr="invalid or unsupported event")

    backend = _perf_backend_with_probe(monkeypatch, fake_run)  # type: ignore[arg-type]
    assert backend.can_collect(frozenset({"cycles"})) is False


def test_perf_can_collect_probe_failure_is_optimistic(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(args: list[str], **kwargs: object) -> SimpleNamespace:
        raise subprocess.TimeoutExpired(args, timeout=10)

    backend = _perf_backend_with_probe(monkeypatch, fake_run)  # type: ignore[arg-type]
    assert backend.can_collect(frozenset({"cycles"})) is True


def test_perf_can_collect_missing_perf_is_optimistic(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(args: list[str], **kwargs: object) -> SimpleNamespace:
        raise AssertionError("probe must not run when perf is missing")

    monkeypatch.setattr(perf_backend.shutil, "which", lambda name: None)
    monkeypatch.setattr(perf_backend.subprocess, "run", fake_run)
    backend = PerfBackend(MetricResolutionConfig())
    assert backend.can_collect(frozenset({"cycles"})) is True


def test_perf_can_collect_empty_set_skips_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(args: list[str], **kwargs: object) -> SimpleNamespace:
        raise AssertionError("probe must not run for an empty event set")

    backend = _perf_backend_with_probe(monkeypatch, fake_run)  # type: ignore[arg-type]
    assert backend.can_collect(frozenset()) is True


def test_perf_partition_uses_probe_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    """partition_events with a real PerfBackend splits on probe-reported overcommit."""
    def fake_run(args: list[str], **kwargs: object) -> SimpleNamespace:
        events = next(a for a in args if a.startswith("duration_time,")).split(",")
        n = len(events) - 1  # drop duration_time
        rows = "\n".join(f"1000,,{e},550492,{100 if n <= 2 else 50},00,," for e in events[1:])
        return SimpleNamespace(returncode=0, stdout=rows + "\n", stderr="")

    backend = _perf_backend_with_probe(monkeypatch, fake_run)
    chunks = partition_events(["a", "b", "c", "d", "e"], backend.can_collect)
    assert chunks == [["a", "b"], ["c", "d"], ["e"]]


def test_perf_parse_output_warns_on_multiplexed_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A run whose CSV shows <not counted> events warns post-hoc but still parses."""
    warns: list[str] = []
    monkeypatch.setattr(perf_backend, "warn", lambda *args, **kwargs: warns.append(str(args[0])))
    backend = PerfBackend(MetricResolutionConfig())
    run_spec = RunSpec(output_dir=tmp_path, events="fp_ret_sse_avx_ops.all,ls_dispatch.ld_dispatch")
    (tmp_path / "perf_stat.csv").write_text(
        "1000,,fp_ret_sse_avx_ops.all,550492,100,00,,\n"
        "<not counted>,,ls_dispatch.ld_dispatch,0,0,00,,\n"
    )
    ranks = backend._parse_output(run_spec)
    assert any("time-multiplexed" in w and "ls_dispatch.ld_dispatch" in w for w in warns)
    assert ranks[0].threads[0].regions[0].counters == {"fp_ret_sse_avx_ops.all": 1000}


def test_perf_check_prerequisites_warns_when_resolved_set_overcommits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The resolved event set failing the probe warns during the session probe."""
    monkeypatch.setattr(shared, "perf_event_paranoid", lambda: 2)
    monkeypatch.setattr(perf_backend, "parse_perf_available_events", lambda: frozenset({"A", "B"}))
    impl = MetricDefinition(
        type=MetricType.FLOPS,
        required_events=frozenset({"A", "B"}),
        compute=lambda e, ctx: 0.0,
        priority=50,
        description="FLOPS",
    )
    monkeypatch.setattr(
        perf_backend, "resolve_perf_metrics", lambda available, config: {MetricType.FLOPS: impl}
    )
    monkeypatch.setattr(PerfBackend, "can_collect", lambda self, events: False)
    warns: list[str] = []
    monkeypatch.setattr(perf_backend, "warn", lambda *args, **kwargs: warns.append(str(args[0])))
    backend = PerfBackend(MetricResolutionConfig())
    assert backend.check_prerequisites() is True
    assert any("may not fit" in w for w in warns)


def test_perf_event_paranoid_reads_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paranoid_file = tmp_path / "perf_event_paranoid"
    paranoid_file.write_text("3\n")
    monkeypatch.setattr(shared, "_PERF_PARANOID_PATH", paranoid_file)
    assert shared.perf_event_paranoid() == 3


@pytest.mark.parametrize("exc", [FileNotFoundError, PermissionError, OSError])
def test_perf_event_paranoid_unreadable_returns_none(monkeypatch: pytest.MonkeyPatch, exc: type[OSError]) -> None:
    class UnreadablePath:
        def read_text(self, *args: object, **kwargs: object) -> str:
            raise exc()

    monkeypatch.setattr(shared, "_PERF_PARANOID_PATH", UnreadablePath())
    assert shared.perf_event_paranoid() is None


def test_perf_event_paranoid_non_integer_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paranoid_file = tmp_path / "perf_event_paranoid"
    paranoid_file.write_text("garbage\n")
    monkeypatch.setattr(shared, "_PERF_PARANOID_PATH", paranoid_file)
    assert shared.perf_event_paranoid() is None


@pytest.mark.parametrize("paranoid", [3, 4])
def test_check_perf_event_paranoid_raises_when_too_high(monkeypatch: pytest.MonkeyPatch, paranoid: int) -> None:
    monkeypatch.setattr(shared, "perf_event_paranoid", lambda: paranoid)
    with pytest.raises(UserError, match="perf_event_paranoid"):
        shared.check_perf_event_paranoid()


@pytest.mark.parametrize("paranoid", [-1, 0, 1, 2])
def test_check_perf_event_paranoid_accepts_2_or_lower(monkeypatch: pytest.MonkeyPatch, paranoid: int) -> None:
    monkeypatch.setattr(shared, "perf_event_paranoid", lambda: paranoid)
    warns: list[str] = []
    monkeypatch.setattr(shared, "warn", lambda *args, **kwargs: warns.append(str(args[0])))
    shared.check_perf_event_paranoid()
    assert not any("perf_event_paranoid" in w for w in warns)


def test_check_perf_event_paranoid_unreadable_warns_and_proceeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shared, "perf_event_paranoid", lambda: None)
    warns: list[str] = []
    monkeypatch.setattr(shared, "warn", lambda *args, **kwargs: warns.append(str(args[0])))
    shared.check_perf_event_paranoid()
    assert sum("perf_event_paranoid" in w for w in warns) == 1


@pytest.mark.parametrize("paranoid", [3, 4])
def test_perf_check_prerequisites_raises_when_paranoid_too_high(
    monkeypatch: pytest.MonkeyPatch, paranoid: int
) -> None:
    monkeypatch.setattr(shared, "perf_event_paranoid", lambda: paranoid)
    monkeypatch.setattr(perf_backend.shutil, "which", lambda name: "/usr/bin/perf")
    monkeypatch.setattr(
        perf_backend,
        "parse_perf_available_events",
        lambda: (_ for _ in ()).throw(AssertionError("must fail before discovery")),
    )
    with pytest.raises(UserError, match="perf_event_paranoid"):
        PerfBackend(MetricResolutionConfig()).check_prerequisites()


@pytest.mark.parametrize("paranoid", [-1, 0, 1, 2])
def test_perf_check_prerequisites_accepts_paranoid_2_or_lower(
    monkeypatch: pytest.MonkeyPatch, paranoid: int
) -> None:
    monkeypatch.setattr(shared, "perf_event_paranoid", lambda: paranoid)
    monkeypatch.setattr(perf_backend.shutil, "which", lambda name: "/usr/bin/perf")
    monkeypatch.setattr(perf_backend, "parse_perf_available_events", lambda: frozenset({"A"}))
    impl = MetricDefinition(
        type=MetricType.FLOPS,
        required_events=frozenset({"A"}),
        compute=lambda e, ctx: 0.0,
        priority=50,
        description="FLOPS",
    )
    monkeypatch.setattr(
        perf_backend, "resolve_perf_metrics", lambda available, config: {MetricType.FLOPS: impl}
    )
    monkeypatch.setattr(PerfBackend, "can_collect", lambda self, events: True)
    warns: list[str] = []
    monkeypatch.setattr(shared, "warn", lambda *args, **kwargs: warns.append(str(args[0])))
    backend = PerfBackend(MetricResolutionConfig())
    assert backend.check_prerequisites() is True
    assert not any("perf_event_paranoid" in w for w in warns)


def test_perf_check_prerequisites_unreadable_paranoid_warns_and_proceeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shared, "perf_event_paranoid", lambda: None)
    monkeypatch.setattr(perf_backend.shutil, "which", lambda name: "/usr/bin/perf")
    monkeypatch.setattr(perf_backend, "parse_perf_available_events", lambda: frozenset({"A"}))
    impl = MetricDefinition(
        type=MetricType.FLOPS,
        required_events=frozenset({"A"}),
        compute=lambda e, ctx: 0.0,
        priority=50,
        description="FLOPS",
    )
    monkeypatch.setattr(
        perf_backend, "resolve_perf_metrics", lambda available, config: {MetricType.FLOPS: impl}
    )
    monkeypatch.setattr(PerfBackend, "can_collect", lambda self, events: True)
    warns: list[str] = []
    monkeypatch.setattr(shared, "warn", lambda *args, **kwargs: warns.append(str(args[0])))
    backend = PerfBackend(MetricResolutionConfig())
    assert backend.check_prerequisites() is True
    assert sum("perf_event_paranoid" in w for w in warns) == 1


@pytest.mark.parametrize("paranoid", [3, 4])
def test_papi_check_prerequisites_raises_when_paranoid_too_high(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, paranoid: int
) -> None:
    _patch_papi_discovery(monkeypatch, tmp_path)
    monkeypatch.setattr(shared, "perf_event_paranoid", lambda: paranoid)
    monkeypatch.setattr(
        papi_backend,
        "parse_available_events",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must fail before discovery")),
    )
    with pytest.raises(UserError, match="perf_event_paranoid"):
        papi_backend.PAPIHLBackend(MetricResolutionConfig()).check_prerequisites()


@pytest.mark.parametrize("paranoid", [-1, 0, 1, 2])
def test_papi_check_prerequisites_accepts_paranoid_2_or_lower(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, paranoid: int
) -> None:
    _patch_papi_discovery(monkeypatch, tmp_path)
    monkeypatch.setattr(shared, "perf_event_paranoid", lambda: paranoid)
    monkeypatch.setattr(papi_backend, "validate_event_set", lambda events: True)
    warns: list[str] = []
    monkeypatch.setattr(shared, "warn", lambda *args, **kwargs: warns.append(str(args[0])))
    backend = papi_backend.PAPIHLBackend(MetricResolutionConfig())
    assert isinstance(backend.check_prerequisites(), bool)
    assert not any("perf_event_paranoid" in w for w in warns)


def test_papi_check_prerequisites_unreadable_paranoid_warns_and_proceeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_papi_discovery(monkeypatch, tmp_path)
    monkeypatch.setattr(shared, "perf_event_paranoid", lambda: None)
    monkeypatch.setattr(papi_backend, "validate_event_set", lambda events: True)
    warns: list[str] = []
    monkeypatch.setattr(shared, "warn", lambda *args, **kwargs: warns.append(str(args[0])))
    backend = papi_backend.PAPIHLBackend(MetricResolutionConfig())
    assert isinstance(backend.check_prerequisites(), bool)
    assert sum("perf_event_paranoid" in w for w in warns) == 1
