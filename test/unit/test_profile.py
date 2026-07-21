"""Unit tests for the profile package: model, aggregation, loaders, output, config."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import DataType

from profiling.aggregation import (
    AggregatedPoint,
    aggregate,
    aggregate_global,
    aggregate_per_rank,
    aggregate_per_region_merged,
    aggregate_per_region_per_thread,
    aggregate_per_thread,
)
from profiling.config import AggregationMode
from profiling.model import RegionMetrics, RunMetadata, RunResults, ThreadMetrics
from profiling.papi_loader import (
    RankMetrics,
    discover_rank_files,
    load_all_ranks,
    parse_rank_file,
)
from profiling.papi_metrics import (
    METRICS,
    _parse_papi_xml_output,
    build_isa_custom_metrics,
    fp_arith_counters_for_isas,
    PAPIMetricRegistry,
    resolve_metrics,
)
from profiling.shared import (
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
    from profiling.config import _default_app_name

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


def test_write_applications_csv(tmp_path: Path) -> None:
    from unittest.mock import MagicMock

    from profiling.output import write_applications_csv

    cfg = MagicMock(machine_name="test_run", output_dir=tmp_path)
    run = _make_sample_run()
    pts = [aggregate_global(run, SAMPLE_RESOLVED, DEFAULT_CTX)]
    write_applications_csv(pts, cfg, run)

    csv_path = tmp_path / "test_run" / "applications.csv"
    assert csv_path.exists()
    text = csv_path.read_text()
    assert "Date,Method,Name,ISA,Precision,Threads,AI,GFLOPS,Bandwidth,Time" in text
    assert "test" in text  # label


def test_write_applications_csv_per_rank(tmp_path: Path) -> None:
    from unittest.mock import MagicMock

    from profiling.output import write_applications_csv

    cfg = MagicMock(machine_name="test_run", output_dir=tmp_path)
    run = _make_sample_run()
    pts = aggregate_per_rank(run, SAMPLE_RESOLVED, DEFAULT_CTX)
    write_applications_csv(pts, cfg, run)

    csv_path = tmp_path / "test_run" / "applications.csv"
    assert csv_path.exists()
    rows = csv_path.read_text().strip().split("\n")
    assert len(rows) == 3  # header + 2 rank points


def test_write_profile_jsonl(tmp_path: Path) -> None:
    from unittest.mock import MagicMock

    from profiling.output import write_profile_jsonl

    cfg = MagicMock(machine_name="test_run", output_dir=tmp_path, aggregation=AggregationMode.GLOBAL)
    run = _make_sample_run()
    pts = [aggregate_global(run, SAMPLE_RESOLVED, DEFAULT_CTX)]
    write_profile_jsonl(run, cfg, pts)

    jsonl_path = tmp_path / "test_run" / "applications.jsonl"
    assert jsonl_path.exists()
    lines = jsonl_path.read_text().strip().split("\n")
    assert len(lines) == 1  # single line per run

    record = json.loads(lines[0])
    assert record["format_version"] == "2.0"
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

    from profiling.output import write_profile_jsonl

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

    from profiling.output import write_profile_jsonl

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
# ISA -> FP_ARITH counter mapping tests
# ---------------------------------------------------------------------------


def test_fp_arith_counters_for_isas_scalar() -> None:
    from isa.x86 import X86Scalar

    counters = fp_arith_counters_for_isas((X86Scalar,), DataType.f64)
    assert counters == {"FP_ARITH_INST_RETIRED:SCALAR_DOUBLE"}

    counters_f32 = fp_arith_counters_for_isas((X86Scalar,), DataType.f32)
    assert counters_f32 == {"FP_ARITH_INST_RETIRED:SCALAR_SINGLE"}


def test_fp_arith_counters_for_isas_sse() -> None:
    from isa.x86 import X86SSE

    counters = fp_arith_counters_for_isas((X86SSE,), DataType.f64)
    assert counters == {"FP_ARITH_INST_RETIRED:128B_PACKED_DOUBLE"}


def test_fp_arith_counters_for_isas_avx2() -> None:
    from isa.x86 import X86AVX2

    counters = fp_arith_counters_for_isas((X86AVX2,), DataType.f64)
    assert counters == {"FP_ARITH_INST_RETIRED:256B_PACKED_DOUBLE"}


def test_fp_arith_counters_for_isas_multiple() -> None:
    from isa.x86 import X86AVX2, X86SSE, X86Scalar

    counters = fp_arith_counters_for_isas(
        (X86AVX2, X86SSE, X86Scalar), DataType.f64
    )
    assert counters == {
        "FP_ARITH_INST_RETIRED:256B_PACKED_DOUBLE",
        "FP_ARITH_INST_RETIRED:128B_PACKED_DOUBLE",
        "FP_ARITH_INST_RETIRED:SCALAR_DOUBLE",
    }


def test_fp_arith_counters_for_isas_non_x86_returns_empty() -> None:
    from isa.arm import ArmNeon

    counters = fp_arith_counters_for_isas((ArmNeon,), DataType.f64)
    assert counters == set()


# ---------------------------------------------------------------------------
# Custom metric factory tests
# ---------------------------------------------------------------------------


def test_build_isa_custom_metrics_returns_correct_events() -> None:
    from isa.x86 import X86AVX2, X86SSE

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
    from isa.arm import ArmNeon

    result = build_isa_custom_metrics((ArmNeon,), DataType.f64)
    assert result is None


def test_build_isa_custom_metrics_empty_isas_returns_none() -> None:
    result = build_isa_custom_metrics((), DataType.f64)
    assert result is None


# ---------------------------------------------------------------------------
# End-to-end resolution with custom metrics
# ---------------------------------------------------------------------------


def test_resolve_metrics_with_custom_isa_outranks_default() -> None:
    from isa.x86 import X86AVX2

    available = frozenset({
        "FP_ARITH_INST_RETIRED:256B_PACKED_DOUBLE",
        "PAPI_LST_INS",
    })

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
