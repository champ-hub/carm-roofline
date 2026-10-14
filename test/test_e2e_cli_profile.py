from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from carm_roofline import carm
from carm_roofline.profiling import papi_backend

pytestmark = pytest.mark.e2e

PAPI_EVENTS = frozenset({"PAPI_FP_OPS", "PAPI_L1_DCA", "PAPI_L1_DCM", "PAPI_L2_DCM"})


def _write_rank_file(output_dir: Path, events: frozenset[str], rank_id: int, thread_count: int) -> None:
    papi_dir = output_dir / "papi_hl_output"
    papi_dir.mkdir(parents=True, exist_ok=True)
    event_values = {
        "PAPI_FP_OPS": 100,
        "PAPI_L1_DCA": 10,
        "PAPI_L1_DCM": 2,
        "PAPI_L2_DCM": 1,
        "FP_ARITH_INST_RETIRED:256B_PACKED_DOUBLE": 10,
        "PAPI_LST_INS": 4,
    }
    regions: dict[str, dict[str, str]] = {}
    for region_id, (name, duration) in enumerate((("compute", 1_000_000_000), ("update", 2_000_000_000))):
        regions[str(region_id)] = {
            "name": name,
            "parent_region_id": "-1",
            "cycles": "0",
            "real_time_nsec": str(duration),
            **{event: str(value) for event, value in event_values.items() if event in events},
        }
    threads = {str(thread_id): {"regions": regions} for thread_id in range(thread_count)}
    record: dict[str, Any] = {
        "papi_version": "7.2.0.0",
        "cpu_info": "test CPU",
        "event_definitions": {event: {"component": "perf_event", "type": "delta"} for event in events},
        "threads": threads,
    }
    (papi_dir / f"rank_{rank_id:06d}.json").write_text(json.dumps(record), encoding="utf-8")


def _enable_papi(
    monkeypatch: pytest.MonkeyPatch,
    max_events_per_run: int = 3,
    command_args: tuple[str, ...] = ("--size", "128"),
    rank_counts: tuple[int, ...] = (2,),
    thread_counts: tuple[int, ...] = (2,),
    available_events: frozenset[str] = PAPI_EVENTS,
) -> list[str]:
    run_event_sets: list[str] = []
    monkeypatch.setattr(papi_backend, "_find_papi_library_path", lambda: None)
    monkeypatch.setattr(papi_backend.shutil, "which", lambda name: "/test/papi_hl_output_writer")
    monkeypatch.setattr(papi_backend, "check_perf_event_paranoid", lambda: None)
    monkeypatch.setattr(papi_backend, "parse_available_events", lambda use_cache=True: available_events)
    monkeypatch.setattr(papi_backend, "validate_event_set", lambda events: len(events) <= max_events_per_run)

    def run_application(
        command: list[str],
        *,
        env: dict[str, str],
        cwd: Path,
        capture_output: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        invocation = len(run_event_sets)
        assert command == ["./hydro-sim", *command_args]
        assert cwd.is_dir()
        assert capture_output is False
        assert check is False
        selected_events = frozenset(env["PAPI_EVENTS"].split(","))
        run_event_sets.append(env["PAPI_EVENTS"])
        output_dir = Path(env["PAPI_OUTPUT_DIRECTORY"])
        for rank_id in range(rank_counts[min(invocation, len(rank_counts) - 1)]):
            _write_rank_file(
                output_dir,
                selected_events,
                rank_id,
                thread_counts[min(invocation, len(thread_counts) - 1)],
            )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(papi_backend.subprocess, "run", run_application)
    return run_event_sets


def _record(output_dir: Path, machine: str) -> dict[str, Any]:
    path = output_dir / machine / "applications.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    return records[0]


@pytest.mark.parametrize(
    ("aggregation", "expected_labels"),
    [
        ("global", ["hydro-sim"]),
        ("rank", ["hydro-sim_rank0", "hydro-sim_rank1"]),
        (
            "thread",
            ["hydro-sim_rank0_thread0", "hydro-sim_rank0_thread1", "hydro-sim_rank1_thread0", "hydro-sim_rank1_thread1"],
        ),
        ("region_merged", ["hydro-sim_compute", "hydro-sim_update"]),
        (
            "region_per_thread",
            [
                f"hydro-sim_rank{rank}_thread{thread}_{region}"
                for rank in range(2)
                for thread in range(2)
                for region in ("compute", "update")
            ],
        ),
    ],
)
def test_papi_cli_parses_rank_files_aggregates_and_writes_consumer_jsonl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    aggregation: str,
    expected_labels: list[str],
) -> None:
    """Exercise CLI, real PAPI parser/metric aggregation/output across MPI × thread × region data."""
    run_event_sets = _enable_papi(monkeypatch)
    exit_code = carm.main(
        [
            "profile",
            "--aggregation",
            aggregation,
            "--machine-name",
            "test-node",
            "--output-dir",
            str(tmp_path),
            "--",
            "./hydro-sim",
            "--size",
            "128",
        ]
    )

    assert exit_code == 0
    assert len(run_event_sets) == 1
    assert set(run_event_sets[0].split(",")) == {"PAPI_FP_OPS", "PAPI_L1_DCA"}
    record = _record(tmp_path, "test-node")
    assert record["format_version"] == "3.0"
    assert record["aggregation"] == aggregation
    assert record["metadata"]["name"] == "hydro-sim"
    assert record["metadata"]["command"] == "./hydro-sim --size 128"
    assert record["metadata"]["method"] == "PAPI HL"
    assert (record["metadata"]["num_ranks"], record["metadata"]["total_threads"]) == (2, 4)
    points = record["points"]
    assert [point["label"] for point in points] == expected_labels
    assert all(point["total_flops"] > 0 and point["total_bytes"] > 0 for point in points)
    assert (tmp_path / "test-node" / "machine.json").is_file()
    from carm_roofline.roofline_assembly import load_applications

    consumer_records = load_applications(tmp_path / "test-node" / "applications.jsonl", machine="test-node")
    assert len(consumer_records) == 1
    assert len(consumer_records[0].points) == len(points)
    assert consumer_records[0].points[0].total_flops == pytest.approx(points[0]["total_flops"])
    assert consumer_records[0].points[0].total_bytes == pytest.approx(points[0]["total_bytes"])

    if aggregation == "global":
        assert points[0]["total_flops"] == pytest.approx(800)
        assert points[0]["total_bytes"] == pytest.approx(640)
        assert points[0]["runtime_s"] == pytest.approx(3)
        assert (points[0]["num_ranks"], points[0]["num_threads"], points[0]["num_regions"]) == (2, 4, 8)
    elif aggregation == "rank":
        assert all(point["total_flops"] == pytest.approx(400) for point in points)
        assert all(point["total_bytes"] == pytest.approx(320) for point in points)
        assert all((point["num_ranks"], point["num_threads"], point["num_regions"]) == (1, 2, 4) for point in points)
    elif aggregation == "thread":
        assert all(point["total_flops"] == pytest.approx(200) for point in points)
        assert all(point["total_bytes"] == pytest.approx(160) for point in points)
        assert all((point["num_ranks"], point["num_threads"], point["num_regions"]) == (1, 1, 2) for point in points)
    elif aggregation == "region_merged":
        assert [point["total_flops"] for point in points] == pytest.approx([400, 400])
        assert [point["total_bytes"] for point in points] == pytest.approx([320, 320])
        assert [point["runtime_s"] for point in points] == pytest.approx([1, 2])
        assert all((point["num_ranks"], point["num_threads"], point["num_regions"]) == (2, 4, 4) for point in points)
    else:
        assert all(point["num_ranks"] == 1 and point["num_threads"] == 1 for point in points)
        assert [point["total_flops"] for point in points] == pytest.approx([100] * 8)
        assert all(point["num_regions"] == 1 for point in points)


@pytest.mark.parametrize(("ranks", "threads"), [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_papi_cli_preserves_serial_threaded_mpi_and_hybrid_run_shapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ranks: int,
    threads: int,
) -> None:
    _enable_papi(monkeypatch, command_args=(), rank_counts=(ranks,), thread_counts=(threads,))
    assert carm.main(
        ["profile", "--machine-name", "mode-node", "--output-dir", str(tmp_path), "--", "./hydro-sim"]
    ) == 0

    record = _record(tmp_path, "mode-node")
    assert record["metadata"]["num_ranks"] == ranks
    assert record["metadata"]["total_threads"] == ranks * threads
    point = record["points"][0]
    assert point["total_flops"] == pytest.approx(200 * ranks * threads)
    assert point["total_bytes"] == pytest.approx(160 * ranks * threads)
    assert point["runtime_s"] == pytest.approx(3)


def test_papi_cli_merges_counter_budget_partitions_before_aggregation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_event_sets = _enable_papi(monkeypatch, max_events_per_run=1, command_args=())
    exit_code = carm.main(
        [
            "profile",
            "--merge-runs",
            "--metrics",
            "cache-line-utilization",
            "--aggregation",
            "global",
            "--machine-name",
            "merged-node",
            "--output-dir",
            str(tmp_path),
            "--",
            "./hydro-sim",
        ]
    )

    assert exit_code == 0
    assert len(run_event_sets) == 3
    assert len({event for group in run_event_sets for event in group.split(",")}) == 3
    record = _record(tmp_path, "merged-node")
    assert record["metadata"]["notes"] == "merged from 3 runs"
    assert (record["metadata"]["num_ranks"], record["metadata"]["total_threads"]) == (2, 4)
    point = record["points"][0]
    assert point["total_flops"] == pytest.approx(800)
    assert point["total_bytes"] == pytest.approx(640)
    assert point["runtime_s"] == pytest.approx(3)
    assert point["optional_fractions"]["cache-line-utilization"]["value"] == pytest.approx(0.625)


def test_papi_cli_resolves_isa_and_precision_specific_counters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    available = frozenset({"FP_ARITH_INST_RETIRED:256B_PACKED_DOUBLE", "PAPI_LST_INS"})
    run_event_sets = _enable_papi(monkeypatch, max_events_per_run=2, available_events=available)
    assert carm.main(
        [
            "profile",
            "--isa",
            "x86_avx2",
            "--data-type",
            "f64",
            "--machine-name",
            "isa-node",
            "--output-dir",
            str(tmp_path),
            "--",
            "./hydro-sim",
            "--size",
            "128",
        ]
    ) == 0

    assert len(run_event_sets) == 1
    assert set(run_event_sets[0].split(",")) == available
    point = _record(tmp_path, "isa-node")["points"][0]
    assert point["total_flops"] == pytest.approx(320)
    assert point["total_bytes"] == pytest.approx(1024)


def test_papi_cli_aggregates_cache_residency_into_gui_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_papi(monkeypatch, max_events_per_run=4)
    assert carm.main(
        [
            "profile",
            "--metrics",
            "cache-residency",
            "--machine-name",
            "cache-node",
            "--output-dir",
            str(tmp_path),
            "--",
            "./hydro-sim",
            "--size",
            "128",
        ]
    ) == 0

    point = _record(tmp_path, "cache-node")["points"][0]
    assert point["total_flops"] == pytest.approx(800)
    assert point["total_bytes"] == pytest.approx(640)
    assert point["optional_bytes"]["cache-residency"] == pytest.approx(
        {"l1": 0, "l2": 128, "l3plus": 512}
    )
    assert point["optional_fractions"]["cache-residency"] == pytest.approx(
        {"l1": 0, "l2": 0.2, "l3plus": 0.8}
    )


def test_papi_cli_rejects_rank_shape_changes_between_merged_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_event_sets = _enable_papi(monkeypatch, max_events_per_run=1, command_args=(), rank_counts=(2, 1))
    exit_code = carm.main(
        [
            "profile",
            "--merge-runs",
            "--metrics",
            "cache-line-utilization",
            "--machine-name",
            "mismatch-node",
            "--output-dir",
            str(tmp_path),
            "--",
            "./hydro-sim",
        ]
    )

    assert exit_code == 1
    assert len(run_event_sets) == 3
    assert not (tmp_path / "mismatch-node" / "applications.jsonl").exists()


def test_profile_cli_rejects_missing_application_and_exposes_workflow_options(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert carm.main(["profile"]) != 0
    assert carm.main(["profile", "--"]) != 0

    with pytest.raises(SystemExit) as exc_info:
        carm.main(["profile", "--help"])
    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    for option in ("--backend", "--aggregation", "--merge-runs", "--metrics", "--perf-interval", "--isa", "--data-type"):
        assert option in help_text


@pytest.mark.parametrize(
    ("interval", "expected_method", "expected_points"),
    [(None, "perf full-run", 1), ("100", "perf interval", 2)],
)
def test_perf_cli_parses_backend_csv_for_full_and_interval_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interval: str | None,
    expected_method: str,
    expected_points: int,
) -> None:
    from carm_roofline.profiling import perf_backend

    available = frozenset(
        {
            "fp_ret_sse_avx_ops.all",
            "ls_dispatch.ld_dispatch",
            "ls_dispatch.store_dispatch",
            "ls_dispatch.ld_st_dispatch",
        }
    )
    monkeypatch.setattr(perf_backend.shutil, "which", lambda name: "/test/perf")
    monkeypatch.setattr(perf_backend, "check_perf_event_paranoid", lambda: None)
    monkeypatch.setattr(perf_backend, "parse_perf_available_events", lambda: available)
    monkeypatch.setattr(perf_backend.PerfBackend, "can_collect", lambda self, events: True)
    observed: list[list[str]] = []

    def run_perf(
        command: list[str],
        *,
        cwd: Path,
        capture_output: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        observed.append(command)
        assert cwd.is_dir()
        assert capture_output is False
        assert check is False
        output_path = Path(command[command.index("-o") + 1])
        if "-I" in command:
            assert command[command.index("-I") + 1] == interval
            rows = [
                "0.100,100,,fp_ret_sse_avx_ops.all",
                "0.100,10,,ls_dispatch.ld_dispatch",
                "0.100,2,,ls_dispatch.store_dispatch",
                "0.100,1,,ls_dispatch.ld_st_dispatch",
                "0.200,200,,fp_ret_sse_avx_ops.all",
                "0.200,20,,ls_dispatch.ld_dispatch",
                "0.200,4,,ls_dispatch.store_dispatch",
                "0.200,2,,ls_dispatch.ld_st_dispatch",
            ]
        else:
            assert "-I" not in command
            rows = [
                "300,,duration_time",
                "300,,fp_ret_sse_avx_ops.all",
                "30,,ls_dispatch.ld_dispatch",
                "6,,ls_dispatch.store_dispatch",
                "3,,ls_dispatch.ld_st_dispatch",
            ]
        output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(perf_backend.subprocess, "run", run_perf)
    args = [
        "profile",
        "--backend",
        "perf",
        "--machine-name",
        "perf-node",
        "--output-dir",
        str(tmp_path),
    ]
    if interval is not None:
        args.extend(["--perf-interval", interval, "--aggregation", "region_merged"])
    args.extend(["--", "./hydro-sim", "--size", "128"])

    assert carm.main(args) == 0
    assert len(observed) == 1
    assert observed[0][-3:] == ["./hydro-sim", "--size", "128"]
    record = _record(tmp_path, "perf-node")
    assert record["metadata"]["method"] == expected_method
    points = record["points"]
    assert len(points) == expected_points
    assert sum(point["total_flops"] for point in points) == pytest.approx(300)
    assert sum(point["total_bytes"] for point in points) == pytest.approx(336)
    if interval:
        assert [point["label"] for point in points] == ["hydro-sim_sample_0", "hydro-sim_sample_1"]
        assert [point["runtime_s"] for point in points] == pytest.approx([0.1, 0.1])
    else:
        assert points[0]["label"] == "hydro-sim"
        assert points[0]["runtime_s"] == pytest.approx(3e-7)
