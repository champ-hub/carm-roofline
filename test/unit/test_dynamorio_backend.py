from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from carm_roofline.core import UserError
from carm_roofline.profiling.backends import RunSpec
from carm_roofline.profiling import dynamorio_backend as dynamorio_backend_module
from carm_roofline.profiling.dynamorio_backend import DynamoRIOBackend
from carm_roofline.profiling.dynamorio_loader import load_dynamorio_results
from carm_roofline.profiling.dynamorio_metrics import resolve_dynamorio_metrics
from carm_roofline.profiling.shared import MetricContext, MetricResolutionConfig, MetricType

pytestmark = pytest.mark.unit


def _record(*, timed: bool, stable: bool = True, elapsed_ns: int = 0) -> dict[str, object]:
    row: dict[str, object] = {
        "type": "region",
        "process_index": 0,
        "process_id": 17,
        "thread_index": 2,
        "thread_id": 99,
        "thread_id_stable": stable,
        "name": "kernel",
    }
    if timed:
        row["elapsed_ns"] = elapsed_ns
    else:
        row.update(
            {
                "instruction_counts": {"add_256_f32": 3, "fma_256_f64": 2},
                "read_bytes": 40,
                "write_bytes": 12,
                "elapsed_ns": None,
            }
        )
    return row


def _write_records(path: Path, *records: dict[str, object]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_dynamorio_loader_joins_native_times_on_stable_application_identity(tmp_path: Path) -> None:
    counts = tmp_path / "counts.jsonl"
    times = tmp_path / "times.jsonl"
    _write_records(counts, _record(timed=False))
    _write_records(times, _record(timed=True, elapsed_ns=1_250))

    ranks = load_dynamorio_results(counts, times)

    assert len(ranks) == 1
    assert ranks[0].rank_id == 0
    assert ranks[0].threads[0].thread_id == 2
    region = ranks[0].threads[0].regions[0]
    assert region.name == "kernel"
    assert region.time_nsec == 1_250
    assert region.counters["dr_add_256_f32"] == 3
    assert region.counters["dr_fma_256_f64"] == 2
    assert region.counters["dr_read_bytes"] == 40
    assert region.counters["dr_write_bytes"] == 12


def test_dynamorio_metrics_resolve_fma_lanes_and_memory_reference_bytes() -> None:
    resolved = resolve_dynamorio_metrics()
    context = MetricContext(MetricResolutionConfig())
    counters = {event: 0 for event in resolved[MetricType.FLOPS].required_events}
    counters.update({event: 0 for event in resolved[MetricType.BYTES].required_events})
    counters["dr_add_256_f32"] = 3
    counters["dr_fma_256_f64"] = 2
    counters["dr_read_bytes"] = 40
    counters["dr_write_bytes"] = 12

    assert resolved[MetricType.FLOPS].compute(counters, context) == 3 * 8 + 2 * 8
    assert resolved[MetricType.BYTES].compute(counters, context) == 52


def test_dynamorio_loader_rejects_unstable_thread_ids(tmp_path: Path) -> None:
    counts = tmp_path / "counts.jsonl"
    times = tmp_path / "times.jsonl"
    _write_records(counts, _record(timed=False, stable=False))
    _write_records(times, _record(timed=True))

    with pytest.raises(UserError, match="no stable application thread ID"):
        load_dynamorio_results(counts, times)


def test_dynamorio_loader_rejects_count_timing_identity_mismatch(tmp_path: Path) -> None:
    counts = tmp_path / "counts.jsonl"
    times = tmp_path / "times.jsonl"
    _write_records(counts, _record(timed=False))
    wrong_thread = _record(timed=True)
    wrong_thread["thread_id"] = 100
    _write_records(times, wrong_thread)

    with pytest.raises(UserError, match="identities do not match"):
        load_dynamorio_results(counts, times)

def test_dynamorio_loader_rejects_conflicting_thread_indexes(tmp_path: Path) -> None:
    counts = tmp_path / "counts.jsonl"
    times = tmp_path / "times.jsonl"
    first_count = _record(timed=False)
    second_count = _record(timed=False)
    second_count["name"] = "second"
    second_count["thread_id"] = 100
    first_timing = _record(timed=True, elapsed_ns=10)
    second_timing = dict(second_count)
    second_timing.pop("instruction_counts")
    second_timing.pop("read_bytes")
    second_timing.pop("write_bytes")
    second_timing["elapsed_ns"] = 20
    _write_records(counts, first_count, second_count)
    _write_records(times, first_timing, second_timing)

    with pytest.raises(UserError, match="maps a thread index/ID inconsistently"):
        load_dynamorio_results(counts, times)


def test_dynamorio_loader_sums_repeated_records_and_zero_fills_sparse_counters(tmp_path: Path) -> None:
    counts = tmp_path / "counts.jsonl"
    timings = tmp_path / "timings.jsonl"
    first_count = _record(timed=False)
    second_count = _record(timed=False)
    first_count["instruction_counts"] = {"add_256_f32": 3}
    second_count["instruction_counts"] = {"fma_256_f64": 2}
    _write_records(counts, first_count, second_count)
    _write_records(
        timings,
        _record(timed=True, elapsed_ns=1_250),
        _record(timed=True, elapsed_ns=750),
    )

    ranks = load_dynamorio_results(counts, timings)
    region = ranks[0].threads[0].regions[0]

    assert region.time_nsec == 2_000
    assert region.counters["dr_add_256_f32"] == 3
    assert region.counters["dr_fma_256_f64"] == 2
    assert region.counters["dr_mul_512_f64"] == 0
    assert region.counters["dr_read_bytes"] == 80
    assert region.counters["dr_write_bytes"] == 24


def test_dynamorio_loader_rejects_conflicting_process_indexes(tmp_path: Path) -> None:
    counts = tmp_path / "counts.jsonl"
    timings = tmp_path / "timings.jsonl"
    first_count = _record(timed=False)
    second_count = _record(timed=False)
    second_count["process_id"] = 18
    _write_records(counts, first_count, second_count)
    _write_records(timings, _record(timed=True), {**second_count, "elapsed_ns": 1})

    with pytest.raises(UserError, match="maps a process index/ID inconsistently"):
        load_dynamorio_results(counts, timings)


@pytest.mark.parametrize(
    ("timed", "elapsed_ns"),
    [(False, 1), (True, None)],
)
def test_dynamorio_loader_rejects_wrong_elapsed_ns_nullability(
    tmp_path: Path, timed: bool, elapsed_ns: object
) -> None:
    counts = tmp_path / "counts.jsonl"
    timings = tmp_path / "timings.jsonl"
    count_record = _record(timed=False)
    timing_record = _record(timed=True)
    if timed:
        timing_record["elapsed_ns"] = elapsed_ns
    else:
        count_record["elapsed_ns"] = elapsed_ns
    _write_records(counts, count_record)
    _write_records(timings, timing_record)

    expected_error = "invalid elapsed_ns" if timed else "must have elapsed_ns:null"
    with pytest.raises(UserError, match=expected_error):
        load_dynamorio_results(counts, timings)


def test_count_failure_prevents_native_execution_and_discards_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "dynamorio"
    drrun = home / "bin64" / "drrun"
    drrun.parent.mkdir(parents=True)
    drrun.touch()
    client = tmp_path / "client.so"
    client.touch()
    backend = DynamoRIOBackend(
        MetricResolutionConfig(),
        dynamorio_home=home,
        client_path=client,
    )
    backend.check_prerequisites()
    calls: list[list[str]] = []

    def failed_count(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[list[str]]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 9)

    monkeypatch.setattr(dynamorio_backend_module.subprocess, "run", failed_count)
    output_dir = tmp_path / "run"
    with pytest.raises(UserError, match="Native timing was not run"):
        backend.profile(
            run_spec=RunSpec(output_dir),
            command=["./app"],
            cwd=tmp_path,
        )

    assert len(calls) == 1
    assert not (output_dir / "dynamorio_counts.jsonl").exists()
    assert not (output_dir / "native_timing.jsonl").exists()


def test_explicit_client_override_is_used_without_building(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "dynamorio"
    drrun = home / "bin64" / "drrun"
    drrun.parent.mkdir(parents=True)
    drrun.touch()
    explicit_client = tmp_path / "explicit-client.so"
    explicit_client.touch()
    monkeypatch.setenv("CARM_DYNAMORIO_CLIENT", str(tmp_path / "environment-client.so"))

    def unexpected_build(_drrun: Path) -> Path:
        raise AssertionError("an explicit client override must not trigger an automatic build")

    monkeypatch.setattr(dynamorio_backend_module, "get_or_build_client", unexpected_build)
    reported_details: list[str] = []
    monkeypatch.setattr(dynamorio_backend_module, "detail", reported_details.append)
    backend = DynamoRIOBackend(
        MetricResolutionConfig(),
        dynamorio_home=home,
        client_path=explicit_client,
    )

    backend.check_prerequisites()

    assert f"DynamoRIO client: {explicit_client}" in reported_details


def test_environment_client_override_is_used_without_building(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "dynamorio"
    drrun = home / "bin64" / "drrun"
    drrun.parent.mkdir(parents=True)
    drrun.touch()
    client = tmp_path / "environment-client.so"
    client.touch()
    monkeypatch.setenv("CARM_DYNAMORIO_CLIENT", str(client))

    def unexpected_build(_drrun: Path) -> Path:
        raise AssertionError("an environment client override must not trigger an automatic build")

    monkeypatch.setattr(dynamorio_backend_module, "get_or_build_client", unexpected_build)
    reported_details: list[str] = []
    monkeypatch.setattr(dynamorio_backend_module, "detail", reported_details.append)
    backend = DynamoRIOBackend(MetricResolutionConfig(), dynamorio_home=home)

    backend.check_prerequisites()

    assert f"DynamoRIO client: {client}" in reported_details


def test_invalid_client_override_fails_without_automatic_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "dynamorio"
    drrun = home / "bin64" / "drrun"
    drrun.parent.mkdir(parents=True)
    drrun.touch()
    monkeypatch.setenv("CARM_DYNAMORIO_CLIENT", str(tmp_path / "missing-client.so"))

    def unexpected_build(_drrun: Path) -> Path:
        raise AssertionError("an invalid override must not trigger an automatic build")

    monkeypatch.setattr(dynamorio_backend_module, "get_or_build_client", unexpected_build)
    backend = DynamoRIOBackend(MetricResolutionConfig(), dynamorio_home=home)

    with pytest.raises(UserError, match="CARM_DYNAMORIO_CLIENT"):
        backend.check_prerequisites()
