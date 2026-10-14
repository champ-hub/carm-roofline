from __future__ import annotations

import json
import math
import platform
import shutil
from pathlib import Path

import pytest

from carm_roofline.carm import main


pytestmark = [pytest.mark.e2e, pytest.mark.native]

@pytest.mark.skipif(platform.machine().lower() not in {"x86_64", "amd64"}, reason="requires native x86-64 execution")
@pytest.mark.skipif(shutil.which("gcc") is None, reason="requires gcc to compile the generated benchmark")
@pytest.mark.parametrize(
    ("test_type", "extra_args", "record_type", "row_count"),
    [
        ("arithmetic", ["--instruction", "add", "--num-ops", "32"], "arithmetic", 1),
        ("memory", ["--mem-target", "L1", "--mem-test-sizes", "4KiB"], "memory", 1),
        (
            "mixed",
            [
                "--instruction",
                "add",
                "--num-ops",
                "32",
                "--mem-target",
                "L1",
                "--mem-test-sizes",
                "4KiB",
                "--ai-range",
                "1",
                "2",
                "--ai-points",
                "2",
                "--arith-mem-ratio",
                "1:1",
            ],
            "mixed",
            1,
        ),
        (
            "roofline",
            [
                "--instruction",
                "add",
                "--num-ops",
                "32",
                "--mem-target",
                "L1",
                "--mem-test-sizes",
                "4KiB",
            ],
            None,
            2,
        ),
    ],
)
def test_generated_benchmark_compiles_runs_and_reports_measurements(
    tmp_path: Path, test_type: str, extra_args: list[str], record_type: str | None, row_count: int
) -> None:
    """Run generated x86 scalar kernels through header emission, native compile and measurement."""
    output_dir = tmp_path / "results"
    name = f"codegen-{test_type}"

    assert main(
        [
            "benchmark",
            "--isa",
            "x86",
            "--test",
            test_type,
            "--data-type",
            "f32",
            "--threads",
            "1",
            "--frequency",
            "2GHz",
            "--test-time",
            "0.1",
            "--name",
            name,
            "--output-dir",
            str(output_dir),
            "--output-fmt",
            "jsonl",
            *extra_args,
        ]
    ) == 0

    result_files = list((output_dir / name).glob("*.jsonl"))
    assert len(result_files) == 1
    records = [json.loads(line) for line in result_files[0].read_text(encoding="utf-8").splitlines()]
    assert len(records) == row_count
    expected_types = {record_type} if record_type is not None else {"arithmetic", "memory"}
    assert {record["type"] for record in records} == expected_types
    assert {record["isa"] for record in records} == {"x86"}
    assert all(record["data_type"] == "f32" and record["num_threads"] == 1 for record in records)
    if test_type != "mixed":
        assert all(record["repetitions"] > 0 and record["time_seconds"] > 0 for record in records)

    if test_type == "arithmetic":
        record = records[0]
        assert record["operation"] == "add"
        assert record["num_ops"] == 32
        assert math.isclose(
            record["performance_gops"],
            record["num_ops"] * record["repetitions"] / record["time_seconds"] / 1e9,
            rel_tol=1e-9,
        )
    elif test_type == "memory":
        record = records[0]
        assert record["cache_level"] == "L1"
        assert 0 < record["working_set_bytes"] <= 4096
        assert math.isclose(
            record["bandwidth_gbps"],
            record["working_set_bytes"] * record["repetitions"] / record["time_seconds"] / 1e9,
            rel_tol=1e-9,
        )
    elif test_type == "mixed":
        series = records[0]
        assert series["cache_level"] == "L1"
        assert series["operation"] == "add"
        assert 0 < series["working_set_bytes"] <= 4096
        points = series["points"]
        assert len(points) == 2
        assert {point["point_index"] for point in points} == {0, 1}
        assert {point["requested_arithmetic_intensity"] for point in points} == {1.0, 2.0}
        for point in points:
            assert point["repetitions"] > 0 and point["time_seconds"] > 0
            assert point["operations_per_thread"] > 0
            assert point["performance_gops"] > 0
            assert math.isclose(
                point["performance_gops"],
                point["operations_per_thread"] * point["repetitions"] / point["time_seconds"] / 1e9,
                rel_tol=1e-9,
            )
    else:
        arithmetic = [record for record in records if record["type"] == "arithmetic"]
        memory = [record for record in records if record["type"] == "memory"]
        assert len(arithmetic) == len(memory) == 1
        arith_record = arithmetic[0]
        assert arith_record["operation"] == "add"
        assert arith_record["num_ops"] == 32
        assert math.isclose(
            arith_record["performance_gops"],
            arith_record["num_ops"] * arith_record["repetitions"] / arith_record["time_seconds"] / 1e9,
            rel_tol=1e-9,
        )
        mem_record = memory[0]
        assert mem_record["cache_level"] == "L1"
        assert 0 < mem_record["working_set_bytes"] <= 4096
        assert math.isclose(
            mem_record["bandwidth_gbps"],
            mem_record["working_set_bytes"] * mem_record["repetitions"] / mem_record["time_seconds"] / 1e9,
            rel_tol=1e-9,
        )

@pytest.mark.skipif(platform.machine().lower() not in {"x86_64", "amd64"}, reason="requires native x86-64 execution")
@pytest.mark.skipif(shutil.which("gcc") is None, reason="requires gcc to compile the generated benchmark")
def test_detected_avx2_benchmark_compiles_and_runs(tmp_path: Path) -> None:
    """Exercise vector code generation only when native CPU detection reports AVX2 support."""
    from carm_roofline.architecture.detect import native_detect

    detected = native_detect()
    if detected.isa is None or "x86_avx2" not in detected.isa:
        pytest.skip("native CPU detection did not report x86_avx2")

    output_dir = tmp_path / "results"
    assert main(
        [
            "benchmark",
            "--isa",
            "x86_avx2",
            "--test",
            "arithmetic",
            "--data-type",
            "f32",
            "--threads",
            "1",
            "--instruction",
            "add",
            "--num-ops",
            "32",
            "--frequency",
            "2GHz",
            "--test-time",
            "0.1",
            "--name",
            "codegen-avx2",
            "--output-dir",
            str(output_dir),
            "--output-fmt",
            "jsonl",
        ]
    ) == 0

    result_files = list((output_dir / "codegen-avx2").glob("*.jsonl"))
    assert len(result_files) == 1
    records = [json.loads(line) for line in result_files[0].read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["type"] == "arithmetic"
    assert record["isa"] == "x86_avx2"
    assert record["data_type"] == "f32"
    assert record["operation"] == "add"
    assert record["num_ops"] == 32
    assert record["num_threads"] == 1
    assert record["repetitions"] > 0 and record["time_seconds"] > 0
    assert math.isclose(
        record["performance_gops"],
        record["num_ops"] * record["repetitions"] / record["time_seconds"] / 1e9,
        rel_tol=1e-9,
    )
