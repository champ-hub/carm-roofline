from __future__ import annotations

import csv
import io
import json
import shutil
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path

from PIL import Image
from rich.console import Console

import pytest

import carm_roofline.architecture.architecture as architecture_module
import carm_roofline.benchmark.interface as benchmark_interface
import carm_roofline.output_utils as output_utils
from carm_roofline.architecture.detect import DetectedArchitecture
from carm_roofline.architecture.memory import SimpleMemoryTopology
from carm_roofline.carm import main
from carm_roofline.core import Bytes, Frequency
from carm_roofline.test_bench.builder import MicrobenchmarkFunctionSpec

pytestmark = pytest.mark.e2e


def _detected_architecture() -> DetectedArchitecture:
    return DetectedArchitecture(
        isa=["x86_sse"],
        arch="x86_64",
        vendor="test-vendor",
        model_name="test-cpu",
        memory_topology=SimpleMemoryTopology(
            [Bytes.from_string("32KiB"), Bytes.from_string("256KiB"), Bytes.from_string("8MiB")],
            [1, 1, 1],
            total_cpus=2,
        ),
        frequency=Frequency(2_000_000_000),
        frequency_nominal=Frequency(2_000_000_000),
    )


def _fake_compile(
    _context: object, output_path: Path, include_dirs: Iterable[Path] | None = None
) -> Path:
    assert include_dirs is not None
    assert (next(iter(include_dirs)) / "microbenchmarks.h").is_file()
    output_path.touch()
    return output_path


def _fake_measure(
    _context: object,
    _binary_path: Path,
    specs: Iterable[MicrobenchmarkFunctionSpec],
    on_benchmarks_complete: Callable[[int], None] | None = None,
) -> str:
    names = [spec.function_name for spec in specs]
    if on_benchmarks_complete is not None:
        on_benchmarks_complete(len(names))
    return "".join(f"{name}, 2.5, 100\n" for name in names)


def test_cli_benchmark_runs_generated_matrix_and_roundtrips_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    request: pytest.FixtureRequest,
) -> None:
    monkeypatch.setattr(architecture_module, "detect_for_isa", lambda _isa, threads: _detected_architecture())
    generated_headers: list[Path] = []
    request.addfinalizer(
        lambda: shutil.rmtree(generated_headers[0].parent, ignore_errors=True) if generated_headers else None
    )

    def compile_generated_header(
        context: object, output_path: Path, include_dirs: Iterable[Path] | None = None
    ) -> Path:
        assert include_dirs is not None
        header = next(iter(include_dirs)) / "microbenchmarks.h"
        assert header.is_file()
        generated_headers.append(header)
        return _fake_compile(context, output_path, include_dirs)

    monkeypatch.setattr(benchmark_interface, "compile_test_bench", compile_generated_header)
    monkeypatch.setattr(benchmark_interface, "run_microbenchmarks", _fake_measure)

    output_dir = tmp_path / "results"
    assert main(
        [
            "benchmark",
            "--isa",
            "x86_sse",
            "--test",
            "arithmetic",
            "--instruction",
            "add",
            "--data-type",
            "f32",
            "f64",
            "--threads",
            "1",
            "2",
            "--num-ops",
            "128",
            "--frequency",
            "2GHz",
            "--name",
            "e2e-arithmetic",
            "--output-dir",
            str(output_dir),
            "--output-fmt",
            "jsonl",
            "--progress-format",
            "jsonl",
            "--keep-artifacts",
        ]
    ) == 0

    assert len(generated_headers) == 1
    assert generated_headers[0].is_file()
    result_path = output_dir / "e2e-arithmetic" / "benchmarks.jsonl"
    records = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 4
    assert {(record["data_type"], record["num_threads"]) for record in records} == {
        ("f32", 1),
        ("f32", 2),
        ("f64", 1),
        ("f64", 2),
    }
    for record in records:
        assert record["isa"] == "x86_sse"
        assert record["operation"] == "add"
        assert record["repetitions"] == 100
        assert record["time_seconds"] == pytest.approx(0.0025)
        expected_performance = (
            record["num_ops"] * record["repetitions"] * record["num_threads"] / record["time_seconds"] / 1e9
        )
        assert record["performance_gops"] == pytest.approx(expected_performance)

    progress_lines = [
        line.removeprefix("CARM_PROGRESS ")
        for line in capsys.readouterr().err.splitlines()
        if line.startswith("CARM_PROGRESS ")
    ]
    progress_events = [json.loads(line) for line in progress_lines]
    stage_events = [event["stage"] for event in progress_events if event.get("event") == "stage"]
    assert stage_events == ["initializing", "generating", "compiling", "running", "parsing", "writing"]
    measurements = [event for event in progress_events if event.get("event") == "measurement"]
    assert measurements == [{"event": "measurement", "completed": 4, "total": 4}]
    assert progress_events[-1] == {"event": "complete"}


def test_cli_memory_benchmark_uses_topology_and_computes_bandwidth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(architecture_module, "detect_for_isa", lambda _isa, threads: _detected_architecture())
    monkeypatch.setattr(benchmark_interface, "compile_test_bench", _fake_compile)
    monkeypatch.setattr(benchmark_interface, "run_microbenchmarks", _fake_measure)
    topology_config = tmp_path / "topology.toml"
    topology_config.write_text(
        'total_cpus = 2\nsmt_degree = 1\ncpu_offset = 0\n'
        '[[cache_levels]]\nlevel = 1\ninstances = 2\nsize = "32KiB"\n'
        '[[cache_levels]]\nlevel = 2\ninstances = 1\nsize = "8MiB"\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "results"

    assert main(
        [
            "benchmark",
            "--isa",
            "x86_sse",
            "--test",
            "memory",
            "--mem-target",
            "L1",
            "--mem-test-sizes",
            "16KiB",
            "--ld-st-ratio",
            "1:0",
            "--topology-config",
            str(topology_config),
            "--frequency",
            "2GHz",
            "--name",
            "e2e-memory",
            "--output-dir",
            str(output_dir),
            "--output-fmt",
            "jsonl",
        ]
    ) == 0

    path = output_dir / "e2e-memory" / "benchmarks.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["memory_level_name"] == "L1"
    assert record["load_store_ratio"] == "1:0"
    assert record["size_per_thread_bytes"] == 16 * 1024
    assert record["repetitions"] == 100
    expected_bandwidth_gbps = 16 * 1024 * 100 / 0.0025 / 1e9
    assert record["bandwidth_gbps"] == pytest.approx(expected_bandwidth_gbps)


@pytest.mark.parametrize(
    ("test_type", "extra_args"),
    [
        ("arithmetic", ["--instruction", "add", "--num-ops", "128"]),
        ("mixed", ["--ai-range", "1", "4", "--ai-points", "2", "--arith-mem-ratio", "1:2"]),
    ],
)
def test_cli_benchmark_dry_run_retains_artifacts_without_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    request: pytest.FixtureRequest,
    test_type: str,
    extra_args: list[str],
) -> None:
    retained_workspaces: list[Path] = []
    request.addfinalizer(
        lambda: shutil.rmtree(retained_workspaces[0], ignore_errors=True) if retained_workspaces else None
    )
    original_mkdtemp = tempfile.mkdtemp

    def retain_workspace(prefix: str) -> str:
        workspace = original_mkdtemp(prefix=prefix)
        retained_workspaces.append(Path(workspace))
        return workspace

    monkeypatch.setattr(tempfile, "mkdtemp", retain_workspace)
    monkeypatch.setattr(architecture_module, "detect_for_isa", lambda _isa, threads: _detected_architecture())
    monkeypatch.setattr(
        benchmark_interface,
        "compile_test_bench",
        lambda *_args, **_kwargs: pytest.fail("dry-run must not compile"),
    )
    monkeypatch.setattr(
        benchmark_interface,
        "run_microbenchmarks",
        lambda *_args, **_kwargs: pytest.fail("dry-run must not execute benchmarks"),
    )

    assert main(
        [
            "benchmark",
            "--isa",
            "x86_sse",
            "--test",
            test_type,
            "--verbose",
            "3",
            "--frequency",
            "2GHz",
            "--name",
            f"dry-run-{test_type}",
            "--dry-run",
            "--keep-artifacts",
            "--output-dir",
            str(tmp_path),
            *extra_args,
        ]
    ) == 0
    assert len(retained_workspaces) == 1
    assert retained_workspaces[0].is_dir()
    assert (retained_workspaces[0] / "microbenchmarks.h").is_file()
    assert not (retained_workspaces[0] / "benchmark").exists()
    assert not (tmp_path / f"dry-run-{test_type}" / "benchmarks.jsonl").exists()


def test_cli_compiler_and_simulator_execute_generated_benchmark(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(architecture_module, "detect_for_isa", lambda _isa, threads: _detected_architecture())
    compiler = tmp_path / "test-compiler"
    compiler.write_text(
        "#!/bin/sh\n"
        "while [ \"$1\" != \"-o\" ]; do shift; done\n"
        "shift\n"
        "cat > \"$1\" <<'EOF'\n"
        "#!/bin/sh\n"
        "printf '%s\\n' 'x86_sse_arith_add_f32_t1, 2.5, 100'\n"
        "EOF\n"
        "chmod +x \"$1\"\n",
        encoding="utf-8",
    )
    compiler.chmod(0o755)
    simulator = tmp_path / "test-simulator"
    simulator.write_text("#!/bin/sh\nexec \"$@\"\n", encoding="utf-8")
    simulator.chmod(0o755)
    output_dir = tmp_path / "simulated-results"

    assert main(
        [
            "benchmark",
            "--isa",
            "x86_sse",
            "--test",
            "arithmetic",
            "--instruction",
            "add",
            "--data-type",
            "f32",
            "--threads",
            "1",
            "--num-ops",
            "128",
            "--frequency",
            "2GHz",
            "--compiler",
            str(compiler),
            "--sim-cmd",
            f"{simulator} {{binary}}",
            "--name",
            "simulated-arithmetic",
            "--output-dir",
            str(output_dir),
            "--output-fmt",
            "jsonl",
        ]
    ) == 0

    result_path = output_dir / "simulated-arithmetic" / "benchmarks.jsonl"
    records = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["name"] == "x86_sse_arith_add_f32_t1"
    assert records[0]["repetitions"] == 100

def test_cli_memory_sweep_emits_ordered_measurements(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(architecture_module, "detect_for_isa", lambda _isa, threads: _detected_architecture())
    monkeypatch.setattr(benchmark_interface, "compile_test_bench", _fake_compile)
    monkeypatch.setattr(benchmark_interface, "run_microbenchmarks", _fake_measure)
    output_dir = tmp_path / "results"

    assert main(
        [
            "benchmark",
            "--isa",
            "x86_sse",
            "--test",
            "memory_sweep",
            "--ld-st-ratio",
            "1:0",
            "--data-type",
            "f32",
            "--frequency",
            "2GHz",
            "--name",
            "e2e-memory-sweep",
            "--output-dir",
            str(output_dir),
            "--output-fmt",
            "jsonl",
        ]
    ) == 0

    result_path = output_dir / "e2e-memory-sweep" / "benchmarks.jsonl"
    records = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) > 1
    working_sets = [record["working_set_bytes"] for record in records]
    assert working_sets == sorted(set(working_sets))
    assert working_sets[0] == pytest.approx(0.1 * 32 * 1024, rel=0.01)
    assert working_sets[-1] == pytest.approx(32 * 256 * 1024, rel=1e-6)
    for record in records:
        expected_bandwidth = (
            record["working_set_bytes"] * record["repetitions"] / record["time_seconds"] / 1e9
        )
        assert record["bandwidth_gbps"] == pytest.approx(expected_bandwidth)

def test_cli_roofline_writes_all_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(architecture_module, "detect_for_isa", lambda _isa, threads: _detected_architecture())
    monkeypatch.setattr(benchmark_interface, "compile_test_bench", _fake_compile)
    monkeypatch.setattr(benchmark_interface, "run_microbenchmarks", _fake_measure)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    table_output = io.StringIO()
    monkeypatch.setattr(
        output_utils, "_console_out", Console(file=table_output, theme=output_utils._theme, width=160, markup=False)
    )
    output_dir = tmp_path / "results"
    assert main(
        [
            "benchmark",
            "--isa",
            "x86_sse",
            "--test",
            "roofline",
            "--mem-target",
            "L1",
            "--mem-test-sizes",
            "16KiB",
            "--instruction",
            "add",
            "--ai-range",
            "1",
            "1",
            "--ai-points",
            "1",
            "--frequency",
            "2GHz",
            "--verbose",
            "2",
            "--name",
            "e2e-roofline",
            "--output-dir",
            str(output_dir),
            "--output-fmt",
            "jsonl",
            "csv",
            "table",
            "plot",
        ]
    ) == 0
    rendered_table = table_output.getvalue()
    assert "x86_sse" in rendered_table
    assert "L1" in rendered_table
    assert "MB/s" in rendered_table

    jsonl_path = output_dir / "e2e-roofline" / "benchmarks.jsonl"
    records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert records
    assert {record["type"] for record in records} == {"arithmetic", "memory"}
    csv_path = output_dir / "roofline" / "e2e-roofline_roofline.csv"
    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        csv_rows = list(csv.reader(csv_file))
    assert csv_rows[1][0] == "Date"
    assert any(row and row[0] for row in csv_rows[2:])
    plot_path = output_dir / "roofline" / "roofline.png"
    assert plot_path.is_file()
    with Image.open(plot_path) as image:
        assert image.width > 1 and image.height > 1
        colors = image.convert("RGB").getcolors(maxcolors=1_000_000)
        assert colors is not None
        assert len(colors) > 1
