from __future__ import annotations

import ctypes
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from carm_roofline.core import Bytes, Frequency
from carm_roofline.exec_interface import ExecutionInterface
from carm_roofline.test_bench import builder
from carm_roofline.test_bench.builder import MicrobenchmarkFunctionSpec, run_microbenchmarks

pytestmark = [pytest.mark.integration, pytest.mark.native]


@pytest.mark.skipif(shutil.which("gcc") is None, reason="requires gcc to compile the native harness")
@pytest.mark.parametrize("gap_bytes", [16 * 1024, 24 * 1024])
def test_native_harness_allocates_thread_stride_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gap_bytes: int
) -> None:
    """Run the real C harness and verify its thread pointers and allocation size."""
    harness_dir = Path(__file__).parents[1] / "carm_roofline" / "test_bench"
    header = tmp_path / "microbenchmarks.h"
    header.write_text(
        '#include "test_bench.h"\n'
        "static void *wrapper_stride_gap(void *arg) {\n"
        "    thread_wrapper_data_t *tdata = (thread_wrapper_data_t *)arg;\n"
        "    for (int i = 0; i < 8; i++) barrier_sync();\n"
        "    tdata->elapsed_ns[0] = 1;\n"
        "    return NULL;\n"
        "}\n"
        "static const benchmark_metadata_t metadata_stride_gap = {\n"
        '    .name = "stride_gap", .frequency_ghz = 1.0f, .thread_affinity = NULL,\n'
        "    .num_threads = 2, .read_array_size_bytes = 4096, .write_array_size_bytes = 4096\n"
        "};\n"
        "#define MICROBENCHMARK_LIST X(wrapper_stride_gap, &metadata_stride_gap)\n"
    )
    binary = tmp_path / "harness"
    subprocess.run(
        [
            "gcc",
            "-DVERBOSITY=4",
            "-DNUM_RUNS=1",
            "-pthread",
            "-lm",
            f"-I{tmp_path}",
            f"-I{harness_dir}",
            str(harness_dir / "test_bench.c"),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    debug_output: list[str] = []
    monkeypatch.setattr(builder, "debug", debug_output.append)
    context = SimpleNamespace(
        benchmarking=SimpleNamespace(
            test_time=1,
            interleaved=True,
            thread_stride_gap=Bytes(gap_bytes),
        ),
        exec_interface=ExecutionInterface(SimpleNamespace(sim_cmd=None, compiler=None)),
    )
    spec = MicrobenchmarkFunctionSpec(
        function_name="stride_gap",
        body="",
        read_array_size=Bytes(4096),
        write_array_size=Bytes(4096),
        frequency=Frequency(1e9),
        thread_affinity=[0, 1],
    )

    run_microbenchmarks(context, binary, [spec])
    diagnostics = "\n".join(debug_output)
    pointer_rows = re.findall(
        r"thread (\d+) read_ptr=(0x[0-9a-fA-F]+) per_thread_stride=(\d+) read_size=(\d+)", diagnostics
    )
    assert len(pointer_rows) == 2
    rows = {int(tid): (int(address, 16), int(stride), int(read_size)) for tid, address, stride, read_size in pointer_rows}
    expected_stride = 2 * 4096 + gap_bytes
    assert rows[0][1:] == (expected_stride, 4096)
    assert rows[1][1:] == (expected_stride, 4096)
    assert rows[1][0] - rows[0][0] == expected_stride

    write_rows = re.findall(r"thread (\d+) write_ptr=(0x[0-9a-fA-F]+).*write_size=(\d+)", diagnostics)
    assert len(write_rows) == 2
    for tid, address, write_size in write_rows:
        assert int(address, 16) - rows[int(tid)][0] == 4096
        assert int(write_size) == 4096

    total_match = re.search(r"Combined buffer: (\d+) bytes", diagnostics)
    assert total_match is not None
    assert int(total_match.group(1)) == 2 * expected_stride

    unknown_arg = subprocess.run(
        [str(binary), "--interleaved", "--unrecognized-option"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert unknown_arg.returncode == 0
    default_rows = re.findall(
        r"thread (\d+) read_ptr=(0x[0-9a-fA-F]+) per_thread_stride=(\d+) read_size=(\d+)",
        unknown_arg.stderr,
    )
    assert len(default_rows) == 2
    assert {int(row[2]) for row in default_rows} == {2 * 4096 + 16 * 1024}

    size_max = (1 << (ctypes.sizeof(ctypes.c_size_t) * 8)) - 1
    context.benchmarking.thread_stride_gap = Bytes(size_max)
    with pytest.raises(RuntimeError, match="Per-thread allocation stride overflow"):
        run_microbenchmarks(context, binary, [spec])
    context.benchmarking.thread_stride_gap = Bytes(size_max // 2 + 1 - 2 * 4096)
    with pytest.raises(RuntimeError, match="Total buffer size overflow"):
        run_microbenchmarks(context, binary, [spec])


@pytest.mark.skipif(shutil.which("gcc") is None, reason="requires gcc to compile the native harness")
def test_arithmetic_only_harness_ignores_gap_without_null_pointer_arithmetic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty working set allocates no bytes even with a maximal gap."""
    harness_dir = Path(__file__).parents[1] / "carm_roofline" / "test_bench"
    header = tmp_path / "microbenchmarks.h"
    header.write_text(
        '#include "test_bench.h"\n'
        "static void *wrapper_arithmetic_only(void *arg) {\n"
        "    thread_wrapper_data_t *tdata = (thread_wrapper_data_t *)arg;\n"
        "    tdata->elapsed_ns[0] = 1;\n"
        "    return NULL;\n"
        "}\n"
        "static const benchmark_metadata_t metadata_arithmetic_only = {\n"
        '    .name = "arithmetic_only", .frequency_ghz = 1.0f, .thread_affinity = NULL,\n'
        "    .num_threads = 2, .read_array_size_bytes = 0, .write_array_size_bytes = 0\n"
        "};\n"
        "#define MICROBENCHMARK_LIST X(wrapper_arithmetic_only, &metadata_arithmetic_only)\n"
    )
    binary = tmp_path / "arithmetic-only-harness"
    subprocess.run(
        [
            "gcc",
            "-DVERBOSITY=4",
            "-DNUM_RUNS=1",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=undefined",
            "-pthread",
            "-lm",
            f"-I{tmp_path}",
            f"-I{harness_dir}",
            str(harness_dir / "test_bench.c"),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    debug_output: list[str] = []
    monkeypatch.setattr(builder, "debug", debug_output.append)
    context = SimpleNamespace(
        benchmarking=SimpleNamespace(
            test_time=1,
            interleaved=False,
            thread_stride_gap=Bytes((1 << (ctypes.sizeof(ctypes.c_size_t) * 8)) - 1),
        ),
        exec_interface=ExecutionInterface(SimpleNamespace(sim_cmd=None, compiler=None)),
    )
    spec = MicrobenchmarkFunctionSpec(
        function_name="arithmetic_only",
        body="",
        read_array_size=Bytes(0),
        write_array_size=Bytes(0),
        frequency=Frequency(1e9),
        thread_affinity=[0, 1],
    )

    output = run_microbenchmarks(context, binary, [spec])
    diagnostics = "\n".join(debug_output)
    assert "arithmetic_only," in output
    assert "Combined buffer:" not in diagnostics
    assert "thread 0 read_ptr=" not in diagnostics
    assert "runtime error:" not in diagnostics
