"""Unit tests for streaming benchmark progress reporting."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from rich.console import Console

from carm_roofline.benchmark.interface import _run_microbenchmarks_with_progress
from carm_roofline.core import Bytes, Frequency, Seconds
from carm_roofline.exec_interface import ExecutionInterface
from carm_roofline.output_utils import Verbosity
from carm_roofline.test_bench import builder
from carm_roofline.test_bench.builder import MicrobenchmarkFunctionSpec, run_microbenchmarks


def make_context() -> SimpleNamespace:
    """Create the minimum context required by the benchmark runner."""
    return SimpleNamespace(
        benchmarking=SimpleNamespace(test_time=1, interleaved=False),
        exec_interface=ExecutionInterface(argparse.Namespace(sim_cmd=None, compiler=None)),
    )


def make_spec() -> MicrobenchmarkFunctionSpec:
    """Create a minimal arithmetic benchmark specification."""
    return MicrobenchmarkFunctionSpec(
        function_name="benchmark",
        body="",
        read_array_size=Bytes(0),
        write_array_size=Bytes(0),
        frequency=Frequency(1e9),
        thread_affinity=[0],
    )


def write_executable(path: Path, source: str) -> Path:
    """Write a Python executable used as a benchmark process."""
    path.write_text(f"#!{sys.executable}\n{source}")
    path.chmod(0o755)
    return path


@pytest.mark.unit
def test_streaming_runner_reports_completed_lines_and_debug_stderr(tmp_path, monkeypatch):
    """The streaming path reports flushed result lines and preserves stderr."""
    binary = write_executable(
        tmp_path / "benchmark.py",
        "import sys\n"
        "sys.stdout.write('one, 1.0, 1\\n'); sys.stdout.flush()\n"
        "sys.stderr.write('diagnostic\\n'); sys.stderr.flush()\n"
        "sys.stdout.write('two, 1.0, 1\\n'); sys.stdout.flush()\n",
    )
    completed: list[int] = []
    debug_messages: list[str] = []
    monkeypatch.setattr(builder, "debug", debug_messages.append)

    output = run_microbenchmarks(make_context(), binary, [make_spec(), make_spec()], completed.append)

    assert output == "one, 1.0, 1\ntwo, 1.0, 1\n"
    assert sum(completed) == 2
    assert any("diagnostic" in message for message in debug_messages)


@pytest.mark.unit
def test_streaming_runner_timeout_reports_partial_output_and_kills_child(tmp_path, monkeypatch):
    """The streaming timeout keeps received output and terminates the child."""
    pid_file = tmp_path / "child.pid"
    binary = write_executable(
        tmp_path / "benchmark.py",
        "import os, pathlib, sys, time\n"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
        "sys.stdout.write('one, 1.0, 1\\n'); sys.stdout.flush()\n"
        "time.sleep(10)\n",
    )
    monkeypatch.setattr(builder, "TIMEOUT_MIN", Seconds(0.05))
    monkeypatch.setattr(builder, "TIMEOUT_SAFETY_FACTOR", 0.1)
    monkeypatch.setattr(builder, "ARITHMETIC_FALLBACK_PER_RUN", Seconds(0.001))
    monkeypatch.setattr(builder, "CALIBRATION_START_REPS", 0)
    completed: list[int] = []

    with pytest.raises(subprocess.TimeoutExpired) as caught:
        run_microbenchmarks(make_context(), binary, [make_spec()], completed.append)

    assert sum(completed) == 1
    assert caught.value.output == b"one, 1.0, 1\n"
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


@pytest.mark.unit
def test_streaming_runner_reaps_child_when_progress_callback_fails(tmp_path):
    """A failing progress callback terminates and reaps the benchmark process."""
    pid_file = tmp_path / "child.pid"
    binary = write_executable(
        tmp_path / "benchmark.py",
        "import os, pathlib, sys, time\n"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
        "sys.stdout.write('one, 1.0, 1\\n'); sys.stdout.flush()\n"
        "time.sleep(10)\n",
    )

    def fail_progress(_count):
        raise RuntimeError("progress failed")

    with pytest.raises(RuntimeError, match="progress failed"):
        run_microbenchmarks(make_context(), binary, [make_spec()], fail_progress)

    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


@pytest.mark.unit
@pytest.mark.parametrize("verbosity", [0, 1, 2])
def test_progress_runner_skips_progress_below_detail(verbosity, monkeypatch):
    """Quiet through result verbosity uses the blocking runner without a bar."""
    context = SimpleNamespace(run_config=SimpleNamespace(verbose=verbosity))
    run = Mock(return_value="results")
    console = StringIO()
    monkeypatch.setattr("carm_roofline.benchmark.interface.run_microbenchmarks", run)
    monkeypatch.setattr("carm_roofline.benchmark.interface.get_console", lambda: Console(file=console))

    assert _run_microbenchmarks_with_progress(context, Path("benchmark"), [Mock(), Mock()]) == "results"
    assert "on_benchmarks_complete" not in run.call_args.kwargs
    assert console.getvalue() == ""


@pytest.mark.unit
@pytest.mark.parametrize("verbosity", [Verbosity.CONFIG, Verbosity.DEBUG])
def test_progress_runner_shows_canonical_total(verbosity, monkeypatch):
    """Detail and debug verbosity show completed canonical measurements."""
    context = SimpleNamespace(run_config=SimpleNamespace(verbose=verbosity))
    console = StringIO()

    def run(_context, _binary_path, _specs, on_benchmarks_complete=None):
        assert on_benchmarks_complete is not None
        on_benchmarks_complete(2)
        return "results"

    monkeypatch.setattr("carm_roofline.benchmark.interface.run_microbenchmarks", run)
    monkeypatch.setattr(
        "carm_roofline.benchmark.interface.get_console",
        lambda: Console(file=console, force_terminal=False, width=80),
    )

    canonical_benchmarks = [Mock(), Mock()]
    assert _run_microbenchmarks_with_progress(context, Path("benchmark"), canonical_benchmarks) == "results"
    output = console.getvalue()
    assert "Running benchmarks" in output
    assert "2/2" in output
