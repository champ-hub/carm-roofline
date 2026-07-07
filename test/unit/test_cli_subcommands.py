from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

import carm

pytestmark = pytest.mark.unit


def test_help_includes_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        carm.main(["-h"])

    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "benchmark" in out
    assert "gui" in out
    assert "profile" in out


def test_benchmark_mode_smoke_with_monkeypatch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def fake_execution_interface_init(self: carm.ExecutionInterface, args: argparse.Namespace) -> None:
        calls["exec_args"] = args
        self.sim_cmd = None
        self.compiler = "gcc"

    def fake_architecture_init(self: carm.Architecture, args: argparse.Namespace) -> None:
        calls["arch_args"] = args

    def fake_benchmarking_init(self: carm.Benchmarking, args: argparse.Namespace) -> None:
        calls["bench_args"] = args

    def fake_run_config_init(self: carm.RunConfig, args: argparse.Namespace) -> None:
        calls["run_args"] = args
        self.dry_run = True

    def fake_set_execution_interface(exec_iface: object) -> None:
        calls["exec_iface"] = exec_iface

    def fake_run_full_benchmark(context: object) -> dict[str, object]:
        calls["context"] = context
        return {}

    def fail_output(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("output_benchmark_results should not run in dry-run mode")

    monkeypatch.setattr(carm.ExecutionInterface, "__init__", fake_execution_interface_init)
    monkeypatch.setattr(carm.Architecture, "__init__", fake_architecture_init)
    monkeypatch.setattr(carm.Benchmarking, "__init__", fake_benchmarking_init)
    monkeypatch.setattr(carm.RunConfig, "__init__", fake_run_config_init)
    monkeypatch.setattr(carm, "set_execution_interface", fake_set_execution_interface)
    monkeypatch.setattr(carm, "run_full_benchmark", fake_run_full_benchmark)
    monkeypatch.setattr(carm, "output_benchmark_results", fail_output)
    monkeypatch.setattr(carm, "signature_from_architecture", lambda arch: None)
    monkeypatch.setattr(carm, "generate_run_name", lambda sig: "auto-test-name")

    exit_code = carm.main(["benchmark", "--dry-run", "--test-time", "1"])

    assert exit_code == 0
    assert isinstance(calls["exec_iface"], carm.ExecutionInterface)
    assert calls["bench_args"].test_time == 1.0


def test_no_command_profile_returns_non_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """Calling 'carm profile' without a command should return non-zero."""
    exit_code = carm.main(["profile"])

    assert exit_code != 0


def test_profile_empty_command_returns_non_zero() -> None:
    """Calling 'carm profile' with an empty command via -- should return non-zero."""
    exit_code = carm.main(["profile", "--"])

    assert exit_code != 0


def test_profile_help_shows_options(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        carm.main(["profile", "--help"])

    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "aggregation" in out
    assert "global" in out
    assert "rank" in out
    assert "thread" in out
    assert "region_merged" in out
    assert "region_per_thread" in out
    assert "results-dir" in out
    assert "command" in out


def test_benchmark_emit_config_creates_file(tmp_path: Path) -> None:
    output_path = tmp_path / "topology.toml"

    exit_code = carm.main(["benchmark", "--emit-config", str(output_path)])

    assert exit_code == 0
    assert output_path.exists()
    assert output_path.read_text()


def test_top_level_emit_config_fails_parse(tmp_path: Path) -> None:
    output_path = tmp_path / "topology.toml"

    with pytest.raises(SystemExit) as exc_info:
        carm.main(["--emit-config", str(output_path)])

    assert exc_info.value.code == 2


@pytest.mark.parametrize("mode", ["gui", "profile"])
def test_non_benchmark_emit_config_raises_parse_error(mode: str, tmp_path: Path) -> None:
    output_path = tmp_path / "topology.toml"

    with pytest.raises(SystemExit) as exc_info:
        carm.main([mode, "--emit-config", str(output_path)])

    assert exc_info.value.code == 2
