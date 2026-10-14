from __future__ import annotations

from pathlib import Path

import pytest

import carm_roofline.carm as carm

pytestmark = pytest.mark.integration


def test_invalid_mixed_ratio_exits_parser() -> None:
    """The parser rejects invalid arithmetic-memory ratios."""
    with pytest.raises(SystemExit) as exc_info:
        carm._create_parser().parse_args(["benchmark", "--arith-mem-ratio", "0:2"])
    assert exc_info.value.code == 2



@pytest.mark.parametrize(
    ("arguments", "expected_bytes"),
    [
        ([], 16 * 1024),
        (["--thread-stride-gap", "32KiB"], 32 * 1024),
        (["--thread-stride-gap", "0"], 0),
    ],
)
def test_thread_stride_gap_argument(arguments: list[str], expected_bytes: int) -> None:
    """The benchmark CLI accepts binary size units and defaults to a 16 KiB gap."""
    args = carm._create_parser().parse_args(["benchmark", *arguments])
    assert int(args.thread_stride_gap) == expected_bytes


def test_thread_stride_gap_rejects_negative_value() -> None:
    """The benchmark CLI rejects negative allocation gaps."""
    with pytest.raises(SystemExit) as exc_info:
        carm._create_parser().parse_args(["benchmark", "--thread-stride-gap", "-1"])
    assert exc_info.value.code == 2

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
