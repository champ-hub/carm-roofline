from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

import carm_roofline.carm as carm
import carm_roofline.paraver.shim as shim

pytestmark = pytest.mark.unit


def _translate(argv: Sequence[str]) -> list[str]:
    return shim.translate_args(shim._build_parser().parse_args(argv))


def test_translate_paraver_example() -> None:
    """The exact command Paraver launches must map to the carm gui argv."""
    translated = _translate(
        [
            "--color_csv",
            "--mask_csv",
            "--csv",
            "New_window_#1_lulesh2.0_64p_carm_DP.chop2.csv",
            "lulesh2.0_64p_carm_DP.chop2.prv",
        ]
    )
    assert translated == [
        "gui",
        "--paraver-trace",
        "lulesh2.0_64p_carm_DP.chop2.prv",
        "--paraver-window-csv",
        "New_window_#1_lulesh2.0_64p_carm_DP.chop2.csv",
        "--paraver-use-semantic-window",
    ]


def test_translate_minimal() -> None:
    """Without --mask_csv, no semantic-window flag is added."""
    assert _translate(["--csv", "m.csv", "t.prv"]) == [
        "gui",
        "--paraver-trace",
        "t.prv",
        "--paraver-window-csv",
        "m.csv",
    ]


@pytest.mark.parametrize("debug_flag", ["-d", "--debug"])
def test_translate_debug_appends_verbose(debug_flag: str) -> None:
    """Both -d and --debug map to a trailing -v (carm gui's debug verbosity)."""
    translated = _translate(["--csv", "m.csv", "t.prv", debug_flag])
    assert translated[-1] == "-v"
    assert "-v" not in translated[:-1]


def test_translate_unsupported_flags_warn_and_drop(monkeypatch: pytest.MonkeyPatch) -> None:
    """-ac and --min_dur are dropped from the argv and surfaced as warnings."""
    warnings: list[str] = []
    monkeypatch.setattr(shim, "warn", warnings.append)
    translated = _translate(["--csv", "m.csv", "t.prv", "-ac", "--min_dur", "5"])
    assert "-ac" not in translated
    assert "--min_dur" not in translated
    assert warnings == [
        "--ac (accumulate mode) is not supported by CARM; ignoring",
        "--min_dur 5.0 (duration filter) is not supported by CARM; ignoring",
    ]


def test_main_delegates_to_carm(monkeypatch: pytest.MonkeyPatch) -> None:
    """shim.main must hand the translated argv to carm.main and return its code."""
    assert shim.carm_main is carm.main  # delegation target is the real entry point
    calls: dict[str, Any] = {}

    def fake_main(argv: Sequence[str]) -> int:
        calls["argv"] = argv
        return 7

    monkeypatch.setattr(shim, "carm_main", fake_main)
    assert shim.main(["--csv", "m.csv", "t.prv"]) == 7
    assert list(calls["argv"]) == ["gui", "--paraver-trace", "t.prv", "--paraver-window-csv", "m.csv"]


def test_main_missing_csv_exits_2() -> None:
    with pytest.raises(SystemExit) as exc_info:
        shim.main([])
    assert exc_info.value.code == 2


def test_main_version_banner(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        shim.main(["-v"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith("Paraver_CARM version ")
