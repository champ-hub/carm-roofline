from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import dash
import pandas as pd
import pytest

from carm_roofline.gui.callbacks import paraver
from carm_roofline.gui.export import ExportFile
from carm_roofline.paraver import protocol


def export_callback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol_mode: bool) -> Callable[..., str]:
    monkeypatch.setattr(
        paraver,
        "export_performance",
        lambda *_: [
            ExportFile("a.csv", "a"),
            ExportFile("a.legend.csv", "legend"),
            ExportFile("b.csv", "b"),
            ExportFile("c.csv", "c"),
        ],
    )
    state = SimpleNamespace(
        config=SimpleNamespace(
            paraver_trace=tmp_path / "trace.prv", paraver_window_csv=None, paraver_protocol=protocol_mode
        ),
        records=[],
        paraver_data=SimpleNamespace(trace=pd.DataFrame({"value": [1]})),
        mode=SimpleNamespace(show_time_slider=False, has_export_tab=True),
    )
    app = dash.Dash(__name__)
    paraver.register_paraver_callbacks(app, state)
    return app.callback_map["export-status-performance.children"]["callback"].__wrapped__


@pytest.mark.parametrize("failure", [False, True])
def test_export_waits_for_host_and_stops_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], failure: bool
) -> None:
    commands: list[str] = []
    responses = ["PARAVER_PROTOCOL succeed\n", "PARAVER_PROTOCOL error\n" if failure else "PARAVER_PROTOCOL succeed\n"]
    if not failure:
        responses.append("PARAVER_PROTOCOL succeed\n")
    total = len(responses)

    class Host(io.StringIO):
        def readline(self, size: int = -1) -> str:
            assert len(commands) == total - len(responses) + 1
            return responses.pop(0)

    def emit(command: str, *, flush: bool) -> None:
        assert flush
        commands.append(command)

    monkeypatch.setattr(protocol, "print", emit, raising=False)
    monkeypatch.setattr(protocol.sys, "stdin", Host())
    callback = export_callback(tmp_path, monkeypatch, True)
    status = callback(1, {})
    names = ("a.csv", "b.csv") if failure else ("a.csv", "b.csv", "c.csv")
    assert commands == [f"PARAVER_PROTOCOL open_csv {tmp_path / name}" for name in names]
    assert capsys.readouterr().out.splitlines() == [str(tmp_path / name) for name in names]
    if failure:
        assert "Failed to open export in Paraver" in status
        assert "open_csv" in status
    else:
        assert status == ""
    assert not responses


@pytest.mark.parametrize("response", ["", "bad response\n", "PARAVER_PROTOCOL succeed"])
def test_export_reports_invalid_host_responses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], response: str
) -> None:
    monkeypatch.setattr(protocol.sys, "stdin", io.StringIO(response))
    callback = export_callback(tmp_path, monkeypatch, True)
    assert "Failed to open export in Paraver" in callback(1, {})
    assert capsys.readouterr().out.splitlines() == [
        f"PARAVER_PROTOCOL open_csv {tmp_path / 'a.csv'}", str(tmp_path / "a.csv")
    ]


def test_export_reports_host_read_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class BrokenHost(io.StringIO):
        def readline(self, size: int = -1) -> str:
            raise OSError("broken pipe")

    monkeypatch.setattr(protocol.sys, "stdin", BrokenHost())
    callback = export_callback(tmp_path, monkeypatch, True)
    assert "broken pipe" in callback(1, {})
    assert capsys.readouterr().out.splitlines() == [
        f"PARAVER_PROTOCOL open_csv {tmp_path / 'a.csv'}", str(tmp_path / "a.csv")
    ]


def test_standalone_export_never_reads_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class NoHost(io.StringIO):
        def readline(self, size: int = -1) -> str:
            raise AssertionError("standalone export must not read stdin")

    monkeypatch.setattr(protocol.sys, "stdin", NoHost())
    callback = export_callback(tmp_path, monkeypatch, False)
    assert callback(1, {}) == ""
    assert capsys.readouterr().out.splitlines() == [
        line
        for name in ("a.csv", "b.csv", "c.csv")
        for line in (f"PARAVER_PROTOCOL open_csv {tmp_path / name}", str(tmp_path / name))
    ]
