"""Tests for concurrent Paraver protocol consumers."""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from carm_roofline.paraver import protocol

pytestmark = pytest.mark.unit


def test_concurrent_requests_wait_for_matching_response(monkeypatch: pytest.MonkeyPatch) -> None:
    first_reading = Event()
    release_first = Event()
    second_emitted = Event()
    second_started = Event()
    commands: list[str] = []

    class HostResponses(io.StringIO):
        def readline(self, size: int = -1) -> str:
            if not first_reading.is_set():
                first_reading.set()
                assert release_first.wait(5)
                assert len(commands) == 1
            return "PARAVER_PROTOCOL succeed\n"

    def emit(command: str, *, flush: bool) -> None:
        assert flush
        if len(commands) == 1:
            second_emitted.set()
        commands.append(command)

    def second_request() -> None:
        second_started.set()
        protocol.request(protocol.ParaverCommand.OPEN_CSV, Path("second.csv"))

    monkeypatch.setattr(protocol, "print", emit, raising=False)
    monkeypatch.setattr(protocol.sys, "stdin", HostResponses())
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(protocol.request, protocol.ParaverCommand.GENERATE_CSV, Path("first.cfg"), Path("first.csv"))
        try:
            assert first_reading.wait(5)
            second = executor.submit(second_request)
            assert second_started.wait(5)
            assert not second_emitted.wait(0.2)
            assert not second.done()
        finally:
            release_first.set()
        first.result(timeout=5)
        second.result(timeout=5)

    assert commands == [
        "PARAVER_PROTOCOL generate_csv first.cfg first.csv",
        "PARAVER_PROTOCOL open_csv second.csv",
    ]
