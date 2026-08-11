"""Unit tests for GUI launch guards (parent-death signal and port reservation)."""

from __future__ import annotations

import os
import signal
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from carm_roofline.core.error import UserError
from carm_roofline.gui import launch, run_app

pytestmark = pytest.mark.unit


def _free_port() -> int:
    """Return a currently-free TCP port on 127.0.0.1."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# -- reserve_free_port --------------------------------------------------------


def test_reserve_free_port_returns_base_when_free() -> None:
    """A free base port is returned as-is, with the socket held open."""
    base = _free_port()
    port, reserved = launch.reserve_free_port("127.0.0.1", base)
    try:
        assert port == base
        assert reserved.fileno() != -1  # still open
    finally:
        reserved.close()


def test_reserve_free_port_bumps_past_busy_port() -> None:
    """A busy base port causes a bump to the next free port."""
    base = _free_port()
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("127.0.0.1", base))
    blocker.listen(1)
    try:
        port, reserved = launch.reserve_free_port("127.0.0.1", base)
        try:
            assert port == base + 1
            assert reserved.fileno() != -1  # still open
        finally:
            reserved.close()
    finally:
        blocker.close()


def test_reserve_free_port_holds_reservation_until_closed() -> None:
    """The returned socket holds the port until explicitly closed."""
    base = _free_port()
    port, reserved = launch.reserve_free_port("127.0.0.1", base)
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError):
            probe.bind(("127.0.0.1", port))
    finally:
        probe.close()
        reserved.close()
    # Released: a fresh bind on the same port now succeeds.
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", port))
    finally:
        probe.close()


def test_reserve_free_port_exhaustion_raises_user_error() -> None:
    """Running out of candidate ports raises UserError naming the scan range."""
    base = _free_port()
    blockers: list[socket.socket] = []
    try:
        for offset in range(launch.MAX_PORT_ATTEMPTS):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", base + offset))
            sock.listen(1)
            blockers.append(sock)
        with pytest.raises(UserError, match="free port"):
            launch.reserve_free_port("127.0.0.1", base)
    finally:
        for sock in blockers:
            sock.close()


def test_reserve_free_port_rejects_non_positive_base() -> None:
    """Base ports 0 and negative are rejected up front as UserError."""
    for bad in (0, -1):
        with pytest.raises(UserError, match="Invalid GUI port"):
            launch.reserve_free_port("127.0.0.1", bad)


# -- set_parent_death_signal --------------------------------------------------


def test_set_parent_death_signal_calls_prctl(monkeypatch: pytest.MonkeyPatch) -> None:
    """set_parent_death_signal calls prctl(PR_SET_PDEATHSIG, SIGTERM) on Linux."""
    calls: list[tuple[int, int]] = []

    class _FakeLibc:
        def prctl(self, op: int, arg: int) -> int:
            calls.append((op, arg))
            return 0

    fake = _FakeLibc()
    monkeypatch.setattr(launch.ctypes, "CDLL", lambda name: fake)

    launch.set_parent_death_signal()

    assert calls == [(1, signal.SIGTERM)]


@pytest.mark.parametrize("fail_mode", ["cdll_raises", "prctl_nonzero"])
def test_set_parent_death_signal_warns_and_continues_on_failure(
    monkeypatch: pytest.MonkeyPatch, fail_mode: str
) -> None:
    """A failed prctl setup warns once and never raises, so the GUI still launches."""
    warnings: list[str] = []

    def record_warning(msg: str) -> None:
        warnings.append(msg)

    monkeypatch.setattr(launch, "warn", record_warning)

    if fail_mode == "cdll_raises":

        def raising_cdll(name: str) -> None:
            raise OSError("libc unavailable")

        monkeypatch.setattr(launch.ctypes, "CDLL", raising_cdll)
    else:

        class _FakeLibc:
            def prctl(self, op: int, arg: int) -> int:
                return -1

        monkeypatch.setattr(launch.ctypes, "CDLL", lambda name: _FakeLibc())

    launch.set_parent_death_signal()  # must not raise

    assert len(warnings) == 1
    assert "parent-death watchdog" in warnings[0]


def test_parent_death_signal_kills_guarded_process(tmp_path: Path) -> None:
    """A process armed with PR_SET_PDEATHSIG dies when its parent exits (real kernel check)."""
    if not sys.platform.startswith("linux"):
        pytest.skip("PR_SET_PDEATHSIG is Linux-only")

    armed = tmp_path / "armed"
    received = tmp_path / "received"

    middle = os.fork()
    if middle == 0:  # middle process
        try:
            child = os.fork()
            if child == 0:  # grandchild: the guarded sleeper
                def _on_sigterm(signum: int, frame: object) -> None:
                    received.write_text("sigterm")
                    os._exit(0)

                signal.signal(signal.SIGTERM, _on_sigterm)
                launch.set_parent_death_signal()
                armed.write_text(str(os.getpid()))
                while True:
                    time.sleep(60)
            # middle: wait until the grandchild is fully armed, then die.
            deadline = time.monotonic() + 10
            while not armed.exists():
                if time.monotonic() > deadline:
                    os._exit(2)
                time.sleep(0.05)
            os._exit(0)
        finally:
            os._exit(3)

    _, status = os.waitpid(middle, 0)
    assert os.waitstatus_to_exitcode(status) == 0

    # The grandchild is not our child; wait for its SIGTERM handler, then clean up.
    deadline = time.monotonic() + 5
    while not received.exists():
        if time.monotonic() > deadline:
            os.kill(int(armed.read_text()), signal.SIGKILL)
            break
        time.sleep(0.05)
    assert received.exists()


# -- run_app ordering and cleanup ---------------------------------------------


class _FakeApp:
    """Minimal stand-in for the Dash app: records run() kwargs."""

    def __init__(self) -> None:
        self.runs: list[dict[str, object]] = []

    def run(self, **kwargs: object) -> None:
        self.runs.append(kwargs)


def test_run_app_checks_before_create_app_and_uses_selected_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guards run before create_app and app.run receives the selected port."""
    events: list[tuple[object, ...]] = []
    fake_app = _FakeApp()
    reserved_sock: socket.socket

    def fake_create_app(config: object) -> _FakeApp:
        events.append(("create",))
        return fake_app

    def fake_set_parent_death_signal() -> None:
        events.append(("death",))

    def fake_reserve_free_port(host: str, base_port: int) -> tuple[int, socket.socket]:
        events.append(("reserve", host, base_port))
        nonlocal reserved_sock
        reserved_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        reserved_sock.bind(("127.0.0.1", 0))
        return 8051, reserved_sock

    monkeypatch.setattr("carm_roofline.gui.factory.create_app", fake_create_app)
    monkeypatch.setattr("carm_roofline.gui.launch.set_parent_death_signal", fake_set_parent_death_signal)
    monkeypatch.setattr("carm_roofline.gui.launch.reserve_free_port", fake_reserve_free_port)

    config = SimpleNamespace(gui_host="0.0.0.0", gui_port=8050, gui_debug=False)
    run_app(config)

    assert events == [("death",), ("reserve", "0.0.0.0", 8050), ("create",)]
    assert reserved_sock.fileno() == -1  # reservation released before app.run
    assert fake_app.runs == [{"host": "0.0.0.0", "port": 8051, "debug": False}]


def test_run_app_closes_reservation_when_create_app_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A create_app failure still releases the reserved port (finally path)."""
    events: list[tuple[object, ...]] = []
    reserved_sock: socket.socket

    def fake_create_app(config: object) -> _FakeApp:
        raise RuntimeError("boom")

    monkeypatch.setattr("carm_roofline.gui.factory.create_app", fake_create_app)
    monkeypatch.setattr("carm_roofline.gui.launch.set_parent_death_signal", lambda: None)

    def fake_reserve_free_port(host: str, base_port: int) -> tuple[int, socket.socket]:
        events.append(("reserve", host, base_port))
        nonlocal reserved_sock
        reserved_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        reserved_sock.bind(("127.0.0.1", 0))
        return 8051, reserved_sock

    monkeypatch.setattr("carm_roofline.gui.launch.reserve_free_port", fake_reserve_free_port)

    config = SimpleNamespace(gui_host="127.0.0.1", gui_port=8050, gui_debug=False)
    with pytest.raises(RuntimeError):
        run_app(config)

    assert events == [("reserve", "127.0.0.1", 8050)]
    assert reserved_sock.fileno() == -1  # finally path released the reservation
