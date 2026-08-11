from __future__ import annotations

import ctypes
import errno
import signal
import socket

from carm_roofline.core.error import UserError
from carm_roofline.output_utils import warn

MAX_PORT_ATTEMPTS = 5

PR_SET_PDEATHSIG = 1  # <linux/prctl.h>


def set_parent_death_signal() -> None:
    """Arrange for SIGTERM when the parent process dies (Linux PR_SET_PDEATHSIG).

    Best-effort: on non-Linux or when the syscall is unavailable, warn and
    continue so the GUI still launches.
    """
    try:
        libc = ctypes.CDLL("libc.so.6")
        if libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM) != 0:
            raise OSError("prctl(PR_SET_PDEATHSIG) failed")
    except (OSError, AttributeError) as exc:
        warn(
            f"Could not arm the parent-death watchdog ({exc}); the GUI will not exit "
            "automatically when its launcher dies"
        )


def reserve_free_port(host: str, base_port: int, max_attempts: int = MAX_PORT_ATTEMPTS) -> tuple[int, socket.socket]:
    """Reserve the first free TCP port at or after ``base_port``.

    Returns ``(port, reserved_socket)``. The caller MUST close ``reserved_socket``
    immediately before handing the port to the server, and on every error path —
    while it stays open it holds the port so concurrent instances cannot steal it
    during the slow app setup. Raises ``UserError`` when no port is free.
    """
    if base_port <= 0:
        raise UserError(f"Invalid GUI port {base_port}; use a port between 1 and 65535")
    port = base_port
    for _ in range(max_attempts):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
        except OSError as exc:
            sock.close()
            if exc.errno == errno.EADDRINUSE:
                warn(f"Port {port} is in use; trying port {port + 1}")
                port += 1
                continue
            raise UserError(f"Cannot bind the GUI to {host}:{port}: {exc}") from exc
        return port, sock
    raise UserError(
        f"Could not find a free port between {base_port} and {base_port + max_attempts - 1}; "
        "another instance may already be running. Close it and retry."
    )
