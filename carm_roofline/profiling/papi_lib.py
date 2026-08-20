"""Access to the native libpapi via ctypes for event collectability checks.

The PAPI event catalog from ``papi_xml_event_info`` lists events the installed
library cannot always add (absent kernel PMUs, uncollectable presets).
:func:`collectable_events` tests each event against the real library so metric
resolution only sees events that can actually be counted.
"""

from __future__ import annotations

import ctypes
import re
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

from carm_roofline.output_utils import debug, warn


def _find_papi_library_path() -> Path | None:
    """Locate ``libpapi.so`` via multiple discovery strategies.

    Tries in order:
      1. ``ldconfig -p`` (dynamic linker cache, standard installs)
      2. ``pkg-config`` (respects ``PKG_CONFIG_PATH``, custom prefixes)

    Returns:
        Path to ``libpapi.so``, or *None* if not found by any strategy.
    """
    # Strategy 1: ldconfig -p
    try:
        result = subprocess.run(
            ["ldconfig", "-p"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "libpapi.so" in line:
                    # ldconfig -p output format: "	libpapi.so (libc6,x86-64) => /usr/lib/libpapi.so"
                    parts = line.split("=>")
                    if len(parts) == 2:
                        path = Path(parts[1].strip())
                        if path.is_file():
                            return path
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # Strategy 2: pkg-config -> list dir, pick first libpapi.so*
    try:
        result = subprocess.run(
            ["pkg-config", "--variable=libdir", "papi"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            libdir = result.stdout.strip()
            if libdir:
                candidates = sorted(Path(libdir).glob("libpapi.so*"))
                if candidates:
                    return candidates[0]
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    return None


def _papi_version_constant() -> int | None:
    """Return PAPI_VER_CURRENT for the installed library, or None.

    The library exports no version getter symbol; PAPI_VER_CURRENT is a
    compile-time constant ``(major << 24) | (minor << 16)`` (verified against
    papi.h: ``PAPI_VER_CURRENT = PAPI_VERSION & 0xffff0000``). The
    ``papi_version`` utility prints "PAPI Version: 7.2.0.0".
    """
    binary = shutil.which("papi_version")
    if binary is None:
        return None
    try:
        result = subprocess.run([binary], capture_output=True, text=True, timeout=5, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    match = re.search(r"PAPI Version:\s*(\d+)\.(\d+)", result.stdout)
    if match is None:
        return None
    major, minor = (int(g) for g in match.groups())
    return (major << 24) | (minor << 16)


def _load_papi_library() -> ctypes.CDLL | None:
    """Load libpapi via ctypes with call signatures; None when unavailable."""
    path = _find_papi_library_path()
    if path is None:
        return None
    try:
        library = ctypes.CDLL(str(path))
    except OSError:
        return None
    library.PAPI_event_name_to_code.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
    library.PAPI_event_name_to_code.restype = ctypes.c_int
    library.PAPI_create_eventset.argtypes = [ctypes.POINTER(ctypes.c_int)]
    library.PAPI_create_eventset.restype = ctypes.c_int
    library.PAPI_add_event.argtypes = [ctypes.c_int, ctypes.c_int]
    library.PAPI_add_event.restype = ctypes.c_int
    library.PAPI_cleanup_eventset.argtypes = [ctypes.c_int]
    library.PAPI_cleanup_eventset.restype = ctypes.c_int
    library.PAPI_destroy_eventset.argtypes = [ctypes.POINTER(ctypes.c_int)]
    library.PAPI_destroy_eventset.restype = ctypes.c_int
    return library


def collectable_events(events: Iterable[str], library: ctypes.CDLL) -> frozenset[str]:
    """Return the subset of *events* PAPI can add to a fresh event set.

    Each name is resolved with ``PAPI_event_name_to_code`` and added with
    ``PAPI_add_event`` to its own fresh event set; unresolvable or unaddable
    names are dropped. Every event set is cleaned up and destroyed. Returns
    *events* unchanged when PAPI cannot initialize in this process.
    """
    try:
        library.PAPI_library_init.argtypes = [ctypes.c_int]
        library.PAPI_library_init.restype = ctypes.c_int
        try:
            # Some builds export PAPI_version; use it when present.
            library.PAPI_version.restype = ctypes.c_int
            version = library.PAPI_version()
        except AttributeError:
            # PAPI 7.x makes PAPI_version an inline function (no symbol), so
            # derive PAPI_VER_CURRENT from the papi_version utility instead.
            version = _papi_version_constant()
        if version is None:
            debug("PAPI collectability filter skipped: cannot determine PAPI version")
            return frozenset(events)
        if library.PAPI_library_init(version) != version:
            debug("PAPI collectability filter skipped: PAPI_library_init failed")
            return frozenset(events)
    except (AttributeError, OSError) as exc:
        debug(f"PAPI collectability filter skipped: {exc}")
        return frozenset(events)
    try:
        result: set[str] = set()
        for name in events:
            code = ctypes.c_int()
            if library.PAPI_event_name_to_code(name.encode(), ctypes.pointer(code)) != 0:
                continue
            # PAPI_create_eventset requires the slot pre-initialized to
            # PAPI_NULL (-1); a zero slot fails with PAPI_EINVAL.
            eventset = ctypes.c_int(-1)
            if library.PAPI_create_eventset(ctypes.pointer(eventset)) != 0:
                continue
            if library.PAPI_add_event(eventset.value, code.value) == 0:
                result.add(name)
            library.PAPI_cleanup_eventset(eventset.value)
            library.PAPI_destroy_eventset(ctypes.pointer(eventset))
        return frozenset(result)
    except Exception as exc:  # filter failure must degrade, not break profiling
        warn(f"PAPI collectability filter failed; using the unfiltered catalog: {exc}")
        return frozenset(events)
