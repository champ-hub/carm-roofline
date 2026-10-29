from __future__ import annotations

import ctypes
import errno
import os
import sys


def _sysctl_bytes(name: str) -> bytes:
    """Read a sysctlbyname value as its native byte representation."""
    libc = ctypes.CDLL(None, use_errno=True)
    sysctlbyname = libc.sysctlbyname
    sysctlbyname.argtypes = [
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    sysctlbyname.restype = ctypes.c_int

    length = ctypes.c_size_t()
    encoded_name = name.encode("ascii")
    if sysctlbyname(encoded_name, None, ctypes.byref(length), None, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), name)
    if length.value == 0:
        return b""

    buffer = ctypes.create_string_buffer(length.value)
    if sysctlbyname(encoded_name, buffer, ctypes.byref(length), None, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), name)
    return buffer.raw[: length.value]


def sysctl_int(name: str) -> int:
    """Read an integer-valued sysctlbyname key (32- or 64-bit)."""
    value = _sysctl_bytes(name)
    if len(value) not in (4, 8):
        raise ValueError(f"sysctl {name!r} returned {len(value)} bytes, expected 4 or 8")
    return int.from_bytes(value, byteorder=sys.byteorder, signed=False)


def sysctl_string(name: str) -> str:
    """Read a NUL-terminated string-valued sysctlbyname key."""
    return _sysctl_bytes(name).split(b"\0", 1)[0].decode("utf-8", errors="replace")


def sysctl_int_or_none(name: str) -> int | None:
    """Read an optional sysctl key, returning None only when the key is absent."""
    try:
        return sysctl_int(name)
    except OSError as error:
        if error.errno in (errno.ENOENT, errno.EINVAL, errno.ENOTSUP):
            return None
        raise
