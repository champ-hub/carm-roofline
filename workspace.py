"""Temporary workspace management utilities.

Provides a context manager that creates a temporary directory, optionally
keeping it after the context exits for debugging or inspection.

Usage:

    with workspace_context(keep=False, prefix="carm-benchmark-") as tmp:
        workspace = Path(tmp)
        # ... write files to workspace ...
        # Directory is auto-cleaned on exit unless keep=True
"""

from __future__ import annotations

import tempfile
from contextlib import AbstractContextManager, nullcontext


def workspace_context(
    keep: bool = False,
    prefix: str = "carm-",
) -> AbstractContextManager[str]:
    """Create a temporary workspace directory.

    Args:
        keep: If True, the directory is NOT cleaned up on exit (useful for
              dry runs or debugging).
        prefix: Prefix for the temporary directory name.

    Returns:
        A context manager that yields the path to the temporary directory as
        a `str`.  When *keep* is `False` the directory and all its
        contents are removed on context exit.
    """
    if keep:
        path = tempfile.mkdtemp(prefix=prefix)
        return nullcontext(path)
    return tempfile.TemporaryDirectory(prefix=prefix)


__all__ = [
    "workspace_context",
]
