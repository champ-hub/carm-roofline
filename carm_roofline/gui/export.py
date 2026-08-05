"""Export of the displayed roofline data back to paraver.

The paraver export format is an external dependency that has not been defined yet;
`serialize_paraver_export` is the single plug point where the writer will land.
"""

from __future__ import annotations

import pandas as pd

EXPORT_FILENAME = "paraver-export.csv"  # update when the export format is defined


def serialize_paraver_export(trace: pd.DataFrame) -> str:
    """Serialize the given trace table to the paraver export format (not yet defined)."""
    raise NotImplementedError("paraver export format not defined yet")
