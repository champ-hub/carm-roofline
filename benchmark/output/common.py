"""Common utilities for benchmark output handlers.

This module provides shared functionality to eliminate code duplication across handlers.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from output_utils import error, info, warn

if TYPE_CHECKING:
    from benchmark.generation.code_gen import DataType


def format_precision_label(data_type: DataType) -> str:
    """Return legacy-like precision labels (sp/dp) when possible."""
    name = getattr(data_type, "name", None)
    if name == "f32":
        return "sp"
    if name == "f64":
        return "dp"
    return name or str(data_type)


def safe_matplotlib_import() -> tuple[Any | None, Any | None]:
    """Safely import matplotlib and numpy with error handling.

    Returns:
        A tuple of (plt, np) if both are available, otherwise (None, None).
        Logs an error message if imports fail.

    Example:
        >>> plt, np = safe_matplotlib_import()
        >>> if plt is None:
        ...     return  # Skip plotting

    Note:
        This function will not raise exceptions - it gracefully degrades by returning None.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        return plt, np
    except ImportError as e:
        # Determine which package failed for better error message
        missing = "matplotlib" if "matplotlib" in str(e) else "numpy" if "numpy" in str(e) else "required packages"
        warn(f"{missing} not available; skipping plots. Install with: pip install {missing}")
        return None, None


def save_or_show_plot(
    output_path: Path | None, filename: str, show_message: bool = True, plt: Any | None = None
) -> None:
    """Save plot to file or display interactively.

    Centralizes the save/show logic used across all handlers.

    Args:
        output_path: Directory to save plot. If None, displays plot interactively.
        filename: Name of the output file (e.g., "arithmetic_gops.png")
        show_message: Whether to print confirmation message when saving
        plt: matplotlib.pyplot module (if None, will attempt to import)

    Side effects:
        - Creates output_path directory if it doesn't exist (when saving)
        - Saves figure to disk or displays it
        - Prints confirmation message (when show_message=True and saving)

    Example:
        >>> plt.figure()
        >>> plt.plot([1, 2, 3], [4, 5, 6])
        >>> save_or_show_plot(Path("results"), "test.png")
        Saved plot: results/test.png

    Note:
        All I/O errors (permissions, disk full, etc.) are caught and logged.
        Function will not raise exceptions - it gracefully degrades.
    """
    if plt is None:
        try:
            import matplotlib.pyplot as plt_local
        except ImportError:
            warn("matplotlib not available; cannot save or show plot")
            return
        plt = plt_local

    if output_path:
        try:
            # Ensure parent directory exists
            output_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            error(f"Failed to create output directory {output_path}: {e}")
            return

        out_file = output_path / filename
        try:
            plt.savefig(out_file)
            if show_message:
                info(f"Saved plot: {out_file}")
        except OSError as e:
            error(f"Failed to save plot to {out_file}: {e}")
            if "Permission denied" in str(e):
                warn("Hint: Check file permissions")
            elif "No space left" in str(e):
                warn("Hint: Disk may be full")
        except Exception as e:
            error(f"Unexpected error saving plot to {out_file}: {e}")
    else:
        # Display interactively
        try:
            plt.show()
        except Exception as e:
            error(f"Failed to display plot: {e}")
