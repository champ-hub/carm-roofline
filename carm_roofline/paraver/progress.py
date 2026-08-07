"""Legacy-compatible terminal progress bar for Paraver launches.

Ported from the legacy carm-paraver ``Paraver_CARM.py`` (class ``ProgressBar``);
the byte-exact output format is part of the Paraver interop contract (see the
paraver-reference docs, ``08-environment-and-deployment.md`` §4.2). Paraver's
plugin renders the stdout progress line as a popup: it opens on the first bar
line and closes only when a full 100% bar is printed, so ``update`` guarantees
the completed line is emitted at most once (a second one would open a second
popup that never closes).
"""

from __future__ import annotations

import math


class ProgressBar:
    """Prints a terminal progress bar that updates in-place, ensuring 100% is printed exactly once."""

    def __init__(self, total: int, bar_width: int = 30):
        self.total = total
        self.bar_width = bar_width
        self._done = False

    def update(self, processed: int) -> None:
        """Print the progress bar for the given number of processed items.

        When ``processed >= total`` the bar is shown at 100% followed by a
        newline; subsequent calls are no-ops. No-op while ``total <= 0``.
        ``total`` may be assigned after construction (the provider builds the bar
        before paramedir, when the burst count is unknown) — ``update`` reads it
        at call time.
        """
        if self._done:
            return

        if self.total <= 0:
            return

        if processed >= self.total:
            self._print(1.0)
            print()  # noqa: T201 — legacy byte-exact protocol; see module docstring
            self._done = True
        else:
            progress = min(processed / self.total, 0.99)
            self._print(progress)

    def _print(self, progress: float) -> None:
        segments = self.bar_width if progress >= 1.0 else min(math.ceil(self.bar_width * progress), self.bar_width - 1)
        print(  # noqa: T201 — legacy byte-exact protocol; see module docstring
            f"[{'#' * segments}{' ' * (self.bar_width - segments)}] {progress * 100:.1f}%",
            end="\r",
            flush=True,
        )
