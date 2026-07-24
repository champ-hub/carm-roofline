from __future__ import annotations

import shutil
import warnings
from collections.abc import Generator
from contextlib import contextmanager

from carm_roofline.core import UserError
from carm_roofline.gpu import detect


class GPUFrequencyManager:
    """Manage GPU clock frequencies with ``locked()`` context manager support.

    Wraps the lower-level ``detect`` functions with state tracking for clock
    restoration.
    """

    def __init__(self, device: int = 0) -> None:
        """Initialize the manager.

        Args:
            device: GPU device index.

        Raises:
            UserError: nvidia-smi is not on PATH.
        """
        if shutil.which("nvidia-smi") is None:
            raise UserError("nvidia-smi not found — GPU frequency management requires NVIDIA drivers.")
        self._device = device
        self._original_clocks: dict[str, int] | None = None

    def read_clocks(self) -> dict[str, int]:
        """Read current SM and memory clock frequencies.

        Returns:
            Dictionary with keys ``"sm"`` and ``"mem"``, values in MHz.
        """
        return detect.read_gpu_frequencies(device=self._device)

    def enable_persistence_mode(self) -> None:
        """Enable NVIDIA persistence mode (``nvidia-smi -pm 1``).

        Prints a warning on failure but does not raise.
        """
        try:
            import subprocess

            subprocess.run(
                ["nvidia-smi", "-i", str(self._device), "-pm", "1"],
                capture_output=True,
                text=True,
                check=True,
            )
        except Exception:
            warnings.warn(f"failed to enable persistence mode for device {self._device}", stacklevel=2)

    def lock_clocks(self, sm_clock: int | None = None, mem_clock: int | None = None) -> None:
        """Lock SM and/or memory clock frequencies.

        On first call, saves the original clocks so they can be restored later.
        """
        if self._original_clocks is None:
            try:
                self._original_clocks = self.read_clocks()
            except Exception:
                self._original_clocks = {"sm": 0, "mem": 0}

        detect.lock_gpu_frequencies(device=self._device, sm_clock=sm_clock, mem_clock=mem_clock)

    def reset_clocks(self) -> None:
        """Reset GPU clocks to their original values, or to default if unknown."""
        if self._original_clocks is not None:
            detect.lock_gpu_frequencies(
                device=self._device,
                sm_clock=self._original_clocks.get("sm"),
                mem_clock=self._original_clocks.get("mem"),
            )
        else:
            detect.reset_gpu_clocks(device=self._device)

    @contextmanager
    def locked(self, sm_clock: int | None = None, mem_clock: int | None = None) -> Generator[None, None, None]:
        """Context manager that locks clocks on enter and restores on exit.

        Enables persistence mode before locking.
        """
        self.enable_persistence_mode()
        self.lock_clocks(sm_clock=sm_clock, mem_clock=mem_clock)
        try:
            yield
        finally:
            self.reset_clocks()
