from __future__ import annotations

from dataclasses import dataclass

from carm_roofline.gpu.types import GPUVendor


@dataclass(frozen=True)
class ComputeCapability:
    """GPU compute capability identifier.

    For NVIDIA: ``major`` and ``minor`` from CC string (e.g. CC 8.9 → major=8, minor=9,
    as_int=89). ``as_int`` is ``major * 10 + minor``.
    For AMD: ``major`` is the gfx major version (e.g. gfx942 → major=9), ``minor`` is 0,
    ``as_int = major``.
    """

    major: int
    minor: int
    vendor: GPUVendor

    @property
    def as_int(self) -> int:
        if self.vendor == GPUVendor.NVIDIA:
            return self.major * 10 + self.minor
        return self.major

    @staticmethod
    def from_string(s: str, vendor: GPUVendor) -> ComputeCapability:
        """Parse a compute capability string.

        NVIDIA formats accepted: ``"8.9"``, ``"89"``
        AMD formats accepted: ``"gfx942"``, ``"9.4.2"``
        """
        raise NotImplementedError("ComputeCapability.from_string deferred to Phase 1")
