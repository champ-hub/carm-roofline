from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from carm_roofline.gpu.types import GPUVendor

if TYPE_CHECKING:
    from carm_roofline.core import DataType
    from carm_roofline.gpu.types import TensorPrecision


@dataclass(frozen=True)
class ComputeCapability:
    """GPU compute capability identifier.

    For NVIDIA: ``major`` and ``minor`` from CC string (e.g. CC 8.9 → major=8, minor=9,
    as_int=89). ``as_int`` is ``major * 10 + minor``.
    For AMD: ``major`` is the gfx major version (e.g. gfx942 → major=9), ``minor`` is 0,
    ``as_int = major``.

    ``gfx_arch`` stores the raw AMD gfx architecture string (e.g. ``"gfx942"``) for
    AMD cascade tiering, or ``None`` for NVIDIA.
    """

    major: int
    minor: int
    vendor: GPUVendor
    gfx_arch: str | None = None

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
        s = s.strip()
        if vendor == GPUVendor.NVIDIA:
            # Accept "8.9" or "89"
            if "." in s:
                parts = s.split(".")
                major = int(parts[0])
                minor = int(parts[1])
            else:
                # Find where the first major version digits end
                major = int(s[0])
                minor = int(s[1:]) if len(s) > 1 else 0
            return ComputeCapability(major=major, minor=minor, vendor=vendor)
        elif vendor == GPUVendor.AMD:
            # Accept "gfx942", "9.4.2", or "942"
            if s.startswith("gfx"):
                gfx_arch = s
                digits = s[3:]  # "942"
            elif "." in s:
                parts = s.split(".")
                gfx_arch = "gfx" + "".join(parts)
                digits = "".join(parts)
            else:
                gfx_arch = "gfx" + s
                digits = s
            major = int(digits[0]) if digits else 9
            return ComputeCapability(major=major, minor=0, vendor=vendor, gfx_arch=gfx_arch)
        else:
            raise ValueError(f"Unsupported GPU vendor: {vendor}")

    def supported_tensor_precisions(self, model_name: str = "") -> dict[str, TensorPrecision]:
        """Return tensor/matrix precisions supported by this GPU."""
        from carm_roofline.gpu.precision import supported_tensor_precisions as _stp

        return _stp(self, model_name)

    def supported_vector_precisions(self) -> list[DataType]:
        """Return base vector data types supported by this GPU."""
        from carm_roofline.gpu.precision import supported_vector_precisions as _svp

        return _svp(self)
