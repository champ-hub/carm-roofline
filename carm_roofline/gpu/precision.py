from __future__ import annotations

from typing import Any

from carm_roofline.gpu.types import TensorPrecision

# NVIDIA tensor precision table — will be populated in Phase 1
NVIDIA_TENSOR_PRECISIONS: dict[str, TensorPrecision] = {}

# AMD matrix precision table — will be populated in Phase 1
AMD_TENSOR_PRECISIONS: dict[str, TensorPrecision] = {}


def supported_tensor_precisions(compute_capability: Any) -> dict[str, TensorPrecision]:
    """Return supported tensor precisions for a given compute capability.

    Deferred to Phase 1 — returns empty dict.
    """
    return {}


def supported_vector_precisions(compute_capability: Any) -> list[str]:
    """Return supported vector precisions for a given compute capability.

    Deferred to Phase 1 — returns empty list.
    """
    return []
