from __future__ import annotations

from carm_roofline.core import DataType
from carm_roofline.gpu.compute_capability import ComputeCapability
from carm_roofline.gpu.types import GPUVendor, TensorPrecision

# ---------------------------------------------------------------------------
# NVIDIA tensor precision table
# ---------------------------------------------------------------------------
NVIDIA_TENSOR_PRECISIONS: dict[str, TensorPrecision] = {
    "fp16_32": TensorPrecision(
        name="fp16_32",
        precision_triple=(DataType.f16, DataType.f16, DataType.f32),
        tile_mnk=(16, 8, 16),
        flops_per_mma=2 * 16 * 8 * 16,
    ),
    "fp16_16": TensorPrecision(
        name="fp16_16",
        precision_triple=(DataType.f16, DataType.f16, DataType.f16),
        tile_mnk=(16, 8, 16),
        flops_per_mma=2 * 16 * 8 * 16,
    ),
    "int8": TensorPrecision(
        name="int8",
        precision_triple=(DataType.i8, DataType.i8, DataType.i32),
        tile_mnk=(16, 8, 16),
        flops_per_mma=2 * 16 * 8 * 16,
    ),
    "int4": TensorPrecision(
        name="int4",
        precision_triple=(DataType.i4, DataType.i4, DataType.i32),
        tile_mnk=(16, 8, 32),
        flops_per_mma=2 * 16 * 8 * 32,
    ),
    "fp64": TensorPrecision(
        name="fp64",
        precision_triple=(DataType.f64, DataType.f64, DataType.f64),
        tile_mnk=(16, 8, 4),
        flops_per_mma=2 * 16 * 8 * 4,
    ),
    "bf16": TensorPrecision(
        name="bf16",
        precision_triple=(DataType.bf16, DataType.bf16, DataType.f32),
        tile_mnk=(16, 8, 16),
        flops_per_mma=2 * 16 * 8 * 16,
    ),
    "tf32": TensorPrecision(
        name="tf32",
        precision_triple=(DataType.tf32, DataType.tf32, DataType.f32),
        tile_mnk=(16, 8, 8),
        flops_per_mma=2 * 16 * 8 * 8,
    ),
    "int1": TensorPrecision(
        name="int1",
        precision_triple=(DataType.i1, DataType.i1, DataType.i32),
        tile_mnk=(16, 8, 128),
        flops_per_mma=2 * 16 * 8 * 128,
    ),
    "fp8": TensorPrecision(
        name="fp8",
        precision_triple=(DataType.f8, DataType.f8, DataType.f32),
        tile_mnk=(16, 8, 16),
        flops_per_mma=2 * 16 * 8 * 16,
    ),
}

# ---------------------------------------------------------------------------
# AMD matrix precision table (estimated CDNA tile sizes)
# ---------------------------------------------------------------------------
AMD_TENSOR_PRECISIONS: dict[str, TensorPrecision] = {
    "fp32": TensorPrecision(
        name="fp32",
        precision_triple=(DataType.f32, DataType.f32, DataType.f32),
        tile_mnk=(16, 16, 4),
        flops_per_mma=2 * 16 * 16 * 4,
    ),
    "int8": TensorPrecision(
        name="int8",
        precision_triple=(DataType.i8, DataType.i8, DataType.i32),
        tile_mnk=(16, 16, 16),
        flops_per_mma=2 * 16 * 16 * 16,
    ),
    "fp64": TensorPrecision(
        name="fp64",
        precision_triple=(DataType.f64, DataType.f64, DataType.f64),
        tile_mnk=(16, 16, 4),
        flops_per_mma=2 * 16 * 16 * 4,
    ),
    "fp16": TensorPrecision(
        name="fp16",
        precision_triple=(DataType.f16, DataType.f16, DataType.f32),
        tile_mnk=(16, 16, 16),
        flops_per_mma=2 * 16 * 16 * 16,
    ),
    "fp8": TensorPrecision(
        name="fp8",
        precision_triple=(DataType.f8, DataType.f8, DataType.f32),
        tile_mnk=(16, 16, 32),
        flops_per_mma=2 * 16 * 16 * 32,
    ),
}


def _nvidia_precision_cascade(as_int: int, model_name: str) -> dict[str, TensorPrecision]:
    """Determine supported NVIDIA tensor precisions by compute capability and model."""
    # GTX cards do not support tensor cores
    if "GTX" in model_name.upper() and as_int >= 70:
        return {}

    precisions: dict[str, TensorPrecision] = {}
    if as_int >= 70:
        precisions["fp16_32"] = NVIDIA_TENSOR_PRECISIONS["fp16_32"]
    if as_int >= 75:
        precisions["fp16_16"] = NVIDIA_TENSOR_PRECISIONS["fp16_16"]
        precisions["int8"] = NVIDIA_TENSOR_PRECISIONS["int8"]
        precisions["int4"] = NVIDIA_TENSOR_PRECISIONS["int4"]
    if as_int >= 80:
        precisions["bf16"] = NVIDIA_TENSOR_PRECISIONS["bf16"]
        precisions["tf32"] = NVIDIA_TENSOR_PRECISIONS["tf32"]
        precisions["int1"] = NVIDIA_TENSOR_PRECISIONS["int1"]
    if as_int in (80, 90) or as_int >= 100:
        precisions["fp64"] = NVIDIA_TENSOR_PRECISIONS["fp64"]
    if as_int >= 89:
        precisions["fp8"] = NVIDIA_TENSOR_PRECISIONS["fp8"]
    return precisions


def _amd_precision_cascade(gfx_arch: str) -> dict[str, TensorPrecision]:
    """Determine supported AMD matrix precisions by gfx architecture."""
    precisions: dict[str, TensorPrecision] = {}
    if gfx_arch in ("gfx908", "gfx90a", "gfx942"):
        precisions["fp32"] = AMD_TENSOR_PRECISIONS["fp32"]
        precisions["int8"] = AMD_TENSOR_PRECISIONS["int8"]
        precisions["fp16"] = AMD_TENSOR_PRECISIONS["fp16"]
    if gfx_arch in ("gfx90a", "gfx942"):
        precisions["fp64"] = AMD_TENSOR_PRECISIONS["fp64"]
    if gfx_arch == "gfx942":
        precisions["fp8"] = AMD_TENSOR_PRECISIONS["fp8"]
    return precisions


def supported_tensor_precisions(
    compute_capability: ComputeCapability,
    model_name: str = "",
) -> dict[str, TensorPrecision]:
    """Return the set of tensor/matrix precisions available on this GPU.

    Dispatches to vendor-specific cascade logic based on ``compute_capability.vendor``.
    """
    if compute_capability.vendor == GPUVendor.NVIDIA:
        return _nvidia_precision_cascade(compute_capability.as_int, model_name)
    elif compute_capability.vendor == GPUVendor.AMD:
        gfx = compute_capability.gfx_arch or f"gfx{compute_capability.major}"
        return _amd_precision_cascade(gfx)
    return {}


def supported_vector_precisions(compute_capability: ComputeCapability) -> list[DataType]:
    """Return base vector data types supported on this GPU."""
    base: list[DataType] = [DataType.f32, DataType.f64, DataType.i8, DataType.i16, DataType.i32, DataType.i64]

    if compute_capability.vendor == GPUVendor.NVIDIA:
        if compute_capability.as_int >= 60:
            base.append(DataType.f16)
        if compute_capability.as_int >= 80:
            base.append(DataType.bf16)
            base.append(DataType.tf32)
    elif compute_capability.vendor == GPUVendor.AMD:
        base.append(DataType.f16)

    return base
