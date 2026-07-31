from __future__ import annotations

import pytest

from carm_roofline.core import DataType
from carm_roofline.gpu import ComputeCapability, GPULaunchConfig, GPUVendor, TensorPrecision

pytestmark = pytest.mark.unit


class TestGPUVendor:
    def test_members(self):
        assert GPUVendor.NVIDIA.value == "nvidia"
        assert GPUVendor.AMD.value == "amd"

    def test_enum_distinct(self):
        assert GPUVendor.NVIDIA != GPUVendor.AMD


class TestGPULaunchConfig:
    def test_defaults(self):
        cfg = GPULaunchConfig(blocks=1024)
        assert cfg.blocks == 1024
        assert cfg.threads_per_block == 1024
        assert cfg.sm_targets is None

    def test_explicit(self):
        cfg = GPULaunchConfig(blocks=512, threads_per_block=256, sm_targets=80)
        assert cfg.num_threads == 512 * 256

    def test_frozen(self):
        cfg = GPULaunchConfig(blocks=1024)
        with pytest.raises(AttributeError):
            cfg.blocks = 2048


class TestComputeCapability:
    def test_as_int_nvidia(self):
        cc = ComputeCapability(major=8, minor=9, vendor=GPUVendor.NVIDIA)
        assert cc.as_int == 89

    @pytest.mark.parametrize(
        "s,vendor,expected_major,expected_minor,expected_gfx",
        [
            ("8.9", GPUVendor.NVIDIA, 8, 9, None),
            ("89", GPUVendor.NVIDIA, 8, 9, None),
            ("70", GPUVendor.NVIDIA, 7, 0, None),
            ("gfx942", GPUVendor.AMD, 9, 42, "gfx942"),
            ("9.4.2", GPUVendor.AMD, 9, 42, "gfx942"),
        ],
    )
    def test_from_string(self, s, vendor, expected_major, expected_minor, expected_gfx):
        cc = ComputeCapability.from_string(s, vendor)
        assert cc.major == expected_major
        assert cc.minor == expected_minor
        assert cc.vendor == vendor
        assert cc.gfx_arch == expected_gfx

    def test_from_string_nvidia_compact_as_int(self):
        cc = ComputeCapability.from_string("89", GPUVendor.NVIDIA)
        assert cc.as_int == 89

    def test_from_string_amd_as_int(self):
        cc = ComputeCapability.from_string("gfx942", GPUVendor.AMD)
        assert cc.as_int == 9

    def test_gfx_arch_default_none(self):
        cc = ComputeCapability(major=8, minor=9, vendor=GPUVendor.NVIDIA)
        assert cc.gfx_arch is None


class TestTensorPrecision:
    FP16_32 = TensorPrecision(
        name="fp16_32",
        precision_triple=(DataType.f16, DataType.f16, DataType.f32),
        tile_mnk=(16, 8, 16),
        flops_per_mma=2 * 16 * 8 * 16,
    )

    def test_flops_formula(self):
        assert self.FP16_32.flops_per_mma == 4096  # 2 * 16 * 8 * 16

    def test_frozen(self):
        with pytest.raises(AttributeError):
            self.FP16_32.name = "changed"


class TestTensorCoreOperation:
    def test_mma_repr(self):
        from carm_roofline.core import TensorCoreOperation

        assert repr(TensorCoreOperation.mma) == "mma"
        assert TensorCoreOperation.mma.name == "mma"


class TestDataTypeGPU:
    @pytest.mark.parametrize(
        "dt,expected_bytes",
        [
            (DataType.f16, 2),
            (DataType.bf16, 2),
            (DataType.tf32, 4),
            (DataType.f8, 1),
            (DataType.i4, 1),
            (DataType.i1, 1),
        ],
    )
    def test_bytes(self, dt, expected_bytes):
        assert dt.bytes() == expected_bytes

    @pytest.mark.parametrize(
        "dt,expected_bits",
        [
            (DataType.f16, 16),
            (DataType.bf16, 16),
            (DataType.tf32, 32),
            (DataType.f8, 8),
            (DataType.i4, 4),
            (DataType.i1, 1),
        ],
    )
    def test_bits(self, dt, expected_bits):
        assert dt.bits() == expected_bits

    def test_bits_from_bytes_consistency(self):
        """For byte-aligned types, bytes*8 == bits. For sub-byte types, ceil(bits/8) == bytes."""
        assert DataType.f16.bits() == DataType.f16.bytes() * 8
        assert DataType.i4.bytes() == 1  # storage, not arithmetic width
        assert DataType.i4.bits() == 4  # arithmetic width

    def test_to_c_type(self):
        assert DataType.f16.to_c_type() == "f16"
        assert DataType.bf16.to_c_type() == "bf16"


class TestPrecisionCascade:
    """Test supported_tensor_precisions and supported_vector_precisions cascade logic."""

    # -- NVIDIA tensor precision tiers --

    def test_nvidia_cc70(self):
        cc = ComputeCapability(major=7, minor=0, vendor=GPUVendor.NVIDIA)
        precisions = cc.supported_tensor_precisions()
        assert set(precisions.keys()) == {"fp16_32"}

    def test_nvidia_cc75(self):
        cc = ComputeCapability(major=7, minor=5, vendor=GPUVendor.NVIDIA)
        precisions = cc.supported_tensor_precisions()
        assert set(precisions.keys()) == {"fp16_32", "fp16_16", "int8", "int4"}

    def test_nvidia_cc80(self):
        cc = ComputeCapability(major=8, minor=0, vendor=GPUVendor.NVIDIA)
        precisions = cc.supported_tensor_precisions()
        assert set(precisions.keys()) == {"fp16_32", "fp16_16", "int8", "int4", "fp64", "bf16", "tf32", "int1"}

    def test_nvidia_cc89(self):
        """Ada (consumer CC 8.9) has no FP64 tensor cores; fp64 is datacenter-only (CC 80/90)."""
        cc = ComputeCapability(major=8, minor=9, vendor=GPUVendor.NVIDIA)
        precisions = cc.supported_tensor_precisions()
        assert set(precisions.keys()) == {"fp16_32", "fp16_16", "int8", "int4", "bf16", "tf32", "int1", "fp8"}

    def test_nvidia_cc100(self):
        """Future CC 10.0 — should at least match CC >=89 tier."""
        cc = ComputeCapability(major=10, minor=0, vendor=GPUVendor.NVIDIA)
        precisions = cc.supported_tensor_precisions()
        assert "fp8" in precisions

    # -- GTX filter --

    def test_gtx_filter_empty(self):
        """GTX cards with CC >=70 should return empty tensor precisions."""
        cc = ComputeCapability(major=7, minor=5, vendor=GPUVendor.NVIDIA)
        precisions = cc.supported_tensor_precisions(model_name="GTX 1650")
        assert precisions == {}

    def test_rtx_not_filtered(self):
        """RTX cards with CC >=70 should return non-empty tensor precisions."""
        cc = ComputeCapability(major=7, minor=5, vendor=GPUVendor.NVIDIA)
        precisions = cc.supported_tensor_precisions(model_name="RTX 2060")
        assert len(precisions) > 0

    def test_gtx_below_70_not_filtered(self):
        """GTX cards with CC <70 are not filtered (no tensor cores at all)."""
        cc = ComputeCapability(major=6, minor=1, vendor=GPUVendor.NVIDIA)
        precisions = cc.supported_tensor_precisions(model_name="GTX 1080")
        assert precisions == {}

    # -- AMD matrix precision tiers --

    def test_amd_gfx908(self):
        cc = ComputeCapability.from_string("gfx908", GPUVendor.AMD)
        precisions = cc.supported_tensor_precisions()
        assert set(precisions.keys()) == {"fp32", "int8", "fp16"}

    def test_amd_gfx90a(self):
        cc = ComputeCapability.from_string("gfx90a", GPUVendor.AMD)
        precisions = cc.supported_tensor_precisions()
        assert set(precisions.keys()) == {"fp32", "int8", "fp64", "fp16"}

    def test_amd_gfx942(self):
        cc = ComputeCapability.from_string("gfx942", GPUVendor.AMD)
        precisions = cc.supported_tensor_precisions()
        assert set(precisions.keys()) == {"fp32", "int8", "fp64", "fp16", "fp8"}

    # -- Vector precisions --

    def test_vector_precisions_nvidia_baseline(self):
        cc = ComputeCapability(major=5, minor=0, vendor=GPUVendor.NVIDIA)
        vec = cc.supported_vector_precisions()
        assert DataType.f32 in vec
        assert DataType.f16 not in vec

    def test_vector_precisions_nvidia_cc60(self):
        cc = ComputeCapability(major=6, minor=0, vendor=GPUVendor.NVIDIA)
        vec = cc.supported_vector_precisions()
        assert DataType.f16 in vec
        assert DataType.bf16 not in vec

    def test_vector_precisions_nvidia_cc80(self):
        cc = ComputeCapability(major=8, minor=0, vendor=GPUVendor.NVIDIA)
        vec = cc.supported_vector_precisions()
        assert DataType.f16 in vec
        assert DataType.bf16 in vec
        assert DataType.tf32 in vec

    def test_vector_precisions_amd_has_f16(self):
        cc = ComputeCapability(major=9, minor=0, vendor=GPUVendor.AMD)
        vec = cc.supported_vector_precisions()
        assert DataType.f16 in vec
        assert DataType.f32 in vec
